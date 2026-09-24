"""The batched certify-and-repair LP solver.

For every member of a batch the solver tries, in order:

1. **warm**: the basis (or Farkas certificate) that solved this member last time
   -- temporal coherence makes this succeed for most dFBA steps;
2. **pool**: the most frequently used certified bases of the group;
3. **repair**: a few *representative* pending members are solved exactly with a
   warm-started lexicographic dual simplex (HiGHS, CPU, in parallel); each new
   optimal basis is certified and immediately tried on **all** remaining pending
   members (**propagation**), each new Farkas ray likewise.

Steps 1-2 and propagation are a single batched GEMM per distinct basis and run
on the GPU (or vectorised on the CPU); only step 3 pivots.  A member is only
reported ``certified`` when our own arithmetic proves optimality (or
infeasibility); the rare member whose HiGHS basis fails re-certification is
returned with HiGHS' values and ``certified = False``.

**Incoherent batches.**  When fresh bases stop certifying other members (a
batch scattered over as many critical regions as members), propagation is pure
overhead.  A yield guard then switches the rest of the batch to *direct mode*:
members are solved by warm-started simplex in worker processes and each answer
is verified by a point certificate (Theorem 1 for one member, no affine law).
Their bases enter the pool as lazy candidates, factorised only if a later call
reuses them.  This bounds the worst case near multi-process simplex speed
without giving up certification.
"""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from manylp.backend import Backend, get_backend
from manylp.basis import (
    CertificateError,
    Tolerances,
    basis_key,
    build_farkas_entry,
    certify,
    check_farkas,
    explain_certify,
    prepare_basis_entry,
    upload_entry,
)
from manylp.lp import LexLP
from manylp.pool import CertificatePool
from manylp.repair import ERROR, INFEASIBLE, OPTIMAL, UNBOUNDED, HighsLexSolver

# where a member's answer came from
SRC_WARM, SRC_POOL, SRC_PROPAGATED, SRC_REPAIRED, SRC_HIGHS, SRC_TRIVIAL = range(6)
SOURCE_NAMES = ("warm", "pool", "propagated", "repaired", "highs_uncertified", "trivial")
STATUS_NAMES = {0: "unsolved", OPTIMAL: "optimal", INFEASIBLE: "infeasible",
                UNBOUNDED: "unbounded", ERROR: "error"}


@dataclass
class BatchSolution:
    status: np.ndarray        # (B,) int8: 1 optimal, 2 infeasible, 3 unbounded, 4 error
    objective: np.ndarray     # (B, K) stage objective values (0 when not optimal)
    z: Optional[np.ndarray]   # (B, n_out) requested primal values (0 when not optimal)
    basis_id: np.ndarray      # (B,) certificate id, pass back as ``warm_start``
    certified: np.ndarray     # (B,) our arithmetic proved the answer
    unique: np.ndarray        # (B,) the lexicographic optimum is provably unique
    source: np.ndarray        # (B,) see SOURCE_NAMES
    stats: dict = field(default_factory=dict)

    @property
    def optimal(self) -> np.ndarray:
        return self.status == OPTIMAL


class GroupHandle:
    """A registered group: one shared ``A`` and objective list, many bound vectors."""

    _ids = 0

    def __init__(self, lp: LexLP, out_z: np.ndarray, name: str, max_bases: int) -> None:
        self.lp = lp
        self.out_z = np.asarray(out_z, dtype=np.int64)
        self.name = name
        self.pool = CertificatePool(max_bases=max_bases)
        GroupHandle._ids += 1
        self.gid = GroupHandle._ids
        self.totals = {k: 0 for k in SOURCE_NAMES}
        self.totals.update(highs_solves=0, highs_iterations=0, highs_seconds=0.0,
                           cert_failures=0, calls=0, members=0)

    @property
    def n_out(self) -> int:
        return int(self.out_z.size)


class BatchLPSolver:
    """Solve batches of bounds-parametric lexicographic LPs.

    Parameters
    ----------
    device:
        ``"cpu"``, ``"cuda"`` or ``"cuda:<id>"``: where certification runs.
    n_workers:
        CPU threads for repair solves (HiGHS releases the GIL while solving).
    pool_scan:
        How many of the hottest cached bases to try on members whose warm start
        missed, before repairing.
    max_bases:
        Per-group cap on cached bases (each costs ``8*m*|NP|`` bytes of device memory).
    """

    def __init__(self, device: str = "cpu", tol: Optional[Tolerances] = None,
                 n_workers: Optional[int] = None, pool_scan: int = 4,
                 max_bases: int = 512, time_limit: float = 300.0,
                 gpu_min_batch: int = 48, progressive: bool = True,
                 pool_scan_near: int = 64, min_yield: float = 1.5) -> None:
        self.backend: Backend = get_backend(device)
        #: groups smaller than this are certified on the host even on a GPU backend
        self.gpu_min_batch = gpu_min_batch
        self.progressive = progressive
        self.pool_scan_near = pool_scan_near
        #: members resolved per repaired representative below which repair goes direct
        self.min_yield = min_yield
        self.tol = tol or Tolerances()
        self.n_workers = int(n_workers or min(16, os.cpu_count() or 1))
        self.pool_scan = pool_scan
        self.max_bases = max_bases
        self.time_limit = time_limit
        self._tls = threading.local()
        self.debug = bool(os.environ.get("MANYLP_DEBUG"))
        self.debug_log: list = []
        self._executor = ThreadPoolExecutor(max_workers=self.n_workers) if self.n_workers > 1 else None
        #: processes for direct mode (spawned lazily on first use, persistent across groups)
        self._procs = None
        from manylp import cpu_kernels

        cpu_kernels.warm_up()      # JIT once at construction, never inside a timed solve

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown()
            self._executor = None
        if self._procs is not None:
            self._procs.shutdown(cancel_futures=True)
            self._procs = None

    def _process_pool(self):
        if self._procs is None:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor

            self._procs = ProcessPoolExecutor(max_workers=self.n_workers, mp_context=mp.get_context("spawn"))
        return self._procs

    def warm_up_processes(self) -> None:
        """Start the direct-mode worker processes ahead of time (optional)."""
        pool = self._process_pool()
        list(pool.map(abs, range(self.n_workers)))

    # -- registration ----------------------------------------------------------
    def register_group(self, lp: LexLP, out_z: Optional[np.ndarray] = None,
                       name: str = "") -> GroupHandle:
        """Register a group.  ``out_z`` selects the z-indices returned per member
        (default: all ``n`` structural columns)."""
        if out_z is None:
            out_z = np.arange(lp.n)
        return GroupHandle(lp, out_z, name or f"group{GroupHandle._ids + 1}", self.max_bases)

    def _materialise(self, g, e):
        """Factorise and upload a lazy candidate basis in place (``None`` if it fails)."""
        lp = g.lp
        src = lp.full_bounds(*e.anchor) if e.anchor is not None else None
        try:
            prepared = prepare_basis_entry(lp, e.basic, e.status, self.tol, g.out_z, source_bounds=src)
        except CertificateError:
            g.pool.bases.pop(e.id, None)
            return None
        upload_entry(prepared, self.backend)
        e.key, e.basic, e.status = prepared.key, prepared.basic, prepared.status
        e.n_np, e.static_unique = prepared.n_np, prepared.static_unique
        e.host, e.harr, e.dev, e.flags = prepared.host, prepared.harr, prepared.dev, prepared.flags
        g.pool.mark_materialised()
        return e

    def _direct_solve(self, g, idx, hb, ws, pool, rstats, counts, status, certified, unique,
                      basis_id, source, pending, out):
        """Remaining members of a low-yield batch: per-member warm-started simplex in worker
        processes, each answer verified by a point certificate (Theorem 1 for one member)."""
        from manylp.solver import lp_fingerprint as _fp

        lp = g.lp
        key = getattr(g, "_fp", None)
        if key is None:
            key = g._fp = _fp(lp)
        L, U = hb.rows(idx)
        warms = []
        for i, r in enumerate(idx):
            e = pool.get(ws[r]) if ws[r] >= 0 else None
            warms.append(None if e is None else e.status)
        nproc = self.n_workers
        bounds = np.linspace(0, idx.size, min(idx.size, 4 * nproc) + 1).astype(int)
        ex = self._process_pool()
        futs = [(a, b, ex.submit(_dm_solve_chunk, key, lp, self.tol, g.out_z, L[a:b], U[a:b], warms[a:b]))
                for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
        for a, b, f in futs:
            for k, rec in enumerate(f.result()):
                r = idx[a + k]
                rstats["highs_solves"] += 1
                rstats["highs_iterations"] += rec["iters"]
                rstats["highs_seconds"] += rec["seconds"]
                status[r] = rec["status"]
                certified[r] = rec["certified"]
                unique[r] = rec["unique"]
                basis_id[r] = -1
                if rec.get("basic") is not None:
                    raw = basis_key(rec["basic"], rec["zstatus"])
                    basis_id[r] = pool.add_lazy(rec["basic"], rec["zstatus"], raw,
                                                (L[a + k].copy(), U[a + k].copy()))
                src = SRC_REPAIRED if rec["certified"] else SRC_HIGHS
                source[r] = src
                counts[SOURCE_NAMES[src]] += 1
                pending[r] = False
                if rec["status"] == OPTIMAL:
                    out["obj"][r] = rec["obj"]
                    if out["z"] is not None:
                        out["z"][r] = rec["z"]

    # -- persistent basis atlas ----------------------------------------------------
    def save_atlas(self, g: GroupHandle, path: str, max_bases: Optional[int] = None) -> int:
        """Save the group's certified bases (hottest first) to ``path`` (.npz).

        Bases are certificates, not answers: whatever diet, ensemble or run
        loads them later, every member is re-certified against its own bounds,
        so an atlas can only save work, never change a result.
        """
        # everything learned, factorised or still lazy (loading factorises all of them)
        entries = sorted(g.pool.bases.values(), key=lambda e: (e.host.get("lazy", False), -e.hits, -e.last_used))
        entries = entries[: (max_bases or len(entries))]
        basic = np.array([e.basic for e in entries], dtype=np.int32).reshape(len(entries), -1)
        status = np.array([e.status for e in entries], dtype=np.int8).reshape(len(entries), -1)
        anchors = np.array([np.concatenate(e.anchor) if e.anchor is not None
                            else np.full(2 * g.lp.p, np.nan) for e in entries]).reshape(len(entries), -1)
        hits = np.array([e.hits for e in entries], dtype=np.int64)
        np.savez_compressed(path, lp_hash=lp_fingerprint(g.lp), basic=basic, status=status,
                            anchors=anchors, hits=hits)
        return len(entries)

    def load_atlas(self, g: GroupHandle, path: str) -> int:
        """Load bases saved by :meth:`save_atlas` into the group's pool (in parallel)."""
        import os as _os

        if not _os.path.exists(path):
            return 0
        d = np.load(path)
        if str(d["lp_hash"]) != lp_fingerprint(g.lp):
            raise ValueError(f"atlas {path} was built for a different LP")
        lp, tol = g.lp, self.tol

        def build(i):
            try:
                return prepare_basis_entry(lp, d["basic"][i], d["status"][i], tol, g.out_z)
            except CertificateError:
                return None

        n = d["basic"].shape[0]
        if self._executor is not None and n > 1:
            prepared = list(self._executor.map(build, range(n)))
        else:
            prepared = [build(i) for i in range(n)]
        added = 0
        for i, e in enumerate(prepared):
            if e is None:
                continue
            upload_entry(e, self.backend)
            a = d["anchors"][i]
            if np.all(np.isfinite(a)) or not np.all(np.isnan(a)):
                e.anchor = (a[: lp.p].copy(), a[lp.p:].copy())
            e.hits = int(d["hits"][i])
            _, is_new = g.pool.add_basis(e)
            added += int(is_new)
        return added

    def _worker(self, g: GroupHandle) -> HighsLexSolver:
        d = getattr(self._tls, "solvers", None)
        if d is None:
            d = self._tls.solvers = {}
        w = d.get(g.gid)
        if w is None:
            w = d[g.gid] = HighsLexSolver(g.lp, dual_tol=self.tol.dual, time_limit=self.time_limit)
        return w

    # -- solve ---------------------------------------------------------------------
    def solve_batch(self, g: GroupHandle, lb, ub, warm_start=None,
                    return_z: bool = True, device_out: bool = False) -> BatchSolution:
        """Solve ``B`` members given their parametric bounds, shape ``(B, p)``.

        ``warm_start`` is a previous :class:`BatchSolution` for the same members
        (or its ``basis_id`` array).

        On a GPU backend ``lb``/``ub`` may be device (CuPy) arrays: they are then
        never copied to the host in bulk -- only the rows that CPU-side work
        (repairs, infeasibility checks, small groups) touches are fetched.  With
        ``device_out=True`` the ``objective`` and ``z`` of the result stay on the
        device too, so a GPU-resident caller moves no bulk data at all.
        """
        lp, pool, be, tol = g.lp, g.pool, self.backend, self.tol
        dev_in = be.is_gpu and not isinstance(lb, np.ndarray) and hasattr(lb, "__cuda_array_interface__")
        if dev_in:
            xp = be.xp
            Ld_in = xp.ascontiguousarray(xp.atleast_2d(lb), dtype=xp.float64)
            Ud_in = xp.ascontiguousarray(xp.atleast_2d(ub), dtype=xp.float64)
            hb = _LazyHost(Ld_in, Ud_in)
            B = Ld_in.shape[0]
            shape_l, shape_u = Ld_in.shape, Ud_in.shape
        else:
            Lp = np.ascontiguousarray(np.atleast_2d(np.asarray(lb, dtype=float)))
            Up = np.ascontiguousarray(np.atleast_2d(np.asarray(ub, dtype=float)))
            hb = _LazyHost(None, None, Lp, Up)
            B = Lp.shape[0]
            shape_l, shape_u = Lp.shape, Up.shape
        if shape_l != (B, lp.p) or shape_u != (B, lp.p):
            raise ValueError(f"bounds must have shape (B, {lp.p}); got {shape_l}, {shape_u}")
        device_out = bool(device_out and be.is_gpu)
        if isinstance(warm_start, BatchSolution):
            warm_start = warm_start.basis_id
        K, n_out = lp.K, g.n_out
        pool.clock += 1
        t_start = time.perf_counter()
        counts = {k: 0 for k in SOURCE_NAMES}
        rstats = {"highs_solves": 0, "highs_iterations": 0, "highs_seconds": 0.0,
                  "cert_failures": 0, "new_bases": 0, "direct_mode": 0}

        status = np.zeros(B, dtype=np.int8)
        basis_id = np.full(B, -1, dtype=np.int64)
        certified = np.zeros(B, dtype=bool)
        unique = np.zeros(B, dtype=bool)
        source = np.full(B, -1, dtype=np.int8)
        pending = np.ones(B, dtype=bool)

        with be.device_ctx():
            dev_bounds = [Ld_in, Ud_in] if dev_in else []   # device copies, uploaded on first use
            obj = np.zeros((B, K))
            zout = np.zeros((B, n_out)) if (return_z and n_out) else None

            if dev_in:
                cross = be.to_host(be.xp.any(Ld_in > Ud_in, axis=1)).astype(bool)
            else:
                cross = np.any(Lp > Up, axis=1)
            if cross.any():
                status[cross] = INFEASIBLE
                certified[cross] = True
                source[cross] = SRC_TRIVIAL
                pending[cross] = False
                counts["trivial"] += int(cross.sum())

            def member_bounds(idx: np.ndarray):
                # small groups: host (latency-bound); large groups: fused GPU kernels
                if be.is_gpu and idx.size >= self.gpu_min_batch:
                    if not dev_bounds:
                        dev_bounds.extend([be.asarray(hb.L), be.asarray(hb.U)])
                    if idx.size == B:
                        return dev_bounds[0], dev_bounds[1]
                    idx_d = be.asarray(idx)
                    return dev_bounds[0][idx_d], dev_bounds[1][idx_d]
                return hb.rows(idx)

            out = {"obj": obj, "z": zout}
            # device-side outputs for members certified on the GPU (one download at the end)
            dout = {}
            on_dev = np.zeros(B, dtype=bool)

            def try_entry(e, idx: np.ndarray, src: int) -> int:
                if idx.size == 0:
                    return 0
                if e.host.get("lazy"):
                    e = self._materialise(g, e)
                    if e is None:
                        return 0
                Ls, Us = member_bounds(idx)
                res = certify(e, Ls, Us, be, tol, n_out, zout is not None, lp=lp)
                ok = res.ok
                nok = int(ok.sum())
                if nok and not isinstance(res.obj, np.ndarray):
                    xp = be.xp
                    if not dout:
                        dout["obj"] = xp.zeros((B, K))
                        dout["z"] = xp.zeros((B, n_out)) if zout is not None else None
                        dout["u"] = xp.zeros(B, dtype=bool)
                    sel = idx[ok]
                    if nok == idx.size:
                        sel_d = be.asarray(idx)
                        dout["obj"][sel_d] = res.obj
                        if dout["z"] is not None:
                            dout["z"][sel_d] = res.z_out
                        dout["u"][sel_d] = res.unique
                    else:
                        ok_d = be.asarray(ok)
                        sel_d = be.asarray(sel)
                        dout["obj"][sel_d] = res.obj[ok_d]
                        if dout["z"] is not None:
                            dout["z"][sel_d] = res.z_out[ok_d]
                        dout["u"][sel_d] = res.unique[ok_d]
                    on_dev[sel] = True
                elif nok:
                    sel = idx[ok]
                    out["obj"][sel] = res.obj[ok]
                    if out["z"] is not None:
                        out["z"][sel] = res.z_out[ok]
                    unique[sel] = res.unique[ok]
                if nok:
                    sel = idx[ok] if nok < idx.size else idx
                    status[sel] = OPTIMAL
                    certified[sel] = True
                    basis_id[sel] = e.id
                    source[sel] = src
                    pending[sel] = False
                    pool.touch(e, nok)
                    counts[SOURCE_NAMES[src]] += nok
                return nok

            def try_farkas(f, idx: np.ndarray, src: int) -> int:
                if idx.size == 0:
                    return 0
                bad = check_farkas(f, *hb.rows(idx), tol)
                nb = int(bad.sum())
                if nb:
                    sel = idx[bad]
                    status[sel] = INFEASIBLE
                    certified[sel] = True
                    basis_id[sel] = f.id
                    source[sel] = src
                    pending[sel] = False
                    f.hits += nb
                    f.last_used = pool.clock
                    counts[SOURCE_NAMES[src]] += nb
                return nb

            # ---- 1. temporal warm start -------------------------------------------
            t0 = time.perf_counter()
            if warm_start is not None:
                ws = np.asarray(warm_start, dtype=np.int64).reshape(-1)
                if ws.shape[0] != B:
                    raise ValueError("warm_start has the wrong length")
                for eid in np.unique(ws[pending & (ws >= 0)]):
                    idx = np.nonzero(pending & (ws == eid))[0]
                    e = pool.get(eid)
                    if e is not None:
                        try_entry(e, idx, SRC_WARM)
                    elif int(eid) in pool.farkas:
                        try_farkas(pool.farkas[int(eid)], idx, SRC_WARM)
            else:
                ws = np.full(B, -1, dtype=np.int64)

            # ---- 2. pool scan (hottest bases, then all infeasibility certificates) ----
            if pending.any():
                # hottest bases first, then the bases whose anchors lie nearest the
                # pending members (what makes a loaded atlas replace the cold start)
                idx = np.nonzero(pending)[0]
                cands = pool.ranked(self.pool_scan)
                seen = {id(e) for e in cands}
                pts = np.hstack(hb.rows(idx))
                for e in pool.nearest(pts, self.pool_scan_near):
                    if id(e) not in seen:
                        cands.append(e)
                        seen.add(id(e))
                for e in cands:
                    idx = np.nonzero(pending)[0]
                    if idx.size == 0:
                        break
                    try_entry(e, idx, SRC_POOL)
            if pending.any() and pool.farkas:
                for f in list(pool.farkas.values()):
                    idx = np.nonzero(pending)[0]
                    if idx.size == 0:
                        break
                    try_farkas(f, idx, SRC_POOL)
            t_cert = time.perf_counter() - t0

            # ---- 3. repair + propagation ----------------------------------------------
            t0 = time.perf_counter()
            # Progressive repair: with an empty pool, solve one representative
            # first and double the number per round (1, 2, 4, ... n_workers), so
            # later representatives warm-start from real bases instead of many
            # cold solves rediscovering the same few critical regions.
            n_round = 0 if len(pool) == 0 else 2
            # Yield guard: when fresh bases stop certifying *other* members (the batch
            # is scattered over many critical regions), propagation is pure overhead.
            # Switch the rest of the batch to direct mode -- large parallel chunks, each
            # representative certified against its own basis only -- which bounds the
            # worst case by the cost of warm-started parallel simplex.
            direct = False
            while pending.any():
                idx = np.nonzero(pending)[0]
                n_before = idx.size
                if direct:
                    self._direct_solve(g, idx, hb, ws, pool, rstats, counts, status, certified, unique,
                                       basis_id, source, pending, out)
                    break
                else:
                    k = min(self.n_workers, idx.size, 2 ** n_round) if self.progressive else \
                        min(self.n_workers, idx.size)
                    reps = idx[_pick_representatives(*hb.rows(idx), k)]
                n_round += 1
                Lr, Ur = hb.rows(reps)
                warms = []
                for i, r in enumerate(reps):
                    e = pool.get(ws[r]) if ws[r] >= 0 else None
                    if e is None:
                        near = pool.nearest(np.concatenate([Lr[i], Ur[i]])[None, :], 1)
                        e = near[0] if near else None
                    warms.append(None if e is None else e.status)
                bounds = [lp.full_bounds(Lr[i], Ur[i]) for i in range(len(reps))]

                def job(i):
                    # simplex + (host-side) certificate construction, off the GIL
                    res = self._worker(g).solve(bounds[i][0], bounds[i][1], warms[i])
                    entry = None
                    err = None
                    if res.status == OPTIMAL:
                        raw = basis_key(res.basic, res.zstatus)
                        known = pool.lookup_raw(raw)
                        if known is not None:
                            entry = known
                        else:
                            try:
                                entry = prepare_basis_entry(lp, res.basic, res.zstatus, tol,
                                                            g.out_z, source_bounds=bounds[i])
                                entry.host["raw_key"] = raw
                            except CertificateError as ex:
                                err = ex
                    return res, entry, err

                if self._executor is not None and len(reps) > 1:
                    results = list(self._executor.map(job, range(len(reps))))
                else:
                    results = [job(i) for i in range(len(reps))]

                new_entries, new_farkas = [], []
                rep_entry = [None] * len(reps)      # final pool entry of each representative
                for i, (r, (res, entry, err)) in enumerate(zip(reps, results)):
                    rstats["highs_solves"] += 1
                    rstats["highs_iterations"] += res.iterations
                    rstats["highs_seconds"] += res.seconds
                    if err is not None:
                        rstats["cert_failures"] += 1
                    if entry is not None:
                        if entry.id < 0:          # freshly prepared, not yet in the pool
                            upload_entry(entry, be)
                            entry.anchor = (Lr[i].copy(), Ur[i].copy())
                            eid, is_new = pool.add_basis(entry)
                            pool.register_raw(entry.host.get("raw_key"), eid)
                            rstats["new_bases"] += int(is_new)
                            entry = pool.get(eid)
                        rep_entry[i] = entry
                        if all(entry is not x for x in new_entries):
                            new_entries.append(entry)
                    elif res.status == INFEASIBLE and res.farkas_y is not None:
                        try:
                            f = build_farkas_entry(lp, res.farkas_y)
                            fid = pool.add_farkas(f)
                            f = pool.farkas.get(fid, f)
                            if all(f is not x for x in new_farkas):
                                new_farkas.append(f)
                        except CertificateError:
                            rstats["cert_failures"] += 1
                entries = [r[1] for r in results]
                errors = [r[2] for r in results]
                results = [r[0] for r in results]

                # propagate: the representative itself first, then everybody pending
                if direct:
                    # each representative against the basis its own solve produced
                    own: dict = {}
                    for r, e in zip(reps, rep_entry):
                        if e is not None:
                            own.setdefault(id(e), (e, []))[1].append(r)
                    for e, rs in own.values():
                        rs = np.asarray(rs)
                        try_entry(e, rs[pending[rs]], SRC_REPAIRED)
                    new_entries = []
                for e in new_entries:
                    idx = np.nonzero(pending)[0]
                    if idx.size == 0:
                        break
                    rep_mask = np.isin(idx, reps)
                    if rep_mask.any():
                        try_entry(e, idx[rep_mask], SRC_REPAIRED)
                    idx = np.nonzero(pending)[0]
                    try_entry(e, idx, SRC_PROPAGATED)
                for f in new_farkas:
                    idx = np.nonzero(pending)[0]
                    if idx.size == 0:
                        break
                    rep_mask = np.isin(idx, reps)
                    if rep_mask.any():
                        try_farkas(f, idx[rep_mask], SRC_REPAIRED)
                    idx = np.nonzero(pending)[0]
                    try_farkas(f, idx, SRC_PROPAGATED)

                # representatives our certificates could not cover: take HiGHS' answer
                for i, (r, res) in enumerate(zip(reps, results)):
                    if not pending[r]:
                        continue
                    if self.debug and res.status == OPTIMAL:
                        e = entries[i]
                        info = ("no entry: " + repr(errors[i])) if e is None else \
                            explain_certify(e, lp, Lp[r], Up[r], tol, be)
                        self.debug_log.append({"call": pool.clock, "member": int(r), "info": info})
                    pending[r] = False
                    source[r] = SRC_HIGHS
                    counts["highs_uncertified"] += 1
                    status[r] = res.status
                    if res.status == OPTIMAL:
                        out["obj"][r] = res.stage_obj
                        if out["z"] is not None:
                            out["z"][r] = res.z[g.out_z]
                if not direct:
                    # yield of *this* round: members resolved per repaired basis.  A coherent
                    # batch keeps certifying many members per basis; once fresh bases only
                    # certify themselves, the rest of the batch is incoherent.
                    round_yield = (n_before - int(pending.sum())) / max(len(reps), 1)
                    if n_round >= 2 and round_yield < self.min_yield and pending.sum() > self.n_workers:
                        direct = True
                        rstats["direct_mode"] = 1
            t_rep = time.perf_counter() - t0

            # infeasible / failed members report zeros (the dFBA "no growth" policy)
            if device_out:
                xp = be.xp
                if not dout:
                    dout["obj"] = xp.zeros((B, K))
                    dout["z"] = xp.zeros((B, n_out)) if zout is not None else None
                    dout["u"] = xp.zeros(B, dtype=bool)
                host_rows = np.nonzero(~on_dev & (status == OPTIMAL))[0]
                if host_rows.size:
                    hr = be.asarray(host_rows)
                    dout["obj"][hr] = be.asarray(out["obj"][host_rows])
                    if dout["z"] is not None:
                        dout["z"][hr] = be.asarray(out["z"][host_rows])
                if on_dev.any():
                    unique[on_dev] = be.to_host(dout["u"])[on_dev]
                bad = np.nonzero(status != OPTIMAL)[0]
                if bad.size:
                    bd = be.asarray(bad)
                    dout["obj"][bd] = 0.0
                    if dout["z"] is not None:
                        dout["z"][bd] = 0.0
                out["obj"], out["z"] = dout["obj"], dout["z"]
                dout = {}
            if dout:
                from manylp.basis import _to_host_pinned

                if on_dev.all():
                    out["obj"] = dout["obj"].get()
                    unique[:] = dout["u"].get()
                    if dout["z"] is not None:
                        out["z"] = _to_host_pinned(dout["z"])
                else:
                    sel = np.nonzero(on_dev)[0]
                    sel_d = be.asarray(sel)
                    out["obj"][sel] = dout["obj"][sel_d].get()
                    unique[sel] = dout["u"][sel_d].get()
                    if dout["z"] is not None:
                        out["z"][sel] = _to_host_pinned(dout["z"][sel_d])
            obj_h, z_h = out["obj"], out["z"]
            not_opt = status != OPTIMAL
            if not_opt.any() and not device_out:
                obj_h[not_opt] = 0.0
                if z_h is not None:
                    z_h[not_opt] = 0.0

        stats = {
            "B": B,
            "seconds": time.perf_counter() - t_start,
            "certify_seconds": t_cert,
            "repair_seconds": t_rep,
            "pool_size": len(pool),
            "farkas_size": len(pool.farkas),
            **counts,
            **rstats,
        }
        g.totals["calls"] += 1
        g.totals["members"] += B
        for k in SOURCE_NAMES:
            g.totals[k] += counts[k]
        for k in ("highs_solves", "highs_iterations", "highs_seconds", "cert_failures"):
            g.totals[k] += rstats[k]
        return BatchSolution(status=status, objective=obj_h, z=z_h, basis_id=basis_id,
                             certified=certified, unique=unique, source=source, stats=stats)


def _pick_representatives(L: np.ndarray, U: np.ndarray, k: int, max_points: int = 4096) -> np.ndarray:
    """Greedy farthest-point sampling in (normalised) bound space.

    Diverse representatives maximise how many *other* pending members each
    freshly repaired basis can certify.
    """
    n = L.shape[0]
    if k >= n:
        return np.arange(n)
    X = np.hstack([L, U])
    X = np.where(np.isfinite(X), X, 0.0)
    cand = np.arange(n)
    if n > max_points:
        rng = np.random.default_rng(0)
        cand = np.sort(rng.choice(n, max_points, replace=False))
        X = X[cand]
    span = X.max(axis=0) - X.min(axis=0)
    keep = span > 0
    if not keep.any():
        return cand[:1]
    X = X[:, keep] / span[keep]
    chosen = [0]
    d = np.abs(X - X[0]).sum(axis=1)
    for _ in range(k - 1):
        i = int(np.argmax(d))
        if d[i] <= 1e-12:
            break
        chosen.append(i)
        d = np.minimum(d, np.abs(X - X[i]).sum(axis=1))
    return cand[np.asarray(chosen)]


def lp_fingerprint(lp: LexLP) -> str:
    """Hash identifying a group's LP (matrix, objectives, template bounds, parameters)."""
    import hashlib

    h = hashlib.sha256()
    for a in (lp.A.indptr, lp.A.indices, lp.A.data, lp.objectives, lp.col_lb, lp.col_ub,
              lp.row_lb, lp.row_ub, lp.param_cols, lp.param_rows):
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


class _LazyHost:
    """Host view of a batch's bounds; for device inputs rows are fetched on demand."""

    def __init__(self, Ld, Ud, L=None, U=None):
        self._Ld, self._Ud = Ld, Ud
        self._L, self._U = L, U

    def _fetch_all(self):
        if self._L is None:
            self._L, self._U = self._Ld.get(), self._Ud.get()

    @property
    def L(self):
        self._fetch_all()
        return self._L

    @property
    def U(self):
        self._fetch_all()
        return self._U

    def rows(self, idx):
        idx = np.asarray(idx)
        if self._L is not None:
            if idx.size == self._L.shape[0]:
                return self._L, self._U
            return self._L[idx], self._U[idx]
        if idx.size * 4 >= self._Ld.shape[0]:
            self._fetch_all()
            return self._L[idx], self._U[idx]
        import cupy as cp

        i = cp.asarray(idx)
        return self._Ld[i].get(), self._Ud[i].get()


# ---------------------------------------------------------------------------
# direct mode: per-member solve + point certificate in worker processes
# ---------------------------------------------------------------------------

_DM: dict = {}


def _dm_solve_chunk(key, lp, tol, out_z, Lrows, Urows, warms):
    """Solve and certify a chunk of members in a worker process (no GIL contention)."""
    from manylp.basis import build_farkas_entry, certify_point, check_farkas
    from manylp.repair import HighsLexSolver

    ent = _DM.get(key)
    if ent is None:
        ent = _DM[key] = {"solver": HighsLexSolver(lp, dual_tol=tol.dual), "last": None}
    hs = ent["solver"]
    out = []
    for i in range(Lrows.shape[0]):
        lb, ub = lp.full_bounds(Lrows[i], Urows[i])
        warm = warms[i] if warms[i] is not None else ent["last"]
        r = hs.solve(lb, ub, warm_status=warm)
        rec = {"status": r.status, "certified": False, "unique": False, "obj": None, "z": None,
               "iters": r.iterations, "seconds": r.seconds}
        if r.status == OPTIMAL:
            ent["last"] = r.zstatus
            rec["basic"], rec["zstatus"] = r.basic, r.zstatus
            ok, uniq, z, obj = certify_point(lp, r.basic, r.zstatus, lb, ub, tol)
            if ok:
                rec.update(certified=True, unique=uniq, obj=obj, z=z[out_z])
            else:
                rec.update(obj=r.stage_obj, z=r.z[out_z])
        elif r.status == INFEASIBLE and r.farkas_y is not None:
            try:
                f = build_farkas_entry(lp, r.farkas_y)
                rec["certified"] = bool(check_farkas(f, Lrows[i:i + 1], Urows[i:i + 1], tol)[0])
            except Exception:
                pass
        out.append(rec)
    return out
