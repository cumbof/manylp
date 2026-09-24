"""Classical bunching baseline (stochastic programming; Wets 1983; Kall & Wallace 1994, section 3.10).

When only bounds change, a basis that is optimal for one LP stays dual feasible for
every LP of the family.  Bunching keeps the optimal bases found so far and, for each
new LP, tries them one at a time: fix the nonbasic variables at the bounds their
status selects, solve ``B z_B = -N z_N`` with the basis' cached sparse LU factors,
and accept the basis if ``z_B`` lies within the LP's bounds.  On a miss, a
warm-started dual simplex (HiGHS) solves the LP and its basis joins the cache.

This is manylp's primal feasibility test applied per LP and sequentially.  It has
none of manylp's additions: no batched matrix products, no representative selection
or propagation, no lexicographic stages, no Farkas certificates for infeasible LPs,
no persistent atlas.  Each LP checks its own previous basis first, then the most
recently used cached bases, up to ``max_checks`` (default 32, roughly the cost of
one warm simplex solve in LU solves).

Adapters: :class:`Bunching` (per-LP, for ``cmp_solvers.ProcPool``),
:class:`BunchingProcAdapter` (dFBA driver), :func:`bunching_pool_solve` (Netlib).
"""

from __future__ import annotations

import os
import time

import numpy as np
from scipy.sparse.linalg import splu

from manylp.basis import AT_UPPER, AT_ZERO
from manylp.repair import OPTIMAL, HighsLexSolver


class _Basis:
    __slots__ = ("basic", "nb", "st", "zst", "lu", "A_nb")

    def __init__(self, lp, basic, zstatus):
        n_z = lp.Abar.shape[1]
        self.basic = np.asarray(basic, dtype=np.int64)
        mask = np.ones(n_z, dtype=bool)
        mask[self.basic] = False
        self.nb = np.nonzero(mask)[0]
        self.st = np.asarray(zstatus)[self.nb]
        self.zst = np.asarray(zstatus).copy()
        self.lu = splu(lp.Abar[:, self.basic].tocsc())
        self.A_nb = lp.Abar[:, self.nb].tocsc()


class BunchingSolver:
    """Sequential bunching over one LP family (one objective, bounds vary)."""

    def __init__(self, lp, max_checks: int = 32, max_bases: int = 512, tol: float = 1e-9):
        self.lp = lp
        self.max_checks = max_checks
        self.max_bases = max_bases
        self.tol = tol
        self.c = np.asarray(lp.C_z[0], dtype=float)
        self.cache: list = []          # most recently used first
        self.last: dict = {}           # member key -> its last optimal _Basis
        self.hs = HighsLexSolver(lp)
        self.checks = self.hits = self.solves = self.failed = 0

    def _check(self, b, lb, ub):
        zN = np.where(b.st == AT_UPPER, ub[b.nb], np.where(b.st == AT_ZERO, 0.0, lb[b.nb]))
        if not np.all(np.isfinite(zN)):
            return None
        zB = b.lu.solve(-(b.A_nb @ zN))
        t = self.tol * np.maximum(1.0, np.abs(zB))
        if np.all(zB >= lb[b.basic] - t) and np.all(zB <= ub[b.basic] + t):
            z = np.empty(self.lp.Abar.shape[1])
            z[b.basic] = zB
            z[b.nb] = zN
            return z
        return None

    def solve(self, lb, ub, key=None):
        """Return ``(status, objective, z)``; status 1 = optimal (``z`` full z-space)."""
        own = self.last.get(key)
        cand = [own] if own is not None else []
        for b in self.cache:
            if len(cand) >= self.max_checks:
                break
            if b is not own:
                cand.append(b)
        for b in cand:
            self.checks += 1
            z = self._check(b, lb, ub)
            if z is not None:
                self.hits += 1
                if self.cache and self.cache[0] is not b:
                    self.cache.remove(b)
                    self.cache.insert(0, b)
                self.last[key] = b
                return OPTIMAL, float(self.c @ z), z
        # miss: warm-started dual simplex from this LP's previous basis, else the most recent one
        warm = own.zst if own is not None else (self.cache[0].zst if self.cache else None)
        r = self.hs.solve(lb, ub, warm_status=warm)
        self.solves += 1
        if r.status != OPTIMAL:
            return r.status, np.nan, None
        try:
            b = _Basis(self.lp, r.basic, r.zstatus)
        except RuntimeError:           # singular factorisation: answer without caching
            self.failed += 1
            return OPTIMAL, float(r.stage_obj[0]), r.z
        self.cache.insert(0, b)
        del self.cache[self.max_bases:]
        self.last[key] = b
        return OPTIMAL, float(r.stage_obj[0]), r.z

    def stats(self):
        return {"checks": self.checks, "hits": self.hits, "simplex_solves": self.solves,
                "bases": len(self.cache), "uncached": self.failed}


# ---------------------------------------------------------------------------
# cmp_solvers adapter (per-LP; wrap in ProcPool for 1 or 32 processes)
# ---------------------------------------------------------------------------


class Bunching:
    """Plain-FBA bunching for the recorded workloads (``bench_solvers.py``)."""

    name = "bunching"
    batch = False
    device = "cpu"
    warm = True

    def __init__(self, max_checks: int = 32):
        self.max_checks = max_checks
        self.name = f"bunching-k{max_checks}"

    def setup(self, models):
        from manylp.fba import compile_fba

        self.models = models
        self.probs = [compile_fba(m, mode="fba") for m in models]
        self.solvers = {}

    def solve(self, s, ex_lb, member_ids):
        p = self.probs[s]
        sv = self.solvers.get(s)
        if sv is None:
            sv = self.solvers[s] = BunchingSolver(p.lp, max_checks=self.max_checks)
        Lp, Up = p.param_bounds(ex_lb)
        oz = p.out_z()
        B = ex_lb.shape[0]
        ok = np.zeros(B, dtype=bool)
        obj = np.zeros(B)
        Z = np.zeros((B, oz.size))
        for b in range(B):
            st, o, z = sv.solve(*p.lp.full_bounds(Lp[b], Up[b]), key=int(member_ids[b]))
            if st == OPTIMAL:
                ok[b], obj[b], Z[b] = True, o, z[oz]
        return ok, obj, p.fluxes(Z)

    def stats(self):
        tot = {}
        for sv in self.solvers.values():
            for k, v in sv.stats().items():
                tot[k] = tot.get(k, 0) + v
        return tot

    def close(self):
        pass


# ---------------------------------------------------------------------------
# dFBA driver adapter (bench_dfba.py): members pinned to worker processes
# ---------------------------------------------------------------------------

_P: dict = {}


def _dinit(problems, max_checks):
    os.environ["OMP_NUM_THREADS"] = "1"
    _P["problems"] = problems
    _P["solvers"] = {}
    _P["k"] = max_checks


def _dsolve(s, ex_lb, member_ids):
    p = _P["problems"][s]
    sv = _P["solvers"].get(s)
    if sv is None:
        sv = _P["solvers"][s] = BunchingSolver(p.lp, max_checks=_P["k"])
    Lp, Up = p.param_bounds(ex_lb)
    oz = p.out_z(p.model.exchanges)
    B = Lp.shape[0]
    growth = np.zeros(B)
    Z = np.zeros((B, oz.size))
    feas = np.zeros(B, dtype=bool)
    for b in range(B):
        st, o, z = sv.solve(*p.lp.full_bounds(Lp[b], Up[b]), key=int(member_ids[b]))
        if st == OPTIMAL:
            feas[b], growth[b], Z[b] = True, o, z[oz]
    return growth, p.fluxes(Z, p.model.exchanges), feas


def _dstats():
    tot = {}
    for sv in _P["solvers"].values():
        for k, v in sv.stats().items():
            tot[k] = tot.get(k, 0) + v
    return tot


class BunchingProcAdapter:
    """Bunching in ``n_procs`` processes, members pinned by ``member % n_procs`` (plain FBA)."""

    def __init__(self, n_procs: int = 32, max_checks: int = 32):
        self.n_procs = n_procs
        self.max_checks = max_checks
        self.name = f"bunching-p{n_procs}"

    def setup(self, models, mode):
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        from manylp.fba import compile_fba

        if mode != "fba":
            raise ValueError("the bunching baseline implements plain FBA (one objective) only")
        self.problems = [compile_fba(m, mode=mode) for m in models]
        ctx = mp.get_context("fork")
        self._ex = [ProcessPoolExecutor(1, mp_context=ctx, initializer=_dinit,
                                        initargs=(self.problems, self.max_checks)) for _ in range(self.n_procs)]

    def solve(self, s, ex_lb, member_ids):
        k = self.problems[s].model.exchanges.size
        E = member_ids.size
        growth, flux, feas = np.zeros(E), np.zeros((E, k)), np.zeros(E, dtype=bool)
        owner = member_ids % self.n_procs
        futs = []
        for w in np.unique(owner):
            idx = np.nonzero(owner == w)[0]
            futs.append((idx, self._ex[w].submit(_dsolve, s, ex_lb[idx], member_ids[idx])))
        for idx, f in futs:
            growth[idx], flux[idx], feas[idx] = f.result()
        return growth, flux, feas

    def stats(self):
        tot = {}
        for ex in self._ex:
            for k, v in ex.submit(_dstats).result().items():
                tot[k] = tot.get(k, 0) + v
        return tot

    def close(self):
        for ex in self._ex:
            ex.shutdown()


# ---------------------------------------------------------------------------
# Netlib (bench_netlib.py): scenarios split into contiguous chunks per process
# ---------------------------------------------------------------------------

_N: dict = {}


def _ninit(lp, max_checks):
    os.environ["OMP_NUM_THREADS"] = "1"
    _N["sv"] = BunchingSolver(lp, max_checks=max_checks)


def _nsolve(args):
    L, U = args
    sv = _N["sv"]
    st = np.zeros(L.shape[0], dtype=np.int64)
    obj = np.full(L.shape[0], np.nan)
    for b in range(L.shape[0]):
        st[b], obj[b], _ = sv.solve(*sv.lp.full_bounds(L[b], U[b]), key=None)
    return st, obj, sv.stats()


def bunching_pool_solve(lp, L, U, n_procs: int = 32, max_checks: int = 32):
    """Solve all scenarios; returns ``(status, objective, stats, seconds)``."""
    import multiprocessing as mp

    t0 = time.perf_counter()
    if n_procs <= 1:
        _ninit(lp, max_checks)
        st, obj, stats = _nsolve((L, U))
        return st, obj, stats, time.perf_counter() - t0
    ctx = mp.get_context("spawn")        # the caller may hold a CUDA context
    k = min(L.shape[0], n_procs)
    cuts = np.linspace(0, L.shape[0], k + 1).astype(int)
    with ctx.Pool(k, initializer=_ninit, initargs=(lp, max_checks)) as pool:
        t0 = time.perf_counter()     # exclude process start-up, as for the HiGHS baseline pool
        parts = pool.map(_nsolve, [(L[a:b], U[a:b]) for a, b in zip(cuts[:-1], cuts[1:])])
        secs = time.perf_counter() - t0
    tot = {}
    for _, _, s in parts:
        for kk, v in s.items():
            tot[kk] = tot.get(kk, 0) + v
    return (np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]), tot, secs)
