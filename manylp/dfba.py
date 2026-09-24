"""Batched community dynamic FBA (static optimisation approach) for benchmarking.

This reproduces the per-step semantics of muODE's ``DynamicFBA`` engine (and of
COMETS-style shared-pool community dFBA) for ``E`` ensemble members advanced
in lockstep, so that one time step is **one batched solve per species**:

1. uptake bound of every exchange ``j`` of species ``i``::

       vmax_ij = min( Vmax * M_j / (Km + M_j),  diet_limit_j,  M_j / (X_i * dt) )
       lb_ij   = -vmax_ij                      (upper bounds stay at the template)

2. solve each species' FBA LP (growth ``mu_i``, exchange fluxes ``v_ij``);
   infeasible -> ``mu = 0`` and zero fluxes; species below ``min_biomass`` are dormant;
3. explicit Euler::

       X_i += (mu_i - D) X_i dt
       M_j += (influx_j - D M_j + sum_i v_ij X_i) dt,   clamped at 0

The LP solver is pluggable (:class:`FBASolverAdapter`) so every baseline sees the
*identical* sequence of LPs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from manylp.fba import FBAModel, compile_fba


def exchange_metabolites(model: FBAModel) -> List[str]:
    """Metabolite id carried by each exchange reaction of ``model`` (in order)."""
    S = model.S.tocsc()
    out = []
    for j in model.exchanges:
        rows = S.indices[S.indptr[j]:S.indptr[j + 1]]
        if rows.size != 1:
            raise ValueError(f"exchange {j} of {model.name} touches {rows.size} metabolites")
        out.append(model.met_ids[rows[0]])
    return out


class FBASolverAdapter:
    """Interface: solve all members of one species for given exchange lower bounds."""

    name = "abstract"

    def setup(self, models: Sequence[FBAModel], mode: str) -> None:
        raise NotImplementedError

    def solve(self, s: int, ex_lb: np.ndarray, member_ids: np.ndarray):
        """Return ``(growth (Ea,), ex_flux (Ea, k_s), feasible (Ea,))``."""
        raise NotImplementedError

    def close(self) -> None:
        pass

    def stats(self) -> dict:
        return {}


class ManyLPAdapter(FBASolverAdapter):
    """The certify-and-repair batched solver (CPU or GPU)."""

    def __init__(self, device: str = "cpu", n_workers: int = 16, per_lp: bool = False,
                 atlas_dir: Optional[str] = None, **kw) -> None:
        from manylp.solver import BatchLPSolver

        self.atlas_dir = atlas_dir
        if per_lp:
            kw.setdefault("gpu_min_batch", 1)   # force the device path even for one LP
        self.solver = BatchLPSolver(device=device, n_workers=n_workers, **kw)
        self.per_lp = per_lp
        self.name = f"manylp-{device.split(':')[0]}" + ("-perlp" if per_lp else "")

    def setup(self, models, mode):
        self.problems = [compile_fba(m, mode=mode) for m in models]
        self.groups = [self.solver.register_group(p.lp, out_z=p.out_z(p.model.exchanges))
                       for p in self.problems]
        # warm-start ids per (species, member); grown lazily
        self.warm: List[Optional[np.ndarray]] = [None] * len(models)
        self.calls = []
        self.atlas_loaded = 0
        if self.atlas_dir:
            for g in self.groups:
                self.atlas_loaded += self.solver.load_atlas(g, self._atlas_path(g))

    def _atlas_path(self, g) -> str:
        import os

        from manylp.solver import lp_fingerprint

        os.makedirs(self.atlas_dir, exist_ok=True)
        return os.path.join(self.atlas_dir, lp_fingerprint(g.lp)[:24] + ".npz")

    def save_atlas(self) -> int:
        return sum(self.solver.save_atlas(g, self._atlas_path(g)) for g in self.groups)

    def solve(self, s, ex_lb, member_ids):
        if self.per_lp and member_ids.size > 1:
            # naive GPU use: the same solver, one LP per call (no batching)
            parts = [self.solve(s, ex_lb[i:i + 1], member_ids[i:i + 1]) for i in range(member_ids.size)]
            return (np.concatenate([q[0] for q in parts]), np.concatenate([q[1] for q in parts]),
                    np.concatenate([q[2] for q in parts]))
        p, g = self.problems[s], self.groups[s]
        be = self.solver.backend
        if not (be.is_gpu and member_ids.size >= self.solver.gpu_min_batch):
            Lp, Up = p.param_bounds(ex_lb)
        need = int(member_ids.max()) + 1 if member_ids.size else 0
        w = self.warm[s]
        if w is None or w.size < need:
            grow = np.full(need, -1, dtype=np.int64)
            if w is not None:
                grow[: w.size] = w
            w = self.warm[s] = grow
        be = self.solver.backend
        if be.is_gpu and member_ids.size >= self.solver.gpu_min_batch:
            # device-resident: upload the (B x k) exchange bounds once, build the
            # LP bounds and the fluxes on the GPU, download only (B x k) fluxes
            with be.device_ctx():
                Ld, Ud = p.param_bounds(be.asarray(ex_lb))
                sol = self.solver.solve_batch(g, Ld, Ud, warm_start=w[member_ids], device_out=True)
                flux = be.to_host(p.fluxes(sol.z, p.model.exchanges))
                obj0 = be.to_host(sol.objective[:, 0])
        else:
            sol = self.solver.solve_batch(g, Lp, Up, warm_start=w[member_ids])
            flux = p.fluxes(sol.z, p.model.exchanges) if sol.z is not None else None
            obj0 = sol.objective[:, 0]
        w[member_ids] = sol.basis_id
        self.calls.append(sol.stats)
        feas = sol.status == 1
        growth = np.where(feas, obj0, 0.0)
        return growth, flux, feas

    def stats(self):
        tot = {}
        for c in self.calls:
            for k, v in c.items():
                if isinstance(v, (int, float)) and k not in ("pool_size", "farkas_size"):
                    tot[k] = tot.get(k, 0) + v
        tot["pool_sizes"] = [len(g.pool) for g in self.groups]
        tot["unique_all"] = True
        return tot

    def close(self):
        self.solver.close()


class HighsLoopAdapter(FBASolverAdapter):
    """Per-LP HiGHS: persistent model per species, optionally warm, optionally threaded."""

    def __init__(self, warm: bool = True, n_workers: int = 1) -> None:
        self.warm_start = warm
        self.n_workers = n_workers
        self.name = f"highs-{'warm' if warm else 'cold'}-x{n_workers}"

    def setup(self, models, mode):
        import threading
        from concurrent.futures import ThreadPoolExecutor

        self.problems = [compile_fba(m, mode=mode) for m in models]
        self._tls = threading.local()
        self._ex = ThreadPoolExecutor(self.n_workers) if self.n_workers > 1 else None
        self.iters = 0
        self.lps = 0
        self.last_basis: Dict[tuple, np.ndarray] = {}

    def _solver(self, s):
        from manylp.repair import HighsLexSolver

        d = getattr(self._tls, "d", None)
        if d is None:
            d = self._tls.d = {}
        if s not in d:
            d[s] = HighsLexSolver(self.problems[s].lp)
        return d[s]

    def solve(self, s, ex_lb, member_ids):
        p = self.problems[s]
        Lp, Up = p.param_bounds(ex_lb)
        lp = p.lp
        oz = p.out_z(p.model.exchanges)

        def one(b):
            lb, ub = lp.full_bounds(Lp[b], Up[b])
            key = (s, int(member_ids[b]))
            warm = self.last_basis.get(key) if self.warm_start else None
            r = self._solver(s).solve(lb, ub, warm_status=warm)
            if r.status == 1 and self.warm_start:
                self.last_basis[key] = r.zstatus
            return r

        idx = range(len(member_ids))
        res = list(self._ex.map(one, idx)) if self._ex is not None else [one(b) for b in idx]
        feas = np.array([r.status == 1 for r in res])
        growth = np.array([r.stage_obj[0] if r.status == 1 else 0.0 for r in res])
        Z = np.array([r.z[oz] if r.status == 1 else np.zeros(oz.size) for r in res])
        self.iters += sum(r.iterations for r in res)
        self.lps += len(res)
        return growth, p.fluxes(Z, p.model.exchanges), feas

    def stats(self):
        return {"highs_iterations": self.iters, "lps": self.lps}

    def close(self):
        if self._ex is not None:
            self._ex.shutdown()


_PROC = {}


def _proc_init(problems):
    import os

    os.environ["OMP_NUM_THREADS"] = "1"
    _PROC["problems"] = problems
    _PROC["solvers"] = {}
    _PROC["basis"] = {}


def _proc_solve(s, ex_lb, member_ids, warm_start):
    from manylp.repair import HighsLexSolver

    p = _PROC["problems"][s]
    sv = _PROC["solvers"].get(s)
    if sv is None:
        sv = _PROC["solvers"][s] = HighsLexSolver(p.lp)
    Lp, Up = p.param_bounds(ex_lb)
    oz = p.out_z(p.model.exchanges)
    B = Lp.shape[0]
    growth = np.zeros(B)
    Z = np.zeros((B, oz.size))
    feas = np.zeros(B, dtype=bool)
    iters = 0
    for b in range(B):
        lb, ub = p.lp.full_bounds(Lp[b], Up[b])
        key = (s, int(member_ids[b]))
        r = sv.solve(lb, ub, warm_status=_PROC["basis"].get(key) if warm_start else None)
        iters += r.iterations
        if r.status == 1:
            feas[b] = True
            growth[b] = r.stage_obj[0]
            Z[b] = r.z[oz]
            if warm_start:
                _PROC["basis"][key] = r.zstatus
    return growth, p.fluxes(Z, p.model.exchanges), feas, iters


class HighsProcAdapter(FBASolverAdapter):
    """Strong CPU baseline: warm-started HiGHS in ``n_procs`` *processes* (no GIL).

    Each ensemble member is pinned to one process (``member % n_procs``), which
    keeps a persistent HiGHS model per species and that member's last optimal
    basis, so every LP is a warm-started dual simplex from its own previous step.
    """

    def __init__(self, n_procs: int = 32, warm: bool = True) -> None:
        self.n_procs = n_procs
        self.warm_start = warm
        self.name = f"highs-{'warm' if warm else 'cold'}-p{n_procs}"

    def setup(self, models, mode):
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        self.problems = [compile_fba(m, mode=mode) for m in models]
        ctx = mp.get_context("fork")
        self._ex = [ProcessPoolExecutor(1, mp_context=ctx, initializer=_proc_init,
                                        initargs=(self.problems,)) for _ in range(self.n_procs)]
        self.iters = 0
        self.lps = 0

    def solve(self, s, ex_lb, member_ids):
        k = self.problems[s].model.exchanges.size
        E = member_ids.size
        growth = np.zeros(E)
        flux = np.zeros((E, k))
        feas = np.zeros(E, dtype=bool)
        owner = member_ids % self.n_procs
        futs = []
        for w in np.unique(owner):
            idx = np.nonzero(owner == w)[0]
            futs.append((idx, self._ex[w].submit(_proc_solve, s, ex_lb[idx], member_ids[idx],
                                                  self.warm_start)))
        for idx, f in futs:
            g, v, ok, it = f.result()
            growth[idx], flux[idx], feas[idx] = g, v, ok
            self.iters += it
        self.lps += E
        return growth, flux, feas

    def stats(self):
        return {"highs_iterations": self.iters, "lps": self.lps}

    def close(self):
        for ex in self._ex:
            ex.shutdown()


class ScipyLinprogAdapter(FBASolverAdapter):
    """muODE's current path: one ``scipy.optimize.linprog(method="highs")`` per LP (FBA only)."""

    name = "scipy-linprog"

    def setup(self, models, mode):
        if mode != "fba":
            raise ValueError("scipy adapter implements plain FBA only (muODE's LinprogOrganism)")
        self.models = models

    def solve(self, s, ex_lb, member_ids):
        from scipy.optimize import linprog

        mdl = self.models[s]
        k = mdl.exchanges.size
        E = ex_lb.shape[0]
        growth = np.zeros(E)
        flux = np.zeros((E, k))
        feas = np.zeros(E, dtype=bool)
        bm = np.zeros(mdl.m)
        for b in range(E):
            lb = mdl.lb.copy()
            lb[mdl.exchanges] = ex_lb[b]
            res = linprog(-mdl.c, A_eq=mdl.S, b_eq=bm, bounds=np.stack([lb, mdl.ub], axis=1),
                          method="highs")
            if res.status == 0:
                feas[b] = True
                growth[b] = mdl.c @ res.x
                flux[b] = res.x[mdl.exchanges]
        return growth, flux, feas


class CobraAdapter(FBASolverAdapter):
    """muODE's production path: a ``cobra.Model`` per species, ``optimize()`` (or pFBA) per LP."""

    def __init__(self, sbml_files: Sequence[str], solver: str = "glpk") -> None:
        self.files = list(sbml_files)
        self.solver_name = solver
        self.name = f"cobra-{solver}"

    def setup(self, models, mode):
        import cobra

        if mode not in ("fba", "pfba"):
            raise ValueError("cobra adapter supports fba and pfba")
        self.mode = mode
        self.cmodels = []
        for f in self.files:
            mdl = cobra.io.read_sbml_model(f)
            mdl.solver = self.solver_name
            self.cmodels.append(mdl)
        self.models = models
        self.base = [np.array([r.lower_bound for r in mdl.reactions]) for mdl in self.cmodels]

    def solve(self, s, ex_lb, member_ids):
        from cobra.flux_analysis import pfba

        mdl, fm = self.cmodels[s], self.models[s]
        rx = [mdl.reactions[j] for j in fm.exchanges]
        E = ex_lb.shape[0]
        growth = np.zeros(E)
        flux = np.zeros((E, fm.exchanges.size))
        feas = np.zeros(E, dtype=bool)
        bm = int(np.argmax(fm.c))
        for b in range(E):
            for r, v in zip(rx, ex_lb[b]):
                r.lower_bound = float(v)
            try:
                sol = pfba(mdl) if self.mode == "pfba" else mdl.optimize()
            except Exception:
                continue
            if sol.status != "optimal":
                continue
            fx = sol.fluxes.values
            feas[b] = True
            growth[b] = fx[bm] if self.mode == "pfba" else sol.objective_value
            flux[b] = fx[fm.exchanges]
        return growth, flux, feas


@dataclass
class Community:
    models: List[FBAModel]
    env_mets: List[str]
    M0: np.ndarray                 # (nM,)
    influx: np.ndarray             # (nM,)
    max_uptake: np.ndarray         # (nM,) inf where unconstrained
    X0: np.ndarray                 # (S,)
    vmax: float = 10.0
    km: float = 0.01
    ex_env: List[np.ndarray] = field(default_factory=list)   # per species: env index of each exchange

    def __post_init__(self):
        idx = {m: i for i, m in enumerate(self.env_mets)}
        if not self.ex_env:
            self.ex_env = [np.array([idx[m] for m in exchange_metabolites(md)]) for md in self.models]


@dataclass
class DFBAResult:
    times: np.ndarray
    X: np.ndarray            # (T, E, S)
    M: np.ndarray            # (T, E, nM)
    mu: np.ndarray           # (T, E, S)
    solve_seconds: float
    wall_seconds: float
    n_lps: int
    solver_stats: dict


def run_dfba(comm: Community, adapter: FBASolverAdapter, E: int = 1, t_end: float = 10.0,
             dt: float = 0.1, mode: str = "pfba-unique", dilution: float = 0.0,
             min_biomass: float = 1e-9, perturb: Optional[np.ndarray] = None,
             setup: bool = True, record_every: int = 1) -> DFBAResult:
    """Integrate ``E`` ensemble members.  ``perturb`` (E, nM) scales initial concentrations."""
    S = len(comm.models)
    nM = len(comm.env_mets)
    if setup:
        adapter.setup(comm.models, mode)
    X = np.tile(comm.X0[None, :], (E, 1)).astype(float)
    M = np.tile(comm.M0[None, :], (E, 1)).astype(float)
    if perturb is not None:
        M *= perturb
    n_steps = int(round(t_end / dt))
    rec = list(range(0, n_steps + 1, record_every))
    Xh = np.zeros((len(rec), E, S))
    Mh = np.zeros((len(rec), E, nM))
    muh = np.zeros((len(rec), E, S))
    t_solve = 0.0
    n_lps = 0
    t0 = time.perf_counter()
    ri = 0
    for step in range(n_steps + 1):
        mu = np.zeros((E, S))
        flux_env = np.zeros((E, nM))  # sum_i v_ij X_i
        for s in range(S):
            active = np.nonzero(X[:, s] > min_biomass)[0]
            if active.size == 0:
                continue
            envi = comm.ex_env[s]
            conc = M[np.ix_(active, envi)]
            mm = np.where(conc > 0, comm.vmax * conc / (comm.km + conc), 0.0)
            mm = np.minimum(mm, comm.max_uptake[envi][None, :])
            cap = conc / (X[active, s][:, None] * dt) if dt > 0 else np.inf
            ex_lb = -np.minimum(mm, cap)
            ts = time.perf_counter()
            growth, v, feas = adapter.solve(s, ex_lb, active)
            t_solve += time.perf_counter() - ts
            n_lps += active.size
            mu[active, s] = growth
            flux_env[np.ix_(active, envi)] += v * X[active, s][:, None]
        if ri < len(rec) and rec[ri] == step:
            Xh[ri], Mh[ri], muh[ri] = X, M, mu
            ri += 1
        if step == n_steps:
            break
        X = np.maximum(0.0, X + (mu - dilution) * X * dt)
        M = np.maximum(0.0, M + (comm.influx[None, :] - dilution * M + flux_env) * dt)
    wall = time.perf_counter() - t0
    return DFBAResult(times=np.array(rec) * dt, X=Xh, M=Mh, mu=muh, solve_seconds=t_solve,
                      wall_seconds=wall, n_lps=n_lps, solver_stats=adapter.stats())


def load_gut_community(gem_dir: str, diet_csv: str, names: Optional[Sequence[str]] = None,
                       abundance_tsv: Optional[str] = None, total_biomass: float = 0.01,
                       vmax: float = 10.0, km: float = 0.01) -> Community:
    """Build the muODE ``gut_western`` community (gapseq GEMs + ModelSEED western diet)."""
    import csv
    import glob
    import os

    import cobra

    files = sorted(glob.glob(os.path.join(gem_dir, "*.xml.gz")))
    if names is not None:
        files = [f for f in files if any(os.path.basename(f).startswith(n) for n in names)]
    models = []
    for f in files:
        mdl = cobra.io.read_sbml_model(f)
        fm = FBAModel.from_cobra(mdl)
        fm.name = os.path.basename(f).replace(".xml.gz", "")
        models.append(fm)
    env = sorted({m for fm in models for m in exchange_metabolites(fm)})
    conc, infl, cap = {}, {}, {}
    with open(diet_csv) as fh:
        rows = [ln for ln in fh if not ln.startswith("#")]
    for r in csv.DictReader(rows):
        r = {k.strip().lower(): (v or "").strip() for k, v in r.items()}
        mid = r["metabolite"]
        conc[mid] = float(r.get("concentration") or 0.0)
        infl[mid] = float(r.get("influx") or 0.0)
        if r.get("max_uptake"):
            cap[mid] = float(r["max_uptake"])
    env = sorted(set(env) | set(conc))
    M0 = np.array([conc.get(m, 0.0) for m in env])
    influx = np.array([infl.get(m, 0.0) for m in env])
    max_up = np.array([cap.get(m, np.inf) for m in env])
    ab = np.ones(len(models))
    if abundance_tsv:
        vals = {}
        with open(abundance_tsv) as fh:
            for ln in fh:
                parts = ln.strip().split("\t")
                if len(parts) >= 2 and not ln.startswith("#"):
                    try:
                        vals[parts[0]] = float(parts[1])
                    except ValueError:
                        pass
        ab = np.array([vals.get(fm.name, 1.0) for fm in models])
    X0 = total_biomass * ab / ab.sum()
    return Community(models=models, env_mets=env, M0=M0, influx=influx, max_uptake=max_up,
                     X0=X0, vmax=vmax, km=km)
