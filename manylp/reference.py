"""Independent CPU ground truth and CPU baselines.

:func:`reference_solve` implements lexicographic optimisation the textbook way
-- after each stage, add the row ``c_k^T x >= z_k* - delta`` and optimise the next
objective -- with a *fresh* HiGHS model per member.  It shares no code path with
the certify-and-repair solver (which uses reduced-cost fixing and cached bases),
so agreement between the two is meaningful validation.

:class:`HighsBaseline` is the *strong* CPU baseline: one persistent HiGHS model per
process, only bounds change between solves (so HiGHS warm-starts from its own
previous basis), lexicographic stages by reduced-cost fixing, and
``multiprocessing`` across cores.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from manylp.lp import LexLP
from manylp.repair import ERROR, INFEASIBLE, OPTIMAL, UNBOUNDED


@dataclass
class RefResult:
    status: int
    stage_obj: np.ndarray
    x: Optional[np.ndarray]


def reference_solve(lp: LexLP, lb_z: np.ndarray, ub_z: np.ndarray,
                    rel_slack: float = 0.0) -> RefResult:
    """Lexicographic optimum of one member by objective-bound rows (fresh HiGHS).

    ``rel_slack`` relaxes each objective row to ``c_k^T x >= z_k - rel_slack*|z_k|``.
    Keep it at 0: lexicographic trade-offs amplify any slack (a ``1e-9`` slack
    moved later stages by ``~1e-6`` in tests).  If a stage turns numerically
    infeasible, it is retried once with slack ``1e-10``.
    """
    if rel_slack == 0.0:
        r = _reference_solve(lp, lb_z, ub_z, 0.0)
        if r.status == ERROR:
            r = _reference_solve(lp, lb_z, ub_z, 1e-10)
        return r
    return _reference_solve(lp, lb_z, ub_z, rel_slack)


def _reference_solve(lp: LexLP, lb_z: np.ndarray, ub_z: np.ndarray, rel_slack: float) -> RefResult:
    import highspy

    n, m, K = lp.n, lp.m, lp.K
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", 1)
    h.setOptionValue("primal_feasibility_tolerance", 1e-9)
    h.setOptionValue("dual_feasibility_tolerance", 1e-9)
    model = highspy.HighsLp()
    A = lp.A
    model.num_col_ = n
    model.num_row_ = m
    model.col_cost_ = lp.objectives[0].copy()
    model.col_lower_ = lb_z[:n].copy()
    model.col_upper_ = ub_z[:n].copy()
    model.row_lower_ = lb_z[n:].copy()
    model.row_upper_ = ub_z[n:].copy()
    model.sense_ = highspy.ObjSense.kMaximize
    model.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    model.a_matrix_.num_col_ = n
    model.a_matrix_.num_row_ = m
    model.a_matrix_.start_ = A.indptr.astype(np.int32)
    model.a_matrix_.index_ = A.indices.astype(np.int32)
    model.a_matrix_.value_ = A.data.copy()
    h.passModel(model)
    MS = highspy.HighsModelStatus
    stage_obj = np.zeros(K)
    for k in range(K):
        if k > 0:
            h.changeColsCost(n, np.arange(n, dtype=np.int32), lp.objectives[k])
            prev = lp.objectives[k - 1]
            nz = np.nonzero(prev)[0].astype(np.int32)
            zk = stage_obj[k - 1]
            h.addRow(zk - rel_slack * max(1.0, abs(zk)), highspy.kHighsInf,
                     nz.size, nz, prev[nz])
        h.run()
        ms = h.getModelStatus()
        if ms != MS.kOptimal:
            if k == 0 and ms in (MS.kInfeasible, MS.kUnboundedOrInfeasible):
                return RefResult(INFEASIBLE, stage_obj, None)
            if ms == MS.kUnbounded:
                return RefResult(UNBOUNDED, stage_obj, None)
            return RefResult(ERROR, stage_obj, None)
        x = np.asarray(h.getSolution().col_value)
        stage_obj[k] = float(lp.objectives[k] @ x)
    return RefResult(OPTIMAL, stage_obj, x)


def reference_batch(lp: LexLP, Lp: np.ndarray, Up: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Loop :func:`reference_solve` over a batch -> ``(status, stage_obj, x)``."""
    B = Lp.shape[0]
    status = np.zeros(B, dtype=np.int8)
    obj = np.zeros((B, lp.K))
    X = np.zeros((B, lp.n))
    for b in range(B):
        lb, ub = lp.full_bounds(Lp[b], Up[b])
        r = reference_solve(lp, lb, ub)
        status[b] = r.status
        obj[b] = r.stage_obj
        if r.x is not None:
            X[b] = r.x
    return status, obj, X


# ---------------------------------------------------------------------------
# strong CPU baseline: persistent, warm-started, multi-process HiGHS
# ---------------------------------------------------------------------------

_G = {}


def _init_worker(lp: LexLP, out_z: np.ndarray, warm: bool) -> None:
    from manylp.repair import HighsLexSolver

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _G["solver"] = HighsLexSolver(lp)
    _G["lp"] = lp
    _G["out_z"] = out_z
    _G["warm"] = warm


def _solve_chunk(args):
    Lp, Up = args
    s = _G["solver"]
    lp = _G["lp"]
    out_z = _G["out_z"]
    B = Lp.shape[0]
    status = np.zeros(B, dtype=np.int8)
    obj = np.zeros((B, lp.K))
    Z = np.zeros((B, out_z.size))
    iters = 0
    last = None
    for b in range(B):
        lb, ub = lp.full_bounds(Lp[b], Up[b])
        # warm: restart from the previous member's final basis (HiGHS' own warm start)
        res = s.solve(lb, ub, warm_status=last if _G["warm"] else None)
        iters += res.iterations
        status[b] = res.status
        if res.status == OPTIMAL:
            obj[b] = res.stage_obj
            Z[b] = res.z[out_z]
            last = res.zstatus
    return status, obj, Z, iters


class HighsBaseline:
    """Strong CPU baseline (persistent + warm-started + multi-process HiGHS)."""

    def __init__(self, lp: LexLP, out_z: Optional[np.ndarray] = None, n_procs: int = 1,
                 warm: bool = True) -> None:
        self.lp = lp
        self.out_z = np.arange(lp.n) if out_z is None else np.asarray(out_z)
        self.n_procs = n_procs
        self.warm = warm
        self._pool = None
        if n_procs > 1:
            import multiprocessing as mp

            # spawn: forking after CUDA / thread-pool initialisation is unsafe
            ctx = mp.get_context("spawn")
            self._pool = ctx.Pool(n_procs, initializer=_init_worker,
                                  initargs=(lp, self.out_z, warm))
        else:
            _init_worker(lp, self.out_z, warm)

    def solve(self, Lp: np.ndarray, Up: np.ndarray):
        t0 = time.perf_counter()
        B = Lp.shape[0]
        if self._pool is None:
            st, obj, Z, iters = _solve_chunk((Lp, Up))
        else:
            # contiguous chunks keep temporal/ensemble neighbours on the same worker
            k = min(B, self.n_procs)
            bounds = np.linspace(0, B, k + 1).astype(int)
            parts = self._pool.map(_solve_chunk, [(Lp[a:b], Up[a:b]) for a, b in zip(bounds[:-1], bounds[1:])])
            st = np.concatenate([p[0] for p in parts])
            obj = np.concatenate([p[1] for p in parts])
            Z = np.concatenate([p[2] for p in parts])
            iters = sum(p[3] for p in parts)
        return st, obj, Z, {"seconds": time.perf_counter() - t0, "iterations": iters}

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None
