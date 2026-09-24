"""Repair: lexicographic simplex solves of individual members with HiGHS.

Only members that no cached certificate covers reach this path.  Each worker
keeps a persistent ``highspy.Highs`` instance per group -- the matrix is loaded
once and only bounds/costs change between solves -- and warm-starts the dual
simplex from the nearest cached basis, which stays dual feasible under any
bound change.

Lexicographic stages use *reduced-cost fixing*: after stage ``k`` every nonbasic
variable (column or row logical) with a nonzero reduced cost is fixed at its
current value, which by LP duality carves out exactly the optimal face; stage
``k+1`` then optimises its objective primal-warm from the stage-``k`` basis.  No
objective-bound rows are added, so the LP never grows or becomes more
degenerate.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from manylp.basis import AT_LOWER, AT_UPPER, AT_ZERO, BASIC
from manylp.lp import LexLP

OPTIMAL = 1
INFEASIBLE = 2
UNBOUNDED = 3
ERROR = 4


@dataclass
class RepairResult:
    status: int
    basic: Optional[np.ndarray] = None       # (m,) z-indices
    zstatus: Optional[np.ndarray] = None     # (N,) int8
    z: Optional[np.ndarray] = None           # (N,) primal values of the final stage
    stage_obj: Optional[np.ndarray] = None   # (K,)
    farkas_y: Optional[np.ndarray] = None    # (m,) dual ray if infeasible
    iterations: int = 0
    seconds: float = 0.0
    message: str = ""


class HighsLexSolver:
    """A persistent HiGHS instance that solves members of one :class:`LexLP`."""

    def __init__(self, lp: LexLP, dual_tol: float = 1e-9, time_limit: float = 300.0,
                 presolve: bool = False) -> None:
        import highspy

        self._hp = highspy
        self.lp = lp
        self.dual_tol = dual_tol
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("threads", 1)
        h.setOptionValue("presolve", "on" if presolve else "off")
        h.setOptionValue("time_limit", float(time_limit))
        # tighter than default so that HiGHS' optimum agrees with our certificate
        h.setOptionValue("primal_feasibility_tolerance", 1e-9)
        h.setOptionValue("dual_feasibility_tolerance", 1e-9)
        model = highspy.HighsLp()
        A = lp.A
        model.num_col_ = lp.n
        model.num_row_ = lp.m
        model.col_cost_ = lp.objectives[0].copy()
        model.col_lower_ = lp.col_lb.copy()
        model.col_upper_ = lp.col_ub.copy()
        model.row_lower_ = lp.row_lb.copy()
        model.row_upper_ = lp.row_ub.copy()
        model.sense_ = highspy.ObjSense.kMaximize
        model.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        model.a_matrix_.num_col_ = lp.n
        model.a_matrix_.num_row_ = lp.m
        model.a_matrix_.start_ = A.indptr.astype(np.int32)
        model.a_matrix_.index_ = A.indices.astype(np.int32)
        model.a_matrix_.value_ = A.data.copy()
        h.passModel(model)
        self.h = h
        self._stage = 0  # objective currently loaded
        # indices whose bounds currently differ from the template
        self._dirty_cols = np.zeros(0, dtype=np.int32)
        self._dirty_rows = np.zeros(0, dtype=np.int32)
        S = highspy.HighsBasisStatus
        # z-status code -> HiGHS enum, gathered with one fancy index (BASIC=-1 -> slot 3)
        self._enum = np.array([S.kLower, S.kUpper, S.kZero, S.kBasic], dtype=object)
        self._all_cols = np.arange(lp.n, dtype=np.int32)

    # -- bounds ------------------------------------------------------------------
    def _load_bounds(self, lb_z: np.ndarray, ub_z: np.ndarray) -> None:
        lp, h = self.lp, self.h
        n = lp.n
        cols = np.union1d(lp.param_cols, self._dirty_cols).astype(np.int32)
        if cols.size:
            h.changeColsBounds(cols.size, cols, lb_z[cols], ub_z[cols])
        rows = np.union1d(lp.param_rows, self._dirty_rows).astype(np.int32)
        if rows.size:
            h.changeRowsBounds(rows.size, rows, lb_z[n + rows], ub_z[n + rows])
        self._dirty_cols = np.zeros(0, dtype=np.int32)
        self._dirty_rows = np.zeros(0, dtype=np.int32)

    def _set_stage(self, k: int) -> None:
        if self._stage != k:
            self.h.changeColsCost(self.lp.n, self._all_cols, self.lp.objectives[k])
            self._stage = k

    def _set_basis(self, zstatus: np.ndarray) -> bool:
        hp = self._hp
        b = hp.HighsBasis()
        n = self.lp.n
        codes = np.where(zstatus == BASIC, 3, zstatus).astype(np.int64)
        enums = self._enum[codes]
        b.col_status = enums[:n].tolist()
        b.row_status = enums[n:].tolist()
        b.valid = True
        st = self.h.setBasis(b)
        return st == hp.HighsStatus.kOk

    def _basic_z(self) -> np.ndarray:
        """Basic variables as z-indices (HiGHS encodes row ``i`` as ``-1-i``)."""
        bv = np.asarray(self.h.getBasicVariables()[1], dtype=np.int64)
        return np.where(bv >= 0, bv, self.lp.n - 1 - bv)

    def _model_status(self):
        return self.h.getModelStatus()

    # -- main entry ----------------------------------------------------------------
    def solve(self, lb_z: np.ndarray, ub_z: np.ndarray,
              warm_status: Optional[np.ndarray] = None) -> RepairResult:
        """Lexicographic solve with a retry ladder for numerical failures.

        A warm-started dual simplex can stall on ill-conditioned LPs (e.g. Netlib
        bnl1).  On a solver error we retry cold, then once more with presolve on,
        before reporting ``ERROR`` -- an error is "no answer", never a wrong one.
        """
        r = self._solve(lb_z, ub_z, warm_status)
        if r.status == ERROR and warm_status is not None:
            r2 = self._solve(lb_z, ub_z, None)
            r2.iterations += r.iterations
            r = r2
        if r.status == ERROR:
            self.h.setOptionValue("presolve", "on")
            try:
                r3 = self._solve(lb_z, ub_z, None)
            finally:
                self.h.setOptionValue("presolve", "off")
            r3.iterations += r.iterations
            r3.message = (r3.message + " (after presolve retry)").strip()
            r = r3
        return r

    def _solve(self, lb_z: np.ndarray, ub_z: np.ndarray,
               warm_status: Optional[np.ndarray] = None) -> RepairResult:
        hp, h, lp = self._hp, self.h, self.lp
        MS = hp.HighsModelStatus
        t0 = time.perf_counter()
        n, m, K = lp.n, lp.m, lp.K
        self._load_bounds(lb_z, ub_z)
        self._set_stage(0)
        if warm_status is not None:
            if not self._set_basis(warm_status):
                h.clearSolver()
        else:
            h.clearSolver()
        iters = 0
        h.run()
        iters += int(h.getInfo().simplex_iteration_count)
        ms = self._model_status()
        if ms == MS.kInfeasible or ms == MS.kUnboundedOrInfeasible:
            if ms == MS.kUnboundedOrInfeasible:
                # disambiguate: a zero objective cannot be unbounded
                h.clearSolver()
                h.changeColsCost(n, self._all_cols, np.zeros(n))
                self._stage = -1
                h.run()
                iters += int(h.getInfo().simplex_iteration_count)
                if self._model_status() != MS.kInfeasible:
                    return RepairResult(UNBOUNDED, iterations=iters,
                                        seconds=time.perf_counter() - t0)
            y = self._dual_ray()
            return RepairResult(INFEASIBLE, farkas_y=y, iterations=iters,
                                seconds=time.perf_counter() - t0)
        if ms == MS.kUnbounded:
            return RepairResult(UNBOUNDED, iterations=iters, seconds=time.perf_counter() - t0)
        if ms != MS.kOptimal:
            return RepairResult(ERROR, iterations=iters, seconds=time.perf_counter() - t0,
                                message=h.modelStatusToString(ms))

        scales = lp.objective_scales()
        fixed_cols: list[np.ndarray] = []
        fixed_rows: list[np.ndarray] = []
        stage_obj = np.zeros(K)
        for k in range(K):
            if k > 0:
                self._set_stage(k)
                h.run()
                iters += int(h.getInfo().simplex_iteration_count)
                if self._model_status() != MS.kOptimal:
                    self._dirty_cols = np.concatenate(fixed_cols).astype(np.int32) if fixed_cols else self._dirty_cols
                    self._dirty_rows = np.concatenate(fixed_rows).astype(np.int32) if fixed_rows else self._dirty_rows
                    return RepairResult(ERROR, iterations=iters,
                                        seconds=time.perf_counter() - t0,
                                        message=f"stage {k} not optimal")
            sol = h.getSolution()
            x = np.asarray(sol.col_value)
            stage_obj[k] = float(lp.objectives[k] @ x)
            if k == K - 1:
                break
            isb = np.zeros(n + m, dtype=bool)
            isb[self._basic_z()] = True
            cd = np.abs(np.asarray(sol.col_dual))
            rd = np.abs(np.asarray(sol.row_dual))
            thr = self.dual_tol * scales[k]
            fc = np.nonzero(~isb[:n] & (cd > thr) & (lb_z[:n] < ub_z[:n]))[0]
            fr = np.nonzero(~isb[n:] & (rd > thr) & (lb_z[n:] < ub_z[n:]))[0]
            if fc.size:
                v = x[fc]
                h.changeColsBounds(fc.size, fc.astype(np.int32), v, v)
                fixed_cols.append(fc)
            if fr.size:
                r = np.asarray(sol.row_value)[fr]
                h.changeRowsBounds(fr.size, fr.astype(np.int32), r, r)
                fixed_rows.append(fr)

        sol = h.getSolution()
        x = np.asarray(sol.col_value, dtype=float)
        r = np.asarray(sol.row_value, dtype=float)
        z = np.concatenate([x, r])
        basic = np.sort(self._basic_z())
        zstatus = self._to_zstatus(basic, z, lb_z, ub_z)
        self._dirty_cols = np.concatenate(fixed_cols).astype(np.int32) if fixed_cols else np.zeros(0, np.int32)
        self._dirty_rows = np.concatenate(fixed_rows).astype(np.int32) if fixed_rows else np.zeros(0, np.int32)
        return RepairResult(OPTIMAL, basic=basic, zstatus=zstatus, z=z, stage_obj=stage_obj,
                            iterations=iters, seconds=time.perf_counter() - t0)

    @staticmethod
    def _to_zstatus(basic: np.ndarray, z: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
        """Nonbasic statuses w.r.t. the member's *original* bounds, from values.

        Variables fixed by the lexicographic stages report an arbitrary side in
        HiGHS; their true side is whichever original bound their value sits at.
        """
        out = np.full(z.shape, AT_LOWER, dtype=np.int8)
        out[basic] = BASIC
        nb = out != BASIC
        scale = 1e-9 * np.maximum(1.0, np.abs(z))
        at_lo = nb & np.isfinite(lb) & (np.abs(z - lb) <= scale)
        at_up = nb & np.isfinite(ub) & (np.abs(z - ub) <= scale)
        out[at_up & ~at_lo] = AT_UPPER
        free0 = nb & ~at_lo & ~at_up & (np.abs(z) <= scale)
        out[free0] = AT_ZERO
        # anything else (nonbasic strictly between bounds) keeps AT_LOWER and will
        # fail certification, falling back to the HiGHS values for this member
        return out

    def _dual_ray(self) -> Optional[np.ndarray]:
        try:
            res = self.h.getDualRay()
        except Exception:
            return None
        # highspy returns (status, has_ray, values) or (has_ray, values)
        if isinstance(res, tuple):
            vals = res[-1]
            has = res[-2]
        else:  # pragma: no cover
            return None
        if not has:
            return None
        return np.asarray(vals, dtype=float)
