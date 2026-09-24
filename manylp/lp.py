"""Problem specification: a lexicographic, bounds-parametric linear program.

A *group* is one constraint matrix ``A`` and one ordered list of objectives
``c_1, ..., c_K`` shared by every member of a batch.  Members differ only in their
variable and row bounds::

    lex-max  (c_1^T x, c_2^T x, ..., c_K^T x)
    s.t.     row_lb <= A x <= row_ub
             col_lb <= x   <= col_ub

Stage ``k`` maximises ``c_k`` over the optimal face of stage ``k-1``; with ``K = 1``
this is an ordinary LP.  Minimisation objectives are passed negated.

Internally every row gets a *logical* variable ``s = A x`` so the problem becomes

    Abar z = 0,   z = [x; s],   Abar = [A, -I],   lb_z <= z <= ub_z

``Abar`` always has full row rank ``m``, a basis is any ``m`` columns of it, and
**all** data that varies across a batch lives in the bounds of ``z``.  That is
the structure the solver exploits: reduced costs, and therefore dual
feasibility, do not depend on bounds, while the primal solution of a fixed basis
is an affine function of them.

Only a subset of ``z`` -- the *parametric* variables, ``param_cols`` plus the
logicals of ``param_rows`` -- may differ from the template bounds across a batch.
In dynamic FBA this subset is the exchange reactions; everything else is fixed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp


def _as_1d(a, n: int, name: str, fill: float | None = None) -> np.ndarray:
    if a is None:
        if fill is None:
            raise ValueError(f"{name} is required")
        return np.full(n, fill, dtype=float)
    out = np.asarray(a, dtype=float).reshape(-1)
    if out.shape[0] != n:
        raise ValueError(f"{name} has length {out.shape[0]}, expected {n}")
    return out.copy()


@dataclass
class LexLP:
    """A bounds-parametric lexicographic LP group (see module docstring).

    Parameters
    ----------
    A:
        ``m x n`` constraint matrix (any scipy.sparse format or dense array).
    objectives:
        ``K x n`` array (or a single ``n`` vector); row ``k`` is maximised at stage ``k``.
    col_lb, col_ub, row_lb, row_ub:
        Template bounds.  ``row_*`` default to ``0`` (equality ``A x = 0``, the
        FBA mass balance).  Infinite bounds are allowed.
    param_cols, param_rows:
        Indices whose bounds may vary across batch members.  Bounds of every other
        variable are always the template's.
    """

    A: sp.spmatrix
    objectives: np.ndarray
    col_lb: np.ndarray
    col_ub: np.ndarray
    row_lb: np.ndarray | None = None
    row_ub: np.ndarray | None = None
    param_cols: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    param_rows: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    def __post_init__(self) -> None:
        A = sp.csc_matrix(self.A, dtype=float)
        A.eliminate_zeros()
        A.sort_indices()
        self.A = A
        m, n = A.shape
        C = np.atleast_2d(np.asarray(self.objectives, dtype=float))
        if C.shape[1] != n:
            raise ValueError(f"objectives have {C.shape[1]} columns, A has {n}")
        self.objectives = C
        self.col_lb = _as_1d(self.col_lb, n, "col_lb")
        self.col_ub = _as_1d(self.col_ub, n, "col_ub")
        self.row_lb = _as_1d(self.row_lb, m, "row_lb", 0.0)
        self.row_ub = _as_1d(self.row_ub, m, "row_ub", 0.0)
        if np.any(self.col_lb > self.col_ub) or np.any(self.row_lb > self.row_ub):
            raise ValueError("template has a lower bound above its upper bound")
        pc = np.unique(np.asarray(self.param_cols, dtype=np.int64).reshape(-1))
        pr = np.unique(np.asarray(self.param_rows, dtype=np.int64).reshape(-1))
        if pc.size and (pc[0] < 0 or pc[-1] >= n):
            raise ValueError("param_cols out of range")
        if pr.size and (pr[0] < 0 or pr[-1] >= m):
            raise ValueError("param_rows out of range")
        self.param_cols, self.param_rows = pc, pr

        # ---- extended (z-space) view -------------------------------------
        self.Abar = sp.hstack([A, -sp.identity(m, format="csc")], format="csc")
        self.Abar.sort_indices()
        self.AbarT = self.Abar.T.tocsr()
        self.lb_z = np.concatenate([self.col_lb, self.row_lb])
        self.ub_z = np.concatenate([self.col_ub, self.row_ub])
        self.C_z = np.hstack([C, np.zeros((C.shape[0], m))])
        self.param_z = np.concatenate([pc, n + pr]).astype(np.int64)
        self.is_param = np.zeros(n + m, dtype=bool)
        self.is_param[self.param_z] = True
        # position of each parametric z-variable inside a member's bound vector
        self.param_pos = np.full(n + m, -1, dtype=np.int64)
        self.param_pos[self.param_z] = np.arange(self.param_z.size)

    # -- sizes ----------------------------------------------------------------
    @property
    def m(self) -> int:
        return self.A.shape[0]

    @property
    def n(self) -> int:
        return self.A.shape[1]

    @property
    def N(self) -> int:
        return self.n + self.m

    @property
    def K(self) -> int:
        return self.objectives.shape[0]

    @property
    def p(self) -> int:
        return int(self.param_z.size)

    # -- bound helpers ----------------------------------------------------------
    def template_param_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Template bounds of the parametric variables, shape ``(p,)`` each."""
        return self.lb_z[self.param_z].copy(), self.ub_z[self.param_z].copy()

    def full_bounds(self, lp_row: np.ndarray, up_row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Expand one member's parametric bounds to full z-space bounds."""
        lb = self.lb_z.copy()
        ub = self.ub_z.copy()
        lb[self.param_z] = lp_row
        ub[self.param_z] = up_row
        return lb, ub

    def objective_scales(self) -> np.ndarray:
        """Per-stage scale used to make reduced-cost tolerances relative."""
        return np.maximum(1.0, np.abs(self.objectives).max(axis=1))
