"""Flux balance analysis front-end: FBA / pFBA / lexicographic FBA as a :class:`LexLP`.

Flux uniqueness (why this module exists)
----------------------------------------
FBA fixes the growth rate but, for most genome-scale models, not the flux
vector: the optimum is a whole face.  In dynamic FBA the *exchange* fluxes drive
the ODE, so an arbitrary vertex makes trajectories depend on solver, pivoting
order and platform.  Every mode here is a lexicographic LP that the batched
solver certifies exactly, and the result carries a per-member ``unique`` flag:

``"fba"``          max c^T v                                   (vertex, may be non-unique)
``"pfba"``         max c^T v, then min ||v||_1                 (parsimonious FBA, Lewis 2010)
``"pfba-unique"``  max c^T v, then min ||v||_1, then max w^T v (default)

``w`` is a fixed pseudo-random ("generic") vector: on the pFBA optimal face it
selects a single vertex with probability one, deterministically for a given
``seed``.  ``"lex"`` takes a DFBAlab-style priority list of extra objectives.

For the L1 stage every reaction whose sign is not fixed -- reversible
reactions and all parametric (exchange) reactions -- is split ``v = v+ - v-``.
Bounds of the split parts are ``v+ in [max(l,0), max(u,0)]`` and
``v- in [max(-u,0), max(-l,0)]``, so a member's exchange bounds map to
parametric split bounds elementwise and all objectives stay bound-independent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import scipy.sparse as sp

from manylp.lp import LexLP

MODES = ("fba", "pfba", "pfba-unique", "lex")


@dataclass
class FBAModel:
    """A metabolic model: ``S v = 0``, ``lb <= v <= ub``, maximise ``c^T v``."""

    S: sp.spmatrix
    lb: np.ndarray
    ub: np.ndarray
    c: np.ndarray
    exchanges: np.ndarray                     # reaction indices whose bounds vary
    rxn_ids: Optional[Sequence[str]] = None
    met_ids: Optional[Sequence[str]] = None
    name: str = ""

    def __post_init__(self) -> None:
        self.S = sp.csc_matrix(self.S, dtype=float)
        n = self.S.shape[1]
        self.lb = np.asarray(self.lb, dtype=float).reshape(n).copy()
        self.ub = np.asarray(self.ub, dtype=float).reshape(n).copy()
        self.c = np.asarray(self.c, dtype=float).reshape(n).copy()
        self.exchanges = np.asarray(self.exchanges, dtype=np.int64).reshape(-1)

    @property
    def n(self) -> int:
        return self.S.shape[1]

    @property
    def m(self) -> int:
        return self.S.shape[0]

    @classmethod
    def from_cobra(cls, model, exchanges: Optional[Sequence[str]] = None) -> "FBAModel":
        """Extract arrays from a ``cobra.Model``; exchanges default to ``model.exchanges``."""
        from cobra.util.array import create_stoichiometric_matrix
        from cobra.util.solver import linear_reaction_coefficients

        S = create_stoichiometric_matrix(model, array_type="dok")
        rxns = model.reactions
        lb = np.array([r.lower_bound for r in rxns], dtype=float)
        ub = np.array([r.upper_bound for r in rxns], dtype=float)
        c = np.zeros(len(rxns))
        for r, w in linear_reaction_coefficients(model).items():
            c[rxns.index(r)] = float(w)
        if model.objective_direction == "min":
            c = -c
        ids = [r.id for r in rxns]
        index = {rid: i for i, rid in enumerate(ids)}
        if exchanges is None:
            ex = [index[r.id] for r in model.exchanges]
        else:
            ex = [index[r] for r in exchanges]
        return cls(S=S, lb=lb, ub=ub, c=c, exchanges=np.array(sorted(set(ex)), dtype=np.int64),
                   rxn_ids=ids, met_ids=[mt.id for mt in model.metabolites], name=model.id)


@dataclass
class FBAProblem:
    """A compiled FBA model: the :class:`LexLP` plus the maps in and out of it."""

    model: FBAModel
    mode: str
    lp: LexLP
    #: z-index of the positive (or only) part of every reaction
    pos_col: np.ndarray
    #: z-index of the negative part (``-1`` when the reaction is not split)
    neg_col: np.ndarray
    #: parametric positions of each exchange's positive / negative part (``-1`` if none)
    ex_ppos: np.ndarray
    ex_npos: np.ndarray
    split: bool
    objective_names: list = field(default_factory=list)

    # -- bounds in -------------------------------------------------------------------
    def param_bounds(self, ex_lb, ex_ub=None) -> tuple[np.ndarray, np.ndarray]:
        """Map exchange bounds ``(B, k)`` to the LexLP's parametric bounds ``(B, p)``.

        ``ex_ub=None`` keeps the template upper bounds (the dFBA case: only uptake
        -- i.e. lower -- bounds move).
        """
        ex = self.model.exchanges
        xp = _array_module(ex_lb)            # NumPy, or CuPy for device-resident callers
        ex_lb = xp.atleast_2d(xp.asarray(ex_lb, dtype=xp.float64))
        B = ex_lb.shape[0]
        if ex_ub is None:
            ex_ub = xp.broadcast_to(xp.asarray(self.model.ub[ex]), (B, ex.size))
        ex_ub = xp.atleast_2d(xp.asarray(ex_ub, dtype=xp.float64))
        L0, U0 = self.lp.template_param_bounds()
        k = ex.size
        if self._blocked():
            # parametric layout is exactly [v+ of exchanges | v- of exchanges]:
            # contiguous slices instead of strided fancy indexing
            Lp = xp.empty((B, 2 * k))
            Up = xp.empty((B, 2 * k))
            xp.maximum(ex_lb, 0.0, out=Lp[:, :k])
            xp.maximum(ex_ub, 0.0, out=Up[:, :k])
            xp.maximum(-ex_ub, 0.0, out=Lp[:, k:])
            xp.maximum(-ex_lb, 0.0, out=Up[:, k:])
            return Lp, Up
        Lp = xp.broadcast_to(xp.asarray(L0), (B, L0.size)).copy()
        Up = xp.broadcast_to(xp.asarray(U0), (B, U0.size)).copy()
        if self.split:
            Lp[:, self.ex_ppos] = xp.maximum(ex_lb, 0.0)
            Up[:, self.ex_ppos] = xp.maximum(ex_ub, 0.0)
            Lp[:, self.ex_npos] = xp.maximum(-ex_ub, 0.0)
            Up[:, self.ex_npos] = xp.maximum(-ex_lb, 0.0)
        else:
            Lp[:, self.ex_ppos] = ex_lb
            Up[:, self.ex_ppos] = ex_ub
        return Lp, Up

    def _blocked(self) -> bool:
        b = getattr(self, "_blocked_cache", None)
        if b is None:
            k = self.model.exchanges.size
            b = bool(self.split and self.lp.p == 2 * k
                     and np.array_equal(self.ex_ppos, np.arange(k))
                     and np.array_equal(self.ex_npos, np.arange(k, 2 * k)))
            self._blocked_cache = b
        return b

    # -- fluxes out --------------------------------------------------------------------
    def out_z(self, reactions: Optional[np.ndarray] = None) -> np.ndarray:
        """z-indices to request from the solver for the given reactions."""
        r = np.arange(self.model.n) if reactions is None else np.asarray(reactions)
        cols = [self.pos_col[r]]
        neg = self.neg_col[r]
        if self.split:
            cols.append(neg[neg >= 0])
        return np.concatenate(cols)

    def fluxes(self, z_out: np.ndarray, reactions: Optional[np.ndarray] = None) -> np.ndarray:
        """Recover ``v`` (B, len(reactions)) from the solver's ``z_out`` for ``out_z(reactions)``."""
        r = np.arange(self.model.n) if reactions is None else np.asarray(reactions)
        k = r.size
        if self.split:
            neg = self.neg_col[r]
            has = np.nonzero(neg >= 0)[0]
            if has.size == k:
                return z_out[:, :k] - z_out[:, k:2 * k]
            v = z_out[:, :k].copy()
            v[:, has] -= z_out[:, k:k + has.size]
            return v
        return z_out[:, :k].copy()


def compile_fba(model: FBAModel, mode: str = "pfba-unique", seed: int = 0,
                extra_objectives: Optional[Sequence[tuple[np.ndarray, str]]] = None,
                unique_tiebreak: Optional[bool] = None) -> FBAProblem:
    """Build the lexicographic LP for ``mode`` (see module docstring).

    ``extra_objectives``: list of ``(vector over reactions, "max"|"min")`` optimised in
    order right after the primary objective (any mode) -- a policy for choosing among
    alternative optima; pFBA / tie-break stages of the mode follow.  ``unique_tiebreak`` appends
    the generic stage (default: on for ``"pfba-unique"`` and ``"lex"``).
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    S, lb, ub, c = model.S, model.lb, model.ub, model.c
    m, n = S.shape
    ex = model.exchanges
    split = mode in ("pfba", "pfba-unique")
    if unique_tiebreak is None:
        unique_tiebreak = mode in ("pfba-unique", "lex")

    if split:
        is_ex = np.zeros(n, dtype=bool)
        is_ex[ex] = True
        R = np.nonzero(((lb < 0) & (ub > 0)) | is_ex)[0]
        nR = R.size
        A = sp.hstack([S, -S[:, R]], format="csc")
        col_lb = np.concatenate([lb, np.zeros(nR)])
        col_ub = np.concatenate([ub, np.zeros(nR)])
        col_lb[R] = np.maximum(lb[R], 0.0)
        col_ub[R] = np.maximum(ub[R], 0.0)
        col_lb[n:] = np.maximum(-ub[R], 0.0)
        col_ub[n:] = np.maximum(-lb[R], 0.0)
        neg_col = np.full(n, -1, dtype=np.int64)
        neg_col[R] = n + np.arange(nR)
        n2 = n + nR
        c1 = np.concatenate([c, -c[R]])
        # L1 norm: +v for forward-only, -v for backward-only, v+ + v- for split
        l1 = np.where(ub <= 0, -1.0, 1.0)
        l1[(lb == 0) & (ub == 0)] = 0.0
        l1 = np.concatenate([l1, np.ones(nR)])
        l1[R] = 1.0
        objs = [c1, -l1]
        names = ["primary", "min_l1"]
        param_cols = np.concatenate([ex, neg_col[ex]])
    else:
        A = S
        col_lb, col_ub = lb.copy(), ub.copy()
        neg_col = np.full(n, -1, dtype=np.int64)
        n2 = n
        objs = [c.copy()]
        names = ["primary"]
        param_cols = ex.copy()

    def lift(vec):
        vec = np.asarray(vec, dtype=float).reshape(n)
        return np.concatenate([vec, -vec[R]]) if split else vec

    # user objectives go right after the primary one (a "policy" for choosing among
    # alternative optima); pFBA and the tie-break, if any, then make it unique
    extras = []
    for vec, sense in extra_objectives or []:
        v = lift(vec)
        extras.append(v if sense == "max" else -v)
    if extras:
        objs[1:1] = extras
        names[1:1] = [f"lex_{sense}" for _, sense in extra_objectives]
    if unique_tiebreak:
        rng = np.random.default_rng(seed)
        objs.append(rng.uniform(-1.0, 1.0, size=n2))
        names.append("tiebreak")

    lp = LexLP(A=A, objectives=np.vstack(objs), col_lb=col_lb, col_ub=col_ub,
               param_cols=param_cols)
    ex_ppos = lp.param_pos[ex]
    ex_npos = lp.param_pos[neg_col[ex]] if split else np.full(ex.size, -1, dtype=np.int64)
    return FBAProblem(model=model, mode=mode, lp=lp, pos_col=np.arange(n, dtype=np.int64),
                      neg_col=neg_col, ex_ppos=ex_ppos, ex_npos=ex_npos, split=split,
                      objective_names=names)


def _array_module(a):
    if isinstance(a, np.ndarray) or not hasattr(a, "__cuda_array_interface__"):
        return np
    import cupy

    return cupy
