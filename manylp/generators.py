"""Synthetic batched-LP workloads (tests and the non-biological benchmarks)."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from manylp.fba import FBAModel
from manylp.lp import LexLP


def random_feasible_lp(m: int, n: int, density: float = 0.1, n_param: int = 10,
                       n_objectives: int = 1, seed: int = 0, box: float = 10.0) -> tuple[LexLP, np.ndarray]:
    """A random inequality LP that is feasible at a known interior-ish point ``x0``.

    ``rl <= A x <= ru`` with ``A x0`` strictly inside, ``-box <= x <= box``;
    the first ``n_param`` columns are parametric.  Returns ``(lp, x0)``.
    """
    rng = np.random.default_rng(seed)
    A = sp.random(m, n, density=density, random_state=rng, format="csc",
                  data_rvs=lambda k: rng.normal(size=k))
    # every column must appear somewhere or the LP is trivially unbounded along it
    empty = np.nonzero(np.diff(A.indptr) == 0)[0]
    if empty.size:
        rows = rng.integers(0, m, size=empty.size)
        fix = sp.csc_matrix((rng.normal(size=empty.size), (rows, empty)), shape=(m, n))
        A = (A + fix).tocsc()
    x0 = rng.uniform(-box / 2, box / 2, size=n)
    r0 = A @ x0
    slack = rng.uniform(0.5, 2.0, size=m)
    rl = r0 - slack
    ru = r0 + slack
    eq = rng.random(m) < 0.3
    rl[eq] = ru[eq] = r0[eq]
    C = rng.normal(size=(n_objectives, n))
    lp = LexLP(A=A, objectives=C, col_lb=np.full(n, -box), col_ub=np.full(n, box),
               row_lb=rl, row_ub=ru, param_cols=np.arange(n_param))
    return lp, x0


def perturb_param_bounds(lp: LexLP, B: int, scale: float = 0.2, seed: int = 0,
                         shrink_only: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """``B`` random perturbations of the template's parametric bounds."""
    rng = np.random.default_rng(seed)
    L0, U0 = lp.template_param_bounds()
    width = np.where(np.isfinite(U0 - L0), U0 - L0, 1.0)
    dl = rng.uniform(0 if shrink_only else -scale, scale, size=(B, L0.size)) * width
    du = rng.uniform(0 if shrink_only else -scale, scale, size=(B, U0.size)) * width
    L = L0 + dl
    U = U0 - du
    bad = L > U
    mid = (L + U) / 2
    L = np.where(bad, mid, L)
    U = np.where(bad, mid, U)
    return L, U


def random_metabolic_network(n_met: int = 60, n_rxn: int = 150, n_ex: int = 15,
                             seed: int = 0, reversible_frac: float = 0.4) -> FBAModel:
    """A random, mass-balanced-looking metabolic network with exchanges and a biomass.

    Internal reactions convert 1-2 substrates to 1-2 products; exchange reactions
    ``met <->`` let the first ``n_ex`` metabolites enter/leave; a biomass reaction
    drains a random subset of metabolites.  Every metabolite gets a producer and a
    consumer so the network is not trivially blocked.  Uptake bounds are ``-10``.
    """
    rng = np.random.default_rng(seed)
    rows, cols, vals = [], [], []
    j = 0
    lb, ub = [], []
    # exchanges
    for i in range(n_ex):
        rows.append(i); cols.append(j); vals.append(-1.0)
        lb.append(-10.0); ub.append(1000.0); j += 1
    ex = np.arange(n_ex)
    # chain producing every metabolite from an earlier one (guarantees connectivity)
    for i in range(n_ex, n_met):
        src = int(rng.integers(0, i))
        rows += [src, i]; cols += [j, j]; vals += [-1.0, 1.0]
        rev = rng.random() < reversible_frac
        lb.append(-1000.0 if rev else 0.0); ub.append(1000.0); j += 1
    # random internal reactions
    while j < n_rxn - 1:
        k_s = int(rng.integers(1, 3)); k_p = int(rng.integers(1, 3))
        mets = rng.choice(n_met, size=k_s + k_p, replace=False)
        for a in mets[:k_s]:
            rows.append(int(a)); cols.append(j); vals.append(-float(rng.integers(1, 3)))
        for a in mets[k_s:]:
            rows.append(int(a)); cols.append(j); vals.append(float(rng.integers(1, 3)))
        rev = rng.random() < reversible_frac
        lb.append(-1000.0 if rev else 0.0); ub.append(1000.0); j += 1
    # biomass
    bm = rng.choice(np.arange(n_ex, n_met), size=min(8, n_met - n_ex), replace=False)
    for a in bm:
        rows.append(int(a)); cols.append(j); vals.append(-float(rng.uniform(0.1, 1.0)))
    lb.append(0.0); ub.append(1000.0)
    biomass = j
    j += 1
    S = sp.csc_matrix((vals, (rows, cols)), shape=(n_met, j))
    c = np.zeros(j)
    c[biomass] = 1.0
    return FBAModel(S=S, lb=np.array(lb), ub=np.array(ub), c=c, exchanges=ex,
                    rxn_ids=[f"R{i}" for i in range(j)], met_ids=[f"M{i}" for i in range(n_met)],
                    name=f"random{seed}")
