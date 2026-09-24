"""Alternative-optima envelopes for dynamic FBA.

FBA pins the growth rate, not the fluxes: the optimal set of every LP is a
polytope (the *optimal face*), and the exchange fluxes that drive the dFBA ODE
can be anywhere on it.  Which point a conventional solver returns depends on
pivoting rules, presolve and floating-point noise -- so a published dFBA
trajectory silently embeds one arbitrary choice per LP.

Enumerating the face is hopeless (infinitely many optima; exponentially many
vertices; branching at every step multiplies that by the number of steps).
Instead we integrate a set of **selection policies** in lockstep.  A policy is a
lexicographic objective list applied consistently at every step and to every
species::

    max growth  ->  policy objective(s)  ->  min ||v||_1  ->  generic tie-break

so each policy yields a unique, certified, reproducible trajectory, and the
spread across policies -- the *alternative-optima envelope* -- measures how much
of a dFBA prediction is determined by biology and how much by solver choice.

Policies provided here:

* ``canonical``            no policy objective (pFBA-unique);
* ``random:<seed>``        a random generic direction on the optimal face (samples
                           its vertices; many seeds approximate its extent);
* ``max:<met>`` / ``min:<met>``  extreme secretion of one metabolite (e.g. a
                           short-chain fatty acid), wherever a species can.

With the batched solver the policies are just ensemble members (one certified
group per species and policy), so an envelope costs a fraction of a single
conventional trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from manylp.dfba import FBASolverAdapter, exchange_metabolites
from manylp.fba import FBAModel, compile_fba


@dataclass(frozen=True)
class Policy:
    name: str

    def objectives(self, model: FBAModel) -> Optional[list]:
        """Extra lexicographic objectives for ``model`` (``[]`` = canonical)."""
        kind, _, arg = self.name.partition(":")
        if kind == "canonical":
            return []
        if kind == "random":
            rng = np.random.default_rng(int(arg) + 1_000_003)
            return [(rng.normal(size=model.n), "max")]
        if kind in ("max", "min"):
            mets = exchange_metabolites(model)
            if arg not in mets:
                return []                         # species cannot exchange it
            j = model.exchanges[mets.index(arg)]
            e = np.zeros(model.n)
            e[j] = 1.0                            # exchange flux > 0 is secretion
            return [(e, kind)]
        raise ValueError(f"unknown policy {self.name!r}")


def standard_policies(n_random: int = 16, metabolites: Sequence[str] = ()) -> List[Policy]:
    pols = [Policy("canonical")]
    pols += [Policy(f"random:{i}") for i in range(n_random)]
    for m in metabolites:
        pols += [Policy(f"max:{m}"), Policy(f"min:{m}")]
    return pols


class PolicyAdapter(FBASolverAdapter):
    """Batched solver where ensemble member ``e`` follows ``policies[member_policy[e]]``."""

    def __init__(self, policies: Sequence[Policy], member_policy: np.ndarray, device: str = "cuda",
                 n_workers: int = 32) -> None:
        from manylp.solver import BatchLPSolver

        self.policies = list(policies)
        self.member_policy = np.asarray(member_policy, dtype=np.int64)
        self.solver = BatchLPSolver(device=device, n_workers=n_workers)
        self.name = f"policies-{len(self.policies)}"

    def setup(self, models, mode):
        self.models = models
        self.mode = mode
        self.groups: Dict[tuple, tuple] = {}
        self.warm: Dict[tuple, np.ndarray] = {}
        self.unique_counts = np.zeros(2, dtype=np.int64)   # [unique, total]

    def _group(self, s, q):
        key = (s, q)
        g = self.groups.get(key)
        if g is None:
            mdl = self.models[s]
            prob = compile_fba(mdl, mode=self.mode, extra_objectives=self.policies[q].objectives(mdl))
            h = self.solver.register_group(prob.lp, out_z=prob.out_z(mdl.exchanges),
                                           name=f"{mdl.name}/{self.policies[q].name}")
            g = self.groups[key] = (prob, h)
        return g

    def solve(self, s, ex_lb, member_ids):
        k = self.models[s].exchanges.size
        E = member_ids.size
        growth = np.zeros(E)
        flux = np.zeros((E, k))
        feas = np.zeros(E, dtype=bool)
        pol = self.member_policy[member_ids]
        for q in np.unique(pol):
            idx = np.nonzero(pol == q)[0]
            prob, h = self._group(s, q)
            Lp, Up = prob.param_bounds(ex_lb[idx])
            w = self.warm.get((s, q))
            mids = member_ids[idx]
            if w is None or w.size <= mids.max():
                nw = np.full(int(self.member_policy.size), -1, dtype=np.int64)
                if w is not None:
                    nw[: w.size] = w
                w = self.warm[(s, q)] = nw
            sol = self.solver.solve_batch(h, Lp, Up, warm_start=w[mids])
            w[mids] = sol.basis_id
            ok = sol.status == 1
            growth[idx] = np.where(ok, sol.objective[:, 0], 0.0)
            flux[idx] = prob.fluxes(sol.z, prob.model.exchanges)
            feas[idx] = ok
            self.unique_counts += [int(sol.unique[ok].sum()), int(ok.sum())]
        return growth, flux, feas

    def stats(self):
        tot = {"groups": len(self.groups), "unique": int(self.unique_counts[0]),
               "optimal": int(self.unique_counts[1]), "highs_solves": 0, "members": 0}
        for _, h in self.groups.values():
            tot["highs_solves"] += h.totals["highs_solves"]
            tot["members"] += h.totals["members"]
        return tot

    def close(self):
        self.solver.close()
