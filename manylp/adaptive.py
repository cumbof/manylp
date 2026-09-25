"""Regime-aware adaptive integration of community dFBA (opt-in, experimental).

With a certified solver an FBA LP is a cheap, exact function evaluation: within a
metabolic regime (one optimal basis) growth and exchange fluxes are *affine* in
the uptake bounds, so the dFBA right-hand side is smooth there and only kinks at
regime switches (a nutrient running out, a diauxic shift).  That makes
high-order adaptive integrators affordable -- something fixed-step explicit
Euler with a CFL-style uptake cap (the usual dFBA scheme) cannot exploit.

This module integrates the continuous model

    dX_i/dt = mu_i(u) X_i,        u_ij = min(Vmax M_j/(Km + M_j), diet_j)
    dM_j/dt = influx_j - D M_j + sum_i v_ij(u) X_i

with SciPy's embedded Runge-Kutta pairs, every right-hand-side evaluation being
one batched, certified solve per species.  Optionally (``split_regimes=True``)
regime switches are located by event detection: a regime is the pattern of binding
uptake bounds and growing members, which (with unique fluxes) is a function of the
state alone and changes at the kinks, and a step that straddles a switch is split
there.  In our benchmarks this did not improve accuracy, so it is not recommended
by default.

It changes the time discretisation, not the LPs: the answer converges to the
same continuous dFBA solution as fixed-step Euler with dt -> 0.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from manylp.dfba import Community, ManyLPAdapter


@dataclass
class AdaptiveResult:
    times: np.ndarray
    X: np.ndarray          # (T, E, S)
    M: np.ndarray          # (T, E, nM)
    rhs_evals: int
    lp_batches: int
    lps: int
    regime_switches: int
    solve_seconds: float
    wall_seconds: float


class _RHS:
    def __init__(self, comm: Community, adapter: ManyLPAdapter, E: int, dilution: float,
                 min_biomass: float):
        self.c, self.ad, self.E = comm, adapter, E
        self.S, self.nM = len(comm.models), len(comm.env_mets)
        self.D = dilution
        self.min_biomass = min_biomass
        self.evals = 0
        self.batches = 0
        self.lps = 0
        self.t_solve = 0.0
    def fluxes(self, X, M):
        """Growth, environmental fluxes and the regime signature of every member.

        The signature marks, per (member, species, exchange), whether the uptake bound
        binds, plus which members grow.  Basis ids are not used: in degenerate LPs
        several cached bases certify the same point, and which one is found first
        depends on history, so they are not a function of the state.
        """
        c = self.c
        mu = np.zeros((self.E, self.S))
        flux_env = np.zeros((self.E, self.nM))
        binding = []
        Mp = np.maximum(M, 0.0)
        for s in range(self.S):
            active = np.nonzero(X[:, s] > self.min_biomass)[0]
            if active.size == 0:
                continue
            envi = c.ex_env[s]
            conc = Mp[np.ix_(active, envi)]
            up = np.where(conc > 0, c.vmax * conc / (c.km + conc), 0.0)
            up = np.minimum(up, c.max_uptake[envi][None, :])
            t = time.perf_counter()
            g, v, ok = self.ad.solve(s, -up, active)
            self.t_solve += time.perf_counter() - t
            self.batches += 1
            self.lps += active.size
            mu[active, s] = g
            flux_env[np.ix_(active, envi)] += v * X[active, s][:, None]
            b = np.zeros((self.E, envi.size), dtype=bool)
            b[active] = (up > 0) & (np.abs(v + up) <= 1e-9 * np.maximum(1.0, up))
            binding.append(b)
        sig = np.concatenate(binding + [mu > 0], axis=1) if binding else mu > 0
        return mu, flux_env, sig

    def __call__(self, t, y):
        self.evals += 1
        X = y[: self.E * self.S].reshape(self.E, self.S)
        M = y[self.E * self.S:].reshape(self.E, self.nM)
        mu, fe, _ = self.fluxes(X, M)
        dX = (mu - self.D) * np.maximum(X, 0.0)
        dM = self.c.influx[None, :] - self.D * M + fe
        return np.concatenate([dX.ravel(), dM.ravel()])


def run_adaptive(comm: Community, adapter: ManyLPAdapter, E: int = 1, t_end: float = 48.0,
                 rtol: float = 1e-6, atol: float = 1e-9, method: str = "RK45",
                 perturb=None, mode: str = "pfba-unique", dilution: float = 0.0,
                 min_biomass: float = 1e-9, t_eval=None, split_regimes: bool = False) -> AdaptiveResult:
    """Integrate ``E`` members with an adaptive RK method (see module docstring)."""
    from scipy.integrate import solve_ivp

    adapter.setup(comm.models, mode)
    S, nM = len(comm.models), len(comm.env_mets)
    X0 = np.tile(comm.X0[None, :], (E, 1)).astype(float)
    M0 = np.tile(comm.M0[None, :], (E, 1)).astype(float)
    if perturb is not None:
        M0 = M0 * perturb
    rhs = _RHS(comm, adapter, E, dilution, min_biomass)
    t_eval = np.linspace(0, t_end, 49) if t_eval is None else t_eval
    y0 = np.concatenate([X0.ravel(), M0.ravel()])
    wall = time.perf_counter()
    switches = 0
    if not split_regimes:
        sol = solve_ivp(rhs, (0, t_end), y0, method=method, rtol=rtol, atol=atol, t_eval=t_eval)
        Y = sol.y.T
    else:
        # integrate regime by regime: an event fires when any member's basis changes
        seg_t, seg_y = [], []
        t0, y = 0.0, y0
        _, _, sig0 = rhs.fluxes(y[: E * S].reshape(E, S), y[E * S:].reshape(E, nM))

        def switch(t, yy):
            _, _, sg = rhs.fluxes(yy[: E * S].reshape(E, S), yy[E * S:].reshape(E, nM))
            return 0.5 if np.array_equal(sg, switch.sig) else -0.5
        switch.terminal = True
        switch.direction = -1
        while t0 < t_end - 1e-12:
            switch.sig = sig0
            te = t_eval[(t_eval >= t0) & (t_eval <= t_end)]
            try:
                sol = solve_ivp(rhs, (t0, t_end), y, method=method, rtol=rtol, atol=atol, t_eval=te,
                                events=switch)
            except ValueError:
                # event location failed (the signature flickered inside a step, e.g. at a
                # bound that is binding to within rounding): finish without splitting
                sol = solve_ivp(rhs, (t0, t_end), y, method=method, rtol=rtol, atol=atol, t_eval=te)
            if len(sol.t):
                seg_t.append(np.asarray(sol.t))
                seg_y.append(np.asarray(sol.y).T)
            if sol.status == 1 and sol.t_events[0].size:
                t0 = float(sol.t_events[0][0]) + 1e-9
                y = sol.y_events[0][0]
                _, _, sig0 = rhs.fluxes(y[: E * S].reshape(E, S), y[E * S:].reshape(E, nM))
                switches += 1
            else:
                break
        tt = np.concatenate(seg_t)
        order = np.argsort(tt)
        Y = np.concatenate(seg_y)[order]
    Xh = Y[:, : E * S].reshape(-1, E, S)
    Mh = Y[:, E * S:].reshape(-1, E, nM)
    return AdaptiveResult(times=np.asarray(t_eval), X=Xh, M=Mh, rhs_evals=rhs.evals, lp_batches=rhs.batches,
                          lps=rhs.lps, regime_switches=switches, solve_seconds=rhs.t_solve,
                          wall_seconds=time.perf_counter() - wall)
