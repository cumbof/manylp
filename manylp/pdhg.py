"""Batched restarted PDHG (cuPDLP-style) in JAX -- the first-order GPU baseline.

This is the approach usually recommended for GPU linear programming (PDLP /
cuPDLP / cuOpt), implemented *batched*: one shared sparse ``A``, ``B`` members
with their own bounds, all iterating together on the device.  It solves a
single-objective LP::

    max c^T x   s.t.   rl <= A x <= ru,   l <= x <= u

with the ingredients that make PDLP competitive:

* Ruiz (10 sweeps) + Pock-Chambolle (alpha = 1) diagonal preconditioning;
* step size ``eta = 0.99/||A||_2`` (power iteration) split by a primal weight;
* adaptive restarts to the average iterate (PDLP's sufficient/necessary/
  artificial criteria on the relative KKT error), per member;
* primal-weight updates at restarts (smoothing 0.5), per member;
* termination on relative KKT error ``max(primal res, dual res, gap) <= eps``;
* warm starts from a previous solution (the dFBA temporal-coherence lever).

The sparse products use gather + ``segment_sum`` so every iteration is two
``B x nnz`` kernels.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import partial
from typing import Optional

import numpy as np

from manylp.lp import LexLP


def _jax():
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    return jax, jnp


@dataclass
class PDHGResult:
    x: np.ndarray            # (B, n) primal (unscaled)
    y: np.ndarray            # (B, m) dual (unscaled)
    objective: np.ndarray    # (B,) c^T x
    kkt: np.ndarray          # (B,) relative KKT error at exit
    converged: np.ndarray    # (B,)
    iterations: np.ndarray   # (B,) iterations until convergence
    seconds: float
    state: tuple             # warm-start state (scaled x, y, primal weight)


def _ruiz_pc(A, iters: int = 10):
    """Ruiz equilibration followed by Pock-Chambolle(alpha=1) scaling: returns (Dr, Dc)."""
    import scipy.sparse as sp

    A = sp.csr_matrix(A, dtype=float)
    m, n = A.shape
    dr = np.ones(m)
    dc = np.ones(n)
    for _ in range(iters):
        As = sp.diags(dr) @ A @ sp.diags(dc)
        rmax = np.sqrt(np.maximum(abs(As).max(axis=1).toarray().ravel(), 1e-12))
        cmax = np.sqrt(np.maximum(abs(As).max(axis=0).toarray().ravel(), 1e-12))
        dr /= rmax
        dc /= cmax
    As = abs(sp.diags(dr) @ A @ sp.diags(dc))
    rs = np.sqrt(np.maximum(np.asarray(As.sum(axis=1)).ravel(), 1e-12))
    cs = np.sqrt(np.maximum(np.asarray(As.sum(axis=0)).ravel(), 1e-12))
    return dr / rs, dc / cs


class BatchedPDHG:
    """Batched PDHG for one LP group; objective = ``lp.objectives[0]`` (maximised)."""

    def __init__(self, lp: LexLP, eps: float = 1e-6, max_iters: int = 100_000,
                 check_every: int = 64, device: Optional[str] = None) -> None:
        jax, jnp = _jax()
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla

        self.jax, self.jnp = jax, jnp
        self.lp = lp
        self.eps = eps
        self.max_iters = max_iters
        self.check_every = check_every
        dev = None
        if device is not None:
            kind = "gpu" if device.startswith("cuda") or device == "gpu" else "cpu"
            dev = jax.devices(kind)[0]
        self.device = dev

        A = sp.csr_matrix(lp.A)
        dr, dc = _ruiz_pc(A)
        As = sp.csr_matrix(sp.diags(dr) @ A @ sp.diags(dc))
        # ||As||_2 by power iteration (on the host, once)
        v = np.random.default_rng(0).normal(size=As.shape[1])
        for _ in range(60):
            v = As.T @ (As @ v)
            v /= np.linalg.norm(v)
        norm = float(np.sqrt(np.linalg.norm(As.T @ (As @ v))))
        try:
            norm = max(norm, float(spla.svds(As, k=1, return_singular_vectors=False)[0]))
        except Exception:
            pass
        self.eta = 0.99 / norm
        coo = As.tocoo()
        put = (lambda a: jax.device_put(a, dev)) if dev is not None else jnp.asarray
        self.rows = put(coo.row.astype(np.int32))
        self.cols = put(coo.col.astype(np.int32))
        self.vals = put(coo.data)
        self.m, self.n = A.shape
        self.dr = dr
        self.dc = dc
        # minimisation of -c in scaled space
        self.c_s = put(-lp.objectives[0] * dc)
        self.c_norm = float(np.linalg.norm(lp.objectives[0]))
        self._put = put
        self._step = jax.jit(partial(self._chunk, n_inner=check_every))

    # -- sparse products -------------------------------------------------------------
    def _Ax(self, X):
        jax, jnp = self.jax, self.jnp
        prod = X[:, self.cols] * self.vals               # B x nnz
        return jax.ops.segment_sum(prod.T, self.rows, num_segments=self.m).T

    def _ATy(self, Y):
        jax, jnp = self.jax, self.jnp
        prod = Y[:, self.rows] * self.vals
        return jax.ops.segment_sum(prod.T, self.cols, num_segments=self.n).T

    # -- one chunk of PDHG iterations + restart logic ----------------------------------
    def _kkt(self, X, Y, bnd):
        jnp = self.jnp
        l, u, rl, ru, bnorm = bnd
        AX = self._Ax(X)
        pres = AX - jnp.clip(AX, rl, ru)
        lam = self.c_s + self._ATy(Y)
        # dual residual: part of lam that no bound can absorb
        dres = jnp.where(jnp.isfinite(l) & jnp.isfinite(u), 0.0,
                         jnp.where(jnp.isfinite(l), jnp.minimum(lam, 0.0),
                                   jnp.where(jnp.isfinite(u), jnp.maximum(lam, 0.0), lam)))
        pobj = jnp.sum(self.c_s * X, axis=1)
        lo = jnp.where(lam > 0, lam * jnp.where(jnp.isfinite(l), l, 0.0),
                       lam * jnp.where(jnp.isfinite(u), u, 0.0))
        rowc = jnp.where(Y > 0, Y * jnp.where(jnp.isfinite(ru), ru, 0.0),
                         Y * jnp.where(jnp.isfinite(rl), rl, 0.0))
        dobj = jnp.sum(lo, axis=1) - jnp.sum(rowc, axis=1)
        e_p = jnp.linalg.norm(pres, axis=1) / (1.0 + bnorm)
        e_d = jnp.linalg.norm(dres, axis=1) / (1.0 + self.c_norm)
        e_g = jnp.abs(pobj - dobj) / (1.0 + jnp.abs(pobj) + jnp.abs(dobj))
        return jnp.maximum(jnp.maximum(e_p, e_d), e_g)

    def _chunk(self, state, bnd, n_inner):
        jax, jnp = self.jax, self.jnp
        (X, Y, w, Xa, Ya, na, X0, Y0, kkt0, kkt_prev, it_since, it_tot, done, it_done) = state
        l, u, rl, ru, bnorm = bnd
        eta = self.eta
        tau = (eta / w)[:, None]
        sig = (eta * w)[:, None]

        def body(_, carry):
            X, Y, Xa, Ya, na = carry
            Xn = jnp.clip(X - tau * (self.c_s + self._ATy(Y)), l, u)
            V = Y + sig * self._Ax(2.0 * Xn - X)
            Yn = V - sig * jnp.clip(V / sig, rl, ru)
            # frozen members stop moving
            Xn = jnp.where(done[:, None], X, Xn)
            Yn = jnp.where(done[:, None], Y, Yn)
            return Xn, Yn, Xa + Xn, Ya + Yn, na + 1.0

        X, Y, Xa, Ya, na = jax.lax.fori_loop(0, n_inner, body, (X, Y, Xa, Ya, na))
        it_since = it_since + n_inner
        it_tot = it_tot + n_inner
        Xavg, Yavg = Xa / na[:, None], Ya / na[:, None]
        k_cur = self._kkt(X, Y, bnd)
        k_avg = self._kkt(Xavg, Yavg, bnd)
        use_avg = k_avg < k_cur
        k_cand = jnp.minimum(k_cur, k_avg)
        Xc = jnp.where(use_avg[:, None], Xavg, X)
        Yc = jnp.where(use_avg[:, None], Yavg, Y)
        newly = (~done) & (k_cand <= self.eps)
        restart = (~done) & ((k_cand <= 0.2 * kkt0)
                             | ((k_cand <= 0.8 * kkt0) & (k_cand > kkt_prev))
                             | (it_since >= 0.36 * it_tot))
        restart = restart | newly
        # primal weight update at restarts
        dx = jnp.linalg.norm(Xc - X0, axis=1)
        dy = jnp.linalg.norm(Yc - Y0, axis=1)
        ok = restart & (dx > 1e-10) & (dy > 1e-10)
        w = jnp.where(ok, jnp.exp(0.5 * jnp.log(dy / dx) + 0.5 * jnp.log(w)), w)
        R = restart[:, None]
        X = jnp.where(R, Xc, X)
        Y = jnp.where(R, Yc, Y)
        X0 = jnp.where(R, Xc, X0)
        Y0 = jnp.where(R, Yc, Y0)
        Xa = jnp.where(R, 0.0, Xa)
        Ya = jnp.where(R, 0.0, Ya)
        na = jnp.where(restart, 0.0, na)
        kkt0 = jnp.where(restart, k_cand, kkt0)
        it_since = jnp.where(restart, 0.0, it_since)
        kkt_prev = k_cand
        it_done = jnp.where(newly, it_tot, it_done)
        done = done | newly
        return (X, Y, w, Xa, Ya, na, X0, Y0, kkt0, kkt_prev, it_since, it_tot, done, it_done), k_cand

    # -- public ------------------------------------------------------------------------------
    def solve(self, Lp: np.ndarray, Up: np.ndarray, warm: Optional[tuple] = None) -> PDHGResult:
        jnp = self.jnp
        lp = self.lp
        B = Lp.shape[0]
        t0 = time.perf_counter()
        lb = np.tile(lp.col_lb, (B, 1))
        ub = np.tile(lp.col_ub, (B, 1))
        rl = np.tile(lp.row_lb, (B, 1))
        ru = np.tile(lp.row_ub, (B, 1))
        pc = lp.param_cols
        npc = pc.size
        lb[:, pc] = Lp[:, :npc]
        ub[:, pc] = Up[:, :npc]
        if lp.param_rows.size:
            rl[:, lp.param_rows] = Lp[:, npc:]
            ru[:, lp.param_rows] = Up[:, npc:]
        bfin = np.where(np.isfinite(rl), rl, 0.0) ** 2 + np.where(np.isfinite(ru), ru, 0.0) ** 2
        bnorm = np.sqrt(bfin.sum(axis=1) / 2)
        put = self._put
        bnd = (put(lb / self.dc), put(ub / self.dc), put(rl * self.dr), put(ru * self.dr), put(bnorm))
        if warm is not None:
            X, Y, w = (put(a) for a in warm)
        else:
            X = jnp.clip(jnp.zeros((B, self.n)), bnd[0], bnd[1])
            Y = jnp.zeros((B, self.m))
            w = jnp.ones(B)
        X = jnp.clip(X, bnd[0], bnd[1])
        z = jnp.zeros(B)
        big = jnp.full(B, jnp.inf)
        state = (X, Y, w, jnp.zeros_like(X), jnp.zeros_like(Y), z, X, Y, big, big, z, z,
                 jnp.zeros(B, dtype=bool), jnp.full(B, float(self.max_iters)))
        kkt = None
        iters = 0
        while iters < self.max_iters:
            state, kkt = self._step(state, bnd)
            iters += self.check_every
            if bool(state[12].all()):
                break
        X, Y, w = state[0], state[1], state[2]
        Xh = np.asarray(X) * self.dc
        Yh = np.asarray(Y) * self.dr
        obj = Xh @ lp.objectives[0]
        return PDHGResult(x=Xh, y=Yh, objective=obj, kkt=np.asarray(kkt), converged=np.asarray(state[12]),
                          iterations=np.asarray(state[13]), seconds=time.perf_counter() - t0,
                          state=(np.asarray(X), np.asarray(Y), np.asarray(w)))
