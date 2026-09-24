"""Fused, multi-threaded host kernels for batched certification (Numba).

These mirror :mod:`manylp.kernels` (the CUDA versions) one to one: the dense
products stay in (multi-threaded) BLAS, and the per-member elementwise work --
which NumPy would spread over a dozen full-size temporaries -- is fused into
single parallel passes.  If Numba is not installed, :func:`available` is False
and the solver uses its NumPy path.
"""

from __future__ import annotations

import numpy as np

try:
    import numba as nb

    _NUMBA = True
except Exception:  # pragma: no cover
    _NUMBA = False

if _NUMBA:
    import os as _os

    # Numba defaults to one thread per hardware thread; on large shared nodes the
    # fork/join cost of hundreds of threads dwarfs these memory-bound loops.
    _nthreads = int(_os.environ.get("MANYLP_NUM_THREADS", 0)) or min(16, _os.cpu_count() or 1)
    try:
        nb.set_num_threads(min(_nthreads, nb.config.NUMBA_NUM_THREADS))
    except Exception:  # pragma: no cover
        pass


def available() -> bool:
    return _NUMBA


if _NUMBA:

    @nb.njit(parallel=True, cache=True, fastmath=False)
    def np_prep(L, U, pidx, st, tmpl, needs_fixed, nonunique, act_slot, n_act,
                zNP, dact, ok, uniq, need):
        Bg = L.shape[0]
        n_np = pidx.shape[0]
        for b in nb.prange(Bg):
            good_b = True
            uniq_b = uniq[b]
            for j in range(n_np):
                l = L[b, pidx[j]]
                u = U[b, pidx[j]]
                s = st[j]
                z = u if s == 1 else (l if s == 0 else 0.0)
                good = np.isfinite(z)
                if s == 2:
                    good = good and (l <= 0.0) and (u >= 0.0)
                fixed = l == u
                if needs_fixed[j] and not fixed:
                    good = False
                if not good:
                    good_b = False
                    z = tmpl[j]
                if nonunique[j] and not fixed:
                    uniq_b = False
                zNP[b, j] = z
                d = z - tmpl[j]
                a = act_slot[j]
                if a >= 0:
                    dact[b, a] = d
                elif d != 0.0:
                    need[j] = 1          # benign race: every writer stores 1
            ok[b] = good_b
            uniq[b] = uniq_b

    @nb.njit(parallel=True, cache=True, fastmath=False)
    def basic_check(zB, h, lo, hi, bp_of_row, L, U, pabs, prel, ok):
        Bg, m = zB.shape
        for b in nb.prange(Bg):
            good = ok[b]
            for i in range(m):
                v = zB[b, i] + h[i]
                zB[b, i] = v
                q = bp_of_row[i]
                if q >= 0:
                    lb = L[b, q]
                    ub = U[b, q]
                else:
                    lb = lo[i]
                    ub = hi[i]
                if (lb - v > pabs + prel * abs(lb)) or (v - ub > pabs + prel * abs(ub)):
                    good = False
            ok[b] = good

    @nb.njit(parallel=True, cache=True)
    def gather_out(zB, zNP, kind, idx, val, out):
        Bg, n_out = out.shape
        for b in nb.prange(Bg):
            for o in range(n_out):
                k = kind[o]
                if k == 0:
                    out[b, o] = zB[b, idx[o]]
                elif k == 1:
                    out[b, o] = zNP[b, idx[o]]
                else:
                    out[b, o] = val[o]


_WARM = False


def warm_up() -> None:
    """Compile the kernels once (on tiny arrays) so no timed call pays for JIT."""
    global _WARM
    if not _NUMBA or _WARM:
        return
    L = np.zeros((2, 3))
    U = np.ones((2, 3))
    pidx = np.array([0, 1], dtype=np.int64)
    st = np.array([0, 1], dtype=np.int8)
    tmpl = np.zeros(2)
    b = np.zeros(2, dtype=np.bool_)
    act = np.array([0, -1], dtype=np.int32)
    zNP = np.empty((2, 2))
    dact = np.empty((2, 1))
    ok = np.ones(2, dtype=np.bool_)
    uq = np.ones(2, dtype=np.bool_)
    need = np.zeros(2, dtype=np.int8)
    np_prep(L, U, pidx, st, tmpl, b, b, act, 1, zNP, dact, ok, uq, need)
    zB = np.zeros((2, 2))
    basic_check(zB, np.zeros(2), np.full(2, -np.inf), np.full(2, np.inf), np.array([-1, 0], dtype=np.int64),
                L, U, 1e-9, 1e-9, ok)
    out = np.empty((2, 2))
    gather_out(zB, zNP, np.array([0, 1], dtype=np.int8), np.array([0, 1], dtype=np.int64), np.zeros(2), out)
    _WARM = True
