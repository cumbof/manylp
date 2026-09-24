"""Fused CUDA kernels for batched certification (CuPy ``RawKernel``).

The generic array-API certification issues ~100 small device operations per
basis, so for moderate batches it is launch/overhead bound.  These kernels fuse
it into three custom launches around the two cuBLAS products:

1. ``np_prep``    member x nonbasic-parametric: statuses -> values, dual-validity
                  and uniqueness flags, the (lazily active) parameter deltas;
2. cuBLAS GEMM    ``zB = delta_act @ T_act^T``;
3. ``basic_check`` member x basic: add ``h``, primal feasibility against static or
                  the member's own bounds;
4. cuBLAS GEMMs   stage objectives;
5. ``gather_out`` member x requested output.

Validity flags use ``atomicAnd`` only when a check *fails*, so there is no
contention on the common (certified) path.
"""

from __future__ import annotations

_SRC = r"""
extern "C" __global__ void np_prep(
    const double* __restrict__ L, const double* __restrict__ U, const long long p,
    const long long* __restrict__ pidx, const signed char* __restrict__ st,
    const double* __restrict__ tmpl, const unsigned char* __restrict__ needs_fixed,
    const unsigned char* __restrict__ nonunique, const int* __restrict__ act_slot,
    const long long n_act, const long long n_np, const long long Bg,
    double* __restrict__ zNP, double* __restrict__ dact,
    int* __restrict__ ok, int* __restrict__ uniq, int* __restrict__ need)
{
    const long long total = Bg * n_np;
    for (long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x; t < total;
         t += (long long)gridDim.x * blockDim.x) {
        const long long b = t / n_np;
        const long long j = t - b * n_np;
        const double l = L[b * p + pidx[j]];
        const double u = U[b * p + pidx[j]];
        const signed char s = st[j];
        double z = (s == 1) ? u : ((s == 0) ? l : 0.0);
        bool good = isfinite(z);
        if (s == 2) good = good && (l <= 0.0) && (u >= 0.0);
        const bool fixed = (l == u);
        if (needs_fixed[j] && !fixed) good = false;
        if (!good) { atomicAnd(ok + b, 0); z = tmpl[j]; }
        if (nonunique[j] && !fixed) atomicAnd(uniq + b, 0);
        zNP[t] = z;
        const double d = z - tmpl[j];
        const int a = act_slot[j];
        if (a >= 0) dact[b * n_act + a] = d;
        else if (d != 0.0) need[j] = 1;
    }
}

extern "C" __global__ void basic_check(
    double* __restrict__ zB, const double* __restrict__ h,
    const double* __restrict__ lo, const double* __restrict__ hi,
    const long long* __restrict__ bp_of_row,
    const double* __restrict__ L, const double* __restrict__ U, const long long p,
    const double pabs, const double prel, const long long m, const long long Bg,
    int* __restrict__ ok)
{
    const long long total = Bg * m;
    for (long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x; t < total;
         t += (long long)gridDim.x * blockDim.x) {
        const long long b = t / m;
        const long long i = t - b * m;
        const double v = zB[t] + h[i];
        zB[t] = v;
        const long long q = bp_of_row[i];
        double l, u;
        if (q >= 0) { l = L[b * p + q]; u = U[b * p + q]; }
        else { l = lo[i]; u = hi[i]; }
        if ((l - v > pabs + prel * fabs(l)) || (v - u > pabs + prel * fabs(u)))
            atomicAnd(ok + b, 0);
    }
}

extern "C" __global__ void gather_out(
    const double* __restrict__ zB, const double* __restrict__ zNP,
    const signed char* __restrict__ kind, const long long* __restrict__ idx,
    const double* __restrict__ val, const long long m, const long long n_np,
    const long long n_out, const long long Bg, double* __restrict__ out)
{
    const long long total = Bg * n_out;
    for (long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x; t < total;
         t += (long long)gridDim.x * blockDim.x) {
        const long long b = t / n_out;
        const long long o = t - b * n_out;
        const signed char k = kind[o];
        out[t] = (k == 0) ? zB[b * m + idx[o]] : ((k == 1) ? zNP[b * n_np + idx[o]] : val[o]);
    }
}
"""

_MOD = {}


def module():
    import cupy as cp

    dev = cp.cuda.Device().id
    mod = _MOD.get(dev)
    if mod is None:
        mod = _MOD[dev] = cp.RawModule(code=_SRC, options=("--std=c++14",))
    return mod


def launch(kernel_name: str, total: int, args) -> None:
    import cupy as cp

    k = module().get_function(kernel_name)
    threads = 256
    blocks = int(min(max(1, (total + threads - 1) // threads), 65535 * 4))
    if total > 0:
        k((blocks,), (threads,), args)
    del cp
