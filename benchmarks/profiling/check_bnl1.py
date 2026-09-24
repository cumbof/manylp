"""Investigate manylp vs HiGHS status disagreements on Netlib bnl1 under RHS perturbation."""
import sys, numpy as np, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "benchmarks")
from bench_netlib import load_mps
from manylp import BatchLPSolver, LexLP
from manylp.reference import HighsBaseline, reference_solve


def main():
    A, cmin, cl, cu, rl, ru, off = load_mps("../netlib/bnl1.mps")
    rows_p = np.nonzero(np.isfinite(rl) | np.isfinite(ru))[0]
    lp = LexLP(A=A, objectives=-cmin, col_lb=cl, col_ub=cu, row_lb=rl, row_ub=ru, param_rows=rows_p)
    rng = np.random.default_rng(0); L0, U0 = lp.template_param_bounds()
    f = 1.0 + 0.05 * rng.standard_normal((1024, L0.size))
    L = np.where(np.isfinite(L0), L0 * f, L0); U = np.where(np.isfinite(U0), U0 * f, U0); L, U = np.minimum(L, U), np.maximum(L, U)
    s = BatchLPSolver("cpu", n_workers=32); g = s.register_group(lp, out_z=np.zeros(0, dtype=np.int64))
    sol = s.solve_batch(g, L, U, return_z=False)
    hb = HighsBaseline(lp, out_z=np.zeros(0, dtype=np.int64), n_procs=32, warm=False)
    st, obj, _, _ = hb.solve(L, U); hb.close()
    decided = np.isin(st, (1, 2))
    print(f"direct mode: {sol.stats['direct_mode']}; decided by HiGHS: {decided.sum()}; "
          f"agreement on decided: {(st[decided] == sol.status[decided]).mean():.4f}")
    bad = np.nonzero(decided & (st != sol.status))[0]
    print(f"disagreements: {bad.size}; manylp statuses {np.bincount(sol.status[bad], minlength=5)}, HiGHS {np.bincount(st[bad], minlength=5)}")
    print("sources of manylp answers on disagreements:", np.bincount(sol.source[bad].astype(int) + 1, minlength=8))
    for b in bad[:20]:
        lb, ub = lp.full_bounds(L[b], U[b])
        r = reference_solve(lp, lb, ub)
        print(f"member {b:4d}: manylp {sol.status[b]} (src {sol.source[b]}, cert {sol.certified[b]}), highs-cold {st[b]}, reference {r.status}, "
              f"obj manylp {sol.objective[b,0]:.6g} ref {r.stage_obj[0]:.6g}")


if __name__ == "__main__":
    main()
