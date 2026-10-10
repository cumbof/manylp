"""Worst case for propagation: Netlib adlittle, 1024 scenarios scattered over ~500 regions."""
import sys, time, numpy as np, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "benchmarks")
from bench_netlib import load_mps
from manylp import BatchLPSolver, LexLP
from manylp.reference import HighsBaseline


def main():
    for name in ("adlittle", "afiro", "sc105"):
        A, cmin, cl, cu, rl, ru, off = load_mps(f"benchmarks/inputs/netlib/{name}.mps")
        rows_p = np.nonzero(np.isfinite(rl) | np.isfinite(ru))[0]
        lp = LexLP(A=A, objectives=-cmin, col_lb=cl, col_ub=cu, row_lb=rl, row_ub=ru, param_rows=rows_p)
        rng = np.random.default_rng(0); L0, U0 = lp.template_param_bounds()
        f = 1.0 + 0.05 * rng.standard_normal((1024, L0.size))
        L = np.where(np.isfinite(L0), L0 * f, L0); U = np.where(np.isfinite(U0), U0 * f, U0); L, U = np.minimum(L, U), np.maximum(L, U)
        res = {}
        for guard in (1e-9, 1.5):
            s = BatchLPSolver("cpu", n_workers=32, min_yield=guard); g = s.register_group(lp, out_z=np.zeros(0, dtype=np.int64))
            t = time.perf_counter(); sol = s.solve_batch(g, L, U, return_z=False); dt = time.perf_counter() - t
            res[guard] = (dt, sol)
            print(f"{name:9s} guard={'on ' if guard > 1 else 'off'} {1024 / dt:8.0f} LP/s regions {len(g.pool):4d} simplex {sol.stats['highs_solves']:5d} "
                  f"direct {sol.stats['direct_mode']} certified {sol.certified.mean():.3f}", flush=True)
            s.close()
        hb = HighsBaseline(lp, out_z=np.zeros(0, dtype=np.int64), n_procs=32, warm=True)
        st, obj, _, info = hb.solve(L, U); hb.close()
        a = res[1.5][1]
        both = (st == 1) & (a.status == 1)
        print(f"{name:9s} HiGHS hot x32 {1024 / info['seconds']:8.0f} LP/s | status agree {np.mean(st == a.status):.3f} "
              f"max obj err {np.max(np.abs(a.objective[both, 0] - obj[both, 0]) / np.maximum(1, np.abs(obj[both, 0]))):.1e}", flush=True)


if __name__ == "__main__":
    main()
