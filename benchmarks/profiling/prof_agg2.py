"""Where does the time go on an incoherent scenario batch (Netlib agg2, sigma = 5%)?"""
import cProfile, pstats, sys, time, numpy as np, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "benchmarks")
from bench_netlib import load_mps
from manylp import BatchLPSolver, LexLP


def main():
    A, cmin, cl, cu, rl, ru, off = load_mps("benchmarks/inputs/netlib/agg2.mps")
    rows_p = np.nonzero(np.isfinite(rl) | np.isfinite(ru))[0]
    lp = LexLP(A=A, objectives=-cmin, col_lb=cl, col_ub=cu, row_lb=rl, row_ub=ru, param_rows=rows_p)
    rng = np.random.default_rng(0); L0, U0 = lp.template_param_bounds()
    f = 1.0 + 0.05 * rng.standard_normal((1024, L0.size))
    L = np.where(np.isfinite(L0), L0 * f, L0); U = np.where(np.isfinite(U0), U0 * f, U0); L, U = np.minimum(L, U), np.maximum(L, U)
    s = BatchLPSolver("cpu", n_workers=32); s.warm_up_processes()
    g = s.register_group(lp, out_z=np.zeros(0, dtype=np.int64))
    pr = cProfile.Profile(); pr.enable(); t = time.perf_counter()
    sol = s.solve_batch(g, L, U, return_z=False)
    dt = time.perf_counter() - t; pr.disable()
    print(f"{1024 / dt:.0f} LP/s, simplex {sol.stats['highs_solves']}, direct {sol.stats['direct_mode']}, "
          f"sources {dict((k, sol.stats[k]) for k in ('warm', 'pool', 'propagated', 'repaired', 'highs_uncertified'))}")
    pstats.Stats(pr).sort_stats("tottime").print_stats(22)
    s.close()


if __name__ == "__main__":
    main()
