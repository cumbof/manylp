"""Non-biological generality benchmark: Netlib LPs under right-hand-side uncertainty.

For every Netlib instance, ``B`` scenarios perturb every finite row bound by an
independent factor ``1 + sigma * N(0, 1)`` (demand / capacity uncertainty -- the
classic scenario-analysis or stochastic-programming inner loop).  Each
scenario is a full LP sharing ``A`` and ``c``; some become infeasible.

Solvers: manylp (CPU and GPU, cold start) and HiGHS dual simplex in 32
processes, either cold (fresh solve per scenario) or hot-started from the
previous scenario's basis.  Reported: throughput, distinct critical regions,
agreement with HiGHS (status and objective).
"""

import argparse
import glob
import json
import os
import time
import warnings

import numpy as np
import scipy.sparse as sp

warnings.filterwarnings("ignore")

from manylp import BatchLPSolver, LexLP  # noqa: E402
from manylp.reference import HighsBaseline  # noqa: E402


def load_mps(path):
    import highspy

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.readModel(path)
    lp = h.getLp()
    n, m = lp.num_col_, lp.num_row_
    A = sp.csc_matrix((np.asarray(lp.a_matrix_.value_), np.asarray(lp.a_matrix_.index_),
                       np.asarray(lp.a_matrix_.start_)), shape=(m, n))
    c = np.asarray(lp.col_cost_)
    sense = 1.0 if lp.sense_ == highspy.ObjSense.kMinimize else -1.0
    inf = highspy.kHighsInf
    fix = lambda a: np.where(np.abs(a) >= inf, np.sign(a) * np.inf, a)  # noqa: E731
    return A, sense * c, fix(np.asarray(lp.col_lower_)), fix(np.asarray(lp.col_upper_)), \
        fix(np.asarray(lp.row_lower_)), fix(np.asarray(lp.row_upper_)), lp.offset_


def _bunching_row(rec, lp, L, U, cpu, args):
    """Classical bunching (benchmarks/bunching.py) on the same scenarios, checked against manylp."""
    from bunching import bunching_pool_solve

    g = cpu.register_group(lp, out_z=np.zeros(0, dtype=np.int64))
    t = time.perf_counter()
    sol = cpu.solve_batch(g, L, U, return_z=False)
    dt = time.perf_counter() - t
    rec["manylp-cpu"] = {"seconds": dt, "lps_per_second": args.B / dt, "distinct_bases": len(g.pool)}
    for label, procs in (("bunching-p1", 1), (f"bunching-p{args.procs}", args.procs)):
        st, obj, stats, secs = bunching_pool_solve(lp, L, U, n_procs=procs)
        decided = np.isin(st, (1, 2)) & np.isin(sol.status, (1, 2))
        both = (st == 1) & (sol.status == 1)
        rec[label] = {"seconds": secs, "lps_per_second": args.B / secs, **stats,
                      "status_agreement": float((st[decided] == sol.status[decided]).mean()) if decided.any() else None,
                      "undecided": int((~np.isin(st, (1, 2))).sum()),
                      "obj_rel_err_max": float((np.abs(obj[both] - sol.objective[both, 0]) /
                                                np.maximum(1.0, np.abs(sol.objective[both, 0]))).max())
                      if both.any() else None}
    b1, bp = rec["bunching-p1"], rec[f"bunching-p{args.procs}"]
    print(f"{rec['problem']:10s} manylp-cpu {rec['manylp-cpu']['lps_per_second']:9.0f} LP/s | bunching p1 "
          f"{b1['lps_per_second']:8.0f} LP/s ({b1['simplex_solves']} solves) | p{args.procs} {bp['lps_per_second']:8.0f} "
          f"LP/s ({bp['simplex_solves']} solves) | agree {bp['status_agreement']} err {bp['obj_rel_err_max']}", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="../netlib")
    ap.add_argument("--B", type=int, default=1024)
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--procs", type=int, default=32)
    ap.add_argument("--out", default="results/netlib")
    ap.add_argument("--only", default="")
    ap.add_argument("--bunching", action="store_true",
                    help="run only the classical bunching baseline (1 and --procs processes) plus manylp-cpu "
                         "for agreement; the scenarios are identical to the main run (same seed)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    files = sorted(glob.glob(f"{args.dir}/*.mps"))
    if args.only:
        files = [f for f in files if os.path.basename(f)[:-4] in args.only.split(",")]
    cpu = BatchLPSolver("cpu", n_workers=32)
    gpu = BatchLPSolver("cuda", n_workers=32)
    rows = []
    for f in files:
        name = os.path.basename(f)[:-4]
        A, cmin, cl, cu, rl, ru, off = load_mps(f)
        m, n = A.shape
        rows_p = np.nonzero(np.isfinite(rl) | np.isfinite(ru))[0]
        # maximise -c  (== minimise c); rows with finite bounds are parametric
        lp = LexLP(A=A, objectives=-cmin, col_lb=cl, col_ub=cu, row_lb=rl, row_ub=ru,
                   param_rows=rows_p)
        rng = np.random.default_rng(0)
        L0, U0 = lp.template_param_bounds()
        f_ = 1.0 + args.sigma * rng.standard_normal((args.B, L0.size))
        L = np.where(np.isfinite(L0), L0 * f_, L0)
        U = np.where(np.isfinite(U0), U0 * f_, U0)
        # rows that were equalities stay equalities, ranges stay ordered
        L, U = np.minimum(L, U), np.maximum(L, U)
        rec = {"problem": name, "m": m, "n": n, "nnz": int(A.nnz), "B": args.B, "p": int(lp.p)}
        if args.bunching:
            rows.append(_bunching_row(rec, lp, L, U, cpu, args))
            json.dump(rows, open(f"{args.out}/netlib_bunching_B{args.B}.json", "w"), indent=1)
            continue
        res = {}
        for label, solver in (("manylp-cpu", cpu), ("manylp-gpu", gpu)):
            g = solver.register_group(lp, out_z=np.zeros(0, dtype=np.int64))
            t = time.perf_counter()
            sol = solver.solve_batch(g, L, U, return_z=False)
            dt = time.perf_counter() - t
            res[label] = sol
            rec[label] = {"seconds": dt, "lps_per_second": args.B / dt,
                          "simplex_solves": sol.stats["highs_solves"], "distinct_bases": len(g.pool),
                          "farkas": len(g.pool.farkas), "certified": float(sol.certified.mean()),
                          "optimal": int((sol.status == 1).sum()), "infeasible": int((sol.status == 2).sum())}
        for label, warm in (("highs-cold", False), ("highs-hot", True)):
            hb = HighsBaseline(lp, out_z=np.zeros(0, dtype=np.int64), n_procs=args.procs, warm=warm)
            st, obj, _, info = hb.solve(L, U)
            hb.close()
            rec[f"{label}-p{args.procs}"] = {"seconds": info["seconds"],
                                              "lps_per_second": args.B / info["seconds"],
                                              "iterations": int(info["iterations"])}
            res[label] = (st, obj)
        st_h, obj_h = res["highs-cold"]
        obj_h = obj_h[:, 0]
        sol = res["manylp-cpu"]
        both = (st_h == 1) & (sol.status == 1)
        # agreement where HiGHS reached a verdict (optimal / infeasible); HiGHS numerical
        # failures that manylp answered with a certificate are counted separately
        decided = np.isin(st_h, (1, 2))
        rec["status_agreement"] = float((st_h[decided] == sol.status[decided]).mean()) if decided.any() else None
        rec["highs_errors"] = int((~decided).sum())
        rec["highs_errors_certified_by_manylp"] = int((~decided & sol.certified & np.isin(sol.status, (1, 2))).sum())
        rec["manylp_errors"] = int((~np.isin(sol.status, (1, 2))).sum())
        rec["obj_rel_err_max"] = float((np.abs(sol.objective[both, 0] - obj_h[both]) /
                                        np.maximum(1.0, np.abs(obj_h[both]))).max()) if both.any() else None
        rows.append(rec)
        best_h = max(rec[f"highs-cold-p{args.procs}"]["lps_per_second"], rec[f"highs-hot-p{args.procs}"]["lps_per_second"])
        best_m = max(rec["manylp-cpu"]["lps_per_second"], rec["manylp-gpu"]["lps_per_second"])
        print(f"{name:10s} m={m:5d} n={n:5d} | manylp {best_m:9.0f} LP/s ({rec['manylp-cpu']['distinct_bases']:4d} regions, "
              f"{rec['manylp-cpu']['infeasible']:4d} infeas) | HiGHS x{args.procs} {best_h:8.0f} LP/s | "
              f"speedup {best_m / best_h:6.1f}x | agree {rec['status_agreement']} err {rec['obj_rel_err_max']} | "
              f"HiGHS errors {rec['highs_errors']} (certified by manylp: {rec['highs_errors_certified_by_manylp']}), "
              f"manylp errors {rec['manylp_errors']}",
              flush=True)
        json.dump(rows, open(f"{args.out}/netlib_B{args.B}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
