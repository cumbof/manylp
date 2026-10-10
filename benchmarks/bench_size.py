"""Scaling with LP size: steady-state throughput across genome-scale models.

For each model, B members get log-normally perturbed open-exchange uptake bounds;
the batch then drifts by a 1%-per-step random walk for T steps (dFBA-like
coherence).  Steady-state throughput (steps >= 2) of manylp (GPU, CPU) and of
warm-started HiGHS in 32 processes (member-pinned), for plain FBA and pFBA-unique.
"""
import argparse, glob, json, os, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
import cobra
from manylp.dfba import HighsProcAdapter, ManyLPAdapter
from manylp.fba import FBAModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=1024)
    ap.add_argument("--T", type=int, default=12)
    ap.add_argument("--modes", default="fba,pfba-unique")
    ap.add_argument("--highs-steps", type=int, default=4, help="HiGHS is timed on the first steps only")
    args = ap.parse_args()
    gut = os.environ.get("GUT_DIR", "../muODE/examples/gut_western")
    files = [("e_coli_core", "benchmarks/inputs/bigg/e_coli_core.xml.gz"), ("iYO844", "benchmarks/inputs/bigg/iYO844.xml.gz"),
             ("iMM904", "benchmarks/inputs/bigg/iMM904.xml.gz"),
             ("Bl_obeum (gapseq)", f"{gut}/gems/Bl_obeum_A2162.xml.gz"),
             ("B_thetaiotaomicron (gapseq)", f"{gut}/gems/B_thetaiotaomicron_VPI5482.xml.gz"),
             ("iJO1366", "benchmarks/inputs/bigg/iJO1366.xml.gz"), ("iML1515", "benchmarks/inputs/bigg/iML1515.xml.gz"),
             ("Recon3D", "benchmarks/inputs/bigg/Recon3D.xml.gz")]
    os.makedirs("results/size", exist_ok=True)
    rows = []
    for label, f in files:
        mdl = cobra.io.read_sbml_model(f)
        fm = FBAModel.from_cobra(mdl)
        lb0 = fm.lb[fm.exchanges]
        open_ = lb0 < 0
        rng = np.random.default_rng(0)
        base = np.where(open_, np.maximum(lb0, -20.0), 0.0)
        P = rng.lognormal(0, 0.3, size=(args.B, lb0.size))
        steps = []
        x = base[None, :] * P
        for t in range(args.T):
            steps.append(x.copy())
            x = x * np.exp(0.01 * rng.standard_normal(x.shape))
        members = np.arange(args.B)
        for mode in args.modes.split(","):
            rec = {"model": label, "m": fm.m, "n": fm.n, "exchanges": int(fm.exchanges.size), "mode": mode, "B": args.B}
            for name, ad, nsteps in (("manylp-gpu", ManyLPAdapter("cuda", n_workers=32), args.T),
                                     ("manylp-cpu", ManyLPAdapter("cpu", n_workers=32), args.T),
                                     ("highs-warm-p32", HighsProcAdapter(32), args.highs_steps)):
                ad.setup([fm], mode)
                ts = []
                for t in range(nsteps):
                    t0 = time.perf_counter()
                    ad.solve(0, steps[t], members)
                    ts.append(time.perf_counter() - t0)
                st = ad.stats()
                ad.close()
                steady = args.B * (nsteps - 2) / sum(ts[2:])
                rec[name] = {"first_step_s": ts[0], "steady_lps_per_second": steady,
                             "regions": (st.get("pool_sizes") or [None])[0], "simplex_solves": st.get("highs_solves")}
                print(f"{label:28s} n={fm.n:6d} {mode:12s} {name:15s} steady {steady:10,.0f} LP/s  first step {ts[0]:7.2f}s",
                      flush=True)
            rec["speedup_gpu"] = rec["manylp-gpu"]["steady_lps_per_second"] / rec["highs-warm-p32"]["steady_lps_per_second"]
            rec["speedup_cpu"] = rec["manylp-cpu"]["steady_lps_per_second"] / rec["highs-warm-p32"]["steady_lps_per_second"]
            rows.append(rec)
            json.dump(rows, open("results/size/size.json", "w"), indent=1)


if __name__ == "__main__":
    # HighsProcAdapter spawns worker processes, which re-import this module
    main()
