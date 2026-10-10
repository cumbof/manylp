"""End-to-end community dFBA benchmark (muODE gut_western: 12 gapseq GEMs, western diet).

Every solver replays the *identical* dFBA workload (same community, diet,
kinetics, ensemble perturbations); results and trajectories are written to
``results/dfba/<tag>.json|.npz`` for the accuracy analysis.

Example::

    python benchmarks/bench_dfba.py --E 1,64 --solvers manylp-cuda,highs-warm-x32 --t-end 48
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from manylp.dfba import (  # noqa: E402
    CobraAdapter,
    HighsLoopAdapter,
    HighsProcAdapter,
    ManyLPAdapter,
    ScipyLinprogAdapter,
    load_gut_community,
    run_dfba,
)

G = os.environ.get("GUT_DIR", "benchmarks/inputs/gut_western")


def make_adapter(name: str, files):
    if name == "manylp-cuda":
        return ManyLPAdapter("cuda", n_workers=32)
    if name == "manylp-cpu":
        return ManyLPAdapter("cpu", n_workers=32)
    if name == "manylp-cuda-perlp":
        return ManyLPAdapter("cuda", n_workers=32, per_lp=True)
    if name.startswith("highs-warm-p"):
        return HighsProcAdapter(n_procs=int(name.split("p")[-1]), warm=True)
    if name.startswith("highs-warm-x"):
        return HighsLoopAdapter(warm=True, n_workers=int(name.split("x")[-1]))
    if name.startswith("highs-cold-x"):
        return HighsLoopAdapter(warm=False, n_workers=int(name.split("x")[-1]))
    if name == "scipy-linprog":
        return ScipyLinprogAdapter()
    if name == "bunching-batch":             # batched (Kall & Wallace) bunching, one process
        from bunching import BunchingBatchAdapter
        return BunchingBatchAdapter()
    if name.startswith("bunching-p"):         # classical bunching baseline, plain FBA only
        from bunching import BunchingProcAdapter
        return BunchingProcAdapter(n_procs=int(name.split("-p")[-1]))
    if name.startswith("cobra-"):
        return CobraAdapter(files, solver=name.split("-", 1)[1])
    raise ValueError(name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--E", default="1")
    ap.add_argument("--solvers", default="manylp-cuda")
    ap.add_argument("--mode", default="pfba-unique")
    ap.add_argument("--t-end", type=float, default=48.0)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--species", default="")
    ap.add_argument("--sigma", type=float, default=0.3, help="lognormal sd of diet perturbation")
    ap.add_argument("--record-every", type=int, default=10)
    ap.add_argument("--out", default="results/dfba")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    names = args.species.split(",") if args.species else None
    t0 = time.perf_counter()
    comm = load_gut_community(f"{G}/gems", f"{G}/gems/western_gut_modelseed.csv", names=names,
                              abundance_tsv=f"{G}/abundance.tsv")
    files = sorted(glob.glob(f"{G}/gems/*.xml.gz"))
    if names:
        files = [f for f in files if any(os.path.basename(f).startswith(n) for n in names)]
    print(f"community: {len(comm.models)} species, {len(comm.env_mets)} metabolites "
          f"(loaded in {time.perf_counter() - t0:.1f}s)", flush=True)

    for E in [int(e) for e in args.E.split(",")]:
        pert = None
        if E > 1:
            pert = np.random.default_rng(0).lognormal(0.0, args.sigma, size=(E, len(comm.env_mets)))
            pert[0] = 1.0   # member 0 is always the unperturbed diet
        for sname in args.solvers.split(","):
            ad = make_adapter(sname, files)
            r = run_dfba(comm, ad, E=E, t_end=args.t_end, dt=args.dt, mode=args.mode, perturb=pert,
                         record_every=args.record_every)
            ad.close()
            tag = f"{args.tag}{sname}_{args.mode}_E{E}_T{args.t_end:g}"
            st = {k: v for k, v in r.solver_stats.items() if not isinstance(v, (list, dict))}
            rec = {
                "solver": sname, "mode": args.mode, "E": E, "t_end": args.t_end, "dt": args.dt,
                "species": [m.name for m in comm.models], "n_lps": r.n_lps,
                "solve_seconds": r.solve_seconds, "wall_seconds": r.wall_seconds,
                "lps_per_second": r.n_lps / max(r.solve_seconds, 1e-12),
                "solver_stats": st, "pool_sizes": r.solver_stats.get("pool_sizes"),
                "host": platform.node(), "sigma": args.sigma,
            }
            with open(f"{args.out}/{tag}.json", "w") as fh:
                json.dump(rec, fh, indent=1)
            np.savez_compressed(f"{args.out}/{tag}.npz", times=r.times, X=r.X, M=r.M, mu=r.mu)
            extra = ""
            if "highs_solves" in st:
                extra = (f" | simplex solves {st['highs_solves']} ({100 * st['highs_solves'] / r.n_lps:.3f}%)"
                         f" certify {st.get('certify_seconds', 0):.1f}s repair {st.get('repair_seconds', 0):.1f}s")
            print(f"E={E:5d} {sname:16s} {args.mode:12s} LPs {r.n_lps:9d} solve {r.solve_seconds:9.2f}s "
                  f"-> {rec['lps_per_second']:10.0f} LP/s{extra}", flush=True)


if __name__ == "__main__":
    main()
