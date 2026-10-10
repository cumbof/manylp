"""Record the LP instances a real community dFBA ensemble generates.

Runs the 12-species gut community with an E-member ensemble and saves, every
``--every`` steps, the exchange lower bounds of every (species, member) LP --
the exact inputs a solver sees in muODE -- to ``results/workload.npz``.  The
solver comparison (``bench_solvers.py``) replays these instances so every
solver solves the same, realistic LPs.
"""

import argparse
import os
import pickle
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from manylp.dfba import ManyLPAdapter, load_gut_community, run_dfba  # noqa: E402

G = os.environ.get("GUT_DIR", "benchmarks/inputs/gut_western")


class Recorder(ManyLPAdapter):
    def __init__(self, every, n_species, **kw):
        super().__init__(**kw)
        self.every = every
        self.n_species = n_species
        self.k = 0
        self.snap = []   # (step, species, member_ids, ex_lb)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--E", type=int, default=64)
    ap.add_argument("--every", type=int, default=16)
    ap.add_argument("--out", default="results/workload.pkl")
    ap.add_argument("--t-end", type=float, default=48.0)
    args = ap.parse_args()
    comm = load_gut_community(f"{G}/gems", f"{G}/gems/western_gut_modelseed.csv",
                              abundance_tsv=f"{G}/abundance.tsv")
    pert = np.random.default_rng(0).lognormal(0.0, 0.3, size=(args.E, len(comm.env_mets)))
    pert[0] = 1.0
    rec = Recorder(args.every, len(comm.models), device="cuda", n_workers=32)
    # species calls happen in order each step, but dormant species skip calls; record the
    # species index explicitly instead of inferring it from the call counter
    orig = rec.solve
    counter = {"step": 0, "last_s": -1}

    def solve(s, ex_lb, member_ids):
        if s <= counter["last_s"]:
            counter["step"] += 1
        counter["last_s"] = s
        if counter["step"] % args.every == 0:
            rec.snap.append((counter["step"], s, member_ids.copy(), ex_lb.copy()))
        return orig(s, ex_lb, member_ids)

    rec.solve = solve
    run_dfba(comm, rec, E=args.E, t_end=args.t_end, dt=0.1, perturb=pert)
    models = [(m.name, m) for m in comm.models]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as fh:
        pickle.dump({"models": models, "snapshots": rec.snap, "E": args.E, "every": args.every}, fh)
    n = sum(x[3].shape[0] for x in rec.snap)
    print(f"recorded {len(rec.snap)} batches, {n} LPs -> {args.out}")


if __name__ == "__main__":
    main()
