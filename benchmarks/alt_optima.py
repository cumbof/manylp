"""Alternative-optima envelope of the 12-species gut community dFBA.

Integrates one trajectory per selection policy (canonical pFBA-unique, random
directions on the optimal face, extreme secretion of each fermentation product)
in lockstep with the batched solver, and reports how far apart they end up.
"""
import argparse, json, os, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
from manylp.dfba import load_gut_community, run_dfba
from manylp.envelope import PolicyAdapter, standard_policies

G = os.environ.get("GUT_DIR", "benchmarks/inputs/gut_western")
# ModelSEED ids of the main fermentation products / cross-fed metabolites
PRODUCTS = {"acetate": "cpd00029_e0", "butyrate": "cpd00211_e0", "propionate": "cpd00141_e0",
            "L-lactate": "cpd00159_e0", "succinate": "cpd00036_e0", "formate": "cpd00047_e0",
            "H2": "cpd11640_e0", "ethanol": "cpd00363_e0"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-random", type=int, default=32)
    ap.add_argument("--t-end", type=float, default=48.0)
    ap.add_argument("--out", default="results/alt_optima")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    comm = load_gut_community(f"{G}/gems", f"{G}/gems/western_gut_modelseed.csv", abundance_tsv=f"{G}/abundance.tsv")
    mets = [m for m in PRODUCTS.values() if m in comm.env_mets]
    pols = standard_policies(args.n_random, mets)
    E = len(pols)
    ad = PolicyAdapter(pols, np.arange(E), device="cuda")
    t = time.perf_counter()
    r = run_dfba(comm, ad, E=E, t_end=args.t_end, dt=0.1, mode="pfba-unique", record_every=1)
    wall = time.perf_counter() - t
    st = ad.stats()
    names = [m.name for m in comm.models]
    np.savez_compressed(f"{args.out}/envelope.npz", times=r.times, X=r.X, M=r.M, mu=r.mu,
                        policies=np.array([p.name for p in pols]), species=np.array(names),
                        env_mets=np.array(comm.env_mets))
    Xf = r.X[-1]                              # (E, S)
    rel = Xf / Xf.sum(axis=1, keepdims=True)
    summary = {"policies": E, "lps": r.n_lps, "solve_seconds": r.solve_seconds, "wall_seconds": wall,
               "stats": st,
               "final_biomass_range": {n: [float(Xf[:, i].min()), float(Xf[:, i].max())] for i, n in enumerate(names)},
               "final_rel_abundance_range": {n: [float(rel[:, i].min()), float(rel[:, i].max())] for i, n in enumerate(names)},
               "final_product_mM_range": {}}
    for lab, m in PRODUCTS.items():
        if m in comm.env_mets:
            j = comm.env_mets.index(m)
            v = r.M[-1][:, j]
            summary["final_product_mM_range"][lab] = [float(v.min()), float(v[0]), float(v.max())]
    json.dump(summary, open(f"{args.out}/summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    # manylp's direct mode spawns worker processes, which re-import this module
    main()
