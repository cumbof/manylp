"""Cold-start ablation: eager vs progressive repair, and a persistent basis atlas.

The atlas is built from one ensemble (diet perturbation seed A) and used on a
*different* ensemble (seed B), so it is not evaluated on the data it was built from.
"""
import argparse, json, os, shutil, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
from manylp.dfba import ManyLPAdapter, load_gut_community, run_dfba
G = os.environ.get("GUT_DIR", "../muODE/examples/gut_western")
ap = argparse.ArgumentParser()
ap.add_argument("--E", default="1,64")
ap.add_argument("--device", default="cuda")
ap.add_argument("--t-end", type=float, default=48.0)
ap.add_argument("--out", default="results/coldstart")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
comm = load_gut_community(f"{G}/gems", f"{G}/gems/western_gut_modelseed.csv", abundance_tsv=f"{G}/abundance.tsv")

def pert(E, seed):
    if E == 1 and seed == 1:
        return None
    p = np.random.default_rng(seed).lognormal(0, 0.3, size=(E, len(comm.env_mets)))
    return p

rows = []
for E in [int(e) for e in args.E.split(",")]:
    atlas = f"{args.out}/atlas_E{E}"
    shutil.rmtree(atlas, ignore_errors=True)
    # build the atlas on seed A (for E=1: a perturbed diet; the test uses the base diet)
    ad = ManyLPAdapter(args.device, n_workers=32, atlas_dir=atlas)
    run_dfba(comm, ad, E=E, t_end=args.t_end, dt=0.1, perturb=pert(E, 1000))
    n_saved = ad.save_atlas(); ad.close()
    for label, kw in [("eager", dict(progressive=False)), ("progressive", dict()),
                      ("progressive+atlas", dict(atlas_dir=atlas))]:
        ad = ManyLPAdapter(args.device, n_workers=32, **kw)
        t = time.perf_counter()
        r = run_dfba(comm, ad, E=E, t_end=args.t_end, dt=0.1, perturb=pert(E, 1))
        st = r.solver_stats
        rows.append({"E": E, "variant": label, "solve_seconds": r.solve_seconds, "lps": r.n_lps,
                     "simplex_solves": st.get("highs_solves"), "repair_seconds": st.get("repair_seconds"),
                     "certify_seconds": st.get("certify_seconds"), "atlas_bases": getattr(ad, "atlas_loaded", 0),
                     "atlas_saved": n_saved})
        ad.close()
        print(json.dumps(rows[-1]), flush=True)
json.dump(rows, open(f"{args.out}/coldstart_{args.device}.json", "w"), indent=1)
