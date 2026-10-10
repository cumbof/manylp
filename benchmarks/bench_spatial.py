"""COMETS-style spatial dFBA in muODE: one LP per species per grid cell per step.

12 gapseq GEMs on an n x n grid (uniform inoculum, western diet, diffusion),
muODE's historical per-cell path (cobra + GLPK) vs the batched manylp backend,
where every (cell, species) LP of a step is one batched solve per species.
"""
import argparse, json, os, sys, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.environ.get("MUODE_DIR", "../muODE"))
from bench_muode import G, load_community
from muode.backends import make_backend
from muode.diet import load_diet
from muode.kinetics import KineticParameters
from muode.spatial import SpatialDynamicFBA

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--t-end", type=float, default=1.0)
ap.add_argument("--backends", default="legacy,manylp-cpu,manylp-gpu")
args = ap.parse_args()
comm = load_community(12)
diet = load_diet(f"{G}/gems/western_gut_modelseed.csv")
kin = KineticParameters(default_vmax=10.0, default_km=0.01)
os.makedirs("results/spatial", exist_ok=True)
out = {}
for name in args.backends.split(","):
    be = None if name == "legacy" else make_backend(name, flux_rule="vertex", n_workers=32)
    eng = SpatialDynamicFBA(nx=args.n, ny=args.n, dx=1.0, t_end=args.t_end, dt=0.1, backend=be)
    t = time.perf_counter()
    r = eng.run(comm, diet, kin)
    wall = time.perf_counter() - t
    lps = args.n * args.n * len(comm.organisms) * int(round(args.t_end / 0.1))
    tot = {k: float(v[-1].sum()) for k, v in r.biomass.items()}
    out[name] = {"wall_seconds": wall, "lps": lps, "lps_per_second": lps / wall, "final_total_biomass": tot}
    print(f"{name:12s} grid {args.n}x{args.n}  {lps} LPs  wall {wall:8.1f}s  -> {lps / wall:9.1f} LP/s", flush=True)
if "legacy" in out:
    for k in out:
        if k != "legacy":
            d = max(abs(out[k]["final_total_biomass"][s] - out["legacy"]["final_total_biomass"][s]) /
                    max(out["legacy"]["final_total_biomass"][s], 1e-12) for s in out[k]["final_total_biomass"])
            out[k]["max_rel_biomass_diff_vs_legacy"] = d
            print(f"{k}: speed-up {out['legacy']['wall_seconds'] / out[k]['wall_seconds']:.1f}x, "
                  f"max rel. total-biomass difference vs legacy {d:.2e}")
json.dump(out, open(f"results/spatial/spatial_{args.n}x{args.n}.json", "w"), indent=1)
