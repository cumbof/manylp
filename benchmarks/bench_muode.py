"""End-to-end benchmark through muODE's own engine (the number users see).

The 12-species ``gut_western`` community (gapseq GEMs as ``CobraOrganism``),
western diet, muODE's default kinetics, 48 h at dt = 0.1 h.  Compares muODE's
historical per-organism path (cobra + GLPK, optionally threaded) with the new
backends, for single trajectories and lockstep ensembles (``run_ensemble``)
whose members get log-normally perturbed diets.

    python benchmarks/bench_muode.py --backends legacy,manylp-gpu --E 1,64
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
MUODE = os.environ.get("MUODE_DIR", "../muODE")
sys.path.insert(0, MUODE)

import cobra  # noqa: E402
from muode import DynamicFBA  # noqa: E402
from muode.backends import LegacyBackend, make_backend  # noqa: E402
from muode.community import Community  # noqa: E402
from muode.diet import load_diet  # noqa: E402
from muode.kinetics import KineticParameters  # noqa: E402
from muode.organism import CobraOrganism  # noqa: E402

G = os.environ.get("GUT_DIR", "benchmarks/inputs/gut_western")


def load_community(n_species: int) -> Community:
    files = sorted(glob.glob(f"{G}/gems/*.xml.gz"))[:n_species]
    orgs = []
    for f in files:
        oid = os.path.basename(f).replace(".xml.gz", "")
        orgs.append(CobraOrganism(cobra.io.read_sbml_model(f), id=oid))
    ab = {}
    with open(f"{G}/abundance.tsv") as fh:
        for ln in fh:
            p = ln.strip().split("\t")
            if len(p) >= 2:
                try:
                    ab[p[0]] = float(p[1])
                except ValueError:
                    pass
    return Community(orgs, abundances={o.id: ab.get(o.id, 1.0) for o in orgs}, total_biomass=0.01)


def perturbed_diets(base, E: int, sigma: float):
    rng = np.random.default_rng(0)
    out = []
    for e in range(E):
        f = np.ones(len(base.concentrations)) if e == 0 else rng.lognormal(0, sigma, len(base.concentrations))
        conc = {m: c * fi for (m, c), fi in zip(base.concentrations.items(), f)}
        out.append(dataclasses.replace(base, concentrations=conc, name=f"{base.name}#{e}"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", default="legacy,manylp-cpu")
    ap.add_argument("--E", default="1")
    ap.add_argument("--species", type=int, default=12)
    ap.add_argument("--t-end", type=float, default=48.0)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--flux-rule", default="pfba-unique")
    ap.add_argument("--jobs", type=int, default=1, help="threads for the legacy backend")
    ap.add_argument("--sigma", type=float, default=0.3)
    ap.add_argument("--out", default="results/muode")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    t0 = time.perf_counter()
    comm = load_community(args.species)
    diet = load_diet(f"{G}/gems/western_gut_modelseed.csv")
    kin = KineticParameters(default_vmax=10.0, default_km=0.01)
    print(f"{len(comm.organisms)} species loaded in {time.perf_counter() - t0:.1f}s", flush=True)
    eng = DynamicFBA(t_end=args.t_end, dt=args.dt, record_fluxes=False)

    for E in [int(e) for e in args.E.split(",")]:
        diets = perturbed_diets(diet, E, args.sigma)
        for name in args.backends.split(","):
            if name == "legacy":
                be = LegacyBackend(n_jobs=args.jobs)
                label = f"legacy-cobra-glpk-j{args.jobs}"
            else:
                be = make_backend(name, flux_rule=args.flux_rule, n_workers=32)
                label = f"{name}-{args.flux_rule}"
            t = time.perf_counter()
            if E == 1:
                res = [eng.run(comm, diets[0], kin, backend=be)]
            else:
                res = eng.run_ensemble(comm, diets, kin, backend=be)
            wall = time.perf_counter() - t
            st = be.stats()
            be.close()
            n_lps = int(sum((r.biomass.values[:-1] > eng.min_biomass).sum() for r in res)) or None
            rec = {"backend": label, "E": E, "species": len(comm.organisms), "t_end": args.t_end,
                   "dt": args.dt, "wall_seconds": wall,
                   "stats": {k: v for k, v in st.items() if isinstance(v, (int, float))},
                   "final_biomass_member0": res[0].biomass.iloc[-1].to_dict()}
            tag = f"{label}_E{E}_T{args.t_end:g}"
            with open(f"{args.out}/{tag}.json", "w") as fh:
                json.dump(rec, fh, indent=1)
            np.savez_compressed(f"{args.out}/{tag}.npz",
                                X=np.stack([r.biomass.values for r in res]),
                                M=np.stack([r.metabolites.values for r in res]))
            lps = st.get("lps") or n_lps
            print(f"E={E:5d} {label:32s} wall {wall:9.2f}s  LPs~{lps}  "
                  f"simplex {st.get('highs_solves', '-')}", flush=True)


if __name__ == "__main__":
    main()
