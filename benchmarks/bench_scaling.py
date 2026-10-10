"""Scaling muODE + manylp from tens to thousands of species in one community.

Species are drawn (fixed seed) from the 2,168 species-level gapseq reconstructions of the
Unified Human Gastrointestinal Genome catalogue v2 in the MICOM model database v1
(doi:10.5281/zenodo.7739096, ``uhgg201_gtdb207_species_1.qza``).  They use the ModelSEED
namespace of the paper's 12-species community, so the same Western diet, kinetics, initial
total biomass (0.01 gDW/L) and 48 h at dt = 0.1 h apply; initial abundances are log-normal.

For every community size N the script reports, separately, model loading (parallel JSON
parsing), compilation of each model into an LP group, and the simulation itself, plus the
simplex solves and distinct optimal bases per species.  The legacy backend (cobra + GLPK)
can be run on a shorter horizon (``--legacy-t-end``) to measure its per-step cost when the
full 48 h would take days.

    python benchmarks/bench_scaling.py --models <dir of *.json> --N 100,1000 \\
        --backends manylp-cpu --E 1
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import os
import resource
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
MUODE = os.environ.get("MUODE_DIR", "../muODE")
sys.path.insert(0, MUODE)
G = os.environ.get("GUT_DIR", "benchmarks/inputs/gut_western")


def _load_one(path):
    import cobra

    return os.path.basename(path)[:-5], cobra.io.load_json_model(path)


def load_models(paths, n_procs):
    """Parse the JSON models in parallel worker processes (cobra models pickle cleanly)."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(n_procs, mp_context=mp.get_context("spawn")) as ex:
        return list(ex.map(_load_one, paths, chunksize=4))


def perturbed_diets(base, E, sigma):
    rng = np.random.default_rng(0)
    out = []
    for e in range(E):
        f = np.ones(len(base.concentrations)) if e == 0 else rng.lognormal(0, sigma, len(base.concentrations))
        conc = {m: c * fi for (m, c), fi in zip(base.concentrations.items(), f)}
        out.append(dataclasses.replace(base, concentrations=conc, name=f"{base.name}#{e}"))
    return out


def gpu_mem_mib():
    try:
        import cupy as cp

        return cp.get_default_memory_pool().total_bytes() / 2**20
    except Exception:
        return None


def main():
    from muode import DynamicFBA
    from muode.backends import LegacyBackend, make_backend
    from muode.community import Community
    from muode.diet import load_diet
    from muode.kinetics import KineticParameters
    from muode.organism import CobraOrganism

    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="directory of COBRA JSON models")
    ap.add_argument("--N", default="10,100")
    ap.add_argument("--E", default="1")
    ap.add_argument("--backends", default="manylp-cpu")
    ap.add_argument("--t-end", type=float, default=48.0)
    ap.add_argument("--legacy-t-end", type=float, default=None,
                    help="shorter horizon for the legacy backend (per-step cost)")
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--sigma", type=float, default=0.3)
    ap.add_argument("--load-procs", type=int, default=32)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", default="results/scaling")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(a.models, "*.json")))
    order = np.random.default_rng(0).permutation(len(paths))
    diet = load_diet(f"{G}/gems/western_gut_modelseed.csv")
    kin = KineticParameters(default_vmax=10.0, default_km=0.01)

    for N in [int(n) for n in a.N.split(",")]:
        N = min(N, len(paths))
        sel = [paths[i] for i in sorted(order[:N])]
        t0 = time.perf_counter()
        loaded = load_models(sel, a.load_procs)
        load_s = time.perf_counter() - t0
        orgs = [CobraOrganism(m, id=oid) for oid, m in loaded]
        w = np.random.default_rng(1).lognormal(0.0, 1.0, N)
        comm = Community(orgs, abundances={o.id: float(x) for o, x in zip(orgs, w)}, total_biomass=0.01)
        print(f"N={N}: loaded in {load_s:.1f}s", flush=True)

        for E in [int(e) for e in a.E.split(",")]:
            diets = perturbed_diets(diet, E, a.sigma)
            for name in a.backends.split(","):
                legacy = name.startswith("legacy")
                t_end = a.legacy_t_end if (legacy and a.legacy_t_end) else a.t_end
                eng = DynamicFBA(t_end=t_end, dt=a.dt, record_fluxes=False)
                if legacy:
                    be, label, compile_s = LegacyBackend(n_jobs=1), "legacy-cobra-glpk-j1", 0.0
                else:
                    be = make_backend(name, flux_rule="pfba-unique", n_workers=a.workers)
                    label = f"{name}-pfba-unique"
                    t = time.perf_counter()
                    for o in comm.organisms:          # LP extraction + compilation per model
                        be._group(o, 0)
                    compile_s = time.perf_counter() - t
                t = time.perf_counter()
                res = ([eng.run(comm, diets[0], kin, backend=be)] if E == 1
                       else eng.run_ensemble(comm, diets, kin, backend=be))
                wall = time.perf_counter() - t
                st = be.stats()
                bases = [len(h.pool.bases) for _, _, h in getattr(be, "_groups", {}).values() if h is not None]
                mem = gpu_mem_mib()
                be.close()
                fin = np.stack([r.biomass.iloc[-1].values for r in res])
                rec = {"backend": label, "N": N, "E": E, "t_end": t_end, "dt": a.dt,
                       "steps": int(round(t_end / a.dt)) + 1,
                       "load_seconds": load_s, "compile_seconds": compile_s, "wall_seconds": wall,
                       "stats": {k: v for k, v in st.items() if isinstance(v, (int, float))},
                       "bases_per_species": {"median": float(np.median(bases)) if bases else None,
                                             "mean": float(np.mean(bases)) if bases else None,
                                             "max": int(max(bases)) if bases else None,
                                             "total": int(sum(bases))},
                       "species_growing": int((fin[0] > comm.total_biomass * 1e-12).sum()),
                       "species_increased": int((res[0].biomass.iloc[-1] > res[0].biomass.iloc[0] * 1.01).sum()),
                       "gpu_pool_mib": mem,
                       "max_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20}
                tag = f"{label}_N{N}_E{E}_T{t_end:g}"
                with open(f"{a.out}/{tag}.json", "w") as fh:
                    json.dump(rec, fh, indent=1)
                bm = res[0].biomass                 # member 0: biomass time course per species
                np.savez_compressed(f"{a.out}/{tag}.npz", species=np.array(list(bm.columns)),
                                    times=bm.index.values, X=bm.values)
                print(f"N={N:5d} E={E:4d} {label:28s} compile {compile_s:7.1f}s  wall {wall:9.1f}s  "
                      f"LPs {st.get('lps', '-')}  simplex {st.get('highs_solves', '-')}  "
                      f"bases/sp median {rec['bases_per_species']['median']}", flush=True)


if __name__ == "__main__":
    main()
