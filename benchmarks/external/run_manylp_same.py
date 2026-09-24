"""manylp on exactly the scenarios/ensembles given to the external tools (for side-by-side tables).

Runs in the ``manylp`` env.  For each case: regime-aware adaptive RK45 (``run_adaptive``)
and fixed-step Euler, E = 1 and the E-member log-normal ensemble (sigma 0.3, seed 0,
member 0 unperturbed -- the same draws as run_dfba_pkg.py / run_surfinfba.py), errors
against ``results/external/ref_*.npz`` with the metric of bench_diauxie.py.
Also reports the max deviation from the external tools' trajectories if present.
Writes ``results/external/manylp_<scenario>.json``.
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
OUT = os.path.join(ROOT, "results", "external")

from manylp.adaptive import run_adaptive  # noqa: E402
from manylp.dfba import Community, ManyLPAdapter, run_dfba  # noqa: E402


def load_comm(name):
    from make_references import diauxie_community

    if name == "diauxie":
        return diauxie_community()
    import cobra

    from manylp.fba import FBAModel

    sc = json.load(open(os.path.join(OUT, f"scenario_{name}.json")))
    fm = FBAModel.from_cobra(cobra.io.read_sbml_model(os.path.join(OUT, sc["sbml"])))
    fm.name = name
    c = Community(models=[fm], env_mets=sc["env_mets"], M0=np.array(sc["M0"]), influx=np.zeros(len(sc["env_mets"])),
                  max_uptake=np.full(len(sc["env_mets"]), np.inf), X0=np.array(sc["X0"]), vmax=sc["vmax"], km=sc["km"])
    assert list(c.env_mets) == sc["env_mets"]
    return c


def err(X, M, Xr, Mr, j, floor=1e-2):
    """X (T, E, S) / Xr (T, E, S): biomass summed over species (the external tools merge identical species)."""
    xs, xr = X.sum(-1), Xr.sum(-1)
    ex = float(np.max(np.abs(xs - xr) / np.maximum(np.abs(xr), 1e-9)))
    em = float(np.max(np.abs(M[..., j] - Mr[..., j]) / np.maximum(np.abs(Mr[..., j]), floor)))
    return ex, em


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="diauxie")
    ap.add_argument("--E", type=int, default=32)
    a = ap.parse_args()
    comm = load_comm(a.scenario)
    te = np.linspace(0, 12, 121)
    j = [comm.env_mets.index(m) for m in ("glc__D_e", "ac_e")]
    pert = np.random.default_rng(0).lognormal(0.0, 0.3, size=(a.E, len(comm.env_mets)))
    pert[0] = 1.0
    ref1 = np.load(os.path.join(OUT, f"ref_{a.scenario}.npz"))
    refE = os.path.join(OUT, f"ref_{a.scenario}_ens.npz")
    refE = np.load(refE) if os.path.exists(refE) else None
    rows, trajs = [], {}
    for E in (1, a.E):
        p = None if E == 1 else pert
        Xr = ref1["X"][:, None] if E == 1 else (refE["X"] if refE is not None else None)
        Mr = ref1["M"][:, None] if E == 1 else (refE["M"] if refE is not None else None)
        if Xr is not None and Xr.ndim == 2:
            Xr = Xr[..., None]
        cases = [(f"RK45 rtol={r:g}" + (" + regime events" if s else ""), "rk", r, s) for r in (1e-6, 1e-8) for s in (False, True)]
        cases += [(f"Euler dt={dt:g}", "euler", dt, None) for dt in (0.01, 0.001)]
        for label, kind, par, split in cases:
            ad = ManyLPAdapter("cpu", n_workers=8)
            t = time.perf_counter()
            if kind == "rk":
                try:
                    r = run_adaptive(comm, ad, E=E, t_end=12.0, rtol=par, atol=par * 1e-3, t_eval=te,
                                     split_regimes=split, perturb=p)
                except Exception as e:   # record, keep going
                    ad.close()
                    row = {"scheme": label, "E": E, "error": f"{type(e).__name__}: {e}"}
                    rows.append(row)
                    print(json.dumps(row), flush=True)
                    continue
                lps, extra = r.lps, {"rhs_evals": r.rhs_evals, "regime_switches": r.regime_switches}
            else:
                r = run_dfba(comm, ad, E=E, t_end=12.0, dt=par, record_every=int(round(0.1 / par)), perturb=p)
                lps, extra = r.n_lps, {}
            w = time.perf_counter() - t
            ad.close()
            row = {"scheme": label, "E": E, "wall_seconds": w, "wall_per_trajectory": w / E, "lps": lps, **extra}
            if Xr is not None:
                row["max_rel_err_biomass"], row["max_rel_err_glc_ac"] = err(r.X, r.M, Xr, Mr, j)
            rows.append(row)
            trajs[f"{label}|E{E}"] = (r.X, r.M)
            print(json.dumps(row), flush=True)
    # deviation from the external tools' trajectories (same member draws)
    cmp = {}
    for fn in sorted(os.listdir(OUT)):
        if fn.startswith(f"dfba_{a.scenario}_") and fn.endswith(".npz"):
            d = np.load(os.path.join(OUT, fn))
            Xd = d["X"]
            Md = d["M"]
            if Xd.ndim == 1:
                key = "RK45 rtol=1e-08|E1"
                Xd, Md = Xd[:, None, None], Md[:, None, :]
            else:
                key = f"RK45 rtol=1e-08|E{a.E}"
                Xd, Md = np.moveaxis(Xd, 0, 1)[..., None], np.moveaxis(Md, 0, 1)
            if key not in trajs:
                key = key.replace("RK45 rtol=1e-08", "RK45 rtol=1e-06")
            if key not in trajs:
                continue
            Xm, Mm = trajs[key]
            cmp.setdefault("_against", {})[fn] = key
            k, T = Xd.shape[1], Xd.shape[0]
            cmp[fn] = dict(zip(("max_rel_dev_biomass", "max_rel_dev_glc_ac"),
                               err(Xd, Md, Xm[:T, :k], Mm[:T, :k], j)))
    json.dump({"scenario": a.scenario, "rows": rows, "deviation_from_manylp_rk45_1e-8": cmp},
              open(os.path.join(OUT, f"manylp_{a.scenario}.json"), "w"), indent=1)
    print(json.dumps(cmp, indent=1))


if __name__ == "__main__":
    main()
