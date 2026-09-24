"""Regime-aware adaptive integration vs fixed-step Euler (12-species community, 48 h).

Reference: explicit Euler at dt = 0.001 h (48,000 steps; only affordable because
every LP is a certified cache hit).  Compared: the production scheme (Euler at
dt = 0.1 h with the CFL-style uptake cap), finer Euler steps, and adaptive RK45
with and without exact regime-switch splitting.  Error = max relative deviation
of biomass and of the major fermentation products over the 48 hourly samples.
"""
import json, os, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
from manylp.adaptive import run_adaptive
from manylp.dfba import ManyLPAdapter, load_gut_community, run_dfba

G = os.environ.get("GUT_DIR", "../muODE/examples/gut_western")
comm = load_gut_community(f"{G}/gems", f"{G}/gems/western_gut_modelseed.csv", abundance_tsv=f"{G}/abundance.tsv")
PRODUCTS = [m for m in ("cpd00029_e0", "cpd00211_e0", "cpd11640_e0") if m in comm.env_mets]
pj = [comm.env_mets.index(m) for m in PRODUCTS]
os.makedirs("results/adaptive", exist_ok=True)
DEV = os.environ.get("DEV", "cpu")


def euler(dt):
    ad = ManyLPAdapter(DEV, n_workers=32)
    t = time.perf_counter()
    every = int(round(1.0 / dt))
    r = run_dfba(comm, ad, E=1, t_end=48.0, dt=dt, record_every=every)
    wall = time.perf_counter() - t
    st = r.solver_stats
    ad.close()
    return {"X": r.X[:, 0], "M": r.M[:, 0], "lp_batches": int(st.get("B", 0) and len(ad.calls)),
            "lps": r.n_lps, "wall": wall, "solve": r.solve_seconds}


def adaptive(rtol, split):
    ad = ManyLPAdapter(DEV, n_workers=32)
    r = run_adaptive(comm, ad, E=1, t_end=48.0, rtol=rtol, atol=rtol * 1e-3, split_regimes=split)
    ad.close()
    return {"X": r.X[:, 0], "M": r.M[:, 0], "lp_batches": r.lp_batches, "lps": r.lps,
            "wall": r.wall_seconds, "solve": r.solve_seconds, "rhs_evals": r.rhs_evals,
            "switches": r.regime_switches}


def err(a, ref):
    ex = np.max(np.abs(a["X"] - ref["X"]) / np.maximum(np.abs(ref["X"]), 1e-9))
    em = np.max(np.abs(a["M"][:, pj] - ref["M"][:, pj]) / np.maximum(np.abs(ref["M"][:, pj]), 1e-3))
    return float(ex), float(em)


ref = euler(0.001)
print(f"reference Euler dt=0.001: {ref['lps']} LPs, {ref['wall']:.1f}s", flush=True)
rows = []
cases = [("Euler dt=0.1 (production)", lambda: euler(0.1)), ("Euler dt=0.05", lambda: euler(0.05)),
         ("Euler dt=0.01", lambda: euler(0.01)),
         ("RK45 rtol=1e-4", lambda: adaptive(1e-4, False)), ("RK45 rtol=1e-6", lambda: adaptive(1e-6, False)),
         ("RK45 rtol=1e-4 + regime events", lambda: adaptive(1e-4, True)),
         ("RK45 rtol=1e-6 + regime events", lambda: adaptive(1e-6, True))]
for label, fn in cases:
    r = fn()
    ex, em = err(r, ref)
    row = {"scheme": label, "lps": r["lps"], "wall_seconds": r["wall"], "solve_seconds": r["solve"],
           "rhs_evals": r.get("rhs_evals"), "regime_switches": r.get("switches"),
           "max_rel_err_biomass": ex, "max_rel_err_products": em}
    rows.append(row)
    print(json.dumps(row), flush=True)
json.dump(rows, open(f"results/adaptive/adaptive_{DEV}.json", "w"), indent=1)
