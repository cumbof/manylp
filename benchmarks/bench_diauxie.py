"""Regime switches matter: E. coli core diauxie (glucose exhaustion -> acetate).

Same comparison as bench_adaptive.py, on a scenario that crosses several
metabolic regimes, against a fine Euler reference (dt = 1e-4 h)."""
import json, os, time, warnings, sys
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, "tests")
from test_extras import _ecoli_community
from manylp.adaptive import run_adaptive
from manylp.dfba import ManyLPAdapter, run_dfba


def euler(dt):
    t = time.perf_counter()
    r = run_dfba(comm, ManyLPAdapter("cpu", n_workers=8), E=1, t_end=T, dt=dt, record_every=int(round(0.1 / dt)))
    return {"X": r.X[:, 0], "M": r.M[:, 0], "lps": r.n_lps, "wall": time.perf_counter() - t}


def adapt(rtol, split):
    r = run_adaptive(comm, ManyLPAdapter("cpu", n_workers=8), E=1, t_end=T, rtol=rtol, atol=rtol * 1e-3,
                     t_eval=te, split_regimes=split)
    return {"X": r.X[:, 0], "M": r.M[:, 0], "lps": r.lps, "wall": r.wall_seconds, "evals": r.rhs_evals,
            "switches": r.regime_switches}


def main():
    global comm, T, te, j
    comm = _ecoli_community()
    comm.M0[comm.env_mets.index("glc__D_e")] = 15.0
    comm.M0[comm.env_mets.index("o2_e")] = 1e3
    T = 12.0
    te = np.linspace(0, T, 121)
    j = [comm.env_mets.index(m) for m in ("glc__D_e", "ac_e")]
    ref = euler(1e-4)
    rows = []
    for label, fn in [("Euler dt=0.1", lambda: euler(0.1)), ("Euler dt=0.01", lambda: euler(0.01)),
                      ("RK45 1e-6", lambda: adapt(1e-6, False)), ("RK45 1e-6 + regime events", lambda: adapt(1e-6, True)),
                      ("RK45 1e-8", lambda: adapt(1e-8, False)), ("RK45 1e-8 + regime events", lambda: adapt(1e-8, True))]:
        r = fn()
        ex = float(np.max(np.abs(r["X"] - ref["X"]) / np.maximum(np.abs(ref["X"]), 1e-9)))
        em = float(np.max(np.abs(r["M"][:, j] - ref["M"][:, j]) / np.maximum(np.abs(ref["M"][:, j]), 1e-2)))
        row = {"scheme": label, "lps": r["lps"], "wall": r["wall"], "rhs_evals": r.get("evals"),
               "regime_switches": r.get("switches"), "max_rel_err_biomass": ex, "max_rel_err_glc_ac": em}
        rows.append(row)
        print(json.dumps(row), flush=True)
    os.makedirs("results/adaptive", exist_ok=True)
    json.dump(rows, open("results/adaptive/diauxie.json", "w"), indent=1)
    print("final glucose/acetate (ref):", ref["M"][-1, j], "biomass", ref["X"][-1])


if __name__ == "__main__":
    # manylp's direct mode spawns worker processes, which re-import this module
    main()
