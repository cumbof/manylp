"""manylp restricted to one core on the external-tool scenarios, for per-core comparisons.

The dfba package and surfinFBA run single-threaded, one trajectory after another.  This
runs manylp's RK45 cases of ``run_manylp_same.py`` with one Numba thread, one BLAS thread
and one repair worker (use ``taskset -c <core>`` as well), so that seconds per trajectory
can be compared core for core.  Writes ``results/external/manylp_<scenario>_1thread.json``.
"""
import os

for _v in ("MANYLP_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "..", "..", "results", "external")


def main():
    from run_manylp_same import err, load_comm

    from manylp.adaptive import run_adaptive
    from manylp.dfba import ManyLPAdapter

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
    rows = []
    for E in (1, a.E):
        Xr = ref1["X"][:, None] if E == 1 else (refE["X"] if refE is not None else None)
        Mr = ref1["M"][:, None] if E == 1 else (refE["M"] if refE is not None else None)
        if Xr is not None and Xr.ndim == 2:
            Xr = Xr[..., None]
        for rtol in (1e-6, 1e-8):
            ad = ManyLPAdapter("cpu", n_workers=1)
            t = time.perf_counter()
            r = run_adaptive(comm, ad, E=E, t_end=12.0, rtol=rtol, atol=rtol * 1e-3, t_eval=te,
                             perturb=None if E == 1 else pert)
            w = time.perf_counter() - t
            ad.close()
            row = {"scheme": f"RK45 rtol={rtol:g}", "E": E, "threads": 1, "wall_seconds": w,
                   "wall_per_trajectory": w / E, "lps": r.lps, "rhs_evals": r.rhs_evals}
            if Xr is not None:
                row["max_rel_err_biomass"], row["max_rel_err_glc_ac"] = err(r.X, r.M, Xr, Mr, j)
            rows.append(row)
            print(json.dumps(row), flush=True)
    json.dump({"scenario": a.scenario, "threads": 1, "rows": rows},
              open(os.path.join(OUT, f"manylp_{a.scenario}_1thread.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
