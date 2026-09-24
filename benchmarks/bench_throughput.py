"""Solver-level throughput vs batch size (one genome-scale group, warm pool).

Measures steady-state certification throughput -- the regime a dFBA or
ensemble workload spends almost all of its time in -- for the CPU paths (NumPy,
fused Numba) and the GPU paths (host inputs/outputs, device-resident), with
per-call latency, on B. thetaiotaomicron (m = 1645, n = 1980, 170 exchanges).
"""

import json
import os
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

import cobra  # noqa: E402

import manylp.basis as mb  # noqa: E402
from manylp import BatchLPSolver  # noqa: E402
from manylp.fba import FBAModel, compile_fba  # noqa: E402

G = os.environ.get("GUT_DIR", "../muODE/examples/gut_western")


def main(out="results/throughput"):
    os.makedirs(out, exist_ok=True)
    mdl = cobra.io.read_sbml_model(f"{G}/gems/B_thetaiotaomicron_VPI5482.xml.gz")
    fm = FBAModel.from_cobra(mdl)
    prob = compile_fba(fm, "pfba-unique")
    lb0 = np.maximum(np.where(fm.lb[fm.exchanges] < 0, fm.lb[fm.exchanges], -10.0), -10.0)
    rows = []
    variants = [("cpu-numpy", "cpu", False), ("cpu-fused", "cpu", False), ("gpu-host-io", "cuda", False),
                ("gpu-device-resident", "cuda", True)]
    for label, dev, resident in variants:
        mb.HOST_FUSED_MIN_BATCH = 10 ** 9 if label == "cpu-numpy" else 16
        s = BatchLPSolver(dev, n_workers=16)
        g = s.register_group(prob.lp, out_z=prob.out_z(fm.exchanges))
        for B in [1, 4, 16, 64, 256, 1024, 4096, 16384, 65536]:
            if label == "cpu-numpy" and B > 16384:
                continue
            rng = np.random.default_rng(0)
            ex_lb = lb0 * (0.8 + 0.001 * rng.random((B, lb0.size)))
            Lp, Up = prob.param_bounds(ex_lb)
            ws = s.solve_batch(g, Lp, Up).basis_id           # warm the pool
            if resident:
                import cupy as cp

                Lp, Up = cp.asarray(Lp), cp.asarray(Up)
            reps = max(3, min(50, int(2e5 // B)))
            for _ in range(2):
                s.solve_batch(g, Lp, Up, warm_start=ws, device_out=resident)
            if dev == "cuda":
                import cupy as cp

                cp.cuda.Device().synchronize()
            t = time.perf_counter()
            for _ in range(reps):
                sol = s.solve_batch(g, Lp, Up, warm_start=ws, device_out=resident)
            if dev == "cuda":
                cp.cuda.Device().synchronize()
            dt = (time.perf_counter() - t) / reps
            rows.append({"variant": label, "B": B, "seconds_per_call": dt, "lps_per_second": B / dt,
                         "distinct_bases": int(len(np.unique(ws)))})
            print(f"{label:20s} B={B:6d} {dt * 1e3:9.3f} ms/call {B / dt:12,.0f} LP/s", flush=True)
        s.close()
    json.dump(rows, open(f"{out}/throughput.json", "w"), indent=1)


if __name__ == "__main__":
    main()
