"""Head-to-head solver comparison on the recorded dFBA workload.

Every solver replays the same LP instances (``results/workload.pkl``: 12 gut
GEMs x 31 snapshots x 64 ensemble members), in temporal order, as batches of
one species' ensemble members -- the unit a batched dFBA engine hands a solver.
The returned flux vectors are audited *independently* of what each solver
claims: objective error against the exact reference, primal infeasibility
``max(||S v||_inf, bound violation)``, status agreement, and exchange-flux
deviation from the canonical (pFBA-unique) solution.

    python benchmarks/bench_solvers.py --solver highs-simplex-warm-p32 --budget 900
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import resource
import subprocess
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import cmp_solvers as C  # noqa: E402


def make(spec: str):
    if spec.startswith("manylp-"):
        parts = spec.split("-")
        dev = "cuda" if parts[1] == "gpu" else "cpu"
        per_lp = "perlp" in parts
        mode = "pfba-unique" if spec.endswith("pfba-unique") else "fba"
        return C.ManyLP(dev, mode=mode, per_lp=per_lp)
    if spec.startswith("cuopt-"):
        # cuopt-<method>[-batch][-warm][-xover]-e<eps>   (eps may itself contain '-', e.g. e1e-4)
        head, _, eps_s = spec.rpartition("-e")
        eps = float(eps_s) if head else 1e-6
        p = (head or spec).split("-")
        return C.CuOpt(method=p[1], eps=eps, batch="batch" in p, warm="warm" in p, crossover="xover" in p)
    if spec.startswith("mpax"):
        return C.MPAX(eps=float(spec.split("-e")[-1]) if "-e" in spec else 1e-6)
    if spec.startswith("ourpdhg"):
        return C.OurPDHG(eps=float(spec.split("-e")[-1]) if "-e" in spec else 1e-6)
    # per-LP CPU solvers: <name>[-args]-p<N>
    base, procs = spec.rsplit("-p", 1)
    n = int(procs)
    if base.startswith("highs-lex"):
        return C.ProcPool("highs-lex", n=n, mode="pfba-unique")
    if base.startswith("highs-"):
        _, method, warm = base.split("-")
        return C.ProcPool("highs", n=n, method=method, warm=(warm == "warm"))
    if base.startswith("ortools-pdlp"):
        return C.ProcPool("ortools-pdlp", n=n, eps=1e-6)
    if base.startswith("gurobi-"):
        _, method, warm = base.split("-")
        return C.ProcPool("gurobi", n=n, method=method, warm=(warm == "warm"))
    if base.startswith("xpress-"):
        return C.ProcPool("xpress", n=n, method=base.split("-")[1])
    if base.startswith("osqp"):
        return C.ProcPool("osqp", n=n, eps=1e-6)
    return C.ProcPool(base, n=n)


class GPUMem(threading.Thread):
    """Peak device memory of the visible GPU, sampled with nvidia-smi."""

    def __init__(self):
        super().__init__(daemon=True)
        self.peak = 0
        self.stop = False
        vis = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        self.idx = vis

    def run(self):
        while not self.stop:
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "-i", self.idx,
                                      "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                     timeout=5).stdout
                self.peak = max(self.peak, int(out.strip().split()[0]))
            except Exception:
                pass
            time.sleep(0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", required=True)
    ap.add_argument("--workload", default="results/workload.pkl")
    ap.add_argument("--reference", default=None, help="default: <workload>_reference.npz or the snapshot reference")
    ap.add_argument("--budget", type=float, default=900.0, help="wall-clock cap (s)")
    ap.add_argument("--max-batches", type=int, default=0)
    ap.add_argument("--out", default="results/solvers")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    W = pickle.load(open(args.workload, "rb"))
    models = [m for _, m in W["models"]]
    snaps = W["snapshots"]
    if args.max_batches:
        snaps = snaps[: args.max_batches]
    if args.reference is None:
        cand = args.workload.replace(".pkl", "_reference.npz")
        args.reference = cand if os.path.exists(cand) else "results/solvers/reference.npz"
    ref = None
    if os.path.exists(args.reference):
        # materialise once: NpzFile re-reads (and unpickles) a key on every access
        with np.load(args.reference, allow_pickle=True) as z:
            ref = {k: z[k] for k in z.files}

    ad = make(args.solver)
    mem = GPUMem()
    mem.start()
    base_gpu = None
    t0 = time.perf_counter()
    ad.setup(models)
    t_setup = time.perf_counter() - t0

    rows = []
    t_solve = 0.0
    n_lps = 0
    first_batch_time = None
    V_ex = []
    for bi, (step, s, members, ex_lb) in enumerate(snaps):
        if t_solve > args.budget:
            break
        mdl = models[s]
        t = time.perf_counter()
        ok, obj, V = ad.solve(s, ex_lb, members)
        dt = time.perf_counter() - t
        if first_batch_time is None:
            first_batch_time = dt
        t_solve += dt
        n_lps += len(members)
        # independent audit of the returned fluxes
        lb = np.tile(mdl.lb, (len(members), 1))
        lb[:, mdl.exchanges] = ex_lb
        ub = mdl.ub[None, :]
        res_S = np.abs(mdl.S @ V.T).max(axis=0)
        res_b = np.maximum(np.maximum(lb - V, V - ub), 0).max(axis=1)
        infeas = np.maximum(res_S, res_b)
        rec = {"batch": bi, "step": step, "species": s, "B": len(members), "seconds": dt,
               "ok": ok.tolist(), "obj": obj.tolist(), "infeas": infeas.tolist()}
        if ref is not None:
            r_obj = ref["obj"][bi]
            r_ok = ref["ok"][bi]
            rec["obj_err"] = (np.abs(obj - r_obj) / np.maximum(1.0, np.abs(r_obj))).tolist()
            rec["status_agree"] = (ok == r_ok).tolist()
            vex = V[:, mdl.exchanges]
            rec["ex_dev"] = np.abs(vex - ref["vex"][bi]).max(axis=1).tolist()
        rows.append(rec)
    mem.stop = True
    wall = time.perf_counter() - t0
    ad.close()

    def cat(key):
        return np.concatenate([np.asarray(r[key], dtype=float) for r in rows]) if rows and key in rows[0] else np.array([])

    ok = cat("ok").astype(bool)
    obj_err = cat("obj_err")
    infeas = cat("infeas")
    agree = cat("status_agree")
    ex_dev = cat("ex_dev")
    ref_ok = np.concatenate([ref["ok"][r["batch"]] for r in rows]).astype(bool) if ref is not None and rows else ok
    summary = {
        "solver": args.solver, "name": ad.name, "device": ad.device, "batched": bool(ad.batch),
        "warm_start": bool(ad.warm),
        "lps": int(n_lps), "batches": len(rows), "total_lps": int(sum(len(x[2]) for x in snaps)),
        "completed_all": len(rows) == len(snaps),
        "setup_seconds": t_setup, "solve_seconds": t_solve, "wall_seconds": wall,
        "lps_per_second": n_lps / max(t_solve, 1e-12),
        # steady state: throughput over the second half of the processed batches
        "steady_lps_per_second": (sum(r["B"] for r in rows[len(rows) // 2:]) /
                                  max(sum(r["seconds"] for r in rows[len(rows) // 2:]), 1e-12)) if rows else None,
        "workload": os.path.basename(args.workload),
        "first_batch_seconds": first_batch_time,
        "success_rate": float(ok[ref_ok].mean()) if ok.size else 0.0,
        "status_agreement": float(agree.mean()) if agree.size else None,
        "obj_rel_err_median": float(np.median(obj_err[ref_ok])) if obj_err.size else None,
        "obj_rel_err_p99": float(np.quantile(obj_err[ref_ok], 0.99)) if obj_err.size else None,
        "obj_rel_err_max": float(obj_err[ref_ok].max()) if obj_err.size else None,
        "infeas_median": float(np.median(infeas[ref_ok])) if infeas.size else None,
        "infeas_max": float(infeas[ref_ok].max()) if infeas.size else None,
        "exchange_dev_median": float(np.median(ex_dev[ref_ok])) if ex_dev.size else None,
        "exchange_dev_max": float(ex_dev[ref_ok].max()) if ex_dev.size else None,
        "frac_exchange_dev_gt_1e-6": float((ex_dev[ref_ok] > 1e-6).mean()) if ex_dev.size else None,
        "gpu_mem_peak_mib": mem.peak,
        "host_maxrss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "children_maxrss_mib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024,
    }
    tag = args.solver + ("" if args.workload.endswith("results/workload.pkl") else "@" +
                         os.path.basename(args.workload).replace(".pkl", ""))
    with open(f"{args.out}/{tag}.json", "w") as fh:
        json.dump({"summary": summary, "batches": [{k: r[k] for k in ("batch", "step", "species", "B", "seconds")}
                                                    for r in rows]}, fh, indent=1)
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
