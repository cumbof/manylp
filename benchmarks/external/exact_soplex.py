"""Exact rational LP solving (SoPlex, GMP) of sampled FBA LPs from the coherent workload.

For each sampled LP (``max c^T v, S v = 0, lb <= v <= ub`` with the member's
exchange lower bounds, as ``lp_bounds`` in ``cmp_solvers.py``):

1. write it as free MPS (``min -c^T v``) with every number as its shortest
   round-trip decimal (``repr``, written without exponent), so the rational LP
   SoPlex reads is exactly the decimal LP we audit against;
2. solve it with SoPlex in exact rational mode
   (``--readmode=1 --solvemode=2 -f0 -o0``: iterative refinement / precision
   boosting to a rational optimum) and write the rational primal and dual solution;
3. independently verify the rational answer here in Python ``fractions``: exact
   primal feasibility (``S x = 0``, bounds) and exact KKT optimality (reduced costs
   ``d = -c - S^T y`` sign-consistent with the variables' positions at their bounds);
4. solve it again with SoPlex in plain floating-point mode (defaults,
   ``--solvemode=0``) for the cost comparison;
5. compare the exact objective with ``workload_coherent_reference.npz``.

Run with ``soplexenv`` (conda-forge ``soplex`` 8.1, GMP/MPFR/Boost) and the manylp
env's Python (the script only needs numpy):
    python benchmarks/external/exact_soplex.py --soplex $SOPLEXENV/bin/soplex
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction

import numpy as np

INF = 1e30


def dec(x) -> str:
    """Shortest round-trip decimal of a double, written without exponent (SoPlex's
    rational MPS reader rejects ``1e-05``-style numbers: WMPSRD07 malformed rational)."""
    from decimal import Decimal

    x = float(x)
    if x == 0:
        return "0"
    return format(Decimal(repr(x)), "f")


def write_mps(path, S, lb, ub, c):
    """Free MPS of ``min (-c)^T v, S v = 0, lb <= v <= ub``; returns the decimal strings used."""
    S = S.tocsc()
    m, n = S.shape
    cmin = [dec(-x) for x in c]
    L = [f"NAME fba\nROWS\n N obj\n"]
    L += [f" E R{i}\n" for i in range(m)]
    L.append("COLUMNS\n")
    for j in range(n):
        a, b = S.indptr[j], S.indptr[j + 1]
        if c[j] != 0:
            L.append(f" C{j} obj {cmin[j]}\n")
        for i, v in zip(S.indices[a:b], S.data[a:b]):
            L.append(f" C{j} R{i} {dec(v)}\n")
        if a == b and c[j] == 0:
            L.append(f" C{j} obj 0\n")          # declare the column
    L.append("RHS\nBOUNDS\n")
    for j in range(n):
        lo, up = float(lb[j]), float(ub[j])
        if lo == up:
            L.append(f" FX BND C{j} {dec(lo)}\n")
            continue
        if lo <= -INF:
            L.append(f" MI BND C{j}\n")
        else:
            L.append(f" LO BND C{j} {dec(lo)}\n")
        if up >= INF:
            L.append(f" PL BND C{j}\n")
        else:
            L.append(f" UP BND C{j} {dec(up)}\n")
    L.append("ENDATA\n")
    with open(path, "w") as f:
        f.writelines(L)


def read_sol(path):
    """SoPlex -x/-X/-y/-Y file: ``name value`` lines (value float or p/q); missing = 0."""
    out = {}
    with open(path) as f:
        for line in f:
            t = line.split()
            if len(t) == 2 and (t[0][0] in "CR") and t[0][1:].isdigit():
                out[t[0]] = t[1]
    return out


def parse_log(txt):
    st = re.search(r"SoPlex status\s*:\s*(.*)", txt)
    tm = re.search(r"Solving time \(sec\)\s*:\s*([0-9.eE+-]+)", txt)
    it = re.search(r"Iterations\s*:\s*(\d+)", txt)
    return (st.group(1).strip() if st else "?", float(tm.group(1)) if tm else float("nan"),
            int(it.group(1)) if it else -1)


def exact_check(S, lb, ub, c, xs, ys):
    """Exact primal feasibility + KKT optimality of rational (x, y) for min -c^T v."""
    S = S.tocsc()
    m, n = S.shape
    x = [Fraction(xs.get(f"C{j}", "0")) for j in range(n)]
    y = [Fraction(ys.get(f"R{i}", "0")) for i in range(m)]
    F = lambda v: Fraction(repr(float(v)))  # noqa: E731  (the decimal written to the MPS)
    rows = [Fraction(0)] * m
    cmin = [(-F(c[j]) if c[j] != 0 else Fraction(0)) for j in range(n)]
    sty = [Fraction(0)] * n                  # (S^T y)_j
    for j in range(n):
        a, b = S.indptr[j], S.indptr[j + 1]
        for i, v in zip(S.indices[a:b], S.data[a:b]):
            fv = F(v)
            rows[i] += fv * x[j]
            sty[j] += fv * y[i]
    row_ok = all(r == 0 for r in rows)
    bnd_ok = True
    kkt = {+1: True, -1: True}  # reduced cost d = cmin - sgn * S^T y (both dual sign conventions tried)
    for j in range(n):
        lo = None if lb[j] <= -INF else F(lb[j])
        up = None if ub[j] >= INF else F(ub[j])
        if (lo is not None and x[j] < lo) or (up is not None and x[j] > up):
            bnd_ok = False
        at_lo = lo is not None and x[j] == lo
        at_up = up is not None and x[j] == up
        if at_lo and at_up:
            continue
        for sgn in (+1, -1):
            dj = cmin[j] - sgn * sty[j]
            if (at_lo and dj < 0) or (at_up and dj > 0) or (not at_lo and not at_up and dj != 0):
                kkt[sgn] = False
    obj = sum((F(c[j]) * x[j] for j in range(n) if c[j] != 0), Fraction(0))
    return obj, row_ok, bnd_ok, (kkt[1] or kkt[-1]), (1 if kkt[1] else (-1 if kkt[-1] else 0))


def run_one(args):
    k, bi, j, S, lb, ub, c, soplex, tmp, tlim = args
    d = os.path.join(tmp, f"lp{k}")
    os.makedirs(d, exist_ok=True)
    mps = os.path.join(d, "lp.mps")
    write_mps(mps, S, lb, ub, c)
    rec = {"k": k, "batch": int(bi), "member_pos": int(j)}
    env = dict(os.environ, OMP_NUM_THREADS="1")
    # exact rational
    cmd = [soplex, "--readmode=1", "--solvemode=2", "-f0", "-o0", f"-t{tlim}", "-c",
           f"-X={d}/x.sol", f"-Y={d}/y.sol", mps]
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    rec["exact_wall_s"] = time.perf_counter() - t0
    st, stime, iters = parse_log(p.stdout)
    rec.update(exact_status=st, exact_solve_s=stime, exact_iters=iters,
               exact_primal_check="Primal solution feasible in original problem (max. violation = 0)" in p.stdout,
               exact_dual_check="Dual solution feasible in original problem (max. violation = 0)" in p.stdout)
    if "[optimal]" in st and os.path.exists(f"{d}/x.sol"):
        xs, ys = read_sol(f"{d}/x.sol"), read_sol(f"{d}/y.sol")
        obj, row_ok, bnd_ok, kkt_ok, sgn = exact_check(S, lb, ub, c, xs, ys)
        rec.update(obj_exact=f"{obj.numerator}/{obj.denominator}", obj_exact_float=float(obj),
                   obj_exact_den_digits=len(str(obj.denominator)),
                   py_row_exact=row_ok, py_bounds_exact=bnd_ok, py_kkt_exact=kkt_ok, py_dual_sign=sgn)
    # floating point (SoPlex defaults)
    cmd = [soplex, "--readmode=0", "--solvemode=0", f"-t{tlim}", f"-x={d}/xf.sol", mps]
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    rec["float_wall_s"] = time.perf_counter() - t0
    st, stime, iters = parse_log(p.stdout)
    rec.update(float_status=st, float_solve_s=stime, float_iters=iters)
    if "[optimal]" in st and os.path.exists(f"{d}/xf.sol"):
        xs = read_sol(f"{d}/xf.sol")
        x = np.zeros(len(c))
        for key, v in xs.items():
            if key[0] == "C":
                x[int(key[1:])] = float(v)
        rec["obj_float"] = float(c @ x)
        rec["float_row_resid"] = float(np.max(np.abs(S @ x)))
    shutil.rmtree(d, ignore_errors=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", default="results/workload_coherent.pkl")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--soplex", default="soplex", help="SoPlex 8.1 binary (conda-forge soplex)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--time-limit", type=float, default=3600)
    ap.add_argument("--out", default="results/external/soplex_exact.json")
    args = ap.parse_args()

    W = pickle.load(open(args.workload, "rb"))
    models = W["models"]
    snaps = W["snapshots"]
    with np.load(args.workload.replace(".pkl", "_reference.npz"), allow_pickle=True) as z:
        ref_obj, ref_ok = z["obj"], z["ok"]
    sizes = np.array([len(s[2]) for s in snaps])
    offs = np.concatenate([[0], np.cumsum(sizes)])
    rng = np.random.default_rng(args.seed)
    flat = np.sort(rng.choice(offs[-1], size=args.n, replace=False))
    jobs = []
    tmp = tempfile.mkdtemp(prefix="soplex_", dir=os.path.dirname(os.path.abspath(args.out)))
    for k, f in enumerate(flat):
        bi = int(np.searchsorted(offs, f, side="right") - 1)
        j = int(f - offs[bi])
        step, s, members, ex_lb = snaps[bi]
        mdl = models[s][1]
        lb = mdl.lb.copy()
        lb[mdl.exchanges] = ex_lb[j]
        jobs.append((k, bi, j, mdl.S, lb, mdl.ub.copy(), mdl.c.copy(), args.soplex, tmp, args.time_limit))
    meta = [{"species": models[snaps[bi][1]][0], "step": int(snaps[bi][0]), "member": int(snaps[bi][2][j]),
             "n": int(models[snaps[bi][1]][1].n), "m": int(models[snaps[bi][1]][1].m),
             "ref_ok": bool(ref_ok[bi][j]), "obj_ref": float(ref_obj[bi][j])}
            for (_, bi, j, *_r) in jobs]
    rows = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(args.workers) as ex:
        for rec in ex.map(run_one, jobs):
            rec.update(meta[rec["k"]])
            if "obj_exact_float" in rec:
                oe = Fraction(rec["obj_exact"])
                rec["relerr_ref_vs_exact"] = float(abs(Fraction(rec["obj_ref"]) - oe) / max(1, abs(oe)))
                rec["abserr_ref_vs_exact"] = float(abs(Fraction(rec["obj_ref"]) - oe))
                rec["relerr_obj_ref_vs_exact"] = float(abs(Fraction(rec["obj_ref"]) - oe) / abs(oe)) if oe else None
            if "obj_float" in rec and "obj_exact_float" in rec and rec["obj_exact_float"] != 0:
                rec["relerr_obj_float_vs_exact"] = abs(rec["obj_float"] - rec["obj_exact_float"]) / abs(
                    rec["obj_exact_float"])
            if "obj_float" in rec and "obj_exact_float" in rec:
                rec["relerr_float_vs_exact"] = abs(rec["obj_float"] - rec["obj_exact_float"]) / max(
                    1.0, abs(rec["obj_exact_float"]))
            rows.append(rec)
            print(f"[{len(rows)}/{len(jobs)}] {rec['species']} exact {rec['exact_status']} "
                  f"{rec['exact_wall_s']:.2f}s float {rec['float_wall_s']:.2f}s "
                  f"relerr {rec.get('relerr_ref_vs_exact')}", flush=True)
    elapsed = time.perf_counter() - t0
    shutil.rmtree(tmp, ignore_errors=True)

    def stats(key):
        v = np.array([r[key] for r in rows if key in r and np.isfinite(r[key])])
        return {"median": float(np.median(v)), "mean": float(v.mean()), "max": float(v.max()),
                "min": float(v.min()), "total": float(v.sum()), "lps_per_s_single_core": float(len(v) / v.sum())}

    ex_opt = [r for r in rows if "obj_exact_float" in r]
    exact_ok = ["[optimal]" in r["exact_status"] for r in rows]
    summ = {
        "soplex": subprocess.run([args.soplex, "--version"], capture_output=True, text=True).stdout.splitlines()[0],
        "exact_cmd": "soplex --readmode=1 --solvemode=2 -f0 -o0 -c -X= -Y= lp.mps",
        "float_cmd": "soplex --readmode=0 --solvemode=0 -x= lp.mps (default tolerances)",
        "workload": args.workload, "n": len(rows), "seed": args.seed, "workers": args.workers,
        "elapsed_s": elapsed, "species_counts": {s: sum(r["species"] == s for r in rows)
                                                 for s in sorted({r["species"] for r in rows})},
        "exact_wall_s": stats("exact_wall_s"), "exact_solve_s": stats("exact_solve_s"),
        "float_wall_s": stats("float_wall_s"), "float_solve_s": stats("float_solve_s"),
        "exact_status_counts": {s: sum(r["exact_status"] == s for r in rows) for s in {r["exact_status"] for r in rows}},
        "float_status_counts": {s: sum(r["float_status"] == s for r in rows) for s in {r["float_status"] for r in rows}},
        "status_agree_exact_vs_ref": int(sum(e == r["ref_ok"] for e, r in zip(exact_ok, rows))),
        "n_ref_ok": int(sum(r["ref_ok"] for r in rows)),
        "max_relerr_ref_vs_exact": max((r["relerr_ref_vs_exact"] for r in ex_opt if r["ref_ok"]), default=None),
        "median_relerr_ref_vs_exact": float(np.median([r["relerr_ref_vs_exact"] for r in ex_opt if r["ref_ok"]]))
        if ex_opt else None,
        "max_abserr_ref_vs_exact": max((r["abserr_ref_vs_exact"] for r in ex_opt if r["ref_ok"]), default=None),
        "max_relerr_obj_ref_vs_exact (|d|/|obj|)": max((r["relerr_obj_ref_vs_exact"] for r in ex_opt
                                                         if r["ref_ok"] and r["relerr_obj_ref_vs_exact"] is not None),
                                                        default=None),
        "median_relerr_obj_ref_vs_exact (|d|/|obj|)": float(np.median([r["relerr_obj_ref_vs_exact"] for r in ex_opt
                                                                       if r["ref_ok"] and r["relerr_obj_ref_vs_exact"]
                                                                       is not None])) if ex_opt else None,
        "max_relerr_float_vs_exact": max((r["relerr_float_vs_exact"] for r in ex_opt if "relerr_float_vs_exact" in r),
                                         default=None),
        "max_relerr_obj_float_vs_exact (|d|/|obj|)": max((r["relerr_obj_float_vs_exact"] for r in ex_opt
                                                           if "relerr_obj_float_vs_exact" in r), default=None),
        "float_status_agree_exact": int(sum(("[optimal]" in r["float_status"]) == e for e, r in zip(exact_ok, rows))),
        "py_exact_verified": int(sum(r.get("py_row_exact", False) and r.get("py_bounds_exact", False)
                                     and r.get("py_kkt_exact", False) for r in ex_opt)),
        "soplex_primal_dual_check_zero_violation": int(sum(r["exact_primal_check"] and r["exact_dual_check"]
                                                           for r in ex_opt)),
        "obj_exact_denominator_digits_max": max((r["obj_exact_den_digits"] for r in ex_opt), default=None),
    }
    json.dump({"summary": summ, "per_lp": rows}, open(args.out, "w"), indent=1)
    print(json.dumps(summ, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    sys.exit(main())
