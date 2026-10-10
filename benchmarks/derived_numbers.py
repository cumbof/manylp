"""In-text numbers of the paper that are arithmetic on the result files, printed with their sources.

    python benchmarks/derived_numbers.py                       # reads results/
    python benchmarks/derived_numbers.py --root benchmarks/data
"""
import argparse
import glob
import json

import numpy as np


def _summary(root, name):
    return json.load(open(f"{root}/solvers/{name}.json"))["summary"]


def coherent_speedups(root):
    m = _summary(root, "manylp-cpu-fba@workload_coherent")["steady_lps_per_second"]
    print(f"manylp CPU, coherent workload: {m:,.0f} LP/s (steady)")
    for s, lab in (("highs-simplex-warm-p32", "HiGHS"), ("gurobi-dual-warm-p32", "Gurobi"),
                   ("xpress-dual-p32", "Xpress"), ("glpk-p32", "GLPK"), ("glop-p32", "GLOP"),
                   ("bunching-p8", "per-LP bunching, 8 processes")):
        b = _summary(root, f"{s}@workload_coherent")["steady_lps_per_second"]
        print(f"  vs {lab:30s} {b:9,.0f} LP/s  -> {m / b:5.1f}x")


def ensemble_scaling(root):
    gpu = json.load(open(f"{root}/dfba_v3/manylp-cuda_pfba-unique_E16384_T48.json"))
    cpu = json.load(open(f"{root}/dfba_v3/manylp-cpu_pfba-unique_E16384_T48.json"))
    st = gpu["solver_stats"]
    print(f"E=16,384: {gpu['n_lps']:,} LPs, {st['highs_solves']} simplex solves, GPU {gpu['lps_per_second']:,.0f} LP/s, "
          f"CPU {cpu['lps_per_second']:,.0f} LP/s -> {gpu['lps_per_second'] / cpu['lps_per_second']:.1f}x")


def muode_overhead(root):
    end = json.load(open(f"{root}/muode_v2/manylp-gpu-pfba-unique_E1024_T48.json"))["wall_seconds"]
    solo = json.load(open(f"{root}/dfba_v3/manylp-cuda_pfba-unique_E1024_T48.json"))["solve_seconds"]
    print(f"E=1,024: muODE end to end {end:.0f} s, manylp alone {solo:.0f} s -> "
          f"{100 * (1 - solo / end):.0f}% of the run time outside the LP solver")


def highs_agreement(root):
    """manylp vs an independent HiGHS implementation of the same pFBA-unique rule, through muODE."""
    for E in (1, 16):
        a = np.load(f"{root}/muode/manylp-cpu-pfba-unique_E{E}_T48.npz")
        b = np.load(f"{root}/muode/highs-pfba-unique_E{E}_T48.npz")
        print(f"E={E:2d}: max |biomass difference| {np.abs(a['X'] - b['X']).max():.2e} gDW/L, "
              f"max |metabolite difference| {np.abs(a['M'] - b['M']).max():.2e} mM")


def legacy_difference(root):
    """COBRApy + GLPK (arbitrary vertex) vs manylp pFBA-unique, final biomass per species."""
    leg = json.load(open(f"{root}/muode_v2/legacy-cobra-glpk-j1_E1_T48.json"))["final_biomass_member0"]
    man = json.load(open(f"{root}/muode_v2/manylp-cpu-pfba-unique_E1_T48.json"))["final_biomass_member0"]
    rel = {k: abs(man[k] / leg[k] - 1) for k in leg if leg[k] > 0}
    k = max(rel, key=rel.get)
    print(f"largest final-biomass difference: {100 * rel[k]:.2f}% ({k})")


def size_wins(root):
    rows = json.load(open(f"{root}/size/size.json"))
    sp = []
    for r in rows:
        best = max(r["manylp-gpu"]["steady_lps_per_second"], r["manylp-cpu"]["steady_lps_per_second"])
        sp.append((best / r["highs-warm-p32"]["steady_lps_per_second"], r["model"], r["mode"]))
    win = [s for s in sp if s[0] > 1]
    print(f"manylp faster than 32-process HiGHS on {len(win)} of {len(sp)} configurations, "
          f"{min(win)[0]:.1f}x to {max(win)[0]:.1f}x; slower on "
          + ", ".join(f"{m} ({mode})" for s, m, mode in sp if s <= 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results")
    a = ap.parse_args()
    for title, fn in (("Coherent workload (Table 1)", coherent_speedups),
                      ("Ensemble scaling (Figure 1A)", ensemble_scaling),
                      ("muODE overhead (Table 3 text)", muode_overhead),
                      ("Agreement with HiGHS under pFBA-unique", highs_agreement),
                      ("COBRApy + GLPK vs pFBA-unique", legacy_difference),
                      ("Model size (Figure 3A)", size_wins)):
        print(f"\n## {title}")
        try:
            fn(a.root)
        except FileNotFoundError as e:
            print(f"  missing input: {e.filename}")


if __name__ == "__main__":
    main()
