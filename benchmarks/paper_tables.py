"""Supplementary Tables S1-S3 and the Netlib summary statistics quoted in the paper.

Works on the raw results or on the exported summaries, which hold the same JSON files:

    python benchmarks/paper_tables.py                       # reads results/
    python benchmarks/paper_tables.py --root benchmarks/data
"""
import argparse
import glob
import json
import os

import numpy as np


def _sci(x):
    if x is None:
        return "–"
    if x == 0:
        return "0"
    return f"{x:.1e}".replace("e-0", "e-").replace("e+0", "e")


def _rate(x):
    if not x:
        return "–"
    return f"{x:,.0f}" if x >= 10 else f"{x:.2f}"


def _gm(a):
    return float(np.exp(np.mean(np.log(a))))


def solver_tables(root):
    """Tables S1 (coherent workload) and S2 (snapshot workload)."""
    out = []
    for title, tag in (("Table S1: coherent workload (185,088 LPs)", "@workload_coherent"),
                       ("Table S2: snapshot workload (23,808 LPs)", "")):
        out += [f"\n### {title}\n",
                "| Configuration | Device | LP/s (overall) | LP/s (steady) | Done | Solved | Status agreement | "
                "Max obj. error | Max infeasibility | Exchange dev. median / max |",
                "|" + " --- |" * 10]
        rows = []
        for p in glob.glob(f"{root}/solvers/*.json"):
            b = os.path.basename(p)[:-5]
            # classical bunching is reported separately (Table 6), not in S1/S2
            if "reference" in b or b.startswith("bunching") or (tag and not b.endswith(tag)) or (not tag and "@" in b):
                continue
            rows.append(json.load(open(p))["summary"])
        for s in sorted(rows, key=lambda s: -s["lps_per_second"]):
            out.append(
                f"| {s['solver'].replace('@workload_coherent', '')} | {s['device']} | {_rate(s['lps_per_second'])} | "
                f"{_rate(s.get('steady_lps_per_second'))} | {100 * s['lps'] / s['total_lps']:.1f}% | "
                f"{100 * s['success_rate']:.1f}% | {100 * (s.get('status_agreement') or 0):.1f}% | "
                f"{_sci(s.get('obj_rel_err_max'))} | {_sci(s.get('infeas_max'))} | "
                f"{_sci(s.get('exchange_dev_median'))} / {_sci(s.get('exchange_dev_max'))} |")
    return out


def _netlib(root):
    nl = [r for r in json.load(open(f"{root}/netlib/netlib_B1024.json")) if r["m"] > 0 and r["n"] > 0]
    patch = {r["problem"]: r for f in glob.glob(f"{root}/netlib_patch/*.json") for r in json.load(open(f))}
    return [patch.get(r["problem"], r) for r in nl]


def netlib_table(root):
    """Table S3 and the HiGHS comparison by number of distinct bases."""
    out = ["\n### Table S3: Netlib under right-hand-side uncertainty (1,024 scenarios per problem)\n",
           "| Problem | m | n | Distinct bases | Infeasible | Simplex solves | Certified | manylp CPU LP/s | "
           "manylp GPU LP/s | HiGHS ×32 LP/s (best) | Speed-up | Status agreement | Max obj. error | "
           "Errors HiGHS / manylp |", "|" + " --- |" * 14]
    few, many = [], []
    for r in sorted(_netlib(root), key=lambda r: r["problem"]):
        c, g = r["manylp-cpu"], r["manylp-gpu"]
        bh = max(r[k]["lps_per_second"] for k in r if k.startswith("highs-"))
        bm = max(c["lps_per_second"], g["lps_per_second"])
        sp, reg = bm / bh, c["distinct_bases"]
        (few if reg < 100 else many).append(sp)
        out.append(f"| {r['problem']} | {r['m']:,} | {r['n']:,} | {reg}{'+' if reg >= 512 else ''} | "
                   f"{c['infeasible']} | {c['simplex_solves']:,} | {100 * c['certified']:.1f}% | "
                   f"{c['lps_per_second']:,.0f} | {g['lps_per_second']:,.0f} | {bh:,.0f} | {sp:.2f}× | "
                   f"{100 * r['status_agreement']:.1f}% | {_sci(r['obj_rel_err_max'])} | "
                   f"{r['highs_errors']} / {r['manylp_errors']} |")
    few, many = np.array(few), np.array(many)
    out += ["\n**manylp vs HiGHS (best of cold/warm, 32 processes)**\n",
            f"- fewer than 100 distinct bases: {few.size} problems, manylp faster on {(few > 1).sum()}, "
            f"geometric mean {_gm(few):.1f}×, max {few.max():.0f}×",
            f"- 100 or more distinct bases: {many.size} problems, HiGHS faster on {(many < 1).sum()}, "
            f"geometric mean {_gm(many):.2f}×, HiGHS up to {1 / many.min():.1f}× faster"]
    return out


def netlib_bunching(root):
    """manylp (the faster of CPU and GPU, as for HiGHS) against the best classical bunching variant."""
    rows = {r["problem"]: r for r in json.load(open(f"{root}/netlib_bunching/netlib_bunching_B1024.json"))}
    pb = f"{root}/netlib_bunching/netlib_bunching_batch_B1024.json"
    batch = {r["problem"]: r for r in json.load(open(pb))} if os.path.exists(pb) else {}
    main = {r["problem"]: r for r in _netlib(root)}
    few, many = [], []
    for p, r in rows.items():
        reg = r["manylp-cpu"]["distinct_bases"]
        m = max(main[p]["manylp-cpu"]["lps_per_second"], main[p]["manylp-gpu"]["lps_per_second"])
        cand = [r[k]["lps_per_second"] for k in r if k.startswith("bunching")]
        cand += [batch[p][k]["lps_per_second"] for k in batch.get(p, {}) if k.startswith("bunching")]
        (few if reg < 100 else many).append(m / max(cand))
    few, many = np.array(few), np.array(many)
    return ["\n**manylp (faster of CPU and GPU) vs the best classical bunching variant "
            "(per-LP with 1 or 32 processes, or batched)**\n",
            f"- fewer than 100 distinct bases: {few.size} problems, manylp faster on {(few > 1).sum()}, "
            f"geometric mean {_gm(few):.2f}×",
            f"- 100 or more distinct bases: {many.size} problems, manylp faster on {(many > 1).sum()}, "
            f"geometric mean {_gm(many):.2f}× (bunching {1 / _gm(many):.1f}× faster)"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results")
    ap.add_argument("--out", default=None, help="also write the Markdown here")
    a = ap.parse_args()
    md = "\n".join(solver_tables(a.root) + netlib_table(a.root) + netlib_bunching(a.root)) + "\n"
    print(md)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        open(a.out, "w").write(md)


if __name__ == "__main__":
    main()
