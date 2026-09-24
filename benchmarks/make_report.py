"""Aggregate every benchmark into tables (Markdown/CSV/JSON) and figures (PNG).

    python benchmarks/make_report.py            # reads results/*, writes results/report/
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np

R = "results"
OUT = f"{R}/report"

# validated categorical palette (fixed slot order) and text inks
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

# ---------------------------------------------------------------------------
# capability matrix (documented behaviour of each solver family)
# ---------------------------------------------------------------------------
CAPS = {
    # family: (method, device, batched, warm start, exact vertex, certified optimality/infeasibility,
    #          unique flux, lexicographic/pFBA native, license)
    "manylp": ("certify-and-repair over cached optimal bases (exact)", "GPU / CPU", "yes (native)",
               "yes (basis per member + atlas)", "yes", "yes (own arithmetic, per LP)", "yes (certified)",
               "yes", "MIT"),
    "highs-simplex": ("dual simplex", "CPU", "no", "yes (basis)", "yes", "solver status", "no", "no", "MIT"),
    "highs-lex": ("HiGHS dual simplex applying manylp's pFBA-unique rule (3 stages)", "CPU", "no", "yes (basis)", "yes",
                  "solver status", "yes (by rule, not certified)", "via repeated solves", "MIT"),
    "highs-ipm": ("interior point + crossover", "CPU", "no", "no", "yes (crossover)", "solver status", "no", "no", "MIT"),
    "highs-pdlp": ("PDLP first-order", "CPU", "no", "no", "no (eps-optimal)", "solver status", "no", "no", "MIT"),
    "glpk": ("primal/dual simplex (cobra/optlang)", "CPU", "no", "yes (basis)", "yes", "solver status", "no", "no", "GPL"),
    "scipy": ("HiGHS via scipy.linprog", "CPU", "no", "no", "yes", "solver status", "no", "no", "BSD/MIT"),
    "glop": ("Google GLOP simplex", "CPU", "no", "yes (incremental)", "yes", "solver status", "no", "no", "Apache-2"),
    "ortools-pdlp": ("Google PDLP first-order", "CPU", "no", "no", "no (eps-optimal)", "solver status", "no", "no", "Apache-2"),
    "gurobi-dual": ("Gurobi dual simplex (commercial)", "CPU", "no", "yes (basis)", "yes", "solver status", "no", "no",
                    "commercial"),
    "gurobi-barrier": ("Gurobi barrier + crossover (commercial)", "CPU", "no", "no", "yes (crossover)", "solver status",
                       "no", "no", "commercial"),
    "xpress": ("FICO Xpress dual simplex (commercial)", "CPU", "no", "yes (basis)", "yes", "solver status", "no", "no",
               "commercial"),
    "osqp": ("ADMM (OSQP)", "CPU", "no", "yes (primal-dual)", "no (eps-optimal)", "solver status", "no", "no", "Apache-2"),
    "cuopt-pdlp": ("NVIDIA cuOpt PDLP", "GPU", "BatchSolve (host threads)", "yes (initial point)", "no (eps-optimal)",
                   "solver status", "no", "no", "Apache-2"),
    "cuopt-barrier": ("NVIDIA cuOpt barrier", "GPU", "no", "no", "no", "solver status", "no", "no", "Apache-2"),
    "cuopt-dual_simplex": ("NVIDIA cuOpt dual simplex (runs on the CPU)", "CPU", "no", "no", "yes", "solver status", "no", "no", "Apache-2"),
    "cuopt-concurrent": ("NVIDIA cuOpt concurrent", "GPU+CPU", "no", "no", "depends on winner", "solver status", "no",
                         "no", "Apache-2"),
    "mpax": ("MPAX r2HPDHG (JAX)", "GPU", "yes (vmap)", "yes (optional)", "no (eps-optimal)", "solver status", "no",
             "no", "MIT"),
    "ourpdhg": ("restarted PDHG, cuPDLP-style (JAX)", "GPU", "yes", "yes", "no (eps-optimal)", "solver status", "no",
                "no", "MIT"),
}


def family(solver: str) -> str:
    for k in sorted(CAPS, key=len, reverse=True):
        if solver.startswith(k):
            return k
    return solver.split("-")[0]


def fmt(x, kind="g"):
    if x is None:
        return "–"
    if isinstance(x, float) and not np.isfinite(x):
        return "–"
    if kind == "sci":
        return "0" if x == 0 else f"{x:.1e}"
    if kind == "int":
        return f"{int(round(x)):,}"
    if kind == "pct":
        return f"{100 * x:.1f}%"
    if kind == "rate":
        return f"{x:,.0f}" if x >= 10 else f"{x:.2f}"
    return f"{x:.3g}"


# ---------------------------------------------------------------------------
# 1. head-to-head solver comparison on the recorded workload
# ---------------------------------------------------------------------------

def comparison_table():
    rows = []
    for f in sorted(glob.glob(f"{R}/solvers/*.json")):
        if f.endswith("reference.json"):
            continue
        d = json.load(open(f))["summary"]
        fam = family(d["solver"])
        cap = CAPS.get(fam, ("?",) * 9)
        rows.append({**d, "family": fam, "method": cap[0], "dev": cap[1], "batched_cap": cap[2],
                     "warm_cap": cap[3], "exact": cap[4], "certificate": cap[5], "unique": cap[6],
                     "lex": cap[7], "license": cap[8]})
    rows.sort(key=lambda r: -r["lps_per_second"])
    best_cpu = max([r["lps_per_second"] for r in rows if r["family"] != "manylp" and r["success_rate"] > 0.99
                    and (r.get("obj_rel_err_max") or 1) < 1e-6] or [np.nan])
    hdr = ("| Solver | Method | Device | Batched | Warm start | Exact | Certified | Unique flux | pFBA/lex native | "
           "LP/s | Speed-up vs best exact baseline | Solved (optimal) | Obj. rel. err (max) | Primal infeas. (max) | "
           "Exchange-flux dev. from canonical (median / max) | Workload done | GPU mem (MiB) | License |")
    lines = [hdr, "|" + "|".join(["---"] * (hdr.count("|") - 1)) + "|"]
    for r in rows:
        sp = r["lps_per_second"] / best_cpu if np.isfinite(best_cpu) else np.nan
        lines.append(
            f"| {r['name']} | {r['method']} | {r['dev']} | {r['batched_cap']} | {r['warm_cap']} | {r['exact']} | "
            f"{r['certificate']} | {r['unique']} | {r['lex']} | {fmt(r['lps_per_second'], 'rate')} | "
            f"{fmt(sp) + 'x' if np.isfinite(sp) else '–'} | {fmt(r['success_rate'], 'pct')} | "
            f"{fmt(r.get('obj_rel_err_max'), 'sci')} | {fmt(r.get('infeas_max'), 'sci')} | "
            f"{fmt(r.get('exchange_dev_median'), 'sci')} / {fmt(r.get('exchange_dev_max'), 'sci')} | "
            f"{fmt(r['lps'] / max(r['total_lps'], 1), 'pct')} | {r.get('gpu_mem_peak_mib') or '–'} | {r['license']} |")
    return rows, "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. dFBA scaling (12-species community, 48 h)
# ---------------------------------------------------------------------------

def dfba_rows(dirs=("dfba", "dfba_v2", "dfba_v3")):
    out = {}
    for d in dirs:
        for f in glob.glob(f"{R}/{d}/*.json"):
            r = json.load(open(f))
            key = (r["solver"], r["mode"], r["E"])
            out[key] = r          # later directories override earlier ones
    return out


def dfba_table(rows):
    lines = ["| Solver | Flux rule | E | LPs | Solve time (s) | LP/s | Simplex solves | % LPs needing simplex |",
             "|---|---|---|---|---|---|---|---|"]
    for (s, mode, E), r in sorted(rows.items(), key=lambda kv: (kv[0][1], kv[0][2], -kv[1]["lps_per_second"])):
        st = r.get("solver_stats", {})
        hs = st.get("highs_solves")
        lines.append(f"| {s} | {mode} | {E} | {r['n_lps']:,} | {r['solve_seconds']:.1f} | "
                     f"{fmt(r['lps_per_second'], 'rate')} | {fmt(hs, 'int') if hs is not None else '–'} | "
                     f"{fmt(hs / r['n_lps'], 'pct') if hs is not None else '–'} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2)


def fig_scaling(rows):
    """(a) community dFBA throughput vs ensemble size; (b) certification throughput vs batch size."""
    import matplotlib.pyplot as plt

    series = [("manylp-cuda", "manylp (GPU)", C[0]), ("manylp-cpu", "manylp (CPU)", C[1]),
              ("highs-warm-p32", "HiGHS warm, 32 processes", C[2]),
              ("manylp-cuda-perlp", "manylp GPU, one LP per call", C[3]), ("highs-warm-x1", "HiGHS warm, 1 process", C[4])]
    tp = f"{R}/throughput/throughput.json"
    ncol = 2 if os.path.exists(tp) else 1
    fig, axes = plt.subplots(1, ncol, figsize=(4.9 * ncol, 3.9), dpi=200, squeeze=False)
    ax = axes[0, 0]
    for key, label, col in series:
        pts = sorted((E, r["lps_per_second"]) for (s, m, E), r in rows.items() if s == key and m == "pfba-unique")
        if not pts:
            continue
        x, y = zip(*pts)
        ax.plot(x, y, "-o", color=col, lw=2, ms=5, label=label)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    _style(ax, "(a) Community dFBA, 12 GEMs, 48 h, pFBA-unique", "ensemble size E", "LPs solved per second")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    if ncol == 2:
        ax = axes[0, 1]
        d = json.load(open(tp))
        for key, label, col in (("gpu-device-resident", "GPU, device-resident I/O", C[0]),
                                ("gpu-host-io", "GPU, host I/O", C[6]),
                                ("cpu-fused", "CPU, fused Numba kernels", C[1]), ("cpu-numpy", "CPU, NumPy", C[3])):
            pts = sorted((r["B"], r["lps_per_second"]) for r in d if r["variant"] == key)
            x, y = zip(*pts)
            ax.plot(x, y, "-o", color=col, lw=2, ms=4, label=label)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        _style(ax, "(b) Certification only, one GEM (n = 1,980)", "batch size B", "LPs certified per second")
        ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_scaling.png")
    plt.close(fig)


def fig_envelope():
    import matplotlib.pyplot as plt

    p = f"{R}/alt_optima/envelope.npz"
    if not os.path.exists(p):
        return
    d = np.load(p, allow_pickle=True)
    t, M, pols, env = d["times"], d["M"], list(d["policies"]), list(d["env_mets"])
    mets = [("cpd00211_e0", "butyrate"), ("cpd00029_e0", "acetate")]
    others = {}
    for s, lab in (("cobra-glpk", "GLPK (cobra)"), ("highs-warm-x1", "HiGHS"), ("scipy-linprog", "scipy linprog")):
        f = glob.glob(f"{R}/dfba*/{s}_fba_E1_T48.npz")
        if f:
            others[lab] = np.load(f[0])
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), dpi=160)
    for ax, (mid, lab) in zip(axes, mets):
        j = env.index(mid)
        Y = M[:, :, j]
        ax.fill_between(t, Y.min(axis=1), Y.max(axis=1), color=C[0], alpha=0.18, lw=0,
                        label=f"envelope over {len(pols)} optimal-flux policies")
        ax.plot(t, Y[:, 0], color=C[0], lw=2, label="canonical (pFBA-unique)")
        for (olab, o), col in zip(others.items(), (C[1], C[2], C[3])):
            ot = o["times"]
            ax.plot(ot, o["M"][:, 0, j], "--", color=col, lw=1.6, label=f"{olab}, plain FBA")
        _style(ax, f"{lab}: same model, same diet", "time (h)", "concentration (mM)")
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_envelope.png")
    plt.close(fig)


def _coherent_rows():
    rows = []
    for f in glob.glob(f"{R}/solvers/*@workload_coherent.json"):
        d = json.load(open(f))["summary"]
        d["family"] = family(d["solver"])
        rows.append(d)
    return rows


def fig_accuracy(_rows=None):
    """Speed vs accuracy on the coherent dFBA workload (steady-state LP/s vs max objective error)."""
    import matplotlib.pyplot as plt

    rows = _coherent_rows()
    if not rows:
        return
    floor = 1e-13          # an exact 0 is drawn on the floor line
    groups = {"manylp": ("manylp (certified)", C[0]), "exact": ("exact CPU/GPU solvers", C[1]),
              "fo": ("first-order GPU/CPU solvers", C[2])}
    # short labels and hand-placed offsets (points) so the dense exact-solver cluster stays readable
    LAB = {"manylp-cpu-fba": ("manylp FBA (CPU \u25cf, GPU \u25cb)", (0, 9), "center"),
           "manylp-cpu-pfba-unique": ("manylp pFBA-unique (CPU, GPU)", (0, 10), "center"),
           "manylp-gpu-perlp-fba": ("manylp GPU, one LP per call", (0, 10), "center"),
           "highs-simplex-warm-p32": ("HiGHS", (7, -1), "left"),
           "highs-lex-pfba-unique-p32": ("HiGHS pFBA-unique", (3, -13), "left"),
           "highs-ipm-cold-p32": ("HiGHS IPM", (-7, -1), "right"),
           "scipy-p32": ("SciPy", (-3, -13), "right"),
           "glpk-p32": ("GLPK", (7, -1), "left"), "gurobi-dual-warm-p32": ("Gurobi / Xpress", (8, 0), "left"),
           "glop-p32": ("GLOP", (7, -1), "left"), "cuopt-concurrent-e1e-6": ("cuOpt concurrent", (7, -1), "left"),
           "cuopt-pdlp-batch-e1e-6": ("cuOpt PDLP (\u03b5 = 10\u207b\u2076)", (7, -1), "left"),
           "cuopt-pdlp-xover-e1e-6": ("cuOpt PDLP + crossover", (7, -1), "left"),
           "cuopt-barrier-xover-e1e-6": ("cuOpt barrier+crossover (0.01% solved)", (7, -1), "left"),
           "mpax-e1e-6": ("MPAX (12% solved)", (-7, -1), "right"),
           "ourpdhg-e1e-6": ("batched PDHG (0% solved)", (0, -13), "center")}
    fig, ax = plt.subplots(figsize=(6.6, 4.4), dpi=200)
    for r in rows:
        err = r.get("obj_rel_err_max")
        x = r.get("steady_lps_per_second") or r["lps_per_second"]
        if err is None or not x:
            continue
        exact = CAPS.get(r["family"], ("", "", "", "", "no"))[4]
        g = "manylp" if r["family"] == "manylp" else ("fo" if str(exact).startswith("no") else "exact")
        gpu = r["device"] != "cpu"
        y = max(err, floor)
        ax.scatter(x, y, s=52, marker="o", facecolor=groups[g][1] if not gpu else "white",
                   edgecolor=groups[g][1], lw=1.8, zorder=3)
        if r["solver"] in LAB:
            text, off, ha = LAB[r["solver"]]
            ax.annotate(text, (x, y), xytext=off, textcoords="offset points", fontsize=6.5, color=INK2, ha=ha,
                        va="center")
    ax.axhline(floor, color=GRID, lw=1)
    ax.annotate("exact 0", (0.01, floor), xycoords=("axes fraction", "data"), xytext=(0, 3),
                textcoords="offset points", fontsize=6.5, color=INK2)
    for g, (lab, col) in groups.items():
        ax.scatter([], [], color=col, label=lab)
    ax.scatter([], [], facecolor="white", edgecolor=INK2, label="open marker = GPU")
    ax.set_xscale("log")
    ax.set_yscale("log")
    _style(ax, "Speed vs accuracy, coherent dFBA workload (185,088 LPs)", "steady-state LPs per second",
           "max relative objective error")
    ax.legend(frameon=False, fontsize=7.5, loc="center right", bbox_to_anchor=(1, 0.64))
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_accuracy.png")
    plt.close(fig)


def _netlib_rows():
    p = f"{R}/netlib/netlib_B1024.json"
    if not os.path.exists(p):
        return []
    nl = [r for r in json.load(open(p)) if r["m"] > 0 and r["n"] > 0]
    patch = {r["problem"]: r for f in glob.glob(f"{R}/netlib_patch/*.json") for r in json.load(open(f))}
    return [patch.get(r["problem"], r) for r in nl]


def fig_netlib():
    """Speed-up over 32-process HiGHS vs the number of distinct optimal bases among the 1,024 scenarios."""
    import matplotlib.pyplot as plt

    nl = _netlib_rows()
    if not nl:
        return
    fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=200)
    for r in nl:
        reg = max(r["manylp-cpu"]["distinct_bases"], 1)
        bm = max(r["manylp-cpu"]["lps_per_second"], r["manylp-gpu"]["lps_per_second"])
        bh = max(r[k]["lps_per_second"] for k in r if k.startswith("highs-"))
        sp = bm / bh
        col = C[0] if sp >= 1 else C[1]
        ax.scatter(reg, sp, s=40, color=col, edgecolor="white", lw=1.2, zorder=3)
        if r["problem"] in ("kb2", "recipe", "afiro", "agg", "scsd8", "bnl1", "share1b", "agg2"):
            off, ha = ((-6, -2), "right") if r["problem"] == "share1b" else ((5, 2), "left")
            ax.annotate(r["problem"], (reg, sp), xytext=off, textcoords="offset points", fontsize=6.5, color=INK2,
                        ha=ha)
    ax.axhline(1, color=INK2, lw=1, ls="--")
    ax.axvline(512, color=INK2, lw=0.8, ls=":")
    ax.annotate("pool cap (512)", (512, 1), xycoords=("data", "axes fraction"), xytext=(-4, -10),
                textcoords="offset points", fontsize=6.5, color=INK2, ha="right")
    ax.scatter([], [], color=C[0], label="manylp faster")
    ax.scatter([], [], color=C[1], label="HiGHS faster")
    ax.set_xscale("log")
    ax.set_yscale("log")
    _style(ax, "Netlib under RHS uncertainty (1,024 scenarios per problem)",
           "distinct optimal bases among the 1,024 scenarios", "speed-up of manylp over HiGHS (32 processes)")
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_netlib.png")
    plt.close(fig)


def fig_size():
    """Speed-up over 32-process HiGHS for genome-scale models of increasing size, against distinct bases."""
    import matplotlib.pyplot as plt

    p = f"{R}/size/size.json"
    if not os.path.exists(p):
        return
    rows = json.load(open(p))
    short = {"e_coli_core": "e_coli_core", "iYO844": "iYO844", "iMM904": "iMM904", "Bl_obeum (gapseq)": "B. obeum",
             "B_thetaiotaomicron (gapseq)": "B. theta", "iJO1366": "iJO1366", "iML1515": "iML1515",
             "Recon3D": "Recon3D"}
    # hand-placed label offsets (points) for the crowded pFBA-unique cluster
    OFF = {("iJO1366", "pfba-unique"): ((-6, 5), "right"), ("iML1515", "pfba-unique"): ((-6, -5), "right"),
           ("iMM904", "pfba-unique"): ((6, 0), "left"), ("e_coli_core", "pfba-unique"): ((-6, 0), "right")}
    fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=200)
    for mode, col, mk, lab in (("fba", C[1], "s", "plain FBA"), ("pfba-unique", C[0], "o", "pFBA-unique")):
        for r in rows:
            if r["mode"] != mode:
                continue
            best = max(r["manylp-gpu"]["steady_lps_per_second"], r["manylp-cpu"]["steady_lps_per_second"])
            sp = best / r["highs-warm-p32"]["steady_lps_per_second"]
            reg = r["manylp-cpu"]["regions"] or 1
            ax.scatter(reg, sp, s=46, color=col, marker=mk, edgecolor="white", lw=1.2, zorder=3)
            off, ha = OFF.get((r["model"], mode), ((5, 2), "left"))
            if reg >= 512:
                off, ha = (-6, 0), "right"
            ax.annotate(f"{short.get(r['model'], r['model'])} ({r['n']:,})", (reg, sp), xytext=off,
                        textcoords="offset points", fontsize=6, color=INK2, ha=ha, va="center")
        ax.scatter([], [], color=col, marker=mk, label=lab)
    ax.axhline(1, color=INK2, lw=1, ls="--")
    ax.axvline(512, color=INK2, lw=0.8, ls=":")
    ax.annotate("pool cap (512)", (512, 1), xycoords=("data", "axes fraction"), xytext=(-4, -10),
                textcoords="offset points", fontsize=6.5, color=INK2, ha="right")
    ax.set_xscale("log")
    ax.set_yscale("log")
    _style(ax, "Genome-scale models, 1,024 members (reactions in brackets)",
           "distinct optimal bases in the batch", "speed-up of manylp over HiGHS (32 processes)")
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_size.png")
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    comp_rows, comp_md = comparison_table()
    drows = dfba_rows()
    parts = ["# manylp benchmark report\n", "## Head-to-head comparison (recorded dFBA workload)\n", comp_md,
             "\n\n## Community dFBA scaling\n", dfba_table(drows)]
    for name in ("coldstart/coldstart_cpu.json", "coldstart/coldstart_cuda.json"):
        if os.path.exists(f"{R}/{name}"):
            rows = json.load(open(f"{R}/{name}"))
            parts.append(f"\n\n## Cold-start ablation ({name})\n")
            parts.append("| E | variant | solve (s) | simplex solves | repair (s) | atlas bases |\n|---|---|---|---|---|---|")
            for r in rows:
                parts.append(f"| {r['E']} | {r['variant']} | {r['solve_seconds']:.2f} | {r['simplex_solves']} | "
                             f"{r['repair_seconds']:.2f} | {r['atlas_bases']} |")
    if os.path.exists(f"{R}/netlib/netlib_B1024.json"):
        nl = _netlib_rows()   # skips failed downloads, applies re-run patches
        parts.append("\n\n## Netlib under RHS uncertainty (B = 1024 scenarios)\n")
        parts.append("| problem | m | n | regions | infeasible | manylp LP/s | HiGHS x32 LP/s (best) | speed-up | "
                     "status agree | max obj err |\n|---|---|---|---|---|---|---|---|---|---|")
        for r in nl:
            hk = [k for k in r if k.startswith("highs-")]
            bh = max(r[k]["lps_per_second"] for k in hk)
            bm = max(r["manylp-cpu"]["lps_per_second"], r["manylp-gpu"]["lps_per_second"])
            parts.append(f"| {r['problem']} | {r['m']} | {r['n']} | {r['manylp-cpu']['distinct_bases']} | "
                         f"{r['manylp-cpu']['infeasible']} | {bm:,.0f} | {bh:,.0f} | {bm / bh:.1f}x | "
                         f"{r['status_agreement']:.3f} | {fmt(r['obj_rel_err_max'], 'sci')} |")
    if os.path.exists(f"{R}/alt_optima/summary.json"):
        s = json.load(open(f"{R}/alt_optima/summary.json"))
        parts.append("\n\n## Alternative-optima envelope (same model, same diet)\n")
        parts.append("| product | min (mM) | canonical | max (mM) |\n|---|---|---|---|")
        for k, v in s["final_product_mM_range"].items():
            parts.append(f"| {k} | {v[0]:.3f} | {v[1]:.3f} | {v[2]:.3f} |")
    md = "\n".join(parts) + "\n"
    open(f"{OUT}/report.md", "w").write(md)
    json.dump({"comparison": comp_rows}, open(f"{OUT}/comparison.json", "w"), indent=1, default=str)
    fig_scaling(drows)
    fig_envelope()
    fig_accuracy()
    fig_netlib()
    fig_size()
    print(md[:6000])


if __name__ == "__main__":
    main()
