"""The overview figure: from the dFBA workload to a 2,168-species gut community.

    python benchmarks/hero_figure.py          # reads results/*, writes results/report/paper/figure_overview.{png,pdf}

a  the workload and how manylp answers it (counts from the 16,384-member community run)
b  exact metabolic phase diagram of E. coli core over glucose and acetate (hero_phase.py)
c  speed against accuracy on the coherent genome-scale workload
d  alternative optima: acetate in the 12-species gut community under 49 equally optimal policies
e  the same policies shift community composition
f  scaling a gut community to 2,168 species (bench_scaling.py)
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

R = "results"
OUT = f"{R}/report/paper"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUE, ORANGE, GREEN = C[0], C[1], C[2]


def _style(ax, xlabel=None, ylabel=None, title=None):
    if title:
        ax.set_title(title, loc="left", fontsize=7, color=INK, pad=3)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=6.5, labelpad=2)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=6.5, labelpad=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
        ax.spines[s].set_linewidth(0.6)
    ax.tick_params(colors=INK2, labelsize=6, width=0.6, length=2.5)


def _label(fig, ax, letter, dx=-0.02, dy=0.01):
    bb = ax.get_position()
    fig.text(bb.x0 + dx, bb.y1 + dy, letter, fontsize=10, fontweight="bold", color=INK, ha="right", va="bottom")


# ---------------------------------------------------------------------------------------------
# a  workload schematic
# ---------------------------------------------------------------------------------------------

def panel_workload(ax):
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 30)
    ax.axis("off")
    st = json.load(open(glob.glob(f"{R}/dfba_v3/manylp-cuda_pfba-unique_E16384_T48.json")[0]))["solver_stats"]
    total = st["B"]

    # stacked "pages": members x (species rows, time columns)
    for k in range(5):
        x0, y0 = 1.5 + 1.3 * k, 6 + 1.5 * k
        ax.add_patch(Rectangle((x0, y0), 15, 12, facecolor="white", edgecolor=INK2, lw=0.5, zorder=2 + k))
        if k == 4:
            for i in range(12):
                for j in range(15):
                    ax.add_patch(Rectangle((x0 + j, y0 + i), 1, 1, facecolor=C[(i // 3) % 4] if False else "#d9e6f7",
                                           edgecolor="white", lw=0.25, zorder=8))
    ax.text(11, 3.2, f"{total / 1e6:.1f} million LPs", ha="center", fontsize=7, color=INK, fontweight="bold")
    ax.text(11, 0.6, "12 species × 481 steps × 16,384 members", ha="center", fontsize=5.6, color=INK2)
    ax.text(22.2, 24.6, "members", fontsize=5.5, color=INK2, rotation=50)
    ax.text(9.2, 19.6 + 6, "", fontsize=5)

    def arrow(x0, x1, y=15):
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=7, color=INK2, lw=0.8))

    arrow(25, 30)
    # the observation
    ax.add_patch(FancyBboxPatch((30.5, 8.5), 21, 13, boxstyle="round,pad=0.4,rounding_size=1.2",
                                facecolor="#f3f2ef", edgecolor="none"))
    ax.text(41, 18.6, "one matrix per species;", ha="center", fontsize=6.2, color=INK)
    ax.text(41, 16.2, "only uptake bounds move", ha="center", fontsize=6.2, color=INK)
    ax.text(41, 12.0, r"$z_B = h + T\,(z_N - \hat z_N)$", ha="center", fontsize=7, color=INK)
    ax.text(41, 9.6, "affine within each critical region", ha="center", fontsize=5.4, color=INK2)
    arrow(52.5, 57)

    # basis pool
    for i in range(31):
        r_, c_ = divmod(i, 8)
        ax.add_patch(Rectangle((57.5 + c_ * 1.6, 9.5 + r_ * 1.6), 1.35, 1.35, facecolor=C[i % 8], edgecolor="none",
                               alpha=0.85))
    ax.text(64, 18.6, f"{st['new_bases']} certified bases", ha="center", fontsize=6.2, color=INK)
    ax.text(64, 6.6, "one GEMM per basis", ha="center", fontsize=5.4, color=INK2)
    arrow(71.5, 75)

    # fates of the 94.6 M LPs, bar length on a log scale
    fates = [("certified by its own basis", st["warm"]), ("certified by another cached basis", st["pool"]),
             ("certified by a new basis (propagation)", st["propagated"]),
             ("solved by simplex (repair)", st["highs_solves"])]
    for k, (lab, n) in enumerate(fates):
        y = 24 - 5.4 * k
        w = 13 * np.log10(max(n, 1)) / np.log10(total)
        ax.add_patch(Rectangle((75.5, y - 1.5), w, 3.0, facecolor=ORANGE if k == 3 else BLUE, edgecolor="none",
                               alpha=1.0 if k == 0 else 0.75))
        pct = 100 * n / total
        ptxt = f"{pct:.2f}%" if pct >= 0.01 else f"{pct:.1e}%"
        ax.text(75.5, y + 2.1, f"{lab}", fontsize=5.3, color=INK2, va="bottom")
        ax.text(75.5 + w + 0.6, y, f"{n:,}  ({ptxt})", fontsize=5.6, color=INK, va="center")
    ax.text(86, 0.6, "fate of every LP (bar length log-scaled)", ha="center", fontsize=5.3, color=INK2)


# ---------------------------------------------------------------------------------------------
# b  phase diagram
# ---------------------------------------------------------------------------------------------

def _modes(sign, mets):
    ig, ia, ifo = mets.index("glc__D_e"), mets.index("ac_e"), mets.index("for_e")
    g, a, f = sign[..., ig], sign[..., ia], sign[..., ifo]
    grow = np.abs(sign).sum(-1) > 0
    m = np.full(g.shape, 5)                                    # no growth
    m[grow & (g < 0) & (a > 0) & (f > 0)] = 0
    m[grow & (g < 0) & (a > 0) & (f <= 0)] = 1
    m[grow & (g < 0) & (a == 0)] = 2
    m[grow & (g < 0) & (a < 0)] = 3
    m[grow & (g == 0) & (a < 0)] = 4
    names = ["glucose →\nacetate + formate", "glucose → acetate", "glucose only",
             "glucose + acetate", "acetate only", "no growth"]
    return m, names


def panel_phase(ax, fig):
    import matplotlib.colors as mcolors

    d = np.load(f"{R}/hero/phase.npz")
    c = d["axis"]
    mets = [str(x) for x in d["ex_mets"]]
    modes, names = _modes(d["sign"], mets)
    tints = ["#c9dcf5", "#dbe8f9", "#f6e3b4", "#cfeedd", "#f6d3c3", "#ebeae7"]
    cmap = mcolors.ListedColormap(tints)
    ax.pcolormesh(c, c, modes, cmap=cmap, vmin=-0.5, vmax=5.5, shading="nearest", rasterized=True)
    # critical-region (basis) boundaries
    B = d["basis"]
    edge = np.zeros(B.shape, dtype=bool)
    edge[:, 1:] |= B[:, 1:] != B[:, :-1]
    edge[1:, :] |= B[1:, :] != B[:-1, :]
    ax.contour(c, c, edge.astype(float), levels=[0.5], colors="white", linewidths=0.35)
    # growth-rate contours
    cs = ax.contour(c, c, d["growth"], levels=[0.1, 0.2, 0.3, 0.4, 0.5], colors=INK2, linewidths=0.45,
                    alpha=0.7)
    G_ = d["growth"]
    row = int(np.searchsorted(c, 3e-2))                         # label each contour low in the plot
    man = []
    for lev in cs.levels:
        j = int(np.argmin(np.abs(G_[row] - lev)))
        if abs(G_[row, j] - lev) < 0.02:
            man.append((c[j], c[row]))
    ax.clabel(cs, fmt=lambda v: f"{v:g} h$^{{-1}}$", fontsize=4.6, inline_spacing=1, manual=man)
    # trajectories
    cg, ca = np.nan_to_num(d["traj_cglc"]), np.nan_to_num(d["traj_cac"])
    for e in range(1, cg.shape[1]):
        ax.plot(cg[:, e], ca[:, e], color=INK2, lw=0.5, alpha=0.45)
    ok = np.ones(cg.shape[0], dtype=bool)
    x, y = cg[ok, 0], ca[ok, 0]
    ax.plot(x, y, color=INK, lw=1.4)
    tb = d["traj_basis"][ok, 0]
    sw = np.nonzero(tb[1:] != tb[:-1])[0] + 1
    ax.scatter(x[sw], y[sw], s=11, facecolor="white", edgecolor=INK, lw=0.8, zorder=5)
    n_sw = len(sw)
    # time arrows along the canonical trajectory
    tt = d["times"][ok]
    for tmark in (1.0, 4.0, 7.0, 9.5):
        i = int(np.searchsorted(tt, tmark))
        if 0 < i < len(x) - 1:
            ax.annotate("", (x[i + 1], y[i + 1]), (x[i - 1], y[i - 1]),
                        arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8, mutation_scale=6))
    lt = float(c[1])
    ax.set_xscale("symlog", linthresh=lt, linscale=0.35)
    ax.set_yscale("symlog", linthresh=lt, linscale=0.35)
    ax.set_xlim(0, c[-1])
    ax.set_ylim(0, c[-1])
    ticks = [0, 1e-3, 1e-2, 1e-1, 1, 10]
    lab = ["0", "10$^{-3}$", "10$^{-2}$", "10$^{-1}$", "1", "10"]
    ax.set_xticks(ticks, lab)
    ax.set_yticks(ticks, lab)
    _style(ax, "glucose (mM)", "acetate (mM)", "$E. coli$ core: exact map of metabolic regimes")
    # direct region labels, placed by hand in data coordinates (glucose, acetate)
    place = {"glucose →\nacetate + formate": (4.0, 0.25), "glucose → acetate": (1.0, 3.0e-3),
             "glucose + acetate": (3e-2, 0.4), "acetate only": (0.0, 0.3), "no growth": (4e-3, 3.5e-2)}
    for k, nm in enumerate(names):
        if (modes == k).sum() < 50 or nm not in place:
            continue
        px, py = place[nm]
        rot = 90 if nm == "acetate only" else 0
        ax.text(px, py, nm, fontsize=5.0, color=INK, ha="center", va="center", rotation=rot,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75))
    n_lps = int(d["grid_lps"])
    ax.annotate("glucose only", (0.3, 0.0), xytext=(0.3, 5.5e-4), fontsize=5.0, color=INK, ha="center",
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.5),
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75))
    ax.text(0.03, 0.10, f"{n_lps:,} LPs\ncertified in {float(d['grid_seconds']):.1f} s\n"
            f"{len(np.unique(B))} critical regions\n{n_sw} simplex solves\non the trajectory",
            transform=ax.transAxes, ha="left", va="center", fontsize=5.2, color=INK)


# ---------------------------------------------------------------------------------------------
# c  speed against accuracy
# ---------------------------------------------------------------------------------------------

def panel_accuracy(ax):
    rows = [json.load(open(f))["summary"] for f in glob.glob(f"{R}/solvers/*@workload_coherent.json")]
    show = {"manylp-cpu-fba": ("manylp CPU", BLUE, "o", True), "manylp-gpu-fba": ("manylp GPU", BLUE, "o", False),
            "manylp-cpu-pfba-unique": ("manylp pFBA-unique", BLUE, "s", True),
            "highs-simplex-warm-p32": ("HiGHS", ORANGE, "o", True),
            "gurobi-dual-warm-p32": ("Gurobi", ORANGE, "o", True), "xpress-dual-p32": ("Xpress", ORANGE, "o", True),
            "glpk-p32": ("GLPK", ORANGE, "o", True), "glop-p32": ("GLOP", ORANGE, "o", True),
            "bunching-p8": ("bunching", ORANGE, "D", True),
            "cuopt-pdlp-batch-e1e-6": ("cuOpt PDLP", GREEN, "o", False),
            "cuopt-pdlp-xover-e1e-6": ("cuOpt PDLP+crossover", GREEN, "o", False),
            "mpax-e1e-6": ("MPAX", GREEN, "o", False), "ourpdhg-e1e-6": ("batched PDHG", GREEN, "o", False)}
    floor = 1e-13
    lab_off = {"manylp CPU": (2, 8, "center"), "manylp GPU": (-5, 0, "right"),
               "manylp pFBA-unique": (0, 7, "center"), "HiGHS": (-4, -4, "right"), "Gurobi": (-4, 4, "right"),
               "Xpress": (4, 3, "left"), "GLPK": (-4, 0, "right"), "GLOP": (-4, 0, "right"),
               "bunching": (0, 7, "center"), "cuOpt PDLP": (4, 0, "left"),
               "cuOpt PDLP+crossover": (4, 0, "left"), "MPAX": (4, 0, "left"), "batched PDHG": (4, 0, "left")}
    for r in rows:
        if r["solver"] not in show:
            continue
        lab, col, mk, filled = show[r["solver"]]
        x = r.get("steady_lps_per_second") or r["lps_per_second"]
        y = max(r.get("obj_rel_err_max") or floor, floor)
        ax.scatter(x, y, s=18, marker=mk, facecolor=col if filled else "white", edgecolor=col, lw=1.0, zorder=3)
        dx, dy, ha = lab_off[lab]
        ax.annotate(lab, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=5.0, color=INK2, ha=ha,
                    va="center")
    ax.axhline(floor, color=GRID, lw=0.6)
    ax.text(0.15, floor * 1.6, "exact 0", fontsize=4.8, color=INK2)
    ax.axvspan(1e4, 1e6, ymax=0.32, color=BLUE, alpha=0.06, lw=0)
    ax.text(1.15e4, 4e-12, "fast and exact", fontsize=5, color=BLUE, va="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.1, 2e5)
    ax.set_ylim(3e-14, 3)
    _style(ax, "LPs per second (steady state)", "max relative objective error",
           "185,088 genome-scale LPs: speed vs accuracy")
    for col, lab in ((BLUE, "manylp (certified)"), (ORANGE, "exact CPU solvers"), (GREEN, "first-order GPU")):
        ax.scatter([], [], s=14, color=col, label=lab)
    ax.scatter([], [], s=14, facecolor="white", edgecolor=INK2, label="open = GPU")
    ax.legend(frameon=False, fontsize=5, loc="upper right", handletextpad=0.2, borderaxespad=0.2)


# ---------------------------------------------------------------------------------------------
# d, e  alternative optima in the gut community
# ---------------------------------------------------------------------------------------------

def panel_envelope(ax):
    d = np.load(f"{R}/alt_optima/envelope.npz", allow_pickle=True)
    t, M, env, pols = d["times"], d["M"], list(d["env_mets"]), list(d["policies"])
    j = env.index("cpd00029_e0")
    Y = M[:, :, j]
    ax.fill_between(t, Y.min(1), Y.max(1), color=BLUE, alpha=0.14, lw=0)
    for q, p in enumerate(pols):
        if p.startswith("random"):
            ax.plot(t, Y[:, q], color=BLUE, lw=0.3, alpha=0.35)
    for q, p in enumerate(pols):
        if p.startswith("max:") or p.startswith("min:"):
            ax.plot(t, Y[:, q], color=ORANGE, lw=0.35, alpha=0.6)
    ax.plot(t, Y[:, 0], color=INK, lw=1.3)
    fin = Y[-1]
    ax.annotate(f"{fin.max() / fin.min():.1f}× spread\nat 48 h", (t[-1], fin.max()), xytext=(-4, -2),
                textcoords="offset points", fontsize=5.2, color=INK, ha="right", va="top")
    ax.text(t[-1], Y[-1, 0], " pFBA-unique", fontsize=5, color=INK, va="center")
    _style(ax, "time (h)", "acetate (mM)", f"{len(pols)} equally optimal flux policies")
    ax.set_xlim(0, 60)
    ax.set_xticks([0, 12, 24, 36, 48])


def panel_composition(ax):
    d = np.load(f"{R}/alt_optima/envelope.npz", allow_pickle=True)
    X, sp = d["X"], [str(s) for s in d["species"]]
    ra = X[-1] / X[-1].sum(-1, keepdims=True)           # (policies, species), relative abundance at 48 h
    rel = 100 * (ra / ra[0] - 1)                        # change against pFBA-unique (%)
    lo, hi = rel.min(0), rel.max(0)
    order = np.argsort(hi - lo)
    for k, i in enumerate(order):
        ax.plot([lo[i], hi[i]], [k, k], color=BLUE, lw=2.8, alpha=0.4, solid_capstyle="butt")
        ax.scatter(lo[i], k, s=5, color=BLUE, zorder=3)
        ax.scatter(hi[i], k, s=5, color=BLUE, zorder=3)
    ax.axvline(0, color=INK, lw=0.7)
    nice = []
    for s_ in sp:
        g, rest = s_.split("_", 1)
        nice.append(f"{g}. {rest.split('_')[0]}")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{nice[i]} ({100 * ra[0, i]:.1f}%)" for i in order], fontsize=5.0, style="italic")
    _style(ax, "relative abundance at 48 h,\nchange vs pFBA-unique (%)", None, "…and community composition")
    ax.tick_params(axis="y", length=0)


# ---------------------------------------------------------------------------------------------
# f  scaling to thousands of species
# ---------------------------------------------------------------------------------------------

def panel_scaling(ax):
    recs = [json.load(open(f)) for f in glob.glob(f"{R}/scaling/*.json")]
    if not recs:
        ax.text(0.5, 0.5, "scaling run pending", transform=ax.transAxes, ha="center")
        return

    def series(pred):
        pts = sorted((r["N"], r) for r in recs if pred(r))
        return [p[0] for p in pts], [p[1] for p in pts]

    # legacy: measured at 48 h, or per-step cost from a short horizon scaled to 481 steps
    N, rs = series(lambda r: r["backend"].startswith("legacy") and r["E"] == 1)
    meas = [(n, r["wall_seconds"]) for n, r in zip(N, rs) if r["t_end"] >= 48]
    extr = [(n, r["wall_seconds"] / r["steps"] * 481) for n, r in zip(N, rs) if r["t_end"] < 48]
    if meas:
        ax.plot(*zip(*meas), "-o", color=ORANGE, ms=3, lw=1.2, label="cobra + GLPK (muODE legacy)")
    if extr:
        pts = sorted(meas[-1:] + extr) if meas else extr
        ax.plot(*zip(*pts), "--", color=ORANGE, lw=1.0)
        ax.plot(*zip(*extr), "o", mfc="white", mec=ORANGE, ms=3)
    N, rs = series(lambda r: r["backend"].startswith("manylp-cpu") and r["E"] == 1)
    if N:
        ax.plot(N, [r["wall_seconds"] for r in rs], "-o", color=BLUE, ms=3, lw=1.4, label="manylp, 1 trajectory")
    N, rs = series(lambda r: r["backend"].startswith("manylp-gpu") and r["E"] > 1)
    if N:
        E = rs[0]["E"]
        ax.plot(N, [r["wall_seconds"] for r in rs], "-s", color=C[6], ms=3, lw=1.2,
                label=f"manylp GPU, {E}-member ensemble")
    for v, lab in ((60, "1 min"), (3600, "1 h"), (86400, "1 day"), (7 * 86400, "1 week")):
        ax.axhline(v, color=GRID, lw=0.5, zorder=0)
        ax.text(1.02, v, lab, transform=ax.get_yaxis_transform(), fontsize=4.8, color=INK2, va="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    _style(ax, "species in the community", "wall time, 48 h simulated", "Scaling to 2,168 gut species")
    ax.legend(frameon=False, fontsize=4.9, loc="upper left", handlelength=1.5)


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": ["Liberation Sans", "Arial", "DejaVu Sans"], "pdf.fonttype": 42,
                         "mathtext.fontset": "dejavusans"})
    os.makedirs(OUT, exist_ok=True)
    W = 7.2
    fig = plt.figure(figsize=(W, 8.4))
    gs = fig.add_gridspec(3, 6, height_ratios=[0.78, 1.75, 1.3], hspace=0.30, wspace=1.15,
                          left=0.07, right=0.955, top=0.975, bottom=0.07)
    axa = fig.add_subplot(gs[0, :])
    axb = fig.add_subplot(gs[1, :3])
    axc = fig.add_subplot(gs[1, 3:])
    axd = fig.add_subplot(gs[2, :2])
    axe = fig.add_subplot(gs[2, 2:4])
    axf = fig.add_subplot(gs[2, 4:])
    panel_workload(axa)
    panel_phase(axb, fig)
    panel_accuracy(axc)
    panel_envelope(axd)
    panel_composition(axe)
    panel_scaling(axf)
    for ax, k in zip((axa, axb, axc, axd, axe, axf), "abcdef"):
        _label(fig, ax, k, dx=0.0 if k == "a" else -0.035)
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/figure_overview.{ext}", dpi=400)
    print("wrote", f"{OUT}/figure_overview.png")


if __name__ == "__main__":
    main()
