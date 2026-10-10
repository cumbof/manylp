"""Build the self-contained HTML report (tables + inline SVG charts) from results/.

    python benchmarks/make_html_report.py   ->  results/report/manylp_report.html
"""

from __future__ import annotations

import glob
import html
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from make_report import CAPS, dfba_rows, family  # noqa: E402

R = os.environ.get("MANYLP_RESULTS", "results")   # benchmarks/data also works
OUT = "results/report/manylp_report.html"


def esc(x) -> str:
    return html.escape(str(x))


def sci(x):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "–"
    if x == 0:
        return "0"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10 ** e
    return f"{m:.1f}×10<sup>{e}</sup>"


def rate(x):
    if x is None:
        return "–"
    return f"{x:,.0f}" if x >= 10 else f"{x:.2f}"


def pct(x):
    return "–" if x is None else f"{100 * x:.1f}%"


# ---------------------------------------------------------------------------
# SVG helpers (theme-aware: every colour is a CSS variable)
# ---------------------------------------------------------------------------

def _ticks_log(lo, hi):
    a, b = math.floor(math.log10(lo)), math.ceil(math.log10(hi))
    return [10 ** k for k in range(a, b + 1)]


def svg_lines(series, xlog2=True, ylog=True, w=720, h=380, xlabel="", ylabel="", title=""):
    """series: list of (label, css_var, [(x, y)], dashed)."""
    pad_l, pad_r, pad_t, pad_b = 64, 150, 16, 44
    xs = [p[0] for s in series for p in s[2]]
    ys = [p[1] for s in series for p in s[2]]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    if ylog:
        yt = _ticks_log(y0, y1)
        y0, y1 = yt[0], yt[-1]
    fx = (lambda v: math.log2(v)) if xlog2 else (lambda v: v)
    fy = (lambda v: math.log10(v)) if ylog else (lambda v: v)
    X = lambda v: pad_l + (fx(v) - fx(x0)) / max(fx(x1) - fx(x0), 1e-12) * (w - pad_l - pad_r)  # noqa: E731
    Y = lambda v: h - pad_b - (fy(v) - fy(y0)) / max(fy(y1) - fy(y0), 1e-12) * (h - pad_t - pad_b)  # noqa: E731
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{esc(title)}" class="chart">']
    for t in (yt if ylog else np.linspace(y0, y1, 5)):
        out.append(f'<line x1="{pad_l}" x2="{w - pad_r}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" class="grid"/>')
        lab = f"10<tspan dy='-5' font-size='9'>{int(math.log10(t))}</tspan>" if ylog else f"{t:g}"
        out.append(f'<text x="{pad_l - 8}" y="{Y(t) + 4:.1f}" class="tick" text-anchor="end">{lab}</text>')
    xvals = sorted(set(xs))
    for v in xvals:
        out.append(f'<text x="{X(v):.1f}" y="{h - pad_b + 18}" class="tick" text-anchor="middle">{v:g}</text>')
    out.append(f'<line x1="{pad_l}" x2="{w - pad_r}" y1="{h - pad_b}" y2="{h - pad_b}" class="axis"/>')
    out.append(f'<text x="{(pad_l + w - pad_r) / 2}" y="{h - 6}" class="lab" text-anchor="middle">{esc(xlabel)}</text>')
    out.append(f'<text x="14" y="{(pad_t + h - pad_b) / 2}" class="lab" text-anchor="middle" '
               f'transform="rotate(-90 14 {(pad_t + h - pad_b) / 2})">{esc(ylabel)}</text>')
    used = []
    for label, var, pts, dashed in series:
        pts = sorted(pts)
        d = " ".join(f"{'M' if i == 0 else 'L'}{X(x):.1f},{Y(y):.1f}" for i, (x, y) in enumerate(pts))
        dash = ' stroke-dasharray="5 4"' if dashed else ""
        out.append(f'<path d="{d}" fill="none" stroke="var({var})" stroke-width="2"{dash}/>')
        for x, y in pts:
            out.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="4.5" fill="var({var})" stroke="var(--surface)" '
                       f'stroke-width="2"><title>{esc(label)} — E = {x:g}: {y:,.0f} LP/s</title></circle>')
        ly = Y(pts[-1][1])
        while any(abs(ly - u) < 13 for u in used):
            ly += 13
        used.append(ly)
        out.append(f'<text x="{X(pts[-1][0]) + 10:.1f}" y="{ly + 4:.1f}" class="dlab">{esc(label)}</text>')
    out.append("</svg>")
    return "\n".join(out)


def svg_envelope(t, lo, hi, canon, others, title, w=520, h=300):
    pad_l, pad_r, pad_t, pad_b = 52, 14, 14, 40
    ymax = max(hi.max(), max((o[2].max() for o in others), default=0)) * 1.05
    X = lambda v: pad_l + v / t[-1] * (w - pad_l - pad_r)  # noqa: E731
    Y = lambda v: h - pad_b - v / ymax * (h - pad_t - pad_b)  # noqa: E731
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{esc(title)}" class="chart">']
    for k in range(5):
        v = ymax * k / 4
        out.append(f'<line x1="{pad_l}" x2="{w - pad_r}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" class="grid"/>')
        out.append(f'<text x="{pad_l - 6}" y="{Y(v) + 4:.1f}" class="tick" text-anchor="end">{v:.2f}</text>')
    for tv in (0, 12, 24, 36, 48):
        out.append(f'<text x="{X(tv):.1f}" y="{h - pad_b + 16}" class="tick" text-anchor="middle">{tv}</text>')
    out.append(f'<text x="{(pad_l + w) / 2}" y="{h - 4}" class="lab" text-anchor="middle">time (h)</text>')
    band = [f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(t, hi)] + [f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(t[::-1], lo[::-1])]
    out.append(f'<polygon points="{" ".join(band)}" fill="var(--s1)" fill-opacity="0.16">'
               f'<title>{esc(title)} range over all optimal-flux policies: {lo[-1]:.3f}–{hi[-1]:.3f} mM at 48 h</title></polygon>')
    d = " ".join(f"{'M' if i == 0 else 'L'}{X(a):.1f},{Y(b):.1f}" for i, (a, b) in enumerate(zip(t, canon)))
    out.append(f'<path d="{d}" fill="none" stroke="var(--s1)" stroke-width="2.4"><title>canonical (pFBA-unique): '
               f'{canon[-1]:.3f} mM at 48 h</title></path>')
    for lab, var, ys, ot in others:
        d = " ".join(f"{'M' if i == 0 else 'L'}{X(a):.1f},{Y(b):.1f}" for i, (a, b) in enumerate(zip(ot, ys)))
        out.append(f'<path d="{d}" fill="none" stroke="var({var})" stroke-width="2" stroke-dasharray="5 4">'
                   f'<title>{esc(lab)}: {ys[-1]:.3f} mM at 48 h</title></path>')
    out.append("</svg>")
    return "\n".join(out)


def svg_scatter(points, w=720, h=400):
    """points: (label, lps, err, css_var, shape)."""
    pad_l, pad_r, pad_t, pad_b = 70, 20, 16, 46
    xs = [p[1] for p in points]
    ys = [max(p[2], 1e-16) for p in points]
    xt, yt = _ticks_log(min(xs), max(xs)), _ticks_log(min(ys), max(ys))
    X = lambda v: pad_l + (math.log10(v) - math.log10(xt[0])) / (math.log10(xt[-1]) - math.log10(xt[0])) * (w - pad_l - pad_r)  # noqa: E731,E501
    Y = lambda v: h - pad_b - (math.log10(v) - math.log10(yt[0])) / (math.log10(yt[-1]) - math.log10(yt[0])) * (h - pad_t - pad_b)  # noqa: E731,E501
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="speed versus accuracy" class="chart">']
    for t in yt:
        out.append(f'<line x1="{pad_l}" x2="{w - pad_r}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" class="grid"/>')
        out.append(f'<text x="{pad_l - 8}" y="{Y(t) + 4:.1f}" class="tick" text-anchor="end">10'
                   f'<tspan dy="-5" font-size="9">{int(math.log10(t))}</tspan></text>')
    for t in xt:
        out.append(f'<text x="{X(t):.1f}" y="{h - pad_b + 18}" class="tick" text-anchor="middle">10'
                   f'<tspan dy="-5" font-size="9">{int(math.log10(t))}</tspan></text>')
    out.append(f'<line x1="{pad_l}" x2="{w - pad_r}" y1="{h - pad_b}" y2="{h - pad_b}" class="axis"/>')
    out.append(f'<text x="{(pad_l + w) / 2}" y="{h - 6}" class="lab" text-anchor="middle">LPs solved per second (log)</text>')
    out.append(f'<text x="14" y="{h / 2}" class="lab" text-anchor="middle" transform="rotate(-90 14 {h / 2})">'
               f'max relative objective error (log)</text>')
    for lab, x, y, var, shape in points:
        cx, cy = X(x), Y(max(y, 1e-16))
        tip = f"<title>{esc(lab)}: {x:,.1f} LP/s, max objective error {y:.1e}</title>"
        if shape == "o":
            out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6.5" fill="var({var})" stroke="var(--surface)" stroke-width="2">{tip}</circle>')
        elif shape == "s":
            out.append(f'<rect x="{cx - 5.5:.1f}" y="{cy - 5.5:.1f}" width="11" height="11" rx="2" fill="var({var})" stroke="var(--surface)" stroke-width="2">{tip}</rect>')
        else:
            out.append(f'<path d="M{cx:.1f},{cy - 7:.1f} L{cx + 6.5:.1f},{cy + 5:.1f} L{cx - 6.5:.1f},{cy + 5:.1f} Z" fill="var({var})" stroke="var(--surface)" stroke-width="2">{tip}</path>')
    out.append("</svg>")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------

def load_comparison(workload="workload.pkl"):
    rows = []
    for f in sorted(glob.glob(f"{R}/solvers/*.json")):
        if "reference" in os.path.basename(f):
            continue
        d = json.load(open(f))["summary"]
        if d.get("workload", "workload.pkl") != workload:
            continue
        fam = family(d["solver"])
        rows.append((d, fam, CAPS.get(fam, ("?",) * 9)))
    rows.sort(key=lambda r: -r[0]["lps_per_second"])
    return rows


def comparison_section(rows, tid="cmp"):
    exact = [d for d, fam, cap in rows if fam != "manylp" and d["success_rate"] > 0.99
             and (d.get("obj_rel_err_max") or 1) < 1e-6]
    best = max((d["lps_per_second"] for d in exact), default=None)
    best_steady = max((d.get("steady_lps_per_second") or 0 for d in exact), default=None) or None
    head = ("<tr><th scope='col'>Solver</th><th scope='col'>Method</th><th scope='col'>Device</th>"
            "<th scope='col' class='num'>LP/s</th><th scope='col' class='num'>Steady-state LP/s</th>"
            "<th scope='col' class='num'>Speed-up vs best exact baseline (overall / steady)</th>"
            "<th scope='col' class='num'>Solved to optimality</th><th scope='col' class='num'>Max obj. error</th>"
            "<th scope='col' class='num'>Max primal infeas.</th><th scope='col' class='num'>Exchange-flux deviation (median / max, mmol·gDW⁻¹·h⁻¹)</th>"
            "<th scope='col'>Batched</th><th scope='col'>Warm start</th><th scope='col'>Exact</th>"
            "<th scope='col'>Certified</th><th scope='col'>Unique flux</th><th scope='col'>pFBA / lex native</th>"
            "<th scope='col' class='num'>Workload done in budget</th><th scope='col' class='num'>GPU MiB</th><th scope='col'>License</th></tr>")
    body = []
    for d, fam, cap in rows:
        mine = fam == "manylp"
        sp = d["lps_per_second"] / best if best else None
        sps = (d.get("steady_lps_per_second") or 0) / best_steady if best_steady else None
        dev = ("GPU" if "gpu" in d["solver"] else "CPU") if mine else cap[1]
        yes = lambda s: f"<span class='{'y' if s.startswith('yes') else 'n'}'>{esc(s)}</span>"  # noqa: E731
        body.append(
            f"<tr class='{'mine' if mine else ''}'><th scope='row'>{esc(d['name'])}</th><td>{esc(cap[0])}</td>"
            f"<td>{esc(dev)}</td><td class='num' data-v='{d['lps_per_second']}'>{rate(d['lps_per_second'])}</td>"
            f"<td class='num' data-v='{d.get('steady_lps_per_second') or 0}'>{rate(d.get('steady_lps_per_second'))}</td>"
            f"<td class='num' data-v='{sps or sp or 0}'>{'–' if sp is None else f'{sp:,.1f}×'}"
            f"{'' if not sps else f' / {sps:,.1f}×'}</td>"
            f"<td class='num' data-v='{d['success_rate']}'>{pct(d['success_rate'])}</td>"
            f"<td class='num' data-v='{d.get('obj_rel_err_max') or 0}'>{sci(d.get('obj_rel_err_max'))}</td>"
            f"<td class='num' data-v='{d.get('infeas_max') or 0}'>{sci(d.get('infeas_max'))}</td>"
            f"<td class='num' data-v='{d.get('exchange_dev_max') or 0}'>{sci(d.get('exchange_dev_median'))} / {sci(d.get('exchange_dev_max'))}</td>"
            f"<td>{yes(cap[2])}</td><td>{yes(cap[3])}</td><td>{yes(cap[4])}</td><td>{yes(cap[5])}</td>"
            f"<td>{yes(cap[6])}</td><td>{yes(cap[7])}</td>"
            f"<td class='num' data-v='{d['lps'] / max(d['total_lps'], 1)}'>{pct(d['lps'] / max(d['total_lps'], 1))}</td>"
            f"<td class='num' data-v='{d.get('gpu_mem_peak_mib') or 0}'>{d.get('gpu_mem_peak_mib') or '–'}</td>"
            f"<td>{esc(cap[8])}</td></tr>")
    return f"<div class='tablewrap'><table id='{tid}' class='sortable'><thead>{head}</thead><tbody>{''.join(body)}</tbody></table></div>"


def build():
    comp = load_comparison("workload.pkl")
    drows = dfba_rows()
    parts = []

    # headline numbers
    def get(s, m, E):
        r = drows.get((s, m, E))
        return None if r is None else r["lps_per_second"]

    hl = []
    for E in (256, 64, 16):
        a = max(filter(None, [get("manylp-cuda", "pfba-unique", E), get("manylp-cpu", "pfba-unique", E)]), default=None)
        b = get("highs-warm-p32", "pfba-unique", E)
        if a and b:
            hl.append((E, a / b))
            break
    mu = {}
    for f in glob.glob(f"{R}/muode/*_E1_T48.json"):
        r = json.load(open(f))
        mu[r["backend"]] = r["wall_seconds"]
    legacy = mu.get("legacy-cobra-glpk-j1")
    ml = min([v for k, v in mu.items() if k.startswith("manylp")] or [None]) if mu else None
    env = json.load(open(f"{R}/alt_optima/summary.json")) if os.path.exists(f"{R}/alt_optima/summary.json") else None
    tiles = []
    if hl:
        tiles.append((f"{hl[0][1]:.0f}×", f"faster than warm-started HiGHS in 32 processes (community dFBA, E = {hl[0][0]})"))
    if legacy and ml:
        tiles.append((f"{legacy / ml:.0f}×", "faster end-to-end in muODE's own engine (12 GEMs, 48 h)"))
    tiles.append(("≈10⁻¹¹", "max objective error on the recorded workload — exact, certified answers"))
    if env:
        b = env["final_product_mM_range"].get("butyrate")
        if b:
            tiles.append((f"{b[2] / b[0]:.1f}×", "spread of predicted butyrate among equally optimal fluxes — made explicit"))
    parts.append("<section class='tiles'>" + "".join(
        f"<div class='tile'><div class='big'>{v}</div><div class='cap'>{esc(c)}</div></div>" for v, c in tiles) + "</section>")

    coh = load_comparison("workload_coherent.pkl")
    parts.append("<h2 id='comparison'>Head-to-head comparison</h2>")
    parts.append("<p class='lede'>Every solver replays the same recorded genome-scale FBA LPs from a 12-species gut-community "
                 "dFBA ensemble (64 members), in temporal order, as batches of one species' members. Returned flux vectors are audited "
                 "independently of what each solver claims. Per-LP CPU solvers run in 32 processes with members pinned, so each keeps "
                 "its warm state. First-order methods use ε = 10⁻⁶ unless noted. Each solver is capped at 900 s; "
                 "<em>workload done</em> shows how much it finished. Click a column header to sort.</p>")
    if coh:
        parts.append("<h3>Coherent workload: consecutive dFBA steps (185,088 LPs, first 24 h)</h3>"
                     "<p class='note'>The access pattern of a real dFBA run. Every step's bounds differ slightly from the previous "
                     "step's, which is the coherence both manylp's cache and warm-started simplex exploit.</p>")
        parts.append(comparison_section(coh, "cmp-coherent"))
    parts.append("<h3>Stress workload: sparse snapshots (23,808 LPs, every 1.6 h over 48 h)</h3>"
                 "<p class='note'>Snapshots 16 steps apart remove most step-to-step coherence, and with so few LPs the one-off cold start "
                 "weighs heavily. This is manylp's least favourable setting.</p>")
    parts.append(comparison_section(comp))
    parts.append("<p class='note'>Exchange-flux deviation is measured from the certified pFBA-unique solution. For "
                 "plain-FBA solvers it is not an error; it shows that <em>different solvers return different optimal fluxes for the "
                 "same LP</em>, which is what drives dFBA trajectories apart.</p>")

    pts = []
    for d, fam, cap in (coh or comp):
        if d.get("obj_rel_err_max") is None:
            continue
        if fam == "manylp":
            var, shp = "--s1", "o"
        elif cap[4].startswith("no"):
            var, shp = "--s3", "^"
        else:
            var, shp = "--s2", "s"
        pts.append((d["name"], d["lps_per_second"], d["obj_rel_err_max"], var, shp))
    if pts:
        parts.append(f"<h3>Speed versus accuracy ({'coherent' if coh else 'snapshot'} workload)</h3><div class='legend'><span><i class='dot' style='background:var(--s1)'></i>manylp</span>"
                     "<span><i class='sq' style='background:var(--s2)'></i>exact (simplex, IPM + crossover)</span>"
                     "<span><i class='tri' style='border-bottom-color:var(--s3)'></i>first-order (PDHG, ADMM)</span></div>")
        parts.append("<figure>" + svg_scatter(pts) + "</figure>")

    # scaling
    series = []
    for key, lab, var, dash in (("manylp-cuda", "manylp GPU", "--s1", False), ("manylp-cpu", "manylp CPU", "--s2", False),
                                ("highs-warm-p32", "HiGHS warm ×32 proc", "--s3", True),
                                ("manylp-cuda-perlp", "GPU, 1 LP per call", "--s4", True)):
        p = sorted((E, r["lps_per_second"]) for (s, m, E), r in drows.items() if s == key and m == "pfba-unique")
        if p:
            series.append((lab, var, p, dash))
    if series:
        parts.append("<h2 id='scaling'>Scaling with ensemble size</h2><p class='lede'>Full 12-species community, 48 h at "
                     "dt = 0.1 h, pFBA-unique fluxes; E ensemble members with log-normally perturbed diets (σ = 0.3). "
                     "Hover a point for its value.</p>")
        parts.append("<figure>" + svg_lines(series, xlabel="ensemble size E", ylabel="LPs per second",
                                            title="dFBA throughput versus ensemble size") + "</figure>")
        parts.append(dfba_table_html(drows))

    # envelope
    p = f"{R}/alt_optima/envelope.npz"
    if os.path.exists(p):
        d = np.load(p, allow_pickle=True)
        t, M, envm = d["times"], d["M"], list(d["env_mets"])
        figs = []
        for mid, lab in (("cpd00211_e0", "butyrate"), ("cpd00029_e0", "acetate")):
            j = envm.index(mid)
            others = []
            for s, olab, var in (("cobra-glpk", "GLPK (cobra), plain FBA", "--s2"), ("highs-warm-x1", "HiGHS, plain FBA", "--s3"),
                                 ("scipy-linprog", "SciPy linprog, plain FBA", "--s4")):
                f = glob.glob(f"{R}/dfba*/{s}_fba_E1_T48.npz")
                if f:
                    o = np.load(f[0])
                    others.append((olab, var, o["M"][:, 0, j], o["times"]))
            figs.append(f"<figure class='half'><figcaption>{lab} (mM)</figcaption>" +
                        svg_envelope(t, M[:, :, j].min(1), M[:, :, j].max(1), M[:, 0, j], others, lab) + "</figure>")
        parts.append("<h2 id='envelope'>Alternative optima change the biology</h2><p class='lede'>Same 12 models, same diet, same "
                     f"kinetics. The band spans {len(d['policies'])} selection policies over the optimal face (random generic "
                     "directions; maximal or minimal secretion of each fermentation product), each integrated as a certified, unique "
                     "trajectory. The dashed lines are what three conventional solvers return for plain FBA. All of them are equally "
                     "optimal at every step.</p>")
        parts.append("<div class='legend'><span><i class='band'></i>envelope over all policies</span>"
                     "<span><i class='ln' style='background:var(--s1)'></i>canonical pFBA-unique</span>"
                     "<span><i class='ln dash' style='border-color:var(--s2)'></i>GLPK</span>"
                     "<span><i class='ln dash' style='border-color:var(--s3)'></i>HiGHS</span>"
                     "<span><i class='ln dash' style='border-color:var(--s4)'></i>SciPy</span></div>")
        parts.append("<div class='figrow'>" + "".join(figs) + "</div>")
        if env:
            rows = "".join(f"<tr><th scope='row'>{esc(k)}</th><td class='num'>{v[0]:.3f}</td><td class='num'>{v[1]:.3f}</td>"
                           f"<td class='num'>{v[2]:.3f}</td></tr>" for k, v in env["final_product_mM_range"].items() if v[2] > 1e-3)
            parts.append("<div class='tablewrap'><table><thead><tr><th scope='col'>Product at 48 h</th><th scope='col' class='num'>min (mM)</th>"
                         "<th scope='col' class='num'>canonical</th><th scope='col' class='num'>max (mM)</th></tr></thead>"
                         f"<tbody>{rows}</tbody></table></div>")

    parts.append(extra_sections())
    return "\n".join(parts)


def dfba_table_html(drows):
    body = []
    for (s, mode, E), r in sorted(drows.items(), key=lambda kv: (kv[0][1], kv[0][2], -kv[1]["lps_per_second"])):
        hs = r.get("solver_stats", {}).get("highs_solves")
        hs_s = "–" if hs is None else f"{hs:,}"
        hs_p = "–" if hs is None else f"{100 * hs / r['n_lps']:.3f}%"
        mine = s.startswith("manylp")
        body.append(f"<tr class='{'mine' if mine else ''}'><th scope='row'>{esc(s)}</th><td>{esc(mode)}</td><td class='num'>{E:,}</td>"
                    f"<td class='num'>{r['n_lps']:,}</td><td class='num'>{r['solve_seconds']:,.1f}</td>"
                    f"<td class='num'>{rate(r['lps_per_second'])}</td>"
                    f"<td class='num'>{hs_s}</td><td class='num'>{hs_p}</td></tr>")
    return ("<details><summary>All community-dFBA runs</summary><div class='tablewrap'><table><thead><tr>"
            "<th scope='col'>Solver</th><th scope='col'>Flux rule</th><th scope='col' class='num'>E</th><th scope='col' class='num'>LPs</th>"
            "<th scope='col' class='num'>Solve (s)</th><th scope='col' class='num'>LP/s</th><th scope='col' class='num'>Simplex solves</th>"
            "<th scope='col' class='num'>LPs needing simplex</th></tr></thead><tbody>" + "".join(body) + "</tbody></table></div></details>")


def extra_sections():
    out = []
    # muODE end to end
    byk = {}
    for d in ("muode", "muode_v2"):               # the rerun (vectorised engine path) overrides
        for f in sorted(glob.glob(f"{R}/{d}/*.json")):
            r = json.load(open(f))
            byk[(r["backend"], r["E"])] = r
    mu = list(byk.values())
    if mu:
        leg = {r["E"]: r["wall_seconds"] for r in mu if r["backend"] == "legacy-cobra-glpk-j1"}
        rows = []
        for r in sorted(mu, key=lambda r: (r["E"], r["wall_seconds"])):
            base = leg.get(1)
            spd = (base * r["E"] / r["wall_seconds"]) if base else None
            extr = "" if r["E"] == 1 or r["backend"].startswith("legacy") else " (vs 16×/E× legacy single run)"
            rows.append(f"<tr class='{'mine' if r['backend'].startswith('manylp') else ''}'><th scope='row'>{esc(r['backend'])}</th>"
                        f"<td class='num'>{r['E']}</td><td class='num'>{r['wall_seconds']:,.1f}</td>"
                        f"<td class='num'>{'–' if spd is None else f'{spd:,.1f}×'}{esc(extr) if spd else ''}</td></tr>")
        out.append("<h2 id='muode'>Inside muODE</h2><p class='lede'>muODE's own engine (coroutine refactor, "
                   "<code>SolverBackend</code>), 12 gapseq GEMs as <code>CobraOrganism</code>, western diet, 48 h. The legacy backend "
                   "is muODE's historical path (cobra + GLPK) and reproduces pre-refactor output byte for byte. Speed-ups for E > 1 "
                   "are relative to E × the measured legacy single-trajectory time, because legacy solves every LP independently.</p>")
        out.append("<div class='tablewrap'><table><thead><tr><th scope='col'>Backend</th><th scope='col' class='num'>E</th>"
                   "<th scope='col' class='num'>Wall (s)</th><th scope='col' class='num'>Speed-up vs legacy</th></tr></thead><tbody>"
                   + "".join(rows) + "</tbody></table></div>")
    # spatial (COMETS-style) dFBA
    sp_rows = []
    legacy_rate = None
    for f in sorted(glob.glob(f"{R}/spatial/spatial_*.json"), key=lambda x: int(x.split("_")[-1].split("x")[0])):
        n = int(f.split("_")[-1].split("x")[0])
        d = json.load(open(f))
        if "legacy" in d:
            legacy_rate = d["legacy"]["lps_per_second"]
        for k, v in d.items():
            sp_rows.append((n, k, v))
    if sp_rows:
        rows = []
        for n, k, v in sp_rows:
            meas = k == "legacy"
            if meas:
                spd = "1×"
            elif legacy_rate:
                spd = f"{v['lps_per_second'] / legacy_rate:,.0f}×"
            else:
                spd = "–"
            est = "" if (meas or n == 8) else " (vs legacy rate measured at 8×8)"
            rows.append(f"<tr class='{'mine' if k.startswith('manylp') else ''}'><th scope='row'>{esc(k)}</th>"
                        f"<td class='num'>{n}×{n}</td><td class='num'>{v['lps']:,}</td><td class='num'>{v['wall_seconds']:,.1f}</td>"
                        f"<td class='num'>{v['lps_per_second']:,.1f}</td><td class='num'>{spd}{esc(est)}</td></tr>")
        out.append("<h2 id='spatial'>Spatial dFBA (COMETS-style grids)</h2><p class='lede'>muODE's spatial engine solves one LP per "
                   "species per grid cell per step: 12 gapseq GEMs, uniform inoculum, western diet, diffusion, 1 h at dt = 0.1 h. "
                   "With a backend, all cells' LPs of a step form one batch per species, with each cell a warm-start member. "
                   "Legacy is muODE's historical per-cell cobra + GLPK path; on larger grids its cost grows linearly with the "
                   "number of LPs, so speed-ups there use its measured 8×8 rate.</p>")
        out.append("<div class='tablewrap'><table><thead><tr><th scope='col'>Backend</th><th scope='col' class='num'>Grid</th>"
                   "<th scope='col' class='num'>LPs</th><th scope='col' class='num'>Wall (s)</th><th scope='col' class='num'>LP/s</th>"
                   "<th scope='col' class='num'>Speed-up vs legacy</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")

    # Netlib
    p = f"{R}/netlib/netlib_B1024.json"
    if os.path.exists(p):
        nl = [r for r in json.load(open(p)) if r["m"] > 0 and r["n"] > 0]   # skip failed downloads
        # rows re-run after a fix (results/netlib_patch/*.json) replace the originals
        patch = {r["problem"]: r for f in glob.glob(f"{R}/netlib_patch/*.json") for r in json.load(open(f))}
        nl = [patch.get(r["problem"], r) for r in nl]
        rows = []
        for r in nl:
            hk = [k for k in r if k.startswith("highs-")]
            bh = max(r[k]["lps_per_second"] for k in hk)
            bm = max(r["manylp-cpu"]["lps_per_second"], r["manylp-gpu"]["lps_per_second"])
            rows.append(f"<tr><th scope='row'>{esc(r['problem'])}</th><td class='num'>{r['m']:,}</td><td class='num'>{r['n']:,}</td>"
                        f"<td class='num'>{r['manylp-cpu']['distinct_bases']}</td><td class='num'>{r['manylp-cpu']['infeasible']}</td>"
                        f"<td class='num'>{bm:,.0f}</td><td class='num'>{bh:,.0f}</td><td class='num'>{bm / bh:,.1f}×</td>"
                        f"<td class='num'>{r['status_agreement']:.3f}</td><td class='num'>{sci(r['obj_rel_err_max'])}</td></tr>")
        sp = np.array([max(r["manylp-cpu"]["lps_per_second"], r["manylp-gpu"]["lps_per_second"]) /
                       max(r[k]["lps_per_second"] for k in r if k.startswith("highs-")) for r in nl])
        out.append("<h2 id='netlib'>Beyond biology: Netlib under uncertainty</h2><p class='lede'>45 Netlib LPs, 1024 scenarios each, "
                   "every finite row bound perturbed by 1 + 0.05·N(0,1). This is the inner loop of scenario analysis and stochastic "
                   f"programming. Geometric-mean speed-up over the best HiGHS configuration (cold or hot-started, 32 processes): "
                   f"<strong>{np.exp(np.log(sp).mean()):.1f}×</strong> (range {sp.min():.1f}–{sp.max():.1f}×).</p>")
        out.append("<details><summary>Per-instance results</summary><div class='tablewrap'><table><thead><tr><th scope='col'>Problem</th>"
                   "<th scope='col' class='num'>m</th><th scope='col' class='num'>n</th><th scope='col' class='num'>Regions</th>"
                   "<th scope='col' class='num'>Infeasible</th><th scope='col' class='num'>manylp LP/s</th><th scope='col' class='num'>HiGHS ×32 LP/s</th>"
                   "<th scope='col' class='num'>Speed-up</th><th scope='col' class='num'>Status agreement</th><th scope='col' class='num'>Max obj. error</th>"
                   "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div></details>")
    # biology-aware mechanisms
    bio = []
    sp = f"{R}/sparsity/sparsity.json"
    if os.path.exists(sp):
        s = json.load(open(sp))
        bio.append(("Limiting-nutrient sparsity", "exact",
                    f"Each certified regime needs {s['lazy']['n_active_mean']:.0f} of {s['lazy']['n_np_mean']:.0f} parametric columns "
                    f"on average ({100 * s['lazy']['n_active_mean'] / s['lazy']['n_np_mean']:.0f}%): only rate-limiting uptakes enter the affine law.",
                    f"{s['dense']['device_MiB'] / s['lazy']['device_MiB']:.1f}× less device memory per regime; throughput unchanged."))
    for name in ("coldstart/coldstart_cuda.json", "coldstart/coldstart_cpu.json"):
        if os.path.exists(f"{R}/{name}"):
            cs = json.load(open(f"{R}/{name}"))
            by = {(r["E"], r["variant"]): r for r in cs}
            e1 = (by.get((1, "progressive")), by.get((1, "progressive+atlas")))
            e64 = (by.get((64, "eager")), by.get((64, "progressive+atlas")))
            txt = []
            if all(e1):
                txt.append(f"E = 1: {e1[0]['solve_seconds']:.1f} → {e1[1]['solve_seconds']:.1f} s, simplex solves {e1[0]['simplex_solves']} → {e1[1]['simplex_solves']}")
            if all(e64):
                txt.append(f"E = 64: {e64[0]['solve_seconds']:.1f} → {e64[1]['solve_seconds']:.1f} s, simplex solves {e64[0]['simplex_solves']} → {e64[1]['simplex_solves']}")
            bio.append(("Metabolic-regime atlas", "exact",
                        "A 48-h community trajectory visits only ~30 regimes (optimal bases); saved per GEM and reloaded by later runs "
                        "with other diets, they are re-certified, never trusted.", "; ".join(txt) + f" ({name.split('_')[-1][:-5]})."))
            break
    for f, lab in ((f"{R}/adaptive/adaptive_cpu.json", "gut community, 48 h"), (f"{R}/adaptive/diauxie.json", "E. coli diauxie, 12 h")):
        if os.path.exists(f):
            a = json.load(open(f))
            by = {r["scheme"]: r for r in a}
            prod = next((r for k, r in by.items() if "production" in k or k == "Euler dt=0.1"), None)
            rk = next((r for k, r in by.items() if k.startswith("RK45") and "1e-6" in k and "event" not in k), None)
            if prod and rk:
                bio.append((f"Regime-aware adaptive integration ({lab})", "changes time discretisation, not LPs",
                            "Certified LPs are cheap, exact function evaluations, so embedded Runge–Kutta integration becomes affordable; "
                            "regime switches (changes in which uptake bounds bind) can optionally be located by event detection.",
                            f"biomass error {prod['max_rel_err_biomass']:.1e} → {rk['max_rel_err_biomass']:.1e} "
                            f"({prod['max_rel_err_biomass'] / rk['max_rel_err_biomass']:.0f}× more accurate) for "
                            f"{rk['lps'] / prod['lps']:.1f}× the LPs of Euler at dt = 0.1."))
    bio.append(("Alternative-optima policies", "exact (each policy unique)",
                "Biologically meaningful selection rules (e.g. maximal / minimal SCFA secretion) integrated in lockstep as ensemble members.",
                "Turns an invisible solver artefact into a quantified uncertainty band (see above) at batched cost."))
    rows = "".join(f"<tr><th scope='row'>{esc(a)}</th><td>{esc(b)}</td><td>{esc(c)}</td><td>{esc(d)}</td></tr>" for a, b, c, d in bio)
    out.append("<h2 id='bio'>Biology-aware mechanisms: what actually helps</h2><p class='lede'>Firewall: biology may inform "
               "performance and memory layout, never which answer is returned (except where a policy is chosen explicitly).</p>")
    out.append("<div class='tablewrap'><table><thead><tr><th scope='col'>Mechanism</th><th scope='col'>Exactness</th>"
               "<th scope='col'>Idea</th><th scope='col'>Measured effect</th></tr></thead><tbody>" + rows + "</tbody></table></div>")
    # trajectory-based dFBA simulators and manylp on the same scenarios (benchmarks/external/summarize.py)
    md = f"{R}/external/summary.md"
    if os.path.exists(md):
        lines = [ln.strip() for ln in open(md) if ln.strip().startswith("|")]
        if len(lines) > 2:
            cells = lambda ln: [c.strip() for c in ln.strip("|").split("|")]  # noqa: E731
            head = "".join(f"<th scope='col'>{esc(c)}</th>" for c in cells(lines[0]))
            body = "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in cells(ln)) + "</tr>" for ln in lines[2:])
            out.append("<h2 id='external'>Trajectory-based dFBA simulators on the same scenarios</h2><p class='lede'>The "
                       "<code>dfba</code> package (Harwood/Barton event method), surfinFBA and manylp on the E. coli core "
                       "diauxie and iJO1366 scenarios exported by <code>benchmarks/external/make_references.py</code>; "
                       "errors against a fine Euler reference.</p>")
            out.append(f"<div class='tablewrap'><table class='sortable'><thead><tr>{head}</tr></thead><tbody>{body}"
                       "</tbody></table></div>")
    return "\n".join(out)


CSS = """
<title>manylp benchmark report</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=Newsreader:opsz,wght@6..72,500;6..72,600&display=swap">
<style>
:root{
  --ground:#f5f7f4; --surface:#ffffff; --ink:#17211c; --ink2:#57625b; --rule:#dde3dc; --mine:#eaf2fc;
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100; --yes:#1f7a3d; --no:#8a918c;
  --grid:#e7ebe6;
}
@media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){
  color-scheme:dark; --ground:#111614; --surface:#181e1b; --ink:#e7ece8; --ink2:#a5afa8; --rule:#2c3530; --mine:#16263a;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500; --yes:#5fc27f; --no:#7d8680; --grid:#232b27;
}}
:root[data-theme="dark"]{
  color-scheme:dark; --ground:#111614; --surface:#181e1b; --ink:#e7ece8; --ink2:#a5afa8; --rule:#2c3530; --mine:#16263a;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500; --yes:#5fc27f; --no:#7d8680; --grid:#232b27;
}
body{background:var(--ground);color:var(--ink);font:15px/1.55 "IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding-inline:20px;padding-block:28px 64px}
header h1{font-family:"Newsreader",Georgia,serif;font-weight:600;font-size:clamp(28px,4vw,40px);line-height:1.1;margin:0 0 8px;text-wrap:balance}
header p{color:var(--ink2);max-width:72ch;margin:0}
h2{font-family:"Newsreader",Georgia,serif;font-weight:600;font-size:26px;margin:48px 0 6px;text-wrap:balance}
h3{font-size:16px;margin:28px 0 6px}
.lede{color:var(--ink2);max-width:80ch;margin:0 0 14px}
.note{color:var(--ink2);font-size:13px;max-width:90ch}
nav{display:flex;flex-wrap:wrap;gap:6px 18px;margin:18px 0 0;font-size:13px}
nav a{color:var(--ink2);text-decoration:none;border-bottom:1px solid var(--rule)}
nav a:hover,nav a:focus-visible{color:var(--ink);border-color:var(--ink)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:26px 0 8px}
.tile{background:var(--surface);border:1px solid var(--rule);border-radius:8px;padding:14px 16px}
.tile .big{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:30px;font-weight:500;color:var(--s1)}
.tile .cap{color:var(--ink2);font-size:13px}
.tablewrap{overflow-x:auto;border:1px solid var(--rule);border-radius:8px;background:var(--surface);margin:8px 0}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:7px 10px;border-bottom:1px solid var(--rule);text-align:left;vertical-align:top}
thead th{font-weight:600;font-size:12px;letter-spacing:.02em;color:var(--ink2);white-space:nowrap;position:sticky;top:0;background:var(--surface)}
tbody th{font-weight:500;white-space:nowrap}
.num{text-align:right;font-family:"IBM Plex Mono",ui-monospace,monospace;font-variant-numeric:tabular-nums;white-space:nowrap}
tr.mine{background:var(--mine)}
tr.mine th::before{content:"";display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--s1);margin-right:7px;vertical-align:1px}
.y{color:var(--yes)} .n{color:var(--no)}
table.sortable thead th{cursor:pointer}
table.sortable thead th:focus-visible{outline:2px solid var(--s1)}
table.sortable thead th[aria-sort]::after{content:" ▾";color:var(--ink2)}
table.sortable thead th[aria-sort="ascending"]::after{content:" ▴"}
figure{margin:10px 0;background:var(--surface);border:1px solid var(--rule);border-radius:8px;padding:10px}
figcaption{font-size:13px;color:var(--ink2);margin:0 0 4px 6px}
.figrow{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}
svg.chart{width:100%;height:auto;display:block}
svg .grid{stroke:var(--grid);stroke-width:1}
svg .axis{stroke:var(--ink2);stroke-width:1}
svg .tick{fill:var(--ink2);font:11px "IBM Plex Mono",ui-monospace,monospace}
svg .lab{fill:var(--ink2);font:12px "IBM Plex Sans",system-ui,sans-serif}
svg .dlab{fill:var(--ink);font:12px "IBM Plex Sans",system-ui,sans-serif}
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:13px;color:var(--ink2);margin:4px 0}
.legend i{display:inline-block;vertical-align:middle;margin-right:6px}
.dot{width:10px;height:10px;border-radius:50%} .sq{width:10px;height:10px;border-radius:2px}
.tri{width:0;height:0;border-left:6px solid transparent;border-right:6px solid transparent;border-bottom:10px solid}
.band{width:18px;height:10px;background:var(--s1);opacity:.25;border-radius:2px}
.ln{width:18px;height:2px} .ln.dash{height:0;border-top:2px dashed;background:none}
details{margin:10px 0} summary{cursor:pointer;color:var(--ink2);font-size:13px}
code{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.92em}
footer{margin-top:48px;color:var(--ink2);font-size:13px;max-width:90ch}
</style>
"""

JS = """
<script>
(function(){
  document.querySelectorAll('table.sortable').forEach(function(t){
    var ths=t.tHead.rows[0].cells;
    Array.prototype.forEach.call(ths,function(th,i){
      th.tabIndex=0;
      function go(){
        var asc=th.getAttribute('aria-sort')!=='ascending';
        Array.prototype.forEach.call(ths,function(x){x.removeAttribute('aria-sort');});
        th.setAttribute('aria-sort',asc?'ascending':'descending');
        var rows=Array.prototype.slice.call(t.tBodies[0].rows);
        rows.sort(function(a,b){
          var A=a.cells[i],B=b.cells[i];
          var va=A.dataset.v!==undefined?parseFloat(A.dataset.v):A.textContent.trim();
          var vb=B.dataset.v!==undefined?parseFloat(B.dataset.v):B.textContent.trim();
          if(typeof va==='number'&&typeof vb==='number')return asc?va-vb:vb-va;
          return asc?String(va).localeCompare(vb):String(vb).localeCompare(va);
        });
        rows.forEach(function(r){t.tBodies[0].appendChild(r);});
      }
      th.addEventListener('click',go);
      th.addEventListener('keydown',function(e){if(e.key==='Enter'||e.key===' '){e.preventDefault();go();}});
    });
  });
})();
</script>
"""


def main():
    body = build()
    page = (CSS + "<div class='wrap'><header><h1>Certify, don't re-solve</h1><p>manylp: exact batched linear programming "
            "for dynamic flux balance analysis. It certifies whole batches of LPs against cached optimal bases on GPU or CPU, "
            "and re-solves only what falls outside them.</p>"
            "<nav><a href='#comparison'>Comparison</a><a href='#scaling'>Scaling</a><a href='#envelope'>Alternative optima</a>"
            "<a href='#muode'>muODE</a><a href='#spatial'>Spatial</a><a href='#netlib'>Netlib</a><a href='#bio'>Biology-aware mechanisms</a><a href='#methods'>Methods</a></nav>"
            "</header>" + body + METHODS + "</div>" + JS)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write(page)
    print(f"wrote {OUT} ({len(page) / 1e3:.0f} kB)")


METHODS = """
<h2 id="methods">Methods and caveats</h2>
<p class="lede">Hardware: one NVIDIA A100-PCIE-40GB; 4× Intel Xeon Platinum 8276L (224 threads) on a <strong>shared node carrying other
users' jobs</strong>, so CPU timings carry run-to-run noise. Timings are single runs, wall-clock. Correctness: every manylp answer is
re-derived by its own arithmetic (primal feasibility and lexicographic dual feasibility of a factorised basis) and cross-checked
against an independent lexicographic reference (fresh HiGHS model per LP, objective-row formulation) to within 1.6×10<sup>-11</sup>.
Baselines run with default settings (tolerances 10<sup>-6</sup> for first-order methods where set); where a solver failed or
was not converged, the table reports what it returned. NVIDIA cuOpt's <code>BatchSolve</code> is deprecated upstream and implemented
as concurrent host threads. Gurobi (size-restricted pip licence) and FICO Xpress (community licence) admit every gut model
(at most 1,981 variables and 1,645 constraints) but not the larger models of the size benchmark; qpth's package
is broken on PyPI and its dense-KKT design cannot hold genome-scale batches.</p>
<footer>Code, raw results and scripts: <code>manylp/benchmarks/</code>. Regenerate with <code>python benchmarks/make_html_report.py</code>.</footer>
"""

if __name__ == "__main__":
    main()
