"""Collect results/external/*.json into one Markdown table (results/external/summary.md)."""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "results", "external")


def f(x, p=3):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{p}g}"
    return str(x)


def main():
    rows = []
    for fn in sorted(glob.glob(os.path.join(OUT, "*.json"))):
        name = os.path.basename(fn)
        if name.startswith(("scenario_", "lex_native", "soplex")):
            continue
        d = json.load(open(fn))
        if name.startswith("manylp_"):
            for r in d["rows"]:
                if "error" in r:
                    rows.append(("manylp", d["scenario"], r["scheme"], r["E"], None, None, None, None, None, r["error"][:80]))
                    continue
                rows.append(("manylp", d["scenario"], r["scheme"], r["E"], r["wall_per_trajectory"], None,
                             r["lps"] / r["E"], r.get("max_rel_err_biomass"), r.get("max_rel_err_glc_ac"), ""))
            continue
        tool = d.get("tool", "?")
        E = d.get("E", 1)
        wall = d.get("wall_per_trajectory", d.get("wall_seconds"))
        jit = d.get("jit_per_trajectory", d.get("jit_seconds"))
        lps = d.get("glpk_simplex_calls_total", d.get("glpk_simplex_calls"))
        if lps is None:
            lps = d.get("gurobi_optimize_calls_total", d.get("gurobi_optimize_calls"))
        if lps is not None and E and "members" in d:
            lps = lps / d.get("members_run", E)
        scheme = name[:-5]   # the file tag encodes mode, eps, algorithm, rtol, E, horizon, tout
        note = d.get("error", "") or d.get("status", "") or ("stopped at t=%.3g" % d["terminated_at"]
                                                            if d.get("terminated_at") else "")
        rows.append((tool, d.get("scenario", d.get("case", "")), scheme, E, wall, jit, lps,
                     d.get("max_rel_err_biomass"), d.get("max_rel_err_glc_ac", d.get("max_rel_err_products")),
                     str(note)[:80]))
    lines = ["| tool | scenario | scheme / result file | E | wall s/traj | of which JIT | LPs/traj | err biomass | err products | note |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append("| " + " | ".join(f(x) for x in r) + " |")
    txt = "\n".join(lines)
    open(os.path.join(OUT, "summary.md"), "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
