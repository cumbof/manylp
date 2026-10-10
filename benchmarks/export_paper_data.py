"""Copy the result summaries behind the paper's tables and figures into benchmarks/data/.

JSON summaries are exported, plus the few raw trajectory arrays (*.npz) that Figure 2 and the
HiGHS/GLPK agreement numbers need (about 37 MB); all other raw arrays stay in results/.

    python benchmarks/export_paper_data.py
"""
import glob
import os
import shutil

SRC, DST = "results", "benchmarks/data"
DIRS = ["solvers", "dfba_v2", "dfba_v3", "muode", "muode_v2", "spatial", "adaptive", "alt_optima", "coldstart",
        "sparsity", "netlib", "netlib_patch", "netlib_bunching", "size", "throughput", "external"]

# raw arrays read by make_report.fig_envelope (Figure 2) and derived_numbers.py
RAW = ["alt_optima/envelope.npz",
       "dfba/cobra-glpk_fba_E1_T48.npz", "dfba/highs-warm-x1_fba_E1_T48.npz", "dfba/scipy-linprog_fba_E1_T48.npz",
       "dfba/cobra-glpk_fba_E1_T48.json", "dfba/highs-warm-x1_fba_E1_T48.json", "dfba/scipy-linprog_fba_E1_T48.json",
       "muode/manylp-cpu-pfba-unique_E1_T48.npz", "muode/manylp-cpu-pfba-unique_E16_T48.npz",
       "muode/highs-pfba-unique_E1_T48.npz", "muode/highs-pfba-unique_E16_T48.npz",
       "muode/legacy-cobra-glpk-j1_E1_T48.npz", "workload_coherent_reference_check.json"]

n = 0
for f in RAW:
    if os.path.exists(f"{SRC}/{f}"):
        os.makedirs(os.path.dirname(f"{DST}/{f}"), exist_ok=True)
        shutil.copy2(f"{SRC}/{f}", f"{DST}/{f}")
        n += 1
for d in DIRS:
    for f in glob.glob(f"{SRC}/{d}/*.json"):
        out = os.path.join(DST, os.path.relpath(f, SRC))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        shutil.copy2(f, out)
        n += 1
for f in ("report/fig_accuracy.png", "report/fig_envelope.png", "report/fig_netlib.png", "report/fig_scaling.png",
          "report/fig_size.png"):
    if os.path.exists(f"{SRC}/{f}"):
        os.makedirs(f"{DST}/figures", exist_ok=True)
        shutil.copy2(f"{SRC}/{f}", f"{DST}/figures/{os.path.basename(f)}")
        n += 1
for f in glob.glob(f"{SRC}/report/paper/figure[0-9].p[nd][gf]"):
    os.makedirs(f"{DST}/figures", exist_ok=True)
    shutil.copy2(f, f"{DST}/figures/{os.path.basename(f)}")
    n += 1
print(f"exported {n} files to {DST}/")
