"""Copy the result summaries behind the paper's tables and figures into paper/data/.

Only JSON summaries are exported (about 10 MB); raw trajectory arrays (*.npz) stay in results/.

    python benchmarks/export_paper_data.py
"""
import glob
import os
import shutil

SRC, DST = "results", "paper/data"
DIRS = ["solvers", "dfba_v2", "dfba_v3", "muode", "muode_v2", "spatial", "adaptive", "alt_optima", "coldstart",
        "sparsity", "netlib", "netlib_patch", "netlib_bunching", "size", "throughput", "external"]

n = 0
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
for f in glob.glob(f"{SRC}/report/paper/figure*.p[nd][gf]"):
    os.makedirs(f"{DST}/figures", exist_ok=True)
    shutil.copy2(f, f"{DST}/figures/{os.path.basename(f)}")
    n += 1
print(f"exported {n} files to {DST}/")
