#!/bin/bash
# GPU solvers on the coherent (consecutive-step) workload
B="python benchmarks/bench_solvers.py --budget 900 --out results/solvers --workload results/workload_coherent.pkl"
for s in manylp-gpu-fba manylp-gpu-pfba-unique manylp-gpu-perlp-fba cuopt-concurrent-e1e-6 cuopt-pdlp-batch-e1e-6 mpax-e1e-6 ourpdhg-e1e-6 ; do
  echo "=== $s $(date)"
  timeout 2400 $B --solver $s 2>&1 | grep -v -i warn | grep -E '"(name|lps|lps_per_second|steady_lps_per_second|success_rate|obj_rel_err_max|exchange_dev_max)"|Traceback|Error'
done
echo COHERENT_GPU_DONE
