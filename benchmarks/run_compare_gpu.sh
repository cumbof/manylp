#!/bin/bash
# GPU solvers of the head-to-head comparison
B="python benchmarks/bench_solvers.py --budget 900 --out results/solvers"
for s in manylp-gpu-fba manylp-gpu-pfba-unique manylp-gpu-perlp-fba cuopt-pdlp-e1e-6 cuopt-pdlp-e1e-4 cuopt-pdlp-warm-e1e-6 cuopt-pdlp-batch-e1e-6 cuopt-barrier-e1e-6 cuopt-concurrent-e1e-6 mpax-e1e-6 mpax-e1e-4 ourpdhg-e1e-6 ; do
  echo "=== $s $(date)"
  timeout 2400 $B --solver $s 2>&1 | grep -v -i warn | grep -E '"(name|lps|lps_per_second|success_rate|obj_rel_err_max|infeas_max|exchange_dev_max)"|Traceback|Error'
done
echo COMPARE_GPU_DONE
