#!/bin/bash
# CPU solvers of the head-to-head comparison (waits for suite v2)
while pgrep -f run_suite_v2.sh > /dev/null; do sleep 60; done
B="python benchmarks/bench_solvers.py --budget 900 --out results/solvers"
for s in manylp-cpu-fba manylp-cpu-pfba-unique highs-simplex-warm-p32 highs-simplex-cold-p32 highs-ipm-cold-p32 highs-pdlp-cold-p32 glpk-p32 scipy-p32 glop-p32 ortools-pdlp-p32 osqp-p32 cuopt-dual_simplex-e1e-6 ; do
  echo "=== $s $(date)"
  timeout 2400 $B --solver $s 2>&1 | grep -v -i warn | grep -E '"(name|lps|lps_per_second|success_rate|obj_rel_err_max|infeas_max|exchange_dev_max)"|Traceback|Error'
done
echo COMPARE_DONE
