#!/bin/bash
# Commercial solvers (size-limited licences cover these LPs); after the CPU comparison.
while pgrep -f "^bash benchmarks/run_compare.sh" > /dev/null || pgrep -f run_suite_v2.sh > /dev/null; do sleep 60; done
B="python benchmarks/bench_solvers.py --budget 900 --out results/solvers"
for s in gurobi-dual-warm-p32 gurobi-barrier-cold-p32 xpress-dual-p32 ; do
  echo "=== $s $(date)"
  timeout 2400 $B --solver $s 2>&1 | grep -v -i warn | grep -E '"(name|lps|lps_per_second|success_rate|obj_rel_err_max|infeas_max|exchange_dev_max)"|Traceback|Error'
done
echo COMMERCIAL_DONE
