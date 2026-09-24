#!/bin/bash
# CPU solvers on the coherent workload (after the snapshot-workload CPU and commercial queues)
while pgrep -f "^bash benchmarks/run_compare\.sh" > /dev/null || pgrep -f "^bash benchmarks/run_compare_commercial\.sh" > /dev/null; do sleep 60; done
B="python benchmarks/bench_solvers.py --budget 900 --out results/solvers"
echo "=== highs-lex-pfba-unique-p32 (snapshot workload) $(date)"
timeout 2400 $B --solver highs-lex-pfba-unique-p32 2>&1 | grep -v -i warn | grep -E '"(name|lps_per_second|success_rate|obj_rel_err_max)"|Traceback|Error'
for s in manylp-cpu-fba manylp-cpu-pfba-unique highs-simplex-warm-p32 highs-lex-pfba-unique-p32 gurobi-dual-warm-p32 xpress-dual-p32 glop-p32 glpk-p32 highs-ipm-cold-p32 scipy-p32 ; do
  echo "=== $s $(date)"
  timeout 2400 $B --workload results/workload_coherent.pkl --solver $s 2>&1 | grep -v -i warn | grep -E '"(name|lps|lps_per_second|steady_lps_per_second|success_rate|obj_rel_err_max|exchange_dev_max)"|Traceback|Error'
done
echo COHERENT_CPU_DONE
