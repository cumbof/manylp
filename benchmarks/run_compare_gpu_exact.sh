#!/bin/bash
# GPU solvers asked to return exact (basic) solutions via crossover, and PDLP at eps = 1e-4.
B="python benchmarks/bench_solvers.py --budget 900 --out results/solvers"
for s in cuopt-pdlp-xover-e1e-6 cuopt-barrier-xover-e1e-6 cuopt-pdlp-e1e-4 ; do
  echo "=== $s $(date)"
  timeout 2400 $B --solver $s 2>&1 | grep -v -i warn | grep -E '"(name|lps_per_second|success_rate|obj_rel_err_max|infeas_max)"|Traceback|Error'
done
for s in cuopt-pdlp-xover-e1e-6 cuopt-barrier-xover-e1e-6 ; do
  echo "=== $s (coherent) $(date)"
  timeout 2400 $B --workload results/workload_coherent.pkl --solver $s 2>&1 | grep -v -i warn | grep -E '"(name|lps_per_second|steady_lps_per_second|success_rate|obj_rel_err_max|infeas_max)"|Traceback|Error'
done
echo GPU_EXACT_DONE
