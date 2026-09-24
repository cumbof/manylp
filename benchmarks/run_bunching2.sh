#!/bin/bash
# Bunching process-count sweep: with ~1 ms per cache hit, dispatch overhead dominates at 32 processes
# (2 LPs per process per 64-LP batch).  Report the best configuration for fairness.
until grep -q BUNCHING_DONE logs/bunching.log 2>/dev/null; do sleep 60; done
set -x
B="python benchmarks/bench_solvers.py --budget 900 --out results/solvers"
for p in 4 8 16; do
  $B --workload results/workload_coherent.pkl --solver bunching-p$p
  $B --solver bunching-p$p
done
echo BUNCHING2_DONE
