#!/bin/bash
# Batched (Kall & Wallace) bunching: each basis checked against all pending LPs of the batch.
set -x
B="python benchmarks/bench_solvers.py --budget 900 --out results/solvers"
$B --workload results/workload_coherent.pkl --solver bunching-batch
$B --solver bunching-batch
python benchmarks/bench_dfba.py --t-end 48 --out results/dfba_v3 --E 64,1024 --mode fba --solvers bunching-batch
python benchmarks/bench_netlib.py --B 1024 --bunching --bunching-batch-only --out results/netlib_bunching
echo BUNCHING3_DONE
