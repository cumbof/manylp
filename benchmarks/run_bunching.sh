#!/bin/bash
# Classical bunching baseline (benchmarks/bunching.py) on every workload manylp was measured on.
set -x
B="python benchmarks/bench_solvers.py --budget 900 --out results/solvers"
$B --workload results/workload_coherent.pkl --solver bunching-p32
$B --solver bunching-p32
$B --workload results/workload_coherent.pkl --solver bunching-p1
$B --workload results/workload_coherent.pkl --solver bunching-k128-p32
python benchmarks/bench_dfba.py --t-end 48 --out results/dfba_v3 --E 64,1024 --mode fba --solvers bunching-p32
python benchmarks/bench_netlib.py --B 1024 --bunching --out results/netlib_bunching
echo BUNCHING_DONE
