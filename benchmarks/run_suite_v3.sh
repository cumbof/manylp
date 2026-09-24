#!/bin/bash
# Final manylp numbers with the finished code (runs after the comparison and Netlib).
set -x
until grep -q NETLIB_DONE logs/netlib.log 2>/dev/null; do sleep 60; done
python benchmarks/bench_throughput.py
D="python benchmarks/bench_dfba.py --t-end 48 --out results/dfba_v3"
$D --E 1,16,64,256,1024,4096 --mode pfba-unique --solvers manylp-cpu,manylp-cuda
$D --E 16384 --mode pfba-unique --solvers manylp-cuda,manylp-cpu
$D --E 64,1024 --mode fba --solvers manylp-cpu,manylp-cuda
python benchmarks/bench_coldstart.py --E 1,64,1024 --device cuda
python benchmarks/bench_coldstart.py --E 1,64,1024 --device cpu
echo V3_DONE
