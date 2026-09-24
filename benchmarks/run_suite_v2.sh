#!/bin/bash
# Final benchmark suite (optimised code). Sequential so timings do not interfere.
set -x
D="python benchmarks/bench_dfba.py --t-end 48 --out results/dfba_v2"
# --- community dFBA replay: manylp vs strong CPU baseline (multi-process warm HiGHS)
$D --E 1 --mode pfba-unique --solvers manylp-cpu,manylp-cuda,highs-warm-x1
$D --E 16,64 --mode pfba-unique --solvers manylp-cpu,manylp-cuda,manylp-cuda-perlp,highs-warm-p32
$D --E 256 --mode pfba-unique --solvers manylp-cpu,manylp-cuda,highs-warm-p32
$D --E 1024,4096 --mode pfba-unique --solvers manylp-cuda,manylp-cpu
$D --E 16384 --mode pfba-unique --solvers manylp-cuda
$D --E 64 --mode fba --solvers manylp-cpu,manylp-cuda,highs-warm-p32
$D --E 1024 --mode fba --solvers manylp-cuda,manylp-cpu
# --- muODE end-to-end (its own engine)
M="python benchmarks/bench_muode.py --t-end 48 --out results/muode"
$M --E 1 --backends legacy,manylp-cpu,manylp-gpu,highs
$M --E 1 --backends legacy --jobs 12
$M --E 16 --backends manylp-cpu,manylp-gpu,highs
$M --E 16 --backends legacy --jobs 12
$M --E 256,1024 --backends manylp-gpu,manylp-cpu
echo SUITE_DONE
