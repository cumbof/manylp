#!/bin/bash
# Yield-guard check, then the Netlib benchmark; NETLIB_DONE releases the final reruns (v3).
python benchmarks/profiling/check_yield_guard.py > logs/check_guard.log 2>&1
python benchmarks/bench_netlib.py --B 1024
echo NETLIB_DONE
