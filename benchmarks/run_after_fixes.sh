#!/bin/bash
# Reruns after the regime-signature fix (adaptive.py) and the Farkas error-bound fix (basis.py).
until grep -q BUNCHING3_DONE logs/bunching3.log 2>/dev/null; do sleep 60; done
set -x
python benchmarks/bench_adaptive.py
python benchmarks/bench_diauxie.py
python benchmarks/external/run_manylp_same.py --scenario diauxie --E 32
python benchmarks/external/run_manylp_same.py --scenario iJO1366 --E 8
python benchmarks/external/summarize.py
echo FIXES_DONE
