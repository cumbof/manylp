#!/bin/bash
# Extra experiments after the final v3 reruns.
until grep -q V3_DONE logs/suite_v3.log 2>/dev/null; do sleep 60; done
python benchmarks/bench_size.py --B 1024 --T 12
echo AFTER_V3_DONE
