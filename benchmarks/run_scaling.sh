#!/bin/bash
# Community-size scaling (bench_scaling.py): manylp chain on the GPU node, legacy chain pinned
# to separate cores.  Models: MICOM model database v1, uhgg201_gtdb207_species_1 (unzipped).
M=${MODELS:-../scaling/models}
B="python benchmarks/bench_scaling.py --models $M --out results/scaling"
set -x
if [ "$1" = legacy ]; then
  taskset -c 200-207 $B --N 10,100 --E 1 --backends legacy
  taskset -c 200-207 $B --N 500,1000,2168 --E 1 --backends legacy --legacy-t-end 1
  echo SCALING_LEGACY_DONE
else
  $B --N 10,100,500,1000,2168 --E 1 --backends manylp-cpu
  $B --N 100,500,1000 --E 64 --backends manylp-gpu
  $B --N 2168 --E 64 --backends manylp-gpu
  echo SCALING_MANYLP_DONE
fi
