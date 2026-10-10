#!/usr/bin/env bash
# Re-run the experiments of the manylp paper, stage by stage (see benchmarks/REPRODUCE.md).
#
#   bash benchmarks/reproduce.sh list            # stages and what they produce
#   bash benchmarks/reproduce.sh <stage> ...     # run some stages
#   bash benchmarks/reproduce.sh all             # everything, in the original order
#
# Run from the repository root, with the conda environment of benchmarks/env/environment.yml active.
# Results go to results/, logs to logs/.  Optional settings (environment variables):
#   BASELINES_VENV  venv with the baseline solvers (benchmarks/env/activate_baselines.sh); default ./baselines
#   DFBA_PY         python of an environment with the dfba package (conda-forge dfba); default: dfba stage skipped
#   SOPLEX          SoPlex 8.1 binary (conda-forge soplex); default: soplex on PATH
#   MUODE_DIR       muODE checkout with the manylp backend (branch manylp-backend); default ../muODE
#   GUT_DIR         gut community (12 gapseq models, diet, abundances); default $MUODE_DIR/examples/gut_western
# Stages tagged [GPU] need an NVIDIA GPU, [LICENCE] Gurobi and FICO Xpress, [MUODE] the muODE simulator.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=. XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
export MUODE_DIR=${MUODE_DIR:-../muODE}
export GUT_DIR=${GUT_DIR:-$MUODE_DIR/examples/gut_western}
BASELINES_VENV=${BASELINES_VENV:-baselines}

run() { local name=$1; shift; mkdir -p results logs; echo "+ [$name] $*" >&2; bash -c "$*" > "logs/$name.log" 2>&1; }
baselines() { source benchmarks/env/activate_baselines.sh "$BASELINES_VENV"; }

stage_workloads() {      # LP workloads and certified references recorded from the gut community   [GPU]
  run record_snapshot "python benchmarks/record_workload.py && python benchmarks/make_reference.py"
  run record_coherent "python benchmarks/record_workload.py --E 64 --every 1 --t-end 24 --out results/workload_coherent.pkl && python benchmarks/make_reference.py results/workload_coherent.pkl"
  # independent cross-check of both references on 400 sampled LPs (Methods)
  run refcheck "python benchmarks/make_reference.py --check-only results/workload_coherent.pkl results/workload_coherent_reference.npz && python benchmarks/make_reference.py --check-only results/workload.pkl results/solvers/reference.npz"
}

stage_alt_optima() {     # Table 2, Figure 2 (envelope)                                     [GPU]
  run alt_optima "python benchmarks/alt_optima.py --n-random 32"
}

stage_sparsity() {       # parameter-sparse affine laws: 9.1% of columns, 7.5x less device memory   [GPU]
  run sparsity "python benchmarks/bench_sparsity.py"
}

stage_dfba() {           # Figure 1, Figure 2 (plain-FBA lines), Table 6 row 2 (manylp)          [GPU]
  run dfba_fig2 "python benchmarks/bench_dfba.py --t-end 48 --E 1 --mode fba --solvers manylp-cpu,highs-warm-x1,scipy-linprog,cobra-glpk"
  D="python benchmarks/bench_dfba.py --t-end 48 --out results/dfba_v2"                  # Figure 1A baselines
  run dfba_baselines "set -e
    $D --E 1 --mode pfba-unique --solvers manylp-cpu,manylp-cuda,highs-warm-x1
    $D --E 16,64 --mode pfba-unique --solvers manylp-cpu,manylp-cuda,manylp-cuda-perlp,highs-warm-p32
    $D --E 256 --mode pfba-unique --solvers manylp-cpu,manylp-cuda,highs-warm-p32"
  D="python benchmarks/bench_dfba.py --t-end 48 --out results/dfba_v3"                  # Figure 1A/B, manylp
  run dfba_manylp "set -e
    python benchmarks/bench_throughput.py
    $D --E 1,16,64,256,1024,4096 --mode pfba-unique --solvers manylp-cpu,manylp-cuda
    $D --E 16384 --mode pfba-unique --solvers manylp-cuda,manylp-cpu
    $D --E 64,1024 --mode fba --solvers manylp-cpu,manylp-cuda"
}

stage_coldstart() {      # Table 5                                                          [GPU]
  run coldstart "python benchmarks/bench_coldstart.py --E 1,64,1024 --device cuda && python benchmarks/bench_coldstart.py --E 1,64,1024 --device cpu"
}

stage_muode() {          # Table 3 (ensembles), agreement with HiGHS, COBRApy + GLPK difference [GPU] [MUODE]
  M="python benchmarks/bench_muode.py --t-end 48 --out results/muode"
  run muode "set -e
    $M --E 1 --backends legacy,manylp-cpu,manylp-gpu,highs
    $M --E 16 --backends manylp-cpu,manylp-gpu,highs
    $M --E 16 --backends legacy --jobs 12"
  run muode_v2 "set -e
    python benchmarks/bench_muode.py --backends manylp-gpu,manylp-cpu --E 1,16,256,1024 --out results/muode_v2
    python benchmarks/bench_muode.py --backends highs,legacy --E 1 --out results/muode_v2
    python benchmarks/bench_muode.py --backends highs --E 16 --out results/muode_v2"
}

stage_spatial() {        # Table 3 (spatial grids)                                          [GPU] [MUODE]
  run spatial "set -e
    python benchmarks/bench_spatial.py --n 8 --t-end 1.0
    for n in 16 32; do python benchmarks/bench_spatial.py --n \$n --t-end 1.0 --backends manylp-cpu,manylp-gpu; done"
}

stage_solvers() {        # Table 1, Tables S1 and S2                                        [GPU] [LICENCE]
  baselines
  run compare_gpu "bash benchmarks/run_compare_gpu.sh"                     # snapshot workload, GPU solvers
  run compare "bash benchmarks/run_compare.sh"                             # snapshot workload, CPU solvers
  run compare_commercial "bash benchmarks/run_compare_commercial.sh"       # snapshot workload, Gurobi, Xpress
  run coherent_gpu "bash benchmarks/run_compare_coherent_gpu.sh"           # coherent workload, GPU solvers
  run coherent_cpu "bash benchmarks/run_compare_coherent_cpu.sh"           # coherent workload, CPU solvers
  run gpu_exact "bash benchmarks/run_compare_gpu_exact.sh"                 # cuOpt with crossover
}

stage_netlib() {         # Figure 3B, Table S3                                              [GPU]
  baselines
  run netlib "python benchmarks/bench_netlib.py --B 1024"
  run netlib_patch "python benchmarks/bench_netlib.py --B 1024 --only bnl1 --out results/netlib_patch"
}

stage_size() {           # Figure 3A                                                        [GPU]
  run size "python benchmarks/bench_size.py --B 1024 --T 12"
}

stage_bunching() {       # Table 6 rows 1-2, classical bunching on the snapshot workload and on Netlib
  run bunching "bash benchmarks/run_bunching.sh"
  run bunching2 "bash benchmarks/run_bunching2.sh"
  run bunching3 "bash benchmarks/run_bunching3.sh"
}

stage_external() {       # SoPlex exact check, native lexicographic modes, dfba package     [LICENCE]
  run ext_soplex "python benchmarks/external/exact_soplex.py --n 200 --workers 1 --soplex ${SOPLEX:-soplex}"
  ( baselines
    run ext_lex_sizes "python benchmarks/external/lex_native.py --solver gurobi --sizes && python benchmarks/external/lex_native.py --solver xpress --sizes"
    run ext_lex "export OMP_NUM_THREADS=1
      python benchmarks/external/lex_native.py --solver gurobi --species 4 --start 1200 --nbatch 8
      python benchmarks/external/lex_native.py --solver xpress --species 4 --start 1200 --nbatch 8
      python benchmarks/external/lex_native.py --solver xpress --start 1200 --nbatch 8 --diagnose
      for mo in 1 3 5; do python benchmarks/external/lex_native.py --solver xpress --start 1200 --nbatch 8 --multiobjops \$mo --diagnose; done" )
  run ext_refs "set -e
    python benchmarks/external/make_references.py --diauxie --linear
    python benchmarks/external/make_references.py --ensemble 32
    python benchmarks/external/make_references.py --gsm iJO1366 --gsm-dt 1e-4"
  if [ -n "${DFBA_PY:-}" ]; then                                    # dfba package, one core per run
    R="$DFBA_PY benchmarks/external/run_dfba_pkg.py"
    run ext_dfba "set -e
      for t in 1e-4 1e-6 1e-8; do $R --rtol \$t --atol \$t; done
      $R --rtol 1e-6 --atol 1e-6 --ensemble 32
      $R --rtol 1e-8 --atol 1e-8 --ensemble 32 --tout 1e-4
      S='--scenario iJO1366 --rtol 1e-6 --atol 1e-6'
      $R \$S --alg direct; $R \$S --mode fba --alg direct; $R \$S --alg direct --ensemble 8
      $R \$S --mode fba --t-end 0.5; $R \$S --t-end 0.5; $R \$S --mode fba --alg direct --t-end 0.5
      $DFBA_PY benchmarks/external/check_dfba_growth.py"
  else
    echo "DFBA_PY not set: skipping the dfba package runs" >&2
  fi
}

stage_adaptive() {       # Table 4; manylp on the dfba scenarios (needs stage_external)
  run adaptive "set -e
    python benchmarks/bench_adaptive.py
    python benchmarks/bench_diauxie.py
    python benchmarks/external/run_manylp_same.py --scenario diauxie --E 32
    python benchmarks/external/run_manylp_same.py --scenario iJO1366 --E 8"
  run onethread "taskset -c 5 python benchmarks/external/run_manylp_1thread.py --scenario diauxie --E 32 && taskset -c 5 python benchmarks/external/run_manylp_1thread.py --scenario iJO1366 --E 8 && python benchmarks/external/summarize.py"
}

stage_report() {         # Figures 1-3, Tables S1-S3, in-text numbers; export to benchmarks/data
  run report "set -e
    python benchmarks/make_report.py
    python benchmarks/paper_tables.py --out results/report/paper_tables.md
    python benchmarks/derived_numbers.py
    python benchmarks/export_paper_data.py"
}

STAGES="workloads alt_optima sparsity dfba coldstart muode spatial solvers netlib size bunching external adaptive report"

case "${1:-list}" in
  list) for s in $STAGES; do grep -m1 "^stage_$s()" "$0" | sed "s/() {[ ]*#/:/"; done ;;
  all) for s in $STAGES; do "stage_$s"; done ;;
  *) for s in "$@"; do "stage_$s"; done ;;
esac
