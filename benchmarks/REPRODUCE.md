# Reproducing the paper

This repository holds what is needed to reproduce the results of the manylp paper:

- the scripts (`benchmarks/`),
- the BiGG and Netlib inputs (`benchmarks/inputs/`, with sources and checksums),
- the result summaries behind every table and figure (`benchmarks/data/`),
- the software environment and hardware (`benchmarks/env/`).

The 12-species gut community that most benchmarks simulate (gapseq models, Western diet and initial
abundances) is distributed with the µODE simulator (`examples/gut_western`); set `GUT_DIR` to that
directory, or `MUODE_DIR` to a µODE checkout.

## 1. Regenerate the tables, figures and numbers from the shipped results

This needs no solver runs and takes under a minute (NumPy and Matplotlib only).

```bash
python benchmarks/paper_tables.py --root benchmarks/data            # Tables S1-S3, Netlib statistics
python benchmarks/derived_numbers.py --root benchmarks/data         # in-text numbers, with their sources
MANYLP_RESULTS=benchmarks/data python benchmarks/make_report.py     # Figures 1-3 -> results/report/paper/
```

## 2. Re-run the experiments

### Software

```bash
conda env create -f benchmarks/env/environment.yml && conda activate manylp
pip install -e .
# baseline solvers (Gurobi, FICO Xpress, NVIDIA cuOpt, MPAX, OR-Tools, OSQP)
python -m venv --system-site-packages baselines
baselines/bin/pip install -r benchmarks/env/requirements-baselines.txt
```

Optional tools:

- **SoPlex 8.1** for the exact rational check: `conda create -n soplexenv -c conda-forge soplex`. Pass the binary as `SOPLEX=<env>/bin/soplex`.
- **dfba package** (Tourigny et al. 2020): `conda create -n dfbaenv -c conda-forge python=3.8 dfba cobra`. Pass its python as `DFBA_PY=<env>/bin/python`.
- **µODE** (branch `manylp-backend`): its `examples/gut_western` directory holds the gut community used by most benchmarks, and its manylp backend runs the end-to-end simulations of Table 3. Point to it with `MUODE_DIR`.

Gurobi and FICO Xpress need licences. The paper used Gurobi's size-limited pip licence and Xpress's community licence. With those, the native lexicographic check covers the gut models that fit within the licence limits.

### Hardware

The paper's timings come from one NVIDIA A100-PCIE-40GB and four Intel Xeon Platinum 8276L CPUs (`benchmarks/env/HARDWARE.md`). Stages tagged `[GPU]` in `reproduce.sh` need an NVIDIA GPU with CUDA 12.

### Running

```bash
bash benchmarks/reproduce.sh list              # stages
bash benchmarks/reproduce.sh workloads dfba    # some stages
bash benchmarks/reproduce.sh all               # everything (about a day on the hardware above)
```

Results go to `results/` and logs to `logs/`. Each stage writes the files listed below. `stage_report` then rebuilds the tables, figures and numbers from `results/` and exports the summaries to `benchmarks/data/`.

## 3. Where every result comes from

All paths below are relative to `results/`. The same files, after export, are under `benchmarks/data/`.

| Paper item | Stage | Result file(s) |
|---|---|---|
| Table 1, Table S1 (coherent workload, 185,088 LPs) | `solvers` | `solvers/*@workload_coherent.json` |
| Table S2 (snapshot workload, 23,808 LPs) | `solvers` | `solvers/*.json` without `@` |
| Coherent workload text (12–33×, 1–8% completed, 0–12% solved) | `solvers` | as Table 1; `derived_numbers.py` prints the ratios |
| Snapshot workload text (bunching 4,414 LP/s, pFBA-unique 4,860 vs 731 LP/s) | `solvers`, `bunching` | `solvers/bunching-p8.json`, `solvers/manylp-*.json`, `solvers/highs-lex-pfba-unique-p32.json` |
| Table 2, Figure 2 (alternative-optima envelope; 49 policies, 282,828 LPs) | `alt_optima`, `dfba` | `alt_optima/summary.json`, `alt_optima/envelope.npz`, `dfba/{cobra-glpk,highs-warm-x1,scipy-linprog}_fba_E1_T48.npz` |
| Exchange fluxes differ by up to 2.7 mmol gDW⁻¹ h⁻¹ | `solvers` | `exchange_dev_max` in `solvers/*@workload_coherent.json` |
| Agreement with HiGHS under pFBA-unique (4×10⁻¹⁵ gDW L⁻¹, 9×10⁻¹³ mM) | `muode` | `muode/{manylp-cpu,highs}-pfba-unique_E{1,16}_T48.npz`; `derived_numbers.py` |
| COBRApy + GLPK final biomass up to 1.4% | `muode` | `muode_v2/{legacy-cobra-glpk-j1,manylp-cpu-pfba-unique}_E1_T48.json`; `derived_numbers.py` |
| SoPlex exact rational check (200 LPs, 1.7×10⁻¹¹) | `external` | `external/soplex_exact.json` |
| FICO Xpress native lexicographic mode wrong on 29% | `external` | `external/lex_native_xpress_b1200_n8.json` (+ `_mo{1,3,5}`) |
| Table 3, ensembles (E = 1 to 1,024) | `muode` | `muode_v2/*.json`; the E = 16 legacy time is `muode/legacy-cobra-glpk-j12_E16_T48.json` (12 threads) |
| Table 3, spatial grids | `spatial` | `spatial/spatial_{8x8,16x16,32x32}.json` |
| 72 simplex solves, 30 critical regions, 23 s standalone (E = 1,024) | `dfba` | `dfba_v3/manylp-cuda_pfba-unique_E1024_T48.json` |
| Figure 1A (ensemble scaling; 94.6 M LPs, 76 solves, 682,000 LP/s) | `dfba` | `dfba_v3/manylp-*_pfba-unique_E*_T48.json`; baselines in `dfba_v2/` |
| Figure 1B (certification throughput) | `dfba` | `throughput/throughput.json` |
| Table 4, regime events (gut community and *E. coli* diauxie) | `adaptive` | `adaptive/adaptive_cpu.json`, `adaptive/diauxie.json` |
| Table 5 (cold start, atlas) | `coldstart` | `coldstart/coldstart_{cpu,cuda}.json` |
| Parameter sparsity (9.1% of columns, 87.1 → 11.6 MiB) | `sparsity` | `sparsity/sparsity.json` |
| Table 6 rows 1–2 (bunching) | `solvers`, `bunching` | `solvers/bunching-p8@workload_coherent.json`, `dfba_v3/{manylp-cpu,bunching-p32}_fba_E1024_T48.json` |
| Table 6 rows 3–5, dfba comparison text | `external`, `adaptive` | `external/manylp_{diauxie,iJO1366}_1thread.json`, `external/dfba_*.json`, `external/summary.md`; the growth rate 0.62 vs 0.70 h⁻¹ is printed by `external/check_dfba_growth.py` |
| Figure 3A (model size; 13 of 16 configurations) | `size` | `size/size.json`; `derived_numbers.py` |
| Figure 3B, Table S3, Netlib text (vs HiGHS and bunching) | `netlib`, `bunching` | `netlib/netlib_B1024.json`, `netlib_patch/netlib_B1024.json` (bnl1), `netlib_bunching/*.json`; `paper_tables.py` |
| Methods: reference cross-check on 400 LPs (1.7×10⁻¹¹) | `workloads` | `workload_coherent_reference_check.json` (snapshot: `solvers/reference_check.json`) |

## 4. Notes

- **The audit reference.** It is manylp's own certified answer for every LP (`make_reference.py`). It is checked independently in two ways:
  - against a fresh HiGHS solve per LP that shares no code with manylp, on 400 sampled LPs;
  - against SoPlex in exact rational arithmetic, on 200 LPs.
  
  The "0 (reference)" error of manylp in Table 1 holds by construction.
- **Extrapolated, not measured.** These Table 3 legacy times were not run:
  - E = 256 and E = 1,024: the number of members times the measured single-trajectory time (653.6 s);
  - the 16×16 and 32×32 grids: the 8×8 legacy rate of 8.84 LP/s applied to their LP counts.
- **Timings vary with the machine.** Absolute numbers depend on it, and the benchmark node was shared. Certified answers, simplex-solve counts and distinct bases are deterministic for a given input.
