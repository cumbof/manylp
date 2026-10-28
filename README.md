# manylp

**Certified, batched solving of many small linear programs that share their constraint matrix
and differ only in their bounds, on GPUs and CPUs, with an application to genome-scale dynamic
flux balance analysis (dFBA).**

This README accompanies the manuscript

> Cumbo F. (2026). *manylp: reproducible, exact, and batched linear programming for
> genome-scale dynamic flux balance analysis.* arXiv:2610.14566.
> https://arxiv.org/abs/2610.14566

| | |
|---|---|
| Source code | https://github.com/cumbof/manylp |
| Python package | https://pypi.org/project/manylp/ (version 0.1.0) |
| Revision used for the paper | `main` at commit `6f6d353dce12cbdc83c778b18493da3dc550fda5` |
| Companion simulator | µODE, https://github.com/cumbof/muODE (branch `manylp-backend`) |
| License | MIT |

The library code (`manylp/`, `tests/`, `pyproject.toml`) is identical in the PyPI release 0.1.0
(tag `v0.1.0`) and in commit `6f6d353`. Commits after the release add only reproduction material
under `benchmarks/` and documentation. To reproduce the paper, clone the repository at that commit
or later.

---

## Contents

1. [Description](#1-description)
2. [Dataset information](#2-dataset-information)
3. [Code information](#3-code-information)
4. [Usage instructions](#4-usage-instructions)
5. [Requirements](#5-requirements)
6. [Methodology](#6-methodology)
7. [Citations](#7-citations)
8. [License and contribution guidelines](#8-license-and-contribution-guidelines)

---

## 1. Description

### The problem

In dynamic flux balance analysis, every species of a microbial community solves a linear program
(LP) at every time step: maximise growth subject to the stoichiometric matrix *S* and to uptake
bounds set by the current metabolite concentrations. Ensembles, spatial grids and parameter sweeps
multiply this count. One simulation can need from 10⁵ to 10⁸ LPs, all with the same matrix and
objective, differing only in a few exchange bounds.

Solving each LP separately with a general-purpose solver has two costs:

- it is slow;
- it is not reproducible. FBA optima are typically degenerate, so different solvers, versions or
  warm starts return different optimal flux vectors, which then feed back into the ODE and change
  the predicted community.

### What manylp does

manylp is a Python library that solves such batches by **certify-and-repair**:

- When only the bounds change, an optimal basis stays dual feasible, and within that basis the
  solution is an affine function of the bounds.
- manylp keeps a pool of certified optimal bases. It certifies every member of a batch against the
  pool with one dense matrix product per basis (GPU or fused multi-threaded CPU kernels).
- Only members that fall outside every cached critical region are re-solved, by a warm-started dual
  simplex (HiGHS). The new basis is then immediately tested on the rest of the batch.

Features:

- **Certified vertices.** Every returned point is a basic optimal solution, checked in double
  precision for primal feasibility and lexicographic dual feasibility (tolerance 1e-9).
- **Unique fluxes (pFBA-unique).** A lexicographic objective — maximum growth, then minimum
  ‖v‖₁, then a generic tie-break — whose certificate also proves that the optimum is unique. dFBA
  trajectories therefore no longer depend on the solver.
- **Certified infeasibility.** Farkas certificates are cached and checked in batch.
- **Basis atlas.** Certified bases can be saved per model and reused by later runs.
- **Alternative-optima envelopes.** Several flux-selection policies can be integrated in lockstep
  to measure how much of a dFBA prediction the model leaves undetermined.
- **Integration with µODE.** manylp is a drop-in backend for the µODE community dFBA simulator.

The repository also contains every script, input (or pointer to it) and result summary behind the
tables and figures of the paper.

---

## 2. Dataset information

No new experimental data were generated. The study uses public metabolic models, a public LP test
set, and LP workloads that the scripts record from simulations.

| Dataset | Location | Content | Source |
|---|---|---|---|
| BiGG genome-scale models | `benchmarks/inputs/bigg/` | 6 SBML models: e_coli_core, iYO844, iMM904, iJO1366, iML1515, Recon3D (`.xml.gz`) | BiGG Models (King et al. 2016), `https://bigg.ucsd.edu/static/models/<id>.xml.gz` |
| Netlib LP problems | `benchmarks/inputs/netlib/` | 44 Netlib LP problems in MPS format | Netlib LP collection (Gay 1985), `https://www.netlib.org/lp/data/`, decompressed with Netlib's `emps.c` |
| 12-species human gut community | µODE repository, `examples/gut_western/` | 12 gapseq genome-scale models (`gems/*.xml.gz`, provenance in `gems/PROVENANCE.md`), Western diet (`gems/western_gut_modelseed.csv`), initial relative abundances (`abundance.tsv`) | https://github.com/cumbof/muODE (MIT); models reconstructed with gapseq (Zimmermann et al. 2021) |
| Recorded LP workloads | generated into `results/` (not shipped) | snapshot workload (23,808 LPs) and coherent workload (185,088 LPs), with a certified reference answer for every LP | `benchmarks/record_workload.py`, `benchmarks/make_reference.py` |
| Synthetic LP batches | generated at run time | random bounds-parametric LP groups for tests | `manylp/generators.py` |
| Result summaries | `benchmarks/data/` | JSON summaries and NPZ trajectories behind every table and figure, plus Figures 1–3 (`figures/figure{1,2,3}.{png,pdf}`) | exported from `results/` by `benchmarks/export_paper_data.py` |
| JSON replicas | `benchmarks/data/json/` | the same data as every NPZ array above, in plain JSON | `benchmarks/export_json_replicas.py` |

SHA-256 checksums of the BiGG and Netlib files are in `benchmarks/inputs/README.md`.

### Human- and machine-readable formats

All data are available in plain-text formats:

- result summaries: JSON;
- SBML models: XML (gzip-compressed);
- Netlib problems: MPS (plain text);
- trajectory arrays (`.npz`): replicated in JSON under `benchmarks/data/json/`, at the same
  relative path. Each replica holds every array as nested lists, with its dtype, shape, axis
  names, units and labels (times, species, metabolites, policies or members). Floats are written
  in their shortest round-trip form, so the values are exact. The 49-policy envelope and the two
  16-member µODE ensembles are split into an `index.json` plus one file per policy or member.

The scripts still read the original files. `python benchmarks/export_json_replicas.py` rebuilds
the replicas and checks each one against its source (`--check` only checks); they match exactly.

### Layout of `benchmarks/data/`

Each subdirectory corresponds to one stage of `benchmarks/reproduce.sh` (Section 4.3):

| Directory | Paper item |
|---|---|
| `solvers/` | Table 1, Tables S1–S2 (solver comparison on the two recorded workloads) |
| `alt_optima/`, `dfba/` | Table 2, Figure 2 (alternative-optima envelope; plain-FBA trajectories) |
| `dfba_v2/`, `dfba_v3/`, `throughput/` | Figure 1 (ensemble scaling, certification throughput) |
| `muode/`, `muode_v2/`, `spatial/` | Table 3 (µODE end-to-end ensembles and spatial grids), agreement with HiGHS |
| `adaptive/` | Table 4 (regime events) |
| `coldstart/` | Table 5 (cold start and basis atlas) |
| `external/`, `netlib_bunching/` | Table 6 (classical bunching, `dfba` package, SoPlex exact check, native lexicographic modes) |
| `size/` | Figure 3A (model size) |
| `netlib/`, `netlib_patch/` | Figure 3B, Table S3 (Netlib under right-hand-side uncertainty) |
| `sparsity/` | parameter sparsity of the affine laws |
| `workload_coherent_reference_check.json` | independent cross-check of the reference answers (Methods) |
| `json/` | JSON replicas of the NPZ arrays above |

The full mapping from each paper item to its result file is in `benchmarks/REPRODUCE.md`
(Section 3).

---

## 3. Code information

Language: Python (≥ 3.10), with CUDA kernels compiled at run time through CuPy and CPU kernels
compiled through Numba. Shell scripts (Bash) orchestrate the benchmarks.

### Library (`manylp/`)

| Module | Contents |
|---|---|
| `lp.py` | `LexLP`: lexicographic, bounds-parametric LP groups |
| `backend.py` | array backend: NumPy on the CPU, CuPy on a CUDA device |
| `basis.py` | basis certificates, affine laws, Farkas certificates, certification |
| `kernels.py`, `cpu_kernels.py` | fused CUDA and Numba certification kernels |
| `pool.py` | per-group cache of certified bases and Farkas certificates |
| `solver.py` | `BatchLPSolver`: the certify-and-repair batched solver and atlas I/O |
| `repair.py` | warm-started lexicographic HiGHS repair |
| `fba.py` | FBA front-end: FBA, pFBA, pFBA-unique and lexicographic modes |
| `dfba.py`, `adaptive.py` | batched dFBA driver (fixed step and adaptive RK45) |
| `envelope.py` | alternative-optima envelopes |
| `reference.py` | independent reference solver and multi-process HiGHS baseline |
| `pdhg.py` | batched restarted PDHG (first-order GPU baseline) |
| `generators.py` | synthetic batched-LP workloads |

### Tests (`tests/`)

`test_core.py`, `test_fba.py` and `test_extras.py` check correctness against the independent
reference solver, on the CPU and, when a CUDA device is visible, on the GPU.

### Benchmarks and reproduction (`benchmarks/`)

| File(s) | Purpose |
|---|---|
| `REPRODUCE.md` | reproduction guide: setup, stages, item → result-file map |
| `reproduce.sh` | runs the experiments stage by stage |
| `record_workload.py`, `make_reference.py` | record LP workloads from the gut community and build/cross-check the reference |
| `bench_solvers.py`, `run_compare*.sh` | solver comparison (Table 1, Tables S1–S2) |
| `alt_optima.py` | alternative-optima envelope (Table 2, Figure 2) |
| `bench_dfba.py`, `bench_throughput.py` | dFBA ensembles and certification throughput (Figure 1) |
| `bench_muode.py`, `bench_spatial.py` | µODE end-to-end runs (Table 3) |
| `bench_adaptive.py`, `bench_diauxie.py` | regime events (Table 4) |
| `bench_coldstart.py` | cold start and atlas (Table 5) |
| `bunching.py`, `run_bunching*.sh` | classical bunching baseline (Table 6) |
| `external/` | `dfba` package, surfinFBA, SoPlex exact check, native Gurobi/Xpress lexicographic modes |
| `bench_size.py`, `bench_netlib.py` | model-size and Netlib benchmarks (Figure 3, Table S3) |
| `bench_sparsity.py` | parameter sparsity |
| `make_report.py` | Figures 1–3 |
| `paper_tables.py` | Tables S1–S3 and Netlib statistics |
| `derived_numbers.py` | every number quoted in the text, with its source file |
| `export_paper_data.py` | exports the result summaries to `benchmarks/data/` |
| `export_json_replicas.py` | JSON replicas of the NPZ arrays, checked against their sources |
| `env/` | pinned environments (`environment.yml`, `requirements-baselines.txt`), `activate_baselines.sh`, `HARDWARE.md` |
| `inputs/` | BiGG and Netlib inputs with sources and checksums |

The scripts `bench_scaling.py`, `run_scaling.sh`, `hero_phase.py`, `hero_figure.py`, `profiling/`
and `run_suite_v*.sh` / `run_after_*.sh` are development and exploratory material; they are not
needed to reproduce the paper.

---

## 4. Usage instructions

### 4.1 Install

From PyPI:

```bash
pip install manylp                  # core: numpy, scipy, highspy
pip install "manylp[gpu]"           # + CuPy for CUDA 12 devices
pip install "manylp[cobra]"         # + COBRApy, to load SBML / COBRA models
pip install numba                   # fused multi-threaded CPU kernels (recommended)
```

From source (needed for the benchmarks):

```bash
git clone https://github.com/cumbof/manylp.git
cd manylp
git checkout 6f6d353dce12cbdc83c778b18493da3dc550fda5     # revision used for the paper
pip install -e ".[cobra,dev]"
pytest                                                    # GPU tests run when CuPy sees a device
```

### 4.2 Use the library

Batches of bounds-parametric LPs:

```python
import numpy as np
from manylp import BatchLPSolver, LexLP

lp = LexLP(A=S, objectives=c, col_lb=lb, col_ub=ub, param_cols=exchange_idx)
solver = BatchLPSolver(device="cuda")                  # or "cpu"
g = solver.register_group(lp)                          # S and c stay resident; only bounds stream in

sol = solver.solve_batch(g, L, U)                      # L, U: (B, p) parametric bounds
sol = solver.solve_batch(g, L2, U2, warm_start=sol)    # next time step

sol.objective, sol.z, sol.status, sol.certified, sol.unique
```

FBA front-end:

```python
import cobra
from manylp import BatchLPSolver
from manylp.fba import FBAModel, compile_fba

cobra_model = cobra.io.read_sbml_model("benchmarks/inputs/bigg/iJO1366.xml.gz")
prob = compile_fba(FBAModel.from_cobra(cobra_model), mode="pfba-unique")
solver = BatchLPSolver(device="cpu")
g = solver.register_group(prob.lp, out_z=prob.out_z(prob.model.exchanges))

Lp, Up = prob.param_bounds(exchange_lower_bounds)      # (B, k) exchange lower bounds
sol = solver.solve_batch(g, Lp, Up)
growth = sol.objective[:, 0]
fluxes = prob.fluxes(sol.z, prob.model.exchanges)
```

Basis atlas:

```python
solver.save_atlas(g, "iJO1366.atlas")    # after a run
solver.load_atlas(g, "iJO1366.atlas")    # before the next one, with any diet or ensemble
```

Inside µODE (branch `manylp-backend`):

```python
from muode import DynamicFBA
from muode.backends import make_backend

engine = DynamicFBA(t_end=48, dt=0.1)
res = engine.run(community, diet, kinetics, backend=make_backend("manylp-gpu"))
```

### 4.3 Reproduce the paper

**(a) Regenerate the tables, figures and numbers from the shipped results.** No solver runs; under
a minute; needs only NumPy and Matplotlib.

```bash
python benchmarks/paper_tables.py --root benchmarks/data            # Tables S1-S3, Netlib statistics
python benchmarks/derived_numbers.py --root benchmarks/data         # in-text numbers, with their sources
MANYLP_RESULTS=benchmarks/data python benchmarks/make_report.py     # Figures 1-3 -> results/report/paper/
```

**(b) Re-run the experiments.**

```bash
conda env create -f benchmarks/env/environment.yml && conda activate manylp
pip install -e .
python -m venv --system-site-packages baselines                     # baseline solvers
baselines/bin/pip install -r benchmarks/env/requirements-baselines.txt

git clone https://github.com/cumbof/muODE.git ../muODE              # gut community + µODE backend
git -C ../muODE checkout manylp-backend

bash benchmarks/reproduce.sh list              # list the stages and what each produces
bash benchmarks/reproduce.sh workloads dfba    # run some stages
bash benchmarks/reproduce.sh all               # everything (about a day on the hardware below)
```

Stages: `workloads`, `alt_optima`, `sparsity`, `dfba`, `coldstart`, `muode`, `spatial`, `solvers`,
`netlib`, `size`, `bunching`, `external`, `adaptive`, `report`. Results go to `results/`, logs to
`logs/`. The final `report` stage rebuilds all tables, figures and numbers from `results/` and
exports them to `benchmarks/data/`.

Optional environment variables:

| Variable | Meaning | Default |
|---|---|---|
| `MUODE_DIR` | µODE checkout (branch `manylp-backend`) | `../muODE` |
| `GUT_DIR` | gut community directory | `$MUODE_DIR/examples/gut_western` |
| `BASELINES_VENV` | virtual environment with the baseline solvers | `./baselines` |
| `SOPLEX` | SoPlex 8.1 binary for the exact rational check | `soplex` on `PATH` |
| `DFBA_PY` | Python of an environment with the `dfba` package | unset: those runs are skipped |
| `MANYLP_RESULTS` | results directory read by `make_report.py` | `results` |

---

## 5. Requirements

### Library

- Python ≥ 3.10
- `numpy` ≥ 1.23, `scipy` ≥ 1.10, `highspy` ≥ 1.7 (required)
- Optional: `cupy-cuda12x` ≥ 13 (GPU, extra `gpu`), `cobra` ≥ 0.29 (SBML/COBRA models, extra
  `cobra`), `jax` ≥ 0.4.30 (PDHG baseline, extra `pdhg`), `numba` (fused CPU kernels),
  `pytest` ≥ 7.0 (tests, extra `dev`)
- Operating system: Linux was used for all results; the CPU path is pure Python/NumPy/Numba.
- GPU: an NVIDIA GPU with CUDA 12 for `device="cuda"`.

### Paper experiments

- The exact environment of the paper: `benchmarks/env/environment.yml` (conda; Python 3.11.16,
  NumPy 2.4.6, SciPy 1.17.1, highspy 1.15.1, COBRApy 0.30.0, GLPK 5.0, CuPy 14.2.0, Numba 0.67.0,
  JAX 0.10.2, Matplotlib 3.11.2).
- Baseline solvers: `benchmarks/env/requirements-baselines.txt` (Gurobi 13.0.3, FICO Xpress 9.9.1,
  NVIDIA cuOpt 26.8.0, MPAX 0.2.4, OR-Tools 9.15, OSQP 1.1.3).
  - Gurobi and FICO Xpress need licences. The paper used Gurobi's size-limited pip licence and
    Xpress's community licence.
- Optional external tools:
  - SoPlex 8.1: `conda create -n soplexenv -c conda-forge soplex`.
  - `dfba` package: `conda create -n dfbaenv -c conda-forge python=3.8 dfba cobra`.
  - µODE, branch `manylp-backend`.
- Hardware of the reported timings (`benchmarks/env/HARDWARE.md`):
  - one NVIDIA A100-PCIE-40GB;
  - 4 × Intel Xeon Platinum 8276L @ 2.20 GHz;
  - Linux 5.14 (RHEL 9).

  The node was shared, so absolute timings may vary by a few percent.

---

## 6. Methodology

### 6.1 Solver

1. **Compilation.** A metabolic model is compiled into a lexicographic, bounds-parametric LP
   group (`compile_fba`). Only the exchange bounds are parameters.
2. **Certification.** For each cached optimal basis, manylp precomputes an affine law
   z_B = h + T·(z_N − z_N,template) over the parameters that move. A batch is evaluated with one
   GEMM per basis and accepted where the result is primal feasible. Dual feasibility does not
   depend on the bounds and is certified once per basis, lexicographically.
3. **Infeasibility.** Members not certified as optimal are checked against cached Farkas rays,
   with a rigorous floating-point error bound.
4. **Repair.** Representatives of the remaining members are re-solved with a warm-started
   lexicographic dual simplex (HiGHS). The new bases are added to the pool and tested on the rest
   of the batch; a yield guard limits wasted repairs.
5. **Persistence (optional).** Certified bases are saved to a per-model atlas.

### 6.2 Experiments

1. **Workloads.** LP workloads are recorded from 48 h dFBA simulations of the 12-species gut
   community on the Western diet (`record_workload.py`):
   - a snapshot workload (23,808 LPs);
   - a temporally coherent workload (64 ensemble members × every step of 24 h; 185,088 LPs).
2. **Reference.** Every LP gets a certified reference answer (`make_reference.py`). The reference
   is checked independently in two ways:
   - against a fresh HiGHS solve that shares no code with manylp, on 400 sampled LPs;
   - against SoPlex in exact rational arithmetic, on 200 LPs.
3. **Accuracy.** Each solver is scored on the same LPs for:
   - throughput (overall and steady state);
   - fraction completed and solved;
   - status agreement;
   - objective error;
   - infeasibility;
   - deviation of the exchange fluxes from the reference.
4. **End-to-end dFBA.** manylp is run as the µODE backend against the original COBRApy + GLPK
   backend and against HiGHS, for ensembles of 1 to 1,024 members and for 8×8 to 32×32 spatial
   grids.
5. **Alternative optima.** 49 flux-selection policies are integrated in lockstep (282,828 LPs) to
   measure the spread of community predictions that plain FBA leaves undetermined.
6. **Generality and limits.** The method is also tested on:
   - BiGG models of increasing size;
   - 44 Netlib problems under right-hand-side uncertainty;
   - classical bunching, the `dfba` package and native lexicographic modes of commercial solvers.
7. **Reporting.** Tables, figures and all quoted numbers are generated by scripts
   (`make_report.py`, `paper_tables.py`, `derived_numbers.py`) from the result files, never typed
   by hand.

### 6.3 Extrapolations

Some legacy (COBRApy + GLPK) times in Table 3 are extrapolated rather than measured:

- E = 256 and E = 1,024: the number of members × the measured single-trajectory time;
- the 16×16 and 32×32 grids: the measured 8×8 rate applied to their LP counts.

Certified answers, simplex-solve counts and distinct bases are deterministic for a given input;
timings depend on the machine.

---

## 7. Citations

If you use manylp, please cite:

> Cumbo F. (2026). manylp: reproducible, exact, and batched linear programming for genome-scale
> dynamic flux balance analysis. arXiv:2610.14566. https://arxiv.org/abs/2610.14566

```bibtex
@misc{cumbo2026manylp,
  author        = {Cumbo, Fabio},
  title         = {manylp: reproducible, exact, and batched linear programming for genome-scale
                   dynamic flux balance analysis},
  year          = {2026},
  eprint        = {2610.14566},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2610.14566}
}
```

Datasets and software used:

- King Z.A., Lu J., Dräger A., Miller P., Federowicz S., Lerman J.A., Ebrahim A., Palsson B.O.,
  Lewis N.E. (2016). BiGG Models: a platform for integrating, standardizing and sharing
  genome-scale models. *Nucleic Acids Research* 44(D1):D515–D522. https://doi.org/10.1093/nar/gkv1049
- Gay D.M. (1985). Electronic mail distribution of linear programming test problems.
  *Mathematical Programming Society COAL Newsletter* 13:10–12.
- Zimmermann J., Kaleta C., Waschina S. (2021). gapseq: informed prediction of bacterial metabolic
  pathways and reconstruction of accurate metabolic models. *Genome Biology* 22:81.
  https://doi.org/10.1186/s13059-021-02295-1
- Huangfu Q., Hall J.A.J. (2018). Parallelizing the dual revised simplex method.
  *Mathematical Programming Computation* 10(1):119–142. https://doi.org/10.1007/s12532-017-0130-5
- Ebrahim A., Lerman J.A., Palsson B.O., Hyduke D.R. (2013). COBRApy: COnstraints-Based
  Reconstruction and Analysis for Python. *BMC Systems Biology* 7:74.
  https://doi.org/10.1186/1752-0509-7-74
- µODE: https://github.com/cumbof/muODE (MIT License).

---

## 8. License and contribution guidelines

### License

manylp is released under the **MIT License** (`LICENSE`, © 2026 Fabio Cumbo). The redistributed
inputs keep the terms of their sources:

- BiGG Models for `benchmarks/inputs/bigg/`;
- the Netlib LP collection for `benchmarks/inputs/netlib/`;
- µODE (MIT) for the gut community.

### Contributing

Contributions are welcome through GitHub:

- **Bugs and questions.** Open an issue at https://github.com/cumbof/manylp/issues. Include:
  - the manylp version (`python -c "import manylp; print(manylp.__version__)"`);
  - the device (CPU or GPU);
  - a minimal example.
- **Code changes.** Open a pull request against `main`.
  - Keep each change focused.
  - Add or update tests in `tests/`.
  - Make sure `pytest` passes on the CPU, and on a GPU if the change touches `kernels.py`,
    `backend.py` or the CUDA path.
- **Releases.** Releases are published to PyPI automatically from GitHub releases
  (`.github/workflows/publish.yml`).
