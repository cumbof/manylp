# manylp

**Certified, batched solving of many small linear programs that share their constraint
matrix and differ only in their bounds, on GPUs and CPUs.**

manylp targets the LP workload of dynamic flux balance analysis (dFBA). That workload has
one LP per species × time step × ensemble member. All of them share the same stoichiometric
matrix and differ only in the exchange (uptake) bounds that the ODE state sets at each step.
The same structure appears in scenario analysis, stochastic programming, parameter sweeps
and model-predictive control. manylp is built for this regime, not for single large LPs.

- [How it works](#how-it-works)
- [Features](#features)
- [When to use it](#when-to-use-it)
- [Installation](#installation)
- [Quick start](#quick-start)
- [muODE integration](#muode-integration)
- [Reproducing the paper](#reproducing-the-paper)
- [Repository layout](#repository-layout)
- [Testing](#testing)
- [License](#license)

## How it works

**Certify, don't re-solve.** When only the bounds change, an optimal basis stays *dual
feasible*, because reduced costs do not depend on bounds. Within one basis the solution is
an affine function of the bounds:

```
z_B = h + T · (z_N − z_N,template)          one GEMV per LP, one GEMM per batch
```

manylp keeps a pool of certified optimal bases. It does not solve each LP. It evaluates and
certifies every member of a batch against the pool with one dense product and a bound check,
on the GPU or with fused multi-threaded CPU kernels.

Only a member that falls outside every cached critical region is re-solved. A warm-started
dual simplex (HiGHS) on the CPU handles it. The new basis is then immediately tested against
the rest of the batch.

## Features

- **Certified vertices.** Every returned point is a basic optimal solution: a vertex, not an
  ε-approximation.
  - manylp checks each point in its own double-precision arithmetic, independently of the
    solver that produced the basis. It checks primal feasibility and lexicographic dual
    feasibility at a tolerance of 1e-9.
  - These are numerical certificates, not interval or rational proofs. On a random sample
    they agree with exact rational optima (SoPlex) to 1.7e-11.
- **Unique fluxes.** A single basis certifies a lexicographic objective, such as max growth,
  then min ‖v‖₁, then a generic tie-break. The certificate also proves when the optimum is
  unique, so trajectories no longer depend on which vertex a solver happens to return.
- **Certified infeasibility.** Farkas rays are cached and checked in batch. The check allows
  for a rigorous floating-point error bound on the certificate itself.
- **Temporal and ensemble coherence.** Over 48 h of a dFBA trajectory, a whole community
  visits only a few dozen critical regions. More than 99.9% of the LPs are certified from
  the cache.
- **Persistent basis atlas.** Certified bases can be saved per model and reused by later
  runs with any diet or ensemble. The atlas stores certificates, not answers.
  - Under a unique flux rule such as pFBA-unique, reusing it never changes a result.
  - Under plain FBA it can change which of several equally optimal vertices is returned.
- **Alternative-optima envelopes.** `manylp.envelope` integrates several *selection
  policies* in lockstep, such as random directions on the optimal face or extreme secretion
  of chosen metabolites. Each policy gives a unique, certified trajectory. Their spread shows
  how much of a dFBA prediction the model leaves undetermined.

## When to use it

manylp pays one simplex solve per critical region, not per LP. It wins when a batch of LPs
that share a matrix and an objective has many more members than distinct optimal bases.
Typical cases are time integration, ensembles, spatial grids, parameter sweeps and scenario
analysis.

On recorded genome-scale dFBA workloads, manylp:

- is 12–33× faster than HiGHS, Gurobi, Xpress, GLPK and GLOP on a CPU;
- is 6.8× faster than classical bunching;
- reaches 682,000 LP/s on one A100 for large ensembles.

When nearly every LP needs its own basis, a conventional simplex code or plain bunching is
faster. See the paper for the full picture.

### Relation to prior work

Checking one basis against many right-hand sides is a classical technique. Stochastic
programming calls it *bunching* (Wets 1983; Haugland & Wallace 1988; Kall & Wallace 1994,
§3.10). manylp adds three things to it:

- **Batched and cheap.** Each basis gets a precomputed affine law over only the parameters
  that move. A whole batch is checked with one GEMM per basis, on a GPU or with fused CPU
  kernels.
- **Certified.** Lexicographic certificates prove uniqueness, and Farkas certificates prove
  infeasibility.
- **Managed.** Representatives are selected for repair, new bases are propagated across the
  batch, a yield guard limits wasted repairs, and the atlas persists bases across runs.

`benchmarks/bunching.py` implements classical bunching as a baseline.

## Installation

Python 3.10 or newer is required.

```bash
pip install -e .                 # core: numpy, scipy, highspy
pip install -e ".[gpu]"          # + CuPy for CUDA 12 devices
pip install -e ".[cobra]"        # + COBRApy, to load SBML / COBRA models
pip install numba                # fused multi-threaded CPU kernels (recommended)
```

| Extra | Adds | Needed for |
|---|---|---|
| `gpu` | `cupy-cuda12x` | `device="cuda"` |
| `cobra` | `cobra` | `FBAModel.from_cobra` |
| `pdhg` | `jax` | the batched PDHG baseline in `manylp/pdhg.py` |
| `dev` | `pytest` | the test suite |

## Quick start

### Batches of bounds-parametric LPs

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

### FBA front-end

Supported modes are FBA, pFBA, pFBA-unique and lexicographic.

```python
from manylp.fba import FBAModel, compile_fba

prob = compile_fba(FBAModel.from_cobra(cobra_model), mode="pfba-unique")
g = solver.register_group(prob.lp, out_z=prob.out_z(prob.model.exchanges))

Lp, Up = prob.param_bounds(exchange_lower_bounds)      # (B, k)
sol = solver.solve_batch(g, Lp, Up)
growth = sol.objective[:, 0]
fluxes = prob.fluxes(sol.z, prob.model.exchanges)
```

### GPU pipelines

Pass CuPy arrays and `device_out=True` to keep a whole pipeline on the device.

### Basis atlas

```python
solver.save_atlas(g, "iJO1366.atlas")    # after a run
solver.load_atlas(g, "iJO1366.atlas")    # before the next one, with any diet or ensemble
```

## muODE integration

The `manylp-backend` branch of muODE adds a `SolverBackend` layer:

```python
from muode import DynamicFBA
from muode.backends import make_backend

engine = DynamicFBA(t_end=48, dt=0.1)
res = engine.run(community, diet, kinetics, backend=make_backend("manylp-gpu"))
runs = engine.run_ensemble(
    community, diets, kinetics,
    backend=make_backend("manylp-gpu", atlas_dir="~/.muode/atlas"),
)
```

The default backend reproduces historical muODE outputs byte-for-byte.

## Reproducing the paper

Every experiment in the paper is a script in `benchmarks/`. The `benchmarks/run_*.sh` files
chain them in the order they were run. Raw results go to `results/`, which is not tracked.

```bash
python benchmarks/make_report.py         # tables and figures  -> results/report/
python benchmarks/make_html_report.py    # HTML report         -> results/report/manylp_report.html
python benchmarks/export_paper_data.py   # JSON summaries      -> benchmarks/data/
```

`benchmarks/data/` holds the exported result summaries and figures behind every table and figure
of the paper.

## Repository layout

| Path | Contents |
|---|---|
| `manylp/lp.py` | lexicographic, bounds-parametric LP groups (z-space form) |
| `manylp/backend.py` | array backend: NumPy on the CPU, CuPy on a CUDA device |
| `manylp/basis.py` | basis certificates, affine laws, Farkas certificates, certification |
| `manylp/kernels.py`, `manylp/cpu_kernels.py` | fused CUDA and Numba certification kernels |
| `manylp/pool.py` | per-group cache of certified bases and Farkas certificates |
| `manylp/solver.py` | the certify-and-repair batched solver and atlas I/O |
| `manylp/repair.py` | warm-started lexicographic HiGHS repair |
| `manylp/fba.py`, `manylp/dfba.py`, `manylp/envelope.py` | FBA front-end, batched dFBA driver, envelopes |
| `manylp/adaptive.py` | adaptive (RK45) batched dFBA integration |
| `manylp/reference.py` | independent reference solver and multi-process HiGHS baseline |
| `manylp/pdhg.py` | batched restarted PDHG (first-order GPU baseline) |
| `manylp/generators.py` | synthetic batched-LP workloads for tests and benchmarks |
| `benchmarks/` | every experiment in the paper |
| `benchmarks/bunching.py` | classical bunching baseline (per-LP and batched) |
| `benchmarks/external/` | comparisons with the `dfba` package, surfinFBA, SoPlex exact and native Gurobi/Xpress lexicographic modes |
| `benchmarks/data/` | the exported result summaries and figures behind every table and figure |
| `tests/` | correctness against the independent reference, on CPU and GPU |

## Testing

```bash
pip install -e ".[dev]"
pytest                   # CPU; GPU variants run automatically when CuPy sees a device
```

## License

manylp is released under the [MIT License](LICENSE).
