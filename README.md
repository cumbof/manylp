# manylp

**Exact, batched solving of many small linear programs that share their constraint
matrix and differ only in their bounds** — on GPUs and CPUs.

This is the LP workload of dynamic flux balance analysis (dFBA): one LP per species ×
time step × ensemble member, all with the same stoichiometric matrix, differing only in
the exchange (uptake) bounds that the ODE state sets each step. It is also the inner
loop of scenario analysis, stochastic programming, parameter sweeps, and model-predictive
control. manylp is built for that regime, not for single large LPs.

## The idea: certify, don't re-solve

When only bounds change, an optimal basis stays **dual feasible**, because reduced costs
do not depend on bounds. Within one basis the solution is an **affine function of the
bounds**:

```
z_B = h + T · (z_N − z_N,template)          (one GEMV per LP, one GEMM per batch)
```

So instead of solving each LP, manylp keeps a pool of certified optimal bases. It
*evaluates and certifies* every member of a batch against them in one dense product and
a bound check, on the GPU or with fused multi-threaded CPU kernels. Only the rare member
that falls outside every cached critical region is re-solved, by a warm-started dual
simplex (HiGHS) on the CPU. Its new basis is then immediately propagated to the rest of
the batch.

- **Exact.** Every returned point is a basic optimal solution (a vertex, not an
  ε-approximation), certified by manylp's own double-precision arithmetic: primal
  feasibility plus lexicographic dual feasibility at a tolerance of 1e-9, checked
  independently of the solver that produced the basis. These are numerical certificates,
  not rigorous interval or rational proofs; on a random sample they agree with exact
  rational optima (SoPlex) to 1.7e-11.
- **Unique fluxes.** Lexicographic objectives (e.g. max growth → min ‖v‖₁ → generic
  tie-break) are certified by a *single* basis. The certificate also proves when the
  optimum is unique, so trajectories no longer depend on which vertex a solver happens
  to return.
- **Infeasibility certified too.** Farkas rays are cached and checked in batch; the check
  accounts for a floating-point error bound of the certificate itself.
- **Temporal and ensemble coherence.** In a dFBA trajectory, a whole community visits
  only a few dozen critical regions over 48 h. More than 99.9% of LPs are certified from
  cache.
- **Persistent basis atlas.** Certified bases can be saved per model and reused by later
  runs with *any* diet or ensemble. Certificates, not answers, are stored, so under a
  unique flux rule (e.g. pFBA-unique) reuse can never change a result; under plain FBA it
  can change which of several equally optimal vertices is returned.

### Relation to prior work

Checking one basis against many right-hand sides is classical: stochastic programming
calls it *bunching* (Wets 1983; Haugland & Wallace 1988; Kall & Wallace 1994, §3.10).
manylp makes it batched, cheap (a precomputed affine law over only the parameters that
move, evaluated as one GEMM per basis on a GPU or with fused CPU kernels), and certified
(lexicographic uniqueness, Farkas infeasibility), with representative selection,
propagation of new bases, a yield guard and a persistent atlas. `benchmarks/bunching.py`
implements classical bunching as a baseline.

### When to use it

manylp pays one simplex solve per *critical region*, not per LP, so it wins when a batch of
LPs sharing a matrix and objective has many more members than distinct optimal bases: time
integration, ensembles, spatial grids, parameter sweeps, scenario analysis. On recorded
genome-scale dFBA workloads it is 12–33× faster than HiGHS, Gurobi, Xpress, GLPK and GLOP on
a CPU, 6.8× faster than classical bunching, and reaches 682,000 LP/s on one A100 for large
ensembles. When nearly every LP needs its own basis, a conventional simplex code (or plain
bunching) is faster; see the paper for the full picture.

## Install

```bash
pip install -e .                 # numpy, scipy, highspy
pip install -e ".[gpu]"          # + cupy (CUDA 12)
pip install numba                # fused multi-threaded CPU kernels (recommended)
```

## Quick start

```python
import numpy as np
from manylp import BatchLPSolver, LexLP

lp = LexLP(A=S, objectives=c, col_lb=lb, col_ub=ub, param_cols=exchange_idx)
solver = BatchLPSolver(device="cuda")            # or "cpu"
g = solver.register_group(lp)                    # S, c resident; only bounds stream in

sol = solver.solve_batch(g, L, U)                # L, U: (B, p) parametric bounds
sol = solver.solve_batch(g, L2, U2, warm_start=sol)   # next time step
sol.objective, sol.z, sol.status, sol.certified, sol.unique
```

FBA front-end (FBA / pFBA / pFBA-unique / lexicographic):

```python
from manylp.fba import FBAModel, compile_fba
prob = compile_fba(FBAModel.from_cobra(cobra_model), mode="pfba-unique")
g = solver.register_group(prob.lp, out_z=prob.out_z(prob.model.exchanges))
Lp, Up = prob.param_bounds(exchange_lower_bounds)          # (B, k)
sol = solver.solve_batch(g, Lp, Up)
growth, fluxes = sol.objective[:, 0], prob.fluxes(sol.z, prob.model.exchanges)
```

On a GPU, pass CuPy arrays and `device_out=True` to keep a whole pipeline on the device.

### Alternative-optima envelopes

FBA fixes the growth rate, not the fluxes. `manylp.envelope` integrates a set of
*selection policies* (random directions on the optimal face, extreme secretion of chosen
metabolites) in lockstep. Each policy gives a unique, certified trajectory, and their
spread shows how much of a dFBA prediction the model leaves undetermined.

## muODE integration

muODE (branch `manylp-backend`) gains a `SolverBackend` layer:

```python
from muode import DynamicFBA
from muode.backends import make_backend

engine = DynamicFBA(t_end=48, dt=0.1)
res  = engine.run(community, diet, kinetics, backend=make_backend("manylp-gpu"))
runs = engine.run_ensemble(community, diets, kinetics, backend=make_backend(
          "manylp-gpu", atlas_dir="~/.muode/atlas"))
```

The default backend reproduces historical muODE outputs byte-for-byte.

## Repository layout

| Path | Contents |
|---|---|
| `manylp/lp.py` | lexicographic, bounds-parametric LP groups (z-space form) |
| `manylp/basis.py` | basis certificates, affine laws, Farkas certificates, certification |
| `manylp/kernels.py`, `manylp/cpu_kernels.py` | fused CUDA / Numba certification kernels |
| `manylp/solver.py` | the certify-and-repair batched solver, atlas I/O |
| `manylp/repair.py` | warm-started lexicographic HiGHS repair |
| `manylp/fba.py`, `manylp/dfba.py`, `manylp/envelope.py` | FBA front-end, batched dFBA driver, envelopes |
| `manylp/reference.py` | independent reference solver and multi-process HiGHS baseline |
| `manylp/pdhg.py` | batched restarted PDHG (first-order GPU baseline) |
| `benchmarks/` | every experiment in the paper (see `benchmarks/run_*.sh`) |
| `benchmarks/bunching.py` | classical bunching baseline (per-LP and batched) |
| `benchmarks/external/` | the `dfba` package, surfinFBA, SoPlex exact and native Gurobi/Xpress lexicographic comparisons |
| `paper/data/` | the result summaries behind every table and figure of the paper (`benchmarks/export_paper_data.py`) |
| `tests/` | correctness against the independent reference, on CPU and GPU |

## Tests

```bash
pytest            # CPU; GPU variants run automatically when CuPy sees a device
```
