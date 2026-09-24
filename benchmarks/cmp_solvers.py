"""Solver adapters for the head-to-head comparison (``bench_solvers.py``).

Every adapter solves FBA LPs ``max c^T v, S v = 0, lb <= v <= ub`` of a
registered model for a batch of exchange lower bounds and returns, per LP,
``(ok, objective, v)`` with the full flux vector ``v`` (so feasibility and
accuracy can be audited independently of what the solver claims).

Per-LP CPU solvers are wrapped by :class:`ProcPool` (members pinned to worker
processes, so every solver keeps its own warm state) -- the strongest fair
way to run them on a many-core node.
"""

from __future__ import annotations

import os
import time
from typing import Dict, List

import numpy as np
import scipy.sparse as sp

INF = 1e30


def lp_bounds(model, ex_lb_row):
    lb = model.lb.copy()
    lb[model.exchanges] = ex_lb_row
    return lb, model.ub


class Adapter:
    name = "abstract"
    batch = False          # consumes the whole batch natively
    device = "cpu"
    warm = False

    def setup(self, models):
        self.models = models

    def solve(self, s, ex_lb, member_ids):
        raise NotImplementedError

    def close(self):
        pass


# ---------------------------------------------------------------------------
# manylp
# ---------------------------------------------------------------------------


class ManyLP(Adapter):
    batch = True
    warm = True

    def __init__(self, device="cuda", mode="fba", per_lp=False):
        self.dev, self.mode, self.per_lp = device, mode, per_lp
        self.device = "gpu" if device.startswith("cuda") else "cpu"
        self.name = f"manylp-{self.device}-{mode}" + ("-perlp" if per_lp else "")
        if per_lp:
            self.batch = False

    def setup(self, models):
        from manylp.dfba import ManyLPAdapter
        from manylp.fba import compile_fba

        self.models = models
        kw = {"gpu_min_batch": 1} if self.per_lp else {}
        self.ad = ManyLPAdapter(self.dev, n_workers=32, **kw)
        self.ad.problems = [compile_fba(m, mode=self.mode) for m in models]
        self.ad.groups = [self.ad.solver.register_group(p.lp, out_z=p.out_z()) for p in self.ad.problems]
        self.ad.warm = [None] * len(models)
        self.ad.calls = []

    def solve(self, s, ex_lb, member_ids):
        ad = self.ad
        if self.per_lp:
            out = [self._one(s, ex_lb[i:i + 1], member_ids[i:i + 1]) for i in range(len(member_ids))]
            return (np.concatenate([o[0] for o in out]), np.concatenate([o[1] for o in out]),
                    np.concatenate([o[2] for o in out]))
        return self._one(s, ex_lb, member_ids)

    def _one(self, s, ex_lb, member_ids):
        ad = self.ad
        p, g = ad.problems[s], ad.groups[s]
        Lp, Up = p.param_bounds(ex_lb)
        need = int(member_ids.max()) + 1
        w = ad.warm[s]
        if w is None or w.size < need:
            nw = np.full(need, -1, dtype=np.int64)
            if w is not None:
                nw[: w.size] = w
            w = ad.warm[s] = nw
        sol = ad.solver.solve_batch(g, Lp, Up, warm_start=w[member_ids])
        w[member_ids] = sol.basis_id
        ok = sol.status == 1
        return ok, sol.objective[:, 0], p.fluxes(sol.z)

    def close(self):
        self.ad.close()


# ---------------------------------------------------------------------------
# HiGHS (dual simplex warm / cold, IPM, PDLP), vanilla highspy
# ---------------------------------------------------------------------------


def _highs_model(model, method):
    import highspy

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", 1)
    if method == "ipm":
        h.setOptionValue("solver", "ipm")
    elif method == "pdlp":
        h.setOptionValue("solver", "pdlp")
    elif method == "simplex":
        h.setOptionValue("solver", "simplex")
    lp = highspy.HighsLp()
    A = model.S.tocsc()
    lp.num_col_, lp.num_row_ = model.n, model.m
    lp.col_cost_ = model.c.copy()
    lp.col_lower_, lp.col_upper_ = model.lb.copy(), model.ub.copy()
    lp.row_lower_ = np.zeros(model.m)
    lp.row_upper_ = np.zeros(model.m)
    lp.sense_ = highspy.ObjSense.kMaximize
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = A.indptr.astype(np.int32)
    lp.a_matrix_.index_ = A.indices.astype(np.int32)
    lp.a_matrix_.value_ = A.data.copy()
    lp.a_matrix_.num_col_, lp.a_matrix_.num_row_ = model.n, model.m
    h.passModel(lp)
    return h


class Highs(Adapter):
    def __init__(self, method="simplex", warm=True):
        self.method, self.warm = method, warm
        self.name = f"highs-{method}-{'warm' if warm else 'cold'}"

    def setup(self, models):
        import highspy

        self.hp = highspy
        self.models = models
        self.h = {}
        self.basis = {}

    def solve(self, s, ex_lb, member_ids):
        hp = self.hp
        mdl = self.models[s]
        B = ex_lb.shape[0]
        ok = np.zeros(B, bool)
        obj = np.zeros(B)
        V = np.zeros((B, mdl.n))
        ex = mdl.exchanges.astype(np.int32)
        for b in range(B):
            if self.warm:
                h = self.h.get(s)
                if h is None:
                    h = self.h[s] = _highs_model(mdl, self.method)
                h.changeColsBounds(ex.size, ex, ex_lb[b], mdl.ub[ex])
                key = (s, int(member_ids[b]))
                if key in self.basis:
                    h.setBasis(self.basis[key])
            else:
                h = _highs_model(mdl, self.method)
                h.changeColsBounds(ex.size, ex, ex_lb[b], mdl.ub[ex])
            h.run()
            if h.getModelStatus() == hp.HighsModelStatus.kOptimal:
                ok[b] = True
                x = np.asarray(h.getSolution().col_value)
                V[b] = x
                obj[b] = mdl.c @ x
                if self.warm:
                    self.basis[(s, int(member_ids[b]))] = h.getBasis()
        return ok, obj, V


# ---------------------------------------------------------------------------
# GLPK through cobra/optlang (muODE's production path), scipy linprog
# ---------------------------------------------------------------------------


class GLPK(Adapter):
    name = "glpk-simplex-warm (cobra/optlang)"
    warm = True

    def setup(self, models):
        import optlang.glpk_interface as gi

        self.models = models
        self.lps = {}
        self.gi = gi

    def _build(self, s):
        gi = self.gi
        mdl = self.models[s]
        m = gi.Model()
        xs = [gi.Variable(f"v{j}", lb=max(-INF, mdl.lb[j]), ub=min(INF, mdl.ub[j])) for j in range(mdl.n)]
        m.add(xs)
        S = mdl.S.tocsr()
        cons = []
        for i in range(mdl.m):
            a, b = S.indptr[i], S.indptr[i + 1]
            cons.append(gi.Constraint(0, lb=0, ub=0, name=f"r{i}"))
        m.add(cons)
        m.update()
        for i in range(mdl.m):
            a, b = S.indptr[i], S.indptr[i + 1]
            cons[i].set_linear_coefficients({xs[j]: v for j, v in zip(S.indices[a:b], S.data[a:b])})
        m.objective = gi.Objective(0, direction="max")
        m.objective.set_linear_coefficients({xs[j]: mdl.c[j] for j in np.nonzero(mdl.c)[0]})
        m.configuration.presolve = False
        return m, xs

    def solve(self, s, ex_lb, member_ids):
        if s not in self.lps:
            self.lps[s] = self._build(s)
        m, xs = self.lps[s]
        mdl = self.models[s]
        B = ex_lb.shape[0]
        ok = np.zeros(B, bool)
        obj = np.zeros(B)
        V = np.zeros((B, mdl.n))
        for b in range(B):
            for k, j in enumerate(mdl.exchanges):
                xs[j].lb = float(ex_lb[b, k])
            st = m.optimize()
            if st == "optimal":
                ok[b] = True
                V[b] = [x.primal for x in xs]
                obj[b] = m.objective.value
        return ok, obj, V


class ScipyLinprog(Adapter):
    name = "scipy-linprog (HiGHS, cold)"

    def solve(self, s, ex_lb, member_ids):
        from scipy.optimize import linprog

        mdl = self.models[s]
        B = ex_lb.shape[0]
        ok = np.zeros(B, bool)
        obj = np.zeros(B)
        V = np.zeros((B, mdl.n))
        for b in range(B):
            lb, ub = lp_bounds(mdl, ex_lb[b])
            r = linprog(-mdl.c, A_eq=mdl.S, b_eq=np.zeros(mdl.m), bounds=np.stack([lb, ub], 1),
                        method="highs")
            if r.status == 0:
                ok[b], V[b], obj[b] = True, r.x, mdl.c @ r.x
        return ok, obj, V


# ---------------------------------------------------------------------------
# Google OR-Tools: GLOP (primal/dual simplex, incremental) and PDLP (first order)
# ---------------------------------------------------------------------------


class Glop(Adapter):
    name = "ortools-glop-warm"
    warm = True

    def setup(self, models):
        self.models = models
        self.m = {}

    def _build(self, s):
        from ortools.linear_solver import pywraplp

        mdl = self.models[s]
        solver = pywraplp.Solver.CreateSolver("GLOP")
        xs = [solver.NumVar(max(-solver.infinity(), mdl.lb[j]), min(solver.infinity(), mdl.ub[j]), f"v{j}")
              for j in range(mdl.n)]
        S = mdl.S.tocsr()
        for i in range(mdl.m):
            ct = solver.Constraint(0.0, 0.0)
            for j, v in zip(S.indices[S.indptr[i]:S.indptr[i + 1]], S.data[S.indptr[i]:S.indptr[i + 1]]):
                ct.SetCoefficient(xs[j], float(v))
        o = solver.Objective()
        for j in np.nonzero(mdl.c)[0]:
            o.SetCoefficient(xs[j], float(mdl.c[j]))
        o.SetMaximization()
        return solver, xs

    def solve(self, s, ex_lb, member_ids):
        from ortools.linear_solver import pywraplp

        if s not in self.m:
            self.m[s] = self._build(s)
        solver, xs = self.m[s]
        mdl = self.models[s]
        B = ex_lb.shape[0]
        ok = np.zeros(B, bool)
        obj = np.zeros(B)
        V = np.zeros((B, mdl.n))
        for b in range(B):
            for k, j in enumerate(mdl.exchanges):
                xs[j].SetLb(float(ex_lb[b, k]))
            if solver.Solve() == pywraplp.Solver.OPTIMAL:
                ok[b] = True
                V[b] = [x.solution_value() for x in xs]
                obj[b] = solver.Objective().Value()
        return ok, obj, V


class OrtoolsPDLP(Adapter):
    def __init__(self, eps=1e-6):
        self.eps = eps
        self.name = f"ortools-pdlp (eps={eps:g})"

    def solve(self, s, ex_lb, member_ids):
        from ortools.pdlp import solve_log_pb2, solvers_pb2
        from ortools.pdlp.python import pdlp

        mdl = self.models[s]
        B = ex_lb.shape[0]
        ok = np.zeros(B, bool)
        obj = np.zeros(B)
        V = np.zeros((B, mdl.n))
        params = solvers_pb2.PrimalDualHybridGradientParams()
        params.termination_criteria.simple_optimality_criteria.eps_optimal_relative = self.eps
        params.termination_criteria.simple_optimality_criteria.eps_optimal_absolute = self.eps
        params.termination_criteria.time_sec_limit = 60.0
        params.num_threads = 1
        for b in range(B):
            lb, ub = lp_bounds(mdl, ex_lb[b])
            qp = pdlp.QuadraticProgram()
            qp.objective_vector = -mdl.c
            qp.constraint_matrix = sp.csr_matrix(mdl.S)
            qp.constraint_lower_bounds = np.zeros(mdl.m)
            qp.constraint_upper_bounds = np.zeros(mdl.m)
            qp.variable_lower_bounds = lb
            qp.variable_upper_bounds = ub
            r = pdlp.primal_dual_hybrid_gradient(qp, params)
            if r.solve_log.termination_reason == solve_log_pb2.TERMINATION_REASON_OPTIMAL:
                ok[b] = True
            V[b] = r.primal_solution
            obj[b] = mdl.c @ V[b]
        return ok, obj, V


# ---------------------------------------------------------------------------
# OSQP (ADMM; LP as a QP with P = 0), warm-started
# ---------------------------------------------------------------------------


class OSQP(Adapter):
    warm = True

    def __init__(self, eps=1e-6):
        self.eps = eps
        self.name = f"osqp-admm-warm (eps={eps:g})"

    def setup(self, models):
        self.models = models
        self.solvers = {}
        self.state = {}

    def solve(self, s, ex_lb, member_ids):
        import osqp

        mdl = self.models[s]
        n, m = mdl.n, mdl.m
        if s not in self.solvers:
            A = sp.vstack([mdl.S, sp.identity(n)], format="csc")
            P = sp.csc_matrix((n, n))
            o = osqp.OSQP()
            o.setup(P=P, q=-mdl.c, A=A, l=np.concatenate([np.zeros(m), mdl.lb]),
                    u=np.concatenate([np.zeros(m), mdl.ub]), eps_abs=self.eps, eps_rel=self.eps,
                    max_iter=100000, verbose=False, polishing=True)
            self.solvers[s] = o
        o = self.solvers[s]
        B = ex_lb.shape[0]
        ok = np.zeros(B, bool)
        obj = np.zeros(B)
        V = np.zeros((B, n))
        for b in range(B):
            lb, ub = lp_bounds(mdl, ex_lb[b])
            o.update(l=np.concatenate([np.zeros(m), lb]), u=np.concatenate([np.zeros(m), ub]))
            key = (s, int(member_ids[b]))
            if key in self.state:
                o.warm_start(x=self.state[key][0], y=self.state[key][1])
            r = o.solve()
            if r.info.status_val in (1, 2):   # solved / solved inaccurate
                ok[b] = r.info.status_val == 1
                V[b] = r.x
                obj[b] = mdl.c @ r.x
                self.state[key] = (r.x.copy(), r.y.copy())
        return ok, obj, V


# ---------------------------------------------------------------------------
# NVIDIA cuOpt (GPU PDLP / barrier / concurrent), per LP or BatchSolve
# ---------------------------------------------------------------------------


class CuOpt(Adapter):
    device = "gpu"

    def __init__(self, method="pdlp", eps=1e-6, batch=False, warm=False, crossover=False):
        self.method, self.eps, self.batch, self.warm, self.crossover = method, eps, batch, warm, crossover
        self.device = "cpu" if method == "dual_simplex" else "gpu"
        self.name = (f"cuopt-{method}{'-batch' if batch else ''}{'-warm' if warm else ''}"
                     f"{'+crossover' if crossover else ''} (eps={eps:g})")

    def setup(self, models):
        from cuopt.linear_programming import DataModel, SolverSettings, SolverMethod
        from cuopt.linear_programming.solver.solver_parameters import (
            CUOPT_ABSOLUTE_DUAL_TOLERANCE, CUOPT_ABSOLUTE_GAP_TOLERANCE,
            CUOPT_ABSOLUTE_PRIMAL_TOLERANCE, CUOPT_METHOD, CUOPT_RELATIVE_DUAL_TOLERANCE,
            CUOPT_RELATIVE_GAP_TOLERANCE, CUOPT_RELATIVE_PRIMAL_TOLERANCE, CUOPT_TIME_LIMIT)

        self.DataModel = DataModel
        self.models = models
        st = SolverSettings()
        meth = {"pdlp": SolverMethod.PDLP, "barrier": SolverMethod.Barrier,
                "dual_simplex": SolverMethod.DualSimplex, "concurrent": SolverMethod.Concurrent}[self.method]
        st.set_parameter(CUOPT_METHOD, meth)
        for p in (CUOPT_ABSOLUTE_DUAL_TOLERANCE, CUOPT_ABSOLUTE_GAP_TOLERANCE, CUOPT_ABSOLUTE_PRIMAL_TOLERANCE,
                  CUOPT_RELATIVE_DUAL_TOLERANCE, CUOPT_RELATIVE_GAP_TOLERANCE, CUOPT_RELATIVE_PRIMAL_TOLERANCE):
            st.set_parameter(p, self.eps)
        st.set_parameter(CUOPT_TIME_LIMIT, 60.0)
        if self.crossover:
            from cuopt.linear_programming.solver.solver_parameters import CUOPT_CROSSOVER

            st.set_parameter(CUOPT_CROSSOVER, True)     # PDLP / barrier -> exact basic solution
        self.settings = st
        self.prev = {}

    def _dm(self, mdl, lb, ub, key=None):
        dm = self.DataModel()
        A = mdl.S.tocsr()
        dm.set_csr_constraint_matrix(A.data.astype(np.float64), A.indices.astype(np.int32),
                                     A.indptr.astype(np.int32))
        dm.set_constraint_bounds(np.zeros(mdl.m))
        dm.set_row_types(np.array(["E"] * mdl.m))
        dm.set_objective_coefficients(mdl.c.astype(np.float64))
        dm.set_maximize(True)
        dm.set_variable_lower_bounds(np.maximum(lb, -INF))
        dm.set_variable_upper_bounds(np.minimum(ub, INF))
        if self.warm and key in self.prev:
            x, y = self.prev[key]
            dm.set_initial_primal_solution(x)
            dm.set_initial_dual_solution(y)
        return dm

    def solve(self, s, ex_lb, member_ids):
        from cuopt.linear_programming import BatchSolve, Solve

        mdl = self.models[s]
        B = ex_lb.shape[0]
        dms = []
        for b in range(B):
            lb, ub = lp_bounds(mdl, ex_lb[b])
            dms.append(self._dm(mdl, lb, ub, (s, int(member_ids[b]))))
        if self.batch:
            sols, _ = BatchSolve(dms, self.settings)
        else:
            sols = [Solve(dm, self.settings) for dm in dms]
        ok = np.zeros(B, bool)
        obj = np.zeros(B)
        V = np.zeros((B, mdl.n))
        for b, so in enumerate(sols):
            try:
                x = np.asarray(so.get_primal_solution(), dtype=float)
            except Exception:
                continue
            if x.size != mdl.n:
                continue
            ok[b] = so.get_termination_status() == 1   # Optimal
            V[b] = x
            obj[b] = mdl.c @ x
            if self.warm:
                self.prev[(s, int(member_ids[b]))] = (x, np.asarray(so.get_dual_solution(), dtype=float))
        return ok, obj, V


# ---------------------------------------------------------------------------
# MPAX (JAX, GPU): batched r2HPDHG via vmap
# ---------------------------------------------------------------------------


class MPAX(Adapter):
    device = "gpu"
    batch = True

    def __init__(self, eps=1e-6):
        self.eps = eps
        self.name = f"mpax-r2hpdhg-batched (eps={eps:g})"

    def setup(self, models):
        import jax

        jax.config.update("jax_enable_x64", True)
        self.models = models
        self.fns = {}

    def _fn(self, s):
        import jax
        import jax.numpy as jnp
        from mpax import create_lp, r2HPDHG

        if s in self.fns:
            return self.fns[s]
        mdl = self.models[s]
        A = jnp.asarray(mdl.S.toarray())
        c = jnp.asarray(-mdl.c)
        b = jnp.zeros(mdl.m)
        solver = r2HPDHG(eps_abs=self.eps, eps_rel=self.eps, verbose=False, iteration_limit=100000)

        def one(l, u):
            lp = create_lp(c, A, b, jnp.zeros((0, mdl.n)), jnp.zeros(0), l, u, use_sparse_matrix=False)
            r = solver.optimize(lp)
            return r.primal_solution, r.termination_status

        f = jax.jit(jax.vmap(one))
        self.fns[s] = f
        return f

    def solve(self, s, ex_lb, member_ids):
        mdl = self.models[s]
        B = ex_lb.shape[0]
        L = np.tile(mdl.lb, (B, 1))
        U = np.tile(mdl.ub, (B, 1))
        L[:, mdl.exchanges] = ex_lb
        X, st = self._fn(s)(L, U)
        X = np.asarray(X)
        st = np.asarray(st)
        ok = st == 2          # mpax TerminationStatus.OPTIMAL
        return ok, X @ mdl.c, X


# ---------------------------------------------------------------------------
# our batched restarted PDHG (JAX, cuPDLP-style)
# ---------------------------------------------------------------------------


class OurPDHG(Adapter):
    device = "gpu"
    batch = True
    warm = True

    def __init__(self, eps=1e-6):
        self.eps = eps
        self.name = f"batched-restarted-pdhg (eps={eps:g})"

    def setup(self, models):
        from manylp.fba import compile_fba

        self.models = models
        self.probs = [compile_fba(m, mode="fba") for m in models]
        self.solvers = {}
        self.state = {}

    def solve(self, s, ex_lb, member_ids):
        from manylp.pdhg import BatchedPDHG

        p = self.probs[s]
        if s not in self.solvers:
            self.solvers[s] = BatchedPDHG(p.lp, eps=self.eps, max_iters=20000, device="cuda")
        Lp, Up = p.param_bounds(ex_lb)
        r = self.solvers[s].solve(Lp, Up)
        return r.converged, r.objective, r.x


# ---------------------------------------------------------------------------
# commercial: Gurobi (restricted pip licence) and FICO Xpress (community licence)
# ---------------------------------------------------------------------------


class Gurobi(Adapter):
    def __init__(self, method="dual", warm=True):
        self.method, self.warm = method, warm
        self.name = f"gurobi-{method}-{'warm' if warm else 'cold'}"

    def setup(self, models):
        self.models = models
        self.m = {}
        self.basis = {}

    def _build(self, s):
        import gurobipy as gp

        mdl = self.models[s]
        env = gp.Env(params={"OutputFlag": 0, "Threads": 1})
        m = gp.Model(env=env)
        x = m.addMVar(mdl.n, lb=np.maximum(mdl.lb, -gp.GRB.INFINITY), ub=np.minimum(mdl.ub, gp.GRB.INFINITY))
        m.addConstr(mdl.S.tocsr() @ x == 0)
        m.setObjective(mdl.c @ x, gp.GRB.MAXIMIZE)
        m.Params.Method = {"dual": 1, "primal": 0, "barrier": 2}[self.method]
        if self.method == "barrier":
            m.Params.Crossover = 1
        m.update()
        return m, x

    def solve(self, s, ex_lb, member_ids):
        import gurobipy as gp

        mdl = self.models[s]
        B = ex_lb.shape[0]
        ok = np.zeros(B, bool)
        obj = np.zeros(B)
        V = np.zeros((B, mdl.n))
        for b in range(B):
            if self.warm:
                if s not in self.m:
                    self.m[s] = self._build(s)
                m, x = self.m[s]
            else:
                m, x = self._build(s)
            xe = x[mdl.exchanges]
            xe.LB = ex_lb[b]
            key = (s, int(member_ids[b]))
            if self.warm and key in self.basis:
                vb, cb = self.basis[key]
                m.setAttr("VBasis", m.getVars(), vb)
                m.setAttr("CBasis", m.getConstrs(), cb)
            m.optimize()
            if m.Status == gp.GRB.OPTIMAL:
                ok[b] = True
                V[b] = x.X
                obj[b] = m.ObjVal
                if self.warm:
                    try:
                        self.basis[key] = (m.getAttr("VBasis", m.getVars()), m.getAttr("CBasis", m.getConstrs()))
                    except Exception:
                        pass
        return ok, obj, V


class Xpress(Adapter):
    warm = True

    def __init__(self, method="dual"):
        self.method = method
        self.name = f"xpress-{method}-warm"

    def setup(self, models):
        self.models = models
        self.p = {}

    def _build(self, s):
        import xpress as xp

        mdl = self.models[s]
        p = xp.problem()
        p.controls.outputlog = 0
        p.controls.threads = 1
        S = mdl.S.tocsc()
        big = xp.infinity
        p.loadproblem("", ["E"] * mdl.m, np.zeros(mdl.m), None, -mdl.c,
                      S.indptr[:-1].astype(np.int64), np.diff(S.indptr).astype(np.int64),
                      S.indices.astype(np.int64), S.data,
                      np.maximum(mdl.lb, -big), np.minimum(mdl.ub, big))
        return p

    def solve(self, s, ex_lb, member_ids):
        mdl = self.models[s]
        if s not in self.p:
            self.p[s] = self._build(s)
        p = self.p[s]
        B = ex_lb.shape[0]
        ok = np.zeros(B, bool)
        obj = np.zeros(B)
        V = np.zeros((B, mdl.n))
        ex = mdl.exchanges.astype(np.int64)
        flag = "d" if self.method == "dual" else ("b" if self.method == "barrier" else "p")
        for b in range(B):
            p.chgbounds(ex.tolist(), ["L"] * ex.size, ex_lb[b].tolist())
            p.lpoptimize(flag)
            if p.attributes.lpstatus == 1:
                ok[b] = True
                x = np.asarray(p.getSolution(), dtype=float)
                V[b] = x
                obj[b] = mdl.c @ x
        return ok, obj, V


class HighsLex(Adapter):
    """Warm-started HiGHS applying manylp's pFBA-unique rule per LP (like-for-like unique fluxes)."""

    warm = True

    def __init__(self, mode="pfba-unique"):
        self.mode = mode
        self.name = f"highs-lex-{mode}-warm"

    def setup(self, models):
        from manylp.fba import compile_fba

        self.models = models
        self.probs = [compile_fba(m, mode=self.mode) for m in models]
        self.solvers = {}
        self.basis = {}

    def solve(self, s, ex_lb, member_ids):
        from manylp.repair import HighsLexSolver

        p = self.probs[s]
        if s not in self.solvers:
            self.solvers[s] = HighsLexSolver(p.lp)
        hs = self.solvers[s]
        Lp, Up = p.param_bounds(ex_lb)
        oz = p.out_z()
        B = ex_lb.shape[0]
        ok = np.zeros(B, bool)
        obj = np.zeros(B)
        Z = np.zeros((B, oz.size))
        for b in range(B):
            key = (s, int(member_ids[b]))
            r = hs.solve(*p.lp.full_bounds(Lp[b], Up[b]), warm_status=self.basis.get(key))
            if r.status == 1:
                ok[b], obj[b], Z[b] = True, r.stage_obj[0], r.z[oz]
                self.basis[key] = r.zstatus
        return ok, obj, p.fluxes(Z)


# ---------------------------------------------------------------------------
# process pool wrapper for per-LP CPU solvers
# ---------------------------------------------------------------------------

_W: Dict[str, object] = {}


def _winit(factory_name, kwargs, models):
    os.environ["OMP_NUM_THREADS"] = "1"
    a = REGISTRY[factory_name](**kwargs)
    a.setup(models)
    _W["a"] = a


def _wsolve(s, ex_lb, member_ids):
    t = time.perf_counter()
    r = _W["a"].solve(s, ex_lb, member_ids)
    return r, time.perf_counter() - t


def _wstats():
    a = _W["a"]
    return a.stats() if hasattr(a, "stats") else {}


class ProcPool(Adapter):
    """Run a per-LP adapter in ``n`` processes; members pinned by ``member % n``."""

    def __init__(self, factory_name, n=32, **kwargs):
        self.fn, self.kw, self.n = factory_name, kwargs, n
        proto = REGISTRY[factory_name](**kwargs)
        self.name = f"{proto.name} x{n}proc"
        self.device, self.warm = proto.device, proto.warm

    def setup(self, models):
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        self.models = models
        ctx = mp.get_context("fork")
        self.ex = [ProcessPoolExecutor(1, mp_context=ctx, initializer=_winit,
                                       initargs=(self.fn, self.kw, models)) for _ in range(self.n)]

    def solve(self, s, ex_lb, member_ids):
        mdl = self.models[s]
        B = ex_lb.shape[0]
        ok = np.zeros(B, bool)
        obj = np.zeros(B)
        V = np.zeros((B, mdl.n))
        owner = member_ids % self.n
        futs = []
        for w in np.unique(owner):
            idx = np.nonzero(owner == w)[0]
            futs.append((idx, self.ex[w].submit(_wsolve, s, ex_lb[idx], member_ids[idx])))
        for idx, f in futs:
            (o, ob, v), _ = f.result()
            ok[idx], obj[idx], V[idx] = o, ob, v
        return ok, obj, V

    def stats(self):
        tot = {}
        for e in self.ex:
            for k, v in e.submit(_wstats).result().items():
                tot[k] = tot.get(k, 0) + v
        return tot

    def close(self):
        for e in self.ex:
            e.shutdown(cancel_futures=True)


from bunching import Bunching  # noqa: E402  (benchmarks/ is on sys.path)

REGISTRY = {
    "bunching": Bunching, "highs": Highs, "glpk": GLPK, "scipy": ScipyLinprog, "glop": Glop, "ortools-pdlp": OrtoolsPDLP,
    "osqp": OSQP, "gurobi": Gurobi, "xpress": Xpress, "highs-lex": HighsLex,
}
