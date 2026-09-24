"""Native hierarchical (lexicographic) multi-objective solving of manylp's pFBA-unique rule.

The rule (``compile_fba(model, "pfba-unique")``) is a 3-stage lexicographic LP over
the split variables ``z`` (``v = v+ - v-`` for reversible and exchange reactions):

  1. max  c1^T z   (growth)
  2. max -l1^T z   (min L1 norm of fluxes) over the stage-1 optimal face
  3. max  w^T z    (fixed seeded generic direction) over the stage-2 optimal face

Here it is handed, unchanged, to a commercial solver's *native* multi-objective API:

* Gurobi   -- ``setObjectiveN(expr, index, priority, weight=1, abstol, reltol)``,
               ``ModelSense = MAXIMIZE``;
* Xpress   -- ``addObjective`` / ``setObjective(..., objidx, priority, weight=1,
               abstol, reltol)`` with ``optimize()``.

Per-objective degradation tolerances are set to 0 (``--objtol``), and primal / dual
feasibility tolerances to 1e-9 (``--feastol``, the same as manylp's HiGHS lex baseline).

Audit against ``<workload>_reference.npz``: stage-1 objective vs ``obj`` (exact FBA
optimum, certified by manylp) and exchange fluxes vs ``vex`` (canonical pFBA-unique
fluxes, certified unique by manylp).

``--sizes`` only reports the split-LP sizes and whether each model is accepted by the
solver's licence (an actual optimise of the template LP).

Run with the comparison env (``cmpenv``: gurobipy restricted pip licence, xpress
community licence); single thread, one persistent problem per species, members
solved sequentially in workload order (the solver keeps its last basis).
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from manylp.fba import compile_fba  # noqa: E402


# ---------------------------------------------------------------------------------
# solver wrappers: build once per species, then (set param bounds, solve) per member
# ---------------------------------------------------------------------------------


class GurobiLex:
    name = "gurobi"

    def __init__(self, lp, objtol=0.0, feastol=1e-9, multiobjops=None):
        import gurobipy as gp

        self.gp = gp
        self.lp = lp
        env = gp.Env(params={"OutputFlag": 0, "Threads": 1})
        m = gp.Model(env=env)
        big = gp.GRB.INFINITY
        x = m.addMVar(lp.n, lb=np.maximum(lp.col_lb, -big), ub=np.minimum(lp.col_ub, big))
        m.addConstr(lp.A.tocsr() @ x == 0)
        m.ModelSense = gp.GRB.MAXIMIZE
        K = lp.K
        for k in range(K):
            m.setObjectiveN(gp.LinExpr(lp.objectives[k].tolist(), x.tolist()), index=k,
                            priority=K - k, weight=1.0, abstol=objtol, reltol=objtol,
                            name=f"stage{k}")
        m.Params.FeasibilityTol = feastol
        m.Params.OptimalityTol = feastol
        m.update()
        self.m, self.x = m, x
        self.pc = lp.param_cols
        self.xp = [x[int(j)] for j in self.pc]
        self.settings = {"ModelSense": "MAXIMIZE", "priorities": list(range(K, 0, -1)), "weights": 1.0,
                         "ObjNAbsTol": objtol, "ObjNRelTol": objtol, "FeasibilityTol": feastol,
                         "OptimalityTol": feastol, "Threads": 1, "gurobi_version": list(gp.gurobi.version())}

    def solve(self, lb, ub):
        gp, m = self.gp, self.m
        xv = self.x[self.pc]
        xv.LB = lb[self.pc]
        xv.UB = ub[self.pc]
        m.optimize()
        if m.Status != gp.GRB.OPTIMAL:
            return False, None, str(m.Status)
        return True, np.asarray(self.x.X, dtype=float), "OPTIMAL"


class XpressLex:
    name = "xpress"

    def __init__(self, lp, objtol=0.0, feastol=1e-9, multiobjops=None):
        import xpress as xp

        self.xp_ = xp
        self.lp = lp
        p = xp.problem()
        p.controls.outputlog = 0
        p.controls.threads = 1
        A = lp.A.tocsc()
        big = xp.infinity
        p.loadproblem("", ["E"] * lp.m, np.zeros(lp.m), None, np.zeros(lp.n),
                      A.indptr[:-1].astype(np.int64), np.diff(A.indptr).astype(np.int64),
                      A.indices.astype(np.int64), A.data,
                      np.maximum(lp.col_lb, -big), np.minimum(lp.col_ub, big))
        cols = p.getVariable()
        K = lp.K

        def expr(vec):
            nz = np.nonzero(vec)[0]
            return xp.Sum(float(vec[j]) * cols[j] for j in nz)

        p.setObjective(expr(lp.objectives[0]), sense=xp.maximize, objidx=0, priority=K, weight=1.0,
                       abstol=objtol, reltol=objtol)
        for k in range(1, K):
            p.addObjective(expr(lp.objectives[k]), priority=K - k, weight=1.0, abstol=objtol, reltol=objtol)
        p.controls.feastol = feastol
        p.controls.optimalitytol = feastol
        if multiobjops is not None:     # bits: 1 ENABLED, 2 PRESOLVE, 4 RCFIXING (default 7)
            p.controls.multiobjops = multiobjops
        self.p = p
        self.pc = lp.param_cols.astype(np.int64).tolist()
        oc = p.objcontrols
        self.settings = {"sense": "maximize", "priorities": list(range(K, 0, -1)), "weights": 1.0,
                         "objective_abstol": [oc[k].abstol for k in range(K)],
                         "objective_reltol": [oc[k].reltol for k in range(K)],
                         "feastol": p.controls.feastol, "optimalitytol": p.controls.optimalitytol,
                         "multiobjops": int(p.controls.multiobjops), "threads": 1,
                         "xpress_version": xp.__version__, "n_objectives": int(p.attributes.objectives)}

    def solve(self, lb, ub):
        xp, p = self.xp_, self.p
        n = len(self.pc)
        p.chgbounds(self.pc + self.pc, ["L"] * n + ["U"] * n,
                    np.concatenate([lb[self.pc], ub[self.pc]]).tolist())
        p.optimize()
        a = p.attributes
        ok = (a.solvestatus == xp.SolveStatus.COMPLETED and a.solstatus == xp.SolStatus.OPTIMAL
              and a.solvedobjs == self.lp.K)
        if not ok:
            return False, None, f"solve={a.solvestatus} sol={a.solstatus} solvedobjs={a.solvedobjs}"
        return True, np.asarray(p.getSolution(), dtype=float), "OPTIMAL"


SOLVERS = {"gurobi": GurobiLex, "xpress": XpressLex}
LIMITS = {"gurobi": "restricted pip licence: <= 2000 variables and <= 2000 linear constraints",
          "xpress": "community licence: rows + columns <= 5000 (error ?120 otherwise)"}


def sizes(models, solver):
    out = []
    for name, mdl in models:
        p = compile_fba(mdl, "pfba-unique")
        lp = p.lp
        rec = {"model": name, "fba_vars": int(mdl.n), "fba_cons": int(mdl.m), "split_vars": int(lp.n),
               "split_cons": int(lp.m), "nnz": int(lp.A.nnz), "objectives": int(lp.K)}
        try:
            s = SOLVERS[solver](lp)
            lb, ub = lp.col_lb, lp.col_ub
            ok, _, msg = s.solve(lb, ub)
            rec["accepted"] = True
            rec["template_solve"] = msg
        except Exception as e:  # licence errors surface here
            rec["accepted"] = False
            rec["error"] = f"{type(e).__name__}: {e}"
        out.append(rec)
        print(json.dumps(rec), flush=True)
    return out


def diagnose(p, lp_row, up_row, vref, stage, hl, s):
    """HiGHS lexicographic re-solve (manylp.repair.HighsLexSolver, cold) of one LP."""
    from manylp.repair import HighsLexSolver

    if s not in hl:
        hl[s] = HighsLexSolver(p.lp)
    lb_z, ub_z = p.lp.full_bounds(lp_row, up_row)
    r = hl[s].solve(lb_z, ub_z)
    out = {"highs_status": int(r.status)}
    if r.status == 1:
        z = r.z[: p.lp.n]
        v = z[: p.model.n].copy()
        has = p.neg_col >= 0
        v[has] -= z[p.neg_col[has]]
        out["highs_stage_obj"] = r.stage_obj.tolist()
        out["highs_vex_maxabs"] = float(np.max(np.abs(v[p.model.exchanges] - vref)))
        # solver minus HiGHS per stage (all stages are maximised: negative = solver worse)
        out["stage_gap"] = (np.asarray(stage) - r.stage_obj).tolist()
    return out


def diag_summary(rows):
    d = [r["diag"] for r in rows if "diag" in r and r["diag"].get("highs_status") == 1]
    if not d:
        return {"n": 0}
    g = np.array([x["stage_gap"] for x in d])
    hv = np.array([x["highs_vex_maxabs"] for x in d])
    sc = np.array([[max(1.0, abs(v)) for v in x["highs_stage_obj"]] for x in d])
    rg = g / sc
    first = [int(np.argmax(np.abs(x) > 1e-9)) if np.any(np.abs(x) > 1e-9) else -1 for x in rg]
    return {"n": len(d), "highs_vex_maxabs_max": float(hv.max()),
            "highs_matches_ref_1e-9": int((hv <= 1e-9).sum()),
            "stage_relgap_min": rg.min(axis=0).tolist(), "stage_relgap_max": rg.max(axis=0).tolist(),
            "first_stage_with_relgap_gt_1e-9": {str(k): first.count(k) for k in sorted(set(first))}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", choices=list(SOLVERS), required=True)
    ap.add_argument("--workload", default="results/workload_coherent.pkl")
    ap.add_argument("--start", type=int, default=1200, help="first snapshot (batch) index")
    ap.add_argument("--nbatch", type=int, default=8)
    ap.add_argument("--species", type=int, default=-1,
                    help="only batches of this species: the nbatch consecutive ones from --start on")
    ap.add_argument("--objtol", type=float, default=0.0)
    ap.add_argument("--feastol", type=float, default=1e-9)
    ap.add_argument("--multiobjops", type=int, default=None, help="Xpress MULTIOBJOPS (default 7)")
    ap.add_argument("--diagnose", action="store_true",
                    help="re-solve LPs whose exchange fluxes deviate (>1e-9) with manylp's HiGHS lex "
                         "solver and record its stage objectives")
    ap.add_argument("--sizes", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    W = pickle.load(open(args.workload, "rb"))
    models = W["models"]
    snaps = W["snapshots"]
    os.makedirs("results/external", exist_ok=True)
    if args.sizes:
        rec = sizes(models, args.solver)
        out = args.out or f"results/external/lex_native_sizes_{args.solver}.json"
        json.dump({"solver": args.solver, "limit": LIMITS[args.solver], "models": rec}, open(out, "w"), indent=1)
        print("wrote", out)
        return

    with np.load(args.workload.replace(".pkl", "_reference.npz"), allow_pickle=True) as z:
        ref = {k: z[k] for k in ("obj", "ok", "vex", "unique")}   # NpzFile re-reads on every access
    if args.species >= 0:
        idx = [i for i in range(args.start, len(snaps)) if snaps[i][1] == args.species][: args.nbatch]
    else:
        idx = list(range(args.start, args.start + args.nbatch))
    probs, solvers, build_t, hl = {}, {}, {}, {}
    rows = []
    t_all = 0.0
    for bi in idx:
        step, s, members, ex_lb = snaps[bi]
        if s not in probs:
            probs[s] = compile_fba(models[s][1], "pfba-unique")
            t0 = time.perf_counter()
            solvers[s] = SOLVERS[args.solver](probs[s].lp, objtol=args.objtol, feastol=args.feastol,
                                              multiobjops=args.multiobjops)
            build_t[models[s][0]] = time.perf_counter() - t0
        p, sv = probs[s], solvers[s]
        lp = p.lp
        ex = p.model.exchanges
        Lp, Up = p.param_bounds(ex_lb)
        for j in range(len(members)):
            lb, ub = lp.full_bounds(Lp[j], Up[j])
            lb, ub = lb[: lp.n], ub[: lp.n]          # z-space -> structural columns
            t0 = time.perf_counter()
            ok, z, msg = sv.solve(lb, ub)
            dt = time.perf_counter() - t0
            t_all += dt
            r = {"batch": bi, "step": int(step), "species": models[s][0], "member": int(members[j]),
                 "seconds": dt, "ok": bool(ok), "status": msg, "ref_ok": bool(ref["ok"][bi][j]),
                 "ref_unique": bool(ref["unique"][bi][j])}
            if ok:
                stage = lp.objectives @ z
                v = z[: p.model.n].copy()
                has = p.neg_col >= 0
                v[has] -= z[p.neg_col[has]]
                vex = v[ex]
                o_ref = float(ref["obj"][bi][j])
                r["stage_obj"] = stage.tolist()
                r["obj_relerr"] = abs(stage[0] - o_ref) / max(1.0, abs(o_ref))
                r["vex_maxabs"] = float(np.max(np.abs(vex - ref["vex"][bi][j])))
                r["vex_maxrel"] = float(np.max(np.abs(vex - ref["vex"][bi][j]) / np.maximum(1.0, np.abs(ref["vex"][bi][j]))))
                r["primal_resid"] = float(np.max(np.abs(lp.A @ z)))
                r["bound_viol"] = float(max(np.max(lb - z), np.max(z - ub), 0.0))
                if args.diagnose and r["vex_maxabs"] > 1e-9:
                    r["diag"] = diagnose(p, Lp[j], Up[j], ref["vex"][bi][j], stage, hl, s)
            rows.append(r)
        print(f"batch {bi} ({models[s][0]}, step {step}): cum {len(rows)} LPs, {t_all:.2f}s", flush=True)

    t = np.array([r["seconds"] for r in rows])
    okr = [r for r in rows if r["ok"] and r["ref_ok"]]
    oe = np.array([r["obj_relerr"] for r in okr]) if okr else np.zeros(0)
    ve = np.array([r["vex_maxabs"] for r in okr]) if okr else np.zeros(0)
    uq = np.array([r["ref_unique"] for r in okr], bool) if okr else np.zeros(0, bool)
    thr = [1e-9, 1e-6, 1e-3]
    summ = {
        "solver": args.solver, "settings": next(iter(solvers.values())).settings, "workload": args.workload,
        "batches": idx, "n_lps": len(rows), "species": sorted({r["species"] for r in rows}),
        "build_seconds": build_t,
        "time_total_s": float(t.sum()), "time_median_ms": float(np.median(t) * 1e3),
        "time_mean_ms": float(t.mean() * 1e3), "time_max_ms": float(t.max() * 1e3),
        "lps_per_s_single_core": float(len(rows) / t.sum()),
        "status_agree": int(sum(r["ok"] == r["ref_ok"] for r in rows)),
        "n_ok": int(sum(r["ok"] for r in rows)), "n_ref_ok": int(sum(r["ref_ok"] for r in rows)),
        "n_audited": len(okr), "n_ref_unique_audited": int(uq.sum()),
        "obj_relerr_max": float(oe.max()) if oe.size else None,
        "obj_relerr_median": float(np.median(oe)) if oe.size else None,
        "vex_maxabs_max": float(ve.max()) if ve.size else None,
        "vex_maxabs_median": float(np.median(ve)) if ve.size else None,
        "vex_maxabs_max_on_ref_unique": float(ve[uq].max()) if uq.any() else None,
        "vex_frac_within": {f"{h:g}": float(np.mean(ve <= h)) for h in thr} if ve.size else None,
        "obj_frac_within": {f"{h:g}": float(np.mean(oe <= h)) for h in thr} if oe.size else None,
        "primal_resid_max": float(max(r["primal_resid"] for r in okr)) if okr else None,
        "bound_viol_max": float(max(r["bound_viol"] for r in okr)) if okr else None,
    }
    tag = f"{args.solver}" + (f"_sp{args.species}" if args.species >= 0 else "") + f"_b{idx[0]}_n{len(idx)}"
    if args.multiobjops is not None:
        tag += f"_mo{args.multiobjops}"
    if args.diagnose:
        summ["diagnosis"] = diag_summary(rows)
    if args.objtol != 0.0 or args.feastol != 1e-9:
        tag += f"_objtol{args.objtol:g}_feastol{args.feastol:g}"
    out = args.out or f"results/external/lex_native_{tag}.json"
    json.dump({"summary": summ, "per_lp": rows}, open(out, "w"), indent=1)
    print(json.dumps(summ, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
