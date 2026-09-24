"""Run surfinFBA (Brunner & Chia 2020, PLoS Comput Biol) on manylp's scenarios.

surfinFBA (github.com/jdbrunner/surfin_fba, v0.9, unmaintained since 2023) integrates
community dFBA with scipy's ``vode`` and re-uses one optimal basis per species until
the reduced linear system leaves the feasible region, then finds a new basis with an
LP whose objective is the *derivative* of growth (its "find_waves" step).  Its LP
solver is Gurobi or CPLEX (no open-source option); here: gurobipy 13.0.3 with the
pip-bundled size-limited licence (<= 2000 variables and <= 2000 constraints).
Runs in the ``surfinenv`` conda env; does not import manylp.

Model semantics (what surfinFBA can express):

* uptake bounds are *linear* in the metabolite, ``u_j = kappa_j y_j`` -- Michaelis-Menten
  is not supported.  We therefore compare against a manylp reference with linear uptake
  (``ref_<scenario>_linear.npz`` from ``make_references.py --linear``) using the MM
  secant ``kappa_j = Vmax / (Km + M0_j)`` (``Vmax/Km`` where ``M0_j = 0``), so the t = 0
  LP equals the MM scenario's;
* SurfMod objects are built directly from the SBML (``prep_cobrapy_models`` only exposes
  exchanges open in ``model.medium`` and relaxes positive internal lower bounds to 0,
  which would drop ATP maintenance); ``--relax-lb`` reproduces that relaxation;
* flux selection: max growth at (re-)initialisation, then surfinFBA's own rule (initial
  secondary objective "min sum v", later bases from the derivative LP) -- not pFBA.

``--example`` runs the package's own BiGG example (``examples/BiGG_model_examples.py``:
random kappas, in/outflow 1, death ~1.1, 1 h) on BiGG e_coli_core instead of iJR904,
which exceeds the size-limited Gurobi licence.

Outputs ``results/external/surfinfba_<scenario>[...].json`` (+ ``.npz``).
"""
import argparse
import copy
import io
import json
import os
import re
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "results", "external")
SURF = os.environ.get("SURFIN_DIR", os.path.expanduser("~/isilon/cumbof/manylp_ws/ext/surfin_fba"))
sys.path.insert(0, SURF)


def surfmod_from_cobra(mdl, ex_rxns, env_mets, ex_mets, kappa_env, name, relax_lb=False):
    """SurfMod with Gamma1 = S[env mets, internal rxns] (net secretion), Gamma2 = internal balance."""
    import surfinFBA as surf
    from cobra.util.array import create_stoichiometric_matrix

    S = create_stoichiometric_matrix(mdl, array_type="dense")
    rid = [r.id for r in mdl.reactions]
    mid = [m.id for m in mdl.metabolites]
    exs = set(ex_rxns)
    internal = [j for j, r in enumerate(rid) if r not in exs]
    ext = set(ex_mets)
    G1 = np.zeros((len(env_mets), len(internal)))
    for k, m in enumerate(env_mets):
        if m in mid:
            G1[k] = S[mid.index(m), internal]
    imets = [i for i, m in enumerate(mid) if m not in ext]
    G2 = S[np.ix_(imets, internal)]
    rx = list(mdl.reactions)
    lilg = np.array([rx[j].objective_coefficient for j in internal], dtype=float)
    ilb = np.array([rx[j].lower_bound for j in internal], dtype=float)
    iub = np.array([rx[j].upper_bound for j in internal], dtype=float)
    if relax_lb:
        ilb = np.minimum(0.0, ilb)
    # exchange secretion bound (-ub of the exchange reaction); metabolites the species does not
    # exchange get kappa = 0 and 0 bounds, as prep_cobrapy_models does
    ub_ex = {m: mdl.reactions.get_by_id(r).upper_bound for r, m in zip(ex_rxns, ex_mets)}
    elb = np.array([-ub_ex.get(m, 0.0) for m in env_mets])
    kap = np.array([kappa_env[k] if m in ub_ex else 0.0 for k, m in enumerate(env_mets)])
    return surf.SurfMod(G1, G2, lilg, ilb, iub, kap, elb, Name=name, deathrate=0.0)


def run_surfin(models, x0, y0, t_end, names, initres=1e-3):
    import surfinFBA as surf

    log = io.StringIO()
    t0 = time.perf_counter()
    x, y, v, t, usage = surf.Surfin_FBA(models, x0, y0, np.zeros(len(y0)), np.zeros(len(y0)), t_end,
                                        metabolite_names=names, concurrent=False, solver="gb",
                                        initres=initres, report_activity=True, detail_activity=True,
                                        flobj=log)
    wall = time.perf_counter() - t0
    txt = log.getvalue()
    n_prep = txt.count("prep_indv_model: LP Status")
    n_fw = txt.count("find_waves: LP Status")
    reinit = re.findall(r"Required (\d+) reinitializations", txt)
    msg = re.findall(r"Surfin_FBA: (Complete|Failed[^ ]*)", txt)
    stats = {"wall_seconds": wall, "gurobi_optimize_calls": 2 * n_prep + n_fw,
             "prep_lps": 2 * n_prep, "basis_lps": n_fw,
             "reinitializations": int(reinit[-1]) if reinit else None,
             "status": msg[-1] if msg else "no status (returned None?)"}
    return x, y, t, stats, txt


def resample(t, series, t_eval):
    t = np.asarray(t, dtype=float)
    s = np.asarray(series, dtype=float)
    out = np.interp(t_eval, t, s)
    out[t_eval > t[-1]] = s[-1]          # hold the last state after an early stop
    return out


def errors(Xg, Mg, ref, names, prod, floor):
    Xr = ref["X"]
    j = [list(ref["env_mets"]).index(m) for m in prod]
    jj = [names.index(m) for m in prod]
    ex = float(np.max(np.abs(Xg - Xr) / np.maximum(np.abs(Xr), 1e-9)))
    em = float(np.max(np.abs(Mg[:, jj] - ref["M"][:, j]) / np.maximum(np.abs(ref["M"][:, j]), floor)))
    return ex, em


def run_example():
    import cobra
    import surfinFBA as surf

    np.random.seed(0)
    mdl = cobra.io.read_sbml_model(os.path.expanduser("~/isilon/cumbof/manylp_ws/bigg/e_coli_core.xml.gz"))
    mdl.name = "E.coli_e_coli_core"
    models, mets, y0 = surf.prep_cobrapy_models({"E.coli": mdl})
    models["E.coli"].deathrate = 1 + 0.2 * np.random.rand()
    x0 = {"E.coli": 1.0}
    t = time.perf_counter()
    log = io.StringIO()
    x, y, v, tt, usage = surf.Surfin_FBA(models, x0, y0, dict((k, 1) for k in mets), dict((k, 1) for k in mets), 1.0,
                                         metabolite_names=list(mets), concurrent=False, solver="gb",
                                         flobj=log, report_activity=True, detail_activity=True)
    w = time.perf_counter() - t
    xs = x["E.coli_e_coli_core"]
    rec = {"tool": "surfinFBA", "case": "package example (BiGG_model_examples.py) on e_coli_core", "wall_seconds": w,
           "n_time_points": len(tt), "t_last": float(tt[-1]), "biomass_first_last": [float(xs[0]), float(xs[-1])],
           "status": re.findall(r"Surfin_FBA: (Complete|Failed[^ ]*)", log.getvalue())[-1:],
           "gurobi_optimize_calls": 2 * log.getvalue().count("prep_indv_model: LP Status")
           + log.getvalue().count("find_waves: LP Status")}
    json.dump(rec, open(os.path.join(OUT, "surfinfba_example_e_coli_core.json"), "w"), indent=1)
    print(json.dumps(rec), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="diauxie", choices=["diauxie", "gut"])
    ap.add_argument("--example", action="store_true")
    ap.add_argument("--relax-lb", action="store_true", help="lb := min(0, lb) as prep_cobrapy_models does")
    ap.add_argument("--initres", type=float, default=1e-3)
    ap.add_argument("--ensemble", type=int, default=0)
    ap.add_argument("--members", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.example:
        return run_example()
    import cobra
    import gurobipy

    sc = json.load(open(os.path.join(OUT, f"scenario_{a.scenario}.json")))
    env = sc["env_mets"]
    t_eval = np.array(sc["t_eval"])
    M0 = np.array(sc["M0"])
    kap = np.where(M0 > 0, sc["vmax"] / (sc["km"] + M0), sc["vmax"] / sc["km"])
    tag = f"surfinfba_{a.scenario}_linear" + ("_relaxlb" if a.relax_lb else "") + (f"_E{a.ensemble}" if a.ensemble else "")
    rec = {"tool": "surfinFBA", "version": "0.9.0 (git master)", "lp_solver": f"Gurobi {gurobipy.gurobi.version()} (size-limited pip licence)",
           "integrator": "scipy.integrate.ode('vode') + basis re-initialisation", "scenario": a.scenario,
           "kinetics": "linear u_j = kappa_j y_j, kappa_j = Vmax/(Km + M0_j) (Vmax/Km if M0_j = 0)",
           "relax_lb": a.relax_lb, "initres": a.initres}
    t0 = time.perf_counter()
    if a.scenario == "diauxie":
        mdl = cobra.io.read_sbml_model(os.path.join(OUT, sc["sbml"]))
        models = [surfmod_from_cobra(mdl, sc["exchange_rxns"][s], env, sc["exchange_mets"][s], kap, nm, a.relax_lb)
                  for s, nm in enumerate(sc["species"])]
        prod, floor = ["glc__D_e", "ac_e"], 1e-2
    else:
        import glob

        files = sorted(glob.glob(os.path.join(sc["gem_dir"], "*.xml.gz")))
        models = []
        for s, nm in enumerate(sc["species"]):
            f = [x for x in files if os.path.basename(x).startswith(nm)][0]
            mdl = cobra.io.read_sbml_model(f)
            models.append(surfmod_from_cobra(mdl, sc["exchange_rxns"][s], env, sc["exchange_mets"][s], kap, nm, a.relax_lb))
        prod, floor = [m for m in ("cpd00029_e0", "cpd00211_e0", "cpd11640_e0") if m in env], 1e-3
        rec["note"] = ("gut diet caps (min(MM, max_uptake)) and influx are not representable; "
                       "kappa*y bounds only")
    rec["build_seconds"] = time.perf_counter() - t0
    rec["lp_sizes"] = [{"vars": int(m.MatrixA.shape[1]), "ineq_rows": int(m.MatrixA.shape[0]),
                        "eq_rows": int(m.Gamma2.shape[0])} for m in models]
    refp = os.path.join(OUT, f"ref_{a.scenario}_linear.npz")
    ref = np.load(refp) if os.path.exists(refp) else None

    def one(y0):
        try:
            x, y, t, st, txt = run_surfin(copy.deepcopy(models), np.array(sc["X0"]), y0, sc["t_end"], env,
                                          a.initres)   # SurfMod.statbds is mutated during a run
        except Exception as e:  # e.g. gurobipy size-limited licence
            return None, None, {"error": f"{type(e).__name__}: {e}"}
        if x is None:
            return None, None, {**st, "error": "Surfin_FBA returned None", "log_tail": txt[-2000:]}
        Xg = np.stack([resample(t, x[m.Name], t_eval) for m in models], axis=1)
        Mg = np.stack([resample(t, y[m], t_eval) for m in env], axis=1)
        st["terminated_at"] = float(t[-1])
        st["n_time_points"] = len(t)
        return Xg, Mg, st

    if not a.ensemble:
        Xg, Mg, st = one(M0.copy())
        rec.update(st)
        if Xg is not None:
            if ref is not None:
                rec["max_rel_err_biomass"], rec["max_rel_err_products"] = errors(Xg, Mg, ref, env, prod, floor)
                rec["max_rel_err_per_product"] = {m: errors(Xg, Mg, ref, env, [m], floor)[1] for m in prod}
                rec["err_floor"] = floor
                rec["reference"] = f"manylp Euler dt={float(ref['dt']):g}, same linear uptake law"
            rec["trajectory"] = {"t": t_eval.tolist(), "X": Xg.tolist(),
                                 **{m: Mg[:, env.index(m)].tolist() for m in prod}}
            np.savez(os.path.join(OUT, tag + ".npz"), times=t_eval, X=Xg, M=Mg, env_mets=np.array(env))
    else:
        E = a.ensemble
        pert = np.random.default_rng(0).lognormal(0.0, 0.3, size=(E, len(env)))
        pert[0] = 1.0
        k = a.members or E
        rows = []
        refe = os.path.join(OUT, f"ref_{a.scenario}_linear_ens.npz")
        refe = np.load(refe) if os.path.exists(refe) else None
        t1 = time.perf_counter()
        for e in range(k):
            Xg, Mg, st = one(M0 * pert[e])
            if refe is not None and Xg is not None:
                assert np.allclose(refe["perturb"][e], pert[e])
                r1 = {"X": refe["X"][:, e], "M": refe["M"][:, e], "env_mets": refe["env_mets"]}
                st["max_rel_err_biomass"], st["max_rel_err_products"] = errors(Xg, Mg, r1, env, prod, floor)
                st["max_rel_err_per_product"] = {m: errors(Xg, Mg, r1, env, [m], floor)[1] for m in prod}
            rows.append(st)
            print(json.dumps({"member": e, **{kk: st.get(kk) for kk in ("wall_seconds", "gurobi_optimize_calls",
                                                                     "status", "terminated_at", "error")}}), flush=True)
        rec.update({"E": E, "members_run": k, "total_wall_seconds": time.perf_counter() - t1,
                    "wall_per_trajectory": (time.perf_counter() - t1) / k,
                    "gurobi_optimize_calls_total": int(sum(r.get("gurobi_optimize_calls") or 0 for r in rows)),
                    "members": rows})
        errs = [r for r in rows if "max_rel_err_biomass" in r]
        if errs:
            rec["max_rel_err_biomass"] = float(max(r["max_rel_err_biomass"] for r in errs))
            rec["max_rel_err_products"] = float(max(r["max_rel_err_products"] for r in errs))
            rec["max_rel_err_per_product"] = {m: float(max(r["max_rel_err_per_product"][m] for r in errs)) for m in prod}
    json.dump(rec, open(os.path.join(OUT, tag + ".json"), "w"), indent=1)
    print(json.dumps({kk: v for kk, v in rec.items() if kk not in ("trajectory", "members")}), flush=True)


if __name__ == "__main__":
    main()
