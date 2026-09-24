"""Run the `dfba` package (Tourigny, Carrasco Muriel & Beber 2020, JOSS) on manylp's scenarios.

`dfba` integrates the dFBA DAE with SUNDIALS IDA and re-solves the LP (GLPK simplex)
only when the current optimal basis stops being primal feasible (Harwood et al. 2016;
``--alg Harwood``), or -- ``--alg direct`` -- re-solves the LP once per output interval
``tout`` and integrates with CVODE holding the fluxes fixed in between (a static-
optimisation scheme with step ``tout``).  Runs in the ``dfbaenv`` conda env
(python 3.8, ``conda install -c conda-forge dfba``); it does NOT import manylp: the
scenario is read from ``results/external/scenario_<name>.json`` written by
``make_references.py``.

Same ODE as manylp's ``run_dfba`` (continuous-time limit):

    dX/dt   = mu X,      dM_j/dt = v_j X,
    lb_j    = -Vmax M_j / (Km + M_j)   for every exchange j (upper bounds = model's),

with the E. coli diauxie's two identical species merged into one (X0 = 0.02 + 0.01;
identical LPs give identical mu and per-biomass fluxes, so the total biomass obeys the
same ODE and each species stays at its initial fraction).

Flux selection (``--mode``):
  fba          maximise growth only (dfba's default; exchange fluxes may be a non-unique vertex)
  pfba         max growth, then min ||v||_1          (manylp's L1 split: reversible + exchanges)
  pfba-unique  ... then max w^T z, w ~ U(-1,1), seed 0 (manylp's default tie-break, same w)
  pfba-weighted  single objective  max c^T v - eps ||v||_1  (eps = 1e-5)
  pfba-unique-weighted  ... + eps2 w^T z  (eps2 = eps/100, manylp's w): approximates pfba-unique
The pfba / pfba-unique stages use dfba's lexicographic objectives on auxiliary sink
reactions (pseudo-metabolites carry ||v||_1 and w^T z), so no dfba code is changed.
NOTE: dfba 0.1.8's lexicographic mode stalls on this problem (the stage-1 objective row
is fixed at a constant value, so the event functions fire again after every IDA step;
observed: thousands of one-step re-initialisations near t = 0).  The weighted modes are
the working way to obtain pFBA fluxes with dfba; ``pfba-unique-weighted`` (default) is
needed on iJO1366, whose pFBA optimum is not unique in the exchange fluxes (GLPK then
returns a vertex with formate 2.0 / acetate 8.4 instead of manylp's 6.1 / 7.4 at t = 0).
Results for eps = 1e-6 ... 1e-4 are identical on the diauxie; eps = 1e-3 is too large.

Known dfba 0.1.8 issues seen here: on iJO1366 the Harwood path gives a wrong growth rate
(0.62 vs 0.70 1/h at t = 0.001 h; 0.25 1/h average over 0.1 h) and is slow (dense
~3000x3000 IDA Jacobian), the direct path is correct; rtol = atol = 1e-10 fails with IDA
"mxstep" at t = 6.35 h; the state at an infeasible-LP stop is only reported at the last
multiple of ``tout`` (hence the fine default ``--tout 1e-3``); the per-run JIT build
needs TMPDIR on a local disk (NFS leaves .nfs files and TemporaryDirectory cleanup fails).

Outputs ``results/external/dfba_<scenario>_<mode>[eps]_<alg>_rtol<r>[_E<E>][_T<h>][_tout<t>]``
``.json`` (+ ``.npz``).
"""
import argparse
import ctypes
import json
import os
import re
import sys
import tempfile
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "results", "external")


# ----------------------------------------------------------------------------- model
def build_cobra(sc, mode, seed=0, eps=1e-5):
    import cobra

    mdl = cobra.io.read_sbml_model(os.path.join(OUT, sc["sbml"]))
    mdl.solver = "glpk"
    rx = list(mdl.reactions)
    n = len(rx)
    ex_ids = set(sc["exchange_rxns"][0])
    lb = np.array([r.lower_bound for r in rx])
    ub = np.array([r.upper_bound for r in rx])
    is_ex = np.array([r.id in ex_ids for r in rx])
    objs, dirs = [], []
    bio = [r.id for r in rx if r.objective_coefficient != 0]
    assert len(bio) == 1, bio
    objs.append(bio[0]); dirs.append("max")
    if mode in ("pfba-weighted", "pfba-unique-weighted"):
        # single objective  max c^T v - eps ||v||_1  (manylp's L1 split); equals the lexicographic
        # pFBA optimum for eps below an LP-dependent threshold (checked against the reference)
        R = np.nonzero(((lb < 0) & (ub > 0)) | is_ex)[0]
        l1 = np.where(ub <= 0, -1.0, 1.0)
        l1[(lb == 0) & (ub == 0)] = 0.0
        inR = np.zeros(n, dtype=bool)
        inR[R] = True
        # manylp's generic tie-break w (same seed and column order), weighted eps2 = eps * 1e-2
        w = (np.random.default_rng(seed).uniform(-1.0, 1.0, size=n + R.size)
             if mode == "pfba-unique-weighted" else np.zeros(n + R.size))
        eps2 = eps * 1e-2
        coef = {r: float(r.objective_coefficient) for r in rx if r.objective_coefficient != 0}
        for j, r in enumerate(rx):
            if not inR[j] and (l1[j] != 0 or w[j] != 0):
                coef[r] = coef.get(r, 0.0) - eps * float(l1[j]) + eps2 * float(w[j])
        new = []
        for j in R:
            r = rx[j]
            P = cobra.Metabolite(f"P{j}__pseudo", compartment="pseudo")
            r.add_metabolites({P: 1.0})                       # P balance: v_j = v+ - v-
            F = cobra.Reaction(f"F{j}__pseudo", lower_bound=0.0, upper_bound=max(ub[j], 0.0) if not is_ex[j] else 1e4)
            B = cobra.Reaction(f"B{j}__pseudo", lower_bound=0.0, upper_bound=max(-lb[j], 0.0) if not is_ex[j] else 1e4)
            F.add_metabolites({P: -1.0})
            B.add_metabolites({P: 1.0})
            new += [F, B]
        mdl.add_reactions(new)
        for k, j in enumerate(R):
            coef[new[2 * k]] = -eps + eps2 * float(w[j])            # v+ (column j in manylp)
            coef[new[2 * k + 1]] = -eps + eps2 * float(w[n + k])    # v- (column n + k)
        mdl.objective = coef
        mdl.objective_direction = "max"
        return mdl, [bio[0]], ["max"], bio[0]
    if mode in ("pfba", "pfba-unique"):
        # manylp.fba.compile_fba: split R = reversible or exchange as v = v+ - v-
        R = np.nonzero(((lb < 0) & (ub > 0)) | is_ex)[0]
        l1 = np.where(ub <= 0, -1.0, 1.0)
        l1[(lb == 0) & (ub == 0)] = 0.0
        n2 = n + R.size
        w = np.random.default_rng(seed).uniform(-1.0, 1.0, size=n2) if mode == "pfba-unique" else None
        L1 = cobra.Metabolite("L1__pseudo", compartment="pseudo")
        TB = cobra.Metabolite("TB__pseudo", compartment="pseudo")
        inR = np.zeros(n, dtype=bool)
        inR[R] = True
        for j, r in enumerate(rx):
            if not inR[j]:
                st = {}
                if l1[j] != 0:
                    st[L1] = float(l1[j])
                if w is not None and w[j] != 0:
                    st[TB] = float(w[j])
                if st:
                    r.add_metabolites(st)
        new = []
        for k, j in enumerate(R):
            r = rx[j]
            P = cobra.Metabolite(f"P{j}__pseudo", compartment="pseudo")
            r.add_metabolites({P: 1.0})                       # P balance: v_j - v+ + v- = 0
            F = cobra.Reaction(f"F{j}__pseudo", lower_bound=0.0, upper_bound=max(ub[j], 0.0) if not is_ex[j] else 1e4)
            B = cobra.Reaction(f"B{j}__pseudo", lower_bound=0.0, upper_bound=max(-lb[j], 0.0) if not is_ex[j] else 1e4)
            fs, bs = {P: -1.0, L1: 1.0}, {P: 1.0, L1: 1.0}
            if w is not None:
                fs[TB] = float(w[j])
                bs[TB] = float(w[n + k])
            F.add_metabolites(fs)
            B.add_metabolites(bs)
            new += [F, B]
        s1 = cobra.Reaction("L1_SUM__pseudo", lower_bound=0.0, upper_bound=1e7)
        s1.add_metabolites({L1: -1.0})
        new.append(s1)
        objs.append(s1.id); dirs.append("min")
        if w is not None:
            s2 = cobra.Reaction("TIE__pseudo", lower_bound=-1e7, upper_bound=1e7)
            s2.add_metabolites({TB: -1.0})
            new.append(s2)
            objs.append(s2.id); dirs.append("max")
        mdl.add_reactions(new)
    return mdl, objs, dirs, bio[0]


def build_dfba(mdl, objs, dirs, bio, sc, M0, X0, rtol, atol, alg):
    from dfba import DfbaModel, ExchangeFlux, KineticVariable

    d = DfbaModel(mdl)
    X = KineticVariable("Biomass__X", initial_condition=float(X0))
    mets = sc["env_mets"]
    kv = {m: KineticVariable(f"M__{m}", initial_condition=float(c)) for m, c in zip(mets, M0)}
    d.add_kinetic_variables([X] + list(kv.values()))
    mu = ExchangeFlux(bio)
    fl = {r: ExchangeFlux(r) for r in sc["exchange_rxns"][0]}
    d.add_exchange_fluxes([mu] + list(fl.values()))
    d.add_rhs_expression("Biomass__X", mu * X)
    for r, m in zip(sc["exchange_rxns"][0], sc["exchange_mets"][0]):
        d.add_rhs_expression(f"M__{m}", fl[r] * X)
    vmax, km = sc["vmax"], sc["km"]
    for r, m in zip(sc["exchange_rxns"][0], sc["exchange_mets"][0]):
        d.add_exchange_flux_lb(r, vmax * (kv[m] / (km + kv[m])), kv[m])
    if len(objs) > 1:
        d.add_objectives(objs, dirs)
    d.solver_data.set_rel_tolerance(rtol)
    d.solver_data.set_abs_tolerance([atol])
    d.solver_data.set_algorithm(alg)
    d.solver_data.set_display("full")          # GLPK + IDA statistics, parsed for LP counts
    return d


class FdCapture:
    """Capture the C/C++ stdout of the extension (GLPK and IDA print there)."""

    def __enter__(self):
        self.libc = ctypes.CDLL(None)
        sys.stdout.flush()
        sys.stderr.flush()
        self.tmp = tempfile.TemporaryFile(mode="w+b")
        self.tmp2 = tempfile.TemporaryFile(mode="w+b")
        self.saved, self.saved2 = os.dup(1), os.dup(2)
        os.dup2(self.tmp.fileno(), 1)
        os.dup2(self.tmp2.fileno(), 2)
        return self

    def __exit__(self, *a):
        self.libc.fflush(None)
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(self.saved, 1)
        os.dup2(self.saved2, 2)
        os.close(self.saved)
        os.close(self.saved2)
        self.tmp.seek(0)
        self.text = self.tmp.read().decode(errors="replace")
        self.tmp2.seek(0)
        err = self.tmp2.read().decode(errors="replace")
        self.err = "\n".join(ln for ln in err.splitlines() if "SId" not in ln and ln.strip())
        self.tmp.close()
        self.tmp2.close()


def simulate_one(sc, mode, M0, X0, rtol, atol, alg, t_end, tout, eps=1e-5):
    import dfba.model as dm

    t0 = time.perf_counter()
    mdl, objs, dirs, bio = build_cobra(sc, mode, eps=eps)
    d = build_dfba(mdl, objs, dirs, bio, sc, M0, X0, rtol, atol, alg)
    t_build = time.perf_counter() - t0
    jit = {"s": 0.0}
    orig = dm.jit.compile

    def timed_compile(directory):
        t = time.perf_counter()
        try:
            return orig(directory)
        finally:
            jit["s"] += time.perf_counter() - t

    dm.jit.compile = timed_compile
    try:
        t1 = time.perf_counter()
        import swiglpk

        swiglpk.glp_term_out(swiglpk.GLP_ON)   # optlang silences GLPK; we count its simplex calls
        with FdCapture() as cap:
            conc, _ = d.simulate(0.0, t_end, tout)
        swiglpk.glp_term_out(swiglpk.GLP_OFF)
        t_sim = time.perf_counter() - t1
    finally:
        dm.jit.compile = orig
    log = cap.text
    stats = {
        "stderr_tail": cap.err[-400:],
        "build_seconds": t_build,
        "simulate_seconds": t_sim,                    # includes JIT C++ compilation
        "jit_seconds": jit["s"],
        "integrate_seconds": t_sim - jit["s"],        # add_to_library + IDA + GLPK
        "cpu_seconds_reported": float(re.findall(r"Total simulation time was ([0-9.eE+-]+)", log)[-1])
        if "Total simulation time" in log else None,
        # GLPK prints one terminal status line per glp_simplex call
        "glpk_simplex_calls": len(re.findall(r"OPTIMAL (?:LP )?SOLUTION FOUND|HAS NO (?:PRIMAL|DUAL) FEASIBLE", log)),
        "lp_reinitialisations": log.count("Final Run Statistics"),
        "infeasible_stop": "Basis not feasible" in log,
        "ida_steps": int(sum(int(x) for x in re.findall(r"Number of steps\s*=\s*(\d+)", log))),
        "ida_residual_evals": int(sum(int(x) for x in re.findall(r"Number of residual evaluations\s*=\s*(\d+)", log))),
    }
    return conc, stats, log


def to_grid(conc, sc, t_eval):
    """Resample onto t_eval; after an early stop (infeasible LP) hold the last output state
    (at most ``tout`` before the stop), which is what manylp does (mu = 0, zero fluxes)."""
    t = conc["time"].values
    X = conc["Biomass__X"].values
    M = np.stack([conc[f"M__{m}"].values for m in sc["env_mets"]], axis=1)
    Xg = np.zeros(len(t_eval))
    Mg = np.zeros((len(t_eval), M.shape[1]))
    for i, te in enumerate(t_eval):
        k = int(np.argmin(np.abs(t - te)))
        if te > t[-1] + 1e-9:
            k = len(t) - 1
        Xg[i], Mg[i] = X[k], M[k]
    return Xg, Mg, float(t[-1])


def errors(Xg, Mg, ref, sc, prod):
    Xr = (ref["X"].sum(axis=1) if ref["X"].ndim == 2 else ref["X"])[: len(Xg)]
    Mr = ref["M"][: len(Xg)]
    j = [list(ref["env_mets"]).index(m) for m in prod]
    jj = [sc["env_mets"].index(m) for m in prod]
    ex = float(np.max(np.abs(Xg - Xr) / np.maximum(np.abs(Xr), 1e-9)))
    em = float(np.max(np.abs(Mg[:, jj] - Mr[:, j]) / np.maximum(np.abs(Mr[:, j]), 1e-2)))
    return ex, em


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="diauxie")
    ap.add_argument("--mode", default="pfba-unique-weighted", choices=["fba", "pfba", "pfba-unique", "pfba-weighted", "pfba-unique-weighted"])
    ap.add_argument("--alg", default="Harwood", choices=["Harwood", "direct"])
    ap.add_argument("--rtol", type=float, default=1e-4)
    ap.add_argument("--atol", type=float, default=1e-4)
    ap.add_argument("--ensemble", type=int, default=0, help="E log-normally perturbed members (loop)")
    ap.add_argument("--members", type=int, default=0, help="run only the first k members")
    ap.add_argument("--keep-log", action="store_true")
    ap.add_argument("--tout", type=float, default=1e-3,
                    help="dfba output interval (fine, so the state at an early infeasible stop is captured)")
    ap.add_argument("--eps", type=float, default=1e-5, help="L1 weight for --mode pfba-weighted")
    ap.add_argument("--t-end", type=float, default=0.0, help="override the scenario horizon (h)")
    a = ap.parse_args()
    import dfba

    sc = json.load(open(os.path.join(OUT, f"scenario_{a.scenario}.json")))
    t_eval = np.array(sc["t_eval"])
    t_end = sc["t_end"]
    if a.t_end:
        t_end = a.t_end
        t_eval = t_eval[t_eval <= t_end + 1e-9]
    X0 = float(np.sum(sc["X0"]))
    M0 = np.array(sc["M0"])
    prod = ["glc__D_e", "ac_e"]
    tag = (f"dfba_{a.scenario}_{a.mode}" + (f"{a.eps:g}" if a.mode.endswith("weighted") else "")
           + f"_{a.alg}_rtol{a.rtol:g}" + (f"_E{a.ensemble}" if a.ensemble else "")
           + (f"_T{a.t_end:g}" if a.t_end else "") + (f"_tout{a.tout:g}" if a.tout != 1e-3 else ""))
    rec = {"tool": "dfba", "version": dfba.__version__, "lp_solver": "GLPK (simplex, via swiglpk/optlang)",
           "integrator": ("SUNDIALS IDA DAE, LP re-solved at basis-infeasibility events (Harwood)" if a.alg == "Harwood"
                          else f"SUNDIALS CVODE, LP re-solved every tout = {a.tout:g} h (direct)"),
           "tout": a.tout, "scenario": a.scenario, "mode": a.mode,
           "rtol": a.rtol, "atol": a.atol, "t_end": t_end,
           "eps": a.eps if a.mode.endswith("weighted") else None}

    if not a.ensemble:
        conc, st, log = simulate_one(sc, a.mode, M0, X0, a.rtol, a.atol, a.alg, t_end, a.tout, a.eps)
        Xg, Mg, t_last = to_grid(conc, sc, t_eval)
        rec.update(st)
        rec["terminated_at"] = t_last
        rec["wall_seconds"] = st["build_seconds"] + st["simulate_seconds"]
        refp = os.path.join(OUT, f"ref_{a.scenario}.npz")
        if os.path.exists(refp):
            ref = np.load(refp)
            rec["max_rel_err_biomass"], rec["max_rel_err_glc_ac"] = errors(Xg, Mg, ref, sc, prod)
            rec["max_rel_err_per_product"] = {m: errors(Xg, Mg, ref, sc, [m])[1] for m in prod}
            rec["reference"] = f"manylp run_dfba Euler dt={float(ref['dt']):g}"
        jj = [sc["env_mets"].index(m) for m in prod]
        rec["trajectory"] = {"t": t_eval.tolist(), "biomass_total": Xg.tolist(),
                             "glc__D_e": Mg[:, jj[0]].tolist(), "ac_e": Mg[:, jj[1]].tolist()}
        np.savez(os.path.join(OUT, tag + ".npz"), times=t_eval, X=Xg, M=Mg, env_mets=np.array(sc["env_mets"]))
        if a.keep_log:
            open(os.path.join(OUT, tag + ".log"), "w").write(log)
    else:
        E = a.ensemble
        pert = np.random.default_rng(0).lognormal(0.0, 0.3, size=(E, len(sc["env_mets"])))
        pert[0] = 1.0
        refp = os.path.join(OUT, f"ref_{a.scenario}_ens.npz")
        ref = np.load(refp) if os.path.exists(refp) else None
        if ref is not None:
            assert np.allclose(ref["perturb"], pert)
        k = a.members or E
        rows, Xs, Ms = [], [], []
        t0 = time.perf_counter()
        for e in range(k):
            conc, st, _ = simulate_one(sc, a.mode, M0 * pert[e], X0, a.rtol, a.atol, a.alg, t_end, a.tout, a.eps)
            Xg, Mg, t_last = to_grid(conc, sc, t_eval)
            st["terminated_at"] = t_last
            st["wall_seconds"] = st["build_seconds"] + st["simulate_seconds"]
            if ref is not None:
                r1 = {"X": ref["X"][:, e], "M": ref["M"][:, e], "env_mets": ref["env_mets"]}
                st["max_rel_err_biomass"], st["max_rel_err_glc_ac"] = errors(Xg, Mg, r1, sc, prod)
            rows.append(st)
            Xs.append(Xg)
            Ms.append(Mg)
            print(json.dumps({"member": e, **{kk: st[kk] for kk in ("wall_seconds", "jit_seconds", "lp_reinitialisations",
                                                                  "glpk_simplex_calls", "terminated_at")},
                              "err": [st.get("max_rel_err_biomass"), st.get("max_rel_err_glc_ac")]}), flush=True)
        tot = time.perf_counter() - t0
        rec.update({"E": E, "members_run": k, "total_wall_seconds": tot, "wall_per_trajectory": tot / k,
                    "jit_per_trajectory": float(np.mean([r["jit_seconds"] for r in rows])),
                    "wall_per_trajectory_excl_jit": float(np.mean([r["wall_seconds"] - r["jit_seconds"] for r in rows])),
                    "lp_reinitialisations_total": int(sum(r["lp_reinitialisations"] for r in rows)),
                    "glpk_simplex_calls_total": int(sum(r["glpk_simplex_calls"] for r in rows)),
                    "members": rows})
        if ref is not None:
            rec["max_rel_err_biomass"] = float(max(r["max_rel_err_biomass"] for r in rows))
            rec["max_rel_err_glc_ac"] = float(max(r["max_rel_err_glc_ac"] for r in rows))
        np.savez(os.path.join(OUT, tag + ".npz"), times=t_eval, X=np.array(Xs), M=np.array(Ms), perturb=pert[:k])
    json.dump(rec, open(os.path.join(OUT, tag + ".json"), "w"), indent=1)
    print(json.dumps({kk: v for kk, v in rec.items() if kk not in ("trajectory", "members")}), flush=True)


if __name__ == "__main__":
    main()
