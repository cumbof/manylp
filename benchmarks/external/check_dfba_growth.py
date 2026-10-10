"""Growth rate of the dfba package's event (Harwood) method on iJO1366 against the LP optimum.

Solves the iJO1366 LP at t = 0 with cobra/GLPK under the same uptake bounds, then integrates the
first 0.003 h with the dfba package and prints the growth rate it reports.  This is the source of
the paper's "0.62 h-1 against the true 0.70 h-1".  Needs the dfba package (see run_dfba_pkg.py)
and the iJO1366 scenario written by make_references.py --gsm iJO1366.

    python benchmarks/external/check_dfba_growth.py
"""
import sys, json, os, time, numpy as np
sys.path.insert(0, "benchmarks/external")
import run_dfba_pkg as R
sc = json.load(open(os.path.join(R.OUT, "scenario_iJO1366.json")))
mdl, objs, dirs, bio = R.build_cobra(sc, "fba")
M0 = np.array(sc["M0"])
# cobra LP at t=0
with mdl:
    for r, m in zip(sc["exchange_rxns"][0], sc["exchange_mets"][0]):
        c = M0[sc["env_mets"].index(m)]
        mdl.reactions.get_by_id(r).lower_bound = -10 * c / (0.5 + c)
    s = mdl.optimize()
    print("cobra glpk mu0", s.objective_value, bio, s.fluxes[bio], "EX_ac", s.fluxes["EX_ac_e"], "EX_glc", s.fluxes["EX_glc__D_e"], flush=True)
d = R.build_dfba(mdl, objs, dirs, bio, sc, M0, 0.03, 1e-6, 1e-6, "Harwood")
d.solver_data.set_display("none")
t = time.perf_counter()
conc, traj = d.simulate(0.0, 0.003, 0.001, [bio, "EX_glc__D_e", "EX_ac_e", "EX_o2_e"])
print(time.perf_counter() - t, flush=True)
print(conc[["time", "Biomass__X", "M__glc__D_e", "M__ac_e"]])
print(traj)
