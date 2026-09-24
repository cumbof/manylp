"""Export the benchmark scenarios for external dFBA tools and compute manylp references.

Runs in the ``manylp`` environment.  Writes to ``results/external/``:

* ``scenario_diauxie.json`` + ``e_coli_core_textbook.xml``: the E. coli core diauxie of
  ``benchmarks/bench_diauxie.py`` (two identical ``textbook`` copies, glucose 15 mM,
  O2 1e3, every exchange metabolite with Michaelis-Menten uptake Vmax = 10, Km = 0.5),
  so tools that cannot import manylp can rebuild exactly the same system;
* ``ref_diauxie.npz``: manylp ``run_dfba`` reference, explicit Euler dt = 1e-4 h
  (as bench_diauxie.py), sampled at the 121 points 0, 0.1, ..., 12 h;
* ``ref_diauxie_linear.npz`` (``--linear``): the same system with *linear* uptake
  bounds ``u_j = kappa_j M_j`` (the only kinetics surfinFBA supports), kappa_j chosen
  as the secant of the Michaelis-Menten law at t = 0 (so the first LP is identical);
* ``ref_diauxie_ens.npz`` (``--ensemble E``): Euler dt = 1e-4 references for the
  log-normally perturbed ensemble (sigma 0.3, seed 0, member 0 unperturbed);
* ``scenario_<model>.json`` / ``ref_<model>.npz`` (``--gsm iJO1366``): single-organism
  genome-scale aerobic glucose case (Euler reference, ``--gsm-dt``, 1e-4 h used);
* ``ref_gut.npz`` (``--gut``): the 12-species gut community of bench_adaptive.py,
  Euler dt = 1e-3 h over 48 h, hourly samples.
"""
import argparse
import gzip
import json
import os
import shutil
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
OUT = os.path.join(ROOT, "results", "external")

from manylp.dfba import Community, ManyLPAdapter, run_dfba, exchange_metabolites  # noqa: E402
from manylp.fba import FBAModel  # noqa: E402


def diauxie_community():
    from test_extras import _ecoli_community

    comm = _ecoli_community()
    comm.M0[comm.env_mets.index("glc__D_e")] = 15.0
    comm.M0[comm.env_mets.index("o2_e")] = 1e3
    return comm


def euler_generic(comm, uptake, E=1, t_end=12.0, dt=1e-4, record_every=1000, perturb=None,
                  min_biomass=1e-9):
    """``run_dfba`` with a pluggable uptake law ``uptake(conc, envi) -> u >= 0`` (same loop)."""
    ad = ManyLPAdapter("cpu", n_workers=8)
    ad.setup(comm.models, "pfba-unique")
    S, nM = len(comm.models), len(comm.env_mets)
    X = np.tile(comm.X0[None, :], (E, 1)).astype(float)
    M = np.tile(comm.M0[None, :], (E, 1)).astype(float)
    if perturb is not None:
        M *= perturb
    n = int(round(t_end / dt))
    rec = list(range(0, n + 1, record_every))
    Xh, Mh = np.zeros((len(rec), E, S)), np.zeros((len(rec), E, nM))
    ri, lps = 0, 0
    for step in range(n + 1):
        mu = np.zeros((E, S))
        fe = np.zeros((E, nM))
        for s in range(S):
            act = np.nonzero(X[:, s] > min_biomass)[0]
            if act.size == 0:
                continue
            envi = comm.ex_env[s]
            conc = M[np.ix_(act, envi)]
            u = np.minimum(uptake(conc, envi), conc / (X[act, s][:, None] * dt))
            g, v, _ = ad.solve(s, -u, act)
            lps += act.size
            mu[act, s] = g
            fe[np.ix_(act, envi)] += v * X[act, s][:, None]
        if ri < len(rec) and rec[ri] == step:
            Xh[ri], Mh[ri] = X, M
            ri += 1
        if step == n:
            break
        X = np.maximum(0.0, X + mu * X * dt)
        M = np.maximum(0.0, M + fe * dt)
    ad.close()
    return np.array(rec) * dt, Xh, Mh, lps


def linear_kappas(comm):
    """Secant of the MM law at t = 0 (slope Vmax/Km where M0 = 0)."""
    M0 = comm.M0
    return np.where(M0 > 0, comm.vmax / (comm.km + M0), comm.vmax / comm.km)


def save_scenario(comm, name, sbml_src, extra=None):
    os.makedirs(OUT, exist_ok=True)
    d = {"env_mets": comm.env_mets, "M0": comm.M0.tolist(), "X0": comm.X0.tolist(),
         "species": [m.name for m in comm.models], "vmax": comm.vmax, "km": comm.km,
         "influx": comm.influx.tolist(), "dilution": 0.0, "min_biomass": 1e-9,
         "mode": "pfba-unique", "sbml": os.path.basename(sbml_src),
         "exchange_rxns": [[m.rxn_ids[j] for j in m.exchanges] for m in comm.models],
         "exchange_mets": [exchange_metabolites(m) for m in comm.models]}
    if extra:
        d.update(extra)
    json.dump(d, open(os.path.join(OUT, f"scenario_{name}.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diauxie", action="store_true")
    ap.add_argument("--linear", action="store_true")
    ap.add_argument("--ensemble", type=int, default=0)
    ap.add_argument("--linear-ensemble", type=int, default=0)
    ap.add_argument("--gsm", default="")
    ap.add_argument("--gsm-dt", type=float, default=1e-3)
    ap.add_argument("--gut", action="store_true")
    ap.add_argument("--scenario-only", action="store_true", help="with --gut: write the scenario, skip the reference")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    comm = diauxie_community()
    j = [comm.env_mets.index(m) for m in ("glc__D_e", "ac_e")]

    if a.diauxie:
        import cobra
        from cobra.io import load_model

        sbml = os.path.join(OUT, "e_coli_core_textbook.xml")
        cobra.io.write_sbml_model(load_model("textbook"), sbml)
        save_scenario(comm, "diauxie", sbml, {"t_end": 12.0, "t_eval": np.linspace(0, 12, 121).tolist(),
                                              "linear_kappa": linear_kappas(comm).tolist()})
        t = time.perf_counter()
        r = run_dfba(comm, ManyLPAdapter("cpu", n_workers=8), E=1, t_end=12.0, dt=1e-4, record_every=1000)
        w = time.perf_counter() - t
        np.savez(os.path.join(OUT, "ref_diauxie.npz"), times=r.times, X=r.X[:, 0], M=r.M[:, 0],
                 env_mets=np.array(comm.env_mets), dt=1e-4, n_lps=r.n_lps, wall=w)
        print(f"ref_diauxie: {r.n_lps} LPs {w:.1f}s  final glc/ac {r.M[-1, 0, j]}  X {r.X[-1, 0]}", flush=True)

    if a.linear:
        kap = linear_kappas(comm)
        t = time.perf_counter()
        tt, X, M, lps = euler_generic(comm, lambda c, envi: np.where(c > 0, kap[envi][None, :] * c, 0.0))
        w = time.perf_counter() - t
        np.savez(os.path.join(OUT, "ref_diauxie_linear.npz"), times=tt, X=X[:, 0], M=M[:, 0],
                 env_mets=np.array(comm.env_mets), dt=1e-4, n_lps=lps, wall=w, kappa=kap)
        print(f"ref_diauxie_linear: {lps} LPs {w:.1f}s final glc/ac {M[-1, 0, j]} X {X[-1, 0]}", flush=True)

    if a.linear_ensemble:
        E = a.linear_ensemble
        kap = linear_kappas(comm)
        pert = np.random.default_rng(0).lognormal(0.0, 0.3, size=(E, len(comm.env_mets)))
        pert[0] = 1.0
        t = time.perf_counter()
        tt, X, M, lps = euler_generic(comm, lambda c, envi: np.where(c > 0, kap[envi][None, :] * c, 0.0),
                                      E=E, perturb=pert)
        w = time.perf_counter() - t
        np.savez(os.path.join(OUT, "ref_diauxie_linear_ens.npz"), times=tt, X=X, M=M, perturb=pert,
                 env_mets=np.array(comm.env_mets), dt=1e-4, n_lps=lps, wall=w, kappa=kap)
        print(f"ref_diauxie_linear_ens E={E}: {lps} LPs {w:.1f}s", flush=True)

    if a.ensemble:
        E = a.ensemble
        pert = np.random.default_rng(0).lognormal(0.0, 0.3, size=(E, len(comm.env_mets)))
        pert[0] = 1.0
        t = time.perf_counter()
        r = run_dfba(comm, ManyLPAdapter("cpu", n_workers=8), E=E, t_end=12.0, dt=1e-4,
                     record_every=1000, perturb=pert)
        w = time.perf_counter() - t
        np.savez(os.path.join(OUT, "ref_diauxie_ens.npz"), times=r.times, X=r.X, M=r.M, perturb=pert,
                 env_mets=np.array(comm.env_mets), dt=1e-4, n_lps=r.n_lps, wall=w)
        print(f"ref_diauxie_ens E={E}: {r.n_lps} LPs {w:.1f}s", flush=True)

    if a.gsm:
        import cobra

        src = os.path.expanduser(f"~/isilon/cumbof/manylp_ws/bigg/{a.gsm}.xml.gz")
        mdl = cobra.io.read_sbml_model(src)
        fm = FBAModel.from_cobra(mdl)
        fm.name = a.gsm
        env = sorted(set(exchange_metabolites(fm)))
        # default BiGG medium: every exchange open for uptake at 1e3 mM (MM bound ~ Vmax),
        # glucose 15 mM, oxygen 1e3 mM (bound ~ 10), everything else absent
        med = {r.id: v for r, v in ((mdl.reactions.get_by_id(k), v) for k, v in mdl.medium.items())}
        conc = {}
        for rid in med:
            for mt in mdl.reactions.get_by_id(rid).metabolites:
                conc[mt.id] = 1e3
        conc["glc__D_e"] = 15.0
        conc["o2_e"] = 1e3
        M0 = np.array([conc.get(m, 0.0) for m in env])
        gcomm = Community(models=[fm], env_mets=env, M0=M0, influx=np.zeros(len(env)),
                          max_uptake=np.full(len(env), np.inf), X0=np.array([0.03]), vmax=10.0, km=0.5)
        dst = os.path.join(OUT, f"{a.gsm}.xml.gz")
        shutil.copy(src, dst)
        save_scenario(gcomm, a.gsm, dst, {"t_end": 12.0, "t_eval": np.linspace(0, 12, 121).tolist()})
        t = time.perf_counter()
        r = run_dfba(gcomm, ManyLPAdapter("cpu", n_workers=8), E=1, t_end=12.0, dt=a.gsm_dt,
                     record_every=int(round(0.1 / a.gsm_dt)))
        w = time.perf_counter() - t
        np.savez(os.path.join(OUT, f"ref_{a.gsm}.npz"), times=r.times, X=r.X[:, 0], M=r.M[:, 0],
                 env_mets=np.array(env), dt=a.gsm_dt, n_lps=r.n_lps, wall=w)
        jj = [env.index(m) for m in ("glc__D_e", "ac_e")]
        print(f"ref_{a.gsm}: {r.n_lps} LPs {w:.1f}s final glc/ac {r.M[-1, 0, jj]} X {r.X[-1, 0]}", flush=True)

    if a.gut:
        from manylp.dfba import load_gut_community

        G = os.path.expanduser("~/isilon/cumbof/manylp_ws/muODE/examples/gut_western")
        gc = load_gut_community(f"{G}/gems", f"{G}/gems/western_gut_modelseed.csv", abundance_tsv=f"{G}/abundance.tsv")
        save_scenario(gc, "gut", f"{G}/gems", {"t_end": 48.0, "t_eval": np.arange(49.0).tolist(), "gem_dir": f"{G}/gems",
                                               "max_uptake": [x if np.isfinite(x) else None for x in gc.max_uptake]})
        if a.scenario_only:
            return
        t = time.perf_counter()
        r = run_dfba(gc, ManyLPAdapter("cpu", n_workers=32), E=1, t_end=48.0, dt=1e-3, record_every=1000)
        w = time.perf_counter() - t
        np.savez(os.path.join(OUT, "ref_gut.npz"), times=r.times, X=r.X[:, 0], M=r.M[:, 0],
                 env_mets=np.array(gc.env_mets), species=np.array([m.name for m in gc.models]),
                 dt=1e-3, n_lps=r.n_lps, wall=w)
        print(f"ref_gut: {r.n_lps} LPs {w:.1f}s", flush=True)


if __name__ == "__main__":
    main()
