"""Data for the metabolic phase diagram: E. coli core critical regions over (glucose, acetate) uptake.

The diauxie scenario of bench_diauxie.py, restricted so that the trajectory lies exactly in a plane:
only glucose, acetate, oxygen, ammonium, phosphate, water and protons can be taken up (other
products are secreted but not re-consumed, as in Mahadevan et al. 2002), and oxygen, ammonium and
phosphate are in excess, so only the glucose and acetate uptake bounds move.  This script

1. solves a G x G grid of the (glucose, acetate) uptake-bound plane under the pFBA-unique rule
   with every other exchange bound fixed at its value at t = 0, and records for each grid point
   the certified basis, the growth rate and the sign pattern of the exchange fluxes;
2. integrates the diauxie trajectory (Euler, dt = 0.01 h) for a small ensemble with perturbed
   initial glucose and acetate, and records the uptake bounds each species saw at every step;
3. certifies every trajectory point with the same solver, so trajectory points and grid points
   share basis ids.

    python benchmarks/hero_phase.py --grid 300 --E 12 --out results/hero/phase.npz
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, "tests")


def main():
    from test_extras import _ecoli_community

    from manylp import BatchLPSolver
    from manylp.dfba import ManyLPAdapter, exchange_metabolites, run_dfba
    from manylp.fba import compile_fba

    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=300)
    ap.add_argument("--E", type=int, default=12)
    ap.add_argument("--cmin", type=float, default=1e-3, help="lowest concentration on the axes (mM)")
    ap.add_argument("--cmax", type=float, default=30.0)
    ap.add_argument("--out", default="results/hero/phase.npz")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    comm = _ecoli_community()
    comm.M0[comm.env_mets.index("glc__D_e")] = 15.0
    for m in ("o2_e", "nh4_e", "pi_e"):
        comm.M0[comm.env_mets.index(m)] = 1e3
    up = ("glc__D_e", "ac_e", "o2_e", "nh4_e", "pi_e", "h2o_e", "h_e")
    comm.max_uptake = np.array([np.inf if m in up else 0.0 for m in comm.env_mets])
    model = comm.models[0]
    mets = exchange_metabolites(model)
    ig, ia = mets.index("glc__D_e"), mets.index("ac_e")

    # --- trajectories: ensemble with perturbed initial glucose and acetate -------------------
    rng = np.random.default_rng(0)
    pert = np.ones((a.E, len(comm.env_mets)))
    pert[1:, comm.env_mets.index("glc__D_e")] = rng.lognormal(0.0, 0.35, a.E - 1)
    dt = 0.01
    r = run_dfba(comm, ManyLPAdapter("cpu", n_workers=8), E=a.E, t_end=12.0, dt=dt, perturb=pert)
    envi = comm.ex_env[0]
    conc = r.M[:, :, envi]                                     # (T, E, n_ex)
    X0 = np.maximum(r.X[:, :, 0], 1e-30)[..., None]
    mm = np.where(conc > 0, comm.vmax * conc / (comm.km + conc), 0.0)
    mm = np.minimum(mm, comm.max_uptake[envi])
    traj_lb = -np.minimum(mm, conc / (X0 * dt))                # bounds species 0 saw, (T, E, n_ex)
    other = np.ones(len(mets), dtype=bool)
    other[[ig, ia]] = False
    drift = np.abs(traj_lb[:, :, other] - traj_lb[:1, :1, other]).max(axis=(0, 1))
    print("largest drift of the other uptake bounds:",
          sorted(zip(drift.round(3), np.array(mets)[other]), reverse=True)[:5], flush=True)

    # --- the grid --------------------------------------------------------------------------
    prob = compile_fba(model, mode="pfba-unique")
    ex = model.exchanges
    solver = BatchLPSolver(device="cpu", n_workers=16)
    g = solver.register_group(prob.lp, out_z=prob.out_z(ex))
    # axes are concentrations (log-spaced); a concentration C gives the uptake bound
    # vmax C / (km + C), as in the simulation
    G = a.grid
    ax = np.concatenate([[0.0], np.geomspace(a.cmin, a.cmax, G - 1)])   # include zero (symlog axes)
    mm_of = lambda c: comm.vmax * c / (comm.km + c)              # noqa: E731
    gg, aa = np.meshgrid(mm_of(ax), mm_of(ax), indexing="xy")  # rows: acetate, cols: glucose
    base = traj_lb[0, 0].copy()
    lb = np.tile(base, (G * G, 1))
    lb[:, ig], lb[:, ia] = -gg.ravel(), -aa.ravel()
    Lp, Up = prob.param_bounds(lb)
    t = time.perf_counter()
    sol = solver.solve_batch(g, Lp, Up)
    grid_s = time.perf_counter() - t
    flux = prob.fluxes(sol.z, ex)
    print(f"grid {G}x{G}: {G * G} LPs in {grid_s:.2f}s, simplex {g.totals.get('highs_solves')}, "
          f"bases {len(g.pool.bases)}, unique {bool(np.all(sol.unique[sol.status == 1]))}", flush=True)

    # --- trajectory points through the same solver ----------------------------------------
    T, E = traj_lb.shape[:2]
    tl = traj_lb.reshape(T * E, -1)
    tl_proj = np.tile(base, (T * E, 1))
    tl_proj[:, ig], tl_proj[:, ia] = tl[:, ig], tl[:, ia]
    Lq, Uq = prob.param_bounds(tl_proj)
    st = solver.solve_batch(g, Lq, Uq)
    sign = lambda v: np.sign(np.where(np.abs(v) < 1e-7, 0.0, v)).astype(np.int8)   # noqa: E731
    np.savez_compressed(
        a.out, axis=ax, basis=sol.basis_id.reshape(G, G), growth=sol.objective[:, 0].reshape(G, G),
        status=sol.status.reshape(G, G), sign=sign(flux).reshape(G, G, -1), ex_mets=np.array(mets),
        times=r.times, traj_glc=-traj_lb[:, :, ig], traj_ac=-traj_lb[:, :, ia],
        # concentration that gives each trajectory bound exactly (differs from the actual
        # concentration only where the depletion cap conc / (X dt) binds)
        traj_cglc=comm.km * (-traj_lb[:, :, ig]) / (comm.vmax + traj_lb[:, :, ig]),
        traj_cac=comm.km * (-traj_lb[:, :, ia]) / (comm.vmax + traj_lb[:, :, ia]),
        traj_basis=st.basis_id.reshape(T, E), traj_growth=st.objective[:, 0].reshape(T, E),
        X=r.X[:, :, 0], M_glc=r.M[:, :, comm.env_mets.index("glc__D_e")],
        M_ac=r.M[:, :, comm.env_mets.index("ac_e")], grid_seconds=grid_s, grid_lps=G * G,
        other_drift=drift.max())
    print("saved", a.out, flush=True)


if __name__ == "__main__":
    main()
