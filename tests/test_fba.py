"""FBA front-end: modes, uniqueness, agreement with cobra and the reference."""

import numpy as np
import pytest

from manylp import BatchLPSolver
from manylp.backend import gpu_available
from manylp.fba import FBAModel, compile_fba
from manylp.generators import random_metabolic_network
from manylp.reference import reference_batch
from manylp.repair import OPTIMAL

DEVICES = ["cpu"] + (["cuda"] if gpu_available() else [])


def _uptake_batch(model: FBAModel, B: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lb0 = model.lb[model.exchanges]
    scale = rng.uniform(0.0, 1.0, size=(B, lb0.size))
    return lb0 * scale


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", ["fba", "pfba", "pfba-unique"])
def test_random_network_modes(device, mode):
    model = random_metabolic_network(n_met=50, n_rxn=140, n_ex=12, seed=1)
    prob = compile_fba(model, mode=mode)
    ex_lb = _uptake_batch(model, 40, seed=2)
    Lp, Up = prob.param_bounds(ex_lb)
    solver = BatchLPSolver(device=device, n_workers=4)
    g = solver.register_group(prob.lp, out_z=prob.out_z())
    sol = solver.solve_batch(g, Lp, Up)
    st, obj, X = reference_batch(prob.lp, Lp, Up)
    assert np.array_equal(sol.status, st)
    opt = st == OPTIMAL
    assert opt.sum() > 20
    np.testing.assert_allclose(sol.objective[opt], obj[opt], rtol=1e-6, atol=1e-6)
    v = prob.fluxes(sol.z)
    v_ref = X[:, prob.pos_col].copy()
    if prob.split:
        has = prob.neg_col >= 0
        v_ref[:, has] -= X[:, prob.neg_col[has]]
    # growth is always unique
    bm = int(np.argmax(model.c))
    np.testing.assert_allclose(v[opt, bm], v_ref[opt, bm], atol=1e-6)
    # mass balance holds for every certified solution
    resid = np.abs(model.S @ v[opt].T).max()
    assert resid < 1e-7
    if mode == "pfba-unique":
        assert sol.unique[opt].all()
        np.testing.assert_allclose(v[opt], v_ref[opt], atol=1e-5)
    solver.close()


def test_pfba_l1_matches_reference():
    model = random_metabolic_network(n_met=40, n_rxn=100, n_ex=10, seed=4)
    prob = compile_fba(model, mode="pfba")
    Lp, Up = prob.param_bounds(_uptake_batch(model, 20, seed=1))
    solver = BatchLPSolver(n_workers=2)
    g = solver.register_group(prob.lp, out_z=prob.out_z())
    sol = solver.solve_batch(g, Lp, Up)
    v = prob.fluxes(sol.z)
    opt = sol.status == OPTIMAL
    # stage-2 objective is -||v||_1
    np.testing.assert_allclose(-np.abs(v[opt]).sum(axis=1), sol.objective[opt, 1], rtol=1e-7, atol=1e-7)


cobra = pytest.importorskip("cobra")


@pytest.fixture(scope="module")
def ecoli():
    from cobra.io import load_model

    return load_model("textbook")


@pytest.mark.parametrize("device", DEVICES)
def test_ecoli_core_growth_matches_cobra(ecoli, device):
    model = FBAModel.from_cobra(ecoli)
    prob = compile_fba(model, mode="pfba-unique")
    ex = model.exchanges
    rng = np.random.default_rng(0)
    B = 24
    ex_lb = np.repeat(model.lb[ex][None, :], B, axis=0)
    glc = model.rxn_ids.index("EX_glc__D_e")
    o2 = model.rxn_ids.index("EX_o2_e")
    kg = int(np.nonzero(ex == glc)[0][0])
    ko = int(np.nonzero(ex == o2)[0][0])
    ex_lb[:, kg] = -rng.uniform(0, 12, B)
    ex_lb[:, ko] = -rng.uniform(0, 25, B)
    Lp, Up = prob.param_bounds(ex_lb)
    solver = BatchLPSolver(device=device, n_workers=4)
    g = solver.register_group(prob.lp, out_z=prob.out_z())
    sol = solver.solve_batch(g, Lp, Up)
    growth_cobra = []
    for b in range(B):
        with ecoli as mdl:
            for i, j in enumerate(ex):
                mdl.reactions[j].lower_bound = ex_lb[b, i]
            growth_cobra.append(mdl.slim_optimize(error_value=0.0))
    growth = np.where(sol.status == OPTIMAL, sol.objective[:, 0], 0.0)
    np.testing.assert_allclose(growth, growth_cobra, atol=1e-6)
    assert sol.unique[sol.status == OPTIMAL].all()
    solver.close()
