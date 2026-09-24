"""Alternative-optima policies and regime-aware adaptive integration."""

import numpy as np
import pytest

from manylp.adaptive import run_adaptive
from manylp.dfba import Community, ManyLPAdapter, exchange_metabolites, run_dfba
from manylp.envelope import PolicyAdapter, standard_policies
from manylp.generators import random_metabolic_network


def _community():
    models = [random_metabolic_network(n_met=40, n_rxn=110, n_ex=10, seed=s) for s in (1, 2)]
    for i, m in enumerate(models):
        m.name = f"sp{i}"
    env = sorted({x for m in models for x in exchange_metabolites(m)})
    M0 = np.full(len(env), 5.0)
    return Community(models=models, env_mets=env, M0=M0, influx=np.zeros(len(env)),
                     max_uptake=np.full(len(env), np.inf), X0=np.array([0.01, 0.01]), vmax=5.0, km=0.5)


def test_policies_keep_growth_and_certify_uniqueness():
    comm = _community()
    pols = standard_policies(n_random=6, metabolites=comm.env_mets[:2])
    E = len(pols)
    ad = PolicyAdapter(pols, np.arange(E), device="cpu", n_workers=4)
    ad.setup(comm.models, "pfba-unique")
    ex_lb = np.tile(-np.full(comm.models[0].exchanges.size, 3.0), (E, 1))
    g, v, ok = ad.solve(0, ex_lb, np.arange(E))
    assert ok.all()
    # growth is unique: every policy returns the same optimum
    np.testing.assert_allclose(g, g[0], rtol=1e-9, atol=1e-12)
    st = ad.stats()
    assert st["unique"] == st["optimal"] == E
    ad.close()


def test_envelope_members_differ_but_canonical_matches_plain_run():
    comm = _community()
    pols = standard_policies(n_random=8)
    E = len(pols)
    ad = PolicyAdapter(pols, np.arange(E), device="cpu", n_workers=4)
    r = run_dfba(comm, ad, E=E, t_end=3.0, dt=0.1, mode="pfba-unique")
    plain = run_dfba(comm, ManyLPAdapter("cpu", n_workers=4), E=1, t_end=3.0, dt=0.1, mode="pfba-unique")
    np.testing.assert_allclose(r.X[:, 0], plain.X[:, 0], rtol=1e-10, atol=1e-14)
    ad.close()


def _ecoli_community():
    cobra = pytest.importorskip("cobra")
    from cobra.io import load_model

    from manylp.fba import FBAModel

    base = FBAModel.from_cobra(load_model("textbook"))
    models = []
    for i in range(2):
        m = FBAModel(S=base.S, lb=base.lb, ub=base.ub, c=base.c, exchanges=base.exchanges,
                     rxn_ids=base.rxn_ids, met_ids=base.met_ids, name=f"ecoli{i}")
        models.append(m)
    env = sorted(set(exchange_metabolites(base)))
    conc = {"glc__D_e": 10.0, "o2_e": 2.0, "nh4_e": 10.0, "pi_e": 10.0, "h2o_e": 1e3, "h_e": 1e3,
            "co2_e": 0.0}
    M0 = np.array([conc.get(m, 0.0) for m in env])
    return Community(models=models, env_mets=env, M0=M0, influx=np.zeros(len(env)),
                     max_uptake=np.full(len(env), np.inf), X0=np.array([0.02, 0.01]), vmax=10.0, km=0.5)


@pytest.mark.parametrize("split", [False, True])
def test_adaptive_integration_converges_to_fine_euler(split):
    comm = _ecoli_community()
    ref = run_dfba(comm, ManyLPAdapter("cpu", n_workers=4), E=1, t_end=6.0, dt=0.0005, record_every=2000)
    r = run_adaptive(comm, ManyLPAdapter("cpu", n_workers=4), E=1, t_end=6.0, rtol=1e-7, atol=1e-10,
                     t_eval=ref.times, split_regimes=split)
    assert np.all(np.isfinite(r.X))
    # Euler's O(dt) error at dt = 5e-4 bounds the achievable agreement
    np.testing.assert_allclose(r.X[:, 0], ref.X[:, 0], rtol=2e-3, atol=1e-8)
    assert r.rhs_evals < 6.0 / 0.0005 / 10
