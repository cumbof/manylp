"""Correctness of the certify-and-repair solver against the independent reference."""

import numpy as np
import pytest
import scipy.sparse as sp

from manylp import BatchLPSolver, LexLP
from manylp.backend import gpu_available
from manylp.generators import perturb_param_bounds, random_feasible_lp
from manylp.reference import reference_batch
from manylp.repair import INFEASIBLE, OPTIMAL

DEVICES = ["cpu"] + (["cuda"] if gpu_available() else [])


def _lex_lp(seed: int, K: int, m: int = 25, n: int = 60) -> LexLP:
    """Random LP whose early objectives are sparse, so optimal faces are large."""
    base, _ = random_feasible_lp(m, n, density=0.15, n_param=14, seed=seed)
    rng = np.random.default_rng(seed + 100)
    objs = []
    for k in range(K):
        c = np.zeros(n)
        nnz = [2, 8, n][min(k, 2)] if K > 1 else n
        idx = rng.choice(n, size=nnz, replace=False)
        c[idx] = rng.normal(size=nnz)
        objs.append(c)
    return LexLP(A=base.A, objectives=np.vstack(objs), col_lb=base.col_lb, col_ub=base.col_ub,
                 row_lb=base.row_lb, row_ub=base.row_ub, param_cols=base.param_cols)


def _check_against_reference(sol, lp, L, U, atol=1e-6):
    st, obj, X = reference_batch(lp, L, U)
    assert np.array_equal(sol.status, st), (sol.status, st)
    opt = st == OPTIMAL
    np.testing.assert_allclose(sol.objective[opt], obj[opt], rtol=1e-6, atol=atol)
    u = opt & sol.unique
    if u.any():
        np.testing.assert_allclose(sol.z[u], X[u], atol=1e-5)
    return st


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("K", [1, 2, 3])
def test_matches_reference(device, K):
    lp = _lex_lp(seed=K, K=K)
    L, U = perturb_param_bounds(lp, 60, scale=0.3, seed=1)
    solver = BatchLPSolver(device=device, n_workers=4)
    g = solver.register_group(lp)
    sol = solver.solve_batch(g, L, U)
    st = _check_against_reference(sol, lp, L, U)
    assert (st == OPTIMAL).sum() > 30
    assert sol.certified[st == OPTIMAL].mean() > 0.95
    if K == 3:
        # a dense generic last stage makes the lexicographic optimum unique
        assert sol.unique[st == OPTIMAL].mean() > 0.95
    solver.close()


@pytest.mark.parametrize("device", DEVICES)
def test_second_call_is_all_cache_hits(device):
    lp = _lex_lp(seed=7, K=2)
    L, U = perturb_param_bounds(lp, 50, scale=0.2, seed=3)
    solver = BatchLPSolver(device=device, n_workers=4)
    g = solver.register_group(lp)
    s1 = solver.solve_batch(g, L, U)
    s2 = solver.solve_batch(g, L, U, warm_start=s1)
    assert s2.stats["highs_solves"] == 0
    assert s2.stats["warm"] == (s1.status == OPTIMAL).sum() + (s1.basis_id[s1.status == INFEASIBLE] >= 0).sum()
    np.testing.assert_array_equal(s1.status, s2.status)
    np.testing.assert_allclose(s1.objective, s2.objective, atol=1e-9)
    solver.close()


@pytest.mark.parametrize("device", DEVICES)
def test_temporal_drift_reuses_bases(device):
    lp = _lex_lp(seed=11, K=2)
    L0, U0 = perturb_param_bounds(lp, 20, scale=0.2, seed=5)
    solver = BatchLPSolver(device=device, n_workers=4)
    g = solver.register_group(lp)
    prev = None
    rng = np.random.default_rng(0)
    L, U = L0.copy(), U0.copy()
    total_solves = 0
    for t in range(15):
        sol = solver.solve_batch(g, L, U, warm_start=prev)
        if t in (0, 7, 14):
            _check_against_reference(sol, lp, L, U)
        total_solves += sol.stats["highs_solves"]
        prev = sol
        step = rng.normal(scale=0.01, size=L.shape)
        L = np.minimum(L + step, U)
    # far fewer simplex solves than LPs
    assert total_solves < 0.3 * 20 * 15
    solver.close()


@pytest.mark.parametrize("device", DEVICES)
def test_infeasible_members_are_certified_by_farkas(device):
    lp = _lex_lp(seed=3, K=1)
    L, U = perturb_param_bounds(lp, 30, scale=0.1, seed=2)
    # force infeasibility on a third of the members: pin params at an extreme corner
    L[::3] = 9.9
    U[::3] = 10.0
    solver = BatchLPSolver(device=device, n_workers=4)
    g = solver.register_group(lp)
    s1 = solver.solve_batch(g, L, U)
    st = _check_against_reference(s1, lp, L, U)
    n_inf = int((st == INFEASIBLE).sum())
    assert n_inf >= 5
    # infeasible members report zeros (dFBA "no growth" policy)
    assert np.all(s1.objective[st == INFEASIBLE] == 0)
    s2 = solver.solve_batch(g, L, U, warm_start=s1)
    assert s2.stats["highs_solves"] == 0
    assert np.array_equal(s2.status, st)
    assert s2.certified[st == INFEASIBLE].all()
    solver.close()


def test_crossing_bounds_are_trivially_infeasible():
    lp = _lex_lp(seed=5, K=1)
    L, U = perturb_param_bounds(lp, 4, scale=0.1, seed=2)
    L[1, 0] = U[1, 0] + 1.0
    solver = BatchLPSolver(device="cpu", n_workers=1)
    g = solver.register_group(lp)
    sol = solver.solve_batch(g, L, U)
    assert sol.status[1] == INFEASIBLE and sol.source[1] == 5


def test_degenerate_equal_bounds_and_fixed_template():
    # template fixes some columns and some members fix parametric columns
    lp = _lex_lp(seed=9, K=2)
    lb = lp.col_lb.copy(); ub = lp.col_ub.copy()
    lb[20:23] = ub[20:23] = 0.5
    lp2 = LexLP(A=lp.A, objectives=lp.objectives, col_lb=lb, col_ub=ub, row_lb=lp.row_lb,
                row_ub=lp.row_ub, param_cols=lp.param_cols)
    L, U = perturb_param_bounds(lp2, 40, scale=0.2, seed=4)
    L[::2, :3] = U[::2, :3] = 0.0
    solver = BatchLPSolver(device="cpu", n_workers=4)
    g = solver.register_group(lp2)
    sol = solver.solve_batch(g, L, U)
    _check_against_reference(sol, lp2, L, U)


def test_equality_constrained_fba_like():
    """S x = 0 with a bounded 'biomass' objective -- the FBA shape."""
    rng = np.random.default_rng(0)
    m, n = 20, 45
    S = sp.random(m, n, density=0.12, random_state=rng, format="csc",
                  data_rvs=lambda k: rng.choice([-2.0, -1.0, 1.0, 2.0], size=k))
    # exchanges: one per metabolite
    S = sp.hstack([S, -sp.identity(m)], format="csc")
    N = S.shape[1]
    lb = np.where(rng.random(N) < 0.4, -100.0, 0.0)
    ub = np.full(N, 100.0)
    ex = np.arange(n, N)
    lb[ex] = -5.0
    c = np.zeros(N); c[0] = 1.0
    lp = LexLP(A=S, objectives=np.vstack([c, rng.uniform(-1, 1, N)]), col_lb=lb, col_ub=ub,
               param_cols=ex)
    L, U = perturb_param_bounds(lp, 50, scale=0.3, seed=1)
    solver = BatchLPSolver(device="cpu", n_workers=4)
    g = solver.register_group(lp)
    sol = solver.solve_batch(g, L, U)
    _check_against_reference(sol, lp, L, U)


def test_fused_host_kernels_match_numpy_path():
    from manylp import cpu_kernels
    from manylp.basis import _certify_host_fused, _certify_host_numpy

    if not cpu_kernels.available():
        pytest.skip("numba not installed")
    lp = _lex_lp(seed=21, K=3)
    L, U = perturb_param_bounds(lp, 200, scale=0.3, seed=9)
    solver = BatchLPSolver(device="cpu", n_workers=4, min_yield=0.0)   # keep propagation on
    g = solver.register_group(lp)
    sol = solver.solve_batch(g, L, U)
    be = solver.backend
    for e in [x for x in g.pool.bases.values() if not x.host.get("lazy")][:5]:
        a = _certify_host_numpy(e, L, U, solver.tol, g.n_out, True, lp, be)
        b = _certify_host_fused(e, L, U, solver.tol, g.n_out, True, lp, be)
        np.testing.assert_array_equal(a.ok, b.ok)
        np.testing.assert_array_equal(a.unique, b.unique)
        np.testing.assert_allclose(a.obj[a.ok], b.obj[b.ok], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(a.z_out[a.ok], b.z_out[b.ok], rtol=1e-12, atol=1e-12)
    assert sol.status.size == 200


def test_atlas_roundtrip_skips_all_simplex_work(tmp_path):
    lp = _lex_lp(seed=31, K=2)
    L, U = perturb_param_bounds(lp, 40, scale=0.2, seed=6)
    s1 = BatchLPSolver(device="cpu", n_workers=4)
    g1 = s1.register_group(lp)
    a = s1.solve_batch(g1, L, U)
    assert a.stats["highs_solves"] > 0
    path = str(tmp_path / "atlas.npz")
    n = s1.save_atlas(g1, path)
    assert n == len(g1.pool)
    s2 = BatchLPSolver(device="cpu", n_workers=4)
    g2 = s2.register_group(lp)
    assert s2.load_atlas(g2, path) == n
    b = s2.solve_batch(g2, L, U)
    assert b.stats["highs_solves"] == 0
    np.testing.assert_array_equal(a.status, b.status)
    np.testing.assert_allclose(a.objective, b.objective, atol=1e-9)
    # an atlas for another LP is refused
    other = _lex_lp(seed=32, K=2)
    g3 = s2.register_group(other)
    with pytest.raises(ValueError):
        s2.load_atlas(g3, path)


@pytest.mark.skipif(not gpu_available(), reason="needs a GPU")
def test_device_resident_inputs_and_outputs_match_host_path():
    import cupy as cp

    lp = _lex_lp(seed=41, K=3)
    L, U = perturb_param_bounds(lp, 300, scale=0.3, seed=12)
    L[::7] = 9.9            # some infeasible members too
    U[::7] = 10.0
    s_h = BatchLPSolver(device="cuda", n_workers=4)
    g_h = s_h.register_group(lp)
    a = s_h.solve_batch(g_h, L, U)
    s_d = BatchLPSolver(device="cuda", n_workers=4)
    g_d = s_d.register_group(lp)
    b = s_d.solve_batch(g_d, cp.asarray(L), cp.asarray(U), device_out=True)
    assert isinstance(b.objective, cp.ndarray)
    np.testing.assert_array_equal(a.status, b.status)
    np.testing.assert_allclose(a.objective, b.objective.get(), atol=1e-10)
    np.testing.assert_allclose(a.z, b.z.get(), atol=1e-9)
    np.testing.assert_array_equal(a.unique, b.unique)
    # warm call on device input: all cache hits
    c = s_d.solve_batch(g_d, cp.asarray(L), cp.asarray(U), warm_start=b, device_out=True)
    assert c.stats["highs_solves"] == 0


def test_direct_mode_answers_are_certified_and_match_reference():
    lp = _lex_lp(seed=51, K=2)
    L, U = perturb_param_bounds(lp, 200, scale=0.45, seed=21)
    L[::9] = 9.9                    # some infeasible members as well
    U[::9] = 10.0
    solver = BatchLPSolver(device="cpu", n_workers=4, min_yield=1e9)   # force direct mode
    g = solver.register_group(lp)
    sol = solver.solve_batch(g, L, U)
    assert sol.stats["direct_mode"] == 1
    _check_against_reference(sol, lp, L, U)
    assert sol.certified.mean() > 0.95
    solver.close()
