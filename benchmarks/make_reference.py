"""Reference answers for the solver comparison.

* exact FBA optimum and status per LP (manylp, certified),
* canonical exchange fluxes (pFBA-unique rule, certified unique),
* an independent cross-check of the objective on a random sample against the
  objective-row lexicographic reference (fresh HiGHS per LP, no shared code path).
"""

import os
import pickle
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from manylp import BatchLPSolver  # noqa: E402
from manylp.fba import compile_fba  # noqa: E402
from manylp.reference import reference_solve  # noqa: E402


def main(workload="results/workload.pkl", out="results/solvers/reference.npz", n_check=400):
    W = pickle.load(open(workload, "rb"))
    models = [m for _, m in W["models"]]
    snaps = W["snapshots"]
    solver = BatchLPSolver("cuda", n_workers=32)
    fba = [compile_fba(m, "fba") for m in models]
    pfu = [compile_fba(m, "pfba-unique") for m in models]
    g_f = [solver.register_group(p.lp, out_z=p.out_z(p.model.exchanges)) for p in fba]
    g_u = [solver.register_group(p.lp, out_z=p.out_z(p.model.exchanges)) for p in pfu]
    objs, oks, vexs, uniq = [], [], [], []
    for step, s, members, ex_lb in snaps:
        Lp, Up = fba[s].param_bounds(ex_lb)
        a = solver.solve_batch(g_f[s], Lp, Up)
        Lp2, Up2 = pfu[s].param_bounds(ex_lb)
        b = solver.solve_batch(g_u[s], Lp2, Up2)
        assert np.array_equal(a.status, b.status)
        # the pFBA stage-1 value must equal the plain FBA optimum
        okm = a.status == 1
        assert np.allclose(a.objective[okm, 0], b.objective[okm, 0], rtol=1e-9, atol=1e-9)
        objs.append(a.objective[:, 0])
        oks.append(okm)
        vexs.append(pfu[s].fluxes(b.z, pfu[s].model.exchanges))
        uniq.append(b.unique)
    # independent cross-check on a random sample
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(n_check):
        bi = int(rng.integers(len(snaps)))
        step, s, members, ex_lb = snaps[bi]
        j = int(rng.integers(len(members)))
        Lp, Up = fba[s].param_bounds(ex_lb[j:j + 1])
        lb, ub = fba[s].lp.full_bounds(Lp[0], Up[0])
        r = reference_solve(fba[s].lp, lb, ub)
        if r.status == 1:
            worst = max(worst, abs(r.stage_obj[0] - objs[bi][j]) / max(1.0, abs(objs[bi][j])))
        else:
            assert not oks[bi][j]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    def objarr(xs):
        a = np.empty(len(xs), dtype=object)
        for i, x in enumerate(xs):
            a[i] = x
        return a

    np.savez_compressed(out, obj=objarr(objs), ok=objarr(oks), vex=objarr(vexs), unique=objarr(uniq))
    n = sum(len(o) for o in objs)
    print(f"reference: {n} LPs, optimal {sum(o.sum() for o in oks)}, unique {sum(u.sum() for u in uniq)}; "
          f"independent cross-check on {n_check} LPs: max rel objective error {worst:.2e}")


if __name__ == "__main__":
    import sys as _s

    if len(_s.argv) > 1:
        wl = _s.argv[1]
        main(workload=wl, out=wl.replace(".pkl", "_reference.npz"))
    else:
        main()
