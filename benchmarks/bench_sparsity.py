"""Limiting-nutrient sparsity: how many parametric columns does each certified
basis actually need, and what does the lazy (sparse-in-parameters) affine law buy
over a dense one?  12-species community, 48 h, E = 256, GPU."""
import json, os, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
import manylp.basis as mb
from manylp.dfba import ManyLPAdapter, load_gut_community, run_dfba
G = os.environ.get("GUT_DIR", "benchmarks/inputs/gut_western")


def main():
    comm = load_gut_community(f"{G}/gems", f"{G}/gems/western_gut_modelseed.csv", abundance_tsv=f"{G}/abundance.tsv")
    E = 256
    pert = np.random.default_rng(0).lognormal(0, 0.3, size=(E, len(comm.env_mets)))
    out = {}
    orig_prepare = mb.prepare_basis_entry
    for label in ("lazy", "dense"):
        if label == "dense":
            # force every nonbasic parametric column into the law up front
            def dense_prepare(lp, basic, status, tol, out_z, source_bounds=None):
                e = orig_prepare(lp, basic, status, tol, out_z, source_bounds)
                mb.ensure_active_host = None
                need = np.ones(e.n_np, dtype=bool)
                new = np.nonzero(need & ~e.host["active"])[0]
                if new.size:
                    Tn = mb._t_columns(lp, e.host["lu"], e.host["np_idx"][new])
                    e.host["T_act"] = np.hstack([e.host["T_act"], Tn])
                    e.host["act_pos"] = np.concatenate([e.host["act_pos"], new])
                    e.host["active"][new] = True
                return e
            import manylp.solver as ms
            ms.prepare_basis_entry = dense_prepare
        ad = ManyLPAdapter("cuda", n_workers=32)
        r = run_dfba(comm, ad, E=E, t_end=48.0, dt=0.1, perturb=pert)
        ents = [e for g in ad.groups for e in g.pool.bases.values()]
        out[label] = {"solve_seconds": r.solve_seconds, "lps_per_second": r.n_lps / r.solve_seconds,
                      "bases": len(ents), "n_np_mean": float(np.mean([e.n_np for e in ents])),
                      "n_active_mean": float(np.mean([e.n_active for e in ents])),
                      "n_active_max": int(max(e.n_active for e in ents)),
                      "device_MiB": float(sum(e.nbytes for e in ents) / 2**20)}
        ad.close()
        print(label, json.dumps(out[label]), flush=True)
    os.makedirs("results/sparsity", exist_ok=True)
    json.dump(out, open("results/sparsity/sparsity.json", "w"), indent=1)


if __name__ == "__main__":
    # manylp's direct mode spawns worker processes, which re-import this module
    main()
