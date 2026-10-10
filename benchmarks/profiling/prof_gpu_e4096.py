import os, numpy as np, warnings, cProfile, pstats, time
warnings.filterwarnings("ignore")
from manylp.dfba import load_gut_community, run_dfba, ManyLPAdapter
G = os.environ.get("GUT_DIR", "benchmarks/inputs/gut_western")
comm = load_gut_community(f"{G}/gems", f"{G}/gems/western_gut_modelseed.csv", names=["B_theta","F_praus","R_bromii"], abundance_tsv=f"{G}/abundance.tsv")
E = 4096
pert = np.random.default_rng(0).lognormal(0, 0.3, size=(E, len(comm.env_mets)))
ad = ManyLPAdapter("cuda", n_workers=32)
run_dfba(comm, ad, E=E, t_end=0.5, dt=0.1, perturb=pert)
ad.setup = lambda *a, **k: None
pr = cProfile.Profile(); pr.enable()
r = run_dfba(comm, ad, E=E, t_end=2.0, dt=0.1, perturb=pert, setup=False)
pr.disable()
print(f"solve {r.solve_seconds:.2f}s wall {r.wall_seconds:.2f}s LPs {r.n_lps} -> {r.n_lps/r.solve_seconds:.0f} LP/s")
pstats.Stats(pr).sort_stats("tottime").print_stats(18)
