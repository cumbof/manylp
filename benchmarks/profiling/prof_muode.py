import cProfile, pstats, sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "benchmarks"); sys.path.insert(0, "../muODE")
from bench_muode import load_community, perturbed_diets, G
from muode import DynamicFBA
from muode.backends import make_backend
from muode.diet import load_diet
from muode.kinetics import KineticParameters
comm = load_community(12)
diets = perturbed_diets(load_diet(f"{G}/gems/western_gut_modelseed.csv"), 64, 0.3)
eng = DynamicFBA(t_end=2.0, dt=0.1, record_fluxes=False)
be = make_backend("manylp-gpu", n_workers=32)
pr = cProfile.Profile(); pr.enable()
eng.run_ensemble(comm, diets, KineticParameters(default_vmax=10.0, default_km=0.01), backend=be)
pr.disable()
pstats.Stats(pr).sort_stats("tottime").print_stats(14)
