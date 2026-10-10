# Activate the baseline-solver environment (Gurobi, Xpress, cuOpt, MPAX, OR-Tools, OSQP).
#
#   conda env create -f benchmarks/env/environment.yml && conda activate manylp
#   python -m venv --system-site-packages baselines
#   baselines/bin/pip install -r benchmarks/env/requirements-baselines.txt
#   source benchmarks/env/activate_baselines.sh baselines
#
# cuOpt and the NVIDIA wheels ship their shared libraries inside their package directories, so
# every site-packages directory that holds a shared library goes on LD_LIBRARY_PATH.
VENV=${1:-baselines}
export PATH="$VENV/bin:$PATH"
_sp=$("$VENV/bin/python" -c "import sys; print('\n'.join(p for p in sys.path if p.endswith('site-packages')))")
_libs=$(for d in $_sp; do find "$d" -name '*.so*' -printf '%h\n' 2>/dev/null; done | sort -u | paste -sd:)
export LD_LIBRARY_PATH="$_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset _sp _libs
