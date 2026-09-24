"""manylp: exact batched solving of many small, structurally-shared LPs.

Members of a group share ``A`` and an (ordered, lexicographic) objective list and
differ only in their bounds.  Solutions are certified in batch against cached
optimal bases -- one GEMM per basis on a GPU or CPU -- and only members outside
every cached critical region are re-solved by a warm-started dual simplex.
"""

from manylp.basis import Tolerances
from manylp.lp import LexLP
from manylp.solver import SOURCE_NAMES, BatchLPSolver, BatchSolution, GroupHandle

__all__ = ["LexLP", "BatchLPSolver", "BatchSolution", "GroupHandle", "Tolerances", "SOURCE_NAMES"]
__version__ = "0.1.0"
