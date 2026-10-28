"""pyIBM: a 2D immersed boundary method in Python.

docs/user-guide.md explains how to set up and run a case, docs/numerics.md the methods.
Public entry points:

    Grid              the uniform grid (pyibm.grid)
    get_kernel        delta-function kernels by name (pyibm.kernels)
    read_input_file   the input2d-style parameter file (pyibm.inputfile)
    read_structure    a structure from its .vertex, .spring, ... files (pyibm.structures)
    run               run a case directory (pyibm.driver)
    Simulation        the time loop of one case, to step it from Python (pyibm.driver)

The command line is ``python -m pyibm run <case_dir>``.
"""

__version__ = "0.1.0.dev0"

from pyibm.grid import Grid                       # noqa: E402
from pyibm.kernels import get_kernel              # noqa: E402
from pyibm.inputfile import read_input_file       # noqa: E402
from pyibm.structures import read_structure       # noqa: E402
from pyibm.driver import run, RunResult, Simulation   # noqa: E402

__all__ = ["Grid", "get_kernel", "read_input_file", "read_structure", "run", "RunResult",
           "Simulation", "__version__"]
