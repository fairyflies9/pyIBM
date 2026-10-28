# Tutorials

Three notebooks on Peskin's immersed boundary formulation, regularized delta functions and the
discretisation. Each one takes the equations of a source paper in order, restates them with
their equation numbers, names the pyIBM function that implements each one, and runs a short
computation that shows it working, and each ends with the full references of the papers it cites.
They are written for people who know fluid mechanics and numerical methods.

| notebook | what it covers |
|---|---|
| `01_peskin_mcqueen_1996_immersed_boundary.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fairyflies9/pyIBM/blob/main/tutorials/01_peskin_mcqueen_1996_immersed_boundary.ipynb) | Peskin & McQueen 1996 section 14 (the first-order periodic scheme my 2008 MATLAB code used: fiber force, spreading, upwind advection, the Fourier solution and its four degenerate modes, interpolation, the 4-point kernel), Peskin 2002 section 7 (the midpoint scheme pyIBM runs by default), target points as tethers, the 2008 heaving plate reproduced to round-off against the MATLAB reference data and then run with a stiffer tether, the midpoint scheme and the `peskin4` kernel, and what pyIBM changes from the 2008 code. |
| `02_projection_method_mac_dirichlet.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fairyflies9/pyIBM/blob/main/tutorials/02_projection_method_mac_dirichlet.ipynb) | The MAC-grid projection back end (`pyibm.fluid.mac`): Chorin's projection, the staggered grid, Dirichlet, periodic and outflow boundaries that vary in time, the Poisson solvers, the immersed boundary on a grid with walls, and the wall slip of the non-incremental projection next to the incremental one. |
| `03_blob_projection_method.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fairyflies9/pyIBM/blob/main/tutorials/03_blob_projection_method.ipynb) | The blob projection method of Cortez & Minion 2000 (`pyibm.coupling.blob`): regularised blob forces, the analytic divergence-free projection, the bicubic stencil and the divergence-free stream-function interpolation pyIBM uses by default, and how the default compares with the delta-function coupling on one elastic ring, with a control run that separates the effect of the interpolation from that of the blob. |

## How to run them

On Google Colab, click a badge above. The first code cell then clones this repository into
the Colab session and moves into `tutorials/`; Colab already has `numpy`, `scipy` and
`matplotlib`, so nothing else needs installing. On your own machine, from the repository root:

```
jupyter lab
```

and open the notebook under `tutorials/`. The first cell of every notebook walks up from its
working directory to the repository root and puts it on `sys.path`, so nothing has to be
installed beyond `numpy`, `scipy`, `matplotlib` and Jupyter. Every notebook uses coarse grids and
short runs and executes in well under three minutes on a laptop; the text says what a converged
run would use. Output files (histories, VTK) go to a temporary directory that the notebook
removes again, never into `examples/`.

To execute them headless, as the tests do:

```
python -m pytest -q -m slow tests/notebooks
```

`tests/notebooks/test_<name>.py` runs each notebook with `nbclient` and fails on any error output
or on a printed local path; the always-on part of those tests only checks that the file loads.
They need `nbformat` and `nbclient` (both come with Jupyter) and are skipped without them.
`tests/notebooks/test_notebook_hygiene.py` needs neither: it checks that the saved notebooks carry
their outputs, no execution timestamps and no local paths. Nor does
`tests/notebooks/test_tutorial_text.py`, which checks that the numbers and claims the text makes
still match what the saved outputs print, so a re-run that changes a printed number fails there
until the text is brought in line.

The saved notebooks are executed with `nbclient` and without execution timestamps, so a re-run
changes only what the cells print (timings, mostly). If you re-run one in Jupyter before
committing it, turn off the recording of cell timings or remove `metadata.execution` from the
cells.
