# pyIBM user guide

How to install pyIBM, run a case, write the input file, the structure files and the hooks,
and read the output. `docs/numerics.md` explains the methods behind each option and
`docs/examples.md` lists the examples.

Contents:

1. [Install](#1-install)
2. [Running a case](#2-running-a-case)
3. [The case folder](#3-the-case-folder)
4. [The input file](#4-the-input-file)
5. [Structure files](#5-structure-files)
6. [Hooks](#6-hooks)
7. [Output](#7-output)
8. [Stepping a case from Python](#8-stepping-a-case-from-python)
9. [Errors and warnings](#9-errors-and-warnings)
10. [Tests and tutorials](#10-tests-and-tutorials)

## 1. Install

pyIBM needs Python 3.11 or newer and only `numpy`, `scipy` and `matplotlib`. From the
repository root:

```
pip install -e .                  # the package
pip install -e ".[test]"          # plus pytest, for the tests
pip install -e ".[notebooks]"     # plus Jupyter, for the tutorials and their tests
```

Without installing, `python -m pyibm ...` still works from the repository root, and every
example's `generate.py` and `analyze.py` puts the repository root on `sys.path` itself, so
they run from a plain clone.

## 2. Running a case

From the command line:

```
python -m pyibm run examples/heaving_plate_2008
python -m pyibm run examples/heaving_plate_2008 --input input2d_midpoint
python -m pyibm run examples/heaving_plate_2008 --steps 10 --quiet
python -m pyibm run examples/heaving_plate_2008 --output-dir viz/trial
python -m pyibm version
```

| option | meaning |
|---|---|
| `case_dir` | the folder holding the input file, the structure files and the hooks file |
| `--input NAME` | the input file inside `case_dir` (default `input2d`) |
| `--steps N` | the number of steps, overriding `Tfinal/dt` |
| `--quiet` | print nothing |
| `--output-dir DIR` | the output folder, relative to `case_dir` unless absolute (default: `Output_Info { output_dir }` of the input file) |

Without `--quiet` a run prints a header line (case, structure, back end, scheme, coupling,
grid, `dt`, steps, output folder), then one line per output step with the step, time, force
on the body, target error in cells, CFL number, kinetic energy and largest speed, and at the
end the wall time per step. The exit code is 0 when the run finishes, 1 when it blows up
(a non-finite velocity or position, reported on stderr), and 2 for a command-line error.
Any other error in the case, such as a bad input file or a hook that raises, ends with
Python's own traceback.

From Python:

```python
import pyibm

result = pyibm.run("examples/heaving_plate_2008", input_file="input2d_midpoint", quiet=True)
h = result.history                      # numpy record array, one row per output
h.t, h.fx_body, h.max_target_error_over_dx
result.solver.u, result.solver.v, result.solver.p    # final fields
result.structure.X                      # final positions
result.wall_time, result.steps, result.t, result.output_dir
```

`pyibm.run(case_dir, input_file="input2d", steps=None, quiet=False, output_dir=None)`
returns a `RunResult` with the fields `history`, `solver`, `structure` (None for a
fluid-only case), `grid`, `parameters`, `wall_time` (seconds), `steps` (taken in this run),
`t` (final time), `output_dir` (the folder actually used) and `simulation` (the
`Simulation`, section 8). An absolute `input_file` is used as it is.

## 3. The case folder

A case is a folder that holds:

- an input file, `input2d` by default; variants are usually named `input2d_<variant>` and
  chosen with `--input`;
- the structure files named by `Lag_Structure_Info { name }`: `<name>.vertex` and whichever
  of `<name>.spring`, `.d_spring`, `.beam`, `.nonInv_beam`, `.target`, `.mass` exist
  (section 5);
- optionally a hooks file, `hooks.py` by default, with prescribed motion, boundary data,
  body forces, an initial velocity and per-output diagnostics (section 6).

Output goes to `<case_dir>/<output_dir>`. The examples keep each input's output under
`viz/` (`viz/<variant>`), which `.gitignore` covers. Every example also has a `generate.py`
that writes its structure and input files, so the parameters live in one place, and an
`analyze.py` that compares the run with its reference.

## 4. The input file

The input file looks like IB2d's `input2d`, with a grammar of our own:

```
% a comment (# works too)
Fluid_Parameters { rho = 1.0  mu = 0.01 }
Temporal_Information {
    Tfinal = 2.0
    dt = 0.002
}
Grid_Parameters { Nx = 64  Ny = 64  Lx = 1.0  Ly = 1.0 }
Solver {
    backend = "periodic_fft"
    time_scheme = "midpoint"
    coupling = "delta"
    kernel = "peskin4"
}
Lag_Structure_Info { name = "band"  ds = 0.0075678631986736845 }
Output_Info { print_dump = 20  output_dir = "viz/delta" }
```

A section is its name followed by `{`, then `key = value` pairs in any layout (several per
line or one per line), then `}`. Values are numbers, quoted strings, `true`/`false` (or
`0`/`1` for the switches), or bare words, which are taken as strings, so `backend = mac`
works as well as `backend = "mac"`. Quoted strings are taken literally (a backslash is an
ordinary character). Unknown sections or keys raise `ValueError` naming them. Keys left out
take the defaults below; the first three sections have no defaults and are required. Files
are read as UTF-8, with or without a byte-order mark.

| section | key | default | meaning |
|---|---|---|---|
| `Fluid_Parameters` | `rho` | required | density [kg/m^2 in 2D], > 0 |
| | `mu` | required | dynamic viscosity [N s/m^2], >= 0 |
| `Temporal_Information` | `Tfinal` | required | final time [s]; the run takes `round(Tfinal/dt)` steps |
| | `dt` | required | time step [s] |
| `Grid_Parameters` | `Nx`, `Ny` | required | cells: an even integer >= 8 in a periodic direction, an integer >= 4 in a direction with walls or an outflow |
| | `Lx`, `Ly` | required | box size [m]; the domain is `[0, Lx) x [0, Ly)` |
| `Solver` | `backend` | `"periodic_fft"` | `"periodic_fft"` (doubly periodic, FFT) or `"mac"` (staggered grid with walls, inflow, outflow or periodic sides) |
| | `time_scheme` | `"midpoint"` | `"midpoint"` (Peskin 2002, second order) or `"pm1996"` (Peskin & McQueen 1996, first order) |
| | `coupling` | `"delta"` | `"delta"` (discrete delta function) or `"blob"` (blob projection, `periodic_fft` only) |
| | `kernel` | `"peskin4"` | delta kernel: `"peskin4"`, `"cosine4"` or `"roma3"` (`roma3` is meant for `mac`) |
| | `remove_mean_force` | `false` | `periodic_fft`: hold the box-mean velocity fixed (needed when a tethered body pushes on a periodic box) |
| | `blob` | `"gaussian4"` | blob: `"gaussian4"` or `"compact"` |
| | `blob_delta` | `3.0` | blob scale in units of `dx` |
| | `patch` | `0` | blob patch half-width in cells; 0 = automatic |
| | `blob_interpolation` | `"streamfunction"` | blob marker velocity: `"streamfunction"` (divergence-free), `"bicubic"` (interpolates `u` and `v` directly) or `"adjoint"` (divergence-free and energy-conserving, slower; see docs/numerics.md) |
| | `advection` | `"blend"` | `mac`: `"blend"` (Seibold's central/upwind blend), `"central"` or `"upwind"` |
| | `viscous` | `"implicit"` | `mac`: `"implicit"` or `"explicit"` |
| | `mac_projection` | `"nonincremental"` | `mac`: `"nonincremental"` (Chorin) or `"incremental"` (removes the first-order wall slip) |
| `Boundary_Conditions` | `left`, `right`, `bottom`, `top` | `"periodic"` | `mac`: `"periodic"`, `"dirichlet"` or `"outflow"` for each side; the data of a Dirichlet side come from the `bc_<side>` hook, a wall at rest without one |
| `Lag_Structure_Info` | `name` | none | the structure file name; leave it out for a fluid-only case |
| | `ds` | median spacing | quadrature weight of every point [m] (section 5) |
| | `gravity_x`, `gravity_y` | `0.0` | gravity on the mass points [m/s^2] |
| `Output_Info` | `print_dump` | `100` | output every this many steps (also at step 0 and the last step) |
| | `save_vtk` | `true` | write the VTK files |
| | `save_history` | `true` | write `history.csv` |
| | `plot` | `false` | write a vorticity PNG at every output |
| | `output_dir` | `"viz"` | output folder, relative to the case folder unless absolute; it may not be the case folder itself |
| `Hooks` | `file` | `"hooks.py"` | the hooks file, relative to the case folder (section 6) |

Checks made when the file is read (and again when a `Simulation` is built from parameters
made in code): `dt`, `Tfinal`, `rho`, `Lx`, `Ly` finite and positive, `mu` finite and not
negative; the cell-count rule above; every option one of its choices; `periodic_fft` needs
all four sides periodic and `dx == dy`; `blob` needs `periodic_fft`; on `mac`, periodic
sides come in opposite pairs and at most one side is an outflow; `ds > 0` when given,
`print_dump >= 1`, `blob_delta > 0`, `patch >= 0`; `Hooks.file` names a `.py` file. A
`ValueError` names the offending key. `roma3` on `periodic_fft` only warns.

To write input files from Python, build `pyibm.inputfile.Parameters` (with `FluidParams`,
`TimeParams`, `GridParams`, `SolverParams`, `BCParams`, `StructureParams`, `OutputParams`,
`HooksParams`) and call `pyibm.inputfile.write_input_file(path, params, header=...)`. It
writes every key, so the file is explicit; `read_input_file(path)` reads it back.

## 5. Structure files

All structure files are whitespace-separated ASCII. The first line is the number of rows,
and it must equal the number of data rows, or we raise `ValueError` naming the file (IB2d
does not use this line, so check it when you bring files over from IB2d). Everything
after `%` or `#` on a line is a comment.

**Indices are 0-based**, as in the Python version of IB2d. Files from the MATLAB version of
IB2d are 1-based: subtract 1 from every index column before using them. A 1-based file is
caught only when it names the point `N`; a partial `.target` or `.mass` file that is still
1-based passes every range check and silently attaches each element to the next point.

| file | columns | meaning |
|---|---|---|
| `.vertex` | `x y` | the positions [m]; required |
| `.spring` | `i j k rest [alpha]` | spring between points `i` and `j`, stiffness `k` per unit length, rest length [m], optional exponent `alpha` (1, Hookean, when missing) |
| `.d_spring` | `i j k rest b` | Hookean spring with a dashpot of coefficient `b` on the relative velocity |
| `.beam` | `p q r k curv` | torsional beam about `q`: energy `k/2 (c - curv)^2` with `c = (X_r - X_q) x (X_q - X_p)`, the preferred cross product `curv` [m^2] (0 for straight) |
| `.nonInv_beam` | `p q r k cx cy` | non-invariant beam: energy `k/2 \|X_r - 2 X_q + X_p - (cx, cy)\|^2` |
| `.target` | `i k [c]` | point `i` tethered to a target with stiffness `k` per unit length and optional damping `c` (0 when missing); the target starts at the vertex position |
| `.mass` | `i k m` | point `i` tethered with stiffness `k` to a ghost of mass `m` per unit length, which starts at the vertex position at rest and feels gravity |

All coefficients are per unit length, because the forces are densities per unit length and
each point carries the quadrature weight `ds` (`docs/numerics.md`, conventions). A physical
spring constant `EA` [N] becomes `k = EA/ds^2`, a bending rigidity `EI` [N m^2] becomes
`k = EI/ds^6` for the torsional beam, and a tether of stiffness `K` [N/m] at one point
becomes `k = K/ds`. The beam sign: a left (counterclockwise) turn from `p` through `q` to
`r` gives `c < 0`.

**ds.** `Lag_Structure_Info { ds = ... }` gives every point the same weight `ds`. Without
it, every point gets the median nearest-neighbour distance among the vertices, and pyIBM
warns when the largest nearest-neighbour distance is more than twice the smallest. Give `ds`
explicitly when you know it. The usual rule is about two points per cell, `ds = dx/2`.

Structure files can be written from Python: build a `pyibm.structures.Structure` with
`Springs`, `DampedSprings`, `Beams`, `NonInvBeams`, `Targets`, `Masses` and call
`write_structure(case_dir, structure)`, which writes the `.vertex` file and one file per
element type present. `read_structure(case_dir, name, ds=None)` reads them back.

## 6. Hooks

The hooks file is a Python file in the case folder that defines any of these functions; the
driver calls the ones that exist.

| hook | called | must return |
|---|---|---|
| `update_targets(t, dt, s)` | before every force evaluation, with the time of that evaluation, and at every output (to measure the target error at the same instant) | nothing; set `s.targets.X` (and `s.targets.V` if the targets are damped) |
| `update_springs(t, dt, s)`, `update_beams`, `update_noninv_beams`, `update_masses` | before every force evaluation | nothing; change `s.springs.rest`, `s.springs.k`, `s.beams.curv`, ... in place |
| `lagrangian_force(t, dt, s, X)` | at every force evaluation, at the positions `X` | an extra force density per unit length, shape `(N, 2)` [N/m], added to the structure force |
| `external_force(t, grid, u, v)` | once per step, at the time of the force evaluation | a body force density `(Fu, Fv)` [N/m^2] on the solver's u and v locations; not part of `fx_body` |
| `initial_velocity(grid, layout_u, layout_v)` | once, when the case is built | `(u, v)` of shapes `layout_u.shape`, `layout_v.shape` |
| `bc_left(y, t)`, `bc_right(y, t)`, `bc_bottom(x, t)`, `bc_top(x, t)` | `mac` only, at the times the substeps need | the velocity `(u, v)` on that Dirichlet side at the coordinates given, arrays of the coordinates' shape or scalars |
| `on_output(step, t, state)` | at every output, after the history row and the files | nothing; `state` is the `Simulation` |

The time of the force evaluation is `t + dt/2` for the midpoint scheme and `t` for `pm1996`
(`docs/numerics.md`, time schemes). Boundary data are evaluated at `t`, `t + dt/2` and
`t + dt`.

Rules:

- **Write everything as a function of absolute time.** Hooks are called at several time
  levels, and `update_targets` again at output times, so a hook that accumulates (adds an
  increment per call) goes wrong. Use `s.X0`, the initial positions, for offsets.
- **Vectorise the `bc_*` hooks.** They get the coordinate arrays of the faces along a side
  (`grid.yc`, `grid.yv` on the left and right, `grid.xc`, `grid.xu` on the bottom and top).
- **Shapes.** `external_force` and `initial_velocity` must return arrays on the solver's
  layouts: `(Nx, Ny)` nodes for both components on `periodic_fft`; the u-faces and v-faces on
  `mac` (`grid.meshgrid("u")`, `grid.meshgrid("v")` give their coordinates). The driver
  raises `ValueError` otherwise. The first step's projection removes any part of the initial
  velocity that is not discretely divergence-free.
- **`on_output`** gets the `Simulation`: `state.solver.u`, `state.solver.p`,
  `state.structure.X`, `state.f` (the Lagrangian force), `state.Fu`, `state.Fv` (the spread
  force), `state.t_force`, `state.output_dir`, `state.history()`. The driver creates the
  output folder only when it writes the history, VTK files or a snapshot; a hook that writes
  its own file should call `os.makedirs(state.output_dir, exist_ok=True)` first.

**Helper modules.** The hooks file may import modules that sit beside it. Import them at the
top of the hooks file: while the hooks file runs, its own folder and the case folder are first
on `sys.path`, but a helper imported later, inside a hook function at call time, is not found.
Only a `name.py` file or a package folder directly beside the hooks file, or directly in the
case folder, counts as a per-case helper. Such helpers are imported fresh for every case, so
two cases with their own `helper.py`, or the same case after you edited its helper, never share
a cached copy; and no bytecode (`.pyc`) is written for them. Give a helper package an
`__init__.py`: a folder without one is imported fresh too, but a module of the same name that
is already imported hides it. Do not name case files after the standard library or after
`pyibm`, `numpy`, `scipy` or `matplotlib`: those are never imported again.

**Choosing the hooks file.** `Hooks { file = "hooks_oscillating.py" }` in the input file
picks another file (a path relative to the case folder, subfolders allowed), so several
inputs in one folder can use different hooks; `examples/lid_driven_cavity` does this. A
missing `hooks.py` means no hooks; any other named file must exist.

A minimal hooks file that moves every target sideways:

```python
import numpy as np

def update_targets(t, dt, s):
    s.targets.X[:, 0] = s.X0[s.targets.idx, 0] + 0.1 * np.sin(2 * np.pi * t)
```

## 7. Output

Output goes to `<case_dir>/<output_dir>` at step 0, every `print_dump` steps and at the
last step of the run.

**`history.csv`**, one row per output, read it with `pyibm.io.history.read_history(path)`
(a record array: `h.t`, `h.fx_body`, ...) or any CSV reader. Numbers have 17 significant
digits and the file is flushed after every row.

| column | meaning |
|---|---|
| `step`, `t` | the step `n` and `t = n dt` of the state the row describes |
| `t_force` | the time level of `fx_body`, `fy_body`: `t - dt/2` (midpoint), `t - dt` (`pm1996`), 0 in row 0 |
| `fx_body`, `fy_body` | the force the fluid puts on the structure [N in 2D], `-sum(F) dx dy` of the spread structure force, without the `external_force` hook |
| `max_target_error_over_dx` | the largest distance between a target and its point, in cells (0 without targets) |
| `cfl` | the CFL number; on `mac` an upper bound, up to twice the pointwise number (`docs/numerics.md`, diagnostics) |
| `mean_u`, `mean_v` | the box-mean velocity |
| `kinetic_energy` | `(rho/2) sum (u^2 + v^2) dx dy` [J in 2D] |
| `max_speed` | the largest speed on the nodes (FFT) or cell centres (MAC) |

The force in a row is the force of the last step, at `t_force`, not at `t`. When you compare
forces between runs, or with a reference, put them at `t_force`.

**VTK files** (legacy ASCII, for ParaView or VisIt), when `save_vtk` is on:
`velocity.NNNNN.vtk`, `vorticity.NNNNN.vtk`, `pressure.NNNNN.vtk` on the nodes (FFT, origin
0) or on the cell centres (MAC, origin `(dx/2, dy/2)`, the face velocities averaged there),
and `lag.NNNNN.vtk` with the Lagrangian points, the spring and damped-spring links, and the
force per unit length as point data `force` (at `t_force`, like the history). With the blob
coupling the pressure includes the membrane pressure `p_F` (`docs/numerics.md`, blob).
`pyibm.io.vtk` reads them back: `read_structured_scalar`, `read_structured_vector`,
`read_lagrangian_points(path) -> X (N, 2)`, and `read_lagrangian_file` for the links and point
data too.

**Snapshots** `snapshot.NNNNN.png`, when `plot` is on: the vorticity with the boundary drawn
on top.

**Clean folder.** The first output of a `Simulation` (so of every command-line run) removes the files of those kinds that an earlier
run left in the output folder (exact names only) and a pyIBM `history.csv`, whatever
`save_vtk`, `plot` and `save_history` are set to, so the folder never mixes two runs. A
`history.csv` that is not a pyIBM history is never deleted: with `save_history` on we raise
`ValueError`, with it off we leave it alone. Files that hooks write (such as `area.csv` in
the examples) are never touched; a hook that appends to its own file should start it afresh
at step 0.

## 8. Stepping a case from Python

`pyibm.Simulation(params, case_dir=".", hooks=None, structure=None, output_dir=None,
quiet=False)` is the loop object behind `run`. It validates `params`, builds the grid, the
fluid solver, the coupling, the structure (from the files, unless one is passed) and the
hooks (from the hooks file, unless passed).

```python
import pyibm
from pyibm.inputfile import read_input_file

params = read_input_file("examples/rubberband/input2d")
sim = pyibm.Simulation(params, "examples/rubberband", quiet=True)
sim.output()            # row 0: history row, VTK files, on_output
for _ in range(100):
    sim.step()          # one step of the chosen scheme, t = n*dt
sim.output()
sim.close()             # close history.csv
```

`sim.run(steps=None)` writes the output of the current step, takes `steps` steps (default
`Tfinal/dt`) with output every `print_dump` steps and at the last step, and returns a
`RunResult`. A second `run()` on the same `Simulation` continues: the history file and
`RunResult.history` hold every row of the Simulation, and no row is written twice. A step
whose history row was written counts as output even if its VTK files, its snapshot or
`on_output` failed afterwards.

`sim.output()` at step 0 evaluates the structure force at the initial positions first (with
the hooks at `t = 0`), so row 0 holds the initial force whichever way you start.

Attributes: `grid`, `solver` (`u`, `v`, `p`), `coupling`, `structure`, `hooks`, `dt`,
`scheme`, `backend`, `n` (steps taken), `t` (`= n*dt`), `f` (the last Lagrangian force
density), `Fu`, `Fv` (the last spread force), `t_force`, `p_F` (blob only), `X_prev`,
`output_dir`. `sim.diagnostics()` returns the history columns of the current state without
writing anything, and `sim.history()` the rows written so far.

The builders are public too: `pyibm.driver.build_grid(params)`, `build_solver(params, grid,
hooks)` and `build_coupling(params, grid)`. Parameters built in code are validated the same
way as an input file.

## 9. Errors and warnings

- **Blow-up.** Every step ends with a check that the velocity, the positions and the ghost
  masses are finite. If they are not, the run raises `FloatingPointError` naming the step
  (the command line prints it and exits with code 1). Reduce `dt` or the stiffness.
- **Wall distance** (`mac` with the delta coupling). Every point must stay at least the
  kernel's half-width from every side that is not periodic: 2 cells for `peskin4` and
  `cosine4`, 1.5 for `roma3`. Closer, part of its force would land on the wall faces and be
  lost, so the driver raises `ValueError` naming the point, the side and the step, when the
  case is built or after any move.
- **CFL warning**, once per run, when the `cfl` column exceeds 1. On `mac` that number is an
  upper bound, so the warning is conservative there.
- **Tether warning**, once, when `k dt^2/(rho dx)` of the stiffest target or mass tether is
  above `pyibm.driver.TETHER_STABLE` (10 for targets, 7 for mass tethers, either scheme; a
  read-only table). It is a rule of thumb from our own test runs, not a stability limit
  (`docs/numerics.md`, diagnostics). Springs and beams are not checked.
- **Explicit viscosity warning** on `mac` with `viscous = "explicit"` when `dt` exceeds the
  explicit limit.
- **Spacing warning** when `ds` is inferred and the nearest-neighbour distances vary by more
  than a factor of two.
- **Input errors** (`ValueError`) name the file and line or the key.

## 10. Tests and tutorials

```
python -m pytest -q                     # the default suite
python -m pytest -q -m slow             # the long comparisons with published or analytic results
python -m pytest -q -m slow tests/notebooks    # execute the tutorials (needs the notebooks extra)
```

Each example has its own test file, `tests/examples/test_<name>.py`, with short smoke runs
in the default suite and the full comparison as a slow test. The tutorials in `tutorials/`
(`tutorials/README.md`) take the equations of Peskin & McQueen 1996, Seibold 2008 and
Cortez & Minion 2000 in order and run each piece with pyIBM.
