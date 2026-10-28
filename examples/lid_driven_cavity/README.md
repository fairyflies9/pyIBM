# Lid-driven cavity

The standard test of a solver with walls: the unit square, three walls at rest, the top wall
(the lid) sliding in +x at U = 1. The flow settles to a steady primary vortex with
secondary eddies in the bottom corners. The reference is U. Ghia, K. N. Ghia and C. T. Shin,
"High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid
method", J. Comput. Phys. 48 (1982) 387-411: a vorticity-stream-function solution on a
129 x 129 uniform grid with a coupled strongly implicit multigrid method. Their Table I
(p. 398) lists u along the vertical centreline x = 0.5 and Table II (p. 399) lists v along
the horizontal centreline y = 0.5, at 17 points each, for Re = 100, 400, 1000 and higher.
The Re = 100, 400 and 1000 columns are copied into `ghia1982.py`.

This example runs the MAC back end only (walls need it), with no structure.

## Physics and parameters

| quantity | value |
|---|---|
| cavity | 1 m x 1 m, Dirichlet on all four sides |
| lid | `bc_top` in `hooks.py`: u = U_LID = 1 m/s, v = 0, uniform all the way into the corners (the usual singular corners) |
| fluid | rho = 1 kg/m^2, mu = 1/Re N s/m^2, so Re = U L/nu |
| scheme | `mac`, `midpoint` (Peskin 2002 sec. 7), `advection = "central"`, `viscous = "implicit"` (Crank-Nicolson in the full step) |
| time step | dt = 0.4 h/U; the solver's CFL number dt (max\|u\|/dx + max\|v\|/dy) settles near 0.6 |
| length | until the relative change of the kinetic energy, \|dKE/dt\|/KE, is below 1e-4 per second: the transient decays at about 0.55/s (Re 100), 0.26/s (Re 400) and 0.11/s (Re 1000) |
| output | history every `print_dump` steps plus the two centreline files written by `on_output` |

The four inputs, all written by `generate.py`:

| input | Re | grid | Tfinal | dt | output |
|---|---|---|---|---|---|
| `input2d` | 100 | 64 x 64 | 18 | 0.00625 | `viz/re100/` every 80 steps |
| `input2d_re400` | 400 | 128 x 128 | 40 | 0.003125 | `viz/re400/` every 640 steps |
| `input2d_re1000` | 1000 | 128 x 128 | 70 | 0.003125 | `viz/re1000/` every 1120 steps (slow test only) |
| `input2d_oscillating` | 100 on the peak speed | 128 x 128 | 4 periods of 1 s | 0.003125 | `viz/oscillating/` every 8 steps, no VTK |

How the walls are handled (`pyibm/fluid/mac.py`; `docs/numerics.md`, fluid solvers): the
normal velocity on a wall face is set exactly; the tangential component uses the ghost
reflection 2 u_wall - u_interior (eq. 35 of B. Seibold, "A compact and fast Matlab code
solving the incompressible Navier-Stokes equations on rectangular domains", course notes for
MIT 18.086, 2008); the Poisson problem is all-Neumann and solved by the DCT fast path with
the null mode pinned; the box is closed, so the compatibility check (zero net inflow) holds
trivially. The projection is the
non-incremental Chorin one, which leaves a tangential slip of order dt at the walls and a
numerical boundary layer in the pressure. At these time steps the velocity profiles do not
show it; the comparison below is the measure.

The centrelines are read straight off the staggered grid, with no averaging: u lives on the
vertical faces x = i dx and face Nx/2 is exactly x = 0.5; v lives on the horizontal faces
y = j dy and face Ny/2 is exactly y = 0.5. `on_output` writes `u_centreline.csv` (y, u from
the bottom wall u = 0 to the lid u = 1) and `v_centreline.csv` (x, v from wall to wall) at
every output, overwriting, so the final files hold the end state. `analyze.py` interpolates
them linearly to the 17 tabulated points.

## The oscillating lid

`input2d_oscillating` is a demonstration with no reference: the same cavity with the lid
speed U(t) = sin(2 pi t) (`hooks_oscillating.py`, a function of absolute time, evaluated by
the driver at the t, t + dt/2 and t + dt levels the midpoint scheme needs). Its input file
names that hooks file in its `Hooks` section,

    Hooks {
        file = "hooks_oscillating.py"
    }

so it sits next to the steady inputs, which use the default `hooks.py`. Re = 100 on the
peak lid speed; the Stokes layer under the lid has thickness
sqrt(2 nu/omega) = sqrt(2 x 0.01/(2 pi)) = 0.056, about 7 cells at 128^2.
`analyze.py` prints the kinetic-energy maximum of each period and saves
`results_oscillating.png`, the kinetic energy over the four periods with U(t) on a second
axis. The energy peaks twice per lid period (the lid does work in both directions) and the
cycle repeats after the first period: the maxima are 0.01203, 0.01073, 0.01065 and
0.01063 J, the last two within 0.2 %.

## How to run

```
python -m pyibm run examples/lid_driven_cavity                        # Re 100, about 15 s
python -m pyibm run examples/lid_driven_cavity --input input2d_re400  # about 4 min
python -m pyibm run examples/lid_driven_cavity --input input2d_re1000 # about 7 min
python -m pyibm run examples/lid_driven_cavity --input input2d_oscillating   # about 25 s
python examples/lid_driven_cavity/analyze.py
```

`analyze.py` prints, for every steady case that has been run, the largest differences from
Tables I and II, the per-point table, and the steadiness of the end state
(|dKE/dt|/KE over the last output interval, from `history.csv`), and saves `results.png`
(u on x = 0.5 and v on y = 0.5 against the tabulated points). Then the oscillating report.
`python generate.py` rewrites the four input files (deterministic; they are committed).

## What to expect

Measured on this laptop (the differences are in units of the lid speed; "steadiness" is
|dKE/dt|/KE at the end of the run):

| input | Re | grid | max \|u - Ghia\| on x = 0.5 | max \|v - Ghia\| on y = 0.5 | steadiness [1/s] | wall time |
|---|---|---|---|---|---|---|
| `input2d` | 100 | 64^2 | 0.0038 | 0.0087 | 2.2e-5 | 13 s (2880 steps) |
| `input2d_re400` | 400 | 128^2 | 0.0021 | 0.0048 | 2.7e-5 | 4 min (12800 steps) |
| `input2d_re1000` | 1000 | 128^2 | 0.0033 | 0.0119 | 5.7e-5 | 7 min (22400 steps) |

The wall times are for an idle laptop, from the cost per step measured on short runs (4.6 ms
at 64^2, 18 ms at 128^2; the time loop is dominated by the two sparse LU solves and the DCT
Poisson solve per substep). The full runs above were made while other jobs shared the
machine and took 1.5 to 2.5 times longer (Re 400: 502 s, Re 1000: 959 s).

The profiles stop moving well before the runs end: at Re 100 the numbers at t = 15 were
0.0037 and 0.0087, at Re 400 the numbers at t = 35 were 0.0020 and 0.0047, at Re 1000 the
numbers at t = 60 were 0.0034 and 0.0118 (steadiness 1.1e-4 there, which is why that run
goes to 70).

Ghia's 129 x 129 grid has the same spacing as our 128^2 cells, so at Re 400 and 1000 the
remaining difference is mostly the two discretisations (their second-order
vorticity-stream-function scheme against our MAC projection) and, at Re 1000, the thin
wall layers: the Re 1000 difference is largest in v at x = 0.95-0.97, inside the layer on
the right wall, and in u at y = 0.05-0.10 near the bottom. At Re 100 on 64^2 the largest
difference sits in the v profile near the right wall (x = 0.86-0.91), where the downward
jet is steepest and our grid has half their points.

The comparison leaves out one tabulated point: v at x = 0.9063 for Re = 400, -0.23827,
which is out of line with its neighbours (-0.22847 at x = 0.9453, -0.44993 at x = 0.8594).
Our run gives -0.387 there. `ghia1982.py` keeps the value as tabulated and lists it in
`LEFT_OUT`; `analyze.py` and the tests leave it out of the pass/fail numbers (they still print
it, marked; `results.png` shows it as a red cross).

## Tests

`tests/examples/test_lid_driven_cavity.py`:

- smoke: three steps of `input2d` and of `input2d_oscillating` into a temporary directory
  (history rows, VTK files or their absence, divergence below 1e-12, the lid drags only the
  top row, the centreline files are the wall-to-wall face lines, the oscillating run uses
  `hooks_oscillating.py` and not `hooks.py`); about 2 s;
- the committed inputs equal what `generate.py` produces, byte for byte; the steady and
  oscillating lid hooks; the Ghia tables compared with themselves give zero error;
- `@pytest.mark.slow`: the three steady runs against Tables I and II with the limits
  0.012 (Re 100), 0.01 (Re 400) and 0.02 (Re 1000) on both centrelines, each run checked
  to be steady (steadiness below 1e-4) and under the CFL limit. Run them with
  `pytest -m slow tests/examples/test_lid_driven_cavity.py`; together about 12 minutes,
  the Re 1000 one alone about 7.

## Files

- `generate.py`: writes `input2d`, `input2d_re400`, `input2d_re1000` and `input2d_oscillating`.
- `hooks.py`: `bc_top` (the steady lid) and `on_output` (the centreline files).
- `hooks_oscillating.py`: `bc_top` with U(t) = sin(2 pi t), named by `input2d_oscillating`.
- `ghia1982.py`: Tables I and II for Re 100, 400, 1000, the point left out, and `compare`.
- `analyze.py`: the comparison, `results.png`, `results_oscillating.png`.
