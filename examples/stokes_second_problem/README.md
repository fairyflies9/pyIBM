# Stokes' second problem

A flat wall slides back and forth in its own plane under a fluid that is at rest far away.
The fluid above it moves as a damped shear wave,

    u(y, t) = U exp(-y/d) cos(omega t - y/d),      d = sqrt(2 nu / omega),

the penetration depth `d` being the distance over which the amplitude falls by a factor e
and the phase lags by one radian (Stokes 1851; `stokes_exact.py` eq. 1, a textbook case:
Batchelor 1967 section 4.3, Schlichting & Gersten chapter 5). It is the simplest check of
the MAC back end's viscous term, time-varying Dirichlet wall (`hooks.py`, the `bc_bottom`
hook) and time stepping together, with nothing else in the way: there is no immersed
boundary, the advection term vanishes, the pressure is uniform and `v = 0`. The case is
fluid-only; `input2d` has no `Lag_Structure_Info { name }`.

The box has a second wall at rest at `y = Ly = 10 d`. The exact solution with that wall is
`stokes_exact.py` eq. 2, a `sinh` ratio, which differs from eq. 1 by at most
`2 U exp(-Ly/d) = 9e-5 U`: `analyze.py` prints the error against both and they agree to
seven digits, so the top wall stands in for "far away".

## Physics and parameters

| quantity | value |
|---|---|
| wall | `u = U cos(omega t)`, `U = 1 m/s`, `omega = 2 pi rad/s` (period `T = 1 s`), bottom of the box; top wall at rest |
| fluid | `rho = 1 kg/m^2`, `nu = omega d^2/2 = 0.031416 m^2/s` chosen so that `d = 0.1 m` |
| box | `Ly = 1 m = 10 d`; periodic in x with `Nx = 8` cells (the smallest the grid allows) and `Lx = 8 dy`, so the cells are square |
| resolution | `Ny = 32, 64, 128`: 3.2, 6.4 and 12.8 cells per penetration depth |
| time step | `dt = T/400 = 0.0025 s`; 5 periods (2000 steps); output every `T/8` (50 steps) |
| scheme | MAC back end, `midpoint`, implicit viscosity, non-incremental projection |

The constants live in `stokes_exact.py` and `generate.py`; the three inputs differ only in
`Ny` and the output directory. With `U = 1` and `d = 0.1` the Reynolds number `U d/nu` is
3.2, but it plays no role: the flow is exactly one-dimensional and the nonlinear term is
identically zero (the computed `v` and `p` stay at round-off, and `u` is the same on every
column of cells to 1e-16).

## Source

Analytic. `tests/examples/test_stokes_second_problem.py::test_exact_solution` checks that
eq. 1 satisfies `u_t = nu u_yy` with the derivatives taken by central differences (the
residual is the truncation error of the differences and falls by 100 when the steps fall
by 10), moves with the wall, is at rest far away, falls by e over one `d` and lags by one
radian there; and that eq. 2 is at rest on the top wall and within `U exp(-Ly/d)` of eq. 1.

## How to run

```
python -m pyibm run examples/stokes_second_problem                      # Ny = 64
python -m pyibm run examples/stokes_second_problem --input input2d_ny32
python -m pyibm run examples/stokes_second_problem --input input2d_ny128
python examples/stokes_second_problem/analyze.py
```

`analyze.py` reads `viz/ny*/history.csv` and the `velocity.NNNNN.vtk` dumps of the last
period (eight phases, `T/8` apart), averages `u` over the periodic direction, prints the
largest `|u - u_exact|/U` per phase for every variant that has been run and saves
`results.png`: the finest run's profiles at four phases on top of the exact curves, and the
largest error against `Ny`. `python generate.py` rewrites the inputs (deterministic; the
inputs are committed).

## What to expect

Measured on this laptop (2000 steps each, 41 history rows):

| input | `Ny` | cells per `d` | max `|u - u_exact|/U`, last period | run time |
|---|---|---|---|---|
| `input2d_ny32` | 32 | 3.2 | 1.92e-2 | 2.1 s |
| `input2d` | 64 | 6.4 | 5.48e-3 | 2.1 s |
| `input2d_ny128` | 128 | 12.8 | 1.45e-3 | 3.6 s |

The error falls by 3.5 and then 3.8 per halving of the cell: second order in `dy`, with the
time error at `dt = T/400` still below the spatial error at `Ny = 128`. It is largest at
`omega t = 90` and 270 degrees, where the wall is momentarily at rest and its acceleration,
and so the curvature `u_yy = u_t/nu` of the profile at the wall, is largest. (The wall shear
`(U/d)(sin omega t - cos omega t)` peaks later, at 135 and 315 degrees, where the error is
smaller: 4.3e-3 against 5.5e-3 at `Ny = 64`.) The error sits in the first few cells above
the wall; by `y = 5 d` the velocity is
below 7e-3 U and the computed and exact profiles are both zero to plotting accuracy. The
non-incremental projection's `O(dt)` wall slip (`pyibm/fluid/mac.py`) does not appear here
because the pressure, and so `phi`, is zero; the Womersley channel example shows what it
costs when it is not.

The start-up transient is gone by the last period: the kinetic energy of the fourth period
differs from the fifth by 3.6e-4 of its peak, and the largest profile error of the fourth
period differs from the fifth by less than 2e-5 U (under 0.3 % of the error itself:
1.4554e-3 against 1.4518e-3 at `Ny = 128`). Run the case longer by raising `PERIODS` in
`generate.py` if you want to see that for yourself.

Run time: about 1.1 ms per step on 8 x 64 cells, 1.8 ms on 8 x 128, dominated by Python
overhead; the three runs together take 8 s. The slow test
(`pytest -m slow tests/examples/test_stokes_second_problem.py`) runs all three and takes
11 s.

## Files

- `stokes_exact.py`: the constants, eqs. 1 and 2, and the finite-difference PDE residual.
- `generate.py`: writes `input2d`, `input2d_ny32`, `input2d_ny128`.
- `hooks.py`: `bc_bottom(x, t) = (U cos(omega t), 0)`, a function of absolute time.
- `analyze.py`: the comparison and `results.png`.
- `tests/examples/test_stokes_second_problem.py`: the exact solution and the phases where
  its wall shear (135 and 315 degrees) and wall acceleration (90 and 270) peak, the
  committed inputs against `generate.py` byte for byte, the hook, three-step smoke runs of
  the three inputs, and the slow full comparison (errors below 2.5e-2, 7e-3 and 2e-3 U with
  the ratios above 3, largest at 90 or 270 degrees).
