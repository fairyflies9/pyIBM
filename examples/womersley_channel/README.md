# Womersley channel

Oscillating flow between two parallel walls driven by the pressure gradient
`-dp/dx = K cos(omega t)`: the plane-channel version of Womersley's pulsatile pipe flow
(Womersley 1955, J. Physiol. 127:553-563, which has Bessel functions where the channel has
a `cosh` ratio). With `u = Re[uhat(y) exp(i omega t)]` the exact profile is

    uhat(y) = (K / (i omega rho)) (1 - cosh(k (y - H/2)) / cosh(k H/2)),   k = (1 + i)/d,
    d = sqrt(2 nu / omega)

(`womersley_exact.py` eq. 1), and the Womersley number `alpha = (H/2) sqrt(omega/nu)`
compares the half-height with the Stokes layer `d`. At `alpha = 3` the profile is a
rounded parabola that lags the pressure gradient a little; at `alpha = 10` the core moves
as a plug `K/(rho omega) sin(omega t)`, 90 degrees behind the gradient, and the velocity
overshoots in the Stokes layers next to the walls (the annular effect, visible in
`results.png`).

The example drives the same flow in two ways:

- **inflow/outflow** (`input2d`, `input2d_alpha10`, `input2d_incremental`, the example
  proper): the exact profile is prescribed as a time-varying Dirichlet inflow on the left
  (`hooks.py`, `bc_left`), outflow on the right, no-slip top and bottom. The solver has to
  produce the pressure gradient `K cos(omega t)` by itself through the projection, so this
  checks the inflow and outflow sides, the time-varying boundary data and the pressure at
  once.
- **periodic, body force** (`input2d_periodic`, `input2d_alpha10_periodic`): periodic in x
  and driven by the uniform body force `K cos(omega t)` (`external_force`). There is no
  inlet and the pressure stays at zero, so the comparison isolates the error of the viscous
  term and the time stepping on the same grid.

No immersed boundary: the case is fluid-only.

## Physics and parameters

| quantity | value |
|---|---|
| fluid | `rho = 1 kg/m^2`, `nu = omega/100 = 0.062832 m^2/s` |
| forcing | `omega = 2 pi rad/s` (period `T = 1 s`), `K = rho omega U_core = 6.2832 N/m^3` so that the plug amplitude `K/(rho omega)` is 1 m/s |
| Stokes layer | `d = sqrt(2 nu/omega) = 0.1414 m` |
| channels | `alpha = 3`: `H = 0.6 m`; `alpha = 10`: `H = 2.0 m`. Same fluid and frequency, as in an experiment; `alpha = H/(sqrt(2) d)` |
| cells | square, `h = 0.025 m`: 5.7 cells per Stokes layer, 24 and 80 cells across the channel |
| length | inflow runs `Lx = 1.2 m` (48 cells) and `2.0 m` (80 cells); periodic runs `Lx = 0.2 m` (8 cells) |
| time step | `dt = T/400 = 0.0025 s`; output every `T/8` (50 steps) |
| duration | inflow 4 and 8 periods; periodic 8 and 50 periods (see below) |
| scheme | MAC back end, `midpoint`, implicit viscosity, non-incremental projection (incremental in `input2d_incremental`) |

| input | `alpha` | `H` | drive | grid | periods |
|---|---|---|---|---|---|
| `input2d` | 3 | 0.6 | inflow/outflow | 48 x 24 | 4 |
| `input2d_alpha10` | 10 | 2.0 | inflow/outflow | 80 x 80 | 8 |
| `input2d_incremental` | 3 | 0.6 | inflow/outflow, `mac_projection = "incremental"` | 48 x 24 | 4 |
| `input2d_periodic` | 3 | 0.6 | periodic, body force | 8 x 24 | 8 |
| `input2d_alpha10_periodic` | 10 | 2.0 | periodic, body force | 8 x 80 | 50 |

One `hooks.py` serves both heights: it reads `H` off the coordinates the solver hands it
(`womersley_exact.channel_height`). The constants live in `womersley_exact.py` and
`generate.py`.

## Source

Analytic. The channel solution is derived in `womersley_exact.py` (eqs. 1-3) and
`tests/examples/test_womersley_channel.py` checks it independently of its algebra:
`test_exact_solution_satisfies_the_momentum_equation` puts eq. 1 into
`u_t = (K/rho) cos(omega t) + nu u_yy` with central differences (the residual is the
truncation error of the differences and falls by 100 when the steps fall by 10), checks
no-slip on both walls, and integrates the profile by adaptive quadrature against the
flow-rate formula eq. 3; `test_exact_solution_limits` checks the quasi-steady parabola at
`alpha = 0.1` and the plug `K/(rho omega) sin(omega t)` at `alpha = 100`.

## How to run

```
python -m pyibm run examples/womersley_channel                                  # alpha 3, inflow
python -m pyibm run examples/womersley_channel --input input2d_alpha10
python -m pyibm run examples/womersley_channel --input input2d_incremental
python -m pyibm run examples/womersley_channel --input input2d_periodic
python -m pyibm run examples/womersley_channel --input input2d_alpha10_periodic
python examples/womersley_channel/analyze.py
```

`analyze.py` reads `viz/*/history.csv` and the velocity and pressure dumps of the last
period (eight phases), takes the profile on the column of cells at `x = Lx/2` and compares
it with eq. 1. Per phase it prints the largest error over the profile, the largest error
with the two cells next to each wall left out, the flow-rate error, and for the inflow runs
the pressure gradient between `x = Lx/4` and `3 Lx/4` next to `K cos(omega (t - dt/2))`.
The pressure in `pressure.NNNNN.vtk` comes from the projection of the step that ends at the
output time `t` and balances the momentum over that whole step, so it belongs to the middle
of the step, `t - dt/2`, and we compare it there. With the default projection the midpoint
predictor has no pressure term and the projection makes the whole pressure; with
`mac_projection = "incremental"` the predictor carries the previous step's pressure and the
projection adds the change. A least-squares fit of `-dp/dx` to `A cos(omega (t - lag))`
gives a lag of 0.64 `dt` for `input2d` and 0.56 `dt` for `input2d_incremental`. The script
saves `results.png`, the two inflow runs at four phases on top of the exact curves. `python
generate.py` rewrites the inputs (deterministic; the inputs are committed).

## What to expect

Measured on this laptop. Velocity errors are in units of the core amplitude
`K/(rho omega) = 1 m/s`, the flux error relative to the exact flow-rate amplitude, all over
the eight phases of the last period:

| input | max error, whole profile | interior (2 wall cells off) | flux error | `-dp/dx` vs `K cos(omega (t - dt/2))` | run time |
|---|---|---|---|---|---|
| `input2d` (alpha 3, inflow) | 1.34e-2 | 6.7e-3 | 1.1e-3 | within 0.50 % of `K` | 2 s |
| `input2d_alpha10` (inflow) | 1.38e-2 | 8.4e-3 | 2.8e-4 | within 0.14 % of `K` | 25 s |
| `input2d_incremental` (alpha 3) | 6.7e-3 | 3.4e-3 | 1.1e-3 | within 0.19 % of `K` | 2 s |
| `input2d_periodic` (alpha 3) | 6.9e-3 | 4.2e-3 | 3.1e-3 | (body force) | 2 s |
| `input2d_alpha10_periodic` | 6.9e-3 | 4.3e-3 | 8.6e-4 | (body force) | 15 s |

The two Womersley numbers give the same error because fluid, frequency, cell and time step
are the same and the Stokes layers are resolved identically; only the channel height
changes. The 6.9e-3 of the periodic runs, at the first cell off the wall, is the
discretisation error at 5.7 cells per Stokes layer and `dt = T/400`. The inflow runs carry
twice that at the wall cell, and the extra is the known `O(dt)` tangential slip of the
non-incremental projection (`pyibm/fluid/mac.py`): the wall velocity ends the step at
`dt * dphi/dx`, and with `dphi/dx = (K/rho) cos(omega t)` that is up to
`dt K/rho = 0.016`, which is what the table shows at the phases where `cos(omega t) = 1`
(the error peaks at `omega t = 0` and 180 degrees in the inflow runs). The body-force runs
have `phi = 0` and no slip. The incremental projection (Solver key
`mac_projection = "incremental"`, the input `input2d_incremental`) removes this slip: the
error at the wall cells falls from 1.34e-2 to 6.7e-3 and in the interior from 6.7e-3 to
3.4e-3, the level of the periodic runs.

Away from the inlet and the outlet the inflow runs are one-dimensional: over the middle
half of the channel `|v|` stays below 1e-3 and `u` varies along x by less than 1e-3 of the
core amplitude; in the periodic runs both are at round-off. The pressure gradient the
projection produces between `Lx/4` and `3 Lx/4` follows `K cos(omega (t - dt/2))` to 0.031
(alpha 3) and 0.0088 (alpha 10), 0.50 % and 0.14 % of `K`. We measure it against the forcing
half a step back because the projection pressure belongs to the middle of the step. Against
`K cos(omega t)` the same runs differ by 0.066 and 0.055, most of which is that half step,
`K sin(omega dt/2) = 0.049` at the phases where `sin(omega t) = 1`. What is left at alpha 3
is mainly a deficit in the amplitude (at `omega t = 0` we get 6.253 against 6.283, 0.48 %
of `K`) with a small extra lag (0.063 against 0.049 at 90 degrees, about 0.14 `dt`). It
comes from the first-order splitting of the non-incremental projection plus the spatial
error: with the incremental projection the largest difference falls to 0.012, 0.19 % of
`K`. The inlet and outlet quarters,
where the outflow condition `p = 0` and the inflow adjust the pressure, are left out of
the pressure measurement.

Start-up transients. The inflow runs impose the flow rate at the inlet, so the transient
leaves within a period or two: the kinetic energy of the second period already agrees
with the last to 1.3e-9 (alpha 3), and the seventh with the eighth to 7e-7 (alpha 10). In
the periodic runs the flow rate is free and the slowest mode, `sin(pi y/H)`, decays on
`H^2/(pi^2 nu)`: 0.58 s at `H = 0.6` (the seventh and eighth periods agree to 9e-6) and
6.5 s at `H = 2`, which is why that run lasts 50 periods (7.7 decay times; the last two
periods then agree to 3.4e-6 of the peak kinetic energy, and the error table above is
the same as for `alpha = 3`).

Run times: 1.3 ms per step on 48 x 24 cells, 8 ms on 80 x 80 (the sparse LU solves), 0.6
and 0.8 ms on the 8-cell-wide periodic channels. The five runs together take about 50 s;
the slow tests (`pytest -m slow tests/examples/test_womersley_channel.py`) rerun them.

## Files

- `womersley_exact.py`: the constants, eqs. 1-3, `channel_height`, and the
  finite-difference PDE residual.
- `generate.py`: writes the five inputs.
- `hooks.py`: `bc_left(y, t)` with the exact profile and `external_force(t, grid, u, v)`
  with the body force (zero when the grid is not periodic in x), both functions of absolute
  time.
- `analyze.py`: the comparison tables and `results.png`.
- `tests/examples/test_womersley_channel.py`: the exact solution and its limits, the
  committed inputs against `generate.py`, the hooks, the time level of the pressure in the
  comparison, three-step smoke runs of the five inputs (divergence-free, mass through the
  inlet equals mass through the outlet), and the slow full comparisons (whole-profile error
  below 2e-2, interior below 1e-2 and `-dp/dx` within 1e-2 `K` for the non-incremental
  inflow runs; 1e-2, 5e-3 and 5e-3 `K` for the incremental one; 1e-2 and 5e-3 for the
  periodic runs). The slow tests also check the fitted lag of the pressure in the two
  alpha 3 inflow runs and in four `pm1996` runs of `input2d` (both projections, `dt = T/400`
  and `T/800`), where it is 0.51 to 0.74 `dt` and `t - dt/2` still matches better than `t`
  or `t - dt`.
