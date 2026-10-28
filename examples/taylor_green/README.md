# Taylor-Green vortex

The one flow in this collection with an exact answer. On the doubly periodic unit box

    u = -cos(2 pi x) sin(2 pi y) exp(-8 pi^2 nu t)
    v =  sin(2 pi x) cos(2 pi y) exp(-8 pi^2 nu t)
    p = -(rho/4) (cos(4 pi x) + cos(4 pi y)) exp(-16 pi^2 nu t)

solves the incompressible Navier-Stokes equations: the nonlinear term is a gradient, the
pressure absorbs it, and the vortex array simply decays. The kinetic energy decays as
exp(-16 pi^2 nu t). We use it to check the two fluid back ends against each other and
against the exact solution, and to measure their orders of accuracy in space and time.
No structure, no forces: a fluid-only run.

## Parameters

| quantity | value |
|---|---|
| box | 1 m x 1 m, all four sides periodic |
| fluid | rho = 1 kg/m^2, mu = 0.01 N s/m^2 (nu = 0.01 m^2/s) |
| default grid | 64 x 64 (dx = 1/64) |
| time step | dt = 0.002 s, Tfinal = 0.5 s (250 steps); the amplitude decays to exp(-8 pi^2 nu 0.5) = 0.674 |
| scheme | `midpoint` (Peskin 2002 sec. 7) on both back ends; `advection = "central"` on the MAC grid |
| initial CFL | dt max(\|u\|/dx + \|v\|/dy) = 0.128 at t = 0 (velocity amplitude 1) |
| output | every 25 steps to `viz/periodic_fft/` and `viz/mac/` |

Two inputs with the same physics, both written by `generate.py`:

| input | back end | notes |
|---|---|---|
| `input2d` | `periodic_fft` | collocated nodes, FFT projection |
| `input2d_mac` | `mac` | staggered grid with all four sides periodic, central advection, implicit (CN) viscosity |

`docs/numerics.md` describes the two fluid solvers. The initial field comes from the
`initial_velocity` hook in `hooks.py`. The driver calls it once, when the run is built, with
the places u and v live on: the nodes on `periodic_fft`, the u and v faces on `mac`. It
writes the exact field at t = 0 there, so `python -m pyibm run examples/taylor_green` (or
`--input input2d_mac`) runs the decaying vortex. `analyze.py` builds the runs through the
same hook.

## What the grid sees

The viscous term on the grid uses the five-point Laplacian, whose symbol for the (1, 1)
mode is not 8 pi^2 but

    lambda_h = (8/h^2) sin^2(pi h) = 8 pi^2 (1 - (pi h)^2/3 + ...).

On both back ends the discrete nonlinear term of the Taylor-Green field is again a
discrete gradient, so the projection removes it exactly and the computed field is
u_exact(x_i) exp(-nu lambda_h t) up to the time-stepping error. Three consequences, all
measured below: the kinetic energy decays at the grid rate 2 nu lambda_h, which at 64^2 is
0.08 % below the exact 16 pi^2 nu; the spatial error against the exact solution is the
O(h^2) symbol error, which is why the orders come out at 2.00 so cleanly; and with
central advection each time step only multiplies the field by a scalar factor of the
viscous step, so the temporal table tests that factor and not the treatment of the
nonlinear term (see "Reading the tables").

The blended advection (`advection = "blend"`, the MAC default, from Seibold's notes, gamma
= min(1.2 dt max(|u|/dx, |v|/dy), 1)) mixes in upwinding with a weight proportional to dt.
That is what makes it robust, and it also adds an artificial viscosity of about 0.6 dt U^2
that stays as h -> 0 at fixed dt, so the blend column of the spatial table does not
converge in h. (At N = 16 its error is smaller than the central one because the extra
damping happens to offset part of the symbol error, which on its own makes the grid decay
slightly slower than the exact rate.) Use `central` when you want to measure the spatial
order; the blend is there for robustness with immersed boundaries at CFL numbers near 1.

## How to run

```
python -m pyibm run examples/taylor_green          # the default periodic_fft run, viz/periodic_fft/
python -m pyibm run examples/taylor_green --input input2d_mac
python examples/taylor_green/analyze.py            # the full tables below, about a minute
python examples/taylor_green/analyze.py --quick    # three resolutions, three time steps, about 15 s
python examples/taylor_green/analyze.py --no-plot  # skip results.png
```

`analyze.py` (1) runs the two default inputs with output to `viz/<backend>/` and measures
the kinetic-energy decay rate from `history.csv` by a least-squares line through
log KE(t); (2) prints a spatial convergence table (N = 16, 32, 64, 128 at dt = 0.001,
midpoint scheme) of the max-norm velocity error against the exact solution at T = 0.5 for
`periodic_fft`, `mac` with central advection and `mac` with the blend; (3) prints a temporal
convergence table (N = 32, dt = 0.01, 0.005, 0.0025, 0.00125) of the max-norm difference
from a run of the same scheme at dt/8, for `midpoint` and `pm1996` on both back ends; and
(4) saves `results.png`. The convergence runs write nothing to disk.

`python generate.py` rewrites the two input files (deterministic; they are committed).

## What to expect

Measured on this laptop (the two default runs: 250 steps at 64^2 in 0.55 s on
`periodic_fft`, 1.5 s on `mac`; the full script 59 s, of which 46 s is the 128^2 spatial
runs):

```
default runs (exact initial condition, output to viz/<backend>/):
  periodic_fft 64^2, dt 0.002, 250 steps: max error 2.136e-04, KE decay rate 1.57787 (exact 16 pi^2 nu = 1.57914, grid 2 nu lambda_h = 1.57787)
  mac          64^2, dt 0.002, 250 steps: max error 2.133e-04, KE decay rate 1.57787 (exact 16 pi^2 nu = 1.57914, grid 2 nu lambda_h = 1.57787)

spatial convergence at dt = 0.001, T = 0.5, midpoint:
         N            periodic_fft             mac central               mac blend
        16      3.410e-03               3.344e-03               6.490e-04
        32      8.541e-04   (2.00)      8.500e-04   (1.98)      2.909e-03  (-2.16)
        64      2.136e-04   (2.00)      2.134e-04   (1.99)      3.510e-03  (-0.27)
       128      5.340e-05   (2.00)      5.338e-05   (2.00)      3.661e-03  (-0.06)

temporal convergence at N = 32, T = 0.5, error against the same scheme at dt_min/8:
        dt   periodic_fft/midpoint     periodic_fft/pm1996            mac/midpoint              mac/pm1996
      0.01      1.370e-06               6.956e-05               1.364e-06               1.019e-03
     0.005      3.423e-07   (2.00)      3.222e-05   (1.11)      3.406e-07   (2.00)      5.025e-04   (1.02)
    0.0025      8.532e-08   (2.00)      1.510e-05   (1.09)      8.491e-08   (2.00)      2.434e-04   (1.05)
   0.00125      2.108e-08   (2.02)      6.935e-06   (1.12)      2.098e-08   (2.02)      1.137e-04   (1.10)
```

(error = max-norm velocity error; order log2(e_coarse/e_fine) in parentheses.)

Reading the tables:

- Both back ends decay at the grid rate 2 nu lambda_h = 1.57787 1/s to five digits, 0.08 %
  below the exact 1.57914; the max errors of the two back ends differ by 3e-7 at 64^2
  (`tests/test_backends_agree.py` checks that each back end returns the Crank-Nicolson
  factor times its initial field on its own points, and that the two decays agree to
  round-off).
- Spatial order 2 on both back ends with central advection; the error is the symbol error
  of the five-point Laplacian, (pi h)^2/3 relative in the decay rate, the same on both
  grids. The blend does not converge in h at fixed dt (see above).
- Temporal order 2 for `midpoint` on both back ends, order 1 for `pm1996`. Read these
  columns for what they are. The vortex is an eigenmode of every step that uses central
  advection, so the `midpoint` columns are, to four digits, the error of the scalar
  Crank-Nicolson factor ((1 - z/2)/(1 + z/2))^n with z = nu lambda_h dt, times the largest
  |u| on the points (1 on the nodes, 0.995 on the 32^2 faces, which is the 0.5 % between the
  two columns). The predictor and the advection of the midpoint step play no part, so this
  table does not test the predictor; for that, `tests/test_periodic_fft.py`
  measures the midpoint order on a multi-mode flow that is not an eigenmode. In the same
  way the `mac/pm1996` column (Seibold's one-step scheme: explicit central advection,
  backward Euler viscosity, Chorin projection) is the error of the backward-Euler factor
  (1 + z)^-n. Only the `periodic_fft/pm1996` column feels the advection: Peskin &
  McQueen's upwind differences of the vortex are not a discrete gradient. Its viscous step
  is backward Euler too, yet its error is 15 times smaller than that of the backward-Euler
  factor alone, so there the time errors of the viscous and the advection steps partly
  cancel. With the midpoint scheme the time error at dt = 0.002 (about 1e-7) is three
  orders below the spatial error (2e-4), so the default runs are spatially limited.

`results.png` has the three panels: KE(t)/KE(0) of the two default runs on the exact
curve, the spatial errors against the h^2 line, the temporal errors against dt and dt^2.

## Tests

`tests/examples/test_taylor_green.py`:

- smoke: three steps of each input at 16^2 from the hook's initial field through
  `analyze.run_case` (KE(0) = 0.25 exactly, max speed decays at exp(-8 pi^2 nu t) to
  0.5 %, max error below 1e-3, history and VTK files written, MAC divergence below 1e-12),
  and three steps of each input as it stands through `pyibm.run` on the case directory,
  the way the command line runs it (KE(0) = 0.25, so the vortex is there);
- the hook returns the exact field on the nodes and on the u and v faces;
- the committed inputs equal what `generate.py` produces, byte for byte;
- the decay rate at 32^2 over 50 steps equals 2 nu lambda_h to 0.1 % on both back ends
  (always on, about a second);
- `@pytest.mark.slow`: the convergence tables with N up to 64 and dt down to 0.0025: spatial
  orders above 1.9 for `periodic_fft` and `mac central`, the blend at least five times worse
  at 64^2, temporal orders above 1.9 for `midpoint` and between 0.9 and 1.4 for `pm1996`,
  and the `midpoint` and `mac/pm1996` errors equal to those of the scalar Crank-Nicolson and
  backward-Euler factors to 1 %. About 10 s
  (`pytest -m slow tests/examples/test_taylor_green.py`).

## Files

- `generate.py`: writes `input2d` and `input2d_mac`.
- `hooks.py`: the exact solution and the `initial_velocity` hook.
- `analyze.py`: the runner (`run_case`), the decay rate and the two convergence tables,
  `results.png`.
- `viz/periodic_fft/`, `viz/mac/`: history and VTK output of the default runs (not committed).
