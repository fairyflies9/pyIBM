# Forced circle (Fovargue 2012): the error trend with the time step

The non-stiff test problem of Lauren Fovargue's thesis, *Addressing the computational cost
of the immersed boundary method through multi-implicit and multi-rate strategies with time
parallelism* (PhD thesis, University of North Carolina at Chapel Hill, 2012), section
5.4.0.1, printed pp. 57-61, Figures 5.4-5.6. A circle of 400 points sits in fluid at rest in
a doubly periodic unit box. The points on its right half are tethered to target points that
follow a prescribed ellipse: 40 % wider than the circle at a quarter period, back to the
circle at the half period, then 40 % narrower. The left half is free and follows through the
membrane. We run one period with the delta-function coupling (`input2d`) and the blob
projection (`input2d_blob`). `analyze.py --study` repeats the run at time steps from dx/2 to
dx/32 against a reference at dx/64 to show how the error falls with dt, which is what her
Figure 5.6 shows for her methods.

## Physics and parameters

| quantity | value |
|---|---|
| box | unit square, doubly periodic, 128 x 128 cells, dx = 1/128 |
| fluid | rho = 1, nu = 1e-4 (mu = 1e-4; "Re = 1000" in the thesis), at rest at t = 0 |
| circle | radius 0.2 about (0.5, 0.5), 400 points, ds = 0.00314 = 0.40 dx |
| targets | the 201 points with cos(theta) >= 0 follow T(theta, t) = (0.5 + a cos theta, 0.5 + b sin theta), a = c (1 + eps sin 2 pi t), b = c/(1 + eps sin 2 pi t), c = 0.2, eps = 0.4; a b = c^2, so the targets' half-ellipse keeps the area of the half-circle (`hooks.py`) |
| tether | k_T = 3200 N/m per m, zero rest length (thesis eq. 5.39) |
| membrane | neighbouring points joined by zero-rest-length springs, k_S = 1e4 N/m per m: a membrane under the constant tension k_S ds L = 0.099 N |
| time | T = 1 (one period); default dt = dx/16 = 4.88e-4 (2048 steps), output every 32 steps |
| scheme | periodic FFT back end, midpoint scheme, `remove_mean_force` on (the targets exert a net force; the box-mean velocity is held at zero, as in her stream-function formulation) |
| blob | `gaussian4`, delta = 4 dx, marker velocity from the interpolated stream function (the default `blob_interpolation`); her compact blob at 3 dx is compared below |

The geometry, the forcing law and nu follow the thesis. The membrane model and the
stiffnesses are our own choices (her membrane has stretching springs with the initial
spacing as rest length, eq. 5.38, plus a bending term sigma kappa n, eq. 5.37), so the
numbers below are a trend to compare with hers, not a reproduction. The choices made here,
each one measured:

- The membrane is a chain under tension (zero rest length) rather than her strain springs
  with a bending term. A chain at its rest length carries no tension, so without a bending
  stiffness nothing keeps it straight. In a delta run with k_S = 1e4 and the initial
  spacing as rest length, the chain had 44 kinked points at t = 1 (defined in the last
  section), all on the targeted half, and the free half came out wavy. The tensioned chain
  had 4 kinked points, 2 on each half, and a smooth free half. A tensioned chain never goes
  slack.
- The targets are soft. The boundary starts at rest while the targets start at 0.5 m/s,
  so the tether rings after the impulsive start at a frequency that grows with sqrt(k_T);
  the error only falls at second order once dt resolves that ringing. With k_T = 3200 the
  ringing is resolved from dt = dx/8 on; with k_T = 1e4 (delta coupling, against dx/64)
  the observed orders between dx/8 and dx/16 fell to 0.4 for E_u and 1.0 for E_X. The
  price is a lag of about one cell, which the tables below report.
- The explicit tether numbers k dt^2/(rho dx) are 0.31 (springs) and 0.10 (targets) at
  the default dt. The driver's rule of thumb checks targets and mass tethers only, against
  10, the largest value at which our own test runs stayed stable. The targets reach 6.25 at
  the coarsest level of the study (dx/2), so the tether warning never fires here; the
  springs reach 19.5 there and the runs stay finite. `analyze.py` prints any warning the
  driver gives next to the level it came from (the delta run at dx/2 gets the CFL warning,
  CFL 1.1).
- The blob variant uses the Gaussian fourth-order blob of Cortez & Minion (J. Comput.
  Phys. 161 (2000) 428-453) at 4 dx. Fovargue used a compact blob at 3 dx; "Blob choices
  at nu = 1e-4" below compares the two on this setup.

## How to run

```
python -m pyibm run examples/forced_circle                      # delta, viz/delta, 10 s
python -m pyibm run examples/forced_circle --input input2d_blob # blob, viz/blob, 50 s
python examples/forced_circle/analyze.py                        # histories + any saved study
python examples/forced_circle/analyze.py --study                # both couplings, about 8 min
python examples/forced_circle/analyze.py --study --coupling delta --factors 8,16 --reference 32
```

`analyze.py` prints the target stiffness next to the largest lag in cells and the force on
the circle for each variant that has been run. `results.png` shows the lag and the x force
against time, the time-step study, and the boundary of both default runs at t = 0.25 and
t = 1. `--study` runs dt = dx/2, dx/4, ..., dx/32 and the reference dx/64 for the chosen
couplings. It measures at t = T the errors of the thesis' eq. 5.40, E_u = max|u -
u_ref|/max|u_ref| over both velocity components and E_X = max_k |X_k - X_ref,k|, against the
finest run of the same coupling, prints the observed orders and saves
`viz/dt_study_<coupling>.csv`. `python generate.py` rewrites the structure and input files
(deterministic; committed, and the tests check that the committed files match it byte for
byte).

## What to expect

Measured on this laptop (errors at t = 1 against the run with dt = dx/64 of the same
coupling; the orders are between consecutive rows; the lag is the largest at the 65 rows
each level records, every T/64, as in the default runs):

delta-function coupling (`input2d`):

| dt | steps | E_u | order | E_X [m] | E_X / dx | order | max lag / dx |
|---|---|---|---|---|---|---|---|
| dx/2 | 256 | 7.7e-1 | - | 7.3e-3 | 0.93 | - | 0.96 |
| dx/4 | 512 | 5.4e-1 | 0.5 | 2.1e-3 | 0.27 | 1.8 | 0.98 |
| dx/8 | 1024 | 1.9e-1 | 1.5 | 7.5e-4 | 0.096 | 1.5 | 0.93 |
| dx/16 | 2048 | 4.3e-2 | 2.1 | 2.1e-4 | 0.027 | 1.8 | 0.93 |
| dx/32 | 4096 | 1.2e-2 | 1.9 | 4.0e-5 | 0.0051 | 2.4 | 0.93 |

blob projection (`input2d_blob`):

| dt | steps | E_u | order | E_X [m] | E_X / dx | order | max lag / dx |
|---|---|---|---|---|---|---|---|
| dx/2 | 256 | 6.7e-1 | - | 5.3e-3 | 0.68 | - | 1.42 |
| dx/4 | 512 | 7.7e-1 | -0.2 | 3.0e-3 | 0.38 | 0.8 | 1.28 |
| dx/8 | 1024 | 1.7e-1 | 2.2 | 1.2e-3 | 0.16 | 1.3 | 1.16 |
| dx/16 | 2048 | 3.4e-2 | 2.3 | 3.6e-4 | 0.046 | 1.8 | 1.17 |
| dx/32 | 4096 | 7.3e-3 | 2.2 | 1.1e-4 | 0.014 | 1.7 | 1.17 |

The thesis, Figure 5.6, standard algorithm (IMEX spectral deferred corrections, fourth
order, her stiffnesses), E_u read off the plot to about 15 %: 4.0e-3 at dt = dx/2, 3.1e-5 at
dx/10, 2.8e-7 at dx/50, 9.0e-8 at dx/100. Her E_u falls by about four orders of magnitude
between dx/2 and dx/50. Her fiber error E_X goes from 1e-5 at dx/2 down to 4e-10.

So: from dx/8 on, E_u falls at the second-order rate of the midpoint scheme for both
couplings (orders 1.5-2.3), and so does E_X for the delta coupling (1.5-2.4). For the blob
E_X falls a little slower, at orders 1.3-1.8. The reference is only twice as fine as the
last level, which blurs the last order. The two couplings give errors within a factor 3 of
each other. The two coarsest levels are outside the asymptotic range: the tether ringing is
not resolved there, and at dx/2 the CFL number reaches 1.1 (delta) and 0.8 (blob). Her E_u
is 170 to 190 times smaller than ours at dt = dx/2 and 3300 to 3700 times smaller at dx/10
(ours interpolated between dx/8 and dx/16). Her scheme is fourth order where ours is
second order, and the membrane model and stiffnesses here are our own, with a tether that
rings after the impulsive start until dt resolves it; so the trend, not the values, is the
comparison. The velocity field at nu = 1e-4 on 128 cells is under-resolved near the
boundary (the boundary layer sqrt(nu T) = 0.01 is about one cell).
The delta run's chain stays smooth (4 kinked points at t = 1, as defined in the next
section). The blob run's chain has 62 kinked points at t = 1, 60 of them on the targeted
half, and they turn by up to 0.79 rad (45 degrees) from one link to the next. The kinks are
at the scale of the marker spacing, half a cell, so they do not show at the scale of
`results.png`. Both runs converge in dt.

Other numbers from the default runs (dt = dx/16), read from the 65 rows of `history.csv`:
the largest target lag is 0.93 cells (delta) and 1.17 cells (blob) at k_T = 3200. The x
force on the circle swings between about -1.5 and +2.6 N (delta) and -2.1 and +3.0 N
(blob), ringing at the tether frequency on top of the forcing (the output every 32 steps
aliases that ringing in `results.png`). The fluid speed at the outputs peaks at 1.52 m/s
(delta) and 1.20 m/s (blob), against a peak target speed of 0.70 m/s (CFL 0.13 and 0.10).
Checked at every step the peaks are a little higher, 1.66 and 1.22 m/s, and checked every
fourth step the lag peaks at 0.96 and 1.19 cells. At t = 1 the targeted half is back on the
circle, while the free half has been pushed in to a bean shape.

## Blob choices at nu = 1e-4

Fovargue used the compact blob B(r) = 6/(pi delta^2) (1 - r^2/delta^2)^5 at delta = 3 dx
(`blob = "compact"` in pyIBM). We ran it on this case next to the Gaussian blob, with the
stream-function and bicubic marker velocities. Every run uses k_S = 1e4, k_T = 3200 and
dt = dx/16 unless stated, to t = 1. "Kinked points" counts the points at t = 1 where the
chain turns by more than 0.3 rad from one link to the next (the initial circle turns by
2 pi/400 = 0.016 rad per point). The peak speed is the largest max|u| over all steps.

| blob | interpolation | nu | runs to t = 1 | peak speed [m/s] | kinked points (of 400) |
|---|---|---|---|---|---|
| gaussian4, 4 dx (`input2d_blob`) | stream function | 1e-4 | yes | 1.22 | 62 |
| gaussian4, 4 dx | bicubic | 1e-4 | yes | 1.26 | 24 |
| gaussian4, 3 dx | stream function | 1e-4 | yes | 1.28 | 120 |
| compact, 3 dx, dt = dx/8, dx/16, dx/32 | stream function | 1e-4 | yes | 2.9, 2.9, 4.2 | 313, 312, 331 |
| compact, 3 dx | bicubic | 1e-4 | yes | 1.62 | 107 |
| compact, 4 dx | stream function | 1e-4 | yes | 6.9 | 320 |
| compact, 4 dx | bicubic | 1e-4 | no, goes unstable at t = 0.88 | - | - |
| compact, 3 dx | stream function | 1e-3 | yes | 1.20 | 46 |
| gaussian4, 4 dx | stream function | 1e-3 | yes | 1.06 | 4 |
| delta function (`input2d`) | delta | 1e-4 | yes | 1.66 | 4 |

Our chain has no bending term (her membrane has one, eq. 5.37), and on it the blob choice
matters at this viscosity. The compact blob at 3 dx runs to the end with either
interpolation. With the stream-function interpolation its chain folds at the marker scale
at every time step tried: most points are kinked, neighbouring links nearly reverse
(turning angles up to 3.1 rad), and the peak speed is 2.4 to 3.5 times the Gaussian blob's.
With bicubic it has 107 kinked points, and at 4 dx the bicubic run goes unstable. At
nu = 1e-3 the compact blob's peak speed is 1.20 m/s against 1.06 m/s for the Gaussian blob,
and its chain has 46 kinked points against 4. The Gaussian blob at 4 dx gives the smoothest
chain of the blob runs in the table. Bicubic gives it a smoother chain than the
stream-function interpolation (24 against 62 kinked points); we keep the default
interpolation, which the time-step study above shows converging.

The blob module has a third marker velocity, the exact adjoint of the projection
(`blob_interpolation = "adjoint"`; `docs/numerics.md` describes it). With the Gaussian blob
at 4 dx it gave the smoothest chain of all our blob runs, 2 kinked points (peak speed
1.77 m/s). With the compact blob it gave 12 kinked points at 4 dx (1.69 m/s) and 151 at
3 dx (1.77 m/s).

So at nu = 1e-4 we use the Gaussian blob at 4 dx. The compact blob runs too; of the runs
above, its smoothest chain is at 4 dx with the adjoint interpolation.

## Files

- `generate.py`: writes `circle.vertex`, `circle.spring`, `circle.target` and the two
  input files.
- `hooks.py`: `update_targets` with the forcing law as a function of absolute time.
- `analyze.py`: the histories, the time-step study (`--study`) and `results.png`.
- `tests/examples/test_forced_circle.py`: smoke runs of both inputs, the committed files
  against `generate.py`, the target motion against the thesis' formula, `analyze.py` on
  short runs, its warning notes and its 65 history rows per level, and a slow test of the
  time-step study for both couplings (orders above 1.5 between dx/8 and dx/16, against
  dx/32).
- `docs/examples.md` summarises this example with the others; `docs/numerics.md` describes
  the blob coupling and its interpolations.

Run times on this laptop with nothing else running: `input2d` 10 s, `input2d_blob` 50 s
(2048 steps; the Gaussian blob's curl is evaluated on a 27 x 27 node patch around each of
the 400 points, so the blob coupling costs about five times the delta coupling here). The
full study took 8.4 min, 1 min for the delta coupling and 7.5 min for the blob (the dx/64
reference alone is 8192 steps). The slow test runs dx/8 and dx/16 against dx/32 for both
couplings in 3 min. These times double or worse when other jobs share the machine.
