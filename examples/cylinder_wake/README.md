# Flow past a circular cylinder (Lai & Peskin 2000)

A rigid circular cylinder in a uniform stream at Re = 100 and 200: the von Karman vortex
street, its Strouhal number, the mean drag coefficient and the lift amplitude, compared
with the immersed-boundary computation in section 4 (Tables I and II) of

M.-C. Lai and C. S. Peskin, "An immersed boundary method with formal second-order accuracy
and reduced numerical viscosity", J. Comput. Phys. 160 (2000) 705-719,
doi:10.1006/jcph.2000.6483.

The cylinder is a ring of target points held by stiff tethers (their eq. 41). The fluid is
the MAC back end with a prescribed free stream on three sides and an outflow on the fourth.

## Physics and parameters

Everything is nondimensional and written in SI names: D = 1 m, U = 1 m/s, rho = 1 kg/m^2,
mu = rho U D / Re. Time is then in units of D/U and the force per unit span the fluid
puts on the ring is already C 0.5 rho U^2 D, so C_D = 2 fx_body and C_L = 2 fy_body
(their eqs. 44, 45, 51, 52). The Strouhal number is St = f D / U with f the frequency of
the lift (their eq. 53).

| quantity | value |
|---|---|
| box | 16 D x 8 D, cylinder centre at (4, 4): 3.5 D upstream, 11.5 D downstream, 3.5 D to each side |
| boundaries | left: inflow (U(t), 0); bottom and top: far field (U(t), 0); right: outflow (p = 0, zero-gradient velocity) |
| free stream | U(t) = U s(t/2) with s the quintic smoothstep 6x^5 - 15x^4 + 10x^3: at U from t = 2 on |
| cylinder | ring of target points on the circle, ds = dx (n = round(pi D / dx)), tethered to their initial positions |
| grids | `input2d`, `input2d_re200`: 192 x 96, D/dx = 12, dt = 0.003125, 38 markers, k = 64000; `input2d_fine`: 256 x 128, D/dx = 16, dt = 0.0025, 50 markers, k = 76800; `input2d_coarse`: 128 x 64, D/dx = 8, dt = 1/140, 25 markers, k = 19200 |
| solver | MAC, midpoint scheme, central advection (no upwind blend), Crank-Nicolson viscosity, 4-point Peskin kernel |
| length | 100 D/U; history every 0.1 D/U to `viz/<variant>/history.csv`; vorticity PNG every 10 D/U |
| trigger | a vertical body force in the near wake between t = 3 and 6 (`hooks.py`) |

The tether stiffness k (force per unit length per unit displacement) is set high enough
that C_D changes by about 1 % or less when it is raised further (see "The tether stiffness"
below). The time step keeps the
tether number k dt^2/(rho dx) at 7.50, 7.68 and 7.84 on the D/dx = 12, 16 and 8 grids. That
is below 10, where the driver starts to warn for targets (a rule of thumb, see
`pyibm/driver.py`), so the runs are quiet. The largest CFL number in the four runs is
0.09 to 0.14. Lai & Peskin's Table I has kappa dt^2/h = 9.95 on both of their grids.

Lai & Peskin's set-up differs from ours in three ways.
- Their box is periodic, [0, 8]^2 with D = 0.3 (27 D wide), the cylinder 6.2 D from the
  inflow side and the free stream imposed in a strip there. Ours is 8 D wide, with the free
  stream prescribed on the top and bottom 3.5 D from the cylinder.
- Their grids are D/h = 19.2 (h = 1/64) and 38.4 (h = 1/128); ours are D/dx = 8, 12 and 16.
- They ran to T = 240 in their time unit, the radius over U, which is 120 D/U; we run 100
  D/U.

### Starting the shedding

The symmetric flow is an exact solution of the discrete equations on this symmetric grid,
so without a push the wake stays symmetric for a very long time. Lai & Peskin seeded the
asymmetry with a random perturbation of the initial velocity. Here `hooks.external_force`
applies a vertical body force in the near wake, a Gaussian patch (half-width 0.5 D,
centred 1.5 D behind the cylinder) with a sin^2 envelope between t = 3 and t = 6 and
peak density 0.5 rho U^2/D, and nothing afterwards. The kick only decides when the street
starts: the lift reaches 95 % of its final amplitude by t = 6.3 in all four runs, and
from then on every shedding cycle is the same. The driver leaves the kick out of `fx_body`,
`fy_body`.

### Why the ramp is smooth

An earlier build of this example ramped the free stream linearly and jerked the cylinder
to start the shedding. Each kink in the forcing puts a jump into the force on the markers,
and with an explicit tether that jump rings the marker-tether system at its own period
(a few time steps) for the rest of the run, because the viscous damping at Re 100 is far
too weak to kill it. The stored drag then swung by more than the drag itself from one
output to the next. With the C^2 ramp and the smooth kick, C_D over t = 60-100 differs from
the sum of its first eight shedding harmonics by at most 2e-4 in the four runs (2e-5 at
Re 100 and 3e-5 at Re 200 on the D/dx = 12 grid).

### Marker spacing

We first built the ring at ds = dx/2, the usual spacing for a boundary that does not leak;
R. Cortez and M. Minion, "The blob projection method for immersed boundary problems",
J. Comput. Phys. 161 (2000) 428-453, give h/2 as the largest spacing that keeps fluid from
leaking across (p. 437). On our ring of tethered target points that spacing grows a
wrinkle: neighbouring markers move apart along and across the ring in patterns with
wavelengths of 1 to 1.7 cells (Fourier modes 15 to 24 of the 50 markers on the D/dx = 8
grid). The 4-point kernel barely sees a
pattern that fine, so the fluid does not push it back, and the tether force in that pattern
barely reaches the fluid either. With ds = dx/2 and k = 2000, the Re 100 run at D/dx = 12
had:

- a target error that grew to 0.64 cells by t = 40 and then wandered, up to 0.70;
- a St of single shedding cycles over t = 60-100 between 0.1645 and 0.1737;
- a per-cycle lift amplitude between 0.43 and 0.53;
- 10-unit means of C_D between 1.71 and 2.11 after t = 20.

So the wake was never periodic. Target damping (the third column of `.target`) did not
help on the D/dx = 8 grid: c = 10 moved C_D, St and the target error by 0.5 % or less, and
at c = 50 the run blew up. A stiffer tether does not cure it either. At k = 64000 on the
D/dx = 12 grid (75 markers, t = 15-30) the ring still sits 0.08 cells out of place in a
pattern of ring modes 33-35, and holding it there takes a force density of 126 N/m^2 on
average and up to 449 N/m^2. The ds = dx ring at the same k needs 7.7 and 16 N/m^2, and the
net load on the cylinder is only about 0.25 N/m^2 of circumference. The two rings give C_D
1.966 and 1.573. The other tight rings we tried behave the same way on the D/dx = 8 grid,
each against the ds = dx ring at the same k (mean force density 6.3-6.9 N/m^2):

- ds = dx/2, k = 307200: 0.025 cells out of place, force density 211 N/m^2 on average and
  up to 955, C_D 2.185 against 1.592;
- ds = 0.75 dx, k = 76800: 40 N/m^2 on average, C_D 1.767 against 1.594, and a lift
  amplitude of only 0.149 against 0.387;
- a disk filled with target points at the ring spacing (tether number 2; at 7.8 it blew up
  by t = 0.18): 56 and 85 N/m^2 on average at k = 19200 and 76800, up to 183 and 286, with
  C_D 1.899 and 2.080, so not settled in k.

The net load is tiny next to these forces, so they are almost entirely self-equilibrated
grid-scale forces, 6 to 33 times those on the ds = dx ring on average and 9 to 115 times at
their peak. We do not use these rings as references.

At ds = dx the wrinkle does not grow, and with a stiff enough tether (below) the target
error stays below 0.005 cells. The cost is a ring that lets fluid through. At t = 100 the
fluid inside r < R - 2 dx moves at up to 0.30 U (mean 0.13 U) on the D/dx = 12 grid,
0.43 U (0.27 U) on D/dx = 8 and 0.23 U (0.10 U) on D/dx = 16. On the D/dx = 12 grid the
ds = dx/2 ring at k = 64000 holds the fluid inside to about half of that (0.15 U and
0.066 U against 0.30 U and 0.13 U at t = 30).

### The tether stiffness

With ds = dx and a soft tether, the ring does not stay round at the grid scale. At
k = 2000 on the D/dx = 12 grid, the stiffness of an earlier build, the target error of
0.10 cells is almost all a static pattern: averaged over t = 15-30, ring modes 9 and up (of
38; the largest are modes 16-19, wavelengths of 2 to 2.4 cells) reach 0.098 cells, modes
0-3 reach 0.007 cells, and the part that changes in time reaches 0.016 cells. The pattern
does not grow, but the forces depend on it. As k rises the pattern shrinks and the forces
settle, once the target error is below about 0.005 cells. Re 100 unless marked, over
t = 20-40 on D/dx = 8 and t = 15-30 on the other grids, tether number 7.3-7.8 except at the
softest k (3.8), the committed k in bold:

| D/dx | k | mean C_D | lift amplitude | St | max target error (cells) |
|---|---|---|---|---|---|
| 8 | 1200 | 2.266 | 0.976 | 0.1594 | 0.15 |
| 8 | 4800 | 1.785 | 0.629 | 0.1677 | 0.028 |
| 8 | **19200** | 1.609 | 0.407 | 0.1704 | 0.0038 |
| 8 | 76800 | 1.594 | 0.387 | 0.1705 | 0.0009 |
| 8 | 307200 | 1.592 | 0.383 | 0.1706 | 0.0002 |
| 8 | 1228800 | 1.591 | 0.382 | 0.1706 | 0.0001 |
| 12 | 2000 | 1.974 | 0.584 | 0.1701 | 0.10 |
| 12 | 9000 | 1.816 | 0.472 | 0.1715 | 0.034 |
| 12 | 16000 | 1.692 | 0.402 | 0.1735 | 0.019 |
| 12 | 32000 | 1.594 | 0.366 | 0.1760 | 0.0075 |
| 12 | **64000** | 1.573 | 0.366 | 0.1767 | 0.0030 |
| 12 | 128000 | 1.568 | 0.367 | 0.1769 | 0.0014 |
| 12 | 256000 | 1.567 | 0.368 | 0.1770 | 0.0006 |
| 16 | 2400 | 1.899 | 0.550 | 0.1722 | 0.13 |
| 16 | 38400 | 1.713 | 0.443 | 0.1735 | 0.017 |
| 16 | **76800** | 1.589 | 0.384 | 0.1787 | 0.0040 |
| 16 | 153600 | 1.579 | 0.379 | 0.1794 | 0.0014 |
| 12, Re 200 | 2000 | 1.931 | 0.939 | 0.1922 | 0.099 |
| 12, Re 200 | 32000 | 1.370 | 0.574 | 0.2033 | 0.0084 |
| 12, Re 200 | **64000** | 1.348 | 0.579 | 0.2039 | 0.0039 |

It is the stiffness, not the time step. On D/dx = 8, a run at about half the time step
(tether number 1.9 instead of 7.3) at k = 76800 changes C_D and the lift amplitude by 1e-4
(1.5943 and 0.3867 against 1.5942 and 0.3868), and a quarter of the time step at k = 1200
gives the same C_D, lift amplitude and St to 1e-4. The committed inputs at dt/2 agree with
the committed runs to four digits too (below).

The committed k put C_D within 1.1 % and St within 0.4 % of the stiffest run on each grid,
and the lift amplitude within 1.3 %, except on the coarse grid, where it is 6.5 % high.
Lai & Peskin chose their kappa so that the largest displacement stays below 5 % of the
radius. On our ring, with ds = dx and our box and grids, the forces stop changing with k only
once the target error is below about 0.005 cells, so we chose k from the sweeps above.

## How to run

```
python examples/cylinder_wake/generate.py             # optional: rewrites the committed files
python -m pyibm run examples/cylinder_wake            # Re 100, D/dx = 12   -> viz/re100
python -m pyibm run examples/cylinder_wake --input input2d_re200    -> viz/re200
python -m pyibm run examples/cylinder_wake --input input2d_fine     -> viz/re100_fine
python -m pyibm run examples/cylinder_wake --input input2d_coarse   -> viz/re100_coarse
python examples/cylinder_wake/analyze.py
```

`analyze.py` reads every `viz/*/history.csv` that exists, converts the forces to C_D and
C_L at the time level of the force (the `t_force` column), and works over the last 40 time
units. It measures:

- St from the upward zero crossings of C_L, cross-checked with the peak of the
  zero-padded lift spectrum, and the St of each cycle;
- the mean C_D over whole shedding cycles, and the peak-to-peak of its oscillation at twice
  the shedding frequency;
- the lift amplitude as the mean over cycles of half the peak-to-peak of C_L.

It prints each next to the paper's values and saves `results.png`: C_D and C_L against
time with the analysis window shaded, and the lift spectrum with the paper's St marked.
The `viz/` folders are not in the repository; the runs write them.

## What to expect

Measured here over t = 60 to 100, the last 5-8 shedding cycles (`python analyze.py` prints
these). In every run, St, the mean C_D and the lift amplitude are the same in every cycle
to within 0.001 % (St), 1e-4 (C_D) and 3e-4 (C_L).

| input | D/dx | k | St (crossings / spectrum) | mean C_D | C_D 2f peak-to-peak | C_L amplitude | max target error | run time |
|---|---|---|---|---|---|---|---|---|
| `input2d` (Re 100) | 12 | 64000 | 0.1765 / 0.175 | 1.5705 | 0.015 | 0.3603 | 0.0034 cells | 8 min |
| `input2d_re200` (Re 200) | 12 | 64000 | 0.2042 / 0.203 | 1.3478 | 0.045 | 0.5757 | 0.0040 cells | 8 min |
| `input2d_fine` (Re 100) | 16 | 76800 | 0.1787 / 0.178 | 1.5878 | 0.018 | 0.3808 | 0.0047 cells | 18 min |
| `input2d_coarse` (Re 100) | 8 | 19200 | 0.1704 / 0.172 | 1.6070 | 0.014 | 0.4025 | 0.0039 cells | 1.6 min |

Lai & Peskin 2000, their second-order scheme (Scheme 2):

| source | D/h | St | C_D | C_L | max displacement |
|---|---|---|---|---|---|
| Table I, Re 100, h = 1/64 | 19.2 | 0.155 | 1.5167 | 0.2904 | 4.54 % of the radius (0.44 cells) |
| Table I, Re 100, h = 1/128 | 38.4 | 0.165 | 1.4473 | 0.3299 | 2.57 % of the radius (0.49 cells) |
| Table II, Re 100 (experiments: Williamson 0.166, Roshko 0.164) | 38.4 | 0.165 | | | |
| Table II, Re 200 (experiments: Williamson 0.197, Roshko 0.190) | 38.4 | 0.190 | | | |

The paper calls its C_D and C_L time-averaged. The mean lift is zero by symmetry, so we
compare its C_L with our lift amplitude.

On the committed grids, with the forces settled in k:
- At Re 100, St is 0.1704, 0.1765 and 0.1787 at D/dx = 8, 12 and 16. That is 3.3 %, 7.0 %
  and 8.3 % above the paper's 0.165, and 2.7-9.0 % above the experiments (0.164 and
  0.166). St rises with resolution, away from the paper.
- At Re 200, St is 0.2042 at D/dx = 12: 7.5 % above the paper and Roshko (0.190), and 3.7 %
  above Williamson (0.197).
- The mean drag is 11.0 %, 8.5 % and 9.7 % above the paper's finest value (1.4473) at
  D/dx = 8, 12 and 16, and the lift amplitude 22 %, 9.2 % and 15 % above (0.4025, 0.3603
  and 0.3808 against 0.3299).
- As in their Fig. 1, the drag oscillates at twice the shedding frequency. Its peak-to-peak
  is 0.015 at Re 100 and 0.045 at Re 200 on the D/dx = 12 grid, and the mean drag drifts by
  less than 2e-6 per time unit over the window.

Both the target error and the paper's maximum displacement are the largest distance of a
boundary point from its rest position. On the committed grids ours is 0.003-0.005 cells
(0.06-0.1 % of the radius); Table I gives 2.57 % and 4.54 % of the radius for theirs. On our
ring the forces settle only once the error is below about 0.005 cells.

What we established about the excess, all at Re 100:

- It is not the tether any more. On each grid the committed k gives C_D within 1.1 % of the
  stiffest run (the k sweeps above).
- It is not the time step. At dt/2, `input2d` over t = 20-40 and `input2d_coarse` over
  t = 60-100 give the same St, C_D and lift amplitude to four digits (0.1766, 1.5716 and
  0.3633; 0.1704, 1.6070 and 0.4025).
- Resolution changes the forces little. At the stiffest k on each grid, C_D is 1.591, 1.567
  and 1.579 at D/dx = 8, 12 and 16, and the lift amplitude 0.382, 0.368 and 0.379. St rises
  with resolution, 0.1706, 0.1770 and 0.1794. The paper's St also rose between its two grids
  (0.155 to 0.165), and its drag fell (1.5167 to 1.4473).
- The box accounts for about 4 % of the drag and 3 % of St. With the far-field sides at
  7.5 D instead of 3.5 D (a 16 x 16 box, 192 x 192 cells, the same ring and k), t = 30-60
  gives C_D 1.514, St 0.1708 and lift amplitude 0.361, against 1.571, 0.1765 and 0.361 in the
  default box over the same window. So the default box raises C_D by 3.7 % and St by 3.3 %
  and leaves the lift alone. In the taller box C_D is 4.6 % above the paper's finest value,
  St 3.5 % and the lift amplitude 9.5 %.
- The ring leaks (see "Marker spacing"), and we cannot tell how much that matters. Every
  tighter ring we tried gives a higher drag (1.77-2.19 on D/dx = 8, 1.97 on D/dx = 12), but
  each also carries grid-scale forces 6 to 115 times larger, so these runs say nothing clean
  about the leak.
- The Strouhal number of the earlier build agreed with the paper partly by cancellation.
  With k = 2000 on D/dx = 12, St was 0.170, 3 % above the paper, while C_D was 1.974, 36 %
  above. The soft tether's static pattern lowered St as it raised the drag; with the forces
  settled, St is 0.1765, 7 % above.

So with the forces settled in k, the Re 100 drag on the committed grids is 8.5-11 % above
the paper's finest value, the lift amplitude 9-22 % and St 3-8 %; at Re 200 St is 7.5 %
above the paper. On the D/dx = 12 grid about half of the drag excess and of the St excess is
the narrow box. The rest may come from the leak through the ring or from the 4-point kernel
smearing the boundary over several cells, and we have not separated those. Take this example
as a periodic vortex street whose forces have converged in the tether stiffness and whose
drag comes within about 10 % of the paper's, not as a converged reproduction of the paper's
numbers.

The run times come from timing runs of 1500 steps on a laptop with nothing else running:
6.9 ms per step at 128 x 64, 15.0 ms at 192 x 96 and 27.2 ms at 256 x 128. The two slow tests
took 77 s (`input2d_coarse` to the end) and 143 s (the first 30 time units of `input2d`).

## Files

- `generate.py`: writes `cylinder*.vertex`, `cylinder*.target` (one structure per grid,
  since the marker spacing follows dx) and the four input files.
- `hooks.py`: free-stream boundary data (`bc_left`, `bc_bottom`, `bc_top`), the kick
  (`external_force`) and the vorticity snapshots (`on_output`).
- `analyze.py`: St, mean C_D, lift amplitude, `results.png`.
- `tests/examples/test_cylinder_wake.py`: three-step smoke runs of the four inputs, the
  committed files against `generate.py`, the hooks, and the analysis on synthetic signals.
  The slow tests (`pytest -m slow`) run `input2d_coarse` to the end and the first 30 time
  units of `input2d`. They check that every shedding cycle has the same St to 1 %, that the
  target error stays below 0.01 cells, that St is near the paper's (within 5 % on the coarse
  grid, 10 % on the default one), and that St, C_D and the lift amplitude match the values
  above: within 2 % of the table for the coarse run, and within 1 % of the k = 64000 row of
  the k sweep (the same window, t = 15-30) for the short `input2d` run.
