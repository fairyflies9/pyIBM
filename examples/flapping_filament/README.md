# Flapping filament in a uniform stream

A heavy, flexible filament pinned at its upstream end in a uniform flow does not sit still:
above a critical length it flaps by itself, like a flag. This is the two-dimensional "flag in
the wind" problem that Zhang et al. studied in a soap film and that Huang, Shin & Sung
simulated with an immersed boundary method (W.-X. Huang, S. J. Shin and H. J. Sung,
*Simulation of flexible filaments in a uniform flow by the immersed boundary method*,
J. Comput. Phys. 226 (2007) 2206-2228). The example reproduces their single filament case
of section 4.2, whose free-end history is their Fig. 13, with pyIBM's MAC back
end: springs for stretching, torsional beams for bending, mass points for inertia and a target
point for the support.

## The paper's case

The filament equations are their eqs. (3)-(4) with the scales of their section 2: length
`L_r`, speed `U_inf`, fluid density `rho_0`, and the filament's excess line density `rho_1`
(the mass per unit length above that of the fluid it displaces; `rho_1 = 0` is neutrally
buoyant). Their nondimensional groups (eqs. 5-6) and the values of the reference run (Fig. 13
caption and the text of section 4.2):

| group | definition | value |
|---|---|---|
| Reynolds number | `Re = rho_0 U_inf L_r / mu` | 200 |
| mass ratio | `rho = rho_1 / (rho_0 L_r)` | 1.5 |
| bending rigidity | `gamma` in units of `rho_1 U_inf^2 L_r^2` | 0.001 |
| Froude number | `Fr = g L_r / U_inf^2`, gravity along the stream | 0.5 |
| length | `L / L_r` | 1 |
| nodes | `N` | 64 |
| initial shape | straight at angle `k = 0.1 pi` from the stream, at rest (eq. 43) | |
| fixed end | simply supported, "BC1": `X = X_O`, zero curvature (eq. 9) | |

Flow and gravity both point along `+x` (their Fig. 1: the filament hangs downstream from the
support at `X_O`, the transverse direction is `y`). The fluid sees Dirichlet `(u, v) = (U_inf, 0)`
at the inflow and the two far-field sides and a convective outflow (their section 3.2.2), on
`-2 <= x <= 6`, `-4 <= y <= 4`, 513 x 251 cells (uniform in `x`, stretched in `y` outside
`|y| < 1`), `dt = 2e-4` to `6e-4`. Their filament is exactly inextensible (the tension is a
Lagrange multiplier, eq. 7); they couple it to the fluid with the feedback forcing of eq. (11).

What they report (Fig. 13, pinned end). I digitised the maxima and minima of that curve up
to t = 38, where its five convergence-test curves still lie within a pixel or two of one
another, to about 0.005 in `Y` and 0.08 in `t`; they are `REF_MAXIMA` and `REF_MINIMA` in
`analyze.py`. The figure in the journal's PDF is a raster with 333 pixels per unit of `Y`,
and its crests are so narrow that the dark run in a crest column takes in both flanks of the
line, so I read each extremum from the outer edge of the 2-pixel line, less half its width
(`analyze.py` says more). The transverse position of the free end `Y(0, t)` falls from the
initial 0.31 to -0.25 at t = 2.0, then swings with maxima of 0.24 (t = 3.7), 0.26 (7.0),
0.28 (10.3), 0.30 (13.5), 0.32 (16.7) and 0.33 (19.9), and settles at an amplitude of 0.33
for `t > 20`, symmetric about zero. The amplitude (half the mean peak-to-trough distance) is
0.29 over 10 < t < 15 and 0.31 over 10 < t < 20. There are 18 maxima between t = 3.7 and
t = 58.2, a period of 3.2 (frequency 0.31); over the first five swings (t < 17) it is 3.27,
over the digitised table 3.22. The text confirms the period: their Fig. 8 shows four
instants "that approximately span a flapping period", t = 9.2, 10.0, 10.8 and 11.6.

## The same case in SI and in pyIBM's elements

I take `L_r = 1 m`, `U_inf = 1 m/s`, `rho_0 = 1 kg/m^2`, so every nondimensional value above
is also its SI value here:

| quantity | value |
|---|---|
| viscosity | `mu = rho_0 U_inf L / Re = 0.005 N s/m^2` |
| excess line density | `rho_1 = 1.5 rho_0 L = 1.5 kg/m` |
| bending rigidity | `gamma = 0.001 rho_1 U_inf^2 L^2 = 0.0015 N m^2` |
| gravity | `g = Fr U_inf^2 / L = 0.5 m/s^2` along `+x` (`gravity_x` in `Lag_Structure_Info`) |
| stretching stiffness | `K_S = 100 rho_1 U_inf^2 = 150 N` (ours; theirs is inextensible, see below) |

pyIBM's Lagrangian forces are densities per unit length and each point carries a quadrature
weight `ds` that multiplies the force when it is spread (`docs/numerics.md`, conventions). The
energy an element contributes is therefore its per-unit-length energy times `ds`, and the
per-unit-length coefficients in the structure files follow from matching those energies to the
continuum ones for a chain of `n` segments of length `ds = L/n`:

- **Torsional beam** at every inner node (`.beam`, `curv = 0`): with `A = X_r - X_q`,
  `B = X_q - X_p` the energy is `k_b/2 (A x B)^2` and `A x B = ds^2 sin(dtheta) ~ kappa ds^3`
  for a turning angle `dtheta = kappa ds`. The physical energy per node is then
  `k_b/2 kappa^2 ds^7`, and the continuum bending energy over the arc `ds` a node represents
  is `gamma/2 kappa^2 ds`, so **`k_b = gamma / ds^6`** (1.61e6 at `ds = 1/32`, 2.52e4 at
  `ds = 1/16`). The test `test_element_stiffness_mapping` bends the committed filament into
  a circle of radius 2 and checks this to 0.1 %.
- **Spring** between neighbours (`.spring`, rest length `ds`): `k_s/2 (|d| - ds)^2 ds =
  k_s/2 eps^2 ds^3` against `K_S/2 eps^2 ds`, so **`k_s = K_S / ds^2`** (1.536e5 and
  3.84e4). The paper's filament is exactly inextensible; their Table 2 compares an
  extensible version with `K_S = 10^2, 10^3, 10^4` (units of `rho_1 U_inf^2`) with it. With
  `K_S = 100` ours stretches by about 1 % under the tension of this case (gravity on the
  heavy filament plus drag). In their Table 2 a stiffer chain needs a smaller `dt`; I have
  not tried one here (the spring tether number `k_s dt^2/(rho_0 dx)` is 0.31 for `input2d`;
  the driver does not check springs).
- **Mass point** on every node but the pinned one (`.mass`): the ghost mass is a density per
  unit length, so **`m = rho_1 = 1.5 kg/m`** directly, and the ghost equation
  `m dV/dt = -k_m (Y - X) + m g` (`docs/numerics.md`, time schemes) is per unit length too.
  Gravity acts on the ghosts only, which is exactly their eq. (3): `rho_1 g` is the weight
  of the excess mass; the fluid's own weight is absorbed in the pressure. The tether
  `k_m = 1e4 N/m` per unit length holds the ghosts within
  `|Y - X| ~ (m a + rho_0 U_inf^2)/k_m ~ 3e-4 m`, a hundredth of a cell, and rings at
  `sqrt(k_m/m) = 82 rad/s`, resolved by `dt`.
- **Target point** at the pinned node (`.target`): `f = k_t (X_T - X)` per unit length is a
  physical tether of stiffness `K_T = k_t ds`; I use `K_T = 1500 N/m`, so **`k_t = K_T/ds`**
  (4.8e4 and 2.4e4). The tension at the support is about 1 N, so the node sits within
  `1/1500 m = 0.02 cells` of `X_O` (`max_target_error_over_dx` in `history.csv`).
- **BC1 (pinned)**: the held node carries no beam, so the moment there is zero, as eq. (9)
  asks. **BC2 (clamped, `input2d_clamped`)**: the last two nodes are both targets, which pins
  the first segment horizontal, eq. (10). The free end carries no beam and no target (eq. 8).
- **Initial shapes**: eq. (43), a straight line at `0.1 pi` from the stream, for the pinned
  runs; their eq. (47) for the clamped run (segment `j` from the support turned by `j a`, so
  the first one is horizontal; with a total turn of 0.64 rad the free end starts at
  `X_O + (0.93, 0.31)`, the position they quote for `a = 0.01`, `N = 64`).

Every point has the same quadrature weight `ds = L/n`, so the `n + 1` points spread over a
length `L + ds`, 3 % more than `L` at 32 segments (the mass is exact: the pinned node has
none).

## What is different from the paper

- **Domain**: 6 x 4 with the support at `(1.5, 2)` instead of their 8 x 8 with the support 2
  behind the inlet and 4 from the sides, and a uniform grid. The far-field sides are the
  same Dirichlet free stream, so the blockage is about `(0.6/4)^2 ~ 2 %` in the velocity.
  This keeps the fine run at 192 x 128 cells.
- **Resolution**: 32 segments on `dx = 1/32` (`input2d`) or 16 on `dx = 1/16` (`input2d_coarse`)
  against their 64 nodes on a 513 x 251 grid. Their Fig. 14 (their convergence test in `N`,
  for `gamma = 0`) shows `N = 48` already deviating slightly from 56-72, so expect the coarse
  run to be off by more than the reading tolerance.
- **Start**: the fluid starts at rest and the free stream is ramped over one time unit with a
  half cosine (`hooks.py`); an impulsive start makes the mass tethers ring for several time
  units. The fine run's maxima and minima come 0.44 to 0.57 time units after the paper's
  up to t = 15, about half the ramp.
- **Stretching**: springs with `K_S = 100` instead of the inextensibility constraint
  (`tip.csv` records the length and the largest segment strain).
- **Coupling**: pyIBM spreads the elastic forces of the massless fluid points and tethers the
  mass to them (the penalty formulation of Kim & Peskin), not the feedback forcing of their
  eq. (11).

## Inputs

All three are written by `generate.py` (parameters live there only; the stiffnesses depend on
`ds`, so each input has its own structure files):

| input | structure | grid | segments | `dt` | output every | output dir |
|---|---|---|---|---|---|---|
| `input2d` | `filament` | 192 x 128 on 6 x 4 (`dx = 1/32`) | 32, pinned | 2.5e-4 | 200 steps (0.05) | `viz/fine` |
| `input2d_coarse` | `filament_coarse` | 96 x 64 (`dx = 1/16`) | 16, pinned | 8e-4 | 250 steps (0.2), with VTK | `viz/coarse` |
| `input2d_clamped` | `filament_clamped` | 192 x 128 | 32, clamped (BC2) | 2.5e-4 | 200 steps | `viz/clamped` |

`Tfinal = 30` for all three. Solver: `mac`, `midpoint`, `peskin4`, blended advection, implicit
viscosity; `left = dirichlet`, `right = outflow`, `bottom = top = dirichlet` with the free
stream from `hooks.bc_left/bc_bottom/bc_top`. The tether numbers `k dt^2/(rho_0 dx)` are
0.31 (spring), 0.10 (target) and 0.02 (mass) for `input2d`, and 0.39, 0.25, 0.10 for the
coarse input.

## How to run

```
python examples/flapping_filament/generate.py          # optional, rewrites the committed files
python -m pyibm run examples/flapping_filament --input input2d_coarse
python -m pyibm run examples/flapping_filament
python -m pyibm run examples/flapping_filament --input input2d_clamped
python examples/flapping_filament/analyze.py [--from T0]
```

`hooks.on_output` writes `<output_dir>/tip.csv` (free end, support, filament length, largest
segment strain at every output). `analyze.py` reads `tip.csv` and `history.csv` of every
variant that has been run, finds the maxima and minima of `Y(0, t) = y_tip - y_support` over
the last 10 time units, or the second half of a shorter run (or from `--from T0`), prints the
amplitude (half the mean peak to trough distance) and the period (mean spacing of successive
maxima and minima) next to the paper's over the same window, and saves `results.png`
(free-end history with the paper's extrema as squares and its settled amplitude as dotted
lines, length error, streamwise force on the filament at its own time, the `t_force` column
of `history.csv`).

## What to expect

The paper (Fig. 13, pinned end, from the digitised extrema): amplitude 0.29 over
10 < t < 15, 0.31 over 10 < t < 20 and 0.33 once settled (t > 20); period 3.2. `analyze.py`
compares each run with the paper over the run's own window (`paper_stats`). Measured here
(laptop, one process; per-step times are from runs with nothing else running; a rerun of
`input2d_coarse` with the current code on 2026-10-27 gave the same `tip.csv` to the last
digit):

| run | steps | wall time | amplitude | period | max strain | max target error |
|---|---|---|---|---|---|---|
| `input2d_coarse` to t = 30 | 37500 | 9-13 ms/step, 5.5-8 min alone (754 s with the fine run competing; the slow test's 25000 steps took 128-958 s, depending on what else was running) | 0.46 over 20 < t < 30 (paper 0.33) | 3.74 (paper 3.18); 3.60 over 10 < t < 30 (paper 3.21) | 2.5 % | 0.035 cells |
| `input2d` to t = 15 | 60000 | 3802 s (63.4 ms/step; t = 30 would take about 2.1 h) | 0.35 over 10 < t < 15 (paper 0.29), still growing (last maximum 0.37 at t = 14.1) | 3.26 (paper 3.23, a single gap between two maxima); 3.31 over 1 < t < 15 (paper 3.29) | 1.4 % | 0.044 cells |
| `input2d_clamped`, 2000 steps | 2000 | 70 ms/step | starts cleanly: free end from +0.30 to +0.26 by t = 0.5 | | 0.9 % | 0.017 cells |

The coarse run flaps steadily from t = 10 on (maxima 0.46, 0.44, 0.48, 0.45, 0.45, 0.47 at
t = 11.1, 14.7, 18.3, 21.7, 25.4, 29.1; minima to -0.48), but over 20 < t < 30 it overshoots
the paper by 39 % in amplitude and 18 % in period, and the free end comes within x = 0.67 of
the support where a 0.33 swing of a nearly straight filament would keep it beyond 0.9. The
domain is not the reason: the same coarse input on a 6 x 8 box with the support 4 from the
sides gives peaks within 6 % of these (0.27, 0.36, 0.43 against 0.28, 0.39, 0.46 at
t = 4.3, 7.6, 11.1) and, over 10 < t < 20, amplitude 0.44 and period 3.63 against 0.46 and
3.60: the side walls are worth about 5 % of the amplitude and nothing in the period.
Resolution is: the 16-segment filament is 16 cells long and the 4-point kernel spreads every
point over 4 cells, so the coarse filament is a quarter as thick as it is long and bends far
too easily. The fine filament keeps its tip beyond x = 0.81 (0.94 over the first swing).
Its first swing (-0.19 at t = 2.5, +0.20 at t = 4.2) is 25 % (trough) and 14 % (peak)
smaller than the paper's (-0.25 at t = 2.0, +0.24 at t = 3.7), 19 % peak to trough, and half
a time unit later because of the ramp.

To t = 15 the fine run's maxima are 0.20, 0.27, 0.33, 0.37 at t = 4.2, 7.5, 10.8, 14.1 and
its minima -0.19, -0.24, -0.30, -0.36 at t = 2.5, 5.9, 9.2, 12.5, so the swing is still
growing when the run stops. The paper's maxima are 0.24, 0.26, 0.28, 0.30 at t = 3.7, 7.0,
10.3, 13.5 and its minima -0.25, -0.25, -0.27, -0.29 at t = 2.0, 5.4, 8.7, 11.9. Ours starts
below the paper and grows faster: its maxima are 14 % below, then 2 %, 17 % and 23 % above.
Over 10 < t < 15 ours gives an amplitude of 0.35 against their 0.29 over the same window,
21 % high and already above their settled 0.33, and a period of 3.26 against their 3.23
(over 1 < t < 15, 3.31 against 3.29). Their 3.23 is the one gap between their maxima at
t = 10.3 and 13.5, each read to 0.08, so it is good to about 0.1. A straight swing of 0.37
would keep the tip beyond x = 0.93, so this filament also bends more than theirs, though far
less than the coarse one. Over
10 < t < 15, going from 16 to 32 segments took the amplitude from 0.45 to 0.35 and the period
from 3.53 to 3.26; the paper's 64 nodes on a grid twice as fine again in x (and four times in
y near the filament) are the next step, about 8 h per 30 time units with this code, and I have
not run it. So: the period is reproduced to 1 %, and the amplitude is not converged: the
fine run overshoots the paper's and is still growing at t = 15.

Use `input2d_coarse` to see the mechanism in minutes (it writes VTK output) and `input2d`
for numbers. The slow test runs the coarse input to t = 20 and holds it to its own flapping
state (amplitude 0.25-0.6, period 2.6-4.4, strain under 4 %), which also admits the paper's
values; it is a regression guard, not a validation.

## Files

- `generate.py`: parameters and the structure files (`filament*.vertex/.spring/.beam/.mass/.target`)
  and the three inputs.
- `hooks.py`: the ramped free stream on the left, bottom and top sides and the `tip.csv` recorder.
- `analyze.py`: amplitude and period against the digitised extrema of Fig. 13 over the same
  window, `results.png`.
- `tests/examples/test_flapping_filament.py`: three-step smoke runs of the three inputs, the
  committed files against `generate.py`, the nondimensional groups, the stiffness mapping
  (energy of a bent and a stretched filament), the hooks, and the slow coarse run to t = 20
  against Fig. 13.
