# Clap and fling: the two-winged fling half-stroke of Miller & Peskin 2005

Two rigid wings, 1/6 chord apart and parallel at an angle of attack of 90 degrees, rotate
apart about their trailing edges to 45 degrees and then translate apart 4.5 chords: the
"fling" of Weis-Fogh, as my 2005 paper simulated it with target points on a doubly periodic
grid. The example runs the fling at Re 128 and Re 8 and compares the lift and drag
coefficient of each wing against time with the paper's Figs. 9 and 11. Source: L. A. Miller
and C. S. Peskin, *A computational fluid dynamics of 'clap and fling' in the smallest
insects*, J. Exp. Biol. 208 (2005) 195-212; the kinematics are eqs. 2-6 and the text below
them in Materials and methods, the target force with its damping eq. 11, the coefficients
eqs. 15-16. The 2009 paper (L. A. Miller and C. S. Peskin, *Flexible clap and fling in tiny
insect flight*, J. Exp. Biol. 212 (2009) 3076-3090) is where the target stiffness and the
force filtering come from.

## Physics and parameters

| quantity | value | paper |
|---|---|---|
| box | 1 m x 1 m, doubly periodic (`input2d`, `input2d_re8`) or no-slip (`input2d_mac`) | 1 m x 1 m periodic |
| grid | 512 x 512, dx = 1.95 mm | 1230 x 1230, dx = 0.81 mm |
| fluid | rho = 1000 kg/m^2, mu = 0.0235 N s/m^2 (the Dickinson & Goetz sucrose solution) | same |
| wing | chord c = 0.05 m, 51 target points at ds = c/51 = dx/2 | 120 points at dx/2 |
| gap between the wings | c/6 = 8.3 mm = 4.3 cells | 10 cells |
| speed V, Re = rho V c/mu | 0.06 m/s (Re 128), 0.00375 m/s (Re 8) | 0.00375 to 0.06 m/s |
| half-stroke | tau = t V/c from 0 to 6: 5 s at Re 128, 80 s at Re 8; 4.49 chords of travel | tau = 6, "about 4.5 chords" |
| rotation (eqs. 5-6) | about the trailing edge, 90 to 45 degrees, omega = (omega_rot/2)[1 - cos(2 pi tau/1.74)] from tau = 0 to 1.74 | same |
| translation (eq. 2) | v = (V/2)[1 + cos(pi + pi (tau - 0.86)/1.3)] from tau = 0.86 to 2.16, then V; no deceleration | same |
| walls | four lines of target points 0.025 m inside the periodic box (`input2d`, `input2d_re8`), 3876 points | 30 mesh widths inside (0.024 m) |
| target stiffness k | 2e5 N/m per m (Re 128), 900 (Re 8) | 1.44e5 in the 2009 paper |
| target damping c (eq. 11) | 1500 N s/m per m on the wing points at Re 128, none at Re 8; the walls are not damped | eq. 11 has a damping term c_targ |
| time step | dt = 0.002 s (Re 128), 0.032 s (Re 8): 2500 steps per half-stroke, tether number k dt^2/(rho dx) = 0.41 and 0.47, damping number c dt/(rho dx) = 1.54 at Re 128 | |
| scheme | `midpoint`, `peskin4` kernel, `delta` coupling, output every 5 steps (501 rows) | Peskin & McQueen 1996 with skew-symmetric advection |

Lift and drag per wing are the force the fluid exerts on that wing, minus the sum of `f ds`
over its 51 points (what `hooks.on_output` writes to `viz/<variant>/wing_forces.csv` every
output, with the time `t_force` of that force, half a step before the output time),
normalised by rho c V^2 / 2 (eqs. 15-16). Lift is the +y component; drag opposes the motion,
+x on the left wing and -x on the right. The two wings are mirror images and their
coefficients agree to 1e-11.

Three inputs, all written by `generate.py` (the stiffness lives in the `.target` file, so
each input has its own structure):

| input | structure | back end | what it is |
|---|---|---|---|
| `input2d` | `fling_re128` (wings + walls) | periodic FFT, 512^2 | Re 128, the paper's set-up |
| `input2d_re8` | `fling_re8` (wings + walls) | periodic FFT, 512^2 | Re 8 |
| `input2d_mac` | `fling_re128_box` (wings only) | MAC, 512^2, Dirichlet box | Re 128 in a real no-slip box instead of walls of target points |

`fling_kinematics.py` holds the geometry, the kinematics as functions of absolute time,
and the coefficient definitions; `hooks.py` moves the wing targets with it (positions and
velocities, which the damping uses) and writes the per-wing forces; `analyze.py` does the
comparison.

## How to run

```
python -m pyibm run examples/clap_and_fling                        # Re 128, about 4 minutes
python -m pyibm run examples/clap_and_fling --input input2d_re8    # Re 8, about 4 minutes
python -m pyibm run examples/clap_and_fling --input input2d_mac    # Re 128 in the MAC box, about 20 minutes
python examples/clap_and_fling/analyze.py
```

`analyze.py` prints, for every variant that has been run, the target stiffness with the
largest target error in cells, the landmarks of the left wing's C_L and C_D curves (raw and
filtered) next to the values read off the paper's figures, and saves `results.png`. VTK
output is off in the inputs (16 MB per dump at 512^2); set `save_vtk = true` to get it.

## What to expect

The reference values come from Fig. 9 (lift per wing, two-winged fling, Re 8 to 128) and
Fig. 11 (drag). I took them from the vector paths of the curves in the journal's PDF, with
the axes calibrated on their tick marks, so they are the plotted values to better than 0.01
in the coefficient and 0.005 in the fraction of the stroke. The Re 128 lift curve oscillates
by about +-0.05 near its rotation peak and +-0.09 over the last 0.05 of the stroke, so the
landmarks read there carry an uncertainty of that size.
"Rotation peak" is the first maximum (wing rotation), "trough" the minimum between the
peaks, "acceleration peak" the second maximum (translational acceleration), and "end" the
mean over the last 0.025 of the half-stroke. An earlier reading of Fig. 11 off a 300 dpi
rendering had the Re 128 drag trough at 1.6; the curve bottoms at 0.97 (at 0.234 of the
stroke), as the paper's own Fig. 3 does on the same mesh (0.98), and
`tests/examples/test_clap_and_fling.py` now checks the reference values against Fig. 3.

Re 128, `input2d` (left wing, filtered as described below; raw values in brackets):

| landmark | paper (frac, value) | this run (frac, value) | ratio |
|---|---|---|---|
| lift rotation peak | 0.16, 1.83 | 0.134, 2.59 (2.66) | 1.42 |
| lift trough | 0.24, 0.84 | 0.220, 1.00 (0.97) | 1.19 |
| lift acceleration peak | 0.34, 2.57 | 0.322, 3.09 (3.11) | 1.20 |
| lift minimum during translation | 0.80, 0.78 | 0.746, 0.77 (0.74) | 0.98 |
| lift at the end | 1.0, 1.03 | 1.0, 1.15 (1.15) | 1.11 |
| drag rotation peak | 0.08, 5.30 | 0.080, 7.23 (7.32) | 1.36 |
| drag trough | 0.23, 0.97 | 0.214, 0.89 (0.82) | 0.91 |
| drag acceleration peak | 0.33, 2.71 | 0.316, 3.06 (3.09) | 1.13 |
| drag at the end | 1.0, 1.40 | 1.0, 1.52 (1.52) | 1.09 |

Re 8, `input2d_re8` (raw: the Re 8 input is not damped and its traces are not filtered):

| landmark | paper (frac, value) | this run (frac, value) | ratio |
|---|---|---|---|
| lift rotation peak | 0.13, 3.00 | 0.112, 3.20 | 1.07 |
| lift trough | 0.24, 1.02 | 0.236, 0.90 | 0.88 |
| lift acceleration peak | 0.33, 2.99 | 0.316, 2.96 | 0.99 |
| lift at the end | 1.0, 1.67 | 1.0, 1.67 | 1.00 |
| drag rotation peak | 0.076, 24.5 | 0.068, 35.4 | 1.44 |
| drag trough | 0.22, 2.39 | 0.212, 2.47 | 1.03 |
| drag acceleration peak | 0.32, 5.70 | 0.296, 5.93 | 1.04 |
| drag at the end | 1.0, 3.03 | 1.0, 3.17 | 1.05 |

The Re 8 lift trace is not clean either. About a 7-row moving mean it wiggles by 0.06-0.13
peak to peak over the first 0.45 of the stroke (0.03-0.09 at Re 128 with its damping), and
the lift trough of 0.90 sits on one of those wiggles: through the 11-row moving average used
at Re 128 it is 0.965 (at 0.232), a ratio of 0.95 instead of 0.88. The same average moves
the other Re 8 landmarks by 3 % or less. Part of the wiggle is tether ringing. With k = 900
the period 2 pi sqrt(m_added/k) is 0.021 of the stroke at Re 8, and over 0.05-0.36 of the
stroke the strongest line in the lift trace is there, 0.031 in C_L at 0.022 (0.034 over
0.05-0.45). In trial runs over the first 0.36 of the stroke, tethers four times stiffer
(k = 3600) moved the strongest line to 0.011 of the stroke (0.041 in C_L), the period that
the same 67 kg/m of added mass gives, and damping the wings as at Re 128 (c = 94 N s/m^2,
damping number 1.54) cut it to 0.014 and raised the raw lift trough from 0.90 to 0.93. The
rest of the wiggle comes as the wings cross grid lines: from 0.45 of the stroke on, at full
speed, its period is 0.0065 of the stroke in the Re 8 and the Re 128 run alike, the time the
wings take to travel one cell. The damping took the wiggle about a 7-row mean from 0.024 to
0.016 rms and moved the Re 8 landmarks it reached by 3 % or less, so the committed Re 8
input stays undamped and its table is raw.

Re 128 in the MAC box, `input2d_mac` (filtered): rotation peaks 2.55 (lift, at 0.126) and
7.65 (drag, at 0.076), troughs 0.98 and 0.86, acceleration peaks 3.00 and 3.00, lift
minimum 0.76 at 0.746, end values 1.15 (lift) and 1.39 (drag), target error 0.53 cells.
Every landmark is within 9 % of the periodic run with walls of target points (the drag at
the end differs most, 1.39 against 1.52), so the paper's walls do their job and the two
back ends agree on this problem.

The shape and timing are the paper's: the rotation and acceleration landmarks of the two
periodic runs fall within 0.03 of the stroke of where the paper has them (at most 0.026,
the Re 128 lift rotation peak; 0.034 for that peak in the MAC box), most of them a little
early (the Re 128 minimum during translation 0.05 early), the Re 8 curves are within 12 %
everywhere except the drag peak of rotation (the 12 % is the raw lift trough; 5 % through
the moving average), and the Re 128 curves have the right peaks in the right order, troughs
within 20 % (lift 19 % high, drag 9 % low) and end within 11 % of the paper. The magnitudes
during rotation read high, by 36-42 % at Re 128 and 44 % for the Re 8 drag peak. That is
the grid. The gap between the wings is 4.3 cells here against 10 in the paper, and the
4-point kernel spreads each wing over 4 cells, so the fluid between the wings is squeezed
through a channel that is numerically much narrower than c/6; the force to pull the wings
apart is overestimated most at low Re, where it is a viscous squeeze. The paper's
refinement test (Fig. 3, a 0.5 m box at Re 128) shows the same trend: halving the mesh width
lowered the rotation peaks from 5.30 to 4.89 (drag) and from 1.85 to 1.69 (lift). Fig. 3 is
on a different time axis from Figs. 9 and 11, so only its peak and trough heights are
quoted. Running at 1230^2 would cost 6 times more per step (and more steps) than `input2d`;
the example stays at 512^2 and says so.

### Target lag, tether ringing and damping

The wings are held by tethers of stiffness k, damped at Re 128 with c (eq. 11 of the paper
has both terms; the value of c here is our own choice, below). The largest target error is
0.59 cells at Re 128 (0.57 near the lift peak of rotation, at 0.13 of the stroke, and 0.59
at the end of the stroke), 0.94 cells at Re 8 and 0.53 cells in the MAC box, above the 0.1 cells at which we
flag a lag. The tether numbers k dt^2/(rho dx) are 0.41 (Re 128) and 0.47 (Re 8), far below
the driver's rule of thumb of 10 for targets. A ten times stiffer wing at Re 128 (k = 2e6,
tether number 4.1, undamped) ran over the first 0.45 of the stroke and lagged only
0.11 cells, but rang at a period of 0.0075 of the stroke with an amplitude of 0.2 in C_L
(the raw lift trough fell to 0.57), so the example keeps k = 2e5. I checked at 256^2 that
halving dt at fixed k does not change the force traces to three digits.

A tethered rigid wing in a fluid is an oscillator: the tether spring against the added mass
of the fluid between and around the wings, with a period 2 pi sqrt(m_added/k) that is
0.115 s (0.023 of the half-stroke) at Re 128 with k = 2e5, so m_added is about 67 kg/m (per
unit length of wing). Undamped, the force traces carried that oscillation at an amplitude
of 0.07 in C_L. With the ten times softer tethers I started with (k = 2e4, 1000 steps) the
period was 0.36 s and the amplitude about 1, which swamped the rotation peak; at 256^2 (gap
2.1 cells) the amplitude was 1.3 even at k = 2e5. The Re 128 inputs therefore damp the wing
points with c = 1500 N s/m^2, about 0.2 of the critical damping
2 sqrt(k m_added) = 7300 N s/m^2. Over the first 0.45 of the stroke this took the ringing
from 0.070 to 0.011 in C_L, moved the filtered landmarks by 1.5 % or less (the lift rotation
peak from 2.63 to 2.59) and brought the raw landmarks close to the filtered ones (the raw
lift trough from 0.85 to 0.97, against 1.00 filtered); the largest lag went from 0.59 to
0.57 cells.
c = 3700 took the ringing to 0.008 but lowered the lift rotation peak by 3 %, and c = 7300
blew up in 67 steps: the damping is explicit, and its damping number c dt/(rho dx) was 7.5
there (1.54 for the committed 1500). At Re 8 the undamped ringing is weaker, about 0.03 in
C_L at its period (see the paragraph after the Re 8 table), and that input is not damped.

The 2009 paper filtered its force coefficients "to remove high frequency noise from the
vibrations of the elastic boundary", and `analyze.py` does the same at Re 128: a centred
moving average over one ringing period (11 rows, 0.022 of the stroke: the odd number of
rows nearest to 0.023 at the 0.002 row spacing), which now mostly smooths the small wiggle
left as the wings cross grid lines. It leaves 4.6 % of a sinusoid of the ringing period. On
curves shaped like the paper's it raises a trough by 2-3 % and lowers a peak by 0.5-1.6 %:
applied to the paper's own Re 128 curves, sampled at the same 0.002 spacing, it takes the
drag trough from 0.97 to 0.995, the lift trough from 0.838 to 0.858 and the drag rotation
peak from 5.30 to 5.25 (the narrow Re 8 lift trough would rise 2.9 %). The Re 8 traces are
not filtered. Raw and filtered landmarks are both printed; the tables above and the slow
tests use the filtered ones.

The box mean velocity stays below 1e-5 m/s in the periodic runs (the walls of target
points hold the fluid), so `remove_mean_force` is off as in the paper.

## Run time

Measured on this laptop (24 cores, `OMP_NUM_THREADS=2`) with the three runs going at once
and other jobs sharing the machine (2026-10-27): `input2d` 236 s and `input2d_re8` 237 s for
the 2500 steps (0.095 s per step; 3978 Lagrangian points), `input2d_mac` 1257 s (0.50 s per
step; 102 points, but the MAC step is dearer). The slow tests run the first 0.45 of the
half-stroke (1125 steps, the rotation and acceleration landmarks) of `input2d` and
`input2d_re8`, 48-460 s each depending on what else was running (three runs; the slowest
under heavy load). The smoke tests coarsen the grids to 128^2 and 64^2 and take a few seconds.

## Files

- `fling_kinematics.py`: parameters, kinematics (eqs. 2-6), wing and wall points, force
  coefficients (eqs. 15-16); shared by the other scripts and the tests.
- `generate.py`: writes `fling_re128.vertex/.target`, `fling_re8.*`, `fling_re128_box.*`
  and the three inputs (deterministic; the generated files are committed).
- `hooks.py`: `update_targets` with the wing motion as a function of absolute time and
  `on_output` writing `wing_forces.csv`.
- `analyze.py`: the comparison and `results.png`.
- `tests/examples/test_clap_and_fling.py`: smoke runs of the three inputs, the committed
  files against `generate.py`, the kinematics against the paper's formulas, the reference
  values against the paper's Fig. 3, the filter on a synthetic trace, the reader of force
  files written before the `t_force` column, and the slow comparisons at Re 128 and Re 8.
