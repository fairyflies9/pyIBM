# Flapping plate at Re 64, the tilted hybrid stroke of Jones et al. 2015

A rigid flat plate hovering with the "tilted stroke" of Jones, Laurenza, Hedrick, Griffith
& Miller, *Lift vs. drag based mechanisms for vertical force production in the smallest
flying insects*, J. Theor. Biol. 384 (2015) 105-120: the plate sweeps back and forth along a
stroke plane tilted 62.8 degrees from the horizontal while it pitches about its midchord, so
it makes vertical force with drag on the downstroke and lift on the upstroke (their Fig. 2c).
The plate is a line of target points pulled toward the prescribed pose, the same model as the
heaving plate but with a real stroke and a real Reynolds number. It runs on both back ends:
`input2d` is the doubly periodic FFT box, `input2d_mac` the same box with no-slip walls at
rest (the paper's boundary condition).

## Physics and parameters

The paper prescribes the midchord position and the angle of attack (section 2.1, eqs. 2-3)

    [x(t), y(t)] = (A0/2) (1 + cos(2 pi f t)) [cos(beta), sin(beta)]
    alpha(t)     = alpha_0 + B sin(2 pi f t)

with `A0/c = 2.5`, `f = 1`, `B = pi/4` for every stroke, and `alpha_0 = 60 deg`,
`beta = 62.8 deg` for the tilted one. `kinematics.py` writes the same motion about the
midpoint of the stroke, `s(t) = (A0/2) cos(2 pi t/T)` along `(cos beta, sin beta)`, so the
stroke is centred in the box. We take the direction of the chord tilt from the paper's
Fig. 2: `alpha` is the counterclockwise angle from the stroke direction
`(cos beta, sin beta)` to the trailing-edge-to-leading-edge vector, which
puts the leading-edge-to-trailing-edge vector at `theta = beta + alpha + pi` in the lab
frame. Fig. 2a (the horizontal stroke, `alpha = 135 deg` at mid-downstroke) settles it: the
chord there runs from the leading edge at the upper left to the trailing edge at the lower
right, 315 deg = 0 + 135 + 180. With this the leading edge leads on both half strokes.
At `t = 0` the plate is at the upper end of the stroke with `alpha = alpha_0`, pitching
nose-down at its peak rate (the paper has no start-up ramp either); the downstroke is the
first half of every cycle (the plate moves down and to the left, the angle of attack rising
to 105 degrees at mid-stroke) and the upstroke the second half (up and to the right,
feathered to 15 degrees). The Reynolds number is their eq. (10), `Re = rho U_max c / mu`
with `U_max = pi f A0` the peak speed of the midchord, and forces are made dimensionless with
`0.5 rho U_rms^2 c`, `U_rms = U_max / sqrt(2)` (section 2.2). The mirror-image convention
`theta = beta - alpha` (trailing edge leading the upstroke) is kept in `kinematics.py` as
`leading_edge_forward = False`; see "What to expect" for what it gives.

The dimensional values are our own choice (they live in `generate.py`) instead of the
paper's dimensionless ones; the coefficients do not care.

| quantity | value |
|---|---|
| chord | c = 0.1 m |
| stroke | A0 = 2.5 c = 0.25 m peak to peak, stroke plane at beta = 62.8 deg |
| pitch | alpha = 60 deg + 45 deg sin(2 pi t/T) about the midchord |
| speed, period | U_max = 24 m/s, so T = pi A0 / U_max = 0.032725 s; U_rms = 16.97 m/s |
| fluid | rho = 10 kg/m^2, mu = rho U_max c / Re = 0.375 N s/m^2 (Re = 64) |
| force scale | q = 0.5 rho U_rms^2 c = 144 N/m |
| box | 1.2 m x 1.2 m (12 chords; the paper used 20), 192 x 192 cells, dx = c/16 |
| plate | 32 target points, ds = dx/2, at (i + 1/2) ds - c/2 along the chord (none on a tip) |
| target stiffness | k = 1e9 N/m per unit length (tether number k dt^2/(rho dx) = 0.42; the driver warns above 10 for targets, a rule of thumb) |
| time step | dt = T / 6400 = 5.113e-6 s; CFL 0.01-0.05 |
| length | 3 cycles (19200 steps); the paper averages cycles 3 and 4 |
| output | a history row every 1/800 cycle (`print_dump = 8`, fine enough to resolve the force oscillation as the plate crosses grid lines), a vorticity PNG every quarter cycle |

Both inputs use the midpoint scheme and the 4-point kernel. `input2d` sets
`remove_mean_force = 1`: in a periodic box the net force on the plate would otherwise
accelerate the whole box, cycle after cycle; removing the mean force is the same as a uniform
body force that holds the box's mean velocity at zero. The forces written to `history.csv`
are the spread structure force before that removal. `input2d_mac` needs no such thing; its
walls are 4.4 chords from the swept region. The CFL number is only 0.01-0.05: the small time
step goes with the stiff targets, and a softer plate would lag its targets more (see the lag
in the tables below and what stiffer plates do after them).

The vertical force coefficient `C_V = F_y/q` and the horizontal one `C_H = F_x/q` are the
paper's; `analyze.py` also gives drag and lift in the frame of the motion (`C_D` against the
instantaneous direction of the midchord, `C_L` along the upward normal to the stroke plane)
and `C_T`, the mean of `|F|/q`, for their ratio `C_V / C_T`. A history row carries the force
of the last step, which the midpoint scheme evaluates half a step before the row's time; the
`t_force` column of `history.csv` gives that time, and `analyze.py` puts every coefficient,
its phase and the direction of motion there (a row still belongs to the cycle of its own
time, so every cycle is the same 800 rows).

## Source and reference values

We run at Re 64, between the Reynolds numbers the paper plots, so every reference number
below is read off a plot and interpolated in Re; the reading error is about 0.03 on
the cycle means, 0.01 on the ratios and 0.3 on the peaks (the plots are small).

- Fig. 11A, cycle-mean `C_V` of the tilted stroke against Re (markers at Re 40, 60, 80,
  100, 200, averaged over cycles 3-4): 0.88, 1.00, 1.10, 1.17, 1.32, so about **1.02 at
  Re 64**. Fig. 12A, `C_V / C_T`: 0.232, 0.284, 0.318, 0.344, 0.441, so about **0.29**.
- Fig. 22B and 23B (rigid plate, black) give the same quantities at Re 5, 10, 50, 100:
  `C_V` 0.92, 0.73, 0.98, 1.15 and `C_V / C_T` 0.113, 0.126, 0.289, 0.353; interpolated
  between 50 and 100 that is 1.03 and 0.31 at Re 64. Figs. 11-12 are the rigid wing: their
  tilted-stroke values sit on the rigid curve of Figs. 22B-23B, not the flexible one
  (`k'_beam = 31`, section 3.3; 0.92, 1.22, 1.48 at Re 10, 50, 100).
- Fig. 21(b.ii) (rigid, black) and the Re 100 curve of Fig. 6, `C_V(t)` and `C_H(t)` at
  Re 100 over cycles 3-4: `C_V` starts the downstroke near 1, peaks at 6.2 at 0.19 of the
  cycle, crosses zero at 0.44, bottoms at -2.2 just after the reversal (0.53) and creeps back
  to zero through the upstroke; `C_H` peaks at 3.4 near 0.10, is -2.9 at the reversal,
  rebounds to -1.0 near 0.65 and bottoms at -3.1 near 0.85. Fig. 21(b.i), the same at
  Re 10: `C_V` 10 / -6.5, `C_H` 5.2 / -7.5. Re 64 should sit between the two, nearer Re 100.
- Their grid (Table 1, section 2.3): 512^2 effective on a 20-chord box, 25.6 cells per
  chord, `dt = 1e-6` in their units (`T/1e6`), no-slip walls. Table 2 says 12.8 cells per
  chord differs from 25.6 by about 15 % at Re 100 and 6 % at Re 10. This example's 16 cells
  per chord is between those, and its periodic 12-chord box is another difference.

## How to run

```
python -m pyibm run examples/flapping_plate_re64                      # periodic box, FFT
python -m pyibm run examples/flapping_plate_re64 --input input2d_mac  # no-slip box, MAC
python examples/flapping_plate_re64/analyze.py
```

`analyze.py` reads `viz/periodic/history.csv` and `viz/mac/history.csv` (whichever exist),
prints per cycle the means of `C_V`, `C_H`, `C_L`, `C_D`, `C_T`, the extrema of `C_V` and
`C_H` with the phase at which they occur and the largest target lag in cells, next to the
paper's values, and saves `results.png`. `python generate.py` rewrites `plate.vertex`,
`plate.target` and the two inputs (deterministic; the generated files are committed).
`generate.write_case(dir, cells_per_chord=..., stiffness=..., steps_per_cycle=...,
cycles=...)` writes a case at another resolution into another directory; the tests use it.

VTK output is off (a row every 1/800 cycle would mean 2400 sets of 2 MB files); set
`save_vtk = 1` and a larger `print_dump` in the input if you want ParaView files. The
vorticity PNGs come from the `on_output` hook in `hooks.py`.

## What to expect

Measured on this laptop while other jobs were running (3 cycles, 19200 steps, 2401 history
rows; rerun with the current code on 2026-10-27: the means and extrema are the same to the
digits shown as in the earlier run; three phases are 0.01 lower, because phases are now taken
at the force time, half a step earlier, and these three sat on a rounding edge). The
cycle means are from the raw rows; the extrema are from a centred running mean over 1/50 of
a cycle (16 rows) of `C_V` and `C_H`, because the raw force oscillates as the plate crosses
grid lines (one cell every 0.008 T at mid-stroke, the usual immersed-boundary artefact of a
stiff penalty body; `analyze.py` prints the raw extrema too, and `results.png` shows the raw
rows in grey under the smoothed curves). The paper's values are the interpolated Re 64 means
of Figs. 11A and 12A and the Re 100 time series of Fig. 21(b.ii); Re 64 should sit a little
above the Re 100 curve in magnitude.

| run | cycle | mean C_V | C_V / C_T | mean C_H | C_V max (phase) | C_V min (phase) | C_H max (phase) | C_H min (phase) | max lag [cells] |
|---|---|---|---|---|---|---|---|---|---|
| paper, interpolated to Re 64 | 3-4 | 1.02 +- 0.03 | 0.29 +- 0.01 | | | | | | |
| paper, Re 100 (Fig. 21 b.ii) | 3-4 | 1.15 | 0.35 | | 6.2 (0.19) | -2.2 (0.53) | 3.4 (0.10) | -3.1 (0.85), -2.9 at the reversal | |
| `input2d`, periodic FFT | 1 | 1.106 | 0.263 | -0.437 | 7.28 (0.17) | -3.47 (0.47) | 3.64 (0.05) | -4.63 (0.53) | 0.126 |
| | 2 | 1.116 | 0.280 | -0.199 | 7.25 (0.16) | -3.00 (0.47) | 4.07 (0.04) | -3.70 (0.49) | 0.109 |
| | 3 | 1.086 | 0.275 | -0.185 | 7.16 (0.16) | -3.00 (0.47) | 4.09 (0.04) | -3.54 (0.79) | 0.109 |
| `input2d_mac`, no-slip MAC | 1 | 1.105 | 0.259 | -0.459 | 7.05 (0.17) | -3.30 (0.47) | 3.59 (0.05) | -4.63 (0.52) | 0.123 |
| | 2 | 1.056 | 0.267 | -0.225 | 6.99 (0.17) | -2.97 (0.45) | 4.01 (0.05) | -3.75 (0.51) | 0.107 |
| | 3 | 1.049 | 0.268 | -0.185 | 6.93 (0.17) | -2.95 (0.45) | 4.06 (0.05) | -3.58 (0.79) | 0.109 |

So the cycle-mean vertical force is within 3-7 % of the paper's interpolated value (1.05
for the no-slip box and 1.09 for the periodic one against 1.02, with a reading error of
0.03 on the paper's number and a 1-3 % drift from cycle 2 to cycle 3 in ours, so the runs
are not yet periodic to better than that; the paper also averaged cycles 3-4), `C_V / C_T`
is within 5-8 % (0.268 and 0.275 against 0.29), and the
time series has the paper's shape: `C_V` peaks near 0.16 of the cycle, falls through zero
near 0.37, bottoms just before the reversal and creeps back to zero through the upstroke;
`C_H` peaks early in the downstroke, dips at the reversal, rebounds to about -0.3 to -0.6
near 0.62 and bottoms again near 0.79. One difference in shape: our `C_V` crosses zero
upwards at 0.90-0.91 of the cycle and is already about 3.2 at the start of the next
downstroke (the smoothed curve of cycles 2 and 3), where the paper's is near 1. In cycles 2
and 3 the extrema are larger than the paper's Re 100 ones by 12-36 % (the `C_V` maxima by
12-17 %, the `C_V` minima by 34-36 %), as a lower Re and a coarser grid (16 against 25.6
cells per chord; the paper's Table 2 puts 15 % between 12.8 and 25.6 at Re 100) both push
them. The two back ends agree with each other to 0.1 % in the first cycle's mean, 1-3 % in
its maxima and 5 % in its `C_V` minimum; by the third cycle the periodic box is 3.5 % above
the no-slip one in the cycle mean (1.09 against 1.05) as the wake fills the box, so the box
(12 chords here, 20 in the paper) and the mean-force removal matter at that level and no
more.

The plate lags its targets by up to 0.126 cells (0.8 % of the chord) in the first upstroke
and 0.109 afterwards, over the 0.1-cell mark at which we flag a lag in these examples.
The lag is what the stiffness k = 1e9 allows: k times the lag is the tether force per unit
length at the worst point, 7e5 N/m, well above the plate's average of about 1e4 N/m
(q C_V / c). A stiffer plate at the same dt lags less but rings more. Over the first half
cycle of `input2d`, k = 1e9, 2e9 and 4e9 (tether numbers 0.42, 0.84 and 1.67, all far under
the driver's rule of thumb of 10) ran stably with largest lags of 0.086, 0.056 and 0.037
cells, but the force oscillation as the plate crosses grid lines grew: the rms of the raw
`C_V` about its running mean went from 0.74 to 0.80 and 1.21, and the mean `C_V` over that
half cycle from 3.17 to 3.21 and 3.25. So the example keeps k = 1e9 and states the lag.

The mirror-image convention (`leading_edge_forward = False`, trailing edge leading the
upstroke) gave, in a coarse check at 8 cells per chord over two cycles, a cycle-mean `C_V` of
0.47 against 1.22 for the paper's convention on the same grid and a positive mean `C_H`;
it does not look like the paper's Fig. 2c or Fig. 21(b).

Run time, with other jobs sharing the machine: 207 s for the periodic run (10.8 ms per step
on 192^2) and 1038 s for the MAC run (54 ms per step; the sparse LU solves of the implicit
viscous step and the Poisson solve cost more than the FFTs). The slow tests took 97-539 s
(two periodic cycles) and 46-172 s (one coarse cycle on both back ends), also with the
machine shared, more or less heavily. The vorticity snapshots show the pair of vortices
shed at every reversal travelling down and to the left, as in the paper's Fig. 9B.

## Files

- `kinematics.py`: the stroke of eqs. (2)-(3) as a `StrokeKinematics` dataclass, with the
  rigid-body positions and velocities of the markers, all as functions of absolute time.
- `generate.py`: every parameter; writes the structure and input files.
- `hooks.py`: `update_targets` (the prescribed pose at the time the driver passes) and
  `on_output` (vorticity snapshots).
- `analyze.py`: coefficients, cycle means, the reference values and the comparison plot.
- `tests/examples/test_flapping_plate_re64.py`: smoke runs of both inputs, the committed
  files against `generate.py`, the kinematics against the paper's definitions, the hooks as
  functions of absolute time, the analysis on synthetic histories (with the force at its own
  time), and two slow runs (two cycles of `input2d`, 97-539 s; one coarse cycle on both
  back ends, 46-172 s).
