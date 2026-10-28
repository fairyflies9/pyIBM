# Jellyfish bell driven at its natural frequency (Hoover & Miller 2015)

A two-dimensional hemielliptical bell that resists stretching and bending, squeezed
periodically at its margin by a band of muscle springs, swims up through a viscous fluid.
This is the model of Hoover & Miller, *A numerical study of the benefits of driving jellyfish
bells at their natural frequency*, J. Theor. Biol. 374 (2015) 13-25: a free-vibration study
gives the bell's natural frequency, f_ref = 0.86 s^-1, and driving the bell near that
frequency gives the largest bell oscillation and the fastest swimming. The pyIBM case is a
free bell of springs and torsional beams in a periodic box, driven through `update_springs`,
with the forward speed taken from the centroid of the bell.

## Physics and parameters

Geometry (eq. 1 and 10, Table 1, L = 1): the preferred shape is the arc of the ellipse
with half width a = 0.5 L and height b = 0.75 L above the centre, cut off d = 0.25 L below
it (theta from -19.5 to 199.5 degrees), so the bell is one diameter wide and one high, open
at the bottom. The points are spaced equally in arclength at ds = h/2 (Table 2). Springs
between neighbours (stiffness ks/ds^2 per unit length, rest length the preferred segment)
and torsional beams about every interior point (stiffness kb/ds^6, preferred cross product
of the hemiellipse) discretise the stretching and bending energies of eq. 6; the discrete
bending energy of the squeezed bell is within 5 % of the continuum integral.

Drive (eq. 2 and 7, Fig. 5): the paper applies F_D(t) = F_Mag sin(2 pi f t) per unit length
on the lower quarter of the bell, along the line joining each point to its mirror image.
Here every point of the lower quarter is joined to its mirror image by a muscle spring of
stiffness k_m = F_Mag/L0 (L0 the preferred width of the band at that point) whose rest
length `hooks.update_springs` sets to L0 (1 - sin(2 pi f t)), a function of absolute time.
Its force k_m (L - rest) is the paper's F_Mag sin(2 pi f t) plus k_m (L - L0), the strain of
the band; the first half of the cycle (sin > 0) is the contraction, the second the expansion.
The frequency is read from the structure name (`bell_f0.86`), and `hooks.on_output` writes
`bell.csv`: the centroid of the bell (unwrapped in y, so the swim distance keeps growing),
the diameter at the widest cross-section (Fig. 2) and the force in the first muscle spring.

Fluid: rho = 1, mu = rho U L/Re with U = f_ref L and Re = 172 (Table 1), so mu = 0.005.

**Stiffness and force scale.** Table 1 gives ks = 1.0563, kb = 0.0528 and F_Mag = 0.0406 as
multiples of rho U^2 L, rho U^2 L^3 and rho U^2 with U = 0.86 (eq. 16-18). pyIBM's spring
and beam constants and forces are per unit length, and the mapping from these values to
them is our own. Used directly in that convention, the values give a bell that in the
free-vibration study creeps from a diameter of 0.90 to 0.92 in 3 s instead of swinging to
1.07 in 0.5 s (Fig. 2), and a drive that moves the margin by a hundredth of a diameter
instead of a quarter (Fig. 6). So the example keeps the paper's ratio ks : kb and calibrates
two factors against the paper's measurements (`generate.py`, `STIFFNESS_SCALE` and
`FORCE_SCALE`):

- the stiffnesses times 430, so that the free vibration of the squeezed bell has the
  recorded frequency 0.86 s^-1, which is how the paper defines its reference bell (measured
  as the inverse of the time between the first two maxima of the diameter; it goes as the
  square root of the scale, 0.66 s^-1 at 256 and 0.94 at 512);
- the force times 118, so that the quasi-static response, the peak-to-peak swing of the
  diameter when driven slowly at f = 0.25 s^-1, is the 0.186 of Fig. 6. The response is not
  linear in the force: in cycle 2 the swing per unit of force scale falls from 1.58e-3 at
  118 to 1.56e-3 at 150 and 1.26e-3 at 430, so a linear scaling from one run misses. So we
  ran the f = 0.25 case itself (`python generate.py --calibration <dir> [force_scale]`, then
  `python -m pyibm run <dir>` and `python analyze.py --calibration <dir>`). The swing in
  cycles 2 and 3 (cycle 1 starts from rest and swings further) is:

  | force scale | 115 | 118 | 120 | 122 | 150 | 430 |
  |---|---|---|---|---|---|---|
  | swing of the diameter | 0.182-0.183 | 0.187-0.188 | 0.190-0.191 | 0.192-0.194 | 0.232-0.233 | 0.543-0.544 |

The resonant amplitude (Fig. 6) and the swimming speed (Fig. 7) are then predictions of the
model, not fits.

| quantity | value |
|---|---|
| box | 4 L x 12 L, doubly periodic, 64 x 192 cells, h = L/16 (the paper: h = L/64 on the finest level) |
| bell | 81 points, ds = L/32, centre at (2 L, 2 L), swims in +y |
| fluid | rho = 1, mu = 0.005 (Re = 172 at f_ref) |
| stiffness | ks = 430 x 0.781 = 336, kb = 430 x 0.0391 = 16.8 (springs 3.4e5, beams 1.8e10 per unit length) |
| drive | F_Mag = 118 x 0.030 = 3.5 per unit length on the lower quarter (9 muscle springs) |
| time step | 2e-4 (the free vibration is the same at 2e-4, 1.5e-4, 1e-4 and 5e-5); the paper: 1e-5 |
| length | 10 cycles at f = 0.86; 5 cycles at the other frequencies; 3 s free vibration |
| output | 40 history and `bell.csv` rows per cycle to `viz/<variant>/` |

| input | what | steps |
|---|---|---|
| `input2d` | f = 0.86 s^-1 (the resonant case), 10 cycles | 58140 |
| `input2d_f0.5`, `_f0.75`, `_f1`, `_f1.25` | the other points of Fig. 6-7, 5 cycles (the P = 5 curve) | 50000, 33335, 25000, 20000 |
| `input2d_freevib` | free vibration from the bell squeezed to 90 % of its width (section 3.1, Fig. 2) | 15000 |
| `input2d_fine` | f = 0.86 at h = L/32, ds = L/64, dt = 1e-4, 10 cycles; two cycles run, see below | 116280 |

The squeezed bell of the free-vibration study keeps the number of points and their spacing
(the paper: "initialized with uniform spacing ds so that the spring forces would be minimized
initially"), so its arc runs a little further down the narrower ellipse, as in Fig. 1.

## How to run

```
python -m pyibm run examples/jellyfish_bell
python -m pyibm run examples/jellyfish_bell --input input2d_freevib
python -m pyibm run examples/jellyfish_bell --input input2d_f0.5      (and _f0.75, _f1, _f1.25)
python examples/jellyfish_bell/analyze.py
```

`analyze.py` reads `viz/*/bell.csv` and prints, for every driven run, the forward speed of
each pulse cycle (the rise of the centroid over the cycle times f, in bell diameters per
second, the V_forward of Fig. 7) and the amplitude of the diameter oscillation in the cycle
(max - min of the diameter, the A_max of Fig. 6), next to the paper's values for the
F_Mag^ref panels, and the rise of the centroid in each contraction (from each trough of the
centroid height to the next peak); for the free vibration it prints the frequency and the
first swing of the diameter next to Fig. 2. It saves `results.png`.

## What to expect

### Free vibration

`input2d_freevib`: from a diameter of 0.90 the bell swings to 1.072 at t = 0.62 s, back to
0.945 at 1.2 s and up to 1.039 at 1.81 s; the frequency from the first two maxima is
0.84 s^-1. The paper's Fig. 2 (k_b^ref): 1.07 at 0.55 s, 0.95 at 1.15 s, 1.03 at 1.75 s, and
the recorded frequency 0.86 s^-1. The frequency is what the stiffness scale was calibrated
to (it is 0.94 s^-1 at scale 512 and 0.66 at 256); the overshoot to 1.07 and the decaying
second swing are not fitted and match the figure.

### Driven bell

Last cycle of each run, against the F_Mag^ref panels of Fig. 6 and Fig. 7. The paper's
values are read from the plotted lines in the vector data of the PDF, to about 0.01:

| f [s^-1] | cycles | amplitude of the diameter | paper, P = 25 | forward speed [L/s] | paper, P = 5 / P = 25 |
|---|---|---|---|---|---|
| 0.5 | 5 | 0.25 | 0.24 | -0.002 | 0.12 / 0.28 |
| 0.75 | 5 | 0.39 | 0.29 | -0.005 | 0.33 / 0.72 |
| 0.86 | 10 | 0.42 (0.38 to 0.40 in cycles 2-9) | 0.285 | -0.083 (net -0.29 L in 10 cycles) | 0.31 / 0.80 |
| 1.0 | 5 | 0.31 | 0.25 | -0.001 | 0.19 / 0.78 |
| 1.25 | 5 | 0.17 | 0.14 | +0.002 | 0.06 / 0.08 |

The amplitude of the oscillation is close to the paper's at f = 0.5 (0.25 against 0.24),
1.2 to 1.5 times the paper's from f = 0.75 to 1.25, and it peaks between f = 0.75 and 0.86
as the paper's does. The force was calibrated to the quasi-static swing at f = 0.25, so the
resonant gain above it is the model's own: the largest amplitude over the quasi-static one is
0.39-0.42/0.187 = 2.1-2.2 here, against 0.289/0.186 = 1.55 in the paper. The bell does not
swim. After the first cycle it moves up by 0.07 to 0.13 L during each contraction at
f = 0.86 (0.04 to 0.13 L over the sweep) and back down during the expansion, and the net is
a slow backward drift, 0.025 L/s on average over the ten cycles at f = 0.86 and 0.08 L/s in
the tenth, where the paper's bell swims forward at 0.31 L/s by the fifth cycle and 0.8 L/s
at steady state.

The reason is resolution. At h = L/16 the four-point kernel smears the bell over a quarter
of its diameter, and the oscillatory boundary layer sqrt(nu/(2 pi f)) = 0.03 L is half a
cell: the starting vortex that the contraction should shed from the margin, and the stopping
vortex that the expansion should leave inside the bell (Fig. 9 of the paper), cannot form,
so the motion is nearly reversible, which is what the paper reports for its low Reynolds
number runs ("moves forward during bell contraction and then moves almost back to its
starting position during bell expansion", section 4). The paper resolved the bell with
h = L/64 and ds = L/128. Two cycles of `input2d_fine` (h = L/32, ds = L/64; 7.7 minutes
with the machine shared) do not settle the question: the per-cycle speeds are
-0.027 and +0.028 L/s and the bell is 0.002 L higher after two cycles, where at h = L/16 they
are -0.042 and +0.026 L/s and the bell is 0.018 L lower; the amplitudes are 0.29 and 0.36
against 0.29 and 0.39. Whether the fine bell reaches the paper's 0.31 L/s by the fifth cycle
takes the full ten-cycle run, about 40 minutes.

So this example reproduces the free vibration of the bell and the shape of the resonance
curve of the oscillation amplitude, and does not reproduce the forward swimming at the
resolution a Python code can afford in minutes. Two scale factors (stiffness x 430, force
x 118) stand between the paper's Table 1 and this input, as explained above; the swimming
would need the finer grid in any case.

Run time on this laptop (64 x 192 grid, 81 points, midpoint scheme): 5.4 ms per step with
the machine shared, so 10 cycles at f = 0.86 s^-1 (58140 steps) took 5.2 minutes and the
5-cycle sweep runs 2.2 to 4.7 minutes each; the six driven runs above were made in parallel.
The fine input runs at 20 ms per step (116280 steps for its ten cycles).


## Files

- `generate.py`: the geometry, the elements, the calibration factors, the seven inputs and
  the calibration case (`--calibration`); the structure files
  (`bell_f<f>.vertex/.spring/.beam`, `bell_free.*`, `bell_fine_f0.86.*`) are committed.
- `hooks.py`: `update_springs` (the muscle law) and `on_output` (`bell.csv`).
- `analyze.py`: the per-cycle speed and amplitude, the rise in each contraction, the
  free-vibration frequency, the swing of a calibration run, `results.png`.
- `tests/examples/test_jellyfish_bell.py`: smoke runs of the driven, the free-vibration and
  the calibration inputs, the committed files against `generate.py`, the geometry, the
  muscle law, the refusal to write the calibration case into the example folder, the rise
  per contraction on a synthetic record, and three slow runs: the free vibration against
  Fig. 2, two cycles of the calibration case against the quasi-static swing of Fig. 6, and
  three cycles at f = 0.86 in which the bell oscillates with the amplitude above and stays
  put.
