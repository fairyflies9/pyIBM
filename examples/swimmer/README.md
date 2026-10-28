# Self-propelled swimmer (travelling-wave curvature through `update_beams`)

A free filament that swims by itself in a doubly periodic box. The body is a line of
points joined by springs (stretching) and torsional beams (bending). Every step
`hooks.update_beams` rewrites the beams' preferred curvature (the `curv` column of the
`.beam` file) so that the body wants to be the Carling midline at that instant: a wave of
lateral deflection that grows from head to tail and runs from head to tail. The fluid
reacts to the bending and the body moves head first. Nothing else acts on it: no targets,
no mass points, no external force, and the spring and beam forces sum to zero, so the box
mean velocity stays zero on its own (`fx_body`, `fy_body` in `history.csv` are 1e-15).

This is a demonstration, not a reproduction of a published number. What it demonstrates:
driving a body through the beam `curv` column with a function of absolute time, measuring
the swimming speed from the centroid, and where the explicit coupling actually stops being
stable for stiff springs and beams. The kinematics come from a paper; the speed does not.

## Kinematics (Carling, Williams & Bowtell 1998, eq. 2)

Carling, J., Williams, T. L. & Bowtell, G. (1998). Self-propelled anguilliform swimming:
simultaneous solution of the two-dimensional Navier-Stokes equations and Newton's laws of
motion. J. Exp. Biol. 201, 3143-3166.

Body coordinate x from the head (x = 0) to the tail (x = L), body length L = 1 m:

    h(x, t) = r(t) A(x) sin(2 pi (x/lambda - t/T)),   A(x) = 0.125 L (x/L + 0.03125)/1.03125

- amplitude envelope A linear from 0.0038 L at the head to 0.125 L at the tail,
- wavelength lambda = L (one wave on the body), period T = 1 s, wave speed V = lambda/T = 1 m/s,
  travelling from head to tail (towards +x), so the body swims towards -x,
- r(t) = sin^2(pi t/(2 T)) for t < T, then 1: a smooth ramp over the first period so the
  start is not impulsive (our addition; the body is straight at t = 0).

The hook treats this graph as the preferred shape traced with arc length: node k at arc
length s_k = k ds along the graph, and the beam about it gets the cross product of two
segments of length ds that turn through the graph's curvature there,

    curv_k = -ds^2 sin(kappa(x(s_k), t) ds),   kappa = h_xx / (1 + h_x^2)^(3/2),

which is exactly what pyibm's torsional beam compares (X_r - X_q) x (X_q - X_p) with
(`beam_energy` and `beam_forces` in `pyibm/structures.py`, and the `.beam` file in
`docs/user-guide.md`; for straight segments of length ds the cross product is
ds^2 sin(theta_B - theta_A)). `tests/examples/test_swimmer.py` checks that the beams get
the cross products of the sampled midline to 1e-3. With the amplitude ramp the body is
straight at t = 0 and the beams are set from the ramped curvature, so no initial transient
needs to be built into the vertex file.

## Parameters

| quantity | `input2d` (fine) | `input2d_coarse` |
|---|---|---|
| box | 3 m x 1.5 m, doubly periodic | same |
| cells, dx | 192 x 96, 1/64 m | 96 x 48, 1/32 m |
| body | 129 points, ds = dx/2 = 1/128 m | 65 points, ds = 1/64 m |
| head at | x = 1.6 m, y = 0.75 m | same |
| fluid | rho = 1 kg/m^2, mu = 0.005 N s/m^2, Re = V L/nu = 200 | same |
| stretching EA = k_spring ds^2 | 250 N (k_spring = 4.1e6) | 250 N (k_spring = 1.0e6) |
| bending EI = k_beam ds^6 | 0.2 N m^2 (k_beam = 8.8e11) | 0.2 N m^2 (k_beam = 1.4e10) |
| scheme | midpoint, peskin4, delta coupling | same |
| dt, length | 1.5e-4 s, 4 periods (26667 steps) | 4e-4 s, 6 periods (15000 steps) |
| output | every 200 steps (0.03 s) | every 100 steps (0.04 s) |

EA and EI are the physical constants (`generate.py`); the per-unit-length coefficients pyibm
wants follow from ds. The conversions come from pyibm's convention that the element energies
k/2 (L - rest)^2 and k/2 (c - curv)^2 are energies per unit length (their gradients are
force densities, `docs/numerics.md`, conventions): the element's energy is ds times that, so
with the strain eps = (L - ds)/ds the springs give (k ds^2/2) eps^2 ds, i.e. EA = k ds^2,
and with c = kappa ds^3 the beams give (k ds^6/2) kappa^2 ds, i.e. EI = k ds^6. (We first
wrote EI = k ds^5 and got a body that did not bend at all; the force per unit length of a
straight body with preferred curvature kappa0 is ds D2(k curv) = k ds^6 kappa0'', the
continuum EI kappa0'', so ds^6 is right.) In the units of this case, EI/(rho V^2 L^3) = 0.2
and EA/(rho V^2 L) = 250.

## How to run

```
python -m pyibm run examples/swimmer                       # fine, about 2 minutes
python -m pyibm run examples/swimmer --input input2d_coarse   # coarse, 30 s
python examples/swimmer/analyze.py
```

`hooks.on_output` appends one row per output to `viz/<name>/swimmer.csv`: centroid
(unwrapped around the head so a body crossing the box edge is whole), centroid velocity
(the mean of the interpolated point velocities, which is exactly what moves the points),
heading of the head-to-tail chord, largest spring stretch, body length, and the RMS of the
realised minus prescribed curvature next to the RMS prescribed curvature. `analyze.py`
prints the period-by-period mean swimming speed U, U/V, the curvature error, the stretch
and the length, and saves `results.png`. `python generate.py` rewrites the structure and
input files (the generated files are committed).

The driver's tether rule of thumb checks only target points and mass tethers, so these
inputs run without a warning. The springs sit at k dt^2/(rho dx) = 5.2 (coarse) and 5.9
(fine), which would be close to the limit for a tethered point; for these springs and beams
the measured limit is 20 to 80 (next section), and the runs are stable. The smoke test
asserts that no tether warning is issued.

## What to expect (measured on this laptop)

Swimming speed, period-averaged, with V = 1 m/s:

| | period 1 (ramp) | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| fine, U/V | 0.06 | 0.096 | 0.090 | 0.091 | | |
| coarse, U/V | 0.04 | 0.03 | 0.02 | 0.03 | 0.03 | 0.04 |

Fine run: curvature error / prescribed 0.047 after the ramp, max stretch 0.9 %, body
length within 3e-4 of 1 m, heading swinging by +-9 degrees, wall time 103 s (3.9 ms/step).
Coarse run: curvature error 0.12, max stretch 1.7 %, length within 6e-4, heading drifting
to -16 degrees by the sixth period, wall time 31 s (2.0 ms/step).

So the body follows the prescribed curvature to within about 5 % RMS on the fine grid and
12 % on the coarse one, stays inextensible, and swims head first at roughly a tenth of the
wave speed on the fine grid and a few hundredths on the coarse one. That is a slow
swimmer: a real eel or lamprey at this kinematics swims at U/V of 0.5 to 0.8. Two things
make this one slow, and both are worth knowing about before using this setup for anything
quantitative:

1. **Resolution.** The 4-point kernel gives a filament an effective thickness of about two
   cells on each side. At dx = 1/32 that is a body 0.125 L thick (as thick as the tail
   amplitude), at dx = 1/64 half that; halving dx about tripled the speed (0.03 to 0.09,
   the mean over t > 1 s). A thin undulating sheet at this Reynolds number is largely a
   viscous swimmer and its thrust is sensitive to that thickness. We did not go finer
   because the beams set the time step (next section) and the cost goes as dx^-4.
2. **Reynolds number.** On the fine grid with mu = 0.002 (Re 500) the speed over the
   third period was 0.09 L/T and with mu = 0.001 (Re 1000, under-resolved, grid Reynolds
   number 15) 0.15 L/T, with a vortex street forming behind the tail; at Re 200 the wake is
   just a pair of thin shear layers that diffuse away.

Also visible in `results.png`: the heading oscillates by a few degrees each period (recoil:
the body is massless and nothing holds it) and drifts slowly, and the instantaneous
centroid speed swings between about 0 and 0.17 m/s within a period.

## Stability limits we measured (coarse grid, dx = 1/32, ds = dx/2, midpoint, peskin4)

Runs of 0.6 to 5 periods, "stable" meaning no growth in 1500 to 12500 steps, "blew up"
meaning non-finite positions within 50 steps:

| element | k dt^2/(rho dx) | result |
|---|---|---|
| springs, k = 1e6, dt = 8e-4 | 20 | stable |
| springs, k = 4e6, dt = 4e-4 | 20 | stable |
| springs, k = 4e6, dt = 8e-4 | 82 | blew up |
| beams (EI = 0.2, k = 1.4e10), dt = 8e-4, with 16 k ds^2 for k | 1100 | stable |
| beams (EI = 1.0, k = 6.9e10), dt = 4e-4 | 1400 | stable |
| beams (EI = 0.5, k = 3.4e10), dt = 8e-4 | 2700 | blew up |
| beams (EI = 1.0, k = 6.9e10), dt = 8e-4 | 5500 | blew up |

So for the springs the limit sits between 20 and 80 in the tether number k dt^2/(rho dx),
above the 10 at which the driver starts to warn for target points (it does not check springs
and beams at all). For the beams the natural stiffness of the worst mode of a
straight chain is 16 k ds^2 (the second-difference operator squared has eigenvalue 16), and
with that the limit is between 1400 and 2700, far above any tether limit.
The reason both are so forgiving is the kernel: the alternating mode of a chain with
ds = dx/2 has wavelength dx and spreads to almost nothing on the grid, so the stiffest modes
never reach the fluid and never get an explicit kick back. The time steps in the inputs are
set at about half the beam limit (fine: 16 k ds^2 dt^2/(rho dx) = 1240) and were stable
for the whole runs. The fine input at dt = 2.5e-4 (number 3400) blew up within a period.

The springs could be made 4 times stiffer at these time steps; EA = 250 N already holds
the stretch to 1 % (coarse) and under 1 % (fine). The beams are the binding constraint on
dt for a body this stiff: dt_max scales as sqrt(rho dx ds^4 / EI) ~ dx^2.5 / sqrt(EI), so
halving dx costs 5.7x more steps on top of 4x more cells.

## Files

- `generate.py`: body points, springs, beams, the two inputs; EA, EI, Re and the
  kinematic constants live here.
- `hooks.py`: `update_beams` (the Carling curvature as a function of absolute time) and
  `on_output` (the body diagnostics in `swimmer.csv`).
- `analyze.py`: swimming speed per period, curvature error, `results.png`.
- `tests/examples/test_swimmer.py`: smoke run of the coarse input (no tether warning), the
  committed files against `generate.py`, the kinematics and the beam `curv` construction,
  and (slow) the coarse run to the end with its speed, curvature error and stretch checked.
