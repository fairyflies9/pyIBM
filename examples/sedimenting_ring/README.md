# Sedimenting elastic ring (MAC back end, closed no-slip box, mass points under gravity)

A closed ring of springs and torsional beams with a mass point on every node is released
at rest near the top of a box of still fluid with four no-slip walls. Gravity acts on the
mass points only (`advance_masses` in `pyibm/structures.py`: each ghost of mass m per unit
length obeys m dV/dt = -k (Y - X) + m g and pulls its fluid point through the tether), so
the ring is a heavy membrane around neutrally buoyant fluid. It falls, flattens into an
oblate shape with a flat bottom, drags the fluid it encloses down with it, and leaves a pair
of shear layers above. This is a demonstration: there is no published number to hit. What it
checks is internal consistency, with three numbers that have to come out right:

1. **Weight transfer.** The vertical force the fluid puts on the ring (`fy_body` in
   `history.csv`, which is minus the spread tether force) has to carry its weight
   M g = m 2 pi R g once gravity is switched on. The ghost update makes it
   fy_body = M (g r(t) - a_ghost) exactly, with r(t) the gravity ramp and a_ghost the
   acceleration of the ghost masses. The ring's own mass, M = 1.0 kg/m, is small next to the
   fluid it carries (rho pi R^2 = 20 kg/m inside it, plus the added mass outside), so a_ghost
   stays small and `fy_body/(M g)` follows the ramp: it is within 2 % of 1 from t = 0.2 s on,
   when the ring is at 11 % of its speed at t = 3 s. So the ratio checks that the weight
   reaches the fluid through the tethers; it does not say how close the ring is to terminal
   speed. For that we fit a straight line to the speed over the last 0.5 s and report its
   slope dU/dt.
2. **Area conservation.** The fluid is incompressible, so the polygon area through the ring
   must stay constant up to the leak of the discrete delta function.
3. **Settling speed** against a rigid circle: `input2d_rigid` drags the same circle, made
   of target points, down the same box at the speed the ring reached, and reports the drag
   the fluid puts on it next to the ring's weight.

## Parameters

| quantity | `input2d` (fine) | `input2d_coarse` | `input2d_rigid` |
|---|---|---|---|
| box | 0.5 m x 1.0 m, four no-slip walls at rest | same | same |
| cells, dx | 64 x 128, 1/128 m | 32 x 64, 1/64 m | 64 x 128 |
| fluid | rho = 1000 kg/m^2, mu = 1 N s/m^2 (nu = 1e-3 m^2/s) | same | same |
| ring | R = 0.08 m (10 cells), centre (0.25, 0.78) | same | same |
| points | 129, ds = dx/2 (rounded so the ring closes) | 64 | 129 |
| mass per unit length m | 2 kg/m, weight M g = 9.862 N | same | targets instead |
| tether k_mass | 5e4 (ghost gap m g/k = 4e-4 m = 0.05 dx at rest) | 5e4 | k_target = 1e7 |
| stretching EA = k_spring ds^2 | 750 N (k_spring = 4.9e7) | 750 N (1.2e7) | - |
| bending EI = k_beam ds^6 | 1e-3 N m^2 (k_beam = 2.9e11) | 1e-3 (4.3e9) | - |
| gravity | 9.81 m/s^2 down, ramped in over 0.2 s (`hooks.py`) | same | U_RIGID = 0.137 m/s, same ramp |
| scheme | mac, midpoint, peskin4, blend advection, implicit viscosity | same | same |
| dt, length | 2.5e-4 s, 3 s (12000 steps) | 5e-4 s, 3 s | 5e-4 s, 3 s |
| output | every 200 steps (0.05 s) | every 100 steps | every 100 steps |

EA and EI are the physical constants and the coefficients per unit length follow from ds
(`generate.py`; the derivation of EA = k ds^2 and EI = k ds^6 from pyibm's
energy-per-unit-length convention is in `examples/swimmer/README.md`). The beams' `curv`
is the circle's own cross product, so the rest shape is the initial circle. The spring
rest length is the chord. The ring and the enclosed fluid together weigh rho pi R^2 = 20
kg/m plus the ring's M = 1 kg/m, with a comparable added mass outside, and only M g drives
the fall, so the ring is slow to reach its terminal speed: at t = 3 s it is still speeding
up by 4.6 % per second.

Why the gravity ramp: switching the full weight on at t = 0 rings the ghost-tether mode
(sqrt(k_mass/m) = 158 rad/s, undamped in the symplectic Euler update) with an amplitude
of the order of the weight, which then shows up aliased in `fy_body` (it swung between 0.1
and 2 times M g in our first run). With the ramp `fy_body/(M g)` is smooth. The tether
number k dt^2/(rho dx) is 0.4 for the springs of the fine case and 0.2 for the coarse one.
The driver's rule of thumb checks the mass tethers (4e-4 fine, 8e-4 coarse) and the rigid
reference's targets (0.32), far under its limits of 7 and 10, so these inputs run without a
warning, and sqrt(k_mass/m) dt = 0.04 is far inside the ghost's own limit.

## How to run

```
python -m pyibm run examples/sedimenting_ring                        # fine, 3 minutes
python -m pyibm run examples/sedimenting_ring --input input2d_coarse   # 35 s
python -m pyibm run examples/sedimenting_ring --input input2d_rigid    # the rigid reference, 2 minutes
python examples/sedimenting_ring/analyze.py
```

`hooks.on_output` appends one row per output to `viz/<name>/ring.csv`: centroid and
centroid velocity (the mean of the interpolated point velocities), polygon area and
perimeter, the largest spring stretch, the largest ghost-to-point gap in cells, and the
height of the lowest point above the floor in cells (the driver raises if a marker comes
within 2 cells of a wall; `Tfinal` keeps it far above). `analyze.py` prints the speed of
the fall over the last 0.5 s and the slope of a straight-line fit to it, the Reynolds number
on the diameter, the drag coefficient the weight implies, `fy_body/(M g)`, the area and
perimeter drift, the stretch, and the rigid reference's drag when that run exists, and
saves `results.png` with the ring's shape at several times read from the `lag.NNNNN.vtk`
files and `fy_body/(M g)` next to the gravity ramp.

## What to expect (measured on this laptop)

| | fine (64 x 128) | coarse (32 x 64) |
|---|---|---|
| mean speed over the last 0.5 s | 0.135 m/s | 0.118 m/s |
| speed at t = 3 s | 0.137 m/s | 0.121 m/s |
| dU/dt over the last 0.5 s (straight-line fit) | 6.2e-3 m/s^2, 4.6 % of U per second | 9.1e-3 m/s^2, 7.7 % per second |
| (M + rho pi R^2) dU/dt / (M g) | 1.3 % | 2.0 % |
| Re on the diameter | 21.6 | 18.9 |
| `fy_body/(M g)` over the last 0.5 s | 0.9999 | 0.9989 |
| area drift, max over the run | 9.5e-4 | 1.4e-3 |
| perimeter drift | 7.5e-3 | 7.4e-3 |
| max spring stretch (one segment) | 3.4 % | 3.3 % |
| max ghost gap | 0.06 cells | 0.03 cells |
| lowest point at the end | 53 cells above the floor | 29 |
| wall time | 165 s (14 ms/step, machine shared) | 36 s (6.1 ms/step) |

The ring has not reached its terminal speed at t = 3 s. (M + rho pi R^2) dU/dt is a lower
bound on the share of the weight still going into acceleration, since the added mass of the
fluid outside is left out. `fy_body/(M g)` reads 0.9999 all the same, because it measures
the weight reaching the fluid (item 1): the acceleration of the ring's own mass is only
0.06 % of its weight. The lowest point is 53 cells above the floor at t = 3 s, so there is
room to run on for about 2 s before the ring comes within 20 cells of it.

Rigid reference: the circle dragged at U_RIGID = 0.137 m/s feels 8.18 N upward over the
last 0.5 s, 0.83 of the ring's weight (drag coefficient 5.5 on the diameter), with a
target error of 0.06 cells; wall time 99 s.

The drag coefficient the ring's weight implies, C_D = M g/(1/2 rho U^2 2R) = 6.7, is
about three times the unconfined rigid-cylinder value at this Reynolds number (about 2 at
Re = 20 on the standard drag curve). Two reasons, both expected: the box is only 6 radii
wide (32 % blockage, and wall effects on a cylinder's drag are strong at Re of order 10),
and the ring is still accelerating at t = 3 s (its speed rises by 4.6 % per second over the
last 0.5 s). The rigid circle in the same box at the same speed gives the first effect on
its own; the difference between it and the ring's weight is the deformation plus the
remaining acceleration.

The flattened shape and the pair of shear layers in `results.png` are the point of the
example. The bending rigidity hardly matters for the fall (EI from 1e-4 to 1e-2 changed
the settling speed by 3 %): the enclosed incompressible fluid and the stretching stiffness
set the shape, the beams only keep it smooth.

## Files

- `generate.py`: the circle, springs, beams, mass points, the rigid variant's targets, the
  three inputs; all constants live here.
- `hooks.py`: the gravity ramp (`update_masses`), the rigid reference's motion
  (`update_targets`, a no-op for the elastic ring) and `on_output` (`ring.csv`).
- `analyze.py`: the numbers above and `results.png`.
- `tests/examples/test_sedimenting_ring.py`: smoke runs of the coarse and rigid inputs,
  the committed files against `generate.py`, the ramp and the rigid motion, the analysis
  helpers, the weight transfer of the coarse ring while it is still slow, and (slow) the fine
  run to the end with the weight transfer, area, stretch, the speed and its slope checked.
