# Rubber band: an ellipse of springs and beams relaxing to a circle

The classic immersed-boundary demonstration. A closed elastic band is started as an
ellipse in a box of fluid at rest. Its springs are stretched (their rest length is half the
initial spacing), so the band wants to shrink, but it is filled with incompressible fluid
that cannot get out. The only way to shorten its perimeter is to become the circle of the
same area, which it does, overshoots a little, and rings down. The quantity to watch is the
enclosed area: an ideal method keeps it exactly constant. An immersed-boundary method leaks
when the velocity it moves the points with is not exactly divergence-free. The same
structure is run with both couplings pyIBM has, the discrete delta function and the blob
projection of Cortez & Minion (J. Comput. Phys. 161 (2000) 428-453), and `analyze.py`
reports the area against time for both. This is a demonstration, not a comparison with a
published number.

## Physics and parameters

| quantity | value |
|---|---|
| box | 1 m x 1 m, doubly periodic, 64 x 64 cells, dx = 0.015625 m |
| fluid | rho = 1 kg/m^2, mu = 0.01 N s/m^2 |
| band | ellipse with semi-axes 0.2 m (x) and 0.1 m (y) about (0.5, 0.5); 128 points at equal arc length, ds = 0.00757 m = 0.484 dx |
| springs | between neighbours, k = 400 N/m per m, rest length 0.5 x the initial spacing (initial tension k ds (L - rest) = 0.011 N) |
| beams | one torsional beam about every point, k = 5e5 N/m per m, preferred curvature 0 (bending stiffness EI = k ds^6 = 1e-7 N m^2; a token stiffness, see below) |
| time | dt = 0.002 s, 1000 steps to t = 2 s, output every 20 steps |
| scheme | periodic FFT back end, midpoint scheme (second order) |
| final circle | radius sqrt(0.2 x 0.1) = 0.1414 m, perimeter 0.8886 m (initial 0.9687 m) |

Two inputs, same structure, both written by `generate.py`:

| input | coupling | output |
|---|---|---|
| `input2d` | delta function, `peskin4` kernel | `viz/delta/` |
| `input2d_blob` | blob projection, `gaussian4` blob, delta = 3 dx, marker velocity from the interpolated stream function (`blob_interpolation = "streamfunction"`, the default) | `viz/blob/` |

The band is 0.3 m from the box edges, so the periodic images do not matter much, and no
net force acts on it (springs and beams are internal), so the box-mean velocity stays at
zero and `remove_mean_force` is not needed. The Reynolds number is small, about 2 with the
peak speed 0.076 m/s and the band size 0.3 m, which is why the ringing is damped within a
couple of oscillations.

The beams are there to show the element in use, not because the band needs them. At t = 0
their largest force is 2.9e-3 N/m against 0.23 N/m for the springs, and in runs without
them the points end up within 1.4e-5 m (0.001 dx) at t = 2 s of where they are with them,
for both couplings. The stretched springs keep the band smooth on their own.

`hooks.py` has one function, `on_output`, which writes `area.csv` next to `history.csv`:
the enclosed area (shoelace formula on the polygon through the points, unwrapped with the
grid's minimum image), the perimeter and the shape ratio rmax/rmin of the distances from
the centroid (2 for the initial ellipse, 1 for a circle).

## How to run

```
python -m pyibm run examples/rubberband
python -m pyibm run examples/rubberband --input input2d_blob
python examples/rubberband/analyze.py
```

`analyze.py` prints the relative area change, the perimeter and the shape ratio against
time for every variant that has been run, and saves `results.png` (area change, shape
ratio, perimeter, and the band at t = 0, 0.5, 1 and 2 s from the `lag.NNNNN.vtk` files).
`python generate.py` rewrites the structure and input files (deterministic; the generated
files are committed, and the tests check that they match it byte for byte).

## What to expect

Measured on this machine (1000 steps, 51 outputs):

| input | area/area(0) - 1 at t = 1 s | at t = 2 s | largest over the run | rmax/rmin at t = 1 s | at t = 2 s |
|---|---|---|---|---|---|
| `input2d` (delta) | +3.6e-5 | -5.7e-4 | 5.7e-4 | 1.011 | 1.111 |
| `input2d_blob` (blob) | +3.1e-5 | +2.8e-5 | 3.2e-5 | 1.009 | 1.111 |

Both couplings keep the area to better than 0.1 % over the two seconds, one part in a
thousand of an area of 0.0628 m^2 covered by about 260 cells. They do it in different ways.

The delta coupling swings between -1.5e-4 (t = 0.2 s) and +1.3e-4 (t = 0.76 s) while the
band moves fast, then loses area slowly and steadily once it has rounded up, to -5.7e-4 at
t = 2 s. Halving the time step (dt = 0.001 s) changes these numbers by less than 0.1 %, so
this is not time-stepping error. It is the leak of the delta function's interpolated
velocity, which is divergence-free only in the discrete sense on the grid: in a scratch run
where we gave the delta coupling the blob module's stream-function interpolation (the
package has no input key for that), the change at t = 2 s was -6e-6 and never more than
1.8e-5.

The blob run moves its points with a velocity that has exactly zero divergence, so it does
not leak. What it shows, at most 3.2e-5, is about the accuracy of the area measure itself:
the polygon through 128 points is not the curve, and the smooth (trigonometric) curve through
the same points gives a change of -4.7e-5 at t = 2 s instead of +2.8e-5. Both numbers are the
same at dt = 0.001 s. With the bicubic interpolation of u and v instead
(`blob_interpolation = "bicubic"` in the Solver section), whose field is divergence-free only
to O(h^2), the blob run gains +6.6e-4 by t = 1 s and +7.5e-4 by t = 2 s, a little more than
the delta run loses. And the delta run with the stream-function interpolation does as well
as the blob run: -6e-6 by the polygon and -8.2e-5 by the smooth curve at t = 2 s, against
+2.8e-5 and -4.7e-5. So on this case the difference between the two runs comes from the
marker interpolation, and at this resolution the area measures cannot tell the two couplings
apart once they share it.
`examples/blob_ellipse` measures the same effect on the test problem of Cortez & Minion,
where, with the stream-function interpolation for both, the blob still holds the area 3 to
11 times better than the delta function (largest error over the run at 128^2, depending on
the area measure).

The shape ratio drops from 2 to 1.01 at t = 1.0 s, when the band is within 1 % of a circle,
overshoots to 1.14 at t = 1.64 s with the long axis now along y, and is on its way back at
t = 2 s. The perimeter goes from 0.969 m to within 0.01 % of the final circle's 0.8886 m by
t = 1 s. The net force on the band is zero to round-off (1e-17 N) in both runs, as it must
be for an isolated elastic structure.

Run time: 2.3 s for `input2d` and 5.4 s for `input2d_blob` (1000 steps on a 64 x 64
grid, laptop with nothing else running; the blob run spends its extra time evaluating the
Gaussian blob's curl on a 21 x 21 node patch around each of the 128 points).

## Files

- `generate.py`: writes `band.vertex`, `band.spring`, `band.beam` and the two input files.
- `hooks.py`: `on_output` writing `area.csv`.
- `analyze.py`: the tables and `results.png`.
- `tests/examples/test_rubberband.py`: smoke runs of both inputs, the committed files
  against `generate.py`, the polygon area of the initial ellipse, `area.csv` with no other
  output switched on, `analyze.py` on short runs, and a slow test that runs both inputs to
  the end and checks the area and the final shape.
- `docs/examples.md` summarises this example with the others.
