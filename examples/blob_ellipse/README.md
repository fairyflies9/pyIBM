# The perturbed ellipse of Cortez & Minion 2000: blob projection against the delta function

The one dynamic test problem of R. Cortez and M. Minion, "The blob projection method for
immersed boundary problems", J. Comput. Phys. 161 (2000) 428-453, section 4.1 (pp. 436-440).
We run it with pyIBM's blob projection coupling and with the discrete delta function, both on
the periodic FFT back end. A closed membrane under constant tension, started as a dented
ellipse in fluid at rest, pulls itself toward a circle and rings. The membrane encloses
incompressible fluid, so its area should stay constant. The relative area error against time
is the paper's measure of how well a coupling conserves volume (their Figures 4-8), and it is
what `analyze.py` reports.

There are two inputs, with the same structure:

| input | coupling | marker velocity | output |
|---|---|---|---|
| `input2d` | blob projection, `gaussian4`, delta = 4 h | curl of the interpolated stream function (`blob_interpolation = "streamfunction"`, pyIBM's default) | `viz/blob/` |
| `input2d_delta` | delta function, `peskin4` | the delta function | `viz/delta/` |

## Physics and parameters (the paper's, section 4.1)

| quantity | value |
|---|---|
| box | unit square, doubly periodic, 128 x 128 cells, h = 1/128 |
| fluid | rho = 1, nu = 0.005 (mu = 0.005), at rest at t = 0 |
| membrane | z(theta) = (1/2, 1/2) + r (cos(theta - theta0), sin(theta - theta0)), r^2 = a^2 cos^2 theta + b^2 sin^2 theta + eps(-(4/3) exp(-3(theta - pi)^2) - exp(-5(theta - theta1)^2) + exp(-8(theta - theta2)^2)) with a = 0.2, b = 0.25, eps = 0.012, theta0 = pi/3, theta1 = 4 pi/5, theta2 = 2 pi/3 |
| points | 400 at equal arc length, ds = 0.003575 = 0.458 h (below the h/2 the paper quotes as the leak-free rule for the IB method) |
| force | f = sigma kappa n with sigma = 1/5 (the force per unit length of a membrane under the constant tension sigma) |
| time | dt = 0.1 h = 7.8125e-4, 1280 steps to t = 1, output every 40 steps (0.03125) |
| blob | `gaussian4` (the paper's fourth-order cutoff, eq. 15), delta = 4 h (the paper's main choice) |
| delta | `peskin4` kernel (the paper used the cosine kernel, eq. 8; it makes no difference here, see below) |

The membrane is a closed chain of zero-rest-length Hookean springs, the way L. Fovargue
discretised the same membrane in her 2012 PhD thesis (UNC Chapel Hill). The force per unit
length pyIBM computes for a spring is k (L - rest) along the link, and the coupling multiplies
it by ds, so a link carries the tension k ds L. `hooks.py` resets k = sigma/(ds L) at every
force evaluation. Then every link carries exactly the tension sigma, and the force on point k
is sigma (tau_{k+1/2} - tau_{k-1/2}), the integral of sigma kappa n over the point's segment.
The committed `ellipse.spring` holds the stiffness at the initial spacing, sigma/(ds L0) =
1.56e4, so the structure also runs without the hook (the tension then drifts with the
spacing). The paper evaluates kappa by fitting a sixth-order polynomial through seven points.
The chain gives the same force to O(ds^2), which `tests/examples/test_blob_ellipse.py`
checks against the exact curvature of the curve. Note that the curve is not convex: the eps
term dents it near theta = pi, where kappa changes sign.

`results.png` draws the membrane turned by half a turn about the centre of the box, the way
round it sits in the paper's Fig. 3 (dent at the lower right). The periodic box and its grid
are symmetric under that turn, so the runs do not depend on it: started from the turned
curve, the stream-function run is the turned run to 1e-14 in the positions after all 1280
steps, with the same area error (the tests check the symmetry over the first 4 steps).
Turned that way, our membrane at t = 0.3125 lies within 0.6 h of the curve in the paper's
right frame on average, and within 1.8 h everywhere (we traced the printed figure).

The driver's tether rule of thumb checks target points and mass tethers only, so these
springs raise no warning; the tests check that the runs raise none.

`hooks.py` also writes `area.csv` next to `history.csv` at every output: the enclosed area
(shoelace formula on the polygon through the points), its error relative to the initial
area, and the perimeter.

## How to run

```
python -m pyibm run examples/blob_ellipse                         # viz/blob
python -m pyibm run examples/blob_ellipse --input input2d_delta   # viz/delta
python examples/blob_ellipse/analyze.py
```

`analyze.py` prints the relative area error at t = 0.25, 0.5, 0.75 and 1 for each run that
has been done, the paper's values and the delta-over-blob ratio. It saves `results.png`:
both runs on one axis, the blob run on its own scale with the paper's +-1e-5 band, the
membrane at t = 0 and at t = 0.3125 (the times of the two frames of the paper's Fig. 3,
turned by half a turn as that figure draws it, for comparison by eye), and the perimeter.
`python generate.py` rewrites the structure and input files (deterministic; committed, and
the tests check that the committed files match it byte for byte).

## What to expect

Relative area error (area/area(0) - 1, polygon through the points) measured on this machine:

| run | t = 0.25 | t = 0.5 | t = 0.75 | t = 1 | largest over the run |
|---|---|---|---|---|---|
| `input2d`, blob, stream-function interpolation | +1.0e-6 | -4.1e-6 | +2.6e-6 | +2.5e-6 | 5.7e-6 |
| `input2d_delta`, delta function, peskin4 | -1.7e-3 | -3.4e-3 | -5.2e-3 | -6.9e-3 | 6.9e-3 |

The paper's values at t = 1 for the same grid, points, dt and nu, read off its figures
(about +-10 % of the printed scale):

| paper | t = 1 |
|---|---|
| blob projection, second-order version, fourth-order blob, delta = 4h (Fig. 7 top, Fig. 8) | within +-1e-5, mostly +-0.5e-5 |
| blob projection, delta = 2h (Fig. 8) | within +-1.2e-5 |
| immersed boundary (second order), decoupled centred projection (Fig. 4) | -1.4e-2 |
| immersed boundary, spectral projection (Fig. 4) | -0.8e-3 |
| immersed boundary, Peskin-Printz projection (Fig. 7 top, the paper's best IB run) | -1.6e-4 |

**The blob run reproduces the paper.** With the stream-function interpolation the
area error wanders between -5.7e-6 (t = 0.44) and +2.9e-6 and ends at +2.5e-6, inside the
paper's +-1e-5 band and with the same noise-about-zero look as its Figs. 7 and 8. Most of
what is left is how well the polygon measures the area: the smooth (trigonometric) curve
through the same 400 points changes its area by at most 1.2e-6 over the run.

**The delta function** leaks about half as much as the paper's IB run with the decoupled
centred projection, which is the projection the periodic FFT back end uses (its divergence
is the wide centred difference D0; Peskin & McQueen 1996, section 14). The leak grows
linearly in time, as in the paper, and halves when grid, points and dt are refined by two
(-3.5e-3 at t = 1 on 256 x 256 with 800 points; the paper's Fig. 5: -7.4e-3, "linear
dependence on the numerical parameters"). With the cosine kernel of the paper instead of
`peskin4` the delta run gives -6.9e-3 as well.

**Blob against delta.** How much better the blob is depends on the marker velocity the two
couplings use. At 128^2, with the polygon area:

| blob run | delta run | delta error / blob error at t = 1 | the same for the largest error over the run |
|---|---|---|---|
| stream function (`input2d`) | delta function (`input2d_delta`) | 2800 | 1200 |
| stream function | stream function (scratch) | 7 | 3 |

The package offers the stream-function interpolation only for the blob coupling, so for the
last row we patched the delta run's interpolation in a scratch script, with the spreading left
as the delta function. Delta spreading with the stream-function interpolant gives -1.8e-5,
against +2.5e-6 for the blob. So the large factors in the first row are mostly the
interpolation; with the same divergence-free marker velocity the blob spreading still gains a
factor of 3 to 45. Comparing the largest error over the run, the blob is 3 times better by the
polygon area (5.7e-6 against 1.8e-5) and 11 times better by the area of the smooth curve
through the points (1.2e-6 against 1.3e-5). At 256^2 with 800 points the same comparison gives
4 times by the polygon (1.6e-6 against 6.2e-6) and 45 times by the smooth curve (1.4e-7
against 6.3e-6). Comparing at t = 1 alone gives bigger factors, but they are not a stable
measure: the blob's error is noise about zero, and at t = 1 it happens to be small. The smooth
curve is the better area measure here (most of the blob's polygon error is the polygon's own,
see above), and by it the blob is an order of magnitude better or more, as in the paper's
Fig. 7, where the blob is "at least 10 times better" than its best IB run.
The two errors also look different. The blob's wanders about zero; the delta run's is a
steady loss even though its marker velocity has no divergence. That loss is the same at
dt = 0.05 h (-1.8e-5), so it is not the time stepping. Our reading is that the delta
function's velocity varies more at the grid scale, so the curve bends between neighbouring
points in ways that neither the polygon nor the smooth curve through the points can see.

**Checks**, all at 128^2 unless stated:

- Time step: at dt = 0.05 h the stream-function blob run gives +2.5e-6 (largest 5.6e-6), the
  same as at dt = 0.1 h.
- Blob width: at delta = 2 h the stream-function run gives +4.1e-6 (largest 5.2e-6), inside
  the paper's +-1.2e-5 for 2h (Fig. 8).
- Refinement, 256 x 256 with 800 points and dt = 0.1 h: stream function +7.0e-7 (largest
  1.6e-6; the paper's Fig. 7 bottom: within about 2e-6), delta function -3.5e-3.

Other numbers from the runs: the peak speed is 0.46 (CFL 0.06) in both. The perimeter
drops from 1.430 to 1.386 by t = 0.25 and then rings with a period of about 0.45 (minima at
t = 0.25 and 0.72). The net force on the membrane is zero to round-off (1e-16).

Run time on this laptop with nothing else running: 33 s for `input2d` (1280 steps, 400
points; the Gaussian blob's curl is evaluated on a 27 x 27 node patch around every point, so
the blob coupling costs about six times the delta coupling here) and 5.4 s for
`input2d_delta`. These times double or worse when other jobs
share the machine.

## Files

- `generate.py`: writes `ellipse.vertex`, `ellipse.spring` and the two input files.
- `hooks.py`: `update_springs` (constant tension) and `on_output` (`area.csv`).
- `analyze.py`: the tables and `results.png`.
- `tests/examples/test_blob_ellipse.py`: smoke runs of the two inputs, the committed files
  against `generate.py`, the curve against the paper's formula, the chain force against
  sigma kappa n, the half-turn symmetry, `area.csv` with no other output switched on,
  `analyze.py` on short runs, and a slow test of the two runs to t = 1 against the numbers
  above.
- `docs/examples.md` summarises this example with the others; `docs/numerics.md` describes
  the blob coupling and its interpolations.

References: R. Cortez and M. Minion, The blob projection method for immersed boundary
problems, J. Comput. Phys. 161 (2000) 428-453. C. S. Peskin and D. M. McQueen, Fluid dynamics
of the heart and its valves, chapter 14 of Othmer, Adler, Lewis and Dallon (eds.), Case
Studies in Mathematical Modeling (Prentice Hall, 1996), pp. 313-342. L. Fovargue, Addressing
the computational cost of the immersed boundary method through multi-implicit and multi-rate
strategies with time parallelism, PhD thesis, University of North Carolina at Chapel Hill
(2012).
