# pyIBM numerics

This is what pyIBM computes and how: the grids, the delta-function kernels, the structure
forces, the two ways of coupling the structure to the fluid, the two time schemes, the two
fluid solvers, and the known limitations. Every module docstring in `pyibm/` cites the
equations it implements; this page puts them together. The references are at the end.
`docs/user-guide.md` explains how to set up and run a case, and `docs/examples.md` how well
each example reproduces its reference.

Contents:

1. [Conventions](#1-conventions)
2. [Grids](#2-grids)
3. [Delta-function kernels](#3-delta-function-kernels)
4. [Delta-function coupling](#4-delta-function-coupling)
5. [Structure forces](#5-structure-forces)
6. [Blob projection coupling](#6-blob-projection-coupling)
7. [Time schemes](#7-time-schemes)
8. [Fluid solvers](#8-fluid-solvers)
9. [Diagnostics](#9-diagnostics)
10. [Known limitations](#10-known-limitations)
11. [What the tests check](#11-what-the-tests-check)
12. [References](#12-references)

## 1. Conventions

- **Units.** SI throughout, in two dimensions: density `rho` in kg/m^2, dynamic viscosity
  `mu` in N s/m^2, pressure in N/m, a body force density on the grid in N/m^2.
- **Arrays** are shaped `(Nx, Ny)` and indexed `[i, j]` with `i` along x
  (`np.meshgrid(x, y, indexing="ij")`), the same convention as the 2008 MATLAB code and
  Seibold's notes. Plotting and VTK output transpose where they have to; nothing else does.
- **Domain** `[0, Lx) x [0, Ly)` with `dx = Lx/Nx`, `dy = Ly/Ny`. The periodic FFT back end
  needs `dx == dy`; the MAC back end allows `dx != dy`.
- **Time** is `t_n = n*dt`, computed from `n`, never accumulated.
- **Lagrangian points** `X` are `(N, 2)` arrays of positions in metres. In a periodic
  direction they are wrapped into the box after every move, and separations between points
  use the minimum image, `d -= L*round(d/L)`.
- **Forces per unit length.** The Lagrangian force `f`, shape `(N, 2)`, is a force density per
  unit length of fibre [N/m]. Each point carries a quadrature weight `ds` [m], and spreading
  multiplies by it. So every structure coefficient (spring, beam, target and mass-tether
  stiffness, and the mass of a mass point) is per unit length too. A physical tether of
  stiffness `K` [N/m] at one point is `k = K/ds` in a `.target` file.
- Everything is `float64`.

## 2. Grids

`pyibm.grid.Grid(Lx, Ly, Nx, Ny, periodic_x, periodic_y)` holds every coordinate array:

- collocated nodes `x_i = i dx`, `y_j = j dy` (the periodic FFT back end, and the blob
  coupling);
- cell centres `xc = (i + 1/2) dx`, `yc = (j + 1/2) dy` (MAC pressure);
- u-faces `xu = i dx`, `yu = yc`, with `Nx + 1` values of `i` when x has walls (both wall
  faces are stored) and `Nx` when x is periodic;
- v-faces `xv = xc`, `yv = j dy`, likewise `Ny + 1` or `Ny` values of `j`.

`grid.meshgrid("nodes" | "centres" | "u" | "v")` returns the 2-D coordinate arrays. A
`FieldLayout` (`layout_collocated`, `layout_u`, `layout_v`) says where a field lives, so the
same spreading and interpolation code addresses nodes, u-faces and v-faces.

Cell counts: a periodic direction needs an even `N >= 8` (the FFT solver treats the
grid-scale wave number `N/2` on its own, and it exists only for even `N`; we use the same rule
for a periodic direction of the MAC grid). A direction with walls or an outflow needs
`N >= 4` and may be odd: four cells is the smallest box in which a 4-point stencil fits
between the walls on both MAC velocity fields.

## 3. Delta-function kernels

The coupling uses the tensor-product delta function

    delta_h(x, y) = phi(x/hx) phi(y/hy) / (hx hy)

(Peskin & McQueen 1996 eq. 14.62; Peskin 2002 eq. 6.1). Three 1-D kernels `phi(r)`, with `r`
in cells (`pyibm.kernels`):

| name | formula | support | properties |
|---|---|---|---|
| `peskin4` (default) | `(3 - 2r + sqrt(1 + 4r - 4r^2))/8` for `r <= 1`; `(5 - 2r - sqrt(-7 + 12r - 4r^2))/8` for `1 < r <= 2` (Peskin & McQueen 1996 eq. 14.73; Peskin 2002 eq. 6.27) | 4 nodes | even/odd sum rule, first moment, sum of squares 3/8 |
| `cosine4` | `(1 + cos(pi r/2))/4` for `r < 2` (Peskin 1977; Peskin 2002 eq. 6.28) | 4 nodes | even/odd sum rule and sum of squares 3/8, without the first-moment condition: it does not interpolate linear fields exactly. Kept to reproduce the 2008 MATLAB code |
| `roma3` | `(1 + sqrt(1 - 3r^2))/3` for `r <= 1/2`; `(5 - 3r - sqrt(1 - 3(1 - r)^2))/6` for `1/2 < r <= 3/2` (Roma, Peskin & Berger 1999) | 3 nodes | partition of unity, first moment, sum of squares 1/2, but not the even/odd sum rule |

The even/odd sum rule (Peskin & McQueen 1996 eq. 14.63) matters on the collocated periodic
grid: the centred-difference gradient there cannot see the grid-scale mode `(-1)^j`, and a
kernel without the rule lets that mode in. So `roma3` is meant for the MAC back end; on
`periodic_fft` the input-file check warns (Peskin 2002, section 6, recommends the 3-point
kernel only with a scheme without that null space).

The stencil along one axis takes the nodes within the kernel's reach: for the 4-point
kernels `i0 = floor((X - x0)/h)` and `i0 - 1 .. i0 + 2`, for `roma3` `i0 = round((X - x0)/h)`
and `i0 - 1 .. i0 + 1`, with the unwrapped index used for the distance and wrapped
(`idx % n`) in a periodic direction.

## 4. Delta-function coupling

`pyibm.coupling.delta.DeltaCoupling(kernel)` implements the two interaction equations
(Peskin & McQueen 1996 eqs. 14.58 and 14.61; Peskin 2002 eqs. 4.18-4.19):

    F(x_ij) = sum_k f_k ds_k delta_h(x_ij - X_k)                 (spread)
    U_k     = sum_ij u(x_ij) delta_h(x_ij - X_k) hx hy           (interpolate)

Both use the same (index, weight) tables, so they are adjoint to round-off:
`sum_ij F u hx hy = sum_k f . U ds`. That is the discrete power identity: the power the
structure puts into the fluid equals the power the fluid takes out of the structure
(Peskin 2002, section 5). On the MAC grid each velocity component is spread to and
interpolated from its own faces.

**Walls.** On the MAC back end every Lagrangian point must stay at least the kernel's
half-width from every side that is not periodic: 2 cells for `peskin4` and `cosine4`, 1.5
cells for `roma3`. A point closer than that would put part of its force on the wall faces,
whose values the solver overwrites with the boundary data, so that part of the force would be
lost without a trace (and still be counted in `fx_body`). The driver checks the distance when
the case is built and after every move (the half-step positions and the new positions) and
raises `ValueError` naming the point, the side and the step.

## 5. Structure forces

`pyibm.structures.compute_forces` returns the total force density `f (N, 2)` [N/m]. Every
elastic force is `-dE/dX` of an energy per unit length, so the spread force is energy
consistent, and for springs and beams the forces of one element sum to zero. With
`d = X_j - X_i` (minimum image), `L = |d|`, `e = d/L`:

| element | file | energy per unit length, force |
|---|---|---|
| spring | `.spring` | `E = k/(alpha+1) \|L - rest\|^(alpha+1)`; `f_i += k sgn(L - rest) \|L - rest\|^alpha e`, `f_j -=` the same. `alpha = 1` is Hookean |
| damped spring | `.d_spring` | the Hookean spring plus a dashpot `f_i += b (V_j - V_i)`, `f_j -=` the same |
| torsional beam | `.beam` | with `A = X_r - X_q`, `B = X_q - X_p`, `c = A x B = ax by - ay bx`: `E = k/2 (c - curv)^2` |
| non-invariant beam | `.nonInv_beam` | `D = X_r - 2 X_q + X_p - (cx, cy)`, `E = k/2 \|D\|^2`; `f_r += -k D`, `f_p += -k D`, `f_q += 2 k D` |
| target | `.target` | `f_i += k (X_target - X_i) + c (V_target - V_i)`, the tether of Miller & Peskin 2005 eq. 11 with optional damping `c` |
| mass | `.mass` | `f_i += k (Y - X_i)`, a tether to a ghost point `Y` of mass `m` per unit length |

Beam sign: `c = |A||B| sin(theta_B - theta_A)`, so a left (counterclockwise) turn from `p`
through `q` to `r` gives `c < 0`, and a convex polygon numbered counterclockwise has
`curv < 0` at every beam. For a chain of straight segments of length `ds` that turns by
`dtheta` at a node, `c = ds^2 sin(dtheta)`. Matching the energy of an element (its energy per
unit length times `ds`) to the continuum gives the per-unit-length coefficients in terms of
the physical ones: `k_spring = EA/ds^2` and `k_beam = EI/ds^6` for a chain with spacing `ds`
(`examples/swimmer/README.md` derives this and `examples/flapping_filament` checks it).

The ghost of a mass point obeys `m dV/dt = -k (Y - X) + m g` outside the fluid; gravity
acts on the ghosts only. The weight of the excess mass reaches the fluid through the tether.

**Velocities.** The dashpots and the target damping need the point velocities, which we form
from the previous positions, `V = min_image(X - X_prev)/dt`. `X_prev` is the previous
half-step position in the midpoint scheme and `X^{n-1}` in `pm1996`. In both schemes that
velocity is centred about half a step before the time at which the force is evaluated. The
target velocity `V_target` is zero unless a hook sets it, so a hook that moves damped targets
should set it too, or the damping pulls the points towards rest.

## 6. Blob projection coupling

`pyibm.coupling.blob.BlobProjection` is the blob projection method of Cortez & Minion 2000,
for the periodic FFT back end only. Instead of spreading each point force with a discrete
delta function, it is replaced by a smooth radial blob of scale `delta` (their eq. 13),

    F~(x) = sum_k ds_k f_k phi_delta(x - X_k),        phi_delta(r) = phi(r/delta)/delta^2,

and the fluid receives only the divergence-free part `P F~` of that regularised force (their
eq. 11). Two blobs, both of unit mass:

- `gaussian4`: `phi_delta = (2 - rho^2) exp(-rho^2)/(pi delta^2)`, `rho = r/delta`, the
  fourth-order Gaussian of Cortez & Minion 2000 eq. 15 (infinite support);
- `compact`: `phi_delta = 6 (1 - rho^2)^5/(pi delta^2)` for `rho <= 1`, Fovargue 2012
  eq. 2.30 (formally second order).

**Projection.** Following section 3.2 of the paper, the analytic curl of every blob force is
accumulated on a patch of nodes around its point, one FFT Poisson solve `-L psi = curl` gives
a stream function, and `P F~ = (D0_y psi, -D0_x psi)` is exactly divergence-free in the
centred-difference (`D0`) sense of the periodic back end. The box mean of the force, which a
stream function cannot carry, is added separately, so the total force is exact. `L` is the
five-point ("tight") Laplacian by default. The wide `D0_x^2 + D0_y^2` symbol decouples the
grid into four subgrids and gives a far-field checkerboard error of order one for a blob only
a few cells wide (the `BlobProjection` docstring has the numbers); it is still available as
`laplacian="wide"` when the class is built in code. The default patch holds every node within
a radius `R` of the point, `R = delta` for the compact blob and `R = 3 delta` for `gaussian4`
(a disc of radius `2 delta` would change the Gaussian blob's mass by 5.5 %, because its tail
beyond `sqrt(2) delta` is negative). `patch = 0` in the input file means this default.

**Pressure.** The gradient part of `F~` never reaches the fluid, so the fluid solver's
pressure holds no trace of the membrane. After every step the driver adds `p_F`, the solution
of `L p_F = div F~` for the same force and positions the step used. The reported blob
pressure is therefore the regularised pressure: a membrane under tension shows its Laplace
jump, smeared over the width of the blob. For a pressurised circle (`T/R = 4`, `gaussian4`,
`delta = 4h`) the measured jump is 4.347, 4.025 and 4.0019 on 32^2, 64^2 and 128^2.

**Marker velocity.** The Solver key `blob_interpolation` picks how the grid velocity is
taken to the points:

- `"streamfunction"` (the default): the curl of the bicubic interpolant of the five-point
  stream function of the grid velocity, `psi = (-L)^{-1}(D0_x v - D0_y u)`, plus the
  box-mean velocity. This marker velocity has zero divergence everywhere, so the markers of a
  closed membrane keep the area they enclose up to the time-stepping error. It is second
  order, but not exact at the nodes, and it drops the three checkerboard modes, which a
  stream function cannot carry.
- `"bicubic"`: bicubic Lagrange interpolation of `u` and `v` directly, on the 4 x 4 nodes
  around each point. The grid velocity is divergence-free only in the `D0` sense, so the
  interpolated field is divergence-free only to `O(h^2)`, in a band about `delta` wide around
  the membrane, and the area inside a closed membrane can drift slowly. The default
  stream-function interpolation is exactly divergence-free.
- `"adjoint"`: the exact adjoint of the projection, the curl of a blob-weighted sum of `psi`. It
  is divergence-free too, and the only one of the three with a discrete power identity.

On the perturbed ellipse of section 4.1 of Cortez & Minion 2000 (400 points on 128^2,
`gaussian4`, `delta = 4h`, `nu = 0.005`, `dt = 0.1h`) the relative area error at `t = 1` with
the two divergence-free interpolations, with the largest over the run in brackets, is

| grid | `streamfunction` | `adjoint` |
|---|---|---|
| 64^2 | +1.8e-5 (2.0e-5) | +5.7e-6 |
| 128^2 | +2.5e-6 (5.7e-6) | +4.1e-7 |
| 256^2 | +7.0e-7 (1.6e-6) | -1.5e-7 |

At 128^2 the paper reports errors within +-1e-5 (its Figs. 7 and 8). `examples/blob_ellipse`
runs this case with the default interpolation and with the delta function.

**How much the blob gains on area.** It depends on the force and on the interpolation. In
the steady leak test of `tests/test_invariants.py` (an ellipse in the Stokes flow its own
force drives), with the `"bicubic"` interpolation:

- for a constant-tension force, normal to the membrane, the blob leaks more than the delta
  function on 32^2, where the blob is as wide as the minor semi-axis, as much on 64^2, and 8
  and 75 times less on 128^2 and 256^2 (`test_static_leak_normal_force_blob_converges_fast`);
- for a purely tangential force both couplings leak at about first order with similar sizes
  (`test_static_leak_tangential_force`), so for spring networks, which carry tangential
  forces, the blob with the `"bicubic"` interpolation gains little over the delta function.

The default stream-function interpolation leaks far less in both cases (the test docstrings
have the rates). Most of the improvement over the delta function in our runs comes from the
divergence-free interpolation: on the rubber band, the delta coupling with the stream-function
interpolation (patched in by a scratch script; the input file offers it only for the blob)
keeps the area as well as the blob does, and on the Cortez & Minion ellipse, with the same
divergence-free interpolation for both, the blob is 3 times better by the polygon area and 11
times better by the area of the smooth curve through the points (largest error over the run at
128^2; `examples/rubberband/README.md` and `examples/blob_ellipse/README.md`).

**Quadrature.** Each point carries a fixed weight `ds`. When the force is given as a density
per unit arc length, such as `f = sigma kappa n`, and the points do not stay equally spaced,
fixed weights add a spurious net force. Integrating the force over each point's segment
avoids this: `examples/blob_ellipse` does it with zero-rest-length springs whose stiffness a
hook resets so that every link carries the tension `sigma`, which gives
`sigma (tau_{k+1/2} - tau_{k-1/2})` on point `k`.

**Resolution guidance**, measured on this implementation: `gaussian4` wants
`delta >= 3-4h` and `compact` wants `delta >= 4h` for about 1 % error in the far field; at
`3h` the compact blob still works, with a larger far-field error. Cortez & Minion's rules are
`delta > h`, so the velocity is regularised over several nodes, and `delta > ds`; points
still want `ds` somewhat below `h`. Tutorial 03 (section 7) has the same guidance with the
measurements behind it.

## 7. Time schemes

`pyibm.driver.Simulation.step` advances from `t = n dt` to `(n+1) dt` with one of two
schemes, chosen by the Solver key `time_scheme`.

**`midpoint`** (Peskin 2002 section 7, eqs. 7.8-7.12; the default on both back ends):

1. `U = interp(u^n, X^n)`; `X_h = X^n + (dt/2) U`; ghosts `Y_h = Y^n + (dt/2) V^n`.
2. The `update_*` hooks with `t + dt/2`.
3. `f` at `(X_h, Y_h)` plus the `lagrangian_force` hook; `F = spread(f, X_h)` (or the blob
   projection); `F += external_force(t + dt/2)`.
4. `u_h = half_step(F)`, then `full_step(F, u_h)`: the fluid over the full step with `u_h`
   in the advection term.
5. `U_h = interp(u_h, X_h)`; `X^{n+1} = X^n + dt U_h`. Ghosts:
   `a_h = -k (Y_h - X_h)/m + g`, `V_h = V^n + (dt/2) a_h`, `Y^{n+1} = Y^n + dt V_h`,
   `V^{n+1} = V^n + dt a_h`.

It is second order in time, with the caveats of section 8.2 on the MAC back end.

**`pm1996`** (Peskin & McQueen 1996 eqs. 14.55-14.61; the 2008 MATLAB code):

1. The `update_*` hooks with `t`; `f` at `X^n` plus `lagrangian_force`;
   `F = spread(f, X^n) + external_force(t)`.
2. One first-order fluid step.
3. `U = interp(u^{n+1}, X^n)`; `X^{n+1} = X^n + dt U`; ghosts by one symplectic Euler step,
   `V += dt (-k (Y - X)/m + g)`, then `Y += dt V`.

This is the order of the 2008 loop, which is why `tests/test_matlab_regression.py` reproduces
the MATLAB state to round-off.

**Time levels.** Hooks are called with the time at which the forces are evaluated
(`t + dt/2` or `t`) and must be functions of absolute time. Boundary data on the MAC back end
are evaluated at the start, middle and end of the substeps (`t`, `t + dt/2`, `t + dt`). The
history rows and output files describe the state at `t = n dt`, with two exceptions that
section 10 lists: the force columns and the pressure.

## 8. Fluid solvers

Both back ends solve

    rho (u_t + u . grad u) + grad p = mu Lap u + F,        div u = 0,

with the force density `F` [N/m^2] given on their own velocity locations.

### 8.1 Periodic FFT back end (`backend = "periodic_fft"`)

`pyibm.fluid.periodic_fft.PeriodicFFTSolver`: a doubly periodic box of collocated nodes,
`dx = dy = h`. The gradient and divergence are the centred difference `D0`, the viscous term
the tight five-point Laplacian `L_h` (Peskin & McQueen 1996 eqs. 14.29-14.33). Once the
nonlinear term and the force are moved into a known vector `w`, every step is a linear
constant-coefficient system that is diagonal in Fourier space, so one FFT per component
solves it exactly (eqs. 14.46-14.52). With `s1 = sin(2 pi k1/Nx)`, `s2 = sin(2 pi k2/Ny)`:
`D0 <-> i s/h`, `L_h <-> -(4/h^2)(sin^2(pi k1/Nx) + sin^2(pi k2/Ny))`, and the projection is
`u_hat = w_hat - s q` with `q = (s1 w_u + s2 w_v)/(s1^2 + s2^2)`. Four modes have
`s1 = s2 = 0`, the box mean and the three checkerboards (p. 323, eq. 14.53); the projection
leaves them alone and the pressure is zero there.

- `pm1996` (`step`): upwind advection, backward Euler for the viscous term, the projection
  (Peskin & McQueen 1996 section 14.4). It reproduces the 2008 MATLAB code (same upwind rule,
  operation order, symbols and degenerate modes).
- `midpoint` (`half_step`, `full_step`): skew-symmetric advection
  `S(u) phi = (u D0_x phi + v D0_y phi)/2 + (D0_x(u phi) + D0_y(v phi))/2`, backward Euler
  over `dt/2` for the half step, then Crank-Nicolson over `dt` for the full step (Peskin 2002
  section 7).

**`remove_mean_force`.** The box mean (`k = 0`) passes through the projection untouched, so
a net force on the fluid, for example from a body held by target points, accelerates the
whole box without end. With `remove_mean_force = 1`, every solve sets the `k = 0` coefficient
of `w` to that of `u^n`. That removes the box mean of every term that changes `u` in the step
(the force, the advection and the viscous term), so the box-mean velocity keeps its starting
value to round-off in both schemes. In `pm1996` this matters for the advection too, since
upwinding is not conservative and its box mean is not zero. The history's `fx_body`,
`fy_body` are the spread force before the removal. Off by default, as in the MATLAB code.

### 8.2 MAC projection back end (`backend = "mac"`)

`pyibm.fluid.mac.MACSolver` follows Seibold's MIT 18.086 notes (Seibold 2008): a staggered
grid with pressure at cell centres, `u` on the vertical faces and `v` on the horizontal faces,
explicit advection in conservative form, implicit viscosity by pre-factored sparse LU, and
Chorin's projection with a pressure Poisson solve. We extend it to velocity data that vary in
space and time on every side, an outflow side, periodic directions, a body force on the faces,
the midpoint scheme, and an incremental projection.

**Sides** (`Boundary_Conditions`; periodic must be set on both sides of a direction, and at
most one side may be an outflow):

- `dirichlet`: the normal velocity on the wall faces is set exactly to the data of the
  `bc_<side>` hook (zero, a wall at rest, when there is no hook); the tangential component
  uses the ghost reflection `2 u_wall - u_interior` (Seibold eq. 35). An inflow is a Dirichlet
  side with a nonzero normal velocity.
- `outflow`: zero-gradient ghosts for both components, `p = 0` on the boundary (a Dirichlet
  condition for the Poisson problem), and the normal face velocity on that side is corrected
  by the projection like an interior face.
- `periodic`.

In a closed box (no outflow, no periodic direction) every step checks that the net inflow
through the walls is zero and raises `ValueError` otherwise.

**Advection.** `advection = "blend"` (the default) is Seibold's blend of central and upwind
differences with `gamma = min(1.2 dt max(|u|/dx, |v|/dy), 1)`; `"central"` is `gamma = 0`,
`"upwind"` is `gamma = 1`. Because the blend's `gamma` is proportional to `dt`, it adds a
numerical viscosity of about `gamma |u| dx/2` that is `O(dt)` and does not go away as the
grid is refined, so with `"blend"` the midpoint scheme is first order in time even in a
periodic box. On a 32^2 periodic Taylor-Green vortex (`nu = 0.01`, `T = 0.5`) the midpoint
errors against a fine-`dt` run were 6.2e-2, 3.3e-2, 1.7e-2, 8.6e-3 at `dt` = 0.02, 0.01,
0.005, 0.0025 with `"blend"` and 5.5e-6, 1.4e-6, 3.4e-7, 8.5e-8 with `"central"`. Use
`"central"` when the cell Reynolds number `max|u| dx/nu` is about 2 or less.

**Viscosity.** A substep has the length `h_sub`: `dt/2` in `half_step`, `dt` in `full_step`
and `step`. `viscous = "implicit"` solves `(I - theta nu h_sub L_h) u** = rhs` with the
boundary data moved to the right-hand side (Seibold's `Ubc`, `Vbc`): backward Euler
(`theta = 1`) in `step` and `half_step`, Crank-Nicolson (`theta = 1/2`) in `full_step`.
`"explicit"` adds `nu L_h u` to the predictor and warns when `dt` exceeds
`1/(2 nu (1/dx^2 + 1/dy^2))`.

**Projection.** Each substep solves `-L_h phi = -(1/h_sub) D_h u**` and sets
`u = u** - h_sub G_h phi`. The Solver key `mac_projection` picks the pressure update:

- `"nonincremental"` (the default, Chorin 1968): the predictor carries no pressure gradient
  and `phi` is the whole pressure over `rho`. With implicit viscosity the splitting error is
  first order in `dt`: the viscous solve imposes the wall data on `u**`, and the projection
  then leaves the tangential velocity next to a Dirichlet wall with a slip of
  `h_sub d(phi)/d(tau)`, which is `O(dt)` wherever the pressure varies along the wall
  (section 10).
- `"incremental"`: the predictor carries the previous pressure, `phi` is only the increment,
  and the wall slip is `O(dt^2)`. This is the pressure-increment projection with the
  rotational update (Brown, Cortez & Minion 2001; reviewed by Guermond, Minev & Shen 2006).
  The first step starts from `p = 0`, so it has the non-incremental slip once.

With `viscous = "explicit"`, or in a doubly periodic box, the two give the same velocity and
pressure to round-off.

**Reported pressure.** With implicit viscosity, `phi` is a smoothed pressure, close to
`(I - c L_h)^{-1} p` with `c = theta nu h_sub`, so a pressure jump at an immersed boundary
would be smeared over about `sqrt(c)`. In both variants the solver therefore reports the
rotational form `rho (phi - c L_h phi)` (plus the previous pressure when incremental), the
pressure that matches the discrete momentum equation (Brown, Cortez & Minion 2001;
Timmermans, Minev & van de Vosse 1996). For a pressurised circle in a closed box the jump
error was -5.5e-5, -8.5e-6, -2.7e-6 on 32^2, 64^2, 128^2 with this form, against -4.1e-3,
-5.9e-3, -7.2e-3 with `p = rho phi`. In the non-incremental variant the velocities do not
depend on this choice.

**Poisson solvers** (`pyibm.fluid.poisson`): the five-point Laplacian on cell centres with a
Neumann condition at every Dirichlet wall, a zero-value condition half a cell outside the
last centre at an outflow side, and wrap-around in a periodic direction (Seibold eqs. 20,
36, 37). A sparse matrix built with `kron` of 1-D tridiagonals and factored once with
`splu`, or the type-2 discrete cosine transform for the all-Neumann box (chosen
automatically). Without a Dirichlet side the operator is singular; both solvers then drop the
mean of the right-hand side and return the zero-mean solution.

**Order in time.** On this back end the midpoint scheme is second order in time only when
the advection blend does not depend on `dt` (`advection = "central"`), and, with the
non-incremental projection and implicit viscosity, no Dirichlet wall carries a tangential
pressure gradient. The one-step scheme (`pm1996`, which here is Seibold's step: explicit
advection, backward Euler viscosity, projection) is first order in every case.

## 9. Diagnostics

The driver writes one history row every `print_dump` steps (`docs/user-guide.md`, output):

- `fx_body, fy_body`: the force the fluid puts on the structure, `-sum(F) hx hy` of the
  spread structure force (springs, beams, targets, mass tethers and the `lagrangian_force`
  hook; not the `external_force` hook). It sums the delta-spread force, or `P F~` for the
  blob, which has the same total. Its time level is `t_force` (section 10).
- `max_target_error_over_dx`: the largest distance between a target and its point, in cells
  of `dx`, with the targets refreshed to the row's `t` by `update_targets`.
- `cfl`: on `periodic_fft`, the pointwise `dt max(|u|/dx + |v|/dy)` over the nodes. On `mac`,
  the upper bound `dt (max|u|/dx + max|v|/dy)`, which can be up to twice the pointwise
  number. The `cfl > 1` warning is therefore conservative on the MAC back end.
- `mean_u, mean_v`: the box-mean velocity.
- `kinetic_energy`: `(rho/2) sum (u^2 + v^2) dx dy` over the nodes (FFT), or with the
  trapezoidal rule over the faces (MAC), in J in 2D.
- `max_speed`: the largest `sqrt(u^2 + v^2)` on the nodes (FFT) or on the cell centres after
  averaging the face velocities there (MAC).

**Tether rule of thumb.** Explicit fibre forces limit the time step through the stiffness
(Stockie & Wetton 1999 analyse this for a stretched fibre). For target points and mass
tethers the driver computes the tether number `k dt^2/(rho dx)` of the stiffest element and
warns once when it is above the largest value at which our own test runs stayed stable,
`pyibm.driver.TETHER_STABLE`: 10 for targets and 7 for mass tethers, with either scheme. The
test: a ring of radius 0.125 in the unit periodic box, 64^2, `rho = 1`, `mu = 0.01`,
`peskin4`, started in a uniform stream and held by target points or by ghost masses of
`m = 1` kg/m. Targets ran 3000 steps at 10 and blew up within 65 steps at 12; mass tethers
ran 3000 steps at 7 and blew up after 360-690 steps at 8. It is a reason to check `dt` or
`k`, not a stability limit: the limit depends on the set-up. Springs and beams are not
checked. Along a chain the stiffest modes are at the marker scale and barely reach the fluid,
so their limits are much higher (the swimmer example measured the spring limit between 20 and
80 in the same number).

## 10. Known limitations

- **First-order wall slip of the non-incremental MAC projection.** With the default
  `mac_projection = "nonincremental"` and implicit viscosity, the tangential velocity at a
  Dirichlet wall slips by `O(dt)` wherever the pressure varies along the wall. Measured on a
  Taylor-Green vortex shifted so that the pressure varies along every wall, with the exact
  velocity as time-varying wall data (closed box, 32^2, `nu = 0.01`, `T = 0.5`, midpoint,
  central advection), against an incremental run at `dt = T/1600` on the same grid, at
  `dt` = 0.02, 0.01, 0.005, 0.0025:

  | projection | max velocity error |
  |---|---|
  | non-incremental | 1.1e-2, 5.5e-3, 2.8e-3, 1.4e-3 (first order) |
  | incremental | 3.9e-4, 9.7e-5, 2.4e-5, 6.0e-6 (second order) |

  (A non-incremental run at `dt = T/1600` still carries its own slip, 1.8e-4 off the
  incremental one, which is why the reference is incremental.) In a Poiseuille channel with
  an outflow side (64 x 32, `T = 1`) the non-incremental steady state is off the discrete
  fully developed `dp/dx` by 1.3e-3 relative with the midpoint scheme and 2.5e-3 with
  `pm1996` at `dt = 2e-3`, and that halves with `dt`; the incremental one matches it to
  3.4e-5 at every `dt` tried. In the Womersley example the slip doubles the error at the
  wall cells (1.34e-2 of the core speed, against 6.7e-3 with the incremental projection).
  Set `mac_projection = "incremental"` when the pressure varies along a wall.
- **Time levels in the history.** A row describes the state at `t = n dt`, except:
  - the force columns `fx_body, fy_body` (and `Simulation.f`, `Fu`, `Fv` and the `force`
    point data of `lag.NNNNN.vtk`) hold the force of the last step, which the scheme
    evaluates at `t_force = t - dt/2` (midpoint) or `t - dt` (`pm1996`); row 0 holds the
    force at the initial positions with `t_force = 0`. Compare forces at `t_force`, not at
    `t`: row for row, a `pm1996` run and a midpoint run of the heaving plate differ by 1.9 %
    of the peak force, and at matched `t_force` by 0.49 %.
  - the MAC pressure comes from the projection of the step that ends at `t` and balances the
    momentum over that whole step, so it is best read at `t - dt/2`. Fitting `-dp/dx` of the
    Womersley channel to `K cos(omega (t - lag))` gave a lag of 0.64 `dt` (non-incremental)
    and 0.56 `dt` (incremental) with the midpoint scheme, and 0.68-0.74 `dt` and 0.51-0.57
    `dt` with `pm1996`.
  - the velocity of the damped springs and damped targets is centred half a step before the
    force time (section 5).
- **Blob energy.** With `"streamfunction"` or `"bicubic"` marker velocities the blob coupling
  has no discrete power identity: the power the fluid receives is not the power the structure
  gives, because the interpolation is not the adjoint of the projection. For a stretched
  ellipse in steady Stokes flow the ratio of the two is 0.64, 0.78, 0.92, 0.97 on 32^2 to
  256^2 (`gaussian4`, streamfunction). Released from rest with `mu = 0` on 32^2, kinetic plus
  elastic energy changes by -22 % (`gaussian4`) and -12 % (compact) by `t = 0.5` with the
  stream-function interpolation, and longer runs gain energy. Do not read the energy history
  of such a run as physical, and do not use these interpolations for inviscid or nearly
  inviscid runs. `"adjoint"` conserves energy to the time-stepping error.
- **Compact blob at low viscosity.** On the forced circle at `nu = 1e-4`, whose membrane has
  no bending term, the compact blob runs to the end at 3h with `"streamfunction"` or
  `"bicubic"`, but the chain folds at the marker scale (312-331 of 400 points kinked with
  `"streamfunction"`, 107 with `"bicubic"`, against 62 for `gaussian4` at 4h), and at 4h
  `"bicubic"` blows up. With a stiffer membrane (`k_S = 3e4`) `"streamfunction"` reaches
  speeds of 37 at 3h and blows up at 4h. `"adjoint"` kept the peak speed at 1.5-1.8 in every
  case we ran and runs at 4h, but at 3h its chain still had 151 kinked points. For low
  viscosity on such a membrane use `gaussian4` (`examples/forced_circle/README.md` and the
  `pyibm.coupling.blob` docstring have the runs).
- **Blob with the wide Laplacian.** With `laplacian = "wide"` (code only) the
  stream-function interpolation amplifies the modes near the checkerboard, and `p_F` carries
  a subgrid offset of about 9 % of `T/R` for the compact blob at 3h. The default `"tight"`
  has neither problem.
- **Wall distance.** On the MAC back end a point must stay at least the kernel's half-width
  from every non-periodic side (section 4); the driver raises rather than lose force. A body
  that has to come closer to a wall needs a finer grid.
- **The blob needs the periodic FFT back end.** There is no blob coupling with walls.
- **At most one outflow side**, and periodic sides come in opposite pairs.
- **`roma3` on the periodic back end** has a checkerboard null space; the input-file check
  warns.
- **Tether stability** is checked only by a rule of thumb, and only for targets and mass
  tethers (section 9). Every step ends with a check that the velocity, the positions and the
  ghosts are finite; a blow-up raises `FloatingPointError` naming the step.
- **Speed.** Everything is vectorised NumPy and SciPy, single process. The run times in
  `docs/examples.md` range from a fraction of a second to about two hours.

## 11. What the tests check

`python -m pytest` runs the default suite; `python -m pytest -m slow` runs only the long runs
that compare with published or analytic results. A few of the numerical checks, with the
tolerance the test uses and what we measured (the test docstrings hold the details):

| test | check | tolerance | measured |
|---|---|---|---|
| `tests/test_matlab_regression.py` | the `pm1996`/`cosine4` port against the unmodified 2008 MATLAB code, 100 steps, eight checkpoints, two stiffnesses | 1e-10 relative in `u`, `v`, `X`, `p`, forces | about 1e-15 |
| `tests/test_mac.py::test_poiseuille_channel` | parabolic inflow, outflow, no-slip walls: profile and `dp/dx = -8 mu U/Ly^2` | 1e-2 | profile 1.6e-3, `dp/dx` 3.8e-3 (the wall-ghost error of the discrete parabola plus the non-incremental slip) |
| `tests/test_mac.py::test_poiseuille_channel_incremental` | the same with the incremental projection, against the discrete fully developed flow | | `dp/dx` 3.4e-5, profile 1e-6 |
| `tests/test_mac.py::test_lid_driven_cavity_ghia` | Re 100 on 64^2 with the one-step scheme against Ghia, Ghia & Shin 1982 | 0.05 lid units; 5 % where `\|u\| > 0.1` or `\|v\| > 0.15`; 2 % for `v` left of the centre | `u` within 0.0071, `v` within 0.0068; `v` within 1.15 % left of the centre |
| `tests/test_mac.py::test_solid_body_rotation_time_varying_walls` | walls moving as a rigid rotation `Omega(t) = sin t` | | interior off the rigid rotation by 1.6e-4 at `t = pi/2`, 4e-3 at `t = pi` |
| `tests/test_mac.py::test_taylor_green_periodic` | mode amplitude against the discrete eigenvalue and the time factor of the scheme | | 3e-14 |
| `tests/test_backends_agree.py` | Taylor-Green through both back ends via the driver | each back end returns the Crank-Nicolson factor times its initial field to 1e-12, and the two decay rates agree to 1e-12; the advected vortex within 3e-3 of the exact solution on each back end's own points | 1.85e-3 on both (0.29 with the advection switched off) |
| `tests/test_invariants.py` | momentum balance, energy balance, Laplace's law, mirror symmetry, Galilean invariance, area of closed membranes, convergence under refinement, back ends agreeing | per test | per test |

## 12. References

- Battista, N. A., Strickland, W. C. and Miller, L. A. (2017). IB2d: a Python and MATLAB
  implementation of the immersed boundary method. Bioinspiration & Biomimetics 12, 036003.
- Battista, N. A., Strickland, W. C., Barrett, A. and Miller, L. A. (2018). IB2d Reloaded: a
  more powerful Python and MATLAB implementation of the immersed boundary method.
  Mathematical Methods in the Applied Sciences 41, 8455-8480.
- Brown, D. L., Cortez, R. and Minion, M. L. (2001). Accurate projection methods for the
  incompressible Navier-Stokes equations. Journal of Computational Physics 168, 464-499.
- Chorin, A. J. (1968). Numerical solution of the Navier-Stokes equations. Mathematics of
  Computation 22, 745-762.
- Cortez, R. and Minion, M. (2000). The blob projection method for immersed boundary
  problems. Journal of Computational Physics 161, 428-453.
- Fovargue, L. (2012). Addressing the computational cost of the immersed boundary method
  through multi-implicit and multi-rate strategies with time parallelism. PhD thesis,
  University of North Carolina at Chapel Hill.
- Guermond, J. L., Minev, P. and Shen, J. (2006). An overview of projection methods for
  incompressible flows. Computer Methods in Applied Mechanics and Engineering 195,
  6011-6045.
- Miller, L. A. and Peskin, C. S. (2005). A computational fluid dynamics of 'clap and fling'
  in the smallest insects. Journal of Experimental Biology 208, 195-212.
- Peskin, C. S. (1977). Numerical analysis of blood flow in the heart. Journal of
  Computational Physics 25, 220-252.
- Peskin, C. S. (2002). The immersed boundary method. Acta Numerica 11, 479-517.
- Peskin, C. S. and McQueen, D. M. (1996). Fluid dynamics of the heart and its valves. In
  Othmer, H. G., Adler, F. R., Lewis, M. A. and Dallon, J. C. (eds.), Case Studies in
  Mathematical Modeling: Ecology, Physiology, and Cell Biology, chapter 14, pp. 313-342.
  Prentice Hall.
- Roma, A. M., Peskin, C. S. and Berger, M. J. (1999). An adaptive version of the immersed
  boundary method. Journal of Computational Physics 153, 509-534.
- Seibold, B. (2008). A compact and fast Matlab code solving the incompressible Navier-Stokes
  equations on rectangular domains. Course notes for MIT 18.086,
  http://www-math.mit.edu/18086.
- Stockie, J. M. and Wetton, B. R. (1999). Analysis of stiffness in the immersed boundary
  method and implications for time-stepping schemes. Journal of Computational Physics 154,
  41-64.
- Timmermans, L. J. P., Minev, P. D. and van de Vosse, F. N. (1996). An approximate projection
  scheme for incompressible flow using spectral elements. International Journal for
  Numerical Methods in Fluids 22, 673-688.

The papers each example reproduces are listed in `docs/examples.md`.
