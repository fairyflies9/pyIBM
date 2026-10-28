"""Blob projection method coupling (Cortez & Minion 2000; Fovargue 2012 chapter 2).

Instead of spreading the Lagrangian force with a discrete delta function, each point force
f_k ds_k is replaced by a smooth radial blob of scale delta (eq. 13 of Cortez & Minion 2000),

    F~(x) = sum_k ds_k f_k phi_delta(x - X_k),      phi_delta(r) = phi(r/delta)/delta^2,

and what we put on the grid is the divergence-free part P F~ of that regularized force. In free
space P F~ is known in closed form (eq. 14: a regularized dipole per point). On the doubly
periodic grid we follow section 3.2 of the paper: the analytic curl of every blob force is
accumulated on a small patch of nodes around its point, one FFT Poisson solve gives the stream
function, and the velocity-like field (D0_y psi, -D0_x psi) is exactly discretely divergence-free.
The box mean of the force, which a stream function cannot carry, is added separately so the total
force is preserved.

The gradient part of F~ never reaches the fluid, so the fluid solver's pressure holds no trace of
the membrane. ``force_pressure`` returns the missing part, p_F with L p_F = div F~, so the
reported pressure can be solver.p + p_F; a membrane under tension then shows its Laplace jump.

Velocity at the points
----------------------
Three interpolations are offered (``interpolation`` in the constructor). The first and the last
use the stream function of the grid velocity, psi = (-L)^{-1} omega, where omega = D0_x v - D0_y u
is the discrete vorticity and L the constructor's Laplacian (``stream_function``).

- "streamfunction" (the default): the exact curl of the bicubic Lagrange interpolant of psi,
  plus the box-mean velocity. This field has zero divergence everywhere, so the markers of a
  closed membrane keep the area they enclose, up to the time-stepping error.
- "bicubic": interpolates u and v directly, with the bicubic Lagrange interpolant on the 4 x 4
  nodes around each point. The grid velocity is divergence-free only in the centred-difference
  (D0) sense, so the interpolated field is divergence-free only to O(h^2), and the area inside a
  closed membrane can drift slowly. The default stream-function interpolation is exactly
  divergence-free.
- "adjoint": the adjoint of the projection, the curl of a blob-weighted sum of psi. It is
  divergence-free too, and it is the only one of the three with a power identity (see Energy
  below).

Which stream function. The obvious one is the least-squares stream function of D0, which gives
a D0-divergence-free velocity back exactly; it is what L = D0_x^2 + D0_y^2 ("wide") gives. We
do not use it with the default "tight" projection. For a mode that D0 barely sees (close to the
checkerboard in one direction) it is large for its grid speed, and the exact derivative of its
interpolant amplifies the mode: 34 times the grid speed at k1 = N/2 - 1, k2 = 1 on 128^2, where
the five-point Laplacian gives 0.04 and bicubic 1.0. With it the compact blob blew up on the
forced circle at nu = 1e-4 (3h, k_S = 1e4: t = 0.30, 0.36, 0.41 at dt = dx/8, dx/16, dx/32) where
it now runs (see below). The price: the five-point psi is not the least-squares one, so a grid
velocity u = D0_y psi0, v = -D0_x psi0 comes back to second order instead of third (on a sampled
smooth velocity both are second order and the five-point psi is the more accurate, 1.3e-2
against 2.6e-2 at 128^2 for a field of speed 14), and the middle modes are damped (gain 0.77 at
k1 = N/4, against 1.53 for the least-squares psi and 1.0 for bicubic).

Area. On the perturbed ellipse of section 4.1 of the paper (400 points on 128^2, scaled with
the grid; gaussian4 blob, delta = 4h, nu = 0.005, dt = 0.1h) we measured these relative area
errors at t = 1, with the largest over the run in brackets:

    grid    streamfunction       adjoint
    64^2    +1.8e-5 (2.0e-5)     +5.7e-6 (2.1e-5)
    128^2   +2.5e-6 (5.7e-6)     +4.1e-7 (7.2e-6)
    256^2   +7.0e-7 (1.6e-6)     -1.5e-7 (1.5e-6)

Both wander about zero. With "bicubic" the area falls slowly and steadily on the same runs, as
described above. The membrane positions at t = 1 on 128^2 are 0.21-0.24h (largest marker
distance) from the 256^2 runs for "streamfunction" and "bicubic", which differ from each other
by 0.027h, and 0.35-0.38h for "adjoint", whose velocity is a blob average and so smoother than
the point value.

Energy. With "streamfunction" or "bicubic" the coupling has no discrete power identity: the
power the fluid receives, sum_ij F_ij . u_ij h^2, is not the power the structure gives,
sum_k f_k . U_k ds_k, because the interpolation is not the adjoint of the projection. For the
stretched ellipse of tests/test_invariants.py in the steady Stokes flow its own force drives
(delta = 4h), the ratio of the two at 32^2, 64^2, 128^2, 256^2 is

    streamfunction   gaussian4 0.64 0.78 0.92 0.97    compact 0.67 0.83 0.92 0.96
    bicubic          gaussian4 0.63 0.77 0.91 0.96    compact 0.64 0.82 0.91 0.96
    adjoint          1 to round-off, both blobs

So kinetic plus elastic energy is not conserved, not even up to the time-stepping error.
Released from rest with mu = 0 on 32^2, the total at t = 0.5 changes by -22 % (streamfunction)
and -30 % (bicubic) with gaussian4, and by -12 % and +15 % with compact, the same for dt = 2e-3
and 1e-3 to within 3 %. Longer runs gain energy: with "streamfunction" gaussian4 is at 1.5
times its initial energy at t = 2, compact at 14-21 times. With "adjoint" the change at t = 0.5
is +4.1e-5 (gaussian4) and +3.4e-4 (compact) at dt = 2e-3 and falls as dt^2, which is the
midpoint scheme's own error. Do not read the energy history of a "streamfunction" or "bicubic"
run as physical, and do not use them for inviscid or nearly inviscid runs.

Compact blob at low viscosity. On the forced circle of examples/forced_circle (128^2, 400
points at 0.4h, zero-rest-length springs, targets with k_T = 3200, nu = 1e-4, one period) we ran
the compact blob at dt = dx/8, dx/16 and dx/32 with k_S = 1e4 (the example's value). max|u| is
the largest fluid speed, checked every 16th step; the table in examples/forced_circle/README.md
checks every step, so its peaks for the same runs are up to 7 % higher:

- at 3h all three interpolations run to t = 1, but with "streamfunction" the chain is rough at
  the marker scale: max|u| 2.8, 2.8, 4.0, against 1.6 for "bicubic" (all three dt), 1.7 for
  "adjoint" (dx/16) and 1.2 for gaussian4 at 4h with "streamfunction";
- at 4h "streamfunction" runs to t = 1 but rougher still (max|u| 8.1, 6.8, 6.5), "adjoint" runs
  cleanly (1.6 at dx/16), and "bicubic" blows up at t = 0.67, 0.88, 0.98.

With k_S = 3e4 and dt = dx/16 only "adjoint" stays clean (max|u| 1.7 at 3h, 1.6 at 4h, and
1.5-1.7 at dx/8 and dx/32): "streamfunction" grows to max|u| 37 at 3h (at dx/8 it blows up at
t = 0.72) and blows up at t = 0.55 at 4h, "bicubic" blows up at t = 0.71 (3h) and 0.41 (4h). At
nu = 1e-3 (k_S = 1e4, dt = dx/16) "streamfunction" and "bicubic" run the compact blob to t = 1
(streamfunction: max|u| 1.2 at 3h, 1.1 at 4h), but the chain is still kinked at the marker
scale (46 kinked points at 3h against 4 for gaussian4 at 4h, examples/forced_circle). The
gaussian4 blob at 3h and 4h runs to t = 1 with "streamfunction" (max|u| 1.3, 1.2) and
"bicubic", and with "adjoint" at 4h. So on a membrane like this one, which has no bending
term, use gaussian4 at low viscosity, or "adjoint" if you need the compact blob.

Choosing. "streamfunction" keeps the area and puts the membrane as close to the fine-grid answer
as "bicubic" does, for 0.5 ms per call (bicubic: 0.2 ms) with 400 points on 128^2.
"adjoint" keeps the area, conserves energy and is stable where the others are not, but it
costs about one patch evaluation per call (15 ms for gaussian4 at 4h, as much as
``projected_force``), its markers sit further from the fine-grid answer (0.35h against 0.22h
above), and with the compact blob at 3h its velocity carries a 3-6 % error on smooth fields that
does not shrink with h.

Conventions (docs/numerics.md): arrays (Nx, Ny) indexed [i, j] with i along x, nodes at
x_i = i*h on the collocated periodic grid, SI units, float64. This module only needs the grid; it
is the periodic FFT back end's companion.
"""

from __future__ import annotations

import math

import numpy as np

from pyibm.grid import Grid

__all__ = ["Blob", "Gaussian4Blob", "CompactBlob", "get_blob", "BlobProjection"]

INTERPOLATIONS = ("streamfunction", "bicubic", "adjoint")


# ---------------------------------------------------------------------------------------------
# Blobs
# ---------------------------------------------------------------------------------------------
class Blob:
    """A radial cutoff (blob) function with unit mass in 2D and scale delta.

    Subclasses supply ``phi``, ``dphi_over_r`` and ``shape_factor``. All three take the radius
    ``r`` (any array shape, metres) and the scale ``delta`` (metres) and are vectorized.

    phi(r, delta)          phi_delta(r) = phi(r/delta)/delta^2, the blob itself [1/m^2].
    dphi_over_r(r, delta)  phi_delta'(r)/r [1/m^4], the factor in the analytic curl of a blob
                           force (finite at r = 0).
    shape_factor(r, delta) F(r) = 2 pi int_0^r s phi_delta(s) ds, the fraction of the blob mass
                           inside radius r (Cortez & Minion 2000, text after eq. 14). F -> 1.
                           Written without the cancellation of 1 - (a number close to 1), so
                           F/r^2 keeps full relative accuracy as r -> 0.
    """

    name: str = ""
    compact: bool = False

    def phi(self, r: np.ndarray, delta: float) -> np.ndarray:
        raise NotImplementedError

    def dphi_over_r(self, r: np.ndarray, delta: float) -> np.ndarray:
        raise NotImplementedError

    def shape_factor(self, r: np.ndarray, delta: float) -> np.ndarray:
        raise NotImplementedError

    def kernels(self, r: np.ndarray, delta: float) -> tuple[np.ndarray, np.ndarray]:
        """The two radial kernels of the free-space projected force, eq. 14 rearranged.

        Cortez & Minion 2000 eq. 14 reads, per point,
            P F~ = ds [ (1/2) f phi_delta(r) + (1/(2 pi)) (f - 2 (f.x^) x^) (r F' - 2 F)/(2 r^2) ].
        With F'(r) = 2 pi r phi_delta(r) this is ds [ p1(r) f - p2(r) (f.x^) x^ ] with
            p1 = (r F' - F)/(2 pi r^2) = phi_delta - F/(2 pi r^2),
            p2 = (r F' - 2 F)/(2 pi r^2) = phi_delta - F/(pi r^2),
        the form used by Fovargue 2012 eq. 2.34. Both are finite at r = 0: since
        F = pi phi_delta(0) r^2 + O(r^4), p1(0) = phi_delta(0)/2 and p2(0) = 0. Far from the
        blob (F = 1, phi = 0) they are -1/(2 pi r^2), -1/(pi r^2): the classical 2D potential
        dipole.

        The shape factors are cancellation free, so F/(pi r^2) is accurate to round-off for
        every r > 0 and p1, p2 are accurate to round-off relative to phi_delta(0). For
        r < 1e-8 delta we return the r = 0 values; the error is below 7.5 (r/delta)^2 < 1e-15
        of p1(0) for the compact blob (2.25 (r/delta)^2 for gaussian4), which is round-off, and
        the branch avoids 0/0 at r = 0.
        """
        r = np.asarray(r, dtype=float)
        near = (r / delta) ** 2 < 1e-16
        r_safe = np.where(near, delta, r)
        phi = self.phi(r_safe, delta)
        F_over = self.shape_factor(r_safe, delta) / (np.pi * r_safe**2)   # F/(pi r^2) -> phi(0)
        p1 = np.where(near, 0.5 * self.phi(np.zeros_like(r), delta), phi - 0.5 * F_over)
        p2 = np.where(near, 0.0, phi - F_over)
        return p1, p2


class Gaussian4Blob(Blob):
    """Fourth-order Gaussian cutoff, Cortez & Minion 2000 eq. 15.

    phi_delta(r) = (1/(pi delta^2)) (2 - rho^2) exp(-rho^2), rho = r/delta. Unit mass, zero second
    moment, infinite support (the paper's workhorse blob, used with delta = 4h or 2h).
    """

    name = "gaussian4"
    compact = False

    def phi(self, r, delta):
        rho2 = (np.asarray(r, dtype=float) / delta) ** 2
        return (2.0 - rho2) * np.exp(-rho2) / (np.pi * delta**2)

    def dphi_over_r(self, r, delta):
        """phi_delta'(r)/r = -(2/(pi delta^4)) (3 - rho^2) exp(-rho^2); differentiate eq. 15."""
        rho2 = (np.asarray(r, dtype=float) / delta) ** 2
        return -2.0 * (3.0 - rho2) * np.exp(-rho2) / (np.pi * delta**4)

    def shape_factor(self, r, delta):
        """F(r) = 1 - (1 - rho^2) exp(-rho^2): integrate 2 pi s phi_delta(s) from 0 to r with
        t = s^2/delta^2, int_0^{rho^2} (2 - t) e^{-t} dt = [(t - 1) e^{-t}]_0^{rho^2}.
        Evaluated as (1 - exp(-rho^2)) + rho^2 exp(-rho^2) with expm1, which has no
        cancellation near r = 0."""
        rho2 = (np.asarray(r, dtype=float) / delta) ** 2
        return -np.expm1(-rho2) + rho2 * np.exp(-rho2)


class CompactBlob(Blob):
    """Compactly supported polynomial blob, Fovargue 2012 eq. 2.30.

    phi_delta(r) = (6/(pi delta^2)) (1 - rho^2)^5 for rho = r/delta <= 1, zero beyond. Unit mass,
    C^4 at the edge of its support, nonzero second moment (delta^2/7), so formally second order.
    Fovargue ran it with delta = 3h.
    """

    name = "compact"
    compact = True

    def phi(self, r, delta):
        rho2 = (np.asarray(r, dtype=float) / delta) ** 2
        w = np.clip(1.0 - rho2, 0.0, None)
        return 6.0 * w**5 / (np.pi * delta**2)

    def dphi_over_r(self, r, delta):
        """phi_delta'(r)/r = -(60/(pi delta^4)) (1 - rho^2)^4 inside the support, zero outside."""
        rho2 = (np.asarray(r, dtype=float) / delta) ** 2
        w = np.clip(1.0 - rho2, 0.0, None)
        return -60.0 * w**4 / (np.pi * delta**4)

    def shape_factor(self, r, delta):
        """F(r) = 1 - (1 - rho^2)^6 for rho <= 1 and 1 beyond: with w = 1 - s^2/delta^2,
        2 pi int_0^r s phi_delta ds = -6 int_1^{1 - rho^2} w^5 dw (Fovargue 2012 eq. 9.4).
        Evaluated as -expm1(6 log1p(-rho^2)), which has no cancellation near r = 0."""
        rho2 = (np.asarray(r, dtype=float) / delta) ** 2
        inside = rho2 < 1.0
        F = -np.expm1(6.0 * np.log1p(-np.where(inside, rho2, 0.0)))
        return np.where(inside, F, 1.0)


_BLOBS: dict[str, type[Blob]] = {"gaussian4": Gaussian4Blob, "compact": CompactBlob}


def get_blob(name: str) -> Blob:
    """Return the blob called ``name``: "gaussian4" (Cortez & Minion eq. 15) or "compact"
    (Fovargue eq. 2.30)."""
    try:
        return _BLOBS[name]()
    except KeyError:
        raise ValueError(f"unknown blob {name!r}; choose one of {sorted(_BLOBS)}") from None


# ---------------------------------------------------------------------------------------------
# Bicubic Lagrange weights
# ---------------------------------------------------------------------------------------------
def _cubic_lagrange_weights(t: np.ndarray) -> np.ndarray:
    """Cubic Lagrange basis on the nodes -1, 0, 1, 2 evaluated at t in [0, 1); shape t.shape + (4,).

    L_{-1} = -t (t-1) (t-2)/6, L_0 = (t+1)(t-1)(t-2)/2, L_1 = -(t+1) t (t-2)/2, L_2 = (t+1) t (t-1)/6.
    They sum to one and reproduce every cubic polynomial exactly.
    """
    t = np.asarray(t, dtype=float)
    tm1, tp1, tm2 = t - 1.0, t + 1.0, t - 2.0
    return np.stack(
        [-t * tm1 * tm2 / 6.0, tp1 * tm1 * tm2 / 2.0, -tp1 * t * tm2 / 2.0, tp1 * t * tm1 / 6.0],
        axis=-1,
    )


def _cubic_lagrange_derivative_weights(t: np.ndarray) -> np.ndarray:
    """d/dt of the weights of ``_cubic_lagrange_weights``; shape t.shape + (4,). They sum to zero.

    L_{-1}' = -(3t^2 - 6t + 2)/6, L_0' = (3t^2 - 4t - 1)/2, L_1' = -(3t^2 - 2t - 2)/2,
    L_2' = (3t^2 - 1)/6.
    """
    t = np.asarray(t, dtype=float)
    t2 = t * t
    return np.stack(
        [-(3.0 * t2 - 6.0 * t + 2.0) / 6.0, (3.0 * t2 - 4.0 * t - 1.0) / 2.0,
         -(3.0 * t2 - 2.0 * t - 2.0) / 2.0, (3.0 * t2 - 1.0) / 6.0],
        axis=-1,
    )


# ---------------------------------------------------------------------------------------------
# Blob projection on the periodic collocated grid
# ---------------------------------------------------------------------------------------------
class BlobProjection:
    """Blob projection coupling on a doubly periodic collocated grid (docs/numerics.md).

    Parameters
    ----------
    grid : Grid
        Doubly periodic, dx == dy (the periodic FFT back end's grid).
    blob : Blob
        The cutoff function, from ``get_blob``.
    delta : float
        Blob scale in metres (Cortez & Minion use 4h or 2h; Fovargue 3h; delta > h is required to
        regularize over several nodes, and delta > ds to keep the boundary error bounded).
    patch_half_width_cells : int, optional
        Number of nodes on each side of the node nearest to a point on which that point's curl
        is evaluated, so each patch is a (2w+1) x (2w+1) block of nodes. The default makes the
        patch contain every node within a radius R of the point, w = ceil(R/h + 1/2), with
        R = delta for the compact blob (its exact support) and R = 3 delta for the Gaussian
        blob. The +1/2 is because the nearest node can be h/2 away from the point. For the
        Gaussian we chose R = 3 delta to keep the induced far field accurate: R = 3 delta
        leaves 8 exp(-9) = 0.1 % of the blob mass outside the patch and gives a 0.3 % error at
        r = 8 delta (R = 4 delta: 0.05-0.2 %; R = 2 delta: 8-12 %, independent of h), for
        about twice the patch work of R = 2 delta. Pass the value explicitly to trade accuracy
        for speed.
    laplacian : {"tight", "wide"}
        Which discrete Laplacian the stream function solve -L psi = curl inverts (and the
        pressure solve L p_F = div of ``force_pressure``). "wide" is D0_x^2 + D0_y^2, symbol
        -(s1^2 + s2^2)/h^2 with s = sin(2 pi k/N); it is the one behind the periodic back end's
        own projection, but it decouples the grid into four subgrids (Peskin & McQueen 1996
        section 14), so the field induced by a blob that is only a few cells wide carries a
        checkerboard whose far-field error is of order one (measured: 160 % at r = 8 delta for
        the compact blob with delta = 3h, 50 % with delta = 4h, 9 % for gaussian4 with
        delta = 2h). "tight" is the five-point Laplacian, symbol
        -(4/h^2)(sin^2(pi k1/Nx) + sin^2(pi k2/Ny)), which couples the subgrids; the same cases
        give 6 %, 1.4 % and 0.7 %, with no checkerboard, and with delta = 8h both blobs match
        the free-space formula at r = 8 delta to better than 1e-3. Either way
        P F~ = (D0_y psi, -D0_x psi) is exactly D0-divergence-free, the total force is exact,
        and the discretization is second order (the two agree to O((h/delta)^2)). The
        difference is only in what the discrete curl D0_x Fy - D0_y Fx returns: the sampled curl
        itself for "wide", a low-pass filtered version of it for "tight". We default to "tight"
        because the default blobs (gaussian4 at 4h, compact at 3h) need it. The same Laplacian
        gives the stream function of the velocity for the "streamfunction" and "adjoint"
        interpolations (``stream_function``); with "wide" that is the least-squares stream
        function of D0, which amplifies the modes near the checkerboard (module docstring).
    interpolation : {"streamfunction", "bicubic", "adjoint"}
        How ``interpolate`` takes the grid velocity to the points: the curl of the bicubic
        interpolant of the discrete stream function (``interpolate_streamfunction``, exactly
        divergence-free, the default), the bicubic interpolant of u and v themselves
        (``interpolate_bicubic``, divergence-free only to O(h^2)), or the adjoint of the
        projection (``interpolate_adjoint``, divergence-free with an exact power identity).
        The module docstring has the measured trade-offs: area, energy, stability of the
        compact blob at low viscosity, cost.
    """

    def __init__(self, grid: Grid, blob: Blob, delta: float,
                 patch_half_width_cells: int | None = None, laplacian: str = "tight",
                 interpolation: str = "streamfunction") -> None:
        if not (grid.periodic_x and grid.periodic_y):
            raise ValueError("the blob projection method needs a doubly periodic grid")
        if not np.isclose(grid.dx, grid.dy, rtol=1e-12, atol=0.0):
            raise ValueError(f"the blob projection method needs dx == dy, got {grid.dx} and {grid.dy}")
        if delta <= 0.0:
            raise ValueError(f"delta must be positive, got {delta}")
        if laplacian not in ("tight", "wide"):
            raise ValueError(f"laplacian must be 'tight' or 'wide', got {laplacian!r}")
        if interpolation not in INTERPOLATIONS:
            raise ValueError(f"interpolation must be one of {INTERPOLATIONS}, got {interpolation!r}")
        self.grid = grid
        self.blob = blob
        self.delta = float(delta)
        self.h = float(grid.dx)
        self.laplacian = laplacian
        self.interpolation = interpolation
        if patch_half_width_cells is None:
            radius = self.delta if blob.compact else 3.0 * self.delta
            patch_half_width_cells = math.ceil(radius / self.h + 0.5)
        w = int(patch_half_width_cells)
        if w < 1:
            raise ValueError("patch_half_width_cells must be at least 1")
        self.patch_half_width_cells = w
        # Patch offsets (both directions), flattened to one axis so a point's patch is a 1-D list.
        d = np.arange(-w, w + 1)
        DI, DJ = np.meshgrid(d, d, indexing="ij")
        self._di = DI.ravel()
        self._dj = DJ.ravel()

        # FFT symbols, the same ones the periodic back end uses (Peskin & McQueen 1996 eqs.
        # 14.46-14.47): D0_x <-> i s1/h, D0_y <-> i s2/h; the wide Laplacian D0_x^2 + D0_y^2
        # <-> -(s1^2 + s2^2)/h^2, zero on the degenerate set Z (both sines vanish: k1 in
        # {0, Nx/2}, k2 in {0, Ny/2}, tested with the index sets, never with sin == 0); the tight
        # five-point Laplacian <-> -(4/h^2)(sin^2(pi k1/Nx) + sin^2(pi k2/Ny)), zero only at k = 0.
        # The inverse is set to zero on the null modes, which drops the box mean of the curl (a
        # periodic curl field has none anyway; the patch truncation leaves a tiny one).
        Nx, Ny = grid.Nx, grid.Ny
        k1, k2 = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing="ij")
        s1 = np.sin(2.0 * np.pi * k1 / Nx)
        s2 = np.sin(2.0 * np.pi * k2 / Ny)
        degenerate = np.isin(k1, (0, Nx // 2)) & np.isin(k2, (0, Ny // 2))
        wide = (s1**2 + s2**2) / self.h**2
        inv_wide = np.zeros_like(wide)
        np.divide(1.0, wide, out=inv_wide, where=~degenerate)
        if laplacian == "wide":
            inv = inv_wide
        else:
            tight = 4.0 * (np.sin(np.pi * k1 / Nx) ** 2 + np.sin(np.pi * k2 / Ny) ** 2) / self.h**2
            inv = np.zeros_like(tight)
            np.divide(1.0, tight, out=inv, where=~((k1 == 0) & (k2 == 0)))
        self._inv_neg_laplacian = inv          # (-L_h)^{-1}, zero on the null modes
        self._D0x = 1j * s1 / self.h
        self._D0y = 1j * s2 / self.h
        # Discrete stream function of a node velocity (``stream_function``), on the rfft2 half
        # spectrum: psi_hat = (-L)^{-1} (D0_x v_hat - D0_y u_hat), the constructor's Laplacian
        # applied to the discrete vorticity. Here the sines are set to exactly 0 at k = N/2 (D0
        # of (-1)^i is 0; sin(pi) is 1.2e-16), so both factors are exactly Hermitian, as D0 is a
        # real operator, and the half spectrum of rfft2 is enough.
        nh = Ny // 2 + 1
        s1e = np.where(k1 == Nx // 2, 0.0, s1)[:, :nh]
        s2e = np.where(k2 == Ny // 2, 0.0, s2)[:, :nh]
        inv_e = inv[:, :nh]
        self._psi_from_u = -1j * s2e / self.h * inv_e             # -D0_y (-L)^{-1}
        self._psi_from_v = 1j * s1e / self.h * inv_e              # D0_x (-L)^{-1}

    # ------------------------------------------------------------------ force to the grid
    def _patch_geometry(self, X: np.ndarray):
        """The patch around each point: wrapped flat node indices i*Ny + j, g = phi_delta'(r)/r
        and the true node-minus-point separations xrel, yrel, all of shape (n, P)."""
        X = self.grid.wrap(np.asarray(X, dtype=float).reshape(-1, 2))
        h, Nx, Ny = self.h, self.grid.Nx, self.grid.Ny
        i0 = np.rint(X[:, 0] / h).astype(np.intp)
        j0 = np.rint(X[:, 1] / h).astype(np.intp)
        I = i0[:, None] + self._di[None, :]                      # (n, P) unwrapped node indices
        J = j0[:, None] + self._dj[None, :]
        xrel = I * h - X[:, 0, None]                             # node minus point, true separation
        yrel = J * h - X[:, 1, None]
        g = self.blob.dphi_over_r(np.hypot(xrel, yrel), self.delta)
        flat = (I % Nx) * Ny + (J % Ny)
        return flat, g, xrel, yrel

    def _patch_terms(self, f: np.ndarray, X: np.ndarray, ds: np.ndarray):
        """``_patch_geometry`` plus the weighted forces ds f_x, ds f_y, shape (n, 1) each: the
        pieces shared by the curl and the divergence of the blob forces."""
        f = np.asarray(f, dtype=float).reshape(-1, 2)
        ds = np.broadcast_to(np.asarray(ds, dtype=float), (f.shape[0],))
        flat, g, xrel, yrel = self._patch_geometry(X)
        return flat, g, xrel, yrel, (ds * f[:, 0])[:, None], (ds * f[:, 1])[:, None]

    def _accumulate(self, flat: np.ndarray, vals: np.ndarray) -> np.ndarray:
        """Scatter-add patch values into an (Nx, Ny) grid (the same accumulation as np.add.at;
        bincount is about twice as fast on numpy 1.26)."""
        Nx, Ny = self.grid.Nx, self.grid.Ny
        return np.bincount(flat.ravel(), weights=vals.ravel(), minlength=Nx * Ny).reshape(Nx, Ny)

    def curl_on_grid(self, f: np.ndarray, X: np.ndarray, ds: np.ndarray) -> np.ndarray:
        """Scalar curl of the regularized force F~ (eq. 13) sampled on the nodes, shape (Nx, Ny).

        For one blob force ds f phi_delta(|x - X|) the curl (d_x F_y - d_y F_x, the sign of
        Cortez & Minion eq. 5) is analytic:
            curl = ds (phi_delta'(r)/r) [ (x - X) f_y - (y - Y) f_x ],   r = |x - X|.
        We evaluate it on the (2w+1)^2 nodes around the node nearest to each point and add the
        patches into the grid with periodic index wrap (section 3.2 of the paper). Node
        coordinates use the unwrapped index, i*h, so the separation is the true one; if a patch is
        wider than the box a node simply receives the contributions of several periodic images,
        which is still the periodic sum.
        """
        flat, g, xrel, yrel, fx, fy = self._patch_terms(f, X, ds)
        return self._accumulate(flat, g * (xrel * fy - yrel * fx))

    def divergence_on_grid(self, f: np.ndarray, X: np.ndarray, ds: np.ndarray) -> np.ndarray:
        """Divergence of the regularized force F~ (eq. 13) sampled on the nodes, shape (Nx, Ny).

        The companion of ``curl_on_grid``, on the same patches: for one blob force
            div = ds (phi_delta'(r)/r) [ (x - X) f_x + (y - Y) f_y ].
        """
        flat, g, xrel, yrel, fx, fy = self._patch_terms(f, X, ds)
        return self._accumulate(flat, g * (xrel * fx + yrel * fy))

    def _project_curl(self, curl: np.ndarray, f: np.ndarray, ds: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray]:
        """(D0_y psi, -D0_x psi) with -L psi = curl, plus the box mean of F~."""
        psi_hat = np.fft.fft2(curl) * self._inv_neg_laplacian
        Fx = np.real(np.fft.ifft2(self._D0y * psi_hat))
        Fy = np.real(np.fft.ifft2(-self._D0x * psi_hat))
        area = self.grid.Lx * self.grid.Ly
        Fx += np.sum(ds * f[:, 0]) / area
        Fy += np.sum(ds * f[:, 1]) / area
        return Fx, Fy

    def _pressure_from_div(self, div: np.ndarray) -> np.ndarray:
        """p with L p = div (L the constructor's Laplacian), zero on the null modes."""
        return np.real(np.fft.ifft2(-np.fft.fft2(div) * self._inv_neg_laplacian))

    def projected_force(self, f: np.ndarray, X: np.ndarray, ds: np.ndarray
                        ) -> tuple[np.ndarray, np.ndarray]:
        """P F~ on the nodes: the divergence-free part of eq. 13, returned as (Fx, Fy), (Nx, Ny).

        Cortez & Minion 2000 section 3.2 with the stream-function projection of their eq. 5:
        solve -L psi = curl F~ (the curl from ``curl_on_grid``) by FFT with the Laplacian chosen
        in the constructor (null modes zeroed), then
            P F~ = (D0_y psi, -D0_x psi),   D0 <-> i sin(2 pi k/N)/h,
        which satisfies D0_x Fx + D0_y Fy = 0 to round-off whatever L is. A stream function
        carries no box mean, and the mean of F~ (which P preserves on a periodic box) is
        sum_k ds_k f_k / (Lx Ly) because every blob has unit mass, so we add it back; then
        sum_ij F_ij h^2 = sum_k ds_k f_k exactly. f is the force density per unit length (N/m),
        ds the quadrature weight (m); the result is a force density (N/m^2).
        """
        f = np.asarray(f, dtype=float).reshape(-1, 2)
        ds = np.broadcast_to(np.asarray(ds, dtype=float), (f.shape[0],))
        return self._project_curl(self.curl_on_grid(f, X, ds), f, ds)

    def force_pressure(self, f: np.ndarray, X: np.ndarray, ds: np.ndarray) -> np.ndarray:
        """p_F on the nodes, shape (Nx, Ny): the pressure of the gradient part of F~ (eq. 13).

        The Helmholtz split F~ = P F~ + grad p_F (Cortez & Minion 2000 eq. 3 applied to the
        force) gives L p_F = div F~. The fluid only receives P F~, so the solver's pressure holds
        no trace of the membrane; p_F is the rest, and solver.p + p_F is the pressure of the
        unprojected problem. A membrane under tension T around a region of curvature radius R
        gives the Laplace jump T/R in p_F. We take div F~ analytically on the same patches as the
        curl (``divergence_on_grid``) and invert the constructor's Laplacian by FFT with the null
        modes zeroed, so p_F has zero mean over the box (and, with "wide", nothing on the three
        checkerboard modes). Units: f in N/m (so F~ in N/m^2), p_F in N/m (2D pressure).

        With "wide" the four subgrids decouple (see the constructor) and a blob only a few cells
        wide leaves them at different levels. On the pressurised circle of tests/test_blob.py
        (R = 0.25, 64^2 and 128^2) the compact blob at 3h gives four subgrid levels 9 % of the
        jump T/R apart inside the circle, and node values that spread over 11-13 % of T/R
        (compact at 4h: 2 % and 3-4 %; gaussian4 at 4h: subgrid levels within 2.3e-4 of T/R);
        the mean jump is still within 1.1e-3 of T/R. With "tight" the compact blob's spread is
        0.2-0.3 %. Use "tight" for the pressure you report.
        """
        return self._pressure_from_div(self.divergence_on_grid(f, X, ds))

    def projected_force_and_pressure(self, f: np.ndarray, X: np.ndarray, ds: np.ndarray
                                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(Fx, Fy, p_F) = ``projected_force`` and ``force_pressure`` together, with the blob
        kernel evaluated on the patches once (the patch evaluation is the expensive part)."""
        f = np.asarray(f, dtype=float).reshape(-1, 2)
        ds = np.broadcast_to(np.asarray(ds, dtype=float), (f.shape[0],))
        flat, g, xrel, yrel, fx, fy = self._patch_terms(f, X, ds)
        Fx, Fy = self._project_curl(self._accumulate(flat, g * (xrel * fy - yrel * fx)), f, ds)
        p_F = self._pressure_from_div(self._accumulate(flat, g * (xrel * fx + yrel * fy)))
        return Fx, Fy, p_F

    # ------------------------------------------------------------------ analytic reference
    def analytic_projected_force(self, f: np.ndarray, X: np.ndarray, ds: np.ndarray,
                                 points: np.ndarray) -> np.ndarray:
        """Eq. 14 of Cortez & Minion 2000 evaluated directly at ``points`` (M, 2) -> (M, 2).

        P F~(x) = sum_k ds_k { (1/2) f_k phi_delta(r) + (1/(2 pi)) [f_k - 2 (f_k.x^) x^]
                               (r F'(r) - 2 F(r)) / (2 r^2) },  r = |x - X_k|, x^ = (x - X_k)/r,
        computed as sum_k ds_k [p1(r) f_k - p2(r) (f_k.x^) x^] with the kernels of
        ``Blob.kernels`` (Fovargue 2012 eq. 2.34). This is the free-space field: plain
        separations, no periodic images. Cost O(M N); meant for tests and diagnostics, never for
        the whole grid (the paper's section 3.2 explains why).

        To compare with ``projected_force`` on a square periodic box, add half the box mean,
        sum_k ds_k f_k / (2 Lx Ly), to this field. The free-space dipole field integrates to
        only half the force over any disc around the point (the other half lives in the
        conditionally convergent 1/r^2 tail, the 2D Lorentz-cavity factor), while the periodic
        projection keeps the full mean; the remaining difference near the point is the image
        field, O(r^2/L^4), harmonic and even about the point.
        """
        f = np.asarray(f, dtype=float).reshape(-1, 2)
        X = np.asarray(X, dtype=float).reshape(-1, 2)
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        n = X.shape[0]
        ds = np.broadcast_to(np.asarray(ds, dtype=float), (n,))
        dx = pts[:, 0, None] - X[None, :, 0]                      # (M, n)
        dy = pts[:, 1, None] - X[None, :, 1]
        r = np.hypot(dx, dy)
        p1, p2 = self.blob.kernels(r, self.delta)
        r_safe = np.where(r == 0.0, 1.0, r)
        ex = np.where(r == 0.0, 0.0, dx / r_safe)
        ey = np.where(r == 0.0, 0.0, dy / r_safe)
        fx = (ds * f[:, 0])[None, :]
        fy = (ds * f[:, 1])[None, :]
        fdotx = fx * ex + fy * ey
        out = np.empty((pts.shape[0], 2))
        out[:, 0] = np.sum(p1 * fx - p2 * fdotx * ex, axis=1)
        out[:, 1] = np.sum(p1 * fy - p2 * fdotx * ey, axis=1)
        return out

    # ------------------------------------------------------------------ velocity to the points
    def interpolate(self, u: np.ndarray, v: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Velocity of the node fields u, v at the points X (N, 2) -> (N, 2).

        ``interpolate_streamfunction``, ``interpolate_bicubic`` or ``interpolate_adjoint``, as
        chosen by the constructor's ``interpolation``.
        """
        if self.interpolation == "streamfunction":
            return self.interpolate_streamfunction(u, v, X)
        if self.interpolation == "adjoint":
            return self.interpolate_adjoint(u, v, X)
        return self.interpolate_bicubic(u, v, X)

    def _stencil(self, X: np.ndarray):
        """Indices (N, 4) of the nodes i0-1..i0+2, j0-1..j0+2 around each point, i0 = floor(X/h)
        with periodic wrap, and the fractional offsets tx, ty in [0, 1)."""
        X = self.grid.wrap(np.asarray(X, dtype=float).reshape(-1, 2))
        h, Nx, Ny = self.h, self.grid.Nx, self.grid.Ny
        sx = X[:, 0] / h
        sy = X[:, 1] / h
        i0 = np.floor(sx).astype(np.intp)
        j0 = np.floor(sy).astype(np.intp)
        offs = np.arange(-1, 3)
        I = (i0[:, None] + offs[None, :]) % Nx                   # (N, 4)
        J = (j0[:, None] + offs[None, :]) % Ny
        return I, J, sx - i0, sy - j0

    def interpolate_bicubic(self, u: np.ndarray, v: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Bicubic Lagrange interpolation of the node fields u, v at the points X (N, 2) -> (N, 2).

        A 4 x 4 patch of nodes around each point and a bicubic polynomial interpolant,
        fourth-order accurate: the tensor-product cubic Lagrange form on the nodes i0-1..i0+2
        with i0 = floor(X/h) (and the same in y), periodic index wrap, weights from
        ``_cubic_lagrange_weights``:
            U(X) = sum_{i,j} u[i, j] L_i(X) L_j(Y).
        It reproduces u, v exactly at the nodes and every cubic exactly. It interpolates u and
        v directly, so for a D0-divergence-free u the interpolated field is divergence-free
        only to O(h^2), and the area inside a closed membrane can drift slowly;
        ``interpolate_streamfunction`` is exactly divergence-free (see the module docstring).
        """
        I, J, tx, ty = self._stencil(X)
        wx = _cubic_lagrange_weights(tx)                          # (N, 4)
        wy = _cubic_lagrange_weights(ty)
        out = np.empty((I.shape[0], 2))
        for c, field in enumerate((u, v)):
            field = np.asarray(field, dtype=float)
            patch = field[I[:, :, None], J[:, None, :]]           # (N, 4, 4)
            out[:, c] = np.einsum("nij,ni,nj->n", patch, wx, wy)
        return out

    def stream_function(self, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, float, float]:
        """The discrete stream function of the node velocity: (psi, mean u, mean v).

        psi (Nx, Ny) solves -L psi = omega for the discrete vorticity omega = D0_x v - D0_y u,
        with L the constructor's Laplacian (null modes zeroed), mode by mode by FFT:
            psi_hat = (D0_x v_hat - D0_y u_hat) / (-L symbol).
        A discrete gradient (D0_x phi, D0_y phi) and the three checkerboard modes have no
        vorticity, so they are dropped; the box mean is returned separately. This is the psi
        that ``interpolate_streamfunction`` and ``interpolate_adjoint`` take to the points.

        With "wide", -L = |D0_x|^2 + |D0_y|^2 and psi is the least-squares solution of
        u = D0_y psi, v = -D0_x psi: it gives back a D0-divergence-free field exactly. But for
        a mode D0 barely sees (close to the checkerboard in one direction) that psi is large for
        its grid speed, and the exact derivative of its interpolant amplifies the mode: 34 times
        the grid speed at k1 = N/2 - 1, k2 = 1 on 128^2, which made the compact blob blow up at
        low viscosity (module docstring). With "tight" (the default), -L is the five-point
        Laplacian, which is not |D0|^2: (D0_y psi, -D0_x psi) is the input only to O(h^2) on a
        smooth field, and the near-checkerboard modes are damped instead (gain 0.04 for the same
        mode, 0.77 at k1 = N/4).
        """
        Nx, Ny = self.grid.Nx, self.grid.Ny
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        psi_hat = self._psi_from_u * np.fft.rfft2(u) + self._psi_from_v * np.fft.rfft2(v)
        return np.fft.irfft2(psi_hat, s=(Nx, Ny)), float(np.mean(u)), float(np.mean(v))

    def interpolate_streamfunction(self, u: np.ndarray, v: np.ndarray, X: np.ndarray
                                   ) -> np.ndarray:
        """Exactly divergence-free velocity at the points X (N, 2) -> (N, 2).

        With psi the discrete stream function of (u, v) (``stream_function``) and I psi its
        bicubic Lagrange interpolant on the same 4 x 4 stencil as ``interpolate_bicubic``,
            U(X) = ( d_y I psi, -d_x I psi ) + (mean u, mean v),
        with the derivatives taken exactly (``_cubic_lagrange_derivative_weights``). Inside a
        cell U is the curl of a polynomial, so d_x U_x + d_y U_y = 0 there. On a cell line
        x = x_i both neighbouring stencils reduce to the same 1D cubic in y through the nodes of
        that line, so I psi and its tangential derivative, which is the normal velocity, are
        continuous across it (the tangential velocity is not). The normal flux is therefore
        continuous and U is divergence-free everywhere: the flux through any closed curve is
        zero. U approximates the smooth velocity behind the grid values to second order (the
        O(h^2) difference between L and D0^2 and between D0 and the exact derivative); it does
        not reproduce u exactly at the nodes. Modes close to the checkerboard are damped with the
        default "tight" Laplacian and amplified with "wide" (see ``stream_function``).
        """
        psi, um, vm = self.stream_function(u, v)
        I, J, tx, ty = self._stencil(X)
        h = self.h
        wx = _cubic_lagrange_weights(tx)                          # (N, 4)
        wy = _cubic_lagrange_weights(ty)
        dwx = _cubic_lagrange_derivative_weights(tx) / h
        dwy = _cubic_lagrange_derivative_weights(ty) / h
        P = psi[I[:, :, None], J[:, None, :]]                     # (N, 4, 4)
        out = np.empty((I.shape[0], 2))
        out[:, 0] = np.einsum("ni,ni->n", wx, np.einsum("nij,nj->ni", P, dwy)) + um
        out[:, 1] = -np.einsum("ni,ni->n", dwx, np.einsum("nij,nj->ni", P, wy)) + vm
        return out

    def interpolate_adjoint(self, u: np.ndarray, v: np.ndarray, X: np.ndarray) -> np.ndarray:
        """The adjoint of ``projected_force``: velocity at the points X (N, 2) -> (N, 2).

        With omega = D0_x v - D0_y u, psi = (-L)^{-1} omega (``stream_function``: the
        constructor's Laplacian, the null modes zeroed) and the same patches as ``curl_on_grid``,
            U(X) = curl_X Psi(X) + (mean u, mean v),   Psi(X) = sum_ij h^2 psi_ij phi_delta(x_ij - X),
            U_x = -sum_ij h^2 psi_ij g(r) (y_j - Y),   U_y = sum_ij h^2 psi_ij g(r) (x_i - X),
        with g = phi_delta'(r)/r. Summation by parts and the symmetry of (-L)^{-1} give
            sum_ij h^2 F_ij . u_ij = sum_k ds_k f_k . U_k
        for every u and every set of forces, F = ``projected_force(f, X, ds)``: the power the
        fluid receives is the power the structure gives, as for the delta function with the
        adjoint kernel interpolation. U is also the curl of a smooth function of X, so it has
        no divergence; for the compact blob exactly, for the Gaussian blob up to the tail its
        patch leaves out (0.1 % of the blob mass at the default patch). The price: U is a blob
        average of the velocity, so it is smoother than the point value and costs a patch
        evaluation per call, about as much as ``projected_force``.
        """
        psi, um, vm = self.stream_function(u, v)
        flat, g, xrel, yrel = self._patch_geometry(X)
        Pg = psi.ravel()[flat] * g                                # (n, P)
        h2 = self.h**2
        out = np.empty((flat.shape[0], 2))
        out[:, 0] = -h2 * np.sum(Pg * yrel, axis=1) + um
        out[:, 1] = h2 * np.sum(Pg * xrel, axis=1) + vm
        return out
