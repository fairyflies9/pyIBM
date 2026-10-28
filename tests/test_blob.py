"""Tests for the blob projection coupling (pyibm/coupling/blob.py; docs/numerics.md).

Every check here is independent of the implementation: closed forms typed from the papers,
numerical quadrature, discrete identities, translation equivariance, and convergence rates.
"""

import numpy as np
import pytest
from scipy import integrate

from pyibm.grid import Grid
from pyibm.coupling.blob import BlobProjection, CompactBlob, Gaussian4Blob, get_blob


# ------------------------------------------------------------------------------------ helpers
def D0x(F, h):
    """Centred difference along x, the operator with symbol i sin(2 pi k1/Nx)/h."""
    return (np.roll(F, -1, axis=0) - np.roll(F, 1, axis=0)) / (2.0 * h)


def D0y(F, h):
    return (np.roll(F, -1, axis=1) - np.roll(F, 1, axis=1)) / (2.0 * h)


def phi_typed(name, r, delta):
    """The blobs as printed: Cortez & Minion 2000 eq. 15 and Fovargue 2012 eq. 2.30."""
    rho2 = (r / delta) ** 2
    if name == "gaussian4":
        return (2.0 - rho2) * np.exp(-rho2) / (np.pi * delta**2)
    return np.where(rho2 <= 1.0, 6.0 * np.clip(1.0 - rho2, 0.0, None) ** 5 / (np.pi * delta**2), 0.0)


def shape_factor_typed(name, r, delta):
    """Closed forms: 1 - (1 - rho^2) exp(-rho^2) and 1 - (1 - rho^2)^6 (one beyond the support)."""
    rho2 = (r / delta) ** 2
    if name == "gaussian4":
        return 1.0 - (1.0 - rho2) * np.exp(-rho2)
    return 1.0 - np.clip(1.0 - rho2, 0.0, None) ** 6


def eq14_direct(name, f, X, ds, delta, points):
    """Cortez & Minion 2000 eq. 14 exactly as printed, with F' = 2 pi r phi_delta.

    P F~(x) = sum_k ds_k { (1/2) f_k phi_delta(r) + (1/(2 pi)) [f_k - 2 (f_k.x^) x^]
                           (r F'(r) - 2 F(r)) / (2 r^2) }.
    Points coinciding with a blob centre are not handled (none are used).
    """
    out = np.zeros((len(points), 2))
    for fk, Xk, dsk in zip(f, X, ds):
        d = points - Xk
        r = np.hypot(d[:, 0], d[:, 1])
        xhat = d / r[:, None]
        phi = phi_typed(name, r, delta)
        F = shape_factor_typed(name, r, delta)
        Fp = 2.0 * np.pi * r * phi
        coef = (r * Fp - 2.0 * F) / (2.0 * r**2)
        fdot = xhat @ fk
        out += dsk * (0.5 * phi[:, None] * fk + coef[:, None] * (fk - 2.0 * fdot[:, None] * xhat) / (2.0 * np.pi))
    return out


def fovargue_kernels(r, delta):
    """p1, p2 for the compact blob as polynomials in rho: the kernels of Fovargue 2012
    eq. 2.34 for the blob of eq. 2.30, multiplied out by hand (no cancellation near r = 0)."""
    rho = r / delta
    rho2 = rho**2
    inside = rho < 1.0
    p1_in = (6 - 45 * rho2 + 100 * rho2**2 - 105 * rho2**3 + 54 * rho2**4 - 11 * rho2**5) / (2 * np.pi * delta**2)
    p2_in = -rho2 * (15 - 40 * rho2 + 45 * rho2**2 - 24 * rho2**3 + 5 * rho2**4) / (np.pi * delta**2)
    with np.errstate(divide="ignore"):
        p1_out = -1.0 / (2.0 * np.pi * r**2)
        p2_out = -1.0 / (np.pi * r**2)
    return np.where(inside, p1_in, p1_out), np.where(inside, p2_in, p2_out)


def gaussian4_kernel_series(rho):
    """p1, p2 of the gaussian4 blob in units of 1/(pi delta^2) near r = 0, from the Taylor
    series of eq. 15 and its shape factor (truncation O(rho^8))."""
    r2 = rho**2
    p1 = 1.0 - 9.0 * r2 / 4.0 + 5.0 * r2**2 / 3.0 - 35.0 * r2**3 / 48.0
    p2 = -3.0 * r2 / 2.0 + 4.0 * r2**2 / 3.0 - 5.0 * r2**3 / 8.0
    return p1, p2


def random_divfree_velocity(N, rng, mean=(0.0, 0.0)):
    """A rough D0-divergence-free node field u = D0_y psi, v = -D0_x psi from a random psi,
    plus a box mean."""
    grid = Grid.periodic_box(1.0, 1.0, N, N)
    h = grid.dx
    psi = rng.normal(size=(N, N)) * h
    return grid, D0y(psi, h) + mean[0], -D0x(psi, h) + mean[1]


def smooth_psi(x, y):
    """A smooth periodic stream function on the unit box and its two derivatives."""
    a, b = 2 * np.pi * x, 2 * np.pi * y
    psi = np.sin(a) * np.cos(2 * b) + 0.3 * np.cos(3 * a + b)
    psi_x = 2 * np.pi * (np.cos(a) * np.cos(2 * b) - 0.9 * np.sin(3 * a + b))
    psi_y = 2 * np.pi * (-2 * np.sin(a) * np.sin(2 * b) - 0.3 * np.sin(3 * a + b))
    return psi, psi_x, psi_y


def fd_divergence(fun, P, e):
    """d_x U_x + d_y U_y at the points P by the fourth-order central difference with step e,
    exact for the cubic in x of U_x and the cubic in y of U_y inside one cell."""
    ex, ey = np.array([e, 0.0]), np.array([0.0, e])
    dU = (-fun(P + 2 * ex)[:, 0] + 8 * fun(P + ex)[:, 0] - 8 * fun(P - ex)[:, 0] + fun(P - 2 * ex)[:, 0]) / (12 * e)
    dV = (-fun(P + 2 * ey)[:, 1] + 8 * fun(P + ey)[:, 1] - 8 * fun(P - ey)[:, 1] + fun(P - 2 * ey)[:, 1]) / (12 * e)
    return dU + dV, max(np.abs(dU).max(), np.abs(dV).max())


# ------------------------------------------------------------------------------------ blobs
@pytest.mark.parametrize("name", ["gaussian4", "compact"])
def test_blob_formulas_as_printed(name):
    blob = get_blob(name)
    delta = 0.37
    r = np.linspace(0.0, 3.0 * delta, 301)
    assert np.allclose(blob.phi(r, delta), phi_typed(name, r, delta), rtol=1e-14, atol=1e-14)
    assert np.allclose(blob.shape_factor(r, delta), shape_factor_typed(name, r, delta), rtol=1e-14, atol=1e-14)
    assert blob.name == name and blob.compact == (name == "compact")


@pytest.mark.parametrize("name", ["gaussian4", "compact"])
def test_blob_unit_mass_by_quadrature(name):
    blob = get_blob(name)
    delta = 0.21
    upper = delta if blob.compact else np.inf
    mass, err = integrate.quad(lambda s: 2.0 * np.pi * s * blob.phi(s, delta), 0.0, upper, epsabs=1e-13, epsrel=1e-13)
    assert abs(mass - 1.0) < 1e-10


@pytest.mark.parametrize("name", ["gaussian4", "compact"])
def test_shape_factor_is_the_mass_inside_r(name):
    blob = get_blob(name)
    delta = 0.21
    for r in (0.05 * delta, 0.4 * delta, 0.9 * delta, 1.3 * delta, 2.5 * delta):
        num, _ = integrate.quad(lambda s: 2.0 * np.pi * s * blob.phi(s, delta), 0.0, r, epsabs=1e-13, epsrel=1e-13)
        assert abs(blob.shape_factor(r, delta) - num) < 1e-10


@pytest.mark.parametrize("name", ["gaussian4", "compact"])
@pytest.mark.parametrize("delta", [0.05, 0.3, 1.7])
def test_dphi_over_r_matches_finite_difference(name, delta):
    blob = get_blob(name)
    r = np.array([0.1, 0.25, 0.5, 0.8, 0.95, 1.4]) * delta
    eps = 1e-6 * delta
    fd = (blob.phi(r + eps, delta) - blob.phi(r - eps, delta)) / (2.0 * eps) / r
    assert np.allclose(blob.dphi_over_r(r, delta), fd, rtol=1e-7, atol=1e-7 / delta**4)
    # The limit at r = 0 is finite and equals phi''(0). phi is even, so the symmetric second
    # difference 2 (phi(e) - phi(0))/e^2 has truncation error (e^2/12) phi''''(0)/phi''(0):
    # 6.7e-7 (gaussian4) and 2.0e-6 (compact) relative at e = 1e-3 delta, with round-off near
    # 1e-10. (With e = 1e-6 delta the round-off alone was about 1.5e-4.)
    eps0 = 1e-3 * delta
    fd0 = 2.0 * (blob.phi(eps0, delta) - blob.phi(0.0, delta)) / eps0**2
    assert np.isclose(blob.dphi_over_r(0.0, delta), fd0, rtol=1e-5)


@pytest.mark.parametrize("name", ["gaussian4", "compact"])
def test_kernels_accurate_near_the_origin(name):
    """p1 and p2 keep full accuracy (relative to p1(0)) as r -> 0. Written as
    phi - F/(pi r^2) with F = 1 - (1 - rho^2) exp(-rho^2) they lost it all: 22 % at
    rho = 1e-8, and for rho below about 1e-9 F rounds to 0 so p1 = p2 = phi(0)."""
    blob = get_blob(name)
    delta = 0.3
    rho = np.array([1e-200, 1e-15, 1e-12, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3])
    p1, p2 = blob.kernels(rho * delta, delta)
    if name == "gaussian4":
        q1, q2 = gaussian4_kernel_series(rho)
        q1, q2 = q1 / (np.pi * delta**2), q2 / (np.pi * delta**2)
    else:
        q1, q2 = fovargue_kernels(rho * delta, delta)
    p10 = 0.5 * blob.phi(0.0, delta)
    assert np.all(np.isfinite(p1)) and np.all(np.isfinite(p2))
    assert np.abs(p1 - q1).max() < 1e-13 * p10
    assert np.abs(p2 - q2).max() < 1e-13 * p10
    # the shape factor itself has full relative accuracy: F ~ pi phi(0) r^2 (1 + O(rho^2))
    F = blob.shape_factor(rho[1:] * delta, delta)
    lead = np.pi * blob.phi(0.0, delta) * (rho[1:] * delta) ** 2
    assert np.allclose(F[:6], lead[:6], rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("name", ["gaussian4", "compact"])
def test_analytic_projected_force_continuous_at_a_blob_centre(name):
    """A point at round-off distance from a blob centre (0.1 + 0.2 against 0.3, r = 5.6e-17)
    gets the r = 0 value ds phi(0) f/2, not zero."""
    grid = Grid.periodic_box(1.0, 1.0, 64, 64)
    bp = BlobProjection(grid, get_blob(name), 4.0 * grid.dx)
    f = np.array([[1.0, -0.5]])
    ds = np.array([grid.dx])
    want = 0.5 * bp.blob.phi(0.0, bp.delta) * ds[0] * f[0]
    at_centre = bp.analytic_projected_force(f, np.array([[0.3, 0.3]]), ds, np.array([[0.3, 0.3]]))
    near = bp.analytic_projected_force(f, np.array([[0.1 + 0.2, 0.3]]), ds, np.array([[0.3, 0.3]]))
    assert np.allclose(at_centre[0], want, rtol=1e-14)
    assert np.allclose(near[0], want, rtol=1e-12)


def test_compact_kernels_match_fovargue_polynomials():
    blob = get_blob("compact")
    delta = 0.3
    r = np.concatenate([[0.0], np.logspace(-14, -2, 13) * delta, np.linspace(0.02, 2.0, 200) * delta])
    p1, p2 = blob.kernels(r, delta)
    q1, q2 = fovargue_kernels(r, delta)
    q1[0], q2[0] = 3.0 / (np.pi * delta**2), 0.0            # the values at r = 0
    assert np.allclose(p1, q1, rtol=1e-12, atol=1e-12 / delta**2)
    assert np.allclose(p2, q2, rtol=1e-12, atol=1e-12 / delta**2)


@pytest.mark.parametrize("name", ["gaussian4", "compact"])
def test_kernels_at_origin_and_far_field(name):
    blob = get_blob(name)
    delta = 0.3
    p1, p2 = blob.kernels(np.array([0.0]), delta)
    assert p1[0] == pytest.approx(0.5 * blob.phi(0.0, delta))   # half the blob peak survives P
    assert p2[0] == 0.0
    r = np.array([8.0 * delta])
    p1, p2 = blob.kernels(r, delta)
    assert p1[0] == pytest.approx(-1.0 / (2.0 * np.pi * r[0] ** 2), rel=1e-8)
    assert p2[0] == pytest.approx(-1.0 / (np.pi * r[0] ** 2), rel=1e-8)


def test_get_blob_unknown():
    with pytest.raises(ValueError):
        get_blob("hat")
    assert isinstance(get_blob("gaussian4"), Gaussian4Blob)
    assert isinstance(get_blob("compact"), CompactBlob)


# ------------------------------------------------------------------------------------ analytic field
@pytest.mark.parametrize("name", ["gaussian4", "compact"])
def test_analytic_projected_force_is_eq14(name):
    rng = np.random.default_rng(3)
    grid = Grid.periodic_box(1.0, 1.0, 64, 64)
    delta = 3.0 * grid.dx
    bp = BlobProjection(grid, get_blob(name), delta)
    X = rng.uniform(0.3, 0.7, size=(6, 2))
    f = rng.normal(size=(6, 2))
    ds = rng.uniform(0.5, 1.5, size=6) * grid.dx
    points = rng.uniform(0.0, 1.0, size=(300, 2))
    got = bp.analytic_projected_force(f, X, ds, points)
    want = eq14_direct(name, f, X, ds, delta, points)
    assert np.allclose(got, want, rtol=1e-11, atol=1e-11 * np.abs(want).max())
    # the field is divergence-free away from the blobs: central differences on a fine stencil
    eps = 1e-4 * delta
    far = np.array([[0.1, 0.1], [0.9, 0.2], [0.05, 0.95]])
    for p in far:
        ux = bp.analytic_projected_force(f, X, ds, p + [[eps, 0.0], [-eps, 0.0]])
        uy = bp.analytic_projected_force(f, X, ds, p + [[0.0, eps], [0.0, -eps]])
        div = (ux[0, 0] - ux[1, 0] + uy[0, 1] - uy[1, 1]) / (2.0 * eps)
        assert abs(div) < 1e-6 * np.abs(ux).max() / delta


# ------------------------------------------------------------------------------------ grid field
@pytest.mark.parametrize("name", ["gaussian4", "compact"])
@pytest.mark.parametrize("laplacian", ["tight", "wide"])
def test_projected_force_divergence_free_and_total_force(name, laplacian):
    rng = np.random.default_rng(7)
    grid = Grid.periodic_box(2.0, 1.0, 128, 64)
    h = grid.dx
    bp = BlobProjection(grid, get_blob(name), 3.0 * h, laplacian=laplacian)
    n = 50
    X = rng.uniform(0.0, 1.0, size=(n, 2)) * [grid.Lx, grid.Ly]
    f = rng.normal(size=(n, 2))
    ds = rng.uniform(0.5, 1.5, size=n) * h
    Fx, Fy = bp.projected_force(f, X, ds)
    assert Fx.shape == (128, 64) and Fy.shape == (128, 64)
    div = D0x(Fx, h) + D0y(Fy, h)
    scale = max(np.abs(D0x(Fx, h)).max(), np.abs(D0y(Fy, h)).max())
    assert np.abs(div).max() < 1e-12 * scale
    total = np.array([Fx.sum(), Fy.sum()]) * h * h
    want = (ds[:, None] * f).sum(axis=0)
    assert np.allclose(total, want, rtol=1e-12, atol=1e-13 * np.abs(ds[:, None] * f).sum())


def test_discrete_curl_of_wide_solution_is_the_patch_curl():
    """With the wide Laplacian, D0_x Fy - D0_y Fx returns the accumulated curl exactly except on
    the four null modes (checked in Fourier space)."""
    rng = np.random.default_rng(11)
    grid = Grid.periodic_box(1.0, 1.0, 64, 64)
    h = grid.dx
    bp = BlobProjection(grid, get_blob("gaussian4"), 4.0 * h, laplacian="wide")
    X = rng.uniform(0, 1, size=(5, 2))
    f = rng.normal(size=(5, 2))
    ds = np.full(5, h)
    curl = bp.curl_on_grid(f, X, ds)
    Fx, Fy = bp.projected_force(f, X, ds)
    got = np.fft.fft2(D0x(Fy, h) - D0y(Fx, h))
    want = np.fft.fft2(curl)
    k1, k2 = np.meshgrid(np.arange(64), np.arange(64), indexing="ij")
    null = np.isin(k1, (0, 32)) & np.isin(k2, (0, 32))
    assert np.allclose(got[~null], want[~null], rtol=1e-11, atol=1e-11 * np.abs(want).max())
    assert np.allclose(got[null], 0.0, atol=1e-11 * np.abs(want).max())


@pytest.mark.parametrize("name", ["gaussian4", "compact"])
def test_translation_equivariance_and_periodic_wrap(name):
    """Moving the point by whole cells rolls the grid field: exercises the index wrap across
    all four box edges for the curl patches."""
    grid = Grid.periodic_box(1.0, 1.0, 48, 48)
    h = grid.dx
    bp = BlobProjection(grid, get_blob(name), 3.0 * h)
    X0 = np.array([[0.5 + 0.31 * h, 0.5 - 0.17 * h]])
    f = np.array([[0.8, -0.4]])
    ds = np.array([0.7 * h])
    Fx0, Fy0 = bp.projected_force(f, X0, ds)
    p0 = bp.force_pressure(f, X0, ds)
    for mi, mj in ((23, 0), (-24, 23), (23, -24), (-22, -22)):
        Fx, Fy = bp.projected_force(f, X0 + [[mi * h, mj * h]], ds)
        assert np.allclose(Fx, np.roll(Fx0, (mi, mj), axis=(0, 1)), rtol=1e-11, atol=1e-11 * np.abs(Fx0).max())
        assert np.allclose(Fy, np.roll(Fy0, (mi, mj), axis=(0, 1)), rtol=1e-11, atol=1e-11 * np.abs(Fy0).max())
        p = bp.force_pressure(f, X0 + [[mi * h, mj * h]], ds)
        assert np.allclose(p, np.roll(p0, (mi, mj), axis=(0, 1)), rtol=1e-11, atol=1e-11 * np.abs(p0).max())


def test_patch_default_and_support_coverage():
    grid = Grid.periodic_box(1.0, 1.0, 64, 64)
    h = grid.dx
    assert BlobProjection(grid, get_blob("compact"), 3.0 * h).patch_half_width_cells == 4      # ceil(3 + 1/2)
    assert BlobProjection(grid, get_blob("gaussian4"), 4.0 * h).patch_half_width_cells == 13   # ceil(12 + 1/2)
    assert BlobProjection(grid, get_blob("gaussian4"), 4.0 * h, 12).patch_half_width_cells == 12
    # the compact blob's default patch holds its whole support: a wider patch changes nothing
    X = np.array([[0.5 + 0.49 * h, 0.5 - 0.49 * h]])
    f = np.array([[1.0, 2.0]])
    ds = np.array([h])
    c1 = BlobProjection(grid, get_blob("compact"), 3.0 * h).curl_on_grid(f, X, ds)
    c2 = BlobProjection(grid, get_blob("compact"), 3.0 * h, 10).curl_on_grid(f, X, ds)
    assert np.array_equal(c1, c2)
    with pytest.raises(ValueError):
        BlobProjection(Grid(1.0, 1.0, 16, 16, periodic_x=False), get_blob("compact"), 3.0 * h)
    with pytest.raises(ValueError):
        BlobProjection(Grid(1.0, 2.0, 16, 16), get_blob("compact"), 3.0 * h)       # dx != dy
    with pytest.raises(ValueError):
        BlobProjection(grid, get_blob("compact"), 3.0 * h, laplacian="spectral")
    with pytest.raises(ValueError):
        BlobProjection(grid, get_blob("compact"), 3.0 * h, interpolation="velocity")
    assert BlobProjection(grid, get_blob("compact"), 3.0 * h).interpolation == "streamfunction"


@pytest.mark.parametrize("name", ["gaussian4", "compact"])
def test_grid_field_matches_analytic_far_field(name):
    """One point force on a fine grid: at r = 8 delta the grid field equals the free-space
    eq. 14 to 1e-3 relative. The periodic field carries the full box mean while the free-space
    dipole field integrates to half the force over any disc, so half the mean is added to the
    analytic field (see analytic_projected_force); the leftover image field is O((r/L)^4)."""
    N = 1024
    grid = Grid.periodic_box(1.0, 1.0, N, N)
    h = grid.dx
    delta = 8.0 * h
    patch = None if name == "compact" else int(np.ceil(4.0 * delta / h + 0.5))   # Gaussian tail below 2e-6
    bp = BlobProjection(grid, get_blob(name), delta, patch)
    X = np.array([[0.5 + 0.37 * h, 0.5 - 0.21 * h]])
    f = np.array([[0.6, 1.0]])
    ds = np.array([0.01])
    Fx, Fy = bp.projected_force(f, X, ds)
    XX, YY = grid.meshgrid("nodes")
    r = np.hypot(XX - X[0, 0], YY - X[0, 1])
    ring = np.abs(r - 8.0 * delta) < 0.5 * h
    pts = np.stack([XX[ring], YY[ring]], axis=1)
    want = bp.analytic_projected_force(f, X, ds, pts) + (ds[:, None] * f).sum(axis=0) / (2.0 * grid.Lx * grid.Ly)
    got = np.stack([Fx[ring], Fy[ring]], axis=1)
    assert np.abs(got - want).max() < 1e-3 * np.abs(want).max()        # measured 4.9e-4, 7.3e-4
    if name == "gaussian4":
        # the default patch (radius 3 delta) truncates the Gaussian tail: 0.3 % in the far field
        Gx, Gy = BlobProjection(grid, get_blob(name), delta).projected_force(f, X, ds)
        got_default = np.stack([Gx[ring], Gy[ring]], axis=1)
        assert np.abs(got_default - want).max() < 5e-3 * np.abs(want).max()
    # and the near field within r < L/8, relative to the peak: second-order discretization error
    disc = r < 0.125
    pts = np.stack([XX[disc], YY[disc]], axis=1)
    want = bp.analytic_projected_force(f, X, ds, pts) + (ds[:, None] * f).sum(axis=0) / (2.0 * grid.Lx * grid.Ly)
    got = np.stack([Fx[disc], Fy[disc]], axis=1)
    assert np.abs(got - want).max() < 2.5e-2 * np.abs(want).max()


def test_curl_route_converges_to_sampled_projection():
    """Sampling F~ on the nodes and projecting it with the back end's symbols (the other route
    to P F~) must agree with the curl route to O((h/delta)^2) with the wide Laplacian:
    they are the same discrete problem up to how the curl is formed."""
    name = "gaussian4"
    errs = []
    for N, dfac in ((128, 4.0), (256, 8.0)):
        grid = Grid.periodic_box(1.0, 1.0, N, N)
        h = grid.dx
        delta = dfac * h
        X = np.array([[0.5 + 0.37 * h, 0.5 - 0.21 * h]])
        f = np.array([[0.6, 1.0]])
        ds = np.array([1.0])
        bp = BlobProjection(grid, get_blob(name), delta, int(np.ceil(4.0 * delta / h + 0.5)), laplacian="wide")
        Fx, Fy = bp.projected_force(f, X, ds)
        XX, YY = grid.meshgrid("nodes")
        phi = phi_typed(name, np.hypot(XX - X[0, 0], YY - X[0, 1]), delta)
        k1, k2 = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
        s1, s2 = np.sin(2 * np.pi * k1 / N), np.sin(2 * np.pi * k2 / N)
        Z = np.isin(k1, (0, N // 2)) & np.isin(k2, (0, N // 2))
        den = np.where(Z, 1.0, s1**2 + s2**2)
        wu, wv = np.fft.fft2(f[0, 0] * phi), np.fft.fft2(f[0, 1] * phi)
        q = np.where(Z, 0.0, (s1 * wu + s2 * wv) / den)
        Px, Py = np.real(np.fft.ifft2(wu - s1 * q)), np.real(np.fft.ifft2(wv - s2 * q))
        errs.append(max(np.abs(Fx - Px).max(), np.abs(Fy - Py).max()) / np.abs(Py).max())
    assert errs[0] < 0.06 and errs[1] < 0.015
    assert errs[0] / errs[1] > 3.0                      # second order in h/delta


def test_tight_and_wide_agree_for_a_well_resolved_blob():
    grid = Grid.periodic_box(1.0, 1.0, 256, 256)
    h = grid.dx
    X = np.array([[0.5, 0.5]])
    f = np.array([[0.6, 1.0]])
    ds = np.array([1.0])
    fields = {}
    for lap in ("tight", "wide"):
        bp = BlobProjection(grid, get_blob("gaussian4"), 8.0 * h, laplacian=lap)
        fields[lap] = np.stack(bp.projected_force(f, X, ds))
    scale = np.abs(fields["wide"]).max()
    assert np.abs(fields["tight"] - fields["wide"]).max() < 2.5e-2 * scale   # measured 1.8e-2


# ------------------------------------------------------------------------------------ force pressure
def pressurised_circle(N, R=0.25, T=1.0, per_cell=4):
    """A circle of radius R with the force of a membrane under tension T, f = -(T/R) n per
    unit length, exact arc-length weights. The Laplace jump is T/R. The centre is off the
    nodes so no symmetry of the grid helps."""
    h = 1.0 / N
    n = int(round(per_cell * 2 * np.pi * R / h))
    th = 2 * np.pi * np.arange(n) / n
    nout = np.column_stack([np.cos(th), np.sin(th)])
    c = np.array([0.5 + 0.3 * h, 0.5 - 0.2 * h])
    return c, c + R * nout, -(T / R) * nout, np.full(n, 2 * np.pi * R / n)


@pytest.mark.parametrize("name,dfac,tol64,tol128", [("gaussian4", 4.0, 1e-2, 1e-3),
                                                     ("compact", 3.0, 2e-3, 2e-4)])
def test_force_pressure_gives_the_laplace_jump(name, dfac, tol64, tol128):
    """p_F of a pressurised circle (R = 0.25, T = 1) is T/R higher inside than outside: mean
    over r < R/2 minus the mean over R + 0.1 < r < 0.45. Measured relative jump error, 64^2 and
    128^2: gaussian4 (4h) 6.7e-3, 4.7e-4; compact (3h) -7.7e-7, -2.1e-5. p_F has zero mean, and
    P F~ stays below 3.1e-5 of the gradient of p_F: the force of a uniform tension is a
    gradient."""
    R, T = 0.25, 1.0
    for N, tol in ((64, tol64), (128, tol128)):
        grid = Grid.periodic_box(1.0, 1.0, N, N)
        c, X, f, ds = pressurised_circle(N, R, T)
        bp = BlobProjection(grid, get_blob(name), dfac * grid.dx)
        p = bp.force_pressure(f, X, ds)
        assert p.shape == (N, N)
        XX, YY = grid.meshgrid("nodes")
        r = np.hypot(XX - c[0], YY - c[1])
        jump = p[r < R / 2].mean() - p[(r > R + 0.1) & (r < 0.45)].mean()
        assert abs(jump - T / R) < tol * T / R
        assert abs(p.mean()) < 1e-12 * T / R
        Fx, Fy = bp.projected_force(f, X, ds)
        assert max(np.abs(Fx).max(), np.abs(Fy).max()) < 1e-4 * np.abs(D0x(p, grid.dx)).max()


def test_force_pressure_wide_laplacian_offsets_the_subgrids():
    """With "wide" the four subgrids (i, j even or odd) decouple, and for the compact blob at 3h
    p_F inside the pressurised circle sits at four levels: the subgrid means over r < R/2 are
    0.087 (64^2) and 0.086 (128^2) of T/R apart, and the node values spread over 0.105 and
    0.129 of T/R. The mean jump is still right (+1.1e-3, +2.7e-4). With "tight" the subgrid
    means agree to 7.7e-5 and 2.3e-5 and the spread is 0.0021 and 0.0031."""
    R, T = 0.25, 1.0
    for N in (64, 128):
        grid = Grid.periodic_box(1.0, 1.0, N, N)
        c, X, f, ds = pressurised_circle(N, R, T)
        XX, YY = grid.meshgrid("nodes")
        inside = np.hypot(XX - c[0], YY - c[1]) < R / 2
        outside = (np.hypot(XX - c[0], YY - c[1]) > R + 0.1) & (np.hypot(XX - c[0], YY - c[1]) < 0.45)
        I, J = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
        res = {}
        for lap in ("tight", "wide"):
            p = BlobProjection(grid, get_blob("compact"), 3.0 * grid.dx, laplacian=lap).force_pressure(f, X, ds)
            means = [p[inside & (I % 2 == a) & (J % 2 == b)].mean() for a in (0, 1) for b in (0, 1)]
            res[lap] = ((max(means) - min(means)) / (T / R), np.ptp(p[inside]) / (T / R),
                        (p[inside].mean() - p[outside].mean()) / (T / R) - 1.0)
        assert res["wide"][0] > 0.05 and res["wide"][1] > 0.08
        assert res["tight"][0] < 1e-3 and res["tight"][1] < 0.01
        assert abs(res["wide"][2]) < 3e-3 and abs(res["tight"][2]) < 3e-4


@pytest.mark.parametrize("name", ["gaussian4", "compact"])
@pytest.mark.parametrize("laplacian", ["tight", "wide"])
def test_force_pressure_completes_the_helmholtz_split(name, laplacian):
    """P F~ + D0 p_F is the regularized force F~ sampled on the nodes, to second order in h at
    a fixed delta = 1/16 (measured, relative to max|F~| at 64, 128, 256: gaussian4 tight
    2.3e-2, 5.8e-3, 1.4e-3; compact tight 7.8e-2, 1.9e-2, 4.9e-3; rates 2.0)."""
    rng = np.random.default_rng(1)
    X = rng.uniform(0.3, 0.7, size=(5, 2))
    f = rng.normal(size=(5, 2))
    ds = rng.uniform(0.5, 1.5, 5) * 0.01
    delta = 1.0 / 16
    errs = []
    for N in (64, 128, 256):
        grid = Grid.periodic_box(1.0, 1.0, N, N)
        h = grid.dx
        patch = None if name == "compact" else int(np.ceil(4 * delta / h + 0.5))
        bp = BlobProjection(grid, get_blob(name), delta, patch, laplacian=laplacian)
        Fx, Fy, p = bp.projected_force_and_pressure(f, X, ds)
        XX, YY = grid.meshgrid("nodes")
        Sx, Sy = np.zeros_like(XX), np.zeros_like(XX)
        for k in range(len(X)):
            d = grid.min_image(np.stack([XX - X[k, 0], YY - X[k, 1]], axis=-1))
            ph = phi_typed(name, np.hypot(d[..., 0], d[..., 1]), delta)
            Sx += ds[k] * f[k, 0] * ph
            Sy += ds[k] * f[k, 1] * ph
        errs.append(max(np.abs(Fx + D0x(p, h) - Sx).max(), np.abs(Fy + D0y(p, h) - Sy).max())
                    / max(np.abs(Sx).max(), np.abs(Sy).max()))
    assert errs[2] < 1.5e-2
    assert np.log2(errs[0] / errs[1]) > 1.8 and np.log2(errs[1] / errs[2]) > 1.8


def test_projected_force_and_pressure_is_the_two_separate_calls():
    rng = np.random.default_rng(4)
    grid = Grid.periodic_box(1.0, 1.0, 32, 32)
    bp = BlobProjection(grid, get_blob("gaussian4"), 3.0 * grid.dx)
    X = rng.uniform(0, 1, size=(20, 2))
    f = rng.normal(size=(20, 2))
    ds = rng.uniform(0.5, 1.5, 20) * grid.dx
    Fx, Fy, p = bp.projected_force_and_pressure(f, X, ds)
    Gx, Gy = bp.projected_force(f, X, ds)
    q = bp.force_pressure(f, X, ds)
    assert np.allclose(Fx, Gx, rtol=0, atol=1e-14 * np.abs(Gx).max())
    assert np.allclose(Fy, Gy, rtol=0, atol=1e-14 * np.abs(Gy).max())
    assert np.allclose(p, q, rtol=0, atol=1e-14 * np.abs(q).max())
    # the divergence on the patches is the analytic one: compare with fourth-order differences
    # of the sampled F~ for a blob much wider than the grid
    g2 = Grid.periodic_box(1.0, 1.0, 256, 256)
    h2 = g2.dx
    bp2 = BlobProjection(g2, get_blob("compact"), 0.1)
    div = bp2.divergence_on_grid(f[:1], np.array([[0.5, 0.5]]), ds[:1])
    XX, YY = g2.meshgrid("nodes")
    ph = phi_typed("compact", np.hypot(XX - 0.5, YY - 0.5), 0.1)

    def d4(F, axis):
        return (8 * (np.roll(F, -1, axis) - np.roll(F, 1, axis))
                - (np.roll(F, -2, axis) - np.roll(F, 2, axis))) / (12 * h2)

    fd = ds[0] * (f[0, 0] * d4(ph, 0) + f[0, 1] * d4(ph, 1))
    assert np.abs(div - fd).max() < 1e-4 * np.abs(fd).max()


# ------------------------------------------------------------------------------------ bicubic interpolation
def test_bicubic_interpolation_exact_for_cubics():
    grid = Grid.periodic_box(1.0, 1.0, 32, 32)
    bp = BlobProjection(grid, get_blob("compact"), 3.0 * grid.dx, interpolation="bicubic")
    XX, YY = grid.meshgrid("nodes")

    def cubic(x, y):
        return 1.0 + 2 * x - 3 * y + 0.5 * x * x - x * y + 2 * y * y + x**3 - 2 * x * x * y + 0.7 * x * y * y - y**3

    rng = np.random.default_rng(5)
    P = rng.uniform(0.2, 0.8, size=(200, 2))          # stencils stay inside the box: no wrap
    U = bp.interpolate(cubic(XX, YY), -2.0 * cubic(XX, YY), P)
    assert np.allclose(U[:, 0], cubic(P[:, 0], P[:, 1]), rtol=1e-12, atol=1e-12)
    assert np.allclose(U[:, 1], -2.0 * cubic(P[:, 0], P[:, 1]), rtol=1e-12, atol=1e-12)
    assert np.array_equal(U, bp.interpolate_bicubic(cubic(XX, YY), -2.0 * cubic(XX, YY), P))
    # a quartic is not reproduced, so the stencil really is cubic
    U = bp.interpolate(XX**4, YY**4, P)
    assert np.abs(U[:, 0] - P[:, 0] ** 4).max() > 1e-8


def test_bicubic_interpolation_fourth_order_on_periodic_field():
    def field(x, y):
        return np.sin(2 * np.pi * x) * np.cos(4 * np.pi * y) + 0.3 * np.cos(6 * np.pi * x + 2 * np.pi * y)

    rng = np.random.default_rng(9)
    P = rng.uniform(0.0, 1.0, size=(400, 2))          # includes points whose stencil wraps
    errs = []
    for N in (32, 64, 128):
        grid = Grid.periodic_box(1.0, 1.0, N, N)
        bp = BlobProjection(grid, get_blob("compact"), 3.0 * grid.dx, interpolation="bicubic")
        XX, YY = grid.meshgrid("nodes")
        U = bp.interpolate(field(XX, YY), field(YY, XX), P)
        errs.append(max(np.abs(U[:, 0] - field(P[:, 0], P[:, 1])).max(),
                        np.abs(U[:, 1] - field(P[:, 1], P[:, 0])).max()))
    rates = [np.log2(errs[i] / errs[i + 1]) for i in range(2)]
    assert min(rates) > 3.7


@pytest.mark.parametrize("mode", ["bicubic", "streamfunction"])
def test_interpolation_periodic_wrap_is_a_roll(mode):
    rng = np.random.default_rng(13)
    grid = Grid.periodic_box(1.0, 1.0, 24, 24)
    h = grid.dx
    bp = BlobProjection(grid, get_blob("compact"), 3.0 * h, interpolation=mode)
    u, v = rng.normal(size=(24, 24)), rng.normal(size=(24, 24))
    P = rng.uniform(0.0, 1.0, size=(100, 2))
    U0 = bp.interpolate(u, v, P)
    for mi, mj in ((5, 0), (-7, 11), (0, -3)):
        U = bp.interpolate(np.roll(u, (mi, mj), axis=(0, 1)), np.roll(v, (mi, mj), axis=(0, 1)), P + [[mi * h, mj * h]])
        assert np.allclose(U, U0, rtol=1e-12, atol=1e-12 * np.abs(U0).max())
    if mode == "bicubic":
        # exact at the nodes themselves
        nodes = np.array([[3 * h, 7 * h], [0.0, 23 * h], [23 * h, 0.0]])
        U = bp.interpolate(u, v, nodes)
        assert np.allclose(U[:, 0], [u[3, 7], u[0, 23], u[23, 0]], atol=1e-13)
        assert np.allclose(U[:, 1], [v[3, 7], v[0, 23], v[23, 0]], atol=1e-13)


# ------------------------------------------------------------------------------------ stream-function interpolation
def test_default_interpolation_is_the_stream_function_curl():
    rng = np.random.default_rng(2)
    grid, u, v = random_divfree_velocity(32, rng)
    bp = BlobProjection(grid, get_blob("gaussian4"), 4.0 * grid.dx)
    P = rng.uniform(0, 1, size=(50, 2))
    assert np.array_equal(bp.interpolate(u, v, P), bp.interpolate_streamfunction(u, v, P))
    assert not np.allclose(bp.interpolate(u, v, P), bp.interpolate_bicubic(u, v, P))


def test_streamfunction_interpolant_is_divergence_free():
    """Fourth-order differences (exact for the polynomials inside a cell) of the interpolated
    field of a rough D0-divergence-free velocity give a divergence at round-off: measured
    2.3e-12 of the size of the derivatives. The bicubic interpolant of the same field has a
    divergence of the same size as its derivatives (0.74 of them here), which is what makes
    markers leak."""
    rng = np.random.default_rng(0)
    grid, u, v = random_divfree_velocity(32, rng, mean=(0.3, -0.1))
    h = grid.dx
    bp = BlobProjection(grid, get_blob("gaussian4"), 4.0 * h)
    P = rng.uniform(0, 1, size=(600, 2))
    t = (P / h) % 1.0
    P = P[np.all((t > 0.01) & (t < 0.99), axis=1)]         # the stencils stay inside one cell
    e = 1e-3 * h
    div, scale = fd_divergence(lambda Q: bp.interpolate_streamfunction(u, v, Q), P, e)
    assert np.abs(div).max() < 1e-9 * scale
    div_b, scale_b = fd_divergence(lambda Q: bp.interpolate_bicubic(u, v, Q), P, e)
    assert np.abs(div_b).max() > 0.1 * scale_b


def test_streamfunction_flux_through_rectangles_is_zero():
    """The outward flux of the interpolated velocity through axis-aligned rectangles (corners
    anywhere, edges crossing many cells, some wrapping across the box edge) vanishes to
    round-off. On each edge the normal component is a polynomial of degree 2 inside each
    cell, so 3-point Gauss-Legendre per cell segment integrates it exactly. The bicubic
    interpolant leaks through the same rectangles."""
    rng = np.random.default_rng(6)
    grid, u, v = random_divfree_velocity(32, rng, mean=(0.2, 0.4))
    h = grid.dx
    bp = BlobProjection(grid, get_blob("compact"), 3.0 * h)
    gx, gw = np.polynomial.legendre.leggauss(3)

    def edge_points(a, b):
        """Gauss points and weights on [a, b] with a node at every cell line in between."""
        cuts = np.concatenate([[a], np.arange(np.floor(a / h) + 1, np.ceil(b / h)) * h, [b]])
        lo, hi = cuts[:-1], cuts[1:]
        pts = (0.5 * (lo + hi))[:, None] + 0.5 * (hi - lo)[:, None] * gx[None, :]
        wts = 0.5 * (hi - lo)[:, None] * gw[None, :]
        return pts.ravel(), wts.ravel()

    def flux(interp, x0, x1, y0, y1):
        ys, wy = edge_points(y0, y1)
        xs, wx = edge_points(x0, x1)
        right = interp(np.column_stack([np.full_like(ys, x1), ys]))[:, 0] @ wy
        left = interp(np.column_stack([np.full_like(ys, x0), ys]))[:, 0] @ wy
        top = interp(np.column_stack([xs, np.full_like(xs, y1)]))[:, 1] @ wx
        bottom = interp(np.column_stack([xs, np.full_like(xs, y0)]))[:, 1] @ wx
        return right - left + top - bottom, (abs(right) + abs(left) + abs(top) + abs(bottom))

    leaks = []
    for _ in range(20):
        x0, y0 = rng.uniform(0.0, 1.0, 2)
        x1, y1 = x0 + rng.uniform(0.05, 0.6), y0 + rng.uniform(0.05, 0.6)
        net, gross = flux(lambda Q: bp.interpolate_streamfunction(u, v, Q), x0, x1, y0, y1)
        assert abs(net) < 1e-12 * gross
        net_b, gross_b = flux(lambda Q: bp.interpolate_bicubic(u, v, Q), x0, x1, y0, y1)
        leaks.append(abs(net_b) / gross_b)
    assert max(leaks) > 1e-3


def test_streamfunction_normal_velocity_is_continuous_across_cell_lines():
    """Across x = x_i the stencil switches cell. U_x (normal) is continuous; U_y (tangential)
    jumps, so the test is not vacuous. The same holds for y = y_j with the roles swapped."""
    rng = np.random.default_rng(8)
    grid, u, v = random_divfree_velocity(32, rng)
    h = grid.dx
    bp = BlobProjection(grid, get_blob("gaussian4"), 4.0 * h)
    i = rng.integers(0, 32, 40)
    y = rng.uniform(0, 1, 40)
    tiny = 1e-9 * h
    scale = np.abs(bp.interpolate_streamfunction(u, v, rng.uniform(0, 1, (400, 2)))).max()
    right = bp.interpolate_streamfunction(u, v, np.column_stack([i * h, y]))
    left = bp.interpolate_streamfunction(u, v, np.column_stack([i * h - tiny, y]))
    assert np.abs(right[:, 0] - left[:, 0]).max() < 1e-6 * scale
    assert np.abs(right[:, 1] - left[:, 1]).max() > 1e-2 * scale
    x = rng.uniform(0, 1, 40)
    j = rng.integers(0, 32, 40)
    top = bp.interpolate_streamfunction(u, v, np.column_stack([x, j * h]))
    bottom = bp.interpolate_streamfunction(u, v, np.column_stack([x, j * h - tiny]))
    assert np.abs(top[:, 1] - bottom[:, 1]).max() < 1e-6 * scale
    assert np.abs(top[:, 0] - bottom[:, 0]).max() > 1e-2 * scale


def test_streamfunction_interpolant_is_second_order():
    """For a smooth stream function (largest speed 14), the interpolated velocity converges at
    second order to the smooth field. Measured max errors at 32, 64, 128, 256 with the default
    "tight" Laplacian:
      grid velocity u = D0_y psi, v = -D0_x psi   6.3e-1, 1.6e-1, 3.9e-2, 9.8e-3 (rates 2.0)
      sampled smooth velocity                     2.3e-1, 5.5e-2, 1.3e-2, 3.3e-3 (2.07, 2.07, 2.01)
      against the grid values at the nodes        3.2e-1, 7.4e-2, 1.8e-2, 4.5e-3 (2.10, 2.04, 2.01)
    so it is not exact at the nodes. With "wide" the stream function of u = D0_y psi is psi
    itself, so only the interpolant's derivative error is left and that case is third order
    (9.7e-2, 1.2e-2, 1.5e-3, 1.9e-4); the other two stay second order (smooth: 4.3e-1, 1.1e-1,
    2.6e-2, 6.6e-3)."""
    rng = np.random.default_rng(10)
    P = rng.uniform(0.0, 1.0, size=(400, 2))
    for lap in ("tight", "wide"):
        e_d0, e_smooth, e_nodes = [], [], []
        for N in (32, 64, 128):
            grid = Grid.periodic_box(1.0, 1.0, N, N)
            h = grid.dx
            bp = BlobProjection(grid, get_blob("gaussian4"), 4.0 * h, laplacian=lap)
            XX, YY = grid.meshgrid("nodes")
            psi, psi_x, psi_y = smooth_psi(XX, YY)
            _, px, py = smooth_psi(P[:, 0], P[:, 1])
            exact = np.column_stack([py, -px])
            u, v = D0y(psi, h), -D0x(psi, h)
            e_d0.append(np.abs(bp.interpolate(u, v, P) - exact).max())
            e_smooth.append(np.abs(bp.interpolate(psi_y, -psi_x, P) - exact).max())
            nodes = np.column_stack([XX.ravel(), YY.ravel()])
            Un = bp.interpolate(u, v, nodes)
            e_nodes.append(max(np.abs(Un[:, 0] - u.ravel()).max(), np.abs(Un[:, 1] - v.ravel()).max()))
        for e in (e_d0, e_smooth, e_nodes):
            assert min(np.log2(e[0] / e[1]), np.log2(e[1] / e[2])) > 1.8
        assert e_nodes[-1] > 1e-6            # not exact at the nodes
        if lap == "wide":
            assert min(np.log2(e_d0[0] / e_d0[1]), np.log2(e_d0[1] / e_d0[2])) > 2.8


def test_streamfunction_keeps_the_mean_and_drops_the_gradient_part():
    """A uniform velocity is reproduced exactly; a discrete gradient (D0_x phi, D0_y phi) has
    no stream function and interpolates to zero; a checkerboard, which D0 cannot see, is
    dropped. So the interpolation acts on the D0-solenoidal part of what it is given."""
    rng = np.random.default_rng(12)
    grid = Grid.periodic_box(1.0, 1.0, 32, 32)
    h = grid.dx
    bp = BlobProjection(grid, get_blob("gaussian4"), 4.0 * h)
    P = rng.uniform(0, 1, size=(100, 2))
    U = bp.interpolate(np.full((32, 32), 0.7), np.full((32, 32), -1.3), P)
    assert np.allclose(U, [0.7, -1.3], rtol=0, atol=1e-14)
    phi = rng.normal(size=(32, 32))
    U = bp.interpolate(D0x(phi, h), D0y(phi, h), P)
    assert np.abs(U).max() < 1e-12 * np.abs(D0x(phi, h)).max()
    i = np.arange(32)
    checker = np.outer((-1.0) ** i, np.ones(32))           # u = (-1)^i: D0_x u = 0
    U = bp.interpolate(checker, np.zeros((32, 32)), P)
    assert np.abs(U).max() < 1e-13


def test_streamfunction_damps_modes_that_d0_barely_sees():
    """A D0-divergence-free mode close to the checkerboard in x (k1 = N/2 - 1, k2 = 1, and the
    middle mode k1 = N/4) on 128^2, largest interpolated speed over the largest grid speed:
      streamfunction, tight (the default)   0.040 (N/4: 0.77)
      streamfunction, wide                  33.6  (N/4: 1.53)
      bicubic                               0.99  (N/4: 1.00)
      adjoint, tight                        0.001 (N/4: 0.34)
    With "wide" the stream function is the least-squares one of D0, of size |u| h/|s| for this
    mode with |s| = sin(2 pi k1/N) small, and the exact derivative of its interpolant is 34
    times the grid speed. That fed marker-scale modes: with it as the default the compact blob
    at 3h blew up on the forced circle at nu = 1e-4 (t = 0.30-0.41) where bicubic ran. The
    five-point Laplacian of the default damps these modes instead."""
    N = 128
    grid = Grid.periodic_box(1.0, 1.0, N, N)
    h = grid.dx
    XX, YY = grid.meshgrid("nodes")
    P = np.random.default_rng(14).uniform(0, 1, size=(3000, 2))
    gains = {}
    for k1 in (N // 2 - 1, N // 4):
        psi = np.cos(2 * np.pi * (k1 * XX + YY) + 0.3)
        u, v = D0y(psi, h), -D0x(psi, h)
        speed = np.hypot(u, v).max()
        for lap in ("tight", "wide"):
            for mode in ("streamfunction", "bicubic", "adjoint"):
                bp = BlobProjection(grid, get_blob("compact"), 3.0 * h, laplacian=lap, interpolation=mode)
                gains[k1, lap, mode] = np.hypot(*bp.interpolate(u, v, P).T).max() / speed
    near = N // 2 - 1
    default = BlobProjection(grid, get_blob("compact"), 3.0 * h)
    assert default.laplacian == "tight" and default.interpolation == "streamfunction"
    assert gains[near, "tight", "streamfunction"] < 0.1
    assert gains[N // 4, "tight", "streamfunction"] < 1.0
    assert gains[near, "wide", "streamfunction"] > 20.0
    assert 0.95 < gains[near, "tight", "bicubic"] < 1.05
    assert gains[near, "tight", "adjoint"] < 0.01


# ------------------------------------------------------------------------------------ adjoint interpolation
@pytest.mark.parametrize("name", ["gaussian4", "compact"])
@pytest.mark.parametrize("laplacian", ["tight", "wide"])
def test_adjoint_interpolation_has_the_power_identity(name, laplacian):
    """sum_ij F_ij . u_ij h^2 = sum_k f_k . U_k ds_k for any velocity (not divergence-free
    here) and any forces, to round-off (measured 2e-16 to 6e-15 relative). The bicubic and
    stream-function interpolations miss it by order one on a field this rough."""
    rng = np.random.default_rng(15)
    grid = Grid.periodic_box(1.0, 1.0, 32, 32)
    h = grid.dx
    bp = BlobProjection(grid, get_blob(name), 3.0 * h, laplacian=laplacian, interpolation="adjoint")
    u, v = rng.normal(size=(32, 32)), rng.normal(size=(32, 32))
    X = rng.uniform(0, 1, size=(30, 2))
    f = rng.normal(size=(30, 2))
    ds = rng.uniform(0.5, 1.5, 30) * h
    Fx, Fy = bp.projected_force(f, X, ds)
    fluid = np.sum(Fx * u + Fy * v) * h * h
    scale = np.sum(np.abs(Fx * u) + np.abs(Fy * v)) * h * h
    U = bp.interpolate(u, v, X)
    assert abs(fluid - np.sum(f * U * ds[:, None])) < 1e-13 * scale
    for other in (bp.interpolate_bicubic, bp.interpolate_streamfunction):
        assert abs(fluid - np.sum(f * other(u, v, X) * ds[:, None])) > 1e-2 * abs(fluid)


@pytest.mark.parametrize("name", ["gaussian4", "compact"])
def test_adjoint_interpolant_is_divergence_free(name):
    """U = curl_X of a blob-weighted sum, so its divergence vanishes to round-off (fourth-order
    differences, step 1e-3 h, kept away from the half-cell lines where the Gaussian blob's
    patch shifts). For the compact blob the patch holds the whole support, so U is one smooth
    field and the half-cell lines need no care."""
    rng = np.random.default_rng(16)
    grid, u, v = random_divfree_velocity(32, rng, mean=(0.1, 0.2))
    h = grid.dx
    bp = BlobProjection(grid, get_blob(name), 3.0 * h, interpolation="adjoint")
    P = rng.uniform(0, 1, size=(600, 2))
    if name == "gaussian4":
        t = (P / h) % 1.0
        P = P[np.all((np.abs(t - 0.5) > 0.01) & (t > 0.01) & (t < 0.99), axis=1)]
    div, scale = fd_divergence(lambda Q: bp.interpolate(u, v, Q), P, 1e-3 * h)
    assert np.abs(div).max() < 1e-8 * scale


def test_adjoint_interpolant_accuracy_on_a_smooth_field():
    """For a smooth stream function and delta = 3h (the blob shrinks with h), relative to the
    largest speed: gaussian4 converges, 1.5e-1, 2.1e-2, 3.4e-3, 7.5e-4 at 32..256 (rates 2.8,
    2.7, 2.2). The compact blob does not: 1.1e-1, 3.1e-2, 2.6e-2, 5.5e-2. The blob-weighted
    sum is a quadrature of a function only six cells wide that is C^4 at its edge, and its error
    depends on delta/h and on where the point sits in its cell, not on h (the same error
    limits the compact blob's projected far field at 3h)."""
    rng = np.random.default_rng(17)
    P = rng.uniform(0.0, 1.0, size=(300, 2))
    errs = {}
    for name in ("gaussian4", "compact"):
        errs[name] = []
        for N in (32, 64, 128):
            grid = Grid.periodic_box(1.0, 1.0, N, N)
            h = grid.dx
            bp = BlobProjection(grid, get_blob(name), 3.0 * h, interpolation="adjoint")
            XX, YY = grid.meshgrid("nodes")
            psi, _, _ = smooth_psi(XX, YY)
            _, px, py = smooth_psi(P[:, 0], P[:, 1])
            exact = np.column_stack([py, -px])
            U = bp.interpolate(D0y(psi, h), -D0x(psi, h), P)
            errs[name].append(np.abs(U - exact).max() / np.abs(exact).max())
    e = errs["gaussian4"]
    assert min(np.log2(e[0] / e[1]), np.log2(e[1] / e[2])) > 2.0 and e[2] < 5e-3
    assert 1e-2 < errs["compact"][2] < 6e-2


def relaxing_ellipse_run(N, dt, T, mode, out_dir, name="gaussian4", mu=0.0):
    """Release a stretched ellipse (0.2 x 0.125, 2N points, Hookean springs k = 1/ds^2 resting
    at the circle of equal area) from rest in the doubly periodic unit box through the driver,
    with case and output directory ``out_dir`` (nothing is written there). Returns
    (W(T) - W0)/W0 with W the kinetic plus elastic energy (mu = 0, so W is conserved by the
    continuous problem)."""
    from pyibm.driver import Simulation
    from pyibm.inputfile import (BCParams, FluidParams, GridParams, OutputParams, Parameters,
                                 SolverParams, StructureParams, TimeParams)
    from pyibm.structures import Hooks, Springs, Structure, structure_energy

    n = 2 * N
    th = 2 * np.pi * np.arange(n) / n
    X = np.column_stack([0.5 + 0.2 * np.cos(th), 0.5 + 0.125 * np.sin(th)])
    i = np.arange(n)
    j = (i + 1) % n
    ds = float(np.mean(np.hypot(*(X[j] - X[i]).T)))
    rest = 2 * np.sqrt(0.2 * 0.125) * np.sin(np.pi / n)
    st = Structure("ellipse", X, np.full(n, ds),
                   springs=Springs(i, j, np.full(n, 1.0 / ds**2), np.full(n, rest)))
    p = Parameters(
        fluid=FluidParams(rho=1.0, mu=mu), time=TimeParams(Tfinal=T, dt=dt),
        grid=GridParams(Nx=N, Ny=N, Lx=1.0, Ly=1.0),
        solver=SolverParams(backend="periodic_fft", time_scheme="midpoint", coupling="blob",
                            blob=name, blob_delta=4.0),
        bc=BCParams(), structure=StructureParams(),
        output=OutputParams(print_dump=10**9, save_vtk=False, save_history=False))
    sim = Simulation(p, str(out_dir), hooks=Hooks(), structure=st, output_dir=str(out_dir),
                     quiet=True)
    sim.coupling = BlobProjection(sim.grid, get_blob(name), 4.0 * sim.grid.dx, interpolation=mode)

    def W():
        return sim.solver.kinetic_energy() + ds * structure_energy(sim.structure, sim.grid)

    W0 = W()
    for _ in range(int(round(T / dt))):
        sim.step()
    return (W() - W0) / W0


@pytest.mark.filterwarnings("ignore:tether number:RuntimeWarning")
def test_adjoint_interpolation_conserves_energy_to_second_order_in_dt(tmp_path):
    """Inviscid release on 32^2 to t = 0.5, gaussian4 at 4h. With the adjoint interpolation
    the energy error is that of the midpoint scheme and falls as dt^2 (measured +1.6e-4,
    +4.1e-5 at dt = 4e-3, 2e-3). With the stream-function interpolation the total changes by
    -22 % whatever dt is (bicubic: -30 %): no power identity, no energy balance."""
    a = relaxing_ellipse_run(32, 4e-3, 0.5, "adjoint", tmp_path)
    b = relaxing_ellipse_run(32, 2e-3, 0.5, "adjoint", tmp_path)
    assert abs(b) < 1e-4 and 3.0 < a / b < 5.0
    assert abs(relaxing_ellipse_run(32, 4e-3, 0.5, "streamfunction", tmp_path)) > 0.1


# ------------------------------------------------------------------------------------ the paper's ellipse
def cortez_minion_ellipse(n=400):
    """The perturbed ellipse of Cortez & Minion 2000 section 4.1 (a = 0.2, b = 0.25,
    eps = 0.012, theta0 = pi/3, theta1 = 4 pi/5, theta2 = 2 pi/3, centre (1/2, 1/2)) with n
    points at equal arc length."""
    def curve(theta):
        r2 = (0.04 * np.cos(theta) ** 2 + 0.0625 * np.sin(theta) ** 2
              + 0.012 * (-(4.0 / 3.0) * np.exp(-3.0 * (theta - np.pi) ** 2)
                         - np.exp(-5.0 * (theta - 0.8 * np.pi) ** 2)
                         + np.exp(-8.0 * (theta - 2.0 * np.pi / 3.0) ** 2)))
        r = np.sqrt(r2)
        return np.column_stack([0.5 + r * np.cos(theta - np.pi / 3), 0.5 + r * np.sin(theta - np.pi / 3)])

    theta = np.linspace(0.0, 2.0 * np.pi, 40001)
    Z = curve(theta)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(Z, axis=0).T))])
    return curve(np.interp(s[-1] * np.arange(n) / n, s, theta))


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore:tether number:RuntimeWarning")
@pytest.mark.parametrize("mode", ["streamfunction", "adjoint"])
def test_paper_ellipse_area_by_interpolation(mode, tmp_path):
    """Cortez & Minion 2000 section 4.1 through the driver: 128^2, 400 points, gaussian4,
    delta = 4h, nu = 0.005, dt = 0.1h, constant tension sigma = 0.2 (a chain of zero-rest-length
    springs whose stiffness is reset to sigma/(ds L) at every force evaluation, which gives
    sigma kappa n in integrated form), to t = 1. Relative area error (shoelace), measured:
      streamfunction  +2.46e-6 at t = 1, at most 5.66e-6 over the run
      adjoint         +4.07e-7 at t = 1, at most 7.24e-6 over the run
    The paper reports errors within +-1e-5. About 30 s per run (adjoint: about 100 s)."""
    from pyibm.driver import Simulation
    from pyibm.inputfile import (BCParams, FluidParams, GridParams, OutputParams, Parameters,
                                 SolverParams, StructureParams, TimeParams)
    from pyibm.structures import Hooks, Springs, Structure

    sigma = 0.2
    N = 128
    h = 1.0 / N
    X = cortez_minion_ellipse()
    n = len(X)
    i = np.arange(n)
    j = (i + 1) % n
    L = np.hypot(*(X[j] - X[i]).T)
    ds = float(L.mean())
    st = Structure("ellipse", X, np.full(n, ds), springs=Springs(i, j, sigma / (ds * L), np.zeros(n)))

    def update_springs(t, dt, s):
        sp = s.springs
        Lk = np.hypot(*(s.X[sp.j] - s.X[sp.i]).T)
        sp.k[:] = sigma / (s.ds[sp.i] * Lk)
        sp.rest[:] = 0.0

    dt = 0.1 * h
    p = Parameters(
        fluid=FluidParams(rho=1.0, mu=0.005), time=TimeParams(Tfinal=1.0, dt=dt),
        grid=GridParams(Nx=N, Ny=N, Lx=1.0, Ly=1.0),
        solver=SolverParams(backend="periodic_fft", time_scheme="midpoint", coupling="blob",
                            blob="gaussian4", blob_delta=4.0),
        bc=BCParams(), structure=StructureParams(),
        output=OutputParams(print_dump=10**9, save_vtk=False, save_history=False))
    sim = Simulation(p, str(tmp_path), hooks=Hooks(update_springs=update_springs), structure=st,
                     output_dir=str(tmp_path / "out"), quiet=True)
    sim.coupling = BlobProjection(sim.grid, get_blob("gaussian4"), 4.0 * h, interpolation=mode)

    def area(Y):
        x, y = Y[:, 0], Y[:, 1]
        return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))

    A0 = area(sim.structure.X)
    errs = []
    for step in range(1, int(round(1.0 / dt)) + 1):
        sim.step()
        if step % 40 == 0:
            errs.append((area(sim.structure.X) - A0) / A0)
    assert np.abs(np.array(errs)).max() < 1e-5


# ------------------------------------------------------------------------------------ the forced circle
@pytest.mark.slow
@pytest.mark.filterwarnings("ignore:tether number:RuntimeWarning")
def test_compact_blob_stays_finite_at_low_viscosity(tmp_path):
    """The forced circle of examples/forced_circle (after Fovargue 2012 section 5.4: 128^2, 400
    points at 0.4h, nu = 1e-4; the membrane model and stiffnesses are our own choice,
    zero-rest-length springs k_S = 1e4 and targets k_T = 3200) with
    the compact blob at 3h, the default interpolation and dt = dx/8, to t = 1. With the
    least-squares stream function of D0 the run blew up at t = 0.30 (dx/16: 0.36, dx/32:
    0.41). With the five-point stream function it runs to t = 1 with max|u| 2.75, checked every
    16th step as below (bicubic: 1.6, gaussian4 at 4h: 1.2), so the chain is still rougher
    than with bicubic. About 15 s."""
    import os
    from pyibm.driver import Simulation
    from pyibm.inputfile import read_input_file

    case = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples", "forced_circle")
    p = read_input_file(os.path.join(case, "input2d_blob"))
    assert (p.grid.Nx, p.grid.Ny) == (128, 128)
    p.fluid.mu = 1e-4
    p.time.dt = 1.0 / 128 / 8
    p.output.save_vtk = False
    p.output.save_history = False
    sim = Simulation(p, case, output_dir=str(tmp_path / "out"), quiet=True)
    sim.structure.springs.k[:] = 1e4
    sim.coupling = BlobProjection(sim.grid, get_blob("compact"), 3.0 * sim.grid.dx)
    assert sim.coupling.interpolation == "streamfunction"
    top = 0.0
    with np.errstate(all="ignore"):
        for n in range(1024):
            sim.step()
            if n % 16 == 15:
                speed = float(np.max(np.hypot(sim.solver.u, sim.solver.v)))
                assert np.isfinite(speed) and speed < 10.0, f"t = {sim.t:.3f}: max|u| = {speed:.3g}"
                top = max(top, speed)
    assert sim.t == pytest.approx(1.0)
    assert np.all(np.isfinite(sim.structure.X))
    assert top < 5.0
