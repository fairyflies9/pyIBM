"""Tests for pyibm.coupling.delta: force conservation, exact linear interpolation, adjointness,
the grid-scale mode of the spread force, periodic wrap, walls, MAC face layouts (odd and
smallest cell counts included), and the 2008 MATLAB forcespread/move_boundary loops.
"""

import math

import numpy as np
import pytest

from pyibm.coupling.delta import DeltaCoupling
from pyibm.grid import FieldLayout, Grid, layout_collocated, layout_u, layout_v
from pyibm.kernels import get_kernel

KERNELS = ["peskin4", "cosine4", "roma3"]


@pytest.fixture
def pbox():
    return Grid.periodic_box(1.0, 1.0, 32, 32)


@pytest.fixture
def mac():
    """Closed box with dx != dy, u on (Nx+1, Ny) faces and v on (Nx, Ny+1) faces."""
    return Grid(1.0, 0.5, 32, 16, periodic_x=False, periodic_y=False)


def _interior_points(rng, layout, N, margin=3.0):
    """Random points at least `margin` cells away from every edge of the layout's node set."""
    x, y = layout.coordinates()
    X = rng.uniform(x[0] + margin * layout.hx, x[-1] - margin * layout.hx, N)
    Y = rng.uniform(y[0] + margin * layout.hy, y[-1] - margin * layout.hy, N)
    return np.column_stack([X, Y])


# ---------------------------------------------------------------------------- conservation
@pytest.mark.parametrize("name", KERNELS)
def test_total_force_conserved_periodic(name, pbox):
    """sum_ij F_ij hx hy = sum_k f_k ds_k, including points whose support wraps around."""
    dc = DeltaCoupling(get_kernel(name))
    lay = layout_collocated(pbox)
    rng = np.random.default_rng(10)
    N = 200
    X = rng.uniform(-0.5, 1.5, (N, 2))               # anywhere, including outside the box
    f = rng.normal(size=N)
    ds = rng.uniform(0.5, 1.5, N) * pbox.dx
    F = dc.spread(f, X, ds, lay)
    assert F.shape == (32, 32) and F.dtype == np.float64
    assert F.sum() * lay.hx * lay.hy == pytest.approx((f * ds).sum(), rel=1e-13, abs=1e-15)
    # a single unit force: sum F h^2 = ds, and the force lands only on the width^2 stencil
    X1 = np.array([[0.3141, 0.2718]])
    F1 = dc.spread(np.array([1.0]), X1, np.array([0.01]), lay)
    assert F1.sum() * lay.hx * lay.hy == pytest.approx(0.01, rel=1e-13)
    assert np.count_nonzero(F1) <= dc.kernel.width ** 2
    assert np.all(F1 >= 0.0)


@pytest.mark.parametrize("name", KERNELS)
def test_total_force_conserved_mac_layouts(name, mac):
    dc = DeltaCoupling(name)
    rng = np.random.default_rng(11)
    for lay in (layout_u(mac), layout_v(mac), layout_collocated(mac)):
        X = _interior_points(rng, lay, 100)
        f = rng.normal(size=100)
        ds = 0.7 * mac.dx
        F = dc.spread(f, X, ds, lay)
        assert F.shape == lay.shape
        assert F.sum() * lay.hx * lay.hy == pytest.approx(f.sum() * ds, rel=1e-13, abs=1e-15)
    assert layout_u(mac).shape == (33, 16) and layout_v(mac).shape == (32, 17)


# ---------------------------------------------------------------------------- interpolation
@pytest.mark.parametrize("name", ["peskin4", "roma3"])
def test_linear_field_interpolated_exactly(name, pbox, mac):
    """Constants and linear fields are reproduced exactly by kernels with a zero first moment."""
    dc = DeltaCoupling(name)
    rng = np.random.default_rng(12)
    a, b, c = 0.7, -1.3, 2.1
    for grid in (pbox, mac):
        for lay in (layout_collocated(grid), layout_u(grid), layout_v(grid)):
            x, y = lay.coordinates()
            field = a + b * x[:, None] + c * y[None, :]
            X = _interior_points(rng, lay, 300)
            U = dc.interpolate(field, X, lay)
            assert U.shape == (300,)
            assert np.max(np.abs(U - (a + b * X[:, 0] + c * X[:, 1]))) < 1e-12
            assert np.allclose(dc.interpolate(np.full(lay.shape, 3.5), X, lay), 3.5, rtol=1e-13)


def test_cosine4_reproduces_constants_but_not_linear_fields(pbox):
    dc = DeltaCoupling("cosine4")
    lay = layout_collocated(pbox)
    rng = np.random.default_rng(13)
    x, y = lay.coordinates()
    X = _interior_points(rng, lay, 300)
    assert np.allclose(dc.interpolate(np.full(lay.shape, 2.0), X, lay), 2.0, rtol=1e-13)
    field = x[:, None] + 0.0 * y[None, :]
    err = np.abs(dc.interpolate(field, X, lay) - X[:, 0]) / lay.hx
    assert err.max() > 1e-3            # nonzero first moment: about 0.02 cells at worst


# ---------------------------------------------------------------------------- adjointness
@pytest.mark.parametrize("name", KERNELS)
def test_spread_and_interpolate_are_adjoint(name, pbox, mac):
    """sum_ij F_ij u_ij hx hy == sum_k f_k ds_k U_k for random data (the power identity,
    Peskin 2002 section 5, eq. 5.2)."""
    dc = DeltaCoupling(name)
    rng = np.random.default_rng(14)
    cases = [(layout_collocated(pbox), rng.uniform(-0.2, 1.2, (60, 2)))]
    for lay in (layout_u(mac), layout_v(mac)):
        cases.append((lay, _interior_points(rng, lay, 60)))
    for lay, X in cases:
        N = X.shape[0]
        f = rng.normal(size=N)
        ds = rng.uniform(0.5, 1.5, N) * lay.hx
        u = rng.normal(size=lay.shape)
        F = dc.spread(f, X, ds, lay)
        U = dc.interpolate(u, X, lay)
        lhs = (F * u).sum() * lay.hx * lay.hy
        rhs = (f * ds * U).sum()
        scale = max(1.0, abs(lhs), (np.abs(F * u)).sum() * lay.hx * lay.hy)
        assert abs(lhs - rhs) <= 1e-13 * scale


# ---------------------------------------------------------------------------- grid-scale mode
@pytest.mark.parametrize("name", KERNELS)
def test_spread_force_has_no_grid_scale_mode(name, pbox):
    """The even/odd rule (Peskin & McQueen 1996 eq. 14.63) makes the discrete Fourier transform
    of the spread force vanish at every wave vector with a component N/2, the mode (-1)^i that
    the centred difference D0 cannot see (the discussion after their eq. 14.66). peskin4 and
    cosine4 have this property; roma3 does not, which is why it is meant for the MAC grid."""
    dc = DeltaCoupling(name)
    lay = layout_collocated(pbox)
    h, n = pbox.dx, pbox.Nx // 2
    rng = np.random.default_rng(18)
    X = rng.uniform(0.0, 1.0, (40, 2))
    F = dc.spread(rng.normal(size=40), X, np.full(40, h), lay)
    Fh = np.fft.fft2(F)
    nyquist = max(np.abs(Fh[n, :]).max(), np.abs(Fh[:, n]).max())
    # one point on a node: Fh[n, 0] / Fh[0, 0] = sum_j (-1)^j phi(j) along x
    F1 = dc.spread(np.array([1.0]), np.array([[5 * h, 0.3141]]), np.array([h]), lay)
    F1h = np.fft.fft2(F1)
    ratio = abs(F1h[n, 0]) / abs(F1h[0, 0])
    if name == "roma3":
        assert ratio == pytest.approx(1.0 / 3.0, abs=1e-12)        # 2/3 - 1/6 - 1/6
        assert nyquist > 0.01 * np.abs(Fh).max()
    else:
        assert ratio < 1e-12
        assert nyquist < 1e-12 * np.abs(Fh).max()


# ---------------------------------------------------------------------------- cell counts
@pytest.mark.parametrize("name", KERNELS)
def test_odd_cell_counts_on_wall_bounded_grid(name):
    """Odd Nx, Ny are allowed along walls: spreading still conserves force and interpolation
    still reproduces constants (and linear fields for peskin4 and roma3) on every layout."""
    grid = Grid(1.0, 0.7, 13, 11, periodic_x=False, periodic_y=False)
    dc = DeltaCoupling(name)
    rng = np.random.default_rng(19)
    for lay in (layout_collocated(grid), layout_u(grid), layout_v(grid)):
        X = _interior_points(rng, lay, 50)
        f = rng.normal(size=50)
        F = dc.spread(f, X, 0.5 * grid.dx, lay)
        assert F.shape == lay.shape
        assert F.sum() * lay.hx * lay.hy == pytest.approx(f.sum() * 0.5 * grid.dx, rel=1e-13, abs=1e-15)
        assert np.allclose(dc.interpolate(np.full(lay.shape, -1.25), X, lay), -1.25, rtol=1e-13)
        if name != "cosine4":
            x, y = lay.coordinates()
            field = 0.3 - 2.0 * x[:, None] + 1.1 * y[None, :]
            U = dc.interpolate(field, X, lay)
            assert np.max(np.abs(U - (0.3 - 2.0 * X[:, 0] + 1.1 * X[:, 1]))) < 1e-12
    assert layout_u(grid).shape == (14, 11) and layout_v(grid).shape == (13, 12)


@pytest.mark.parametrize("name", ["peskin4", "cosine4"])
def test_four_cells_is_the_smallest_wall_box_for_a_4_point_kernel(name):
    """The Grid allows N >= 4 along walls because 4 cells is the smallest box where a 4-point
    stencil fits between the walls on both MAC velocity fields. Along x, with N cells, the
    u-faces have N + 1 nodes and the v-faces N (at the cell centres); along y the two swap.
    With Nx = 3 the v-faces have 3 nodes along x and no point can be placed at all."""
    dc = DeltaCoupling(name)
    msg = "Lagrangian point within kernel support of a non-periodic boundary"
    grid = Grid(1.0, 1.0, 4, 4, periodic_x=False, periodic_y=False)
    X = np.array([[0.5, 0.5]])
    for lay in (layout_u(grid), layout_v(grid)):
        F = dc.spread(np.array([2.0]), X, np.array([0.1]), lay)
        assert F.sum() * lay.hx * lay.hy == pytest.approx(0.2, rel=1e-13)
        assert dc.interpolate(np.ones(lay.shape), X, lay) == pytest.approx(1.0, rel=1e-13)
    # v-faces of a channel 3 cells wide between walls in x and periodic in y (Grid refuses
    # Nx = 3, so we build the layout by hand). When x is periodic the same point is fine: the
    # 4-point stencil wraps onto the 3 nodes, one node takes two contributions, and the total
    # force is still conserved
    h = 1.0 / 3.0
    v_faces3 = FieldLayout(0.5 * h, 0.0, h, h, 3, 8, False, True)
    F = dc.spread(np.array([1.0]), np.array([[0.5, 0.5]]), np.array([0.1]),
                  FieldLayout(0.5 * h, 0.0, h, h, 3, 8, True, True))
    assert F.shape == (3, 8)
    assert F.sum() * h * h == pytest.approx(0.1, rel=1e-13)
    for x in np.linspace(0.0, 1.0, 61):
        with pytest.raises(ValueError, match=msg):
            dc.spread(np.array([1.0]), np.array([[x, 0.5]]), np.array([0.1]), v_faces3)


# ---------------------------------------------------------------------------- periodic wrap
@pytest.mark.parametrize("name", KERNELS)
def test_periodic_wrap_near_box_edges(name, pbox):
    """Shifting a point by whole cells rolls the spread field; a point near (0, 0) and near
    (Lx, Ly) wraps onto the opposite rows/columns; interpolation sees the rolled field."""
    dc = DeltaCoupling(name)
    lay = layout_collocated(pbox)
    h = pbox.dx
    rng = np.random.default_rng(15)
    f, ds = np.array([1.7]), np.array([0.3 * h])
    for corner in (np.array([[0.3 * h, 0.2 * h]]), np.array([[1.0 - 0.4 * h, 1.0 - 0.1 * h]])):
        shift = (5, 7)
        F_edge = dc.spread(f, corner, ds, lay)
        F_in = dc.spread(f, corner + np.array(shift) * h, ds, lay)
        assert np.allclose(np.roll(F_in, (-shift[0], -shift[1]), axis=(0, 1)), F_edge, rtol=1e-13, atol=1e-16)
        # the support straddles the edge: both the first and the last rows/columns are hit
        assert F_edge[0, :].any() and F_edge[-1, :].any() and F_edge[:, 0].any() and F_edge[:, -1].any()
        u = rng.normal(size=lay.shape)
        U_edge = dc.interpolate(u, corner, lay)
        U_in = dc.interpolate(np.roll(u, shift, axis=(0, 1)), corner + np.array(shift) * h, lay)
        assert np.allclose(U_edge, U_in, rtol=1e-13, atol=1e-16)
    # positions outside [0, L) are treated as their periodic image
    X = np.array([[-0.37 * h, 1.0 + 2.2 * h]])
    Fa = dc.spread(f, X, ds, lay)
    Fb = dc.spread(f, pbox.wrap(X), ds, lay)
    assert np.allclose(Fa, Fb, rtol=1e-11, atol=1e-13)


# ---------------------------------------------------------------------------- walls
def test_value_error_near_nonperiodic_wall(mac):
    dc = DeltaCoupling("peskin4")
    msg = "Lagrangian point within kernel support of a non-periodic boundary"
    lu, lv = layout_u(mac), layout_v(mac)
    dx, dy = mac.dx, mac.dy
    f, ds = np.array([1.0]), np.array([dx])
    for lay in (lu, lv, layout_collocated(mac)):
        for X in (np.array([[0.5 * dx, 0.25]]),            # left wall
                  np.array([[0.5, 0.5 - 0.3 * dy]]),       # top wall
                  np.array([[1.0 + dx, 0.25]]),            # outside the box
                  np.array([[0.5, 0.25], [0.5, 0.0]])):    # one bad point among good ones
            with pytest.raises(ValueError, match=msg):
                dc.spread(np.ones(X.shape[0]), X, ds[0], lay)
            with pytest.raises(ValueError, match=msg):
                dc.interpolate(np.zeros(lay.shape), X, lay)
    # the node count per layout matters: u-faces run to x = Lx (33 nodes), v-faces to
    # Lx - dx/2 (32 nodes). A point 1.5 dx from the right wall needs u-faces 29..32 (fine)
    # but v-faces 29..32 of 0..31 (not fine).
    X = np.array([[1.0 - 1.5 * dx, 0.25]])
    dc.spread(f, X, ds, lu)
    with pytest.raises(ValueError, match=msg):
        dc.spread(f, X, ds, lv)
    # and the same point is fine on a grid that is periodic in x
    gpx = Grid(1.0, 0.5, 32, 16, periodic_x=True, periodic_y=False)
    dc.spread(f, X, ds, layout_v(gpx))
    with pytest.raises(ValueError, match=msg):
        dc.spread(f, np.array([[0.5, 0.0]]), ds, layout_v(gpx))


def test_interior_points_fine_on_mac_layouts(mac):
    """Face layouts: the stencil of a point lands on the right faces with the right offsets."""
    dc = DeltaCoupling("peskin4")
    lu, lv = layout_u(mac), layout_v(mac)
    X = np.array([[0.5, 0.25]])                 # on a u-face column (x = 16 dx) and a v-face row
    F = dc.spread(np.array([1.0]), X, np.array([1.0]), lu)
    i, j = np.nonzero(F)
    assert set(i.tolist()) <= {15, 16, 17, 18} and set(j.tolist()) <= {6, 7, 8, 9}
    # y = 0.25 sits half-way between u-face rows 7 (y = 7.5 dy) and 8: equal weights
    assert F[16, 7] == pytest.approx(F[16, 8]) and F[16, 6] == pytest.approx(F[16, 9])
    F = dc.spread(np.array([1.0]), X, np.array([1.0]), lv)
    i, j = np.nonzero(F)
    assert set(j.tolist()) <= {7, 8, 9, 10} and set(i.tolist()) <= {14, 15, 16, 17}
    # x = 0.5 is half-way between v-face columns 15 and 16 (r = 1/2); y = 0.25 is on row 8
    phi = dc.kernel.phi
    assert F[15, 8] == pytest.approx(F[16, 8])
    assert F[16, 8] == pytest.approx(phi(0.5) * phi(0.0) / (mac.dx * mac.dy))
    assert F[14, 8] == pytest.approx(phi(1.5) * phi(0.0) / (mac.dx * mac.dy))


# ---------------------------------------------------------------------------- vector helpers
def test_vector_conveniences(pbox, mac):
    dc = DeltaCoupling("peskin4")
    rng = np.random.default_rng(16)
    lc = layout_collocated(pbox)
    X = rng.uniform(0, 1, (25, 2))
    f = rng.normal(size=(25, 2))
    ds = np.full(25, pbox.dx)
    Fx, Fy = dc.spread_vector(f, X, ds, lc, lc)
    assert np.array_equal(Fx, dc.spread(f[:, 0], X, ds, lc))
    assert np.array_equal(Fy, dc.spread(f[:, 1], X, ds, lc))
    u, v = rng.normal(size=lc.shape), rng.normal(size=lc.shape)
    UV = dc.interpolate_vector(u, v, X, lc, lc)
    assert UV.shape == (25, 2)
    assert np.array_equal(UV[:, 0], dc.interpolate(u, X, lc))
    assert np.array_equal(UV[:, 1], dc.interpolate(v, X, lc))
    # MAC: x component on u-faces, y component on v-faces
    lu, lv = layout_u(mac), layout_v(mac)
    Xm = _interior_points(rng, lv, 25)
    Fx, Fy = dc.spread_vector(f, Xm, ds, lu, lv)
    assert Fx.shape == lu.shape and Fy.shape == lv.shape
    UV = dc.interpolate_vector(np.ones(lu.shape), 2 * np.ones(lv.shape), Xm, lu, lv)
    assert np.allclose(UV[:, 0], 1.0) and np.allclose(UV[:, 1], 2.0)
    with pytest.raises(ValueError):
        dc.spread_vector(f[:, 0], X, ds, lc, lc)


def test_argument_checks_and_edge_cases(pbox):
    lc = layout_collocated(pbox)
    dc = DeltaCoupling(get_kernel("roma3"))
    assert dc.kernel.name == "roma3"
    with pytest.raises(TypeError):
        DeltaCoupling(3)
    with pytest.raises(ValueError):
        dc.spread(np.ones(3), np.zeros((3, 3)), 1.0, lc)                  # X not (N, 2)
    with pytest.raises(ValueError, match="spread_vector"):
        dc.spread(np.ones((3, 2)), np.zeros((3, 2)), 1.0, lc)             # f is (N, 2)
    with pytest.raises(ValueError):
        dc.spread(np.ones(4), np.zeros((3, 2)), 1.0, lc)                  # f length mismatch
    with pytest.raises(ValueError):
        dc.interpolate(np.zeros((32, 31)), np.zeros((3, 2)), lc)          # field shape mismatch
    # no points: zero field, empty result
    F = dc.spread(np.zeros(0), np.zeros((0, 2)), np.zeros(0), lc)
    assert F.shape == lc.shape and not F.any()
    assert dc.interpolate(np.ones(lc.shape), np.zeros((0, 2)), lc).shape == (0,)
    # scalar ds broadcasts; array ds per point
    X = np.array([[0.5, 0.5], [0.25, 0.75]])
    Fa = dc.spread(np.array([1.0, 2.0]), X, 0.01, lc)
    Fb = dc.spread(np.array([1.0, 2.0]), X, np.array([0.01, 0.01]), lc)
    assert np.array_equal(Fa, Fb)
    # inputs are not modified
    X0 = X.copy()
    dc.spread(np.array([1.0, 2.0]), X, 0.01, lc)
    assert np.array_equal(X, X0)


# ---------------------------------------------------------------------------- 2008 MATLAB
def _wrap_matlab(ii, M):
    if ii == -1:
        ii = M - 1
    if ii == -2:
        ii = M - 2
    if ii == -3:
        ii = M - 3
    if ii == M:
        ii = 0
    if ii == M + 1:
        ii = 1
    if ii == M + 2:
        ii = 2
    return ii


def _matlab_forcespread(array1, array2, in_f1, in_f2, Q, dx, ds, M, N):
    """Literal transcription of forcespread.m (2008). MATLAB's f1(ii+1, jj+1) is f1[ii, jj]."""
    f1 = np.zeros((M, N))
    f2 = np.zeros((M, N))
    for i in range(Q):
        x1 = math.ceil(array1[i] / dx)
        x2 = math.ceil(array2[i] / dx)
        for k in range(4):
            for h in range(4):
                ii = _wrap_matlab(x1 + 1 - k, M)
                jj = _wrap_matlab(x2 + 1 - h, N)
                wx = (1 / (4 * dx)) * (1 + math.cos((math.pi / (2 * dx)) * (dx * (x1 + 1 - k) - array1[i])))
                wy = (1 / (4 * dx)) * (1 + math.cos((math.pi / (2 * dx)) * (dx * (x2 + 1 - h) - array2[i])))
                f1[ii, jj] += in_f1[i] * wx * wy * ds
                f2[ii, jj] += in_f2[i] * wx * wy * ds
    return f1, f2


def _matlab_move_boundary_velocity(array1, array2, Q, u, v, dx, M, N):
    """Literal transcription of the interpolation loop of move_boundary.m (2008)."""
    uarray = np.zeros(Q)
    varray = np.zeros(Q)
    for i in range(Q):
        x1 = math.ceil(array1[i] / dx)
        x2 = math.ceil(array2[i] / dx)
        for k in range(4):
            for h in range(4):
                ii = _wrap_matlab(x1 + 1 - k, M)
                jj = _wrap_matlab(x2 + 1 - h, N)
                wx = (1 / (4 * dx)) * (1 + math.cos((math.pi / (2 * dx)) * (dx * (x1 + 1 - k) - array1[i])))
                wy = (1 / (4 * dx)) * (1 + math.cos((math.pi / (2 * dx)) * (dx * (x2 + 1 - h) - array2[i])))
                uarray[i] += u[ii, jj] * wx * wy * dx * dx
                varray[i] += v[ii, jj] * wx * wy * dx * dx
    return uarray, varray


def test_cosine4_matches_matlab_2008_spread_and_interpolate():
    """DeltaCoupling('cosine4') on the 2008 grid (16^2, L = 0.4) equals the MATLAB loops."""
    grid = Grid.periodic_box(0.4, 0.4, 16, 16)
    lay = layout_collocated(grid)
    dx, ds = grid.dx, 0.4 / 32
    dc = DeltaCoupling("cosine4")
    rng = np.random.default_rng(17)
    # the 2008 plate (x ~ 0.2, y in [0.1, 0.29]) plus points at the edges and on nodes
    plate = np.column_stack([0.2 + 0.001 * rng.normal(size=16), 0.1 + ds * np.arange(16)])
    extra = np.array([[0.01, 0.39], [0.395, 0.005], [0.2, 0.2], [0.0, 0.0], [0.3999, 0.0001]])
    X = np.vstack([plate, extra])
    Q = X.shape[0]
    f = rng.normal(size=(Q, 2))
    f1_ref, f2_ref = _matlab_forcespread(X[:, 0], X[:, 1], f[:, 0], f[:, 1], Q, dx, ds, 16, 16)
    Fx, Fy = dc.spread_vector(f, X, ds, lay, lay)
    assert np.allclose(Fx, f1_ref, rtol=1e-12, atol=1e-12 * np.abs(f1_ref).max())
    assert np.allclose(Fy, f2_ref, rtol=1e-12, atol=1e-12 * np.abs(f2_ref).max())
    u, v = rng.normal(size=(16, 16)), rng.normal(size=(16, 16))
    u_ref, v_ref = _matlab_move_boundary_velocity(X[:, 0], X[:, 1], Q, u, v, dx, 16, 16)
    UV = dc.interpolate_vector(u, v, X, lay, lay)
    assert np.allclose(UV[:, 0], u_ref, rtol=1e-12, atol=1e-13)
    assert np.allclose(UV[:, 1], v_ref, rtol=1e-12, atol=1e-13)
