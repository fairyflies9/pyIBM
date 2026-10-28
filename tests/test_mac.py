"""Tests for pyibm.fluid.mac: the MAC projection back end (scheme described in
docs/numerics.md)."""

import warnings

import numpy as np
import pytest

from pyibm.grid import Grid
from pyibm.fluid.mac import BoundaryConditions, MACSolver

D = "dirichlet"
P = "periodic"
O = "outflow"


def closed_box(N, Lx=1.0, Ly=1.0, **bc):
    g = Grid(Lx, Ly, N, N, periodic_x=False, periodic_y=False)
    return g, BoundaryConditions(D, D, D, D, **bc)


def run_midpoint(s, n, Fu=None, Fv=None):
    for k in range(n):
        t = k * s.dt
        uh, vh = s.half_step(Fu, Fv, t)
        s.full_step(Fu, Fv, uh, vh, t)


# ------------------------------------------------------------------ validation
def test_boundary_conditions_validation():
    with pytest.raises(ValueError):
        BoundaryConditions(P, D, D, D)                 # periodic on one side only
    with pytest.raises(ValueError):
        BoundaryConditions(D, D, P, D)
    with pytest.raises(ValueError):
        BoundaryConditions(O, O, D, D)                 # two outflow sides
    with pytest.raises(ValueError):
        BoundaryConditions("wall", D, D, D)
    b = BoundaryConditions(P, P, D, O)
    assert b.periodic_x and not b.periodic_y and b.outflow_side == "top"


def test_solver_validation():
    g = Grid(1.0, 1.0, 16, 16, periodic_x=False, periodic_y=False)
    with pytest.raises(ValueError):
        MACSolver(g, 1.0, 0.1, 1e-3, BoundaryConditions(P, P, D, D))   # grid says not periodic
    with pytest.raises(ValueError):
        MACSolver(g, 1.0, 0.1, 1e-3, BoundaryConditions(D, D, D, D), advection="qwerty")
    with pytest.raises(ValueError):
        MACSolver(g, 1.0, 0.1, 1e-3, BoundaryConditions(D, D, D, D), viscous="semi")
    with pytest.raises(ValueError):
        MACSolver(g, 1.0, 0.1, 1e-3, BoundaryConditions(D, D, D, D), projection="rotational")
    s = MACSolver(g, 1.0, 0.1, 1e-3, BoundaryConditions(D, D, D, D))
    assert s.projection == "nonincremental"
    assert s.u.shape == (17, 16) and s.v.shape == (16, 17) and s.p.shape == (16, 16)
    assert s.layout_u.shape == s.u.shape and s.layout_v.shape == s.v.shape


# ------------------------------------------------------------------ ghost construction
def test_extend_dirichlet_walls():
    g = Grid(1.0, 0.5, 16, 8, periodic_x=False, periodic_y=False)
    left = lambda y, t: (np.sin(y) + t, np.cos(y))
    top = lambda x, t: (2.0 + x, 0.3 * x * t)
    bcs = BoundaryConditions(D, D, D, D, bc_left=left, bc_top=top)
    s = MACSolver(g, 1.0, 0.1, 1e-3, bcs)
    rng = np.random.default_rng(3)
    u = rng.standard_normal(g.shape_u())
    v = rng.standard_normal(g.shape_v())
    t = 0.7
    ue = s.extend_u(u, t)
    ve = s.extend_v(v, t)
    assert ue.shape == (g.Nx + 3, g.Ny + 2) and ve.shape == (g.Nx + 2, g.Ny + 3)
    # interior faces untouched, normal components on the wall faces exact
    assert np.array_equal(ue[2:-2, 1:-1], u[1:-1])
    assert np.allclose(ue[1, 1:-1], np.sin(g.yc) + t)        # left wall faces
    assert np.allclose(ue[-2, 1:-1], 0.0)                     # right wall at rest
    assert np.allclose(ve[1:-1, 1], 0.0)                      # bottom wall
    assert np.allclose(ve[1:-1, -2], 0.3 * g.xc * t)          # top wall v
    # tangential ghosts: the average of ghost and first interior value is the wall value
    assert np.allclose(0.5 * (ue[2:-2, -1] + ue[2:-2, -2]), 2.0 + g.xu[1:-1])   # top, u
    assert np.allclose(0.5 * (ue[2:-2, 0] + ue[2:-2, 1]), 0.0)                 # bottom, u
    assert np.allclose(0.5 * (ve[0, 1:-1] + ve[1, 1:-1]), np.cos(g.yv))        # left, v
    assert np.allclose(0.5 * (ve[-1, 1:-1] + ve[-2, 1:-1]), 0.0)               # right, v


def test_extend_outflow_and_periodic():
    g = Grid(1.0, 0.5, 16, 8, periodic_x=False, periodic_y=False)
    s = MACSolver(g, 1.0, 0.1, 1e-3, BoundaryConditions(D, O, D, D))
    rng = np.random.default_rng(4)
    u = rng.standard_normal(g.shape_u())
    v = rng.standard_normal(g.shape_v())
    ue, ve = s.extend_u(u, 0.0), s.extend_v(v, 0.0)
    assert np.array_equal(ue[-2, 1:-1], u[-1]) and np.array_equal(ue[-1], ue[-2])   # zero gradient
    assert np.array_equal(ve[-1], ve[-2])
    gp = Grid(1.0, 0.5, 16, 8, periodic_x=True, periodic_y=True)
    sp_ = MACSolver(gp, 1.0, 0.1, 1e-3, BoundaryConditions(P, P, P, P))
    u = rng.standard_normal(gp.shape_u())
    v = rng.standard_normal(gp.shape_v())
    ue, ve = sp_.extend_u(u, 0.0), sp_.extend_v(v, 0.0)
    assert np.array_equal(ue[1:gp.Nx + 1, 1:-1], u)
    assert np.array_equal(ue[0, 1:-1], u[-1]) and np.array_equal(ue[gp.Nx + 1, 1:-1], u[0])
    assert np.array_equal(ue[gp.Nx + 2, 1:-1], u[1])
    assert np.array_equal(ue[1:gp.Nx + 1, 0], u[:, -1]) and np.array_equal(ue[1:gp.Nx + 1, -1], u[:, 0])
    assert np.array_equal(ve[1:-1, 1:gp.Ny + 1], v)
    assert np.array_equal(ve[1:-1, 0], v[:, -1]) and np.array_equal(ve[1:-1, gp.Ny + 2], v[:, 1])
    assert np.array_equal(ve[0, 1:gp.Ny + 1], v[-1]) and np.array_equal(ve[-1, 1:gp.Ny + 1], v[0])


@pytest.mark.parametrize("periodic, kinds", [
    ((False, False), (D, D, D, D)),
    ((False, False), (D, O, D, D)),
    ((False, False), (D, D, O, D)),
    ((True, False), (P, P, D, D)),
    ((True, False), (P, P, D, O)),
    ((False, True), (D, D, P, P)),
])
def test_helmholtz_matrix_matches_stencil(periodic, kinds):
    """The kron-built -L_h on the free faces equals the 5-point stencil on the extended
    field minus its boundary-data part (which goes to the right-hand side)."""
    g = Grid(1.0, 0.75, 12, 8, periodic_x=periodic[0], periodic_y=periodic[1])
    lid = lambda x, t: (np.ones_like(x), np.zeros_like(x))
    inflow = lambda y, t: (y * (1 - y), 0.1 * np.ones_like(y))
    kw = {}
    if kinds[3] == D:
        kw["bc_top"] = lid
    if kinds[0] == D:
        kw["bc_left"] = inflow
    s = MACSolver(g, 1.0, 0.2, 1e-2, BoundaryConditions(*kinds, **kw))
    rng = np.random.default_rng(5)
    u = rng.standard_normal(g.shape_u())
    v = rng.standard_normal(g.shape_v())
    Ku, Kv = s._laplacian_matrices()
    t = 0.3
    lap_u = s._lap_u(s.extend_u(u, t)) - s._lap_u(s.extend_u(np.zeros_like(u), t))
    lap_v = s._lap_v(s.extend_v(v, t)) - s._lap_v(s.extend_v(np.zeros_like(v), t))
    uf = s._full_u(u)[s._iu]
    vf = s._full_v(v)[:, s._jv]
    assert np.allclose((Ku @ uf.ravel()).reshape(uf.shape), -lap_u[s._iu], atol=1e-10)
    assert np.allclose((Kv @ vf.ravel()).reshape(vf.shape), -lap_v[:, s._jv], atol=1e-10)
    # the LU solves the Helmholtz system
    c = 0.2 * 1e-2
    lu_u, lu_v = s._helmholtz(c)
    x = lu_u.solve(uf.ravel())
    assert np.allclose(x + c * (Ku @ x), uf.ravel(), atol=1e-12)
    assert s._helmholtz(c) is s._helmholtz(c)          # cached, not rebuilt


# ------------------------------------------------------------------ divergence and compatibility
def test_divergence_free_closed_box_and_periodic():
    g, bcs = closed_box(32, bc_top=lambda x, t: (np.ones_like(x), np.zeros_like(x)))
    s = MACSolver(g, 1.0, 0.01, 5e-3, bcs)
    for k in range(5):
        s.step(None, None, k * s.dt)
        assert np.max(np.abs(s.divergence())) < 1e-12
    run_midpoint(s, 5)
    assert np.max(np.abs(s.divergence())) < 1e-12
    uh, vh = s.half_step(None, None, s.t)
    assert np.max(np.abs(s.divergence(uh, vh))) < 1e-12
    gp = Grid.periodic_box(1.0, 1.0, 32, 32)
    sp_ = MACSolver(gp, 1.0, 0.01, 5e-3, BoundaryConditions(P, P, P, P))
    rng = np.random.default_rng(6)
    sp_.u[:] = rng.standard_normal(gp.shape_u())
    sp_.v[:] = rng.standard_normal(gp.shape_v())
    sp_.step(None, None, 0.0)
    assert np.max(np.abs(sp_.divergence())) < 1e-12
    run_midpoint(sp_, 3)
    assert np.max(np.abs(sp_.divergence())) < 1e-12


def test_divergence_free_with_outflow():
    g = Grid(2.0, 1.0, 32, 16, periodic_x=False, periodic_y=False)
    inflow = lambda y, t: (4 * y * (1 - y) * min(1.0, 2 * t + 0.1), np.zeros_like(y))
    s = MACSolver(g, 1.0, 0.05, 5e-3, BoundaryConditions(D, O, D, D, bc_left=inflow))
    for k in range(5):
        s.step(None, None, k * s.dt)
        assert np.max(np.abs(s.divergence())) < 1e-12
    run_midpoint(s, 5)
    assert np.max(np.abs(s.divergence())) < 1e-12
    # mass conservation: what comes in through the left goes out through the right
    q_in = s.u[0].sum() * g.dy
    q_out = s.u[-1].sum() * g.dy
    assert abs(q_in - q_out) < 1e-12 and q_in > 0.1


def test_net_inflow_without_outflow_raises():
    g, bcs = closed_box(16, bc_left=lambda y, t: (np.ones_like(y), np.zeros_like(y)))
    s = MACSolver(g, 1.0, 0.1, 1e-3, bcs)
    with pytest.raises(ValueError, match="net inflow"):
        s.step(None, None, 0.0)
    with pytest.raises(ValueError, match="net inflow"):
        s.half_step(None, None, 0.0)
    # balanced in/out through opposite walls is fine (pure Neumann Poisson, compatible)
    thru = lambda y, t: (np.ones_like(y), np.zeros_like(y))
    g, bcs = closed_box(16, bc_left=thru, bc_right=thru)
    s = MACSolver(g, 1.0, 0.1, 1e-3, bcs)
    s.step(None, None, 0.0)
    assert np.max(np.abs(s.divergence())) < 1e-12


# ------------------------------------------------------------------ physics
def test_poiseuille_channel():
    """Parabolic inflow on the left, outflow on the right, no-slip walls: the profile stays
    parabolic and dp/dx = -8 mu U / Ly^2 (Chorin 1967 test problem; outflow instead of a
    prescribed outlet). The 1e-2 tolerance covers the O(dy^2) wall-ghost error of the
    discrete parabola (1.5e-3 here, see the incremental test below) and the O(dt) tangential
    slip of the non-incremental projection; measured: profile 1.6e-3, dp/dx 3.8e-3, its
    variation across y 5.7e-3."""
    Lx, Ly, Nx, Ny = 2.0, 1.0, 64, 32
    U, mu, rho = 1.0, 0.1, 1.0
    g = Grid(Lx, Ly, Nx, Ny, periodic_x=False, periodic_y=False)
    inflow = lambda y, t: (4 * U * y * (Ly - y) / Ly**2, np.zeros_like(y))
    s = MACSolver(g, rho, mu, 2e-3, BoundaryConditions(D, O, D, D, bc_left=inflow))
    X, Y = g.meshgrid("u")
    s.u[:] = 4 * U * Y * (Ly - Y) / Ly**2
    run_midpoint(s, int(round(1.5 / s.dt)))
    exact = 4 * U * g.yc * (Ly - g.yc) / Ly**2
    assert np.max(np.abs(s.u[Nx // 2] - exact)) < 1e-2 * U
    assert np.max(np.abs(s.v)) < 1e-2 * U
    dpdx = (s.p[-1] - s.p[0]) / (g.xc[-1] - g.xc[0])
    expected = -8 * mu * U / Ly**2
    assert abs(dpdx.mean() - expected) < 1e-2 * abs(expected)
    assert np.max(np.abs(dpdx - dpdx.mean())) < 1e-2 * abs(expected)     # same at every y
    assert np.max(np.abs(s.divergence())) < 1e-12
    assert abs(s.u[0].sum() - s.u[-1].sum()) * g.dy < 1e-12
    assert abs(s.p[-1].mean()) < 1e-2 * abs(expected) * Lx                 # p = 0 at the outflow


def poiseuille_discrete(g, U, mu, Ly):
    """The discrete fully developed channel flow: with the reflection ghost at the walls the
    5-point Laplacian of y(Ly - y) + dy^2/4 is -2 at every cell centre, so u = A (y(Ly - y)
    + dy^2/4) with A fixed by the inflow flux (midpoint sum of the parabola). Returns the
    profile and dp/dx = -2 mu A, which differs from -8 mu U/Ly^2 by about 1.5 dy^2/Ly^2."""
    dy = g.dy
    A = (4 * U / Ly**2) * (Ly**2 / 6 + dy**2 / 12) / (Ly**2 / 6 + dy**2 / 3)
    return A * (g.yc * (Ly - g.yc) + dy**2 / 4), -2 * mu * A


@pytest.mark.parametrize("scheme, dt", [("midpoint", 2e-3), ("step", 4e-3)])
def test_poiseuille_channel_incremental(scheme, dt):
    """With the incremental projection the steady state has no splitting error (the pressure
    increment vanishes), so the developed flow is the discrete fully developed solution
    whatever dt: at T = 1 dp/dx over the middle half of the channel matches it to 3.4e-5
    relative and the profile to 1e-6 (measured, the same at dt = 1e-3, 2e-3, 4e-3 and for
    both schemes), and dp/dx matches -8 mu U/Ly^2 to 1.5e-3, the wall-ghost error 1.5 dy^2
    of the discrete parabola. The non-incremental projection is 1.3e-3 off the discrete
    dp/dx at dt = 2e-3 with the midpoint scheme (2.5e-3 with step) and halves with dt."""
    Lx, Ly, Nx, Ny = 2.0, 1.0, 64, 32
    U, mu, rho = 1.0, 0.1, 1.0
    g = Grid(Lx, Ly, Nx, Ny, periodic_x=False, periodic_y=False)
    inflow = lambda y, t: (4 * U * y * (Ly - y) / Ly**2, np.zeros_like(y))
    s = MACSolver(g, rho, mu, dt, BoundaryConditions(D, O, D, D, bc_left=inflow),
                  projection="incremental")
    X, Y = g.meshgrid("u")
    s.u[:] = 4 * U * Y * (Ly - Y) / Ly**2
    n = int(round(1.0 / dt))
    if scheme == "midpoint":
        run_midpoint(s, n)
    else:
        for k in range(n):
            s.step(None, None, k * dt)
    u_disc, dpdx_disc = poiseuille_discrete(g, U, mu, Ly)
    i1, i2 = Nx // 4, 3 * Nx // 4
    dpdx = (s.p[i2] - s.p[i1]) / (g.xc[i2] - g.xc[i1])
    assert abs(dpdx.mean() / dpdx_disc - 1) < 1e-4
    assert np.max(np.abs(dpdx / dpdx_disc - 1)) < 2e-4                       # same at every y
    assert np.max(np.abs(s.u[Nx // 2] - u_disc)) < 1e-5 * U
    expected = -8 * mu * U / Ly**2
    assert abs(dpdx.mean() / expected - 1) < 1.5 * g.dy**2 + 1e-4
    assert np.max(np.abs(s.divergence())) < 1e-12
    assert abs(s.u[0].sum() - s.u[-1].sum()) * g.dy < 1e-12


def tg_wall_case(case, N, dt, projection, viscous="implicit", nu=0.01):
    """Taylor-Green vortex shifted by (0.13, 0.29) so no wall is a symmetry line, with the
    exact, decaying velocity as Dirichlet data on the walls. ``case``: "box" (four walls),
    "channel" (periodic x, walls in y), "outflow" (outflow on the right), "outflow_top"
    (periodic x, wall at the bottom, outflow at the top). The pressure varies along every
    wall."""
    x0, y0 = 0.13, 0.29
    F = lambda t: np.exp(-8 * np.pi**2 * nu * t)
    Uex = lambda x, y, t: -np.cos(2 * np.pi * (x + x0)) * np.sin(2 * np.pi * (y + y0)) * F(t)
    Vex = lambda x, y, t: np.sin(2 * np.pi * (x + x0)) * np.cos(2 * np.pi * (y + y0)) * F(t)
    kinds = {"box": (D, D, D, D), "channel": (P, P, D, D), "outflow": (D, O, D, D),
             "outflow_top": (P, P, D, O)}[case]
    g = Grid(1.0, 1.0, N, N, periodic_x=kinds[0] == P, periodic_y=False)
    bc = {}
    for side, kind, at in zip(("left", "right", "bottom", "top"), kinds, (0.0, 1.0, 0.0, 1.0)):
        if kind == D and side in ("left", "right"):
            bc["bc_" + side] = lambda y, t, at=at: (Uex(at, y, t), Vex(at, y, t))
        elif kind == D:
            bc["bc_" + side] = lambda x, t, at=at: (Uex(x, at, t), Vex(x, at, t))
    s = MACSolver(g, 1.0, nu, dt, BoundaryConditions(*kinds, **bc), advection="central",
                  viscous=viscous, projection=projection)
    Xu, Yu = g.meshgrid("u")
    Xv, Yv = g.meshgrid("v")
    s.u[:] = Uex(Xu, Yu, 0.0)
    s.v[:] = Vex(Xv, Yv, 0.0)
    return s


@pytest.mark.parametrize("case", ["box", "channel", "outflow", "outflow_top"])
def test_incremental_projection_second_order_with_moving_walls(case):
    """Time-varying Dirichlet data with a tangential pressure gradient along the walls, the
    midpoint scheme with central advection and implicit viscosity, 16^2, T = 0.5. We compare
    runs at dt = 0.02, 0.01, 0.005: the difference between successive runs falls by 4 with
    the incremental projection (measured 3.92 to 4.08 over the four cases) and by 2 with the
    non-incremental one (1.93 to 2.09), whose O(dt) wall slip limits it to first order.
    Against an incremental dt = T/1600 reference on 32^2 the incremental errors are 3.9e-4
    to 2.4e-5 (box) for dt = 0.02 to 0.005 and the non-incremental 1.1e-2 to 2.8e-3. The
    first step starts from p = 0, which costs nothing measurable here."""
    T = 0.5
    ratios = {}
    for projection in ("nonincremental", "incremental"):
        runs = []
        for dt in (0.02, 0.01, 0.005):
            s = tg_wall_case(case, 16, dt, projection)
            run_midpoint(s, int(round(T / dt)))
            assert np.max(np.abs(s.divergence())) < 1e-12
            assert np.isfinite(s.p).all()
            runs.append(s)
        d = [max(np.max(np.abs(a.u - b.u)), np.max(np.abs(a.v - b.v)))
             for a, b in zip(runs[:-1], runs[1:])]
        ratios[projection] = d[0] / d[1]
    assert ratios["incremental"] > 3.5
    assert ratios["nonincremental"] < 2.5


def test_projections_agree_without_a_splitting_error():
    """The two projections differ only by the gradient h G p_old in the predictor. When the
    viscous solve commutes with the gradient (doubly periodic box) or there is no viscous
    solve (explicit viscosity), the projection removes that gradient exactly, so u, v and p
    are the same to round-off; with implicit viscosity and walls they are not."""
    gp = Grid.periodic_box(1.0, 1.0, 16, 16)
    Xu, Yu = gp.meshgrid("u")
    Xv, Yv = gp.meshgrid("v")
    Xc, Yc = gp.meshgrid("centres")
    rng = np.random.default_rng(7)
    u0, v0 = rng.standard_normal(gp.shape_u()), rng.standard_normal(gp.shape_v())
    out = {}
    for projection in ("nonincremental", "incremental"):
        s = MACSolver(gp, 1.0, 0.02, 2e-3, BoundaryConditions(P, P, P, P), projection=projection)
        s.u[:], s.v[:] = u0, v0
        Gx, Gy = s._gradient(np.sin(2 * np.pi * Xc) * np.cos(4 * np.pi * Yc))
        Fu = s._store_u(Gx) + np.sin(2 * np.pi * Yu)          # gradient plus a shear force
        Fv = s._store_v(Gy) + 0.3 * np.cos(2 * np.pi * Xv)
        for k in range(5):
            s.step(Fu, Fv, k * s.dt)
        run_midpoint(s, 5, Fu, Fv)
        out[projection] = s
    a, b = out["nonincremental"], out["incremental"]
    assert np.max(np.abs(a.u - b.u)) < 1e-12 and np.max(np.abs(a.v - b.v)) < 1e-12
    assert np.max(np.abs((a.p - a.p.mean()) - (b.p - b.p.mean()))) < 1e-11
    for viscous, same in (("explicit", True), ("implicit", False)):
        s = [tg_wall_case("box", 16, 0.01, pr, viscous=viscous)
             for pr in ("nonincremental", "incremental")]
        for x in s:
            run_midpoint(x, 10)
        du = max(np.max(np.abs(s[0].u - s[1].u)), np.max(np.abs(s[0].v - s[1].v)))
        assert (du < 1e-12) if same else (du > 1e-5)


# Ghia, Ghia & Shin 1982, J. Comput. Phys. 48:387-411, Table I, Re = 100:
# u along the vertical line through the cavity centre (y, u).
GHIA_RE100_Y = np.array([1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172,
                         0.5000, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0.0000])
GHIA_RE100_U = np.array([1.00000, 0.84123, 0.78871, 0.73722, 0.68717, 0.23151, 0.00332, -0.13641,
                         -0.20581, -0.21090, -0.15662, -0.10150, -0.06434, -0.04775, -0.04192,
                         -0.03717, 0.00000])
# Table II, Re = 100: v along the horizontal line through the centre (x, v). The same
# tables are in examples/lid_driven_cavity/ghia1982.py.
GHIA_RE100_X = np.array([1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047, 0.5000,
                         0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625, 0.0000])
GHIA_RE100_V = np.array([0.00000, -0.05906, -0.07391, -0.08864, -0.10313, -0.16914, -0.22445,
                         -0.24533, 0.05454, 0.17527, 0.17507, 0.16077, 0.12317, 0.10890, 0.10091,
                         0.09233, 0.00000])


def test_lid_driven_cavity_ghia():
    """Re = 100 on 64^2 with the one-step scheme, compared with Ghia et al. 1982 on the
    centrelines. Measured: u(y) within 0.0071 everywhere (under 5% relative wherever
    |u| > 0.1); v(x) within 0.0068 everywhere, under 5% relative where |v| > 0.15 and within
    1.2% at every point left of the centre (largest 1.15% at x = 0.0703); the largest
    relative errors (6.1-6.7%) are at the four points next to the right wall, x >= 0.9453."""
    N = 64
    g, bcs = closed_box(N, bc_top=lambda x, t: (np.ones_like(x), np.zeros_like(x)))
    s = MACSolver(g, 1.0, 0.01, 0.5 / N, bcs)
    for k in range(int(round(10.0 / s.dt))):
        s.step(None, None, k * s.dt)
    assert s.cfl() < 1.0
    # u on the vertical centreline x = 0.5 is a stored face column; interpolate in y
    yy = np.concatenate([[0.0], g.yc, [1.0]])
    ucl = np.interp(GHIA_RE100_Y, yy, np.concatenate([[0.0], s.u[N // 2], [1.0]]))
    assert np.max(np.abs(ucl - GHIA_RE100_U)) < 0.05                       # 5% of the lid speed
    big = np.abs(GHIA_RE100_U) > 0.1
    assert np.all(np.abs(ucl[big] - GHIA_RE100_U[big]) < 0.05 * np.abs(GHIA_RE100_U[big]))
    xx = np.concatenate([[0.0], g.xc, [1.0]])
    vcl = np.interp(GHIA_RE100_X, xx, np.concatenate([[0.0], s.v[:, N // 2], [0.0]]))
    assert np.max(np.abs(vcl - GHIA_RE100_V)) < 0.05
    big = np.abs(GHIA_RE100_V) > 0.15
    assert np.all(np.abs(vcl[big] - GHIA_RE100_V[big]) < 0.05 * np.abs(GHIA_RE100_V[big]))
    left = (GHIA_RE100_X > 0) & (GHIA_RE100_X < 0.5)
    assert np.all(np.abs(vcl[left] - GHIA_RE100_V[left]) < 0.02 * np.abs(GHIA_RE100_V[left]))
    # the primary vortex spins clockwise: negative vorticity in the core
    om = s.vorticity()
    assert om[N // 2, 3 * N // 4] < 0
    assert np.max(np.abs(s.divergence())) < 1e-12


def test_solid_body_rotation_time_varying_walls():
    """Walls moving as u_b = Omega(t)(-(y - yc), x - xc), Omega = sin t, on a small box with
    large viscosity: the interior follows the rigid rotation. The lag behind the walls is
    O(Omega' L^2/nu) and vanishes where Omega' = 0 (t = pi/2); measured deviation there
    1.6e-4, and 4e-3 at t = pi where the lag peaks."""
    N, L, nu = 32, 1.0, 2.0
    xc = yc = 0.5 * L
    Om = np.sin
    left = lambda y, t: (-Om(t) * (y - yc), Om(t) * (0.0 - xc) * np.ones_like(y))
    right = lambda y, t: (-Om(t) * (y - yc), Om(t) * (L - xc) * np.ones_like(y))
    bottom = lambda x, t: (-Om(t) * (0.0 - yc) * np.ones_like(x), Om(t) * (x - xc))
    top = lambda x, t: (-Om(t) * (L - yc) * np.ones_like(x), Om(t) * (x - xc))
    g = Grid(L, L, N, N, periodic_x=False, periodic_y=False)
    s = MACSolver(g, 1.0, nu, 2e-3, BoundaryConditions(D, D, D, D, left, right, bottom, top))
    Xu, Yu = g.meshgrid("u")
    Xv, Yv = g.meshgrid("v")
    n_quarter = int(round(0.5 * np.pi / s.dt))
    run_midpoint(s, n_quarter)
    t = n_quarter * s.dt
    u_ex, v_ex = -Om(t) * (Yu - yc), Om(t) * (Xv - xc)
    speed = 0.5 * np.sqrt(2) * abs(Om(t))
    assert np.max(np.abs(s.u - u_ex)) < 2e-3 * speed
    assert np.max(np.abs(s.v - v_ex)) < 2e-3 * speed
    for (i, j) in [(N // 4, N // 2), (N // 2, N // 4), (3 * N // 4, 3 * N // 4)]:
        assert s.u[i, j] == pytest.approx(u_ex[i, j], abs=2e-3 * speed)
        assert s.v[i, j] == pytest.approx(v_ex[i, j], abs=2e-3 * speed)
    om = s.vorticity()
    assert np.allclose(om, 2 * Om(t), atol=1e-2)
    assert np.max(np.abs(s.divergence())) < 1e-12
    # the wall faces carry the data at the current time exactly
    assert np.allclose(s.u[0], -Om(t) * (g.yc - yc)) and np.allclose(s.v[:, -1], Om(t) * (g.xc - xc))
    # through Omega = 0 the lag is largest; still within 1% of the peak speed
    for k in range(n_quarter, 2 * n_quarter):
        uh, vh = s.half_step(None, None, k * s.dt)
        s.full_step(None, None, uh, vh, k * s.dt)
    t = 2 * n_quarter * s.dt
    assert np.max(np.abs(s.u + Om(t) * (Yu - yc))) < 1e-2 * 0.5 * np.sqrt(2)


# Linear manufactured solution u = a(t) y + b(t), v = c(t) x + d(t): divergence-free, zero
# Laplacian, and every discrete operator (averages, central products, reflection ghosts) is
# exact on it, so only the time discretisation and the time levels of the boundary data
# are tested.
LIN_A = (lambda t: 1 + 0.5 * np.sin(3 * t), lambda t: 1.5 * np.cos(3 * t))
LIN_B = (lambda t: 0.3 * np.cos(2 * t), lambda t: -0.6 * np.sin(2 * t))
LIN_C = (lambda t: -0.7 + 0.4 * np.sin(5 * t), lambda t: 2.0 * np.cos(5 * t))
LIN_D = (lambda t: 0.2 * np.sin(4 * t), lambda t: 0.8 * np.cos(4 * t))


def linear_mms_error(dt, viscous, projection, N=16, T=0.5, nu=0.05):
    (a, da), (b, db), (c, dc), (d, dd) = LIN_A, LIN_B, LIN_C, LIN_D
    Uex = lambda x, y, t: a(t) * y + b(t)
    Vex = lambda x, y, t: c(t) * x + d(t)
    g = Grid(1.0, 1.0, N, N, periodic_x=False, periodic_y=False)
    bc = dict(bc_left=lambda y, t: (Uex(0.0, y, t), Vex(0.0, y, t)),
              bc_right=lambda y, t: (Uex(1.0, y, t), Vex(1.0, y, t)),
              bc_bottom=lambda x, t: (Uex(x, 0.0, t), Vex(x, 0.0, t)),
              bc_top=lambda x, t: (Uex(x, 1.0, t), Vex(x, 1.0, t)))
    s = MACSolver(g, 1.0, nu, dt, BoundaryConditions(D, D, D, D, **bc), advection="central",
                  viscous=viscous, projection=projection)
    Xu, Yu = g.meshgrid("u")
    Xv, Yv = g.meshgrid("v")
    s.u[:], s.v[:] = Uex(Xu, Yu, 0.0), Vex(Xv, Yv, 0.0)
    n = int(round(T / dt))
    for k in range(n):
        t = k * dt
        th = t + 0.5 * dt                       # u_t + (u.grad)u = f, p = 0
        Fu = da(th) * Yu + db(th) + a(th) * (c(th) * Xu + d(th))
        Fv = dc(th) * Xv + dd(th) + c(th) * (a(th) * Yv + b(th))
        uh, vh = s.half_step(Fu, Fv, t)
        s.full_step(Fu, Fv, uh, vh, t)
    t = n * dt
    return max(np.max(np.abs(s.u - Uex(Xu, Yu, t))), np.max(np.abs(s.v - Vex(Xv, Yv, t))))


@pytest.mark.parametrize("viscous, projection", [("implicit", "nonincremental"),
                                                 ("explicit", "nonincremental"),
                                                 ("implicit", "incremental")])
def test_boundary_data_time_levels(viscous, projection):
    """Pins the time levels of the wall data in the midpoint scheme with the linear
    manufactured solution above (16^2, nu = 0.05, T = 0.5, central advection). Measured
    error at dt = 1.25e-3: 1.18e-7 (implicit), 1.78e-7 (explicit), 1.19e-7 (implicit,
    incremental), falling by 4.0-4.1 from dt = 2.5e-3. Each of these mistakes gives 4.5e-4
    or more and first order: Helmholtz boundary data at the advection time instead of
    t_new (implicit only), the explicit Laplacian's data at t_new, the advection ghosts at
    t_new."""
    e1 = linear_mms_error(2.5e-3, viscous, projection)
    e2 = linear_mms_error(1.25e-3, viscous, projection)
    assert e2 < 1e-6
    assert e1 / e2 > 3.5


@pytest.mark.parametrize("scheme", ["step", "midpoint"])
def test_taylor_green_periodic(scheme):
    """u = -cos(2 pi x) sin(2 pi y) exp(-8 pi^2 nu t) on the doubly periodic box. We measure
    the amplitude of the mode by projection, amp = sum(u u0)/sum(u0 u0), which is free of the
    grid sampling of the initial maximum (max|u0| on the u-faces is sin(15 pi/32) = 0.995).
    With central advection the mode decays with the discrete eigenvalue
    lam_h = (4/dx^2) sin^2(pi dx) + (4/dy^2) sin^2(pi dy) (3.2e-3 relative below 8 pi^2 at 32^2,
    so the amplitude at T = 0.5 is 1.27e-3 above the analytic one) and the time factor of
    the viscous scheme: Crank-Nicolson for the midpoint scheme, backward Euler for step;
    measured to 3e-14. The blend adds about 1.1e-2 of extra damping (measured -1.14e-2
    midpoint, -1.12e-2 step)."""
    N, nu, dt, T = 32, 0.01, 2e-3, 0.5
    g = Grid.periodic_box(1.0, 1.0, N, N)
    Xu, Yu = g.meshgrid("u")
    Xv, Yv = g.meshgrid("v")
    u0 = -np.cos(2 * np.pi * Xu) * np.sin(2 * np.pi * Yu)
    v0 = np.sin(2 * np.pi * Xv) * np.cos(2 * np.pi * Yv)
    n = int(round(T / dt))
    lam_h = (4 / g.dx**2) * np.sin(np.pi * g.dx)**2 + (4 / g.dy**2) * np.sin(np.pi * g.dy)**2
    if scheme == "step":
        factor = (1 + nu * lam_h * dt) ** (-n)
    else:
        factor = ((1 - 0.5 * nu * lam_h * dt) / (1 + 0.5 * nu * lam_h * dt)) ** n
    amp = {}
    for adv in ("central", "blend"):
        s = MACSolver(g, 1.0, nu, dt, BoundaryConditions(P, P, P, P), advection=adv)
        s.u[:], s.v[:] = u0, v0
        ke0 = s.kinetic_energy()
        if scheme == "step":
            for k in range(n):
                s.step(None, None, k * dt)
        else:
            run_midpoint(s, n)
        amp[adv] = (np.sum(s.u * u0) / np.sum(u0 * u0), np.sum(s.v * v0) / np.sum(v0 * v0))
        assert np.max(np.abs(s.divergence())) < 1e-12
        # central keeps the shape of the mode exactly, the blend nearly (3.4e-6)
        assert s.kinetic_energy() == pytest.approx(ke0 * amp[adv][0]**2,
                                                   rel=1e-10 if adv == "central" else 1e-4)
        assert abs(s.mean_velocity()[0]) < 1e-12 and abs(s.mean_velocity()[1]) < 1e-12
    for a in amp["central"]:
        assert abs(a / factor - 1) < 1e-10
        assert abs(a / np.exp(-nu * lam_h * T) - 1) < (1e-6 if scheme == "midpoint" else 1e-3)
    for a, ac in zip(amp["blend"], amp["central"]):
        assert a < ac
        assert -2e-2 < a / factor - 1 < -5e-3


# ------------------------------------------------------------------ forces, schemes, diagnostics
def test_body_force_accelerates_periodic_box():
    """A uniform force F in x on the doubly periodic box adds dt F/rho to the mean velocity
    per step (the projection leaves the box mean alone)."""
    g = Grid.periodic_box(1.0, 1.0, 16, 16)
    rho, F = 2.0, 3.0
    s = MACSolver(g, rho, 0.01, 1e-3, BoundaryConditions(P, P, P, P))
    Fu = np.full(g.shape_u(), F)
    Fv = np.zeros(g.shape_v())
    s.step(Fu, Fv, 0.0)
    assert s.mean_velocity()[0] == pytest.approx(s.dt * F / rho, rel=1e-12)
    uh, vh = s.half_step(Fu, Fv, s.dt)
    assert np.mean(uh) == pytest.approx(s.dt * F / rho * 1.5, rel=1e-12)
    s.full_step(Fu, Fv, uh, vh, s.dt)
    assert s.mean_velocity()[0] == pytest.approx(2 * s.dt * F / rho, rel=1e-12)
    assert abs(s.mean_velocity()[1]) < 1e-14
    # in a closed box the same force is absorbed by the pressure gradient: exactly so for
    # an inviscid fluid; with viscosity the implicit solve sees the predictor's jump at the
    # wall faces and leaves a splitting remainder of order (dt F/rho)*(nu dt/h^2)
    gc, bcs = closed_box(16)
    sc = MACSolver(gc, rho, 0.0, 1e-3, bcs)
    sc.step(np.full(gc.shape_u(), F), np.zeros(gc.shape_v()), 0.0)
    assert np.max(np.abs(sc.u)) < 1e-12 and np.max(np.abs(sc.v)) < 1e-12
    assert np.allclose((sc.p[1:] - sc.p[:-1]) / gc.dx, F, atol=1e-10)
    sv = MACSolver(gc, rho, 0.01, 1e-3, bcs)
    sv.step(np.full(gc.shape_u(), F), np.zeros(gc.shape_v()), 0.0)
    assert np.max(np.abs(sv.u)) < 10 * (sv.dt * F / rho) * (sv.nu * sv.dt / gc.dx**2)
    assert np.allclose((sv.p[1:] - sv.p[:-1]) / gc.dx, F, rtol=1e-2)


@pytest.mark.parametrize("scheme", ["midpoint", "step"])
@pytest.mark.parametrize("projection", ["nonincremental", "incremental"])
def test_pressure_is_sharp_next_to_a_membrane_force(scheme, projection):
    """A pressurised membrane spreads a force whose gradient part jumps by J across a thin
    band. Here the force is exactly that gradient part, F = G_h Q with Q = J inside a circle
    of radius 0.25 and 0 outside, smoothed over 4 cells, so the exact discrete answer is
    u = 0 and p = Q + const. With implicit viscosity the projection potential is a smoothed
    pressure, about (I - c L_h)^{-1} Q with c = theta nu h; the reported rotational pressure
    rho (phi - c L_h phi) undoes that. mu = 1, dt = 1e-3, 32^2 closed box, 5 steps. Measured
    jump error between r < R - 3h and R + 3h < r < 0.45: below 1e-6 relative (it was -8e-3
    midpoint and -2.6e-2 step with p = rho phi); max|p - Q| 1.2e-5 (midpoint), 2e-4 (step),
    in the cells next to the walls, where the viscous solve does not commute with the
    gradient (it was 0.36 and 0.57). The incremental projection gives 3.4e-7 and 1.3e-5."""
    N, J, R = 32, 4.0, 0.25
    g, bcs = closed_box(N)
    s = MACSolver(g, 1.0, 1.0, 1e-3, bcs, projection=projection)
    Xc, Yc = g.meshgrid("centres")
    r = np.hypot(Xc - 0.5, Yc - 0.5)
    h = g.dx
    z = np.clip((r - R) / (2 * h), -1.0, 1.0)
    Q = J * 0.5 * (1.0 - np.sin(0.5 * np.pi * z))
    Gx, Gy = s._gradient(Q)
    Fu, Fv = s._store_u(Gx), s._store_v(Gy)
    for k in range(5):
        t = k * s.dt
        if scheme == "step":
            s.step(Fu, Fv, t)
        else:
            uh, vh = s.half_step(Fu, Fv, t)
            s.full_step(Fu, Fv, uh, vh, t)
    inside, outside = r < R - 3 * h, (r > R + 3 * h) & (r < 0.45)
    jump = s.p[inside].mean() - s.p[outside].mean()
    assert abs(jump / J - 1) < 1e-5
    assert s.p[inside].std() < 1e-5 * J and s.p[outside].std() < 1e-5 * J
    assert np.max(np.abs((s.p - s.p.mean()) - (Q - Q.mean()))) < 1e-3 * J
    assert max(np.max(np.abs(s.u)), np.max(np.abs(s.v))) < 1e-4
    if projection == "nonincremental":
        # the reported pressure does not feed back into the velocity
        s2 = MACSolver(g, 1.0, 1.0, 1e-3, bcs)
        s2.u[:], s2.v[:], s2.p[:] = s.u, s.v, 123.0
        s.step(Fu, Fv, 5 * s.dt)
        s2.step(Fu, Fv, 5 * s.dt)
        assert np.array_equal(s.u, s2.u) and np.array_equal(s.v, s2.v)


def test_half_step_leaves_state_alone():
    g, bcs = closed_box(16, bc_top=lambda x, t: (np.ones_like(x), np.zeros_like(x)))
    s = MACSolver(g, 1.0, 0.01, 1e-3, bcs)
    s.step(None, None, 0.0)
    u0, v0, p0, t0 = s.u.copy(), s.v.copy(), s.p.copy(), s.t
    uh, vh = s.half_step(None, None, s.t)
    assert np.array_equal(s.u, u0) and np.array_equal(s.v, v0) and np.array_equal(s.p, p0)
    assert s.t == t0 and uh.shape == u0.shape and vh.shape == v0.shape
    assert not np.array_equal(uh, u0)


def test_explicit_viscosity_scheme_and_warning():
    g, bcs = closed_box(16, bc_top=lambda x, t: (np.ones_like(x), np.zeros_like(x)))
    with pytest.warns(RuntimeWarning, match="stability limit"):
        MACSolver(g, 1.0, 0.1, 1e-2, bcs, viscous="explicit")      # nu dt / h^2 = 0.256 > 1/4
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s = MACSolver(g, 1.0, 0.1, 5e-4, bcs, viscous="explicit", advection="upwind")
    for k in range(10):
        s.step(None, None, k * s.dt)
    assert np.max(np.abs(s.divergence())) < 1e-12
    run_midpoint(s, 10)
    assert np.max(np.abs(s.divergence())) < 1e-12
    assert np.isfinite(s.u).all() and np.max(np.abs(s.u)) <= 1.0 + 1e-9
    # explicit and implicit agree to O(dt) on a short run
    si = MACSolver(g, 1.0, 0.1, 5e-4, bcs, viscous="implicit", advection="upwind")
    for k in range(10):
        si.step(None, None, k * si.dt)
    run_midpoint(si, 10)
    assert np.max(np.abs(si.u - s.u)) < 2e-2


def test_diagnostics():
    g = Grid(2.0, 1.0, 16, 8, periodic_x=False, periodic_y=False)
    s = MACSolver(g, 3.0, 0.1, 0.01, BoundaryConditions(D, D, D, D))
    s.u[:] = 1.0
    s.v[:] = -0.5
    assert s.cfl() == pytest.approx(0.01 * (1.0 / g.dx + 0.5 / g.dy))
    assert s.kinetic_energy() == pytest.approx(0.5 * 3.0 * (1.0 + 0.25) * g.Lx * g.Ly)
    assert s.mean_velocity() == pytest.approx((1.0, -0.5))
    # vorticity of a rigid rotation about the box centre is 2 Omega at every centre
    Om = 0.7
    Xu, Yu = g.meshgrid("u")
    Xv, Yv = g.meshgrid("v")
    s.u[:] = -Om * (Yu - 0.5)
    s.v[:] = Om * (Xv - 1.0)
    rot = BoundaryConditions(D, D, D, D,
                             bc_left=lambda y, t: (-Om * (y - 0.5), -Om * np.ones_like(y)),
                             bc_right=lambda y, t: (-Om * (y - 0.5), Om * np.ones_like(y)),
                             bc_bottom=lambda x, t: (0.5 * Om * np.ones_like(x), Om * (x - 1.0)),
                             bc_top=lambda x, t: (-0.5 * Om * np.ones_like(x), Om * (x - 1.0)))
    s.bcs = rot
    assert np.allclose(s.vorticity(), 2 * Om)
    assert s.vorticity().shape == g.shape_p()
