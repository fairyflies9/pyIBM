"""Tests for the doubly periodic FFT back end (docs/numerics.md, fluid solvers).

The difference operators and the projection used here are written independently of the
module (np.roll and plain sines), and the pm1996 reference is a loop-for-loop transcription
of the 2008 MATLAB scheme (makew.m, fluid_solve.m, moving_plate.m), so a match is
evidence and not a tautology.
"""

import numpy as np
import pytest

from pyibm.coupling.delta import DeltaCoupling
from pyibm.fluid import FluidSolver
from pyibm.fluid.periodic_fft import PeriodicFFTSolver, periodic_symbols
from pyibm.grid import Grid, layout_collocated


# --------------------------------------------------------------------------- independent helpers
def _d0x(phi, h):
    return (np.roll(phi, -1, 0) - np.roll(phi, 1, 0)) / (2 * h)


def _d0y(phi, h):
    return (np.roll(phi, -1, 1) - np.roll(phi, 1, 1)) / (2 * h)


def _lap(phi, h):
    return (np.roll(phi, -1, 0) + np.roll(phi, 1, 0) + np.roll(phi, -1, 1) + np.roll(phi, 1, 1)
            - 4 * phi) / h**2


def _div(u, v, h):
    return _d0x(u, h) + _d0y(v, h)


def _skew(u, v, h):
    """Peskin 2002 eq. 4.10 applied to (u, v) with advecting field (u, v)."""
    Su = 0.5 * (u * _d0x(u, h) + v * _d0y(u, h)) + 0.5 * (_d0x(u * u, h) + _d0y(v * u, h))
    Sv = 0.5 * (u * _d0x(v, h) + v * _d0y(v, h)) + 0.5 * (_d0x(u * v, h) + _d0y(v * v, h))
    return Su, Sv


def _sines(Nx, Ny):
    k1, k2 = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing="ij")
    s1, s2 = np.sin(2 * np.pi * k1 / Nx), np.sin(2 * np.pi * k2 / Ny)
    Z = np.isin(k1, [0, Nx // 2]) & np.isin(k2, [0, Ny // 2])
    return s1, s2, Z


def _project_fields(u, v):
    """Independent projection in physical space: subtract the D0-gradient part."""
    Nx, Ny = u.shape
    s1, s2, Z = _sines(Nx, Ny)
    uh, vh = np.fft.fft2(u), np.fft.fft2(v)
    S = np.where(Z, 1.0, s1**2 + s2**2)
    q = np.where(Z, 0.0, (s1 * uh + s2 * vh) / S)
    return np.real(np.fft.ifft2(uh - s1 * q)), np.real(np.fft.ifft2(vh - s2 * q))


def _is_discrete_gradient(rx, ry):
    """Largest Fourier component of (rx, ry) orthogonal to s, relative to |r|; zero if r = D0 p."""
    Nx, Ny = rx.shape
    s1, s2, Z = _sines(Nx, Ny)
    rxh, ryh = np.fft.fft2(rx), np.fft.fft2(ry)
    S = np.where(Z, 1.0, s1**2 + s2**2)
    q = np.where(Z, 0.0, (s1 * rxh + s2 * ryh) / S)
    rest = np.hypot(np.abs(rxh - s1 * q), np.abs(ryh - s2 * q))
    scale = max(np.max(np.hypot(np.abs(rxh), np.abs(ryh))), 1e-300)
    return np.max(rest) / scale


def _matlab_pm1996_step(u, v, f1, f2, dx, dt, rho, mu):
    """Loop transcription of makew.m + moving_plate.m (a, b, c, d) + fluid_solve.m, 0-based."""
    M, N = u.shape
    nu = mu / rho
    dt_p = dt / rho
    w1 = np.empty_like(u)
    w2 = np.empty_like(v)
    for i in range(M):
        for j in range(N):
            ip1, i_1 = (i + 1) % M, (i - 1) % M
            jp1, j_1 = (j + 1) % N, (j - 1) % N
            if u[i, j] < 0:
                du1 = (1 / dx) * (u[ip1, j] - u[i, j])
                dv1 = (1 / dx) * (v[ip1, j] - v[i, j])
            else:
                du1 = (1 / dx) * (u[i, j] - u[i_1, j])
                dv1 = (1 / dx) * (v[i, j] - v[i_1, j])
            if v[i, j] < 0:
                du2 = (1 / dx) * (u[i, jp1] - u[i, j])
                dv2 = (1 / dx) * (v[i, jp1] - v[i, j])
            else:
                du2 = (1 / dx) * (u[i, j] - u[i, j_1])
                dv2 = (1 / dx) * (v[i, j] - v[i, j_1])
            w1[i, j] = u[i, j] - dt * (u[i, j] * du1 + v[i, j] * du2) + dt_p * f1[i, j]
            w2[i, j] = v[i, j] - dt * (u[i, j] * dv1 + v[i, j] * dv2) + dt_p * f2[i, j]
    w1ft, w2ft = np.fft.fft2(w1), np.fft.fft2(w2)
    uft = np.empty_like(w1ft)
    vft = np.empty_like(w2ft)
    pft = np.empty_like(w1ft)
    for i in range(M):
        for j in range(N):
            s1, s2 = np.sin(2 * np.pi * i / M), np.sin(2 * np.pi * j / N)
            if (i == 0 or i == M // 2) and (j == 0 or j == N // 2):
                a = b = c = 0.0
                pft[i, j] = 0.0
            else:
                S = s1 * s1 + s2 * s2
                a, b, c = s1 * s1 / S, s1 * s2 / S, s2 * s2 / S
                pft[i, j] = ((rho * dx / dt) * (s1 * w1ft[i, j].imag + s2 * w2ft[i, j].imag) / S
                             - 1j * (rho * dx / dt) * (s1 * w1ft[i, j].real + s2 * w2ft[i, j].real) / S)
            d = 1 / (1 + (4 * nu * dt / (dx * dx)) * (np.sin(np.pi * i / M) ** 2 + np.sin(np.pi * j / N) ** 2))
            uft[i, j] = (w1ft[i, j] - (a * w1ft[i, j] + b * w2ft[i, j])) * d
            vft[i, j] = (w2ft[i, j] - (b * w1ft[i, j] + c * w2ft[i, j])) * d
    return (np.real(np.fft.ifft2(uft)), np.real(np.fft.ifft2(vft)), np.real(np.fft.ifft2(pft)))


def _taylor_green(grid, nu, t, U0=1.0):
    X, Y = grid.meshgrid("nodes")
    e = np.exp(-8 * np.pi**2 * nu * t)
    return (-U0 * np.cos(2 * np.pi * X) * np.sin(2 * np.pi * Y) * e,
            U0 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y) * e)


def _lambda_h(grid):
    """Eigenvalue of -L_h on the (1, 1) Taylor-Green mode of the unit box."""
    return (4 / grid.dx**2) * (np.sin(np.pi / grid.Nx) ** 2 + np.sin(np.pi / grid.Ny) ** 2)


def _random_state(solver, rng, amplitude=1.0):
    """Random discretely divergence-free velocity and random force on the solver's grid."""
    shape = solver.grid.shape_p()
    u, v = _project_fields(rng.standard_normal(shape), rng.standard_normal(shape))
    solver.u, solver.v = amplitude * u, amplitude * v
    return rng.standard_normal(shape), rng.standard_normal(shape)


# --------------------------------------------------------------------------- construction
def test_constructor_and_protocol():
    g = Grid.periodic_box(1.0, 2.0, 16, 32)
    s = PeriodicFFTSolver(g, rho=2.0, mu=0.1, dt=1e-3)
    assert isinstance(s, FluidSolver)
    assert s.layout_u == layout_collocated(g) and s.layout_v == layout_collocated(g)
    assert s.u.shape == s.v.shape == s.p.shape == (16, 32)
    assert s.u.dtype == np.float64 and not s.u.any()
    with pytest.raises(ValueError):
        PeriodicFFTSolver(Grid(1.0, 1.0, 16, 16, periodic_x=False), 1.0, 0.1, 1e-3)
    with pytest.raises(ValueError):
        PeriodicFFTSolver(Grid(1.0, 1.0, 16, 32), 1.0, 0.1, 1e-3)          # dx != dy
    with pytest.raises(ValueError):
        PeriodicFFTSolver(g, 1.0, 0.1, 0.0)
    with pytest.raises(ValueError):
        s.step(np.zeros((16, 16)), np.zeros((16, 32)))
    with pytest.raises(ValueError):
        s.full_step(np.zeros((16, 32)), np.zeros((16, 32)), np.zeros((32, 16)), np.zeros((16, 32)))


def test_symbols():
    Nx, Ny, h = 16, 32, 0.25
    sym = periodic_symbols(Nx, Ny, h)
    assert sym.degenerate.sum() == 4
    for i, j in [(0, 0), (0, 16), (8, 0), (8, 16)]:
        assert sym.degenerate[i, j] and sym.inv_s_sq[i, j] == 0.0
        assert abs(sym.s1[i, j]) < 1e-15 and abs(sym.s2[i, j]) < 1e-15
    assert np.all(np.isfinite(sym.inv_s_sq))
    assert np.all(sym.inv_s_sq[~sym.degenerate] > 0)
    assert sym.lap[0, 0] == 0.0 and np.all(sym.lap <= 0)
    assert sym.lap[Nx // 2, Ny // 2] == pytest.approx(-8 / h**2)
    assert sym.s1[1, 0] == pytest.approx(np.sin(2 * np.pi / Nx))
    assert not sym.s1.flags.writeable
    with pytest.raises(ValueError):
        periodic_symbols(15, 16)


# --------------------------------------------------------------------------- projection
def test_project_divergence_free_idempotent_and_mean():
    rng = np.random.default_rng(1)
    g = Grid.periodic_box(1.0, 1.0, 32, 32)
    wu, wv = rng.standard_normal(g.shape_p()) + 0.7, rng.standard_normal(g.shape_p()) - 0.3
    pu_hat, pv_hat = PeriodicFFTSolver.project(np.fft.fft2(wu), np.fft.fft2(wv))
    pu, pv = np.real(np.fft.ifft2(pu_hat)), np.real(np.fft.ifft2(pv_hat))
    assert np.max(np.abs(_div(pu, pv, g.dx))) < 1e-13
    assert pu.mean() == pytest.approx(wu.mean(), abs=1e-14)       # k = 0 passes through
    assert pv.mean() == pytest.approx(wv.mean(), abs=1e-14)
    qu_hat, qv_hat = PeriodicFFTSolver.project(pu_hat, pv_hat)
    assert np.max(np.abs(qu_hat - pu_hat)) < 1e-12 * np.max(np.abs(pu_hat))
    assert np.max(np.abs(qv_hat - pv_hat)) < 1e-12 * np.max(np.abs(pv_hat))
    # and the same with the solver's precomputed symbols
    s = PeriodicFFTSolver(g, 1.0, 0.01, 1e-3)
    ru_hat, rv_hat = PeriodicFFTSolver.project(np.fft.fft2(wu), np.fft.fft2(wv), s.symbols)
    assert np.array_equal(ru_hat, pu_hat) and np.array_equal(rv_hat, pv_hat)
    # the random field was not divergence-free to begin with (the test is not vacuous)
    assert np.max(np.abs(_div(wu, wv, g.dx))) > 1.0


def test_project_passes_divergence_free_force_unchanged():
    rng = np.random.default_rng(2)
    g = Grid.periodic_box(1.0, 1.0, 32, 32)
    psi = rng.standard_normal(g.shape_p())
    Fx, Fy = _d0y(psi, g.dx), -_d0x(psi, g.dx)        # D0 . F = 0 exactly (D0_x, D0_y commute)
    assert np.max(np.abs(_div(Fx, Fy, g.dx))) < 1e-12 * np.max(np.abs(Fx)) / g.dx
    Fx_hat, Fy_hat = np.fft.fft2(Fx), np.fft.fft2(Fy)
    pu_hat, pv_hat = PeriodicFFTSolver.project(Fx_hat, Fy_hat)
    scale = max(np.max(np.abs(Fx_hat)), np.max(np.abs(Fy_hat)))
    assert np.max(np.abs(pu_hat - Fx_hat)) < 1e-13 * scale
    assert np.max(np.abs(pv_hat - Fy_hat)) < 1e-13 * scale
    # with the box mean added (a blob-projection force carries one) it still passes through
    pu_hat, pv_hat = PeriodicFFTSolver.project(np.fft.fft2(Fx + 2.0), np.fft.fft2(Fy - 1.0))
    assert np.allclose(np.real(np.fft.ifft2(pu_hat)), Fx + 2.0, atol=1e-13 * scale)
    assert np.allclose(np.real(np.fft.ifft2(pv_hat)), Fy - 1.0, atol=1e-13 * scale)


# --------------------------------------------------------------------------- pm1996 scheme
def test_pm1996_matches_matlab_transcription():
    rng = np.random.default_rng(3)
    g = Grid.periodic_box(0.4, 0.4, 16, 16)
    rho, mu = 2.0, 3.0
    dt = g.dx**2 / (mu / rho)                            # in_fluid.m: dt = dx^2/nu
    s = PeriodicFFTSolver(g, rho, mu, dt)
    u0, v0 = rng.standard_normal(g.shape_p()), rng.standard_normal(g.shape_p())
    f1, f2 = 50 * rng.standard_normal(g.shape_p()), 50 * rng.standard_normal(g.shape_p())
    s.u, s.v = u0.copy(), v0.copy()
    s.step(f1, f2, 0.0)
    u_ref, v_ref, p_ref = _matlab_pm1996_step(u0, v0, f1, f2, g.dx, dt, rho, mu)
    assert np.max(np.abs(s.u - u_ref)) < 1e-13 * np.max(np.abs(u_ref))
    assert np.max(np.abs(s.v - v_ref)) < 1e-13 * np.max(np.abs(v_ref))
    assert np.max(np.abs(s.p - p_ref)) < 1e-13 * np.max(np.abs(p_ref))
    assert np.max(np.abs(p_ref)) > 0                      # the pressure check is not vacuous
    # the upwind switch at u == 0 takes the backward difference, as in makew.m (u < 0 test)
    u1 = u0.copy()
    u1[3, 5] = 0.0
    s.u, s.v = u1.copy(), v0.copy()
    s.step(f1, f2, 0.0)
    u_ref, v_ref, _ = _matlab_pm1996_step(u1, v0, f1, f2, g.dx, dt, rho, mu)
    assert np.max(np.abs(s.u - u_ref)) < 1e-13 * np.max(np.abs(u_ref))
    assert np.max(np.abs(s.v - v_ref)) < 1e-13 * np.max(np.abs(v_ref))


def test_pm1996_momentum_residual_random_data():
    """Eqs. 14.40-14.41: (I - (mu dt/rho) L_h) u^{n+1} + (dt/rho) D0 p^{n+1} = w, D0 . u^{n+1} = 0."""
    rng = np.random.default_rng(4)
    g = Grid.periodic_box(1.0, 1.0, 32, 32)
    rho, mu, dt = 1.3, 0.05, 2e-3
    s = PeriodicFFTSolver(g, rho, mu, dt)
    Fu, Fv = _random_state(s, rng)
    u0, v0 = s.u.copy(), s.v.copy()
    Au, Av = s.upwind_advection(u0, v0)
    wu = u0 - dt * Au + dt / rho * Fu
    wv = v0 - dt * Av + dt / rho * Fv
    s.step(Fu, Fv, 0.0)
    ru = s.u - (mu * dt / rho) * _lap(s.u, g.dx) + (dt / rho) * _d0x(s.p, g.dx) - wu
    rv = s.v - (mu * dt / rho) * _lap(s.v, g.dx) + (dt / rho) * _d0y(s.p, g.dx) - wv
    scale = max(np.max(np.abs(wu)), np.max(np.abs(wv)))
    assert np.max(np.abs(ru)) < 1e-13 * scale
    assert np.max(np.abs(rv)) < 1e-13 * scale
    assert np.max(np.abs(_div(s.u, s.v, g.dx))) < 1e-13
    # p is a genuine part of the balance: the residual without it is O(1)
    assert np.max(np.abs(ru + (dt / rho) * _d0x(s.p, g.dx))) > 1e-3 * scale
    # the pressure has no energy in the four degenerate modes (eq. 14.53: p_hat = 0 there)
    p_hat = np.fft.fft2(s.p)
    _, _, Z = _sines(*g.shape_p())
    assert np.max(np.abs(p_hat[Z])) < 1e-12 * np.max(np.abs(p_hat))


def test_upwind_advection_matches_loop_rule():
    rng = np.random.default_rng(5)
    g = Grid.periodic_box(1.0, 1.0, 16, 16)
    s = PeriodicFFTSolver(g, 1.0, 0.1, 1e-3)
    u, v = rng.standard_normal(g.shape_p()), rng.standard_normal(g.shape_p())
    u[2, 2] = 0.0
    Au, Av = s.upwind_advection(u, v)
    N, h = 16, g.dx
    for i in range(N):
        for j in range(N):
            ip, im, jp, jm = (i + 1) % N, (i - 1) % N, (j + 1) % N, (j - 1) % N
            dux = (u[ip, j] - u[i, j]) / h if u[i, j] < 0 else (u[i, j] - u[im, j]) / h
            dvx = (v[ip, j] - v[i, j]) / h if u[i, j] < 0 else (v[i, j] - v[im, j]) / h
            duy = (u[i, jp] - u[i, j]) / h if v[i, j] < 0 else (u[i, j] - u[i, jm]) / h
            dvy = (v[i, jp] - v[i, j]) / h if v[i, j] < 0 else (v[i, j] - v[i, jm]) / h
            assert Au[i, j] == pytest.approx(u[i, j] * dux + v[i, j] * duy, rel=1e-13, abs=1e-13)
            assert Av[i, j] == pytest.approx(u[i, j] * dvx + v[i, j] * dvy, rel=1e-13, abs=1e-13)


def test_pm1996_taylor_green_decay_and_first_order_in_time():
    g = Grid.periodic_box(1.0, 1.0, 32, 32)
    rho, mu, U0 = 1.0, 0.02, 0.05
    nu = mu / rho
    zero = np.zeros(g.shape_p())

    def run(dt, T):
        s = PeriodicFFTSolver(g, rho, mu, dt)
        s.u, s.v = _taylor_green(g, nu, 0.0, U0)
        n = int(round(T / dt))
        for _ in range(n):
            s.step(zero, zero, 0.0)
            assert np.max(np.abs(_div(s.u, s.v, g.dx))) < 1e-13
        return s

    # decay rate of the kinetic energy against 2 * 8 pi^2 nu; first-order upwinding adds
    # numerical viscosity (+1.2 %), while backward Euler and the discrete Laplacian make the
    # decay slower (-0.7 %); net 0.5 % high on this grid, so we ask for 5 %
    T = 0.5
    s = run(0.005, T)
    E0 = 0.5 * rho * np.sum(_taylor_green(g, nu, 0.0, U0)[0] ** 2 * 2) * g.dx**2
    rate = -np.log(s.kinetic_energy() / E0) / T
    assert rate == pytest.approx(16 * np.pi**2 * nu, rel=0.05)
    # measured temporal order by Richardson extrapolation (the spatial error is the same for
    # all three runs, so only the time discretisation error is compared)
    T = 0.2
    u1 = run(0.02, T).u
    u2 = run(0.01, T).u
    u4 = run(0.005, T).u
    order = np.log2(np.max(np.abs(u1 - u2)) / np.max(np.abs(u2 - u4)))
    assert 0.8 < order < 1.25


# --------------------------------------------------------------------------- midpoint scheme
def test_midpoint_residuals_random_data():
    """Eqs. 7.8-7.9 for the half step and 7.11-7.12 for the full step, at 1e-13."""
    rng = np.random.default_rng(6)
    g = Grid.periodic_box(1.0, 1.0, 32, 32)
    rho, mu, dt, h = 0.8, 0.03, 1.5e-3, g.dx
    s = PeriodicFFTSolver(g, rho, mu, dt)
    Fu, Fv = _random_state(s, rng)
    u0, v0 = s.u.copy(), s.v.copy()
    u_h, v_h = s.half_step(Fu, Fv, 0.0)
    assert np.array_equal(s.u, u0) and np.array_equal(s.v, v0)      # state untouched
    assert np.max(np.abs(_div(u_h, v_h, h))) < 1e-13
    Su, Sv = _skew(u0, v0, h)
    rx = rho * ((u_h - u0) / (dt / 2) + Su) - mu * _lap(u_h, h) - Fu
    ry = rho * ((v_h - v0) / (dt / 2) + Sv) - mu * _lap(v_h, h) - Fv
    # the residual must be a discrete pressure gradient D0 p~ and nothing else
    assert _is_discrete_gradient(rx, ry) < 1e-13
    assert np.max(np.hypot(rx, ry)) > 1.0                             # a real pressure is there
    s.full_step(Fu, Fv, u_h, v_h, 0.0)
    Su, Sv = _skew(u_h, v_h, h)
    rx = rho * ((s.u - u0) / dt + Su) + _d0x(s.p, h) - mu * _lap(0.5 * (u0 + s.u), h) - Fu
    ry = rho * ((s.v - v0) / dt + Sv) + _d0y(s.p, h) - mu * _lap(0.5 * (v0 + s.v), h) - Fv
    scale = max(np.max(np.abs(Fu)), rho * np.max(np.abs(s.u - u0)) / dt)
    assert np.max(np.abs(rx)) < 1e-13 * scale
    assert np.max(np.abs(ry)) < 1e-13 * scale
    assert np.max(np.abs(_div(s.u, s.v, h))) < 1e-13


def test_midpoint_taylor_green_crank_nicolson_factor():
    """Taylor-Green is an exact discrete eigenmode, so this checks the viscous CN factor only.

    The skew-symmetric advection of the mode is a discrete gradient and is projected away, so
    each step multiplies the field by the Crank-Nicolson factor (1 - z/2)/(1 + z/2) of the
    discrete Laplacian. The half step, the nonlinear predictor and the use of u_h in the full
    step play no part here; test_midpoint_second_order_in_time_multimode checks those.
    """
    g = Grid.periodic_box(1.0, 1.0, 32, 32)
    rho, mu, U0 = 1.0, 0.1, 1.0
    nu = mu / rho
    lam = nu * _lambda_h(g)
    zero = np.zeros(g.shape_p())

    def run(dt, T):
        s = PeriodicFFTSolver(g, rho, mu, dt)
        s.u, s.v = _taylor_green(g, nu, 0.0, U0)
        n = int(round(T / dt))
        for _ in range(n):
            u_h, v_h = s.half_step(zero, zero, 0.0)
            s.full_step(zero, zero, u_h, v_h, 0.0)
            assert np.max(np.abs(_div(s.u, s.v, g.dx))) < 1e-13
        return s, n

    T, dt = 0.2, 0.02
    s, n = run(dt, T)
    z = lam * dt
    u0, v0 = _taylor_green(g, nu, 0.0, U0)
    factor = ((1 - z / 2) / (1 + z / 2)) ** n
    assert np.max(np.abs(s.u - factor * u0)) < 1e-12
    assert np.max(np.abs(s.v - factor * v0)) < 1e-12
    # it decays at the analytic rate up to O(h^2) + O(dt^2)
    ua, va = _taylor_green(g, nu, T, U0)
    assert np.max(np.abs(s.u - ua)) < 1e-2 * U0
    assert factor == pytest.approx(np.exp(-8 * np.pi**2 * nu * T), rel=5e-3)
    # The error against the exact semi-discrete solution exp(-lam T) u0 is the error of the
    # scalar Crank-Nicolson recursion, so this order is that of the viscous CN factor.
    e = []
    for dt in (0.02, 0.01):
        s, _ = run(dt, T)
        e.append(np.max(np.abs(s.u - np.exp(-lam * T) * u0)))
    order = np.log2(e[0] / e[1])
    assert e[1] > 1e-7                                     # well above round-off
    assert 1.9 < order < 2.1


def _multimode(g, amplitude=0.15):
    """A flow that is not an eigenmode: u = amplitude (D0_y psi, -D0_x psi), D0 . u = 0 exactly.

    psi = sin 2 pi x sin 2 pi y + 0.6 cos(2 pi x + 0.3) sin 4 pi y + 0.4 sin(6 pi x + 1.0).
    """
    X, Y = g.meshgrid("nodes")
    psi = (np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)
           + 0.6 * np.cos(2 * np.pi * X + 0.3) * np.sin(4 * np.pi * Y)
           + 0.4 * np.sin(6 * np.pi * X + 1.0))
    return amplitude * _d0y(psi, g.dx), -amplitude * _d0x(psi, g.dx)


def test_midpoint_second_order_in_time_multimode():
    """Richardson temporal order of the whole midpoint scheme on a flow that advects itself.

    The spatial error is the same in the three runs, so differences between successive dt
    measure the time discretisation alone. Measured: 1.98. With the half step replaced by
    u_h = u^n (a broken predictor) the same measurement gives 1.18.
    """
    g = Grid.periodic_box(1.0, 1.0, 32, 32)
    zero = np.zeros(g.shape_p())

    def run(dt, T=0.4):
        s = PeriodicFFTSolver(g, 1.0, 0.01, dt)
        s.u, s.v = _multimode(g)
        for _ in range(int(round(T / dt))):
            u_h, v_h = s.half_step(zero, zero, 0.0)
            s.full_step(zero, zero, u_h, v_h, 0.0)
        return s

    r = [run(dt) for dt in (0.02, 0.01, 0.005)]
    e1 = max(np.max(np.abs(r[0].u - r[1].u)), np.max(np.abs(r[0].v - r[1].v)))
    e2 = max(np.max(np.abs(r[1].u - r[2].u)), np.max(np.abs(r[1].v - r[2].v)))
    assert e2 > 1e-6                                       # well above round-off
    assert 1.9 < np.log2(e1 / e2) < 2.1


# --------------------------------------------------------------------------- mean velocity
@pytest.mark.parametrize("scheme", ["pm1996", "midpoint"])
@pytest.mark.parametrize("remove", [False, True])
def test_constant_force_mean_velocity(scheme, remove):
    g = Grid.periodic_box(1.0, 1.0, 32, 32)
    rho, dt, F = 2.0, 1e-3, 3.0
    s = PeriodicFFTSolver(g, rho, 0.01, dt, remove_mean_force=remove)
    Fu = np.full(g.shape_p(), F)
    Fv = np.zeros(g.shape_p())
    n = 5
    for _ in range(n):
        if scheme == "pm1996":
            s.step(Fu, Fv, 0.0)
        else:
            u_h, v_h = s.half_step(Fu, Fv, 0.0)
            s.full_step(Fu, Fv, u_h, v_h, 0.0)
    expected = 0.0 if remove else n * dt * F / rho
    mu_, mv_ = s.mean_velocity()
    assert mu_ == pytest.approx(expected, abs=1e-14)
    assert mv_ == pytest.approx(0.0, abs=1e-14)
    assert np.max(np.abs(s.u - expected)) < 1e-14          # the whole field is uniform
    assert np.max(np.abs(s.p)) < 1e-12


def test_midpoint_remove_mean_force_keeps_mean_velocity():
    rng = np.random.default_rng(7)
    g = Grid.periodic_box(1.0, 1.0, 32, 32)
    s = PeriodicFFTSolver(g, 1.0, 0.01, 1e-3, remove_mean_force=True)
    Fu, Fv = _random_state(s, rng)
    s.u += 0.4                       # the random projected field carries a mean of its own
    s.v -= 0.2
    mu0, mv0 = s.mean_velocity()
    for _ in range(3):
        u_h, v_h = s.half_step(Fu + 5.0, Fv - 3.0, 0.0)
        assert np.mean(u_h) == pytest.approx(mu0, abs=1e-14)
        s.full_step(Fu + 5.0, Fv - 3.0, u_h, v_h, 0.0)
    mu_, mv_ = s.mean_velocity()
    assert mu_ == pytest.approx(mu0, abs=1e-14)
    assert mv_ == pytest.approx(mv0, abs=1e-14)
    # without the option the same net force accelerates the box by dt F/rho per step
    s2 = PeriodicFFTSolver(g, 1.0, 0.01, 1e-3, remove_mean_force=False)
    s2.u, s2.v = s.u.copy(), s.v.copy()
    u_h, v_h = s2.half_step(Fu + 5.0, Fv - 3.0, 0.0)
    s2.full_step(Fu + 5.0, Fv - 3.0, u_h, v_h, 0.0)
    assert s2.mean_velocity()[0] == pytest.approx(mu_ + 1e-3 * (5.0 + Fu.mean()), abs=1e-13)
    assert s2.mean_velocity()[1] == pytest.approx(mv_ + 1e-3 * (-3.0 + Fv.mean()), abs=1e-13)


def _tethered_ring(scheme, remove, steps=50, seed=11):
    """A ring of 40 markers tethered to targets displaced at random, plus a random Lagrangian
    force at every step, spread with peskin4 and moved with the fluid (the driver's loop for
    each scheme). The fluid starts from a random divergence-free field with a box mean.
    Returns the solver, the box-mean velocity after every step (steps + 1, 2), the net force
    on the fluid at every step (steps, 2) and the box mean of the upwind advection of the
    starting field (2,).
    """
    rng = np.random.default_rng(seed)
    g = Grid.periodic_box(1.0, 1.0, 32, 32)
    rho, mu, dt, k = 1.0, 0.02, 2e-3, 50.0
    s = PeriodicFFTSolver(g, rho, mu, dt, remove_mean_force=remove)
    _random_state(s, rng, amplitude=0.2)
    s.u += 0.3
    s.v -= 0.2
    adv_mean = np.array([np.mean(a) for a in s.upwind_advection(s.u, s.v)])
    delta = DeltaCoupling("peskin4")
    lay = s.layout_u
    n = 40
    theta = 2 * np.pi * np.arange(n) / n
    X = np.column_stack([0.5 + 0.2 * np.cos(theta), 0.5 + 0.2 * np.sin(theta)])
    XT = X + 0.02 * rng.standard_normal((n, 2))
    ds = 2 * np.pi * 0.2 / n
    means, net = [s.mean_velocity()], []
    for _ in range(steps):
        f_random = rng.standard_normal((n, 2))
        if scheme == "pm1996":
            Fu, Fv = delta.spread_vector(k * (XT - X) + f_random, X, ds, lay, lay)
            s.step(Fu, Fv, 0.0)
            X = g.wrap(X + dt * delta.interpolate_vector(s.u, s.v, X, lay, lay))
        else:
            X_h = g.wrap(X + 0.5 * dt * delta.interpolate_vector(s.u, s.v, X, lay, lay))
            Fu, Fv = delta.spread_vector(k * (XT - X_h) + f_random, X_h, ds, lay, lay)
            u_h, v_h = s.half_step(Fu, Fv, 0.0)
            s.full_step(Fu, Fv, u_h, v_h, 0.0)
            X = g.wrap(X + dt * delta.interpolate_vector(u_h, v_h, X_h, lay, lay))
        net.append((np.sum(Fu) * g.dx * g.dy, np.sum(Fv) * g.dx * g.dy))
        means.append(s.mean_velocity())
    return s, np.array(means), np.array(net), adv_mean


@pytest.mark.parametrize("scheme", ["pm1996", "midpoint"])
def test_remove_mean_force_holds_mean_velocity_tethered_structure(scheme):
    """remove_mean_force keeps the box-mean velocity at its starting value to round-off.

    In pm1996 the upwind advection has a box mean of its own (upwinding is not
    conservative), so removing the mean of the force alone would let the box mean drift;
    the solver removes the mean of the whole increment of u.
    """
    s, means, net, adv_mean = _tethered_ring(scheme, remove=True)
    assert np.all(np.isfinite(s.u)) and np.all(np.isfinite(means))
    assert np.max(np.abs(means - means[0])) < 1e-14
    # the test is not vacuous: the structure pushes the fluid hard, the starting field moves
    # the markers, and the upwind advection alone would change the box mean
    assert np.max(np.abs(net)) > 0.1
    assert np.max(np.abs(adv_mean)) > 1e-3
    # without the option the same case accelerates the box
    _, means_off, _, _ = _tethered_ring(scheme, remove=False)
    assert np.max(np.abs(means_off - means_off[0])) > 1e-3


@pytest.mark.parametrize("scheme", ["pm1996", "midpoint"])
def test_remove_mean_force_holds_mean_velocity_float32_state(scheme):
    """The k = 0 coefficient is summed in float64, whatever the dtype of the state.

    A float32 sum of N^2 values near 0.35 is off by far more than float64 round-off, and
    that error would move the box mean by (error)/N^2 in one step.
    """
    rng = np.random.default_rng(3)
    g = Grid.periodic_box(1.0, 1.0, 128, 128)
    s = PeriodicFFTSolver(g, 1.0, 0.01, 1e-4, remove_mean_force=True)
    shape = g.shape_p()
    u32 = (0.3 + 0.1 * rng.random(shape)).astype(np.float32)
    v32 = (-0.2 + 0.1 * rng.random(shape)).astype(np.float32)
    s.u, s.v = u32, v32
    mean_u, mean_v = np.mean(u32, dtype=np.float64), np.mean(v32, dtype=np.float64)
    Fu, Fv = 1.0 + rng.standard_normal(shape), -2.0 + rng.standard_normal(shape)
    if scheme == "pm1996":
        s.step(Fu, Fv, 0.0)
    else:
        u_h, v_h = s.half_step(Fu, Fv, 0.0)
        assert abs(np.mean(u_h) - mean_u) < 1e-15 and abs(np.mean(v_h) - mean_v) < 1e-15
        s.full_step(Fu, Fv, u_h, v_h, 0.0)
    assert s.u.dtype == np.float64 and s.v.dtype == np.float64
    assert abs(np.mean(s.u) - mean_u) < 1e-15
    assert abs(np.mean(s.v) - mean_v) < 1e-15


# --------------------------------------------------------------------------- diagnostics
def test_diagnostics():
    g = Grid.periodic_box(1.0, 1.0, 64, 64)
    s = PeriodicFFTSolver(g, rho=2.0, mu=0.01, dt=0.01)
    X, Y = g.meshgrid("nodes")
    s.u = np.sin(2 * np.pi * Y)
    s.v = 0.5 * np.cos(2 * np.pi * X)
    # vorticity = D0_x v - D0_y u: centred differences give the sinc factor sin(2 pi h)/(2 pi h)
    sinc = np.sin(2 * np.pi * g.dx) / (2 * np.pi * g.dx)
    omega = sinc * (-np.pi * np.sin(2 * np.pi * X) - 2 * np.pi * np.cos(2 * np.pi * Y))
    assert np.max(np.abs(s.vorticity() - omega)) < 1e-12
    assert s.vorticity().shape == (64, 64)
    assert s.cfl() == pytest.approx(0.01 * np.max(np.abs(s.u) / g.dx + np.abs(s.v) / g.dy))
    assert s.kinetic_energy() == pytest.approx(0.5 * 2.0 * (0.5 + 0.125), rel=1e-12)
    assert s.mean_velocity() == pytest.approx((0.0, 0.0), abs=1e-15)


def test_rectangular_box_steps_are_divergence_free():
    rng = np.random.default_rng(8)
    g = Grid.periodic_box(1.0, 0.5, 64, 32)
    s = PeriodicFFTSolver(g, 1.0, 0.02, 2e-3)
    Fu, Fv = _random_state(s, rng)
    for _ in range(3):
        s.step(Fu, Fv, 0.0)
        assert np.max(np.abs(_div(s.u, s.v, g.dx))) < 1e-13
        u_h, v_h = s.half_step(Fu, Fv, 0.0)
        assert np.max(np.abs(_div(u_h, v_h, g.dx))) < 1e-13
        s.full_step(Fu, Fv, u_h, v_h, 0.0)
        assert np.max(np.abs(_div(s.u, s.v, g.dx))) < 1e-13
    assert np.isfinite(s.kinetic_energy())


def test_returned_arrays_are_not_scratch():
    """The solver reuses scratch arrays between calls; nothing it hands out may be one."""
    rng = np.random.default_rng(9)
    g = Grid.periodic_box(1.0, 1.0, 16, 16)
    s = PeriodicFFTSolver(g, 1.0, 0.05, 1e-3)
    Fu, Fv = _random_state(s, rng)
    Su, Sv = s.skew_advection(s.u, s.v)
    Au, Av = s.upwind_advection(s.u, s.v)
    u_h, v_h = s.half_step(Fu, Fv, 0.0)
    kept = [a.copy() for a in (Su, Sv, Au, Av, u_h, v_h)]
    s.full_step(Fu, Fv, u_h, v_h, 0.0)
    u1, v1, p1 = s.u.copy(), s.v.copy(), s.p.copy()
    s.step(2 * Fu, -Fv, 0.0)
    s.half_step(-Fu, 3 * Fv, 0.0)
    s.skew_advection(s.v, s.u)
    s.upwind_advection(s.v, s.u)
    for a, b in zip((Su, Sv, Au, Av, u_h, v_h), kept):
        assert np.array_equal(a, b)
    # and a repeated step from the same state gives the same answer
    s.u, s.v = u1.copy(), v1.copy()
    s.step(Fu, Fv, 0.0)
    first = (s.u.copy(), s.v.copy(), s.p.copy())
    s.u, s.v = u1.copy(), v1.copy()
    s.step(Fu, Fv, 0.0)
    assert all(np.array_equal(a, b) for a, b in zip((s.u, s.v, s.p), first))
    assert not np.array_equal(p1, s.p)
