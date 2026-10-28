"""Tests for pyibm.fluid.poisson: sparse LU and DCT solvers for -L_h phi = rhs."""

import numpy as np
import pytest

from pyibm.grid import Grid
from pyibm.fluid.poisson import (DCTPoissonSolver, PoissonSolver, k1, laplacian_matrix,
                                 make_poisson_solver)

ALL_NEUMANN = {"left": "neumann", "right": "neumann", "bottom": "neumann", "top": "neumann"}


def neg_laplacian(phi, dx, dy, sides):
    """Reference -L_h with the ghost rules written out: neumann -> ghost = interior,
    dirichlet (on the wall between points) -> ghost = -interior, periodic -> wrap."""
    px = np.empty((phi.shape[0] + 2, phi.shape[1]))
    px[1:-1] = phi
    px[0] = {"neumann": phi[0], "dirichlet": -phi[0], "periodic": phi[-1]}[sides["left"]]
    px[-1] = {"neumann": phi[-1], "dirichlet": -phi[-1], "periodic": phi[0]}[sides["right"]]
    py = np.empty((phi.shape[0], phi.shape[1] + 2))
    py[:, 1:-1] = phi
    py[:, 0] = {"neumann": phi[:, 0], "dirichlet": -phi[:, 0], "periodic": phi[:, -1]}[sides["bottom"]]
    py[:, -1] = {"neumann": phi[:, -1], "dirichlet": -phi[:, -1], "periodic": phi[:, 0]}[sides["top"]]
    return -((px[:-2] - 2 * phi + px[2:]) / dx**2 + (py[:, :-2] - 2 * phi + py[:, 2:]) / dy**2)


def test_k1_entries():
    h = 0.5
    K = k1(5, h, "neumann", "dirichlet_mid").toarray() * h**2
    assert K[0, 0] == 1 and K[-1, -1] == 3 and K[2, 2] == 2
    assert K[0, 1] == -1 and K[1, 0] == -1 and K[0, 2] == 0
    K = k1(5, h, "dirichlet_point", "neumann").toarray() * h**2
    assert K[0, 0] == 2 and K[-1, -1] == 1
    K = k1(6, h, "periodic", "periodic").toarray() * h**2
    assert np.all(np.diag(K) == 2) and K[0, -1] == -1 and K[-1, 0] == -1
    assert np.allclose(K, K.T)
    with pytest.raises(ValueError):
        k1(5, h, "periodic", "neumann")
    with pytest.raises(ValueError):
        k1(5, h, "foo", "neumann")


def test_laplacian_matrix_matches_stencil():
    g = Grid(1.0, 0.6, 16, 12, periodic_x=False, periodic_y=False)
    sides = {"left": "neumann", "right": "dirichlet", "bottom": "neumann", "top": "neumann"}
    A = laplacian_matrix(g.Nx, g.Ny, g.dx, g.dy, "neumann", "dirichlet_mid", "neumann", "neumann")
    rng = np.random.default_rng(1)
    phi = rng.standard_normal(g.shape_p())
    assert np.allclose((A @ phi.ravel()).reshape(g.shape_p()), neg_laplacian(phi, g.dx, g.dy, sides),
                       rtol=0, atol=1e-12 / g.dx**2)


@pytest.mark.parametrize("reg", ["bordered", "pin"])
def test_sparse_and_dct_agree(reg):
    g = Grid(1.0, 0.6, 32, 24, periodic_x=False, periodic_y=False)   # dx != dy on purpose
    rng = np.random.default_rng(0)
    rhs = rng.standard_normal(g.shape_p())
    ps, pd = PoissonSolver(g, ALL_NEUMANN, regularisation=reg), DCTPoissonSolver(g)
    a, b = ps.solve(rhs - rhs.mean()), pd.solve(rhs - rhs.mean())
    assert np.max(np.abs(a - b)) < 1e-10
    assert abs(a.mean()) < 1e-12 and abs(b.mean()) < 1e-12
    # an incompatible rhs: both drop its mean
    a2, b2 = ps.solve(rhs), pd.solve(rhs)
    assert np.max(np.abs(a2 - a)) < 1e-10 and np.max(np.abs(b2 - b)) < 1e-10
    with pytest.raises(ValueError):
        PoissonSolver(g, ALL_NEUMANN, regularisation="other")


def test_factory_selection():
    g = Grid(1.0, 1.0, 16, 16, periodic_x=False, periodic_y=False)
    assert isinstance(make_poisson_solver(g, ALL_NEUMANN), DCTPoissonSolver)
    s = make_poisson_solver(g, {**ALL_NEUMANN, "right": "dirichlet"})
    assert isinstance(s, PoissonSolver) and not s.singular
    gp = Grid(1.0, 1.0, 16, 16, periodic_x=True, periodic_y=False)
    s = make_poisson_solver(gp, {**ALL_NEUMANN, "left": "periodic", "right": "periodic"})
    assert isinstance(s, PoissonSolver) and s.singular


@pytest.mark.parametrize("sides", [
    ALL_NEUMANN,
    {"left": "neumann", "right": "dirichlet", "bottom": "neumann", "top": "neumann"},
    {"left": "dirichlet", "right": "neumann", "bottom": "neumann", "top": "neumann"},
    {"left": "neumann", "right": "neumann", "bottom": "dirichlet", "top": "neumann"},
    {"left": "neumann", "right": "neumann", "bottom": "neumann", "top": "dirichlet"},
    {"left": "periodic", "right": "periodic", "bottom": "periodic", "top": "periodic"},
    {"left": "periodic", "right": "periodic", "bottom": "neumann", "top": "neumann"},
    {"left": "periodic", "right": "periodic", "bottom": "neumann", "top": "dirichlet"},
    {"left": "dirichlet", "right": "neumann", "bottom": "periodic", "top": "periodic"},
])
def test_discrete_residual(sides):
    g = Grid(1.0, 0.75, 24, 16, periodic_x=False, periodic_y=False)
    rng = np.random.default_rng(2)
    rhs = rng.standard_normal(g.shape_p())
    for reg in ("bordered", "pin"):
        s = PoissonSolver(g, sides, regularisation=reg)
        phi = s.solve(rhs)
        target = rhs - rhs.mean() if s.singular else rhs
        res = neg_laplacian(phi, g.dx, g.dy, sides) - target
        assert np.max(np.abs(res)) < 1e-10 * np.max(np.abs(rhs))
        assert np.allclose((s.matrix @ phi.ravel()).reshape(g.shape_p()), target, atol=1e-10)
        if s.singular:
            assert abs(phi.mean()) < 1e-12


def _convergence(make_solver, exact, rhs_fn, sides_periodic=(False, False)):
    errs = []
    for N in (16, 32):
        g = Grid(1.0, 0.8, N, 3 * N // 4, periodic_x=sides_periodic[0], periodic_y=sides_periodic[1])
        X, Y = g.meshgrid("centres")
        s = make_solver(g)
        phi = s.solve(rhs_fn(X, Y))
        ex = exact(X, Y)
        if s.singular:
            ex = ex - ex.mean()
        errs.append(np.max(np.abs(phi - ex)))
    return errs


def test_manufactured_neumann():
    Lx, Ly = 1.0, 0.8
    exact = lambda X, Y: np.cos(np.pi * X / Lx) * np.cos(np.pi * Y / Ly)
    rhs = lambda X, Y: np.pi**2 * (1 / Lx**2 + 1 / Ly**2) * exact(X, Y)
    e_sparse = _convergence(lambda g: PoissonSolver(g, ALL_NEUMANN), exact, rhs)
    e_dct = _convergence(lambda g: DCTPoissonSolver(g), exact, rhs)
    for e in (e_sparse, e_dct):
        assert e[1] < 5e-3
        assert 3.5 < e[0] / e[1] < 4.5          # second order


def test_manufactured_dirichlet_x():
    Lx, Ly = 1.0, 0.8
    sides = {"left": "dirichlet", "right": "dirichlet", "bottom": "neumann", "top": "neumann"}
    exact = lambda X, Y: np.sin(np.pi * X / Lx) * np.cos(np.pi * Y / Ly)
    rhs = lambda X, Y: np.pi**2 * (1 / Lx**2 + 1 / Ly**2) * exact(X, Y)
    e = _convergence(lambda g: PoissonSolver(g, sides), exact, rhs)
    assert e[1] < 5e-3 and 3.5 < e[0] / e[1] < 4.5


def test_manufactured_periodic_and_mixed():
    Lx, Ly = 1.0, 0.8
    per = {"left": "periodic", "right": "periodic", "bottom": "periodic", "top": "periodic"}
    exact = lambda X, Y: np.sin(2 * np.pi * X / Lx) * np.cos(2 * np.pi * Y / Ly)
    rhs = lambda X, Y: 4 * np.pi**2 * (1 / Lx**2 + 1 / Ly**2) * exact(X, Y)
    e = _convergence(lambda g: PoissonSolver(g, per), exact, rhs, (True, True))
    assert e[1] < 2e-2 and 3.5 < e[0] / e[1] < 4.5
    mixed = {"left": "periodic", "right": "periodic", "bottom": "neumann", "top": "neumann"}
    exact2 = lambda X, Y: np.sin(2 * np.pi * X / Lx) * np.cos(np.pi * Y / Ly)
    rhs2 = lambda X, Y: np.pi**2 * (4 / Lx**2 + 1 / Ly**2) * exact2(X, Y)
    e = _convergence(lambda g: PoissonSolver(g, mixed), exact2, rhs2, (True, False))
    assert e[1] < 1e-2 and 3.5 < e[0] / e[1] < 4.5


def test_side_validation():
    g = Grid(1.0, 1.0, 16, 16, periodic_x=False, periodic_y=False)
    with pytest.raises(ValueError):
        PoissonSolver(g, {"left": "neumann", "right": "neumann", "bottom": "neumann"})
    with pytest.raises(ValueError):
        PoissonSolver(g, {**ALL_NEUMANN, "top": "robin"})
    with pytest.raises(ValueError):
        PoissonSolver(g, {**ALL_NEUMANN, "left": "periodic"})
    with pytest.raises(ValueError):
        PoissonSolver(g, ALL_NEUMANN).solve(np.zeros((8, 8)))
