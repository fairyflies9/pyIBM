"""Tests for pyibm.grid: spacing, face coordinates, cell-count rules, periodic helpers."""

import numpy as np
import pytest

from pyibm.grid import Grid, FieldLayout, layout_collocated, layout_u, layout_v


def test_spacing_and_coordinates():
    g = Grid(0.4, 0.8, 16, 32, periodic_x=False, periodic_y=True)
    assert g.dx == pytest.approx(0.025)
    assert g.dy == pytest.approx(0.025)
    assert g.x[1] == pytest.approx(g.dx)
    assert g.xc[0] == pytest.approx(0.5 * g.dx)
    assert g.shape_p() == (16, 32)
    assert g.shape_u() == (17, 32)       # both wall faces stored in a non-periodic direction
    assert g.shape_v() == (16, 32)       # periodic y: face Ny coincides with face 0
    assert g.xu[-1] == pytest.approx(0.4)
    assert len(g.yv) == 32


def test_validation():
    with pytest.raises(ValueError):
        Grid(1.0, 1.0, 15, 16)
    with pytest.raises(ValueError):
        Grid(1.0, 1.0, 4, 16)
    with pytest.raises(ValueError):
        Grid.periodic_box(1.0, 1.0, 16, 32)   # dx != dy
    g = Grid.periodic_box(1.0, 2.0, 16, 32)
    assert g.periodic_x and g.periodic_y
    # the box lengths must be finite and positive (NaN and inf used to slip through and gave
    # NaN coordinates)
    for L in (float("nan"), float("inf"), -float("inf"), -1.0, 0.0):
        with pytest.raises(ValueError, match="Lx and Ly must be finite and positive"):
            Grid(L, 1.0, 16, 16)
        with pytest.raises(ValueError, match="Lx and Ly must be finite and positive"):
            Grid(1.0, L, 16, 16, periodic_x=False, periodic_y=False)


def test_periodic_direction_needs_even_n_at_least_8():
    msg = "in a periodic direction"
    for n in (4, 6, 7, 9, 15):
        with pytest.raises(ValueError, match=msg):
            Grid(1.0, 1.0, n, 16)                                  # x periodic
        with pytest.raises(ValueError, match=msg):
            Grid(1.0, 1.0, 16, n, periodic_x=False)                # y periodic, x walls
    for n in (8, 10, 16):
        g = Grid(1.0, 1.0, n, n)
        assert g.Nx == n and g.Ny == n
    # an odd count is fine along the walls but not along the periodic direction
    g = Grid(1.0, 1.0, 9, 16, periodic_x=False, periodic_y=True)
    assert g.shape_u() == (10, 16) and g.shape_v() == (9, 16)
    with pytest.raises(ValueError, match="Ny must be an even integer >= 8"):
        Grid(1.0, 1.0, 16, 9, periodic_x=False, periodic_y=True)
    with pytest.raises(ValueError, match="Nx must be an even integer >= 8"):
        Grid.periodic_box(1.0, 1.0, 9, 9)


def test_nonperiodic_direction_accepts_any_integer_at_least_4():
    """With walls, N >= 4 is allowed and N may be odd; a smaller N is refused."""
    for nx, ny in ((4, 4), (5, 7), (13, 4), (4, 9)):
        g = Grid(1.0, 0.7, nx, ny, periodic_x=False, periodic_y=False)
        assert (g.Nx, g.Ny) == (nx, ny)
        assert g.dx == pytest.approx(1.0 / nx) and g.dy == pytest.approx(0.7 / ny)
        assert g.shape_p() == (nx, ny)
        assert g.shape_u() == (nx + 1, ny) and g.shape_v() == (nx, ny + 1)
        assert g.xu[-1] == pytest.approx(1.0) and g.yv[-1] == pytest.approx(0.7)
        assert g.xc[-1] == pytest.approx(1.0 - 0.5 / nx)
        lu, lv = layout_u(g), layout_v(g)
        assert lu.shape == g.shape_u() and lv.shape == g.shape_v()
        assert layout_collocated(g).shape == (nx, ny)
    msg = "in a non-periodic direction"
    for n in (1, 2, 3):
        with pytest.raises(ValueError, match="Nx must be an integer >= 4 " + msg):
            Grid(1.0, 1.0, n, 16, periodic_x=False)
        with pytest.raises(ValueError, match="Ny must be an integer >= 4 " + msg):
            Grid(1.0, 1.0, 16, n, periodic_x=True, periodic_y=False)


def test_cell_counts_must_be_integers():
    g = Grid(1.0, 1.0, 16.0, np.int64(9), periodic_y=False)       # integral values are fine
    assert g.Nx == 16 and g.Ny == 9 and type(g.Nx) is int and type(g.Ny) is int
    for bad in (8.5, "16", None, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            Grid(1.0, 1.0, bad, 16)
        with pytest.raises(ValueError):
            Grid(1.0, 1.0, 16, bad, periodic_y=False)


def test_wrap_and_min_image():
    g = Grid(1.0, 1.0, 16, 16)
    X = np.array([[-0.1, 1.2], [0.5, 0.5]])
    W = g.wrap(X)
    assert np.allclose(W, [[0.9, 0.2], [0.5, 0.5]])
    assert X[0, 0] == -0.1                      # input untouched
    d = g.min_image(np.array([[0.9, -0.95]]))
    assert np.allclose(d, [[-0.1, 0.05]])
    gw = Grid(1.0, 1.0, 16, 16, periodic_x=False, periodic_y=False)
    assert np.allclose(gw.wrap(X), X)
    assert np.allclose(gw.min_image(np.array([[0.9, -0.95]])), [[0.9, -0.95]])


def test_layouts():
    g = Grid(1.0, 0.5, 16, 8, periodic_x=False, periodic_y=False)
    lc, lu, lv = layout_collocated(g), layout_u(g), layout_v(g)
    assert lc.shape == (16, 8) and lu.shape == (17, 8) and lv.shape == (16, 9)
    xu, yu = lu.coordinates()
    assert xu[-1] == pytest.approx(1.0) and yu[0] == pytest.approx(0.5 * g.dy)
    xv, yv = lv.coordinates()
    assert xv[0] == pytest.approx(0.5 * g.dx) and yv[-1] == pytest.approx(0.5)
    assert isinstance(lc, FieldLayout)
    X, Y = g.meshgrid("u")
    assert X.shape == (17, 8) and Y[0, 1] == pytest.approx(1.5 * g.dy)
