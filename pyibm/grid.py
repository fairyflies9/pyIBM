"""Uniform Cartesian grid for the 2D immersed boundary solvers.

Conventions (see docs/numerics.md): arrays are shaped (Nx, Ny) and indexed [i, j] with i
along x. The domain is [0, Lx) x [0, Ly). Collocated nodes sit at x_i = i*dx (periodic FFT
back end). The MAC back end keeps pressure at cell centres, u on vertical faces and v on
horizontal faces.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _check_cell_count(name: str, n, periodic: bool) -> int:
    """Return the cell count n as an int, or raise ValueError if the direction cannot use it.

    A periodic direction needs an even n >= 8; a direction with walls needs n >= 4 and may be
    odd. See the ``Grid`` docstring for why.
    """
    try:
        is_int = int(n) == n
    except (TypeError, ValueError, OverflowError):
        is_int = False
    if periodic:
        if not is_int or n < 8 or int(n) % 2 != 0:
            raise ValueError(f"{name} must be an even integer >= 8 in a periodic direction, "
                             f"got {n!r}")
    elif not is_int or n < 4:
        raise ValueError(f"{name} must be an integer >= 4 in a non-periodic direction, got {n!r}")
    return int(n)


@dataclass(frozen=True)
class Grid:
    """Uniform grid on [0, Lx) x [0, Ly) with Nx by Ny cells.

    Cell counts: a periodic direction needs an even N >= 8. The periodic FFT back end treats
    the wave number N/2 (the grid-scale mode (-1)^i) on its own, and that wave number only
    exists when N is even (``pyibm.fluid.periodic_fft``); we use the same rule for a periodic
    direction of the MAC grid. A direction with walls needs N >= 4 and N may be odd: 4 cells
    is the smallest box in which a 4-point kernel stencil fits between the walls on both MAC
    velocity fields. Along a wall-bounded direction the velocity component normal to the walls
    has N + 1 face nodes and the other component has N, at the cell centres (along x the
    u-faces have N + 1 nodes and the v-faces N; along y the reverse). A 4-point stencil on the
    cell-centre nodes needs i0 - 1 >= 0 and i0 + 2 <= N - 1, so N >= 4.

    Attributes
    ----------
    x, y : collocated node coordinates i*dx, j*dy (shapes (Nx,), (Ny,)).
    xc, yc : cell-centre coordinates (i + 1/2)*dx, (j + 1/2)*dy.
    xu, yu : coordinates of the u-faces, x = i*dx for i = 0..Nx (Nx + 1 values) when x is not
        periodic, i = 0..Nx-1 when it is; y = yc.
    xv, yv : coordinates of the v-faces, x = xc; y = j*dy for j = 0..Ny (or 0..Ny-1 periodic).
    """

    Lx: float
    Ly: float
    Nx: int
    Ny: int
    periodic_x: bool = True
    periodic_y: bool = True
    dx: float = field(init=False)
    dy: float = field(init=False)
    x: np.ndarray = field(init=False, repr=False, compare=False)
    y: np.ndarray = field(init=False, repr=False, compare=False)
    xc: np.ndarray = field(init=False, repr=False, compare=False)
    yc: np.ndarray = field(init=False, repr=False, compare=False)
    xu: np.ndarray = field(init=False, repr=False, compare=False)
    yu: np.ndarray = field(init=False, repr=False, compare=False)
    xv: np.ndarray = field(init=False, repr=False, compare=False)
    yv: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not (np.isfinite(self.Lx) and np.isfinite(self.Ly) and self.Lx > 0 and self.Ly > 0):
            raise ValueError(f"Lx and Ly must be finite and positive, got Lx={self.Lx!r}, "
                             f"Ly={self.Ly!r}")
        object.__setattr__(self, "Nx", _check_cell_count("Nx", self.Nx, self.periodic_x))
        object.__setattr__(self, "Ny", _check_cell_count("Ny", self.Ny, self.periodic_y))
        dx = self.Lx / self.Nx
        dy = self.Ly / self.Ny
        object.__setattr__(self, "dx", dx)
        object.__setattr__(self, "dy", dy)
        object.__setattr__(self, "x", np.arange(self.Nx) * dx)
        object.__setattr__(self, "y", np.arange(self.Ny) * dy)
        object.__setattr__(self, "xc", (np.arange(self.Nx) + 0.5) * dx)
        object.__setattr__(self, "yc", (np.arange(self.Ny) + 0.5) * dy)
        nu = self.Nx if self.periodic_x else self.Nx + 1
        nv = self.Ny if self.periodic_y else self.Ny + 1
        object.__setattr__(self, "xu", np.arange(nu) * dx)
        object.__setattr__(self, "yu", self.yc.copy())
        object.__setattr__(self, "xv", self.xc.copy())
        object.__setattr__(self, "yv", np.arange(nv) * dy)

    # ------------------------------------------------------------------ constructors
    @classmethod
    def periodic_box(cls, Lx: float, Ly: float, Nx: int, Ny: int) -> "Grid":
        """Doubly periodic grid for the FFT back end (requires dx == dy)."""
        g = cls(Lx, Ly, Nx, Ny, periodic_x=True, periodic_y=True)
        if not np.isclose(g.dx, g.dy, rtol=1e-12, atol=0.0):
            raise ValueError(f"the periodic FFT back end needs dx == dy, got dx={g.dx}, dy={g.dy}")
        return g

    # ------------------------------------------------------------------ shapes
    def shape_p(self) -> tuple[int, int]:
        """Shape of cell-centred fields (and of collocated node fields)."""
        return (self.Nx, self.Ny)

    def shape_u(self) -> tuple[int, int]:
        """Shape of the u-face field."""
        return (len(self.xu), self.Ny)

    def shape_v(self) -> tuple[int, int]:
        """Shape of the v-face field."""
        return (self.Nx, len(self.yv))

    # ------------------------------------------------------------------ periodic helpers
    def wrap(self, X: np.ndarray) -> np.ndarray:
        """Return a copy of positions X (..., 2) mapped into [0, L) in each periodic direction."""
        X = np.array(X, dtype=float, copy=True)
        if self.periodic_x:
            X[..., 0] = np.mod(X[..., 0], self.Lx)
        if self.periodic_y:
            X[..., 1] = np.mod(X[..., 1], self.Ly)
        return X

    def min_image(self, d: np.ndarray) -> np.ndarray:
        """Return a copy of separation vectors d (..., 2) reduced to the minimum image."""
        d = np.array(d, dtype=float, copy=True)
        if self.periodic_x:
            d[..., 0] -= self.Lx * np.round(d[..., 0] / self.Lx)
        if self.periodic_y:
            d[..., 1] -= self.Ly * np.round(d[..., 1] / self.Ly)
        return d

    def meshgrid(self, which: str = "nodes") -> tuple[np.ndarray, np.ndarray]:
        """Coordinate arrays (X, Y) of shape (nx, ny) for 'nodes', 'centres', 'u' or 'v'."""
        if which == "nodes":
            return np.meshgrid(self.x, self.y, indexing="ij")
        if which == "centres":
            return np.meshgrid(self.xc, self.yc, indexing="ij")
        if which == "u":
            return np.meshgrid(self.xu, self.yu, indexing="ij")
        if which == "v":
            return np.meshgrid(self.xv, self.yv, indexing="ij")
        raise ValueError(f"unknown field location {which!r}")


@dataclass(frozen=True)
class FieldLayout:
    """Where a scalar grid field lives: nodes x0 + i*hx (i < nx), y0 + j*hy (j < ny).

    Used by the delta-function coupling to spread to and interpolate from collocated nodes,
    u-faces or v-faces with one code path.
    """

    x0: float
    y0: float
    hx: float
    hy: float
    nx: int
    ny: int
    periodic_x: bool
    periodic_y: bool

    @property
    def shape(self) -> tuple[int, int]:
        return (self.nx, self.ny)

    def coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        """1-D coordinate arrays (x, y) of the nodes."""
        return self.x0 + np.arange(self.nx) * self.hx, self.y0 + np.arange(self.ny) * self.hy


def layout_collocated(grid: Grid) -> FieldLayout:
    """Collocated nodes x_i = i*dx, y_j = j*dy (periodic FFT back end)."""
    return FieldLayout(0.0, 0.0, grid.dx, grid.dy, grid.Nx, grid.Ny, grid.periodic_x, grid.periodic_y)


def layout_u(grid: Grid) -> FieldLayout:
    """u-faces of the MAC grid: x = i*dx, y = (j + 1/2)*dy."""
    return FieldLayout(0.0, 0.5 * grid.dy, grid.dx, grid.dy, len(grid.xu), grid.Ny,
                       grid.periodic_x, grid.periodic_y)


def layout_v(grid: Grid) -> FieldLayout:
    """v-faces of the MAC grid: x = (i + 1/2)*dx, y = j*dy."""
    return FieldLayout(0.5 * grid.dx, 0.0, grid.dx, grid.dy, grid.Nx, len(grid.yv),
                       grid.periodic_x, grid.periodic_y)
