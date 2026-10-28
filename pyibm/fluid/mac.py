"""MAC-grid projection back end with Dirichlet, inflow, outflow and periodic sides.

This follows B. Seibold, "A compact and fast Matlab code solving the incompressible
Navier-Stokes equations on rectangular domains", course notes for MIT 18.086 (2008),
http://www-math.mit.edu/18086: a staggered (MAC) grid, explicit advection in conservative
form with his central/upwind blend, implicit viscosity by pre-factored sparse LU, and
Chorin's projection with a pressure Poisson solve. We generalise it to

- velocity data that vary in space and time on each side (``BoundaryConditions``),
- an outflow side (zero-gradient velocity ghosts, ``p = 0`` on the boundary, and the normal
  face velocity on that side corrected by the projection like an interior face),
- periodic directions,
- a body force on the faces (the spread immersed-boundary force),
- the midpoint scheme of Peskin 2002 sec. 7 (``half_step`` / ``full_step``) next to the
  one-step scheme (``step``),
- an optional incremental projection (``projection="incremental"``).

Layout (see docs/numerics.md): ``p[i, j]`` at cell centres ``(xc_i, yc_j)``, shape
``(Nx, Ny)``; ``u[i, j]`` on the vertical faces ``(xu_i, yc_j)``, shape ``(Nx+1, Ny)`` with
both wall faces stored, or ``(Nx, Ny)`` when x is periodic; ``v[i, j]`` on the horizontal
faces ``(xc_i, yv_j)``, shape ``(Nx, Ny+1)`` or ``(Nx, Ny)``. Arrays are indexed ``[i, j]``
with ``i`` along x.

Projection. Each substep, of length ``h`` (``dt/2`` in ``half_step``, ``dt`` in
``full_step`` and ``step``), solves ``-L_h phi = -(1/h) D_h u**`` and sets
``u = u** - h G_h phi``. Two variants:

- ``"nonincremental"`` (the default, Chorin 1968): the predictor carries no pressure
  gradient, so ``phi`` is the whole pressure over ``rho``. With implicit viscosity the
  splitting error is first order in ``dt``: the viscous solve imposes the wall data on
  ``u**``, and the projection then leaves the tangential velocity next to a Dirichlet wall
  with a slip of ``h * d(phi)/d(tau)``, which is O(dt) wherever the pressure varies along
  the wall.
- ``"incremental"``: the predictor carries ``-h G_h p_old/rho`` with ``p_old`` the pressure
  of the last ``step`` or ``full_step`` (``self.p``; ``half_step`` uses it too), and ``phi``
  is only the increment, which is O(dt), so the wall slip is O(dt^2). This is the
  pressure-increment projection with the rotational pressure update (Brown, Cortez &
  Minion 2001; reviewed by Guermond, Minev & Shen 2006), here with the advection of our
  midpoint scheme. The first step starts from ``self.p``, zero unless the caller sets it,
  so it has the non-incremental slip once; in our tests that did not change the observed
  order. At a steady state the increment vanishes and so does the splitting error.

Measured, Taylor-Green vortex shifted so that the pressure varies along every wall, exact
velocity as time-varying Dirichlet data, 32^2, nu = 0.01, T = 0.5, central advection,
implicit viscosity, midpoint, error against an incremental dt = T/1600 run on the same
grid (a non-incremental run at that dt still carries its own slip, 1.8e-4 off the
incremental one) for dt = 0.02, 0.01, 0.005, 0.0025: closed box 1.1e-2, 5.5e-3, 2.8e-3,
1.4e-3 (non-incremental) and 3.9e-4, 9.7e-5, 2.4e-5, 6.0e-6 (incremental); with periodic x
and walls in y, with an outflow side, and with periodic x plus an outflow top, the
incremental errors also fall by 3.7 to 4.1 per halving and the non-incremental ones by 1.9
to 2.0. In a Poiseuille channel
with an outflow side (64 x 32) the incremental steady state matches the discrete fully
developed flow, dp/dx to 3.4e-5 relative, the same at dt = 1e-3, 2e-3 and 4e-3; the
non-incremental one is off by 1.3e-3 at dt = 2e-3 with the midpoint scheme (2.5e-3 with
``step``), and that halves with dt.

With ``viscous="explicit"`` there is no viscous solve to split from: the projection removes
the extra gradient exactly, the two variants give the same velocity and pressure to
round-off, and both are second order in the test above. The same holds in a doubly periodic
box with implicit viscosity, where the viscous solve commutes with the gradient.

Reported pressure. With implicit viscosity the force enters the predictor before the
Helmholtz solve ``(I - c L_h) u** = ...``, ``c = theta*nu*h``, so ``phi`` is a smoothed
pressure, close to ``(I - c L_h)^{-1} p``: next to an immersed boundary the pressure jump
would be smeared over about ``sqrt(c)``. In both variants ``self.p`` is therefore the
rotational form ``rho*(phi - c L_h phi)`` (plus ``p_old`` when incremental), the pressure
that matches the discrete momentum equation (Brown, Cortez & Minion 2001; Timmermans,
Minev & van de Vosse 1996). In the non-incremental variant the velocities do not depend on
this choice (they are bit for bit what ``p = rho*phi`` gave); in the incremental one the
rotational update is part of the scheme. ``L_h phi`` is computed as ``D_h G_h phi``, so it
uses the Poisson solver's own boundary rules. For a pressurised circle (T/R = 4, mu = 1,
closed box, midpoint, t = 0.05, mean pressure inside r < R - 3dx minus the mean over
R + 3dx < r < 0.45) the jump error was -4.1e-3, -5.9e-3, -7.2e-3 on 32^2, 64^2, 128^2 with
``p = rho*phi`` (growing, because dt is proportional to dx) and is -5.5e-5, -8.5e-6,
-2.7e-6 with the rotational form.

Order in time. The midpoint scheme is second order in time on this back end only when

- the advection blend parameter does not depend on ``dt``, i.e. ``advection="central"``
  (or ``"upwind"``, which is first order in space). The default ``"blend"`` uses Seibold's
  ``gamma = min(1.2 dt max(|u|/dx, |v|/dy), 1)``, which is proportional to ``dt`` and adds
  a numerical viscosity of about ``gamma*|u|*dx/2``, at most ``0.6 dt max|u| |u|``. That is
  O(dt) and independent of the grid spacing, so with ``"blend"`` the midpoint scheme is
  first order in time even in a doubly periodic box (measured numbers in the ``MACSolver``
  docstring);
- and, with the non-incremental projection and implicit viscosity, no Dirichlet wall
  carries a tangential pressure gradient (otherwise the O(dt) slip above).

Use ``"central"`` when the cell Reynolds number ``max|u| dx/nu`` is about 2 or less; the
extra viscosity of ``"blend"`` can be compared with ``nu`` from the formula above. The
one-step scheme (``step``, backward Euler) is first order in time in every case.

References: A. J. Chorin, "Numerical solution of the Navier-Stokes equations", Math. Comp.
22 (1968) 745-762. D. L. Brown, R. Cortez and M. L. Minion, "Accurate projection methods
for the incompressible Navier-Stokes equations", J. Comput. Phys. 168 (2001) 464-499.
L. J. P. Timmermans, P. D. Minev and F. N. van de Vosse, "An approximate projection scheme
for incompressible flow using spectral elements", Int. J. Numer. Methods Fluids 22 (1996)
673-688. J. L. Guermond, P. Minev and J. Shen, "An overview of projection methods for
incompressible flows", Comput. Methods Appl. Mech. Engrg. 195 (2006) 6011-6045.
C. S. Peskin, "The immersed boundary method", Acta Numerica 11 (2002) 479-517.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from pyibm.fluid.poisson import laplacian_matrix, make_poisson_solver
from pyibm.grid import FieldLayout, Grid, layout_u, layout_v

SIDES = ("left", "right", "bottom", "top")
BC_KINDS = ("periodic", "dirichlet", "outflow")


@dataclass
class BoundaryConditions:
    """Per-side velocity boundary conditions for the MAC back end.

    ``left``, ``right``, ``bottom``, ``top`` are each ``"periodic"``, ``"dirichlet"`` or
    ``"outflow"``. Periodic must be set on both sides of a direction; at most one side may
    be outflow. ``bc_<side>(coord, t) -> (u, v)`` gives the velocity on a Dirichlet side at
    the given coordinates along it (``y`` for left/right, ``x`` for bottom/top) and time;
    ``None`` means a no-slip wall at rest. The callables must be vectorised and return
    arrays shaped like ``coord`` (or scalars).
    """

    left: str
    right: str
    bottom: str
    top: str
    bc_left: Callable | None = None
    bc_right: Callable | None = None
    bc_bottom: Callable | None = None
    bc_top: Callable | None = None

    def __post_init__(self) -> None:
        for s in SIDES:
            kind = getattr(self, s)
            if kind not in BC_KINDS:
                raise ValueError(f"side {s!r}: unknown condition {kind!r}; use one of {BC_KINDS}")
        if (self.left == "periodic") != (self.right == "periodic"):
            raise ValueError("periodic must be set on both left and right")
        if (self.bottom == "periodic") != (self.top == "periodic"):
            raise ValueError("periodic must be set on both bottom and top")
        n_out = sum(getattr(self, s) == "outflow" for s in SIDES)
        if n_out > 1:
            raise ValueError(f"at most one outflow side is supported, got {n_out}")

    @property
    def periodic_x(self) -> bool:
        return self.left == "periodic"

    @property
    def periodic_y(self) -> bool:
        return self.bottom == "periodic"

    @property
    def outflow_side(self) -> str | None:
        for s in SIDES:
            if getattr(self, s) == "outflow":
                return s
        return None


class MACSolver:
    """Projection solver on the MAC grid (the fluid solver interface and scheme are in
    docs/numerics.md).

    ``MACSolver(grid, rho, mu, dt, bcs, advection="blend", viscous="implicit",
    projection="nonincremental")``.

    ``advection`` is ``"blend"`` (Seibold's gamma, eq. 28), ``"central"`` (gamma = 0) or
    ``"upwind"`` (gamma = 1). Because the blend's gamma is proportional to ``dt``, the
    default ``"blend"`` makes the midpoint scheme first order in time (module docstring):
    on a 32^2 periodic Taylor-Green vortex (nu = 0.01, T = 0.5) we measured midpoint errors
    against a fine-``dt`` run of 6.2e-2, 3.3e-2, 1.7e-2, 8.6e-3 for dt = 0.02, 0.01, 0.005,
    0.0025 with ``"blend"``, and 5.5e-6, 1.4e-6, 3.4e-7, 8.5e-8 with ``"central"``.

    ``viscous`` is ``"implicit"`` (backward Euler in ``step`` and ``half_step``,
    Crank-Nicolson in ``full_step``) or ``"explicit"``.

    ``projection`` is ``"nonincremental"`` (Chorin's, O(dt) wall slip with implicit
    viscosity) or ``"incremental"`` (pressure increment with the rotational update, O(dt^2)
    wall slip); see the module docstring. In a doubly periodic box, or with
    ``viscous="explicit"``, the two give the same ``u``, ``v`` and ``p`` to round-off.

    State: ``u``, ``v`` on ``layout_u``, ``layout_v``; ``p`` at cell centres (the rotational
    pressure of the last ``step`` or ``full_step``, and the pressure the incremental
    projection starts from); ``t`` the time of the current state (updated by ``step`` and
    ``full_step``, used by ``vorticity``).
    """

    def __init__(self, grid: Grid, rho: float, mu: float, dt: float, bcs: BoundaryConditions,
                 advection: str = "blend", viscous: str = "implicit",
                 projection: str = "nonincremental"):
        if advection not in ("blend", "central", "upwind"):
            raise ValueError(f"advection must be 'blend', 'central' or 'upwind', got {advection!r}")
        if viscous not in ("implicit", "explicit"):
            raise ValueError(f"viscous must be 'implicit' or 'explicit', got {viscous!r}")
        if projection not in ("nonincremental", "incremental"):
            raise ValueError("projection must be 'nonincremental' or 'incremental', "
                             f"got {projection!r}")
        if bcs.periodic_x != grid.periodic_x or bcs.periodic_y != grid.periodic_y:
            raise ValueError("grid periodicity (periodic_x, periodic_y) must match the "
                             "periodic sides of the BoundaryConditions")
        if rho <= 0 or mu < 0 or dt <= 0:
            raise ValueError("need rho > 0, mu >= 0, dt > 0")
        self.grid = grid
        self.rho = float(rho)
        self.mu = float(mu)
        self.nu = self.mu / self.rho
        self.dt = float(dt)
        self.bcs = bcs
        self.advection = advection
        self.viscous = viscous
        self.projection = projection
        self.layout_u: FieldLayout = layout_u(grid)
        self.layout_v: FieldLayout = layout_v(grid)
        self.u = np.zeros(grid.shape_u())
        self.v = np.zeros(grid.shape_v())
        self.p = np.zeros(grid.shape_p())
        self.t = 0.0

        Nx, Ny = grid.Nx, grid.Ny
        # Faces updated by the predictor and the projection ("free" faces) form a contiguous
        # block: all u-faces except Dirichlet wall faces (and, in periodic x, the duplicate
        # face Nx of the full (Nx+1, Ny) working array); likewise for v.
        if grid.periodic_x:
            self._iu = slice(0, Nx)
        else:
            self._iu = slice(1 if bcs.left == "dirichlet" else 0,
                             Nx if bcs.right == "dirichlet" else Nx + 1)
        if grid.periodic_y:
            self._jv = slice(0, Ny)
        else:
            self._jv = slice(1 if bcs.bottom == "dirichlet" else 0,
                             Ny if bcs.top == "dirichlet" else Ny + 1)
        self._closed = bcs.outflow_side is None

        p_sides = {}
        for s in SIDES:
            kind = getattr(bcs, s)
            p_sides[s] = {"periodic": "periodic", "dirichlet": "neumann", "outflow": "dirichlet"}[kind]
        self.poisson = make_poisson_solver(grid, p_sides)
        self._lu_cache: dict[float, tuple] = {}

        if viscous == "explicit":
            dt_max = 1.0 / (2.0 * self.nu * (1.0 / grid.dx**2 + 1.0 / grid.dy**2)) if self.nu > 0 else np.inf
            if self.dt > dt_max:
                warnings.warn(f"explicit viscosity: dt = {self.dt:g} exceeds the stability limit "
                              f"1/(2 nu (1/dx^2 + 1/dy^2)) = {dt_max:g}", RuntimeWarning, stacklevel=2)

    # ------------------------------------------------------------------ boundary data
    def _wall(self, side: str, coord: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
        """(u, v) prescribed on a Dirichlet side at coordinates ``coord`` and time ``t``."""
        f = getattr(self.bcs, "bc_" + side)
        coord = np.asarray(coord, dtype=float)
        if f is None:
            return np.zeros(coord.shape), np.zeros(coord.shape)
        ub, vb = f(coord, t)
        ub = np.array(np.broadcast_to(np.asarray(ub, dtype=float), coord.shape), dtype=float)
        vb = np.array(np.broadcast_to(np.asarray(vb, dtype=float), coord.shape), dtype=float)
        return ub, vb

    def extend_u(self, u: np.ndarray, t: float) -> np.ndarray:
        """u with one ghost layer on every side: shape ``(Nx+3, Ny+2)``.

        ``ue[1 + i, 1 + j]`` is the u-face ``(i, j)`` for ``i = 0..Nx``, ``j = 0..Ny-1``.
        Dirichlet x-sides: the wall face itself holds the prescribed normal velocity (the
        layer beyond it is a copy, never used by an interior stencil). Dirichlet y-sides:
        the tangential ghost is the reflection ``2*u_wall - u_interior`` (Seibold eq. 35).
        Outflow sides: zero-gradient copy. Periodic sides: wrap, so in periodic x the layer
        ``1 + Nx`` is face 0 again and both outer layers are the wrapped neighbours.
        """
        g, bcs = self.grid, self.bcs
        Nx, Ny = g.Nx, g.Ny
        u = np.asarray(u, dtype=float)
        if u.shape != g.shape_u():
            raise ValueError(f"u has shape {u.shape}, expected {g.shape_u()}")
        uu = u.copy()
        if not g.periodic_x:
            if bcs.left == "dirichlet":
                uu[0, :] = self._wall("left", g.yc, t)[0]
            if bcs.right == "dirichlet":
                uu[Nx, :] = self._wall("right", g.yc, t)[0]
        # tangential ghosts (y) on the stored columns
        uy = np.empty((uu.shape[0], Ny + 2))
        uy[:, 1:-1] = uu
        if g.periodic_y:
            uy[:, 0] = uu[:, -1]
            uy[:, -1] = uu[:, 0]
        else:
            uy[:, 0] = 2.0 * self._wall("bottom", g.xu, t)[0] - uu[:, 0] if bcs.bottom == "dirichlet" else uu[:, 0]
            uy[:, -1] = 2.0 * self._wall("top", g.xu, t)[0] - uu[:, -1] if bcs.top == "dirichlet" else uu[:, -1]
        # normal direction (x)
        ue = np.empty((Nx + 3, Ny + 2))
        if g.periodic_x:
            ue[1:Nx + 1] = uy
            ue[0] = uy[Nx - 1]
            ue[Nx + 1] = uy[0]
            ue[Nx + 2] = uy[1]
        else:
            ue[1:Nx + 2] = uy
            ue[0] = uy[0]
            ue[Nx + 2] = uy[Nx]
        return ue

    def extend_v(self, v: np.ndarray, t: float) -> np.ndarray:
        """v with one ghost layer on every side: shape ``(Nx+2, Ny+3)``.

        ``ve[1 + i, 1 + j]`` is the v-face ``(i, j)`` for ``i = 0..Nx-1``, ``j = 0..Ny``.
        Same rules as ``extend_u`` with x and y swapped: Dirichlet y-sides set the wall
        faces, Dirichlet x-sides give the tangential reflection ghost, outflow copies,
        periodic wraps.
        """
        g, bcs = self.grid, self.bcs
        Nx, Ny = g.Nx, g.Ny
        v = np.asarray(v, dtype=float)
        if v.shape != g.shape_v():
            raise ValueError(f"v has shape {v.shape}, expected {g.shape_v()}")
        vv = v.copy()
        if not g.periodic_y:
            if bcs.bottom == "dirichlet":
                vv[:, 0] = self._wall("bottom", g.xc, t)[1]
            if bcs.top == "dirichlet":
                vv[:, Ny] = self._wall("top", g.xc, t)[1]
        vx = np.empty((Nx + 2, vv.shape[1]))
        vx[1:-1] = vv
        if g.periodic_x:
            vx[0] = vv[-1]
            vx[-1] = vv[0]
        else:
            vx[0] = 2.0 * self._wall("left", g.yv, t)[1] - vv[0] if bcs.left == "dirichlet" else vv[0]
            vx[-1] = 2.0 * self._wall("right", g.yv, t)[1] - vv[-1] if bcs.right == "dirichlet" else vv[-1]
        ve = np.empty((Nx + 2, Ny + 3))
        if g.periodic_y:
            ve[:, 1:Ny + 1] = vx
            ve[:, 0] = vx[:, Ny - 1]
            ve[:, Ny + 1] = vx[:, 0]
            ve[:, Ny + 2] = vx[:, 1]
        else:
            ve[:, 1:Ny + 2] = vx
            ve[:, 0] = vx[:, 0]
            ve[:, Ny + 2] = vx[:, Ny]
        return ve

    # ------------------------------------------------------------------ shape helpers
    def _full_u(self, u: np.ndarray) -> np.ndarray:
        """Working array over faces ``i = 0..Nx`` (appends the wrapped face in periodic x)."""
        u = np.asarray(u, dtype=float)
        if self.grid.periodic_x:
            return np.concatenate([u, u[:1]], axis=0)
        return u.copy()

    def _full_v(self, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=float)
        if self.grid.periodic_y:
            return np.concatenate([v, v[:, :1]], axis=1)
        return v.copy()

    def _store_u(self, uf: np.ndarray) -> np.ndarray:
        return uf[:self.grid.Nx].copy() if self.grid.periodic_x else uf

    def _store_v(self, vf: np.ndarray) -> np.ndarray:
        return vf[:, :self.grid.Ny].copy() if self.grid.periodic_y else vf

    def _weights_u(self) -> np.ndarray:
        """Trapezoidal face weights along x for u sums: 1/2 on stored wall faces."""
        w = np.ones(self.grid.shape_u()[0])
        if not self.grid.periodic_x:
            w[0] = w[-1] = 0.5
        return w

    def _weights_v(self) -> np.ndarray:
        w = np.ones(self.grid.shape_v()[1])
        if not self.grid.periodic_y:
            w[0] = w[-1] = 0.5
        return w

    # ------------------------------------------------------------------ operators
    def _gamma(self, ua: np.ndarray, va: np.ndarray, h: float) -> float:
        """Seibold eq. 28: ``gamma = min(1.2 h max(|u|/dx, |v|/dy), 1)`` for ``"blend"``.

        ``h`` is the substep (``dt/2`` or ``dt``), so the blend's numerical viscosity, about
        ``gamma*|u|*dx/2 <= 0.6 h max|u| |u|``, is O(dt) and does not shrink with the grid:
        with ``"blend"`` the midpoint scheme is first order in time. ``"central"`` and
        ``"upwind"`` give a ``gamma`` that does not depend on ``dt``.
        """
        if self.advection == "central":
            return 0.0
        if self.advection == "upwind":
            return 1.0
        s = max(np.max(np.abs(ua)) / self.grid.dx, np.max(np.abs(va)) / self.grid.dy)
        return float(min(1.2 * h * s, 1.0))

    def _advection(self, ue: np.ndarray, ve: np.ndarray, gamma: float) -> tuple[np.ndarray, np.ndarray]:
        """Conservative advection terms on the faces (Seibold eqs. 26-34).

        Returns ``((u^2)_x + (uv)_y`` on the u-faces ``(Nx+1, Ny)``,
        ``(uv)_x + (v^2)_y`` on the v-faces ``(Nx, Ny+1))``. Products are formed from
        averages (eqs. 23-25) and the central/donor-cell blend subtracts
        ``gamma*|advecting velocity|*(half difference)`` from each flux (eqs. 29-32), which is
        plain averaged central differencing for ``gamma = 0`` and conservative upwinding for
        ``gamma = 1`` (eq. 34).
        """
        dx, dy = self.grid.dx, self.grid.dy
        # (u^2)_x on u-faces: u^2 at the centres between consecutive faces
        Ua = 0.5 * (ue[:-1, 1:-1] + ue[1:, 1:-1])
        Ud = 0.5 * (ue[1:, 1:-1] - ue[:-1, 1:-1])
        Fxx = Ua * Ua - gamma * np.abs(Ua) * Ud
        U2x = (Fxx[1:] - Fxx[:-1]) / dx
        # (uv) at the cell corners (face i, face j), i = 0..Nx, j = 0..Ny
        Uav = 0.5 * (ue[1:-1, :-1] + ue[1:-1, 1:])      # u averaged vertically
        Udv = 0.5 * (ue[1:-1, 1:] - ue[1:-1, :-1])
        Vah = 0.5 * (ve[:-1, 1:-1] + ve[1:, 1:-1])      # v averaged horizontally
        Vdh = 0.5 * (ve[1:, 1:-1] - ve[:-1, 1:-1])
        UV = Uav * Vah
        Fxy = UV - gamma * np.abs(Vah) * Udv            # eq. 31: u transported by v
        UVy = (Fxy[:, 1:] - Fxy[:, :-1]) / dy
        Fyx = UV - gamma * np.abs(Uav) * Vdh            # eq. 32: v transported by u
        UVx = (Fyx[1:, :] - Fyx[:-1, :]) / dx
        # (v^2)_y on v-faces
        Va = 0.5 * (ve[1:-1, :-1] + ve[1:-1, 1:])
        Vd = 0.5 * (ve[1:-1, 1:] - ve[1:-1, :-1])
        Fyy = Va * Va - gamma * np.abs(Va) * Vd
        V2y = (Fyy[:, 1:] - Fyy[:, :-1]) / dy
        return U2x + UVy, UVx + V2y

    def _lap_u(self, ue: np.ndarray) -> np.ndarray:
        """5-point Laplacian (Seibold eq. 20) of an extended u on the faces ``(Nx+1, Ny)``."""
        dx, dy = self.grid.dx, self.grid.dy
        c = ue[1:-1, 1:-1]
        return ((ue[:-2, 1:-1] - 2.0 * c + ue[2:, 1:-1]) / dx**2
                + (ue[1:-1, :-2] - 2.0 * c + ue[1:-1, 2:]) / dy**2)

    def _lap_v(self, ve: np.ndarray) -> np.ndarray:
        """5-point Laplacian of an extended v on the faces ``(Nx, Ny+1)``."""
        dx, dy = self.grid.dx, self.grid.dy
        c = ve[1:-1, 1:-1]
        return ((ve[:-2, 1:-1] - 2.0 * c + ve[2:, 1:-1]) / dx**2
                + (ve[1:-1, :-2] - 2.0 * c + ve[1:-1, 2:]) / dy**2)

    def _laplacian_matrices(self) -> tuple[sp.csc_matrix, sp.csc_matrix]:
        """``-L_h`` restricted to the free u-faces and free v-faces (homogeneous data).

        End conditions per direction (Seibold eq. 37 and his table of ``a11``): a Dirichlet
        wall face on the point itself is removed from the unknowns (``a11 = 2``), a
        tangential Dirichlet wall half a cell away gives the reflection (``a11 = 3``), an
        outflow side gives the zero-gradient ghost (``a11 = 1``), periodic is circulant.
        """
        g, bcs = self.grid, self.bcs
        Nx, Ny = g.Nx, g.Ny

        def end(kind, on_point):
            if kind == "periodic":
                return "periodic"
            if kind == "outflow":
                return "neumann"
            return "dirichlet_point" if on_point else "dirichlet_mid"

        nxu = self._iu.stop - self._iu.start
        Ku = laplacian_matrix(nxu, Ny, g.dx, g.dy,
                              end(bcs.left, True), end(bcs.right, True),
                              end(bcs.bottom, False), end(bcs.top, False))
        nyv = self._jv.stop - self._jv.start
        Kv = laplacian_matrix(Nx, nyv, g.dx, g.dy,
                              end(bcs.left, False), end(bcs.right, False),
                              end(bcs.bottom, True), end(bcs.top, True))
        return Ku, Kv

    def _helmholtz(self, c: float) -> tuple:
        """LU factors of ``I + c*(-L_h)`` for u and v, cached by ``c`` (rebuilt if dt changes)."""
        key = float(c)
        if key not in self._lu_cache:
            if len(self._lu_cache) >= 4:
                self._lu_cache.clear()
            Ku, Kv = self._laplacian_matrices()
            Au = (sp.identity(Ku.shape[0], format="csc") + c * Ku).tocsc()
            Av = (sp.identity(Kv.shape[0], format="csc") + c * Kv).tocsc()
            self._lu_cache[key] = (splu(Au), splu(Av))
        return self._lu_cache[key]

    def _impose_walls(self, uf: np.ndarray, vf: np.ndarray, t: float) -> None:
        """Write the prescribed normal velocities at ``t`` on Dirichlet wall faces (in place)
        and refresh the duplicate wrapped face in a periodic direction."""
        g, bcs = self.grid, self.bcs
        if g.periodic_x:
            uf[g.Nx] = uf[0]
        else:
            if bcs.left == "dirichlet":
                uf[0] = self._wall("left", g.yc, t)[0]
            if bcs.right == "dirichlet":
                uf[g.Nx] = self._wall("right", g.yc, t)[0]
        if g.periodic_y:
            vf[:, g.Ny] = vf[:, 0]
        else:
            if bcs.bottom == "dirichlet":
                vf[:, 0] = self._wall("bottom", g.xc, t)[1]
            if bcs.top == "dirichlet":
                vf[:, g.Ny] = self._wall("top", g.xc, t)[1]

    def _check_compatibility(self, uf: np.ndarray, vf: np.ndarray) -> None:
        """All-closed box: the prescribed normal velocities must carry no net flux.

        The pure Neumann/periodic Poisson problem is solvable only if the right-hand side
        sums to zero, i.e. ``sum over walls of u_n * face length = 0``. We require
        ``|flux| <= 1e-10 * Umax * perimeter`` with ``Umax`` the largest prescribed normal
        speed, else ``ValueError``.
        """
        g = self.grid
        flux = 0.0
        umax = 0.0
        if not g.periodic_x:
            flux += (uf[g.Nx].sum() - uf[0].sum()) * g.dy
            umax = max(umax, np.max(np.abs(uf[0])), np.max(np.abs(uf[g.Nx])))
        if not g.periodic_y:
            flux += (vf[:, g.Ny].sum() - vf[:, 0].sum()) * g.dx
            umax = max(umax, np.max(np.abs(vf[:, 0])), np.max(np.abs(vf[:, g.Ny])))
        if abs(flux) > 1e-10 * umax * 2.0 * (g.Lx + g.Ly):
            raise ValueError(f"net inflow {flux:.6e} m^2/s through the Dirichlet walls with no "
                             "outflow side: the prescribed normal velocities must have zero net flux")

    def _gradient(self, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``G_h phi`` on all faces, with the pressure ghosts matching the Poisson operator:
        Neumann at a velocity wall (gradient 0 on that face), ``-phi`` beyond an outflow side
        (so the outflow face is corrected by ``-2 phi/h``), wrap in a periodic direction.
        This makes ``D_h G_h = L_h`` exactly, so the corrected field is divergence-free to
        round-off."""
        g, bcs = self.grid, self.bcs
        Nx, Ny = g.Nx, g.Ny
        px = np.empty((Nx + 2, Ny))
        px[1:-1] = phi
        if g.periodic_x:
            px[0], px[-1] = phi[-1], phi[0]
        else:
            px[0] = -phi[0] if bcs.left == "outflow" else phi[0]
            px[-1] = -phi[-1] if bcs.right == "outflow" else phi[-1]
        py = np.empty((Nx, Ny + 2))
        py[:, 1:-1] = phi
        if g.periodic_y:
            py[:, 0], py[:, -1] = phi[:, -1], phi[:, 0]
        else:
            py[:, 0] = -phi[:, 0] if bcs.bottom == "outflow" else phi[:, 0]
            py[:, -1] = -phi[:, -1] if bcs.top == "outflow" else phi[:, -1]
        return (px[1:] - px[:-1]) / g.dx, (py[:, 1:] - py[:, :-1]) / g.dy

    def divergence(self, u: np.ndarray | None = None, v: np.ndarray | None = None) -> np.ndarray:
        """``D_h u`` at the cell centres, shape ``(Nx, Ny)`` (Seibold sec. 5.1, eq. 22)."""
        uf = self._full_u(self.u if u is None else u)
        vf = self._full_v(self.v if v is None else v)
        return (uf[1:] - uf[:-1]) / self.grid.dx + (vf[:, 1:] - vf[:, :-1]) / self.grid.dy

    # ------------------------------------------------------------------ one projection substep
    def _substep(self, u0, v0, Fu, Fv, h, ua, va, t_a, t_new, theta, explicit, q_old=None):
        """Advance ``(u0, v0)`` by ``h`` and project. Returns ``(u, v, q)``: the velocity on
        the stored layouts and the pressure divided by ``rho`` at the cell centres.

        Predictor (Seibold eqs. 12-13 plus the force, an optional explicit viscous part and,
        for the incremental projection, the old pressure gradient):
            ``u* = u0 - h*adv(ua, va; data at t_a) + h*F/rho + h*c_e*L_h(u_e; data at t_e)
                   - h*G_h q_old``
        with ``explicit = (c_e, u_e, v_e, t_e)`` or ``None`` and ``q_old = p_old/rho`` or
        ``None`` (non-incremental).
        Implicit viscosity (eqs. 14-15 with weight ``theta``, ``c = theta*nu*h``), boundary
        data at ``t_new`` on the right-hand side (Seibold's ``Ubc``/``Vbc``):
            ``(I - c L_h,hom) u** = u* + c L_h,bc(t_new)``
        Dirichlet wall faces are set to the prescribed normal velocity at ``t_new``.
        Projection (eqs. 16-19, Chorin 1968): ``-L_h phi = -(1/h) D_h u**``,
        ``u = u** - h G_h phi``. Pressure, rotational form (Brown, Cortez & Minion 2001):
            ``q = q_old + phi - c L_h phi``   (``q_old = 0`` when non-incremental),
        with ``L_h phi = D_h G_h phi`` (exact, since ``D_h G_h = L_h``) and ``c = 0`` for
        explicit viscosity.
        """
        g = self.grid
        dx, dy = g.dx, g.dy
        ue, ve = self.extend_u(ua, t_a), self.extend_v(va, t_a)
        adv_u, adv_v = self._advection(ue, ve, self._gamma(ua, va, h))
        uf = self._full_u(u0) - h * adv_u
        vf = self._full_v(v0) - h * adv_v
        if Fu is not None:
            uf += (h / self.rho) * self._full_u(Fu)
        if Fv is not None:
            vf += (h / self.rho) * self._full_v(Fv)
        if explicit is not None:
            c_e, u_e, v_e, t_e = explicit
            if c_e != 0.0:
                uf += h * c_e * self._lap_u(self.extend_u(u_e, t_e))
                vf += h * c_e * self._lap_v(self.extend_v(v_e, t_e))
        if q_old is not None:
            Gx0, Gy0 = self._gradient(q_old)
            uf -= h * Gx0
            vf -= h * Gy0
        c = theta * self.nu * h if (theta > 0.0 and self.nu > 0.0) else 0.0
        if c > 0.0:
            lu_u, lu_v = self._helmholtz(c)
            bu = self._lap_u(self.extend_u(np.zeros(g.shape_u()), t_new))[self._iu]
            rhs = uf[self._iu] + c * bu
            uf[self._iu] = lu_u.solve(rhs.ravel()).reshape(rhs.shape)
            bv = self._lap_v(self.extend_v(np.zeros(g.shape_v()), t_new))[:, self._jv]
            rhs = vf[:, self._jv] + c * bv
            vf[:, self._jv] = lu_v.solve(rhs.ravel()).reshape(rhs.shape)
        self._impose_walls(uf, vf, t_new)
        div = (uf[1:] - uf[:-1]) / dx + (vf[:, 1:] - vf[:, :-1]) / dy
        if self._closed:
            self._check_compatibility(uf, vf)
        phi = self.poisson.solve(-div / h)
        Gx, Gy = self._gradient(phi)
        uf -= h * Gx
        vf -= h * Gy
        q = phi
        if c > 0.0:
            q = phi - c * ((Gx[1:] - Gx[:-1]) / dx + (Gy[:, 1:] - Gy[:, :-1]) / dy)
        if q_old is not None:
            q = q_old + q
        return self._store_u(uf), self._store_v(vf), q

    def _q_old(self) -> np.ndarray | None:
        """``p/rho`` of the current state for the incremental projection, else ``None``."""
        return self.p / self.rho if self.projection == "incremental" else None

    # ------------------------------------------------------------------ public time stepping
    def step(self, Fu: np.ndarray | None, Fv: np.ndarray | None, t: float) -> None:
        """One Seibold step ``u^n -> u^{n+1}`` with the force at ``t^n`` (the pm1996 scheme).

        Explicit advection of ``u^n`` (boundary data at ``t``), backward Euler viscosity
        with data at ``t + dt`` (or explicit with ``u^n``), then the projection. Sets
        ``p = rho*q`` (rotational form, ``_substep``) and ``self.t = t + dt``.
        """
        dt = self.dt
        if self.viscous == "implicit":
            theta, explicit = 1.0, None
        else:
            theta, explicit = 0.0, (self.nu, self.u, self.v, t)
        self.u, self.v, q = self._substep(self.u, self.v, Fu, Fv, dt, self.u, self.v, t,
                                          t + dt, theta, explicit, self._q_old())
        self.p = self.rho * q
        self.t = t + dt

    def half_step(self, Fu, Fv, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Midpoint scheme, first half (Peskin 2002 eqs. 7.8-7.9): returns ``(u_h, v_h)``
        at ``t + dt/2`` without touching the state. Forward Euler in the advection of
        ``u^n`` (data at ``t``), backward Euler viscosity over ``dt/2`` with data at
        ``t + dt/2`` (or explicit with ``u^n``), projection (incremental from ``self.p``
        when ``projection="incremental"``)."""
        h = 0.5 * self.dt
        if self.viscous == "implicit":
            theta, explicit = 1.0, None
        else:
            theta, explicit = 0.0, (self.nu, self.u, self.v, t)
        u_h, v_h, _ = self._substep(self.u, self.v, Fu, Fv, h, self.u, self.v, t, t + h,
                                    theta, explicit, self._q_old())
        return u_h, v_h

    def full_step(self, Fu, Fv, u_h: np.ndarray, v_h: np.ndarray, t: float) -> None:
        """Midpoint scheme, second half (Peskin 2002 eqs. 7.11-7.12): ``u^n -> u^{n+1}``.

        Advection evaluated with ``(u_h, v_h)`` and boundary data at ``t + dt/2``; viscous
        term Crank-Nicolson (explicit half with ``u^n`` at ``t``, implicit half with
        ``u^{n+1}`` at ``t + dt``), or fully explicit with ``u_h`` for ``viscous="explicit"``;
        force ``F``; projection. Sets ``p = rho*q`` (rotational form, a pressure at about
        ``t + dt/2``) and ``self.t = t + dt``.
        """
        dt = self.dt
        if self.viscous == "implicit":
            theta, explicit = 0.5, (0.5 * self.nu, self.u, self.v, t)
        else:
            theta, explicit = 0.0, (self.nu, u_h, v_h, t + 0.5 * dt)
        self.u, self.v, q = self._substep(self.u, self.v, Fu, Fv, dt, u_h, v_h,
                                          t + 0.5 * dt, t + dt, theta, explicit, self._q_old())
        self.p = self.rho * q
        self.t = t + dt

    # ------------------------------------------------------------------ diagnostics
    def vorticity(self) -> np.ndarray:
        """``dv/dx - du/dy`` at the cell centres, shape ``(Nx, Ny)``.

        Computed at the cell corners from the extended fields (boundary data at ``self.t``,
        Seibold sec. 7.1 "curl of the velocity field") and averaged to the centres.
        """
        g = self.grid
        ue, ve = self.extend_u(self.u, self.t), self.extend_v(self.v, self.t)
        w = (ve[1:, 1:-1] - ve[:-1, 1:-1]) / g.dx - (ue[1:-1, 1:] - ue[1:-1, :-1]) / g.dy
        return 0.25 * (w[:-1, :-1] + w[1:, :-1] + w[:-1, 1:] + w[1:, 1:])

    def cfl(self) -> float:
        """``dt * (max|u|/dx + max|v|/dy)``, an upper bound on the pointwise CFL number."""
        g = self.grid
        return float(self.dt * (np.max(np.abs(self.u)) / g.dx + np.max(np.abs(self.v)) / g.dy))

    def kinetic_energy(self) -> float:
        """``(rho/2) * sum (u^2 + v^2) dx dy`` with the trapezoidal rule over the faces
        (wall faces weighted 1/2), i.e. the cell average of ``u^2`` and ``v^2``."""
        g = self.grid
        su = np.sum(self._weights_u()[:, None] * self.u**2)
        sv = np.sum(self._weights_v()[None, :] * self.v**2)
        return float(0.5 * self.rho * (su + sv) * g.dx * g.dy)

    def mean_velocity(self) -> tuple[float, float]:
        """Box-averaged ``(u, v)`` with the same trapezoidal face weights."""
        g = self.grid
        area = g.Lx * g.Ly
        mu_ = np.sum(self._weights_u()[:, None] * self.u) * g.dx * g.dy / area
        mv_ = np.sum(self._weights_v()[None, :] * self.v) * g.dx * g.dy / area
        return float(mu_), float(mv_)
