"""Doubly periodic FFT fluid back end on collocated nodes.

This is the fluid solver of Peskin & McQueen 1996 section 14 (the first-order scheme,
``PeriodicFFTSolver.step``) and of Peskin 2002 section 7 (the second-order midpoint scheme,
``half_step`` followed by ``full_step``), on a doubly periodic box with Nx by Ny collocated
nodes x_i = i h, y_j = j h, h = dx = dy. We discretise

    rho (u_t + u . grad u) + grad p = mu Lap u + F,    div u = 0

with the centred difference D0 for the gradient and the divergence and the tight five-point
Laplacian L_h for the viscous term. Once the nonlinear term and the force are moved into a
known vector w, every time step is a constant-coefficient linear system that is diagonal in
Fourier space (eqs. 14.48-14.52), so we solve it exactly with one FFT and one inverse FFT per
velocity component.

The ``step`` scheme is written to reproduce the 2008 MATLAB code (makew.m, fluid_solve.m and
the symbol arrays in moving_plate.m): same upwind rule and operation order, same symbols,
same four degenerate modes zeroed in the pressure, same handling of the k = 0 mode. Given
the same u, v and F it returns the same u, v, p to round-off.

Conventions (docs/numerics.md): arrays (Nx, Ny) indexed [i, j] with i along x, float64, SI
units. The fields are real, so the solvers use the real-to-complex transforms
``scipy.fft.rfft2`` and ``irfft2``, which keep only the half spectrum k2 = 0..Ny/2 (the
other half is its complex conjugate). The forward transform is unnormalised and the
1/(Nx Ny) sits on the inverse. The 1/N^3 of eq. 14.43 is on the forward transform instead,
which changes nothing because eqs. 14.51-14.52 are homogeneous in the transforms. The MATLAB
code takes the real part of a full complex inverse transform; for the spectrum of a real
field that is the same thing, and the two agree to round-off.

For speed, the spatial differences in the time steps take their neighbours from a copy of
the field padded with one periodic ghost layer (the same values ``np.roll`` gives, without a
new array for each shift), and the solver reuses its scratch arrays from step to step. The
module-level operators ``d0x``, ``d0y``, ``divergence`` and ``laplacian`` keep the plain
``np.roll`` form for reading and for checks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.fft as sfft

from pyibm.grid import FieldLayout, Grid, layout_collocated


# ----------------------------------------------------------------------------- difference operators
def d0x(phi: np.ndarray, h: float) -> np.ndarray:
    """Centred difference in x, (phi[i+1, j] - phi[i-1, j])/(2h), periodic.

    Peskin & McQueen 1996 eq. 14.31; Peskin 2002 eq. 4.7.
    """
    return (np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)) / (2.0 * h)


def d0y(phi: np.ndarray, h: float) -> np.ndarray:
    """Centred difference in y, (phi[i, j+1] - phi[i, j-1])/(2h), periodic (eq. 14.31)."""
    return (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / (2.0 * h)


def divergence(u: np.ndarray, v: np.ndarray, h: float) -> np.ndarray:
    """Discrete divergence D0 . u = D0_x u + D0_y v (Peskin & McQueen 1996 eq. 14.29)."""
    return d0x(u, h) + d0y(v, h)


def laplacian(phi: np.ndarray, h: float) -> np.ndarray:
    """Tight five-point Laplacian sum_alpha D_alpha^+ D_alpha^- phi, periodic.

    Peskin & McQueen 1996 eqs. 14.32-14.33; Peskin 2002 eq. 4.8. Not D0 . D0, whose stencil
    is four mesh widths wide and has a checkerboard null space.
    """
    return (np.roll(phi, -1, axis=0) + np.roll(phi, 1, axis=0)
            + np.roll(phi, -1, axis=1) + np.roll(phi, 1, axis=1) - 4.0 * phi) / (h * h)


def _pad_into(P: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Fill P, shape (2, Nx + 2, Ny + 2), with (u, v) and one periodic ghost layer all round.

    Afterwards P[:, 1:-1, 1:-1] is (u, v), and the neighbours i + 1, i - 1, j + 1, j - 1 of
    every node are the slices P[:, 2:, 1:-1], P[:, :-2, 1:-1], P[:, 1:-1, 2:] and
    P[:, 1:-1, :-2]: the same values np.roll gives, without a new array for each shift.
    """
    nx, ny = P.shape[1] - 2, P.shape[2] - 2
    P[0, 1:-1, 1:-1] = u
    P[1, 1:-1, 1:-1] = v
    P[:, 0, 1:-1] = P[:, nx, 1:-1]
    P[:, nx + 1, 1:-1] = P[:, 1, 1:-1]
    P[:, :, 0] = P[:, :, ny]
    P[:, :, ny + 1] = P[:, :, 1]
    return P


def _laplacian_into(P: np.ndarray, h: float, out: np.ndarray, tmp: np.ndarray) -> np.ndarray:
    """L_h of both components of a padded stack into ``out`` (eq. 14.32).

    The same sums in the same order as ``laplacian``, so the result is the same bit for bit.
    ``tmp`` is a scratch array of the shape of ``out``.
    """
    np.add(P[:, 2:, 1:-1], P[:, :-2, 1:-1], out=out)
    out += P[:, 1:-1, 2:]
    out += P[:, 1:-1, :-2]
    np.multiply(P[:, 1:-1, 1:-1], 4.0, out=tmp)
    out -= tmp
    out /= h * h
    return out


# ----------------------------------------------------------------------------- Fourier symbols
@dataclass(frozen=True)
class PeriodicSymbols:
    """Fourier symbols of the periodic difference operators on an (Nx, Ny) node array.

    For the full ``np.fft.fft2`` spectrum the wavenumber index arrays are k1[i, j] = i and
    k2[i, j] = j. With s1 = sin(2 pi k1/Nx) and s2 = sin(2 pi k2/Ny) (Peskin & McQueen 1996
    eqs. 14.46-14.47):

    - centred difference  D0_x <-> i s1/h,   D0_y <-> i s2/h
    - tight Laplacian     L_h  <-> -(4/h^2) (sin^2(pi k1/Nx) + sin^2(pi k2/Ny))

    The sine in the Laplacian symbol is squared: (e^{i theta} - 1)(1 - e^{-i theta})/h^2 =
    -(4/h^2) sin^2(theta/2), as in eq. 14.50.

    ``degenerate`` marks the four modes k1 in {0, Nx/2}, k2 in {0, Ny/2} where s1 = s2 = 0:
    the box mean and the three checkerboards (p. 323, eq. 14.53). It is built from the index
    arrays, never from a floating-point test on the sines, because sin(pi) evaluates to
    1.2e-16 and not to 0. ``inv_s_sq`` is 1/(s1^2 + s2^2) with 0 on those four modes.
    All arrays are read-only.
    """

    Nx: int
    Ny: int
    h: float
    k1: np.ndarray
    k2: np.ndarray
    s1: np.ndarray
    s2: np.ndarray
    s_sq: np.ndarray
    inv_s_sq: np.ndarray
    degenerate: np.ndarray
    sin2_half: np.ndarray
    lap: np.ndarray


def periodic_symbols(Nx: int, Ny: int, h: float = 1.0) -> PeriodicSymbols:
    """Build the ``PeriodicSymbols`` for an (Nx, Ny) node array with spacing h.

    Nx and Ny must be even: the degenerate set assumes a Nyquist index at N/2.
    """
    if Nx % 2 or Ny % 2 or Nx < 2 or Ny < 2:
        raise ValueError(f"periodic symbols need even Nx, Ny >= 2, got {Nx}, {Ny}")
    k1, k2 = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing="ij")
    # Same expressions as the MATLAB code: sin(2*pi*(i-1)/M) with k = i-1.
    s1 = np.sin(2 * np.pi * k1 / Nx)
    s2 = np.sin(2 * np.pi * k2 / Ny)
    s_sq = s1 * s1 + s2 * s2
    degenerate = ((k1 == 0) | (k1 == Nx // 2)) & ((k2 == 0) | (k2 == Ny // 2))
    inv_s_sq = np.zeros_like(s_sq)
    np.divide(1.0, s_sq, out=inv_s_sq, where=~degenerate)
    sin2_half = np.sin(np.pi * k1 / Nx) ** 2 + np.sin(np.pi * k2 / Ny) ** 2
    lap = -(4.0 / (h * h)) * sin2_half
    arrays = (k1, k2, s1, s2, s_sq, inv_s_sq, degenerate, sin2_half, lap)
    for a in arrays:
        a.flags.writeable = False
    return PeriodicSymbols(int(Nx), int(Ny), float(h), *arrays)


def _project_hat(wu_hat: np.ndarray, wv_hat: np.ndarray, sym: PeriodicSymbols
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Projection in Fourier space; also returns q = (s . w_hat)/|s|^2, which gives the pressure.

    Eliminating p_hat from eqs. 14.48-14.49: u_hat = (w_hat - s q)/A with
    q = (s1 wu_hat + s2 wv_hat)/(s1^2 + s2^2) and q = 0 on the degenerate set (eq. 14.53).
    The MATLAB code writes the same thing with a = s1^2/|s|^2, b = s1 s2/|s|^2,
    c = s2^2/|s|^2 (moving_plate.m); the two differ only in rounding.
    """
    q = (sym.s1 * wu_hat + sym.s2 * wv_hat) * sym.inv_s_sq
    return wu_hat - sym.s1 * q, wv_hat - sym.s2 * q, q


# ----------------------------------------------------------------------------- the solver
class PeriodicFFTSolver:
    """Navier-Stokes on a doubly periodic box of collocated nodes, solved with FFTs.

    Parameters
    ----------
    grid : Grid
        Doubly periodic with dx == dy (``Grid.periodic_box``).
    rho, mu : float
        Density [kg/m^2] and dynamic viscosity [N s/m^2].
    dt : float
        Time step [s]. The Fourier symbols are built for this dt once, here.
    remove_mean_force : bool
        Hold the box-mean velocity (the k = 0 Fourier mode) fixed. The k = 0 mode passes
        through the projection untouched (eq. 14.53), so a net force on the fluid, say from a
        body tethered to target points, accelerates the whole box without end. With this
        option every solve sets the k = 0 coefficient of w to that of u^n. That is the same
        as subtracting the box mean of every term that changes u in the step: the force, the
        advection and the viscous term. In the pm1996 scheme this matters for the advection
        too, because upwinding is not conservative and its box mean is not zero. In the
        midpoint scheme the skew advection of a discretely divergence-free field and the
        viscous term have zero box mean up to round-off. Either way the force drives only
        the flow relative to the box, and the box-mean velocity keeps its starting value to
        round-off. Off by default, as in the MATLAB code.

    Attributes
    ----------
    u, v, p : (Nx, Ny) float64 arrays on the nodes x_i = i h, y_j = j h, zero at start.
    layout_u, layout_v : both ``layout_collocated(grid)``.
    symbols : ``PeriodicSymbols`` shared by both schemes (full spectrum).
    inv_A_pm1996, inv_A_mid : 1/A(k) for the two schemes (eq. 14.50 with dt, and with dt/2),
        full spectrum.

    Two time schemes share the state:

    - ``step`` is the first-order scheme of Peskin & McQueen 1996 section 14.4: upwind
      advection, backward Euler for the viscous term and the pressure. It reproduces the 2008
      MATLAB code (makew.m, fluid_solve.m) to round-off.
    - ``half_step`` + ``full_step`` is the second-order midpoint scheme of Peskin 2002
      section 7: skew-symmetric advection, backward Euler over dt/2 for the half step, then
      Crank-Nicolson over dt for the full step.

    Both ignore ``t``: there is no boundary data on a periodic box. The argument is there for
    the ``FluidSolver`` interface, and the driver decides which time level F belongs to.

    The solver keeps its scratch arrays between calls, so that a step does not allocate its
    large work arrays afresh. Because of this, one solver must not be stepped from two
    threads at once. Every array it hands out (u, v, p, u_h, v_h and the advection terms) is
    a new array, never one of the scratch arrays.
    """

    def __init__(self, grid: Grid, rho: float, mu: float, dt: float,
                 remove_mean_force: bool = False) -> None:
        if not (grid.periodic_x and grid.periodic_y):
            raise ValueError("PeriodicFFTSolver needs a doubly periodic grid (Grid.periodic_box)")
        if not np.isclose(grid.dx, grid.dy, rtol=1e-12, atol=0.0):
            raise ValueError(f"PeriodicFFTSolver needs dx == dy, got dx={grid.dx}, dy={grid.dy}")
        if rho <= 0 or mu < 0 or dt <= 0:
            raise ValueError(f"need rho > 0, mu >= 0, dt > 0; got rho={rho}, mu={mu}, dt={dt}")
        self.grid = grid
        self.rho = float(rho)
        self.mu = float(mu)
        self.dt = float(dt)
        self.h = float(grid.dx)
        self.remove_mean_force = bool(remove_mean_force)
        self.layout_u: FieldLayout = layout_collocated(grid)
        self.layout_v: FieldLayout = layout_collocated(grid)
        shape = grid.shape_p()
        self.u = np.zeros(shape)
        self.v = np.zeros(shape)
        self.p = np.zeros(shape)

        # Symbols, built once. A(k) = 1 + (4 nu dt/h^2)(sin^2(pi k1/Nx) + sin^2(pi k2/Ny)) is
        # eq. 14.50 written with nu = mu/rho, exactly as moving_plate.m builds d = 1/A; we keep
        # 1/A and multiply, as the MATLAB code does. The midpoint scheme uses the same A with
        # dt/2: backward Euler over dt/2 for the half step, and Crank-Nicolson over dt for the
        # full step has the same implicit operator (I - (mu dt/(2 rho)) L_h).
        nu = self.mu / self.rho
        h = self.h
        self.symbols = periodic_symbols(grid.Nx, grid.Ny, h)
        A_pm1996 = 1.0 + (4.0 * nu * self.dt / (h * h)) * self.symbols.sin2_half
        A_mid = 1.0 + (2.0 * nu * self.dt / (h * h)) * self.symbols.sin2_half
        self.inv_A_pm1996 = 1.0 / A_pm1996
        self.inv_A_mid = 1.0 / A_mid
        self.inv_A_pm1996.flags.writeable = False
        self.inv_A_mid.flags.writeable = False

        # The same symbols on the half spectrum k2 = 0..Ny/2 that rfft2 returns, contiguous.
        nr = grid.Ny // 2 + 1
        self._s1_r = np.ascontiguousarray(self.symbols.s1[:, :nr])
        self._s2_r = np.ascontiguousarray(self.symbols.s2[:, :nr])
        self._inv_s_sq_r = np.ascontiguousarray(self.symbols.inv_s_sq[:, :nr])
        self._inv_A_pm1996_r = np.ascontiguousarray(self.inv_A_pm1996[:, :nr])
        self._inv_A_mid_r = np.ascontiguousarray(self.inv_A_mid[:, :nr])
        self._scratch: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------ projection
    @staticmethod
    def project(wu_hat: np.ndarray, wv_hat: np.ndarray, symbols: PeriodicSymbols | None = None
                ) -> tuple[np.ndarray, np.ndarray]:
        """Remove the discrete-gradient part of a vector field given in Fourier space.

        Peskin & McQueen 1996 eqs. 14.48-14.49 and 14.52-14.53: with s = (s1, s2),
        q = (s . w_hat)/|s|^2 and q = 0 on the four modes where s = 0, returns
        (wu_hat - s1 q, wv_hat - s2 q). Afterwards the discrete divergence D0 . u is zero
        exactly (i s . u_hat/h = 0 for every k), the k = 0 mode (the box mean) passes through
        untouched, and so do the three checkerboard modes, which D0 cannot see. A field that
        is already discretely divergence-free (for example a blob-projection force) is
        returned unchanged.

        The inputs are full ``np.fft.fft2`` spectra of shape (Nx, Ny). ``symbols`` defaults
        to ``periodic_symbols(*wu_hat.shape)``; the solver passes its precomputed set. The
        spacing h cancels out of the projection, so it is not needed.
        """
        if symbols is None:
            symbols = periodic_symbols(wu_hat.shape[0], wu_hat.shape[1])
        pu, pv, _ = _project_hat(wu_hat, wv_hat, symbols)
        return pu, pv

    def _solve(self, w: np.ndarray, inv_A_r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Eqs. 14.48-14.53 for w of shape (2, Nx, Ny), in Fourier space on the half spectrum.

        Returns (u_hat, q): u_hat = (w_hat - s q)/A, a new (2, Nx, Ny/2 + 1) array, and
        q = (s . w_hat)/|s|^2 (zero on the four degenerate modes), a scratch array that is
        valid until the next call. This is ``_project_hat`` followed by the multiplication
        by 1/A, in the same order. With ``remove_mean_force`` the k = 0 coefficient of w is
        first set to that of the current state (u, v), summed in float64, so the box-mean
        velocity stays put.
        """
        w_hat = sfft.rfft2(w)
        if self.remove_mean_force:
            # summed in float64 whatever the dtype of the state, like the rest of the step
            w_hat[0, 0, 0] = np.sum(self.u, dtype=np.float64)
            w_hat[1, 0, 0] = np.sum(self.v, dtype=np.float64)
        q = self._work("q", w_hat.shape[1:], np.complex128)
        tmp = self._work("q_tmp", w_hat.shape[1:], np.complex128)
        np.multiply(self._s1_r, w_hat[0], out=q)
        np.multiply(self._s2_r, w_hat[1], out=tmp)
        q += tmp
        q *= self._inv_s_sq_r
        np.multiply(self._s1_r, q, out=tmp)
        w_hat[0] -= tmp
        np.multiply(self._s2_r, q, out=tmp)
        w_hat[1] -= tmp
        w_hat *= inv_A_r
        return w_hat, q

    def _inverse(self, a_hat: np.ndarray) -> np.ndarray:
        """Real inverse transform of a half spectrum (or a stack of them) to (..., Nx, Ny)."""
        return sfft.irfft2(a_hat, s=self.grid.shape_p())

    def _pressure(self, q: np.ndarray) -> np.ndarray:
        """Eq. 14.52: p_hat = -i (rho h/dt) q, zero on the four degenerate modes (q is zero there)."""
        return self._inverse(-1j * (self.rho * self.h / self.dt) * q)

    # ------------------------------------------------------------------ advection terms
    def upwind_advection(self, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """First-order upwind advection (u D^+-_x + v D^+-_y) applied to u and to v.

        Peskin & McQueen 1996 eq. 14.34: u_alpha D_alpha^+- = u_alpha D_alpha^+ where
        u_alpha < 0 (forward difference) and u_alpha D_alpha^- where u_alpha >= 0 (backward
        difference), the switch taken on the advecting component at the point itself. Same
        rule and same operation order as makew.m, including the ``(1/h) * (difference)``
        factor. Stable under the CFL condition dt (|u| + |v|) <= h (eq. 14.38).
        """
        A = self._upwind(self._padded(u, v), np.empty((2,) + self.grid.shape_p()))
        return A[0], A[1]

    def _upwind(self, P: np.ndarray, out: np.ndarray) -> np.ndarray:
        """``upwind_advection`` of the padded stack P into ``out``, shape (2, Nx, Ny).

        Dx[:, m] = P[:, m + 1] - P[:, m] holds every one-sided difference in x once: the
        forward difference at node i is Dx[:, i + 1] and the backward one is Dx[:, i].
        The same in y. The result is the same, bit for bit, as the np.roll form.
        """
        nx, ny = self.grid.shape_p()
        inv_h = 1.0 / self.h
        u, v = P[0, 1:-1, 1:-1], P[1, 1:-1, 1:-1]
        Dx = self._work("Dx", (2, nx + 1, ny))
        Dy = self._work("Dy", (2, nx, ny + 1))
        np.subtract(P[:, 1:, 1:-1], P[:, :-1, 1:-1], out=Dx)
        np.subtract(P[:, 1:-1, 1:], P[:, 1:-1, :-1], out=Dy)
        # D^+ where the advecting component is negative, D^- elsewhere. np.where makes a new
        # array, but it is much faster than a masked copy into a scratch array.
        gx = np.where(u < 0, Dx[:, 1:], Dx[:, :-1])
        gx *= inv_h
        gy = np.where(v < 0, Dy[:, :, 1:], Dy[:, :, :-1])
        gy *= inv_h
        np.multiply(gx, u, out=out)
        gy *= v
        out += gy
        return out

    def skew_advection(self, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Skew-symmetric advection S_h(u) applied to u and to v (Peskin 2002 eq. 4.10).

        S_h(u) phi = 1/2 (u D0_x phi + v D0_y phi) + 1/2 (D0_x(u phi) + D0_y(v phi)).
        S_h(u) is skew-adjoint for every u (eq. 4.14). Together with D0 . u = 0 this makes the
        semi-discrete scheme conserve energy up to viscous dissipation (eq. 5.15). The fully
        discrete midpoint scheme (eqs. 7.8-7.12) evaluates S_h at u_h but pairs it with
        (u^n + u^{n+1})/2, so with mu = 0 the energy is conserved only up to a small
        time-stepping error (O(dt^2) per unit time or better), and it can drift slightly,
        usually upward. When D0 . u = 0 the sum of S_h(u) phi over the box vanishes, so the
        box momentum is conserved too (eq. 5.12).
        """
        S = self._skew(self._padded(u, v), np.empty((2,) + self.grid.shape_p()))
        return S[0], S[1]

    def _skew(self, P: np.ndarray, out: np.ndarray) -> np.ndarray:
        """``skew_advection`` of the padded stack P into ``out``, shape (2, Nx, Ny).

        With the undivided centred difference d_x phi = phi[i+1] - phi[i-1] = 2h D0_x phi,
        S_h(u)(u, v) = (1/(4h)) [(u d_x + v d_y)(u, v) + d_x(uu, uv) + d_y(uv, vv)]. The
        products uu, uv, vv are formed on the padded arrays, so their ghost layers are
        periodic copies too.
        """
        u, v = P[0, 1:-1, 1:-1], P[1, 1:-1, 1:-1]
        Q = self._work("Q", (3,) + P.shape[1:])
        t = self._work("t", out.shape)
        c = self._work("c", out.shape)
        np.multiply(P[0], P[0], out=Q[0])
        np.multiply(P[0], P[1], out=Q[1])
        np.multiply(P[1], P[1], out=Q[2])
        S = out
        np.subtract(P[:, 2:, 1:-1], P[:, :-2, 1:-1], out=S)    # d_x (u, v)
        S *= u
        np.subtract(P[:, 1:-1, 2:], P[:, 1:-1, :-2], out=t)    # d_y (u, v)
        t *= v
        S += t                                                 # (u d_x + v d_y)(u, v)
        np.subtract(Q[0:2, 2:, 1:-1], Q[0:2, :-2, 1:-1], out=c)  # d_x (uu, uv)
        np.subtract(Q[1:3, 1:-1, 2:], Q[1:3, 1:-1, :-2], out=t)  # d_y (uv, vv)
        c += t
        S += c
        S *= 0.25 / self.h
        return S

    # ------------------------------------------------------------------ time schemes
    def step(self, Fu: np.ndarray, Fv: np.ndarray, t: float = 0.0) -> None:
        """Peskin & McQueen 1996 first-order step: u^n -> u^{n+1}, p^{n+1}, with F at t^n.

        Eq. 14.42:   w = u^n - dt (u D^+-_x u + v D^+-_y u) + (dt/rho) F  (upwind, eq. 14.34).
                     The advection term carries dt and the force dt/rho (eq. 14.28 times
                     dt/rho), as in makew.m.
        Eq. 14.51:   u_hat^{n+1} = (w_hat - s q)/A,  A = 1 + (4 mu dt/(rho h^2))
                     (sin^2(pi k1/Nx) + sin^2(pi k2/Ny))  (eq. 14.50);
        Eq. 14.52:   p_hat^{n+1} = -i (rho h/dt) q,  q = (s . w_hat)/|s|^2, q = 0 on the four
                     modes with s = 0 (eq. 14.53; there u_hat = w_hat/A, including k = 0).
        Projecting first and dividing by A second is exact because both are diagonal in k.
        This is fluid_solve.m, with makew.m for w; u, v, p are the real inverse transforms.
        """
        Fu, Fv = self._check_forces(Fu, Fv)
        dt = self.dt
        P = self._padded(self.u, self.v)
        w = self._upwind(P, self._work("w", (2,) + self.grid.shape_p()))
        w *= dt
        np.subtract(P[:, 1:-1, 1:-1], w, out=w)                # u^n - dt (u D^+- u)
        self._add_force(w, Fu, Fv, dt / self.rho)              # + (dt/rho) F
        u_hat, q = self._solve(w, self._inv_A_pm1996_r)
        uv = self._inverse(u_hat)
        self.u, self.v = uv[0], uv[1]
        self.p = self._pressure(q)

    def half_step(self, Fu: np.ndarray, Fv: np.ndarray, t: float = 0.0
                  ) -> tuple[np.ndarray, np.ndarray]:
        """Peskin 2002 eqs. 7.8-7.9: the preliminary substep, returning (u_h, v_h) at t + dt/2.

        rho [(u_h - u^n)/(dt/2) + S_h(u^n) u^n] + D0 p~ = mu L_h u_h + F,  D0 . u_h = 0:
        forward Euler in the nonlinear term, backward Euler otherwise, over dt/2. So
        w = u^n - (dt/2) S_h(u^n) u^n + (dt/(2 rho)) F,
        u_h_hat = (w_hat - s q)/A_h with A_h = 1 + (2 mu dt/(rho h^2))(sin^2 + sin^2).
        The state (u, v, p) is not touched; the half-step pressure p~ is not kept.
        """
        Fu, Fv = self._check_forces(Fu, Fv)
        half_dt = 0.5 * self.dt
        P = self._padded(self.u, self.v)
        w = self._skew(P, self._work("w", (2,) + self.grid.shape_p()))
        w *= half_dt
        np.subtract(P[:, 1:-1, 1:-1], w, out=w)                # u^n - (dt/2) S_h(u^n) u^n
        self._add_force(w, Fu, Fv, half_dt / self.rho)         # + (dt/(2 rho)) F
        u_hat, _ = self._solve(w, self._inv_A_mid_r)
        uv = self._inverse(u_hat)
        return uv[0], uv[1]

    def full_step(self, Fu: np.ndarray, Fv: np.ndarray, u_h: np.ndarray, v_h: np.ndarray,
                  t: float = 0.0) -> None:
        """Peskin 2002 eqs. 7.11-7.12: the final substep, u^n -> u^{n+1} and p^{n+1/2}.

        rho [(u^{n+1} - u^n)/dt + S_h(u_h) u_h] + D0 p = mu L_h (u^n + u^{n+1})/2 + F,
        D0 . u^{n+1} = 0, with the same F as the half step (f^{n+1/2}). Crank-Nicolson in the
        viscous term, so
        w = u^n - dt S_h(u_h) u_h + (dt/rho) F + (mu dt/(2 rho)) L_h u^n,
        u_hat^{n+1} = (w_hat - s q)/A_h with the same A_h as the half step (CN over dt has the
        same implicit operator as backward Euler over dt/2), and the pressure as in
        eq. 14.52 with dt: p_hat = -i (rho h/dt) q, zero on the four degenerate modes.
        """
        Fu, Fv = self._check_forces(Fu, Fv)
        u_h = self._check_field(u_h, "u_h")
        v_h = self._check_field(v_h, "v_h")
        dt = self.dt
        shape2 = (2,) + self.grid.shape_p()
        w = self._skew(self._padded(u_h, v_h), self._work("w", shape2))
        w *= dt
        P = self._padded(self.u, self.v)                       # the pad now holds u^n
        np.subtract(P[:, 1:-1, 1:-1], w, out=w)                # u^n - dt S_h(u_h) u_h
        self._add_force(w, Fu, Fv, dt / self.rho)              # + (dt/rho) F
        L = _laplacian_into(P, self.h, self._work("c", shape2), self._work("t", shape2))
        L *= 0.5 * self.mu * dt / self.rho
        w += L                                                 # + (mu dt/(2 rho)) L_h u^n
        u_hat, q = self._solve(w, self._inv_A_mid_r)
        uv = self._inverse(u_hat)
        self.u, self.v = uv[0], uv[1]
        self.p = self._pressure(q)

    # ------------------------------------------------------------------ diagnostics
    def vorticity(self) -> np.ndarray:
        """omega = D0_x v - D0_y u on the nodes, shape (Nx, Ny) (centred differences, eq. 14.31)."""
        return d0x(self.v, self.h) - d0y(self.u, self.h)

    def cfl(self) -> float:
        """dt * max(|u|/dx + |v|/dy). The upwind scheme needs this <= 1 (eq. 14.38)."""
        return float(self.dt * np.max(np.abs(self.u) / self.grid.dx + np.abs(self.v) / self.grid.dy))

    def kinetic_energy(self) -> float:
        """(rho/2) sum (u^2 + v^2) h^2 over the box [J in 2D]."""
        return float(0.5 * self.rho * np.sum(self.u * self.u + self.v * self.v) * self.h * self.h)

    def mean_velocity(self) -> tuple[float, float]:
        """Box-mean velocity (the k = 0 mode), (mean u, mean v)."""
        return float(np.mean(self.u)), float(np.mean(self.v))

    # ------------------------------------------------------------------ helpers
    def _work(self, name: str, shape: tuple[int, ...], dtype=np.float64) -> np.ndarray:
        """The scratch array ``name``, made on first use and reused after that."""
        a = self._scratch.get(name)
        if a is None:
            a = self._scratch[name] = np.empty(shape, dtype=dtype)
        return a

    def _padded(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """(u, v) with one periodic ghost layer, in the scratch array "P" (see ``_pad_into``)."""
        nx, ny = self.grid.shape_p()
        P = self._work("P", (2, nx + 2, ny + 2))
        return _pad_into(P, self._check_field(u, "u"), self._check_field(v, "v"))

    def _add_force(self, w: np.ndarray, Fu: np.ndarray, Fv: np.ndarray, c: float) -> None:
        """w += c F, component by component, through the scratch array "t"."""
        t = self._work("t", w.shape)
        np.multiply(Fu, c, out=t[0])
        np.multiply(Fv, c, out=t[1])
        w += t

    def _check_field(self, a: np.ndarray, name: str) -> np.ndarray:
        a = np.asarray(a, dtype=float)
        if a.shape != self.grid.shape_p():
            raise ValueError(f"{name} must have shape {self.grid.shape_p()}, got {a.shape}")
        return a

    def _check_forces(self, Fu: np.ndarray, Fv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Validate the force arrays (with ``remove_mean_force`` the mean goes in ``_solve``)."""
        return self._check_field(Fu, "Fu"), self._check_field(Fv, "Fv")


__all__ = ["PeriodicFFTSolver", "PeriodicSymbols", "periodic_symbols",
           "d0x", "d0y", "divergence", "laplacian"]
