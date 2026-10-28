"""Poisson solvers for the MAC projection back end.

We solve ``-L_h phi = rhs`` for a cell-centred field ``phi`` of shape ``(Nx, Ny)``, where
``L_h`` is the standard 5-point Laplacian on the uniform MAC grid (Seibold 2008 eq. 20).
Each side of the box carries one of three conditions:

- ``"neumann"``   : d(phi)/dn = 0 across the wall, ghost cell ``phi_ghost = phi_interior``
                    (Seibold eq. 36). This is what every Dirichlet-velocity wall needs.
- ``"dirichlet"`` : phi = 0 on the wall, which lies half a cell outside the last centre, so
                    ``phi_ghost = -phi_interior`` (Seibold's "Dirichlet between two points").
                    This is what an outflow side needs.
- ``"periodic"``  : wrap-around; both sides of that direction must be periodic.

Two implementations: ``PoissonSolver`` builds the sparse matrix with ``scipy.sparse.kron``
of 1-D tridiagonals (Seibold eq. 37, sec. 7.3) and factors it once with ``splu``;
``DCTPoissonSolver`` is the fast path for the all-Neumann box, where the operator is
diagonalised exactly by the type-2 discrete cosine transform. ``make_poisson_solver`` picks
the DCT when every side is Neumann and the sparse solver otherwise.

When no side is Dirichlet the operator is singular with the constant vector in its null
space. Both solvers then drop the mean of the right-hand side (the part no solution can
match) and return the zero-mean solution.

Reference: B. Seibold, "A compact and fast Matlab code solving the incompressible
Navier-Stokes equations on rectangular domains", course notes for MIT 18.086 (2008),
http://www-math.mit.edu/18086.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.fft import dctn, idctn
from scipy.sparse.linalg import splu

from pyibm.grid import Grid

SIDES = ("left", "right", "bottom", "top")
SIDE_KINDS = ("neumann", "dirichlet", "periodic")

# Diagonal end entry of Seibold's K1 matrix (eq. 37, table on p. 13) for each 1-D end
# condition. "dirichlet_point" is a Dirichlet value sitting ON the end point (that point is
# known and removed from the unknowns), "dirichlet_mid" is a Dirichlet value half a spacing
# beyond the last unknown (ghost = -last).
_A11 = {"neumann": 1.0, "dirichlet_point": 2.0, "dirichlet_mid": 3.0}


def k1(n: int, h: float, lo: str, hi: str) -> sp.csr_matrix:
    """1-D matrix approximating ``-d^2/dx^2`` on ``n`` points with spacing ``h``.

    Seibold 2008 eq. 37: tridiagonal with ``2`` on the diagonal and ``-1`` beside it, the
    end diagonals replaced by ``a11`` = 1 (Neumann), 2 (Dirichlet on the point) or 3
    (Dirichlet between points), all divided by ``h^2``. ``lo``/``hi`` name the condition at
    the low/high end: ``"neumann"``, ``"dirichlet_point"``, ``"dirichlet_mid"`` or
    ``"periodic"`` (then both must be periodic and the matrix is circulant).
    """
    if n < 2:
        raise ValueError(f"k1 needs at least 2 points, got n={n}")
    if lo == "periodic" or hi == "periodic":
        if lo != hi:
            raise ValueError("periodic must be set on both ends of a direction")
        main = np.full(n, 2.0)
        off = -np.ones(n - 1)
        K = sp.diags([off, main, off], [-1, 0, 1], shape=(n, n), format="lil")
        K[0, n - 1] += -1.0
        K[n - 1, 0] += -1.0
        return (K / h**2).tocsr()
    for name in (lo, hi):
        if name not in _A11:
            raise ValueError(f"unknown 1-D end condition {name!r}")
    main = np.full(n, 2.0)
    main[0] = _A11[lo]
    main[-1] = _A11[hi]
    off = -np.ones(n - 1)
    K = sp.diags([off, main, off], [-1, 0, 1], shape=(n, n), format="csr")
    return (K / h**2).tocsr()


def laplacian_matrix(nx: int, ny: int, dx: float, dy: float,
                     xlo: str, xhi: str, ylo: str, yhi: str) -> sp.csc_matrix:
    """Sparse matrix of ``-L_h`` acting on an ``(nx, ny)`` field raveled in C order.

    With index ``i*ny + j`` (``j`` fastest) the 2-D operator is
    ``kron(Kx, I_ny) + kron(I_nx, Ky)`` (Seibold sec. 7.3; his MATLAB code ravels the other
    way round, so the kron factors are swapped relative to his listing).
    """
    Kx = k1(nx, dx, xlo, xhi)
    Ky = k1(ny, dy, ylo, yhi)
    A = sp.kron(Kx, sp.identity(ny, format="csr")) + sp.kron(sp.identity(nx, format="csr"), Ky)
    return A.tocsc()


def _check_sides(sides: dict) -> dict:
    out = {}
    for s in SIDES:
        if s not in sides:
            raise ValueError(f"missing boundary side {s!r}; need {SIDES}")
        if sides[s] not in SIDE_KINDS:
            raise ValueError(f"side {s!r}: unknown condition {sides[s]!r}; use one of {SIDE_KINDS}")
        out[s] = sides[s]
    if (out["left"] == "periodic") != (out["right"] == "periodic"):
        raise ValueError("periodic must be set on both left and right")
    if (out["bottom"] == "periodic") != (out["top"] == "periodic"):
        raise ValueError("periodic must be set on both bottom and top")
    return out


class PoissonSolver:
    """Sparse direct solver for ``-L_h phi = rhs`` on cell centres with per-side conditions.

    The matrix ``A`` is built once with ``laplacian_matrix`` and factored with ``splu``.
    When no side is Dirichlet, ``A`` has corank 1 (null space = constants) and the problem
    is solvable only for ``rhs`` of zero mean. Two regularisations:

    - ``regularisation="bordered"`` (default): solve the bordered system
      ``[[A, e], [e^T, 0]] [phi; lam] = [rhs; 0]`` with ``e`` the vector of ones. This is
      the Lagrange-multiplier form of "``A phi = rhs`` subject to mean(phi) = 0"; the
      multiplier ``lam = mean(rhs)`` absorbs whatever incompatible part the right-hand side
      has, so the result is the least-squares solution with the residual spread uniformly.
    - ``regularisation="pin"``: Seibold sec. 7.3, add a positive number to one diagonal
      entry. ``B^{-1} b`` then solves ``A phi = b`` for every compatible ``b`` (his eq. 38
      and the proof below it hold for any positive addition); we add
      ``2/dx^2 + 2/dy^2``, the size of a typical diagonal entry, rather than literally 1
      so the regularised matrix is as well conditioned as a Dirichlet problem, and
      subtract the mean of ``phi`` afterwards.

    Both give the same zero-mean solution (to 1e-13 relative in our tests). The bordered
    form is the default because with pinning the whole round-off residual of the solve
    lands in the one pinned cell, scaled by ``1/h^2``, which is about 20 times larger than
    the residual the LU leaves elsewhere; the factorisation cost is the same.
    """

    def __init__(self, grid: Grid, sides: dict[str, str], regularisation: str = "bordered"):
        if regularisation not in ("bordered", "pin"):
            raise ValueError(f"regularisation must be 'bordered' or 'pin', got {regularisation!r}")
        self.grid = grid
        self.sides = _check_sides(sides)
        self.shape = grid.shape_p()
        self.regularisation = regularisation
        m = {"neumann": "neumann", "dirichlet": "dirichlet_mid", "periodic": "periodic"}
        A = laplacian_matrix(grid.Nx, grid.Ny, grid.dx, grid.dy,
                             m[self.sides["left"]], m[self.sides["right"]],
                             m[self.sides["bottom"]], m[self.sides["top"]])
        self.matrix = A                      # the operator -L_h itself
        self.singular = "dirichlet" not in self.sides.values()
        if self.singular and regularisation == "bordered":
            n = A.shape[0]
            e = sp.csc_matrix(np.ones((n, 1)))
            B = sp.bmat([[A, e], [e.T, None]], format="csc")
        elif self.singular:
            B = A.tolil()
            B[0, 0] += 2.0 / grid.dx**2 + 2.0 / grid.dy**2
            B = B.tocsc()
        else:
            B = A
        self._lu = splu(B)

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """Return ``phi`` with ``-L_h phi = rhs``; zero-mean ``phi`` in the singular case
        (then the mean of ``rhs``, which no solution can match, is dropped)."""
        rhs = np.asarray(rhs, dtype=float)
        if rhs.shape != self.shape:
            raise ValueError(f"rhs has shape {rhs.shape}, expected {self.shape}")
        if not self.singular:
            return self._lu.solve(rhs.ravel()).reshape(self.shape)
        if self.regularisation == "bordered":
            x = self._lu.solve(np.concatenate([rhs.ravel(), [0.0]]))
            phi = x[:-1].reshape(self.shape)
        else:
            phi = self._lu.solve((rhs - rhs.mean()).ravel()).reshape(self.shape)
        phi -= phi.mean()
        return phi


class DCTPoissonSolver:
    """Fast solver for the all-Neumann box: ``-L_h phi = rhs`` by the type-2 DCT.

    The 1-D Neumann matrix ``k1(n, h, "neumann", "neumann")`` has eigenvectors
    ``cos(pi k (j + 1/2)/n)`` with eigenvalues ``(4/h^2) sin^2(pi k/(2n))``, which is exactly
    the DCT-II basis. So ``phi = idctn(dctn(rhs)/lambda)`` with
    ``lambda_kl = (4/dx^2) sin^2(pi k/(2Nx)) + (4/dy^2) sin^2(pi l/(2Ny))`` and the
    ``(0, 0)`` mode set to zero (zero-mean solution; the mean of ``rhs`` is dropped).
    """

    def __init__(self, grid: Grid):
        self.grid = grid
        self.shape = grid.shape_p()
        self.sides = {s: "neumann" for s in SIDES}
        self.singular = True
        k = np.arange(grid.Nx)
        l = np.arange(grid.Ny)
        lam = ((4.0 / grid.dx**2) * np.sin(0.5 * np.pi * k / grid.Nx) ** 2)[:, None] \
            + ((4.0 / grid.dy**2) * np.sin(0.5 * np.pi * l / grid.Ny) ** 2)[None, :]
        lam[0, 0] = 1.0          # placeholder; that mode is zeroed in solve()
        self._lam = lam

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """Return the zero-mean ``phi`` with ``-L_h phi = rhs - mean(rhs)``."""
        rhs = np.asarray(rhs, dtype=float)
        if rhs.shape != self.shape:
            raise ValueError(f"rhs has shape {rhs.shape}, expected {self.shape}")
        r = dctn(rhs, type=2, norm="ortho")
        r[0, 0] = 0.0
        r /= self._lam
        return idctn(r, type=2, norm="ortho")


def make_poisson_solver(grid: Grid, sides: dict[str, str]):
    """Pick ``DCTPoissonSolver`` when every side is Neumann, else ``PoissonSolver``."""
    sides = _check_sides(sides)
    if all(v == "neumann" for v in sides.values()):
        return DCTPoissonSolver(grid)
    return PoissonSolver(grid, sides)
