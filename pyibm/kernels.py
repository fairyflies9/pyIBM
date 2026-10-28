"""Discrete delta-function kernels and the 1-D stencils built from them.

The immersed boundary method couples Lagrangian points to the Eulerian grid through a
smoothed delta function

    delta_h(x, y) = phi(x/h) phi(y/h) / h^2

(Peskin & McQueen 1996 eq. 14.62; Peskin 2002 eq. 6.1, both written there in 3-D). This
module holds the 1-D kernels phi and the routine that turns a set of Lagrangian coordinates
into (grid index, weight) pairs along one axis. The spreading and interpolation operators in
``pyibm.coupling.delta`` are tensor products of these 1-D stencils.

Three kernels are provided:

- ``peskin4``: the standard 4-point kernel of Peskin & McQueen 1996 eq. 14.73 (Peskin 2002
  eq. 6.27). Satisfies the even/odd sum rule, the first-moment condition and the
  sum-of-squares condition (Peskin & McQueen 1996 eqs. 14.63-14.65). The default.
- ``cosine4``: Peskin's 1977 cosine kernel (Peskin 2002 eq. 6.28), used in my 2008 MATLAB
  code. It satisfies the even/odd sum rule, so it conserves total force, and has the same
  sum of squares (3/8). Its first moment is not zero, so it does not interpolate linear
  fields exactly. Kept so the 2008 results can be reproduced.
- ``roma3``: the 3-point kernel of Roma, Peskin & Berger 1999, built for staggered (MAC)
  grids. It satisfies the partition of unity but not the even/odd sum rule, so it lets in
  the grid-scale (-1)^j mode that the centred-difference gradient of the collocated
  periodic back end cannot see. Peskin 2002 (section 6, the paragraphs after eq. 6.28) recommends
  the 3-point kernel only with a scheme that does not have that null space, such as MAC.
  Not meant for the collocated periodic back end.

Conventions: ``r`` is a distance in cells (``|x|/h``); ``phi`` is even and returns 0 outside
its support; everything is float64 and vectorized. See docs/numerics.md for how the kernels
fit into the coupling.

References
----------
Peskin, C. S. (1977). Numerical analysis of blood flow in the heart. Journal of
Computational Physics 25, 220-252.

Peskin, C. S. and McQueen, D. M. (1996). Fluid dynamics of the heart and its valves. In
Othmer, H. G., Adler, F. R., Lewis, M. A. and Dallon, J. C. (eds.), Case Studies in
Mathematical Modeling: Ecology, Physiology, and Cell Biology, chapter 14, pp. 313-342.
Prentice Hall.

Peskin, C. S. (2002). The immersed boundary method. Acta Numerica 11, 479-517.

Roma, A. M., Peskin, C. S. and Berger, M. J. (1999). An adaptive version of the immersed
boundary method. Journal of Computational Physics 153, 509-534.
"""

from __future__ import annotations

import numpy as np

__all__ = ["Kernel", "Peskin4", "Cosine4", "Roma3", "get_kernel", "stencil_1d"]


class Kernel:
    """Base class for a 1-D delta kernel phi(r).

    Attributes
    ----------
    name : str
        The name ``get_kernel`` knows the kernel by.
    half_width : float
        Half-width of the support in cells (2.0 for the 4-point kernels, 1.5 for 3-point).
    width : int
        Number of grid nodes touched per direction (4 or 3).

    ``phi(r)`` must be even, continuous, vanish for ``|r| >= half_width``, and satisfy the
    discrete partition of unity

        sum_j phi(r - j) = 1                                  for every real r

    (Peskin & McQueen 1996 eq. 14.66; Peskin 2002 eq. 6.8, or eq. 5.3 written for delta_h),
    which is what makes spreading conserve total force and interpolation reproduce constants.

    The two 4-point kernels also satisfy the stronger even/odd sum rule

        sum_{j even} phi(r - j) = sum_{j odd} phi(r - j) = 1/2      for every real r

    (Peskin & McQueen 1996 eq. 14.63; Peskin 2002 eq. 6.4), and the partition of unity
    follows from it. roma3 satisfies the partition of unity but not the even/odd rule: its
    even and odd sums differ from 1/2 by up to 1/6. The even/odd rule is what keeps a spread
    force free of the grid-scale mode (-1)^j, whose discrete Fourier transform sits at wave
    number N/2 and which the centred difference D0 cannot see (Peskin & McQueen 1996, the
    discussion after eq. 14.66).
    """

    name: str = "base"
    half_width: float = 0.0
    width: int = 0

    def phi(self, r: np.ndarray) -> np.ndarray:
        """Kernel value at r (distance in cells, any sign). Vectorized; 0 outside support."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, width={self.width})"


class Peskin4(Kernel):
    """Peskin's 4-point kernel, Peskin & McQueen 1996 eq. 14.73 (Peskin 2002 eq. 6.27).

        phi(r) = (3 - 2r + sqrt(1 + 4r - 4r^2)) / 8          0 <= r <= 1
        phi(r) = (5 - 2r - sqrt(-7 + 12r - 4r^2)) / 8        1 <  r <= 2
        phi(r) = 0                                           r > 2

    with r = |x|/h. phi(0) = 1/2, phi(1) = 1/4, phi(2) = 0. The radicands are clamped at zero
    before the square root so that round-off at r = 1 and r = 2 cannot produce NaN.
    """

    name = "peskin4"
    half_width = 2.0
    width = 4

    def phi(self, r: np.ndarray) -> np.ndarray:
        r = np.abs(np.asarray(r, dtype=float))
        inner = (3.0 - 2.0 * r + np.sqrt(np.maximum(1.0 + 4.0 * r - 4.0 * r * r, 0.0))) / 8.0
        outer = (5.0 - 2.0 * r - np.sqrt(np.maximum(-7.0 + 12.0 * r - 4.0 * r * r, 0.0))) / 8.0
        return np.where(r <= 1.0, inner, np.where(r <= 2.0, outer, 0.0))


class Cosine4(Kernel):
    """Peskin's 1977 cosine kernel, as used in forcespread.m / move_boundary.m (2008).

        phi(r) = (1 + cos(pi r / 2)) / 4      r < 2          (Peskin 2002 eq. 6.28)
        phi(r) = 0                            r >= 2

    phi(0) = 1/2, phi(1) = 1/4, phi(2) = 0. It satisfies the even/odd sum rule
    (phi(r) + phi(r - 2) = 1/2, Peskin & McQueen 1996 eq. 14.63) and has sum of squares 3/8
    like the 4-point kernel, but its first moment sum_j (r - j) phi(r - j) is not zero, so
    linear fields are not interpolated exactly. Kept to reproduce the 2008 results.
    """

    name = "cosine4"
    half_width = 2.0
    width = 4

    def phi(self, r: np.ndarray) -> np.ndarray:
        r = np.abs(np.asarray(r, dtype=float))
        return np.where(r < 2.0, (1.0 + np.cos(0.5 * np.pi * r)) / 4.0, 0.0)


class Roma3(Kernel):
    """The 3-point kernel of Roma, Peskin & Berger 1999 (J. Comput. Phys. 153, 509-534).

        phi(r) = (1 + sqrt(1 - 3 r^2)) / 3                    0   <= r <= 1/2
        phi(r) = (5 - 3r - sqrt(1 - 3 (1 - r)^2)) / 6         1/2 <  r <= 3/2
        phi(r) = 0                                            r > 3/2

    phi(0) = 2/3, phi(1) = 1/6, phi(3/2) = 0. It satisfies the partition of unity, the
    first-moment condition and the sum-of-squares condition (sum phi^2 = 1/2) on a 3-node
    stencil, but not the separate even/odd sum rule (Peskin & McQueen 1996 eq. 14.63): at an
    integer r one of the two sums is 2/3 and the other 1/3. So it should be used on the
    staggered MAC grid only (Peskin 2002 section 6, after eq. 6.28; docs/numerics.md).
    Radicands are clamped at zero before the sqrt.
    """

    name = "roma3"
    half_width = 1.5
    width = 3

    def phi(self, r: np.ndarray) -> np.ndarray:
        r = np.abs(np.asarray(r, dtype=float))
        inner = (1.0 + np.sqrt(np.maximum(1.0 - 3.0 * r * r, 0.0))) / 3.0
        outer = (5.0 - 3.0 * r - np.sqrt(np.maximum(1.0 - 3.0 * (1.0 - r) ** 2, 0.0))) / 6.0
        return np.where(r <= 0.5, inner, np.where(r <= 1.5, outer, 0.0))


def get_kernel(name: str = "peskin4") -> Kernel:
    """Return a kernel by name: "peskin4" (default), "cosine4" or "roma3" (case-insensitive)."""
    key = str(name).strip().lower()
    if key == "peskin4":
        return Peskin4()
    if key == "cosine4":
        return Cosine4()
    if key == "roma3":
        return Roma3()
    raise ValueError(f"unknown delta kernel {name!r}; choose 'peskin4', 'cosine4' or 'roma3'")


def stencil_1d(kernel: Kernel, X: np.ndarray, x0: float, h: float, n: int, periodic: bool
               ) -> tuple[np.ndarray, np.ndarray]:
    """Grid indices and kernel weights along one axis for the points X.

    Returns ``(indices, weights)``, both of shape ``(N, kernel.width)``, with

        weights[k, m] = phi((x_idx - X_k) / h) / h,     x_idx = x0 + idx * h,

    the 1-D factor of delta_h (Peskin & McQueen 1996 eq. 14.62: delta_h is the product of
    one such factor per direction, with phi from eq. 14.73 for the 4-point kernel). The nodes
    of the field live at ``x0 + i*h`` for ``i = 0 .. n-1``.

    Node selection (the distance always uses the UNWRAPPED index):

    - even width (4-point kernels): ``i0 = floor((X - x0)/h)``, ``idx = i0-1 .. i0+2``. These
      are exactly the nodes with ``|x_idx - X| < 2h`` (plus one node at distance 2h with zero
      weight when X sits on a node). The 2008 MATLAB code used ``x1 = ceil(X/dx)`` and nodes
      ``x1+1-k``, k = 0..3, which is the same set of nodes off the grid points.
    - odd width (3-point kernel): ``i0 = round((X - x0)/h)``, ``idx = i0-1 .. i0+1``. We round
      half up (``floor(s + 1/2)``) so the choice is deterministic; at an exact half the node
      that differs between the two roundings sits at distance 3h/2 and has zero weight.

    ``periodic``: indices are wrapped with ``idx % n``. Otherwise any index outside
    ``0 .. n-1`` raises ``ValueError`` ("Lagrangian point within kernel support of a
    non-periodic boundary"): the kernel would need nodes beyond the wall. ``n`` is the number
    of nodes of the field being addressed along this axis (``Nx+1`` for u-faces in a
    non-periodic x direction, ``Nx`` for cell centres, and so on). The check is on the index
    set, so a point exactly 2h from the last node of a 4-point stencil raises even though
    the offending weight is zero; keeping Lagrangian points at least two cells from a wall
    is the intended contract.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 1:
        raise ValueError(f"X must be a 1-D array of coordinates, got shape {X.shape}")
    if h <= 0:
        raise ValueError(f"grid spacing h must be positive, got {h}")
    if n < 1:
        raise ValueError(f"number of nodes n must be positive, got {n}")
    s = (X - x0) / h
    if not np.all(np.isfinite(s)):
        raise ValueError("non-finite Lagrangian coordinate passed to stencil_1d")

    width = int(kernel.width)
    if width % 2 == 0:
        i0 = np.floor(s).astype(np.int64)
        offsets = np.arange(-(width // 2 - 1), width // 2 + 1, dtype=np.int64)   # -1..2 for 4
    else:
        i0 = np.floor(s + 0.5).astype(np.int64)
        offsets = np.arange(-(width // 2), width // 2 + 1, dtype=np.int64)       # -1..1 for 3
    idx = i0[:, None] + offsets[None, :]                     # unwrapped, shape (N, width)
    weights = kernel.phi(idx - s[:, None]) / h               # (x_idx - X)/h = idx - s

    if periodic:
        idx = np.mod(idx, n)
    elif idx.size:
        bad = (idx < 0) | (idx > n - 1)
        if np.any(bad):
            k = int(np.argmax(np.any(bad, axis=1)))
            raise ValueError(
                "Lagrangian point within kernel support of a non-periodic boundary: "
                f"point {k} at coordinate {X[k]!r} needs nodes {int(idx[k, 0])}..{int(idx[k, -1])} "
                f"of 0..{n - 1}")
    return idx, weights
