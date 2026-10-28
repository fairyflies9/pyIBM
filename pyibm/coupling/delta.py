"""Delta-function coupling: spread Lagrangian forces to the grid, interpolate grid velocities.

These are the two interaction equations of the immersed boundary method (Peskin & McQueen
1996 eqs. 14.58 and 14.61; Peskin 2002 eqs. 4.18-4.19 for the operators and eqs. 6.27-6.28
for the kernels):

    F(x_ij) = sum_k f_k ds_k delta_h(x_ij - X_k)                   (spread, eq. 14.58)
    U_k     = sum_ij u(x_ij) delta_h(x_ij - X_k) hx hy             (interpolate, eq. 14.61)

with delta_h(x, y) = phi(x/hx) phi(y/hy) / (hx hy). The two operators are adjoint with
respect to the inner products sum_ij (.)(.) hx hy on the grid and sum_k (.)(.) ds_k on the
boundary, which is what makes the power the structure puts into the fluid equal the power
the fluid takes out (Peskin 2002 section 5, eq. 5.2). The implementation keeps that property
exactly, up to round-off, by using the same (index, weight) tables for both operations.

Forces ``f`` are force densities per unit fiber length [N/m], each point carries a quadrature
weight ``ds`` [m] (the conventions are in docs/numerics.md), and the spread field ``F`` is a
body force per unit area [N/m^2]. A ``FieldLayout`` (``pyibm.grid``) says where the grid
field lives, so the same code addresses collocated nodes, u-faces and v-faces.

References
----------
Peskin, C. S. and McQueen, D. M. (1996). Fluid dynamics of the heart and its valves. In
Othmer, H. G., Adler, F. R., Lewis, M. A. and Dallon, J. C. (eds.), Case Studies in
Mathematical Modeling: Ecology, Physiology, and Cell Biology, chapter 14, pp. 313-342.
Prentice Hall.

Peskin, C. S. (2002). The immersed boundary method. Acta Numerica 11, 479-517.
"""

from __future__ import annotations

import numpy as np

from pyibm.grid import FieldLayout
from pyibm.kernels import Kernel, get_kernel, stencil_1d

__all__ = ["DeltaCoupling"]


class DeltaCoupling:
    """Spreading and interpolation with a discrete delta function.

    Parameters
    ----------
    kernel : Kernel or str
        The 1-D kernel phi (see ``pyibm.kernels``); a name is looked up with ``get_kernel``.
    """

    def __init__(self, kernel: Kernel | str = "peskin4"):
        self.kernel = get_kernel(kernel) if isinstance(kernel, str) else kernel
        if not isinstance(self.kernel, Kernel):
            raise TypeError(f"kernel must be a Kernel or a kernel name, got {type(kernel).__name__}")

    # ------------------------------------------------------------------ helpers
    def _stencils(self, X: np.ndarray, layout: FieldLayout
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Per-axis (index, weight) tables, each (N, width), for the points X on this layout."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != 2:
            raise ValueError(f"X must have shape (N, 2), got {X.shape}")
        ix, wx = stencil_1d(self.kernel, X[:, 0], layout.x0, layout.hx, layout.nx, layout.periodic_x)
        iy, wy = stencil_1d(self.kernel, X[:, 1], layout.y0, layout.hy, layout.ny, layout.periodic_y)
        return ix, wx, iy, wy

    # ------------------------------------------------------------------ scalar operators
    def spread(self, f: np.ndarray, X: np.ndarray, ds: np.ndarray, layout: FieldLayout) -> np.ndarray:
        """Spread the Lagrangian force density f to the grid field of the given layout.

        Peskin & McQueen 1996 eq. 14.58:

            F(x_ij) = sum_k f_k ds_k delta_h(x_ij - X_k),   delta_h = phi(rx) phi(ry) / (hx hy).

        Parameters
        ----------
        f : (N,) force density per unit length [N/m] (one component).
        X : (N, 2) Lagrangian positions [m].
        ds : (N,) quadrature weight per point [m]; a scalar is broadcast to every point.
        layout : FieldLayout of the target field.

        Returns
        -------
        F : (nx, ny) body force per unit area [N/m^2].

        The 1-D weights from ``stencil_1d`` already carry the 1/hx and 1/hy factors. The
        ``width x width`` contributions of every point are accumulated with ``np.bincount``
        on flattened (i, j) indices, so points that share nodes (or a periodic image) add
        up correctly without a Python loop.
        """
        ix, wx, iy, wy = self._stencils(X, layout)
        N = ix.shape[0]
        f = np.asarray(f, dtype=float)
        if f.shape != (N,):
            raise ValueError(f"f must have shape ({N},) to match X; got {f.shape} "
                             "(use spread_vector for (N, 2) forces)")
        ds = np.broadcast_to(np.asarray(ds, dtype=float), (N,))
        contrib = (f * ds)[:, None, None] * wx[:, :, None] * wy[:, None, :]      # (N, w, w)
        flat = ix[:, :, None] * layout.ny + iy[:, None, :]                        # (N, w, w)
        F = np.bincount(flat.ravel(), weights=contrib.ravel(), minlength=layout.nx * layout.ny)
        return F.reshape(layout.nx, layout.ny)

    def interpolate(self, field: np.ndarray, X: np.ndarray, layout: FieldLayout) -> np.ndarray:
        """Interpolate a grid field to the Lagrangian points.

        Peskin & McQueen 1996 eq. 14.61:

            U_k = sum_ij field(x_ij) delta_h(x_ij - X_k) hx hy.

        Parameters
        ----------
        field : (nx, ny) grid field on ``layout``.
        X : (N, 2) Lagrangian positions [m].

        Returns
        -------
        U : (N,) interpolated values.

        For kernels that satisfy the first-moment condition (peskin4, roma3) a linear field
        is reproduced exactly; every kernel reproduces constants.
        """
        field = np.asarray(field, dtype=float)
        if field.shape != layout.shape:
            raise ValueError(f"field has shape {field.shape} but the layout is {layout.shape}")
        ix, wx, iy, wy = self._stencils(X, layout)
        vals = field[ix[:, :, None], iy[:, None, :]]                              # (N, w, w)
        return np.einsum("kab,ka,kb->k", vals, wx, wy) * (layout.hx * layout.hy)

    # ------------------------------------------------------------------ vector conveniences
    def spread_vector(self, f: np.ndarray, X: np.ndarray, ds: np.ndarray,
                      layout_x: FieldLayout, layout_y: FieldLayout) -> tuple[np.ndarray, np.ndarray]:
        """Spread an (N, 2) force density: x component to ``layout_x``, y component to ``layout_y``.

        Returns ``(Fx, Fy)`` with shapes ``layout_x.shape`` and ``layout_y.shape``. On the
        collocated periodic grid both layouts are the same; on the MAC grid they are the u- and
        v-face layouts.
        """
        f = np.asarray(f, dtype=float)
        if f.ndim != 2 or f.shape[1] != 2:
            raise ValueError(f"f must have shape (N, 2), got {f.shape}")
        Fx = self.spread(f[:, 0], X, ds, layout_x)
        Fy = self.spread(f[:, 1], X, ds, layout_y)
        return Fx, Fy

    def interpolate_vector(self, u: np.ndarray, v: np.ndarray, X: np.ndarray,
                           layout_x: FieldLayout, layout_y: FieldLayout) -> np.ndarray:
        """Interpolate the velocity components u (on ``layout_x``) and v (on ``layout_y``).

        Returns an (N, 2) array of Lagrangian velocities.
        """
        U = self.interpolate(u, X, layout_x)
        V = self.interpolate(v, X, layout_y)
        return np.column_stack([U, V])
