"""Matplotlib snapshots of the vorticity field and the immersed boundary.

Figures are built on matplotlib's Agg canvas directly (``matplotlib.figure.Figure`` plus
``FigureCanvasAgg``) instead of through ``pyplot``. That works on a headless machine, never
touches the global backend or the user's interactive session, and keeps no registry: dropping
the returned ``Figure`` frees it, there is nothing to ``plt.close``. In a notebook, display the
returned figure with ``IPython.display.display(fig)``.

Grid fields are (Nx, Ny) arrays indexed [i, j] with i along x (docs/numerics.md);
``pcolormesh`` wants rows along y, so the transpose happens here and nowhere else.
"""

from __future__ import annotations

import os

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def _extend_periodic(grid, x: np.ndarray, y: np.ndarray, omega: np.ndarray):
    """Append the wrapped column (x = x[0] + Lx) and row (y = y[0] + Ly) in periodic directions.

    Without this the strip between the last node and the box edge would be blank; with it the
    periodic box is drawn fully and the plot shows that the field wraps.
    """
    if grid.periodic_x:
        x = np.append(x, x[0] + grid.Lx)
        omega = np.concatenate([omega, omega[:1, :]], axis=0)
    if grid.periodic_y:
        y = np.append(y, y[0] + grid.Ly)
        omega = np.concatenate([omega, omega[:, :1]], axis=1)
    return x, y, omega


def _figsize(Lx: float, Ly: float) -> tuple[float, float]:
    """Figure size (inches) with the axes in the box's aspect ratio plus room for the colour bar."""
    aspect = Ly / Lx
    w, h = 6.0, 6.0 * aspect
    if h > 8.0:
        w, h = 8.0 / aspect, 8.0
    return (w + 1.6, h + 0.8)


def snapshot(grid, omega: np.ndarray, X=None, targets=None, path=None, vmax=None, title=None,
             location=None) -> Figure:
    """Draw a vorticity field with the immersed boundary on top.

    Parameters
    ----------
    grid : ``pyibm.grid.Grid``.
    omega : vorticity, shape ``(grid.Nx, grid.Ny)`` indexed [i, j].
    X : optional Lagrangian positions (N, 2), drawn as black dots (wrapped into the box in
        periodic directions; dots rather than lines so fibres crossing the box edge do not
        streak across the plot).
    targets : optional target positions (M, 2), drawn as smaller red dots. An object with an
        ``X`` attribute (the ``Targets`` dataclass) is accepted too.
    path : if given, the figure is also saved there (format from the extension, PNG at 150 dpi;
        the parent directory is created).
    vmax : colour limits are ``[-vmax, vmax]``; default ``max |omega|`` (an all-zero field gets
        ``[-1, 1]``). Pass a fixed value to keep the colours comparable between snapshots.
    title : optional axes title.
    location : ``"nodes"`` (periodic FFT back end, values at ``grid.x, grid.y``) or
        ``"centres"`` (MAC back end, values at ``grid.xc, grid.yc``). Default: nodes when the
        grid is doubly periodic, centres otherwise. The grid does not know which back end
        owns it, so the default is right for the FFT back end and for MAC grids with walls
        or a channel, but not for the MAC back end on a doubly periodic grid: its vorticity
        is at the cell centres, so pass ``location="centres"`` there (the driver does), or
        the field is drawn half a cell off in x and y.

    Uses ``pcolormesh(x, y, omega.T, shading="nearest")`` so each cell is centred on its data
    point, a symmetric diverging colour map (zero vorticity is white), equal aspect, and the
    axes clipped to ``[0, Lx] x [0, Ly]``. In periodic directions the wrapped row and column
    are appended so the whole box is painted. Returns the ``Figure``.
    """
    omega = np.asarray(omega, dtype=float)
    if omega.shape != (grid.Nx, grid.Ny):
        raise ValueError(f"omega must have shape {(grid.Nx, grid.Ny)}, got {omega.shape}")
    if location is None:
        location = "nodes" if (grid.periodic_x and grid.periodic_y) else "centres"
    if location == "nodes":
        x, y = grid.x, grid.y
    elif location == "centres":
        x, y = grid.xc, grid.yc
    else:
        raise ValueError(f"location must be 'nodes' or 'centres', got {location!r}")
    x, y, w = _extend_periodic(grid, x, y, omega)

    if vmax is None:
        vmax = float(np.max(np.abs(w))) if w.size else 0.0
    vmax = float(vmax)
    if not vmax > 0.0:
        vmax = 1.0

    fig = Figure(figsize=_figsize(grid.Lx, grid.Ly), layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 1, 1)
    mesh = ax.pcolormesh(x, y, w.T, shading="nearest", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    fig.colorbar(mesh, ax=ax, label="vorticity [1/s]")

    if X is not None:
        Xw = grid.wrap(np.asarray(X, dtype=float).reshape(-1, 2))
        ax.plot(Xw[:, 0], Xw[:, 1], linestyle="none", marker=".", markersize=3.0, color="k",
                zorder=3, label="boundary")
    if targets is not None:
        T = getattr(targets, "X", targets)
        Tw = grid.wrap(np.asarray(T, dtype=float).reshape(-1, 2))
        ax.plot(Tw[:, 0], Tw[:, 1], linestyle="none", marker=".", markersize=2.0, color="r",
                zorder=4, label="targets")

    ax.set_xlim(0.0, grid.Lx)
    ax.set_ylim(0.0, grid.Ly)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    if title is not None:
        ax.set_title(title)

    if path is not None:
        path = os.fspath(path)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        fig.savefig(path, dpi=150)
    return fig
