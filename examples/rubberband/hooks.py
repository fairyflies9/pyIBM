"""Per-output diagnostics of the rubber band: the enclosed area, the perimeter and the
shape, written to <output_dir>/area.csv next to history.csv.

The band is a closed polygon; its area comes from the shoelace formula on the positions
unwrapped along the chain with the grid's minimum image (the band sits in the middle of the
box, but the unwrapping costs nothing). The shape is measured by the ratio of the largest to
the smallest distance from the centroid, 2 for the initial ellipse and 1 for a circle.
"""

import os

import numpy as np

COLUMNS = ("step", "t", "area", "perimeter", "rmax_over_rmin")


def polygon_geometry(X, grid):
    """(area, perimeter, rmax/rmin) of the closed polygon through the rows of X (N, 2)."""
    d = grid.min_image(np.roll(X, -1, axis=0) - X)            # segment k: X[k] -> X[k+1]
    P = X[0] + np.concatenate([np.zeros((1, 2)), np.cumsum(d[:-1], axis=0)])
    x, y = P[:, 0], P[:, 1]
    area = 0.5 * abs(float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))
    perimeter = float(np.sum(np.hypot(d[:, 0], d[:, 1])))
    r = np.hypot(x - x.mean(), y - y.mean())
    return area, perimeter, float(r.max() / r.min())


def on_output(step, t, state):
    area, perimeter, ratio = polygon_geometry(state.structure.X, state.grid)
    # the driver creates the folder only when it writes files of its own
    os.makedirs(state.output_dir, exist_ok=True)
    path = os.path.join(state.output_dir, "area.csv")
    mode = "w" if step == 0 else "a"
    with open(path, mode, encoding="utf-8") as fh:
        if step == 0:
            fh.write(",".join(COLUMNS) + "\n")
        fh.write(f"{step},{t!r},{area!r},{perimeter!r},{ratio!r}\n")
