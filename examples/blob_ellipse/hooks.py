"""The constant-tension membrane and the area diagnostic of the perturbed ellipse.

update_springs: the paper's force is sigma kappa n, the force per unit length of a membrane
under the constant tension sigma. A closed chain of springs reproduces it exactly in
integrated form when every link carries the tension sigma: the total force on point k is then
sigma (tau_{k+1/2} - tau_{k-1/2}), which is the integral of sigma kappa n over the segment
between the two midpoints. pyIBM's spring force per unit length is k (L - rest) along the link
and the coupling multiplies it by ds, so the link tension is k ds (L - rest). With rest = 0 we
set k = sigma/(ds L) at every force evaluation; the driver calls this hook with the structure
at the positions where the forces are taken (the half step for the midpoint scheme), so the
tension is sigma there to round-off. The markers slide freely along the curve, as they do
in the paper, which says the spacing does not stay uniform.

on_output: the enclosed area (shoelace formula on the chain, unwrapped with the minimum
image) and its error relative to the initial area, written to <output_dir>/area.csv. This is
the "relative volume error" of the paper's Figures 4, 7 and 8.
"""

import os

import numpy as np

SIGMA = 0.2           # N, the paper's sigma = 1/5 (also in generate.py)
COLUMNS = ("step", "t", "area", "rel_area_error", "perimeter")


def update_springs(t, dt, s):
    sp = s.springs
    d = s.X[sp.j] - s.X[sp.i]                      # the membrane never crosses the box edge
    L = np.hypot(d[:, 0], d[:, 1])
    sp.k[:] = SIGMA / (s.ds[sp.i] * L)
    sp.rest[:] = 0.0


def polygon_area_perimeter(X, grid):
    d = grid.min_image(np.roll(X, -1, axis=0) - X)
    P = X[0] + np.concatenate([np.zeros((1, 2)), np.cumsum(d[:-1], axis=0)])
    x, y = P[:, 0], P[:, 1]
    area = 0.5 * abs(float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))
    return area, float(np.sum(np.hypot(d[:, 0], d[:, 1])))


def on_output(step, t, state):
    area, perimeter = polygon_area_perimeter(state.structure.X, state.grid)
    area0, _ = polygon_area_perimeter(state.structure.X0, state.grid)
    # the driver creates the folder only when it writes files of its own
    os.makedirs(state.output_dir, exist_ok=True)
    path = os.path.join(state.output_dir, "area.csv")
    mode = "w" if step == 0 else "a"
    with open(path, mode, encoding="utf-8") as fh:
        if step == 0:
            fh.write(",".join(COLUMNS) + "\n")
        fh.write(f"{step},{t!r},{area!r},{(area - area0) / area0!r},{perimeter!r}\n")
