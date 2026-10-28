"""Boundary data and the free-end recorder of the flapping-filament case.

The stream enters at the left and the far-field sides (top and bottom) carry the free
stream too, (u, v) = (U_inf, 0), as in Huang, Shin & Sung 2007 sec. 3.2.2 ("Dirichlet
boundary conditions u = U_inf, v = 0 at the inflow and far-field boundaries"); the right
side is the outflow in the input file, so there is no bc_right here.

The free stream is ramped up over T_RAMP = 1 time unit with a half cosine instead of
being switched on at t = 0. The fluid starts at rest, and an impulsive start makes the
fluid at the filament jump to U_inf in one step while the heavy ghost points are still at
rest: the mass tethers then ring at sqrt(k_m / m) with a force of +/- 40 N on a filament
whose drag is 0.3 N, and the ringing takes several time units to die out. With the ramp
the tether stretch stays below 0.01 cells from the first step. The paper's feedback
forcing has its own start-up transient; the settled flapping does not depend on it.

on_output appends one row per output to <output_dir>/tip.csv: the free end (node 0),
the fixed end (the last node), the filament length and the largest segment strain. The
paper's quantity of interest is Y(0, t), the transverse position of the free end (their
Fig. 13); analyze.py reads this file. The file is started afresh at step 0.
"""

from __future__ import annotations

import os

import numpy as np

U_INF = 1.0     # m/s, must match generate.py
T_RAMP = 1.0    # s, the free stream reaches U_inf at this time

TIP_COLUMNS = ("step", "t", "x_tip", "y_tip", "x_fixed", "y_fixed", "length", "max_strain")


def free_stream_speed(t: float) -> float:
    """U_inf (1 - cos(pi t / T_RAMP)) / 2 for t < T_RAMP, then U_inf."""
    if t >= T_RAMP:
        return U_INF
    return 0.5 * U_INF * (1.0 - np.cos(np.pi * t / T_RAMP))


def _free_stream(coord, t):
    coord = np.asarray(coord, dtype=float)
    return np.full(coord.shape, free_stream_speed(t)), np.zeros(coord.shape)


def bc_left(y, t):
    return _free_stream(y, t)


def bc_bottom(x, t):
    return _free_stream(x, t)


def bc_top(x, t):
    return _free_stream(x, t)


def filament_length(X: np.ndarray, ds: float) -> tuple[float, float]:
    """Length of the polyline X (N, 2) and the largest |segment/ds - 1|."""
    seg = np.hypot(*(X[1:] - X[:-1]).T)
    return float(seg.sum()), float(np.max(np.abs(seg / ds - 1.0)))


def on_output(step, t, state):
    s = state.structure
    X = s.X
    length, strain = filament_length(X, float(s.ds[0]))
    row = (step, t, X[0, 0], X[0, 1], X[-1, 0], X[-1, 1], length, strain)
    path = os.path.join(state.output_dir, "tip.csv")
    os.makedirs(state.output_dir, exist_ok=True)
    fresh = step == 0 or not os.path.isfile(path)
    with open(path, "w" if fresh else "a", encoding="utf-8", newline="\n") as fh:
        if fresh:
            fh.write(",".join(TIP_COLUMNS) + "\n")
        fh.write(",".join(f"{v:.17g}" for v in row) + "\n")
