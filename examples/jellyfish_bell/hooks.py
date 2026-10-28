"""Muscle drive and swimming diagnostics of the jellyfish-bell case (Hoover & Miller 2015).

update_springs: the muscle springs are the ones that join a point to its mirror image across
the bell (every other spring joins neighbours, so |i - j| > 1 picks them out). Their rest
length follows

    rest(t) = L0 (1 - A_m sin(2 pi f t)),

with L0 the preferred width of the band at that point (from the initial positions X0) and
A_m = 1 (generate.py, MUSCLE_AMPLITUDE), so the spring force k_m (L - rest) is the paper's
F_Mag sin(2 pi f t) up to the strain of the band (eq. 2 and 7 of the paper). The frequency f
is read from the structure name, bell[_fine]_f<frequency>; a structure without muscle
springs (bell_free) is left alone. The law is a function of absolute time: the driver passes
t + dt/2 for the midpoint scheme.

on_output: appends to <output_dir>/bell.csv the centroid of the bell (mean of the points,
unwrapped in y so the swim distance keeps growing across the periodic box, measured
relative to the box centre), the diameter (distance between the two points nearest the
widest cross-section, Fig. 2) and the muscle force per unit length of the first muscle
spring. analyze.py turns these into the forward speed per cycle (Fig. 7) and the amplitude
of the diameter oscillation (Fig. 6).
"""

from __future__ import annotations

import os
import re

import numpy as np

MUSCLE_AMPLITUDE = 1.0     # must equal generate.MUSCLE_AMPLITUDE (checked by the test)
_NAME = re.compile(r"_f([0-9.eE+-]+)$")


def frequency(name: str) -> float | None:
    """The driving frequency encoded in the structure name, None for the free vibration."""
    m = _NAME.search(name)
    return None if m is None else float(m.group(1))


def muscle_mask(s) -> np.ndarray:
    """Springs that cross the bell: |i - j| > 1."""
    return np.abs(s.springs.i - s.springs.j) > 1


def update_springs(t, dt, s):
    f = frequency(s.name)
    if f is None or s.springs is None:
        return
    m = muscle_mask(s)
    if not np.any(m):
        return
    L0 = np.hypot(*(s.X0[s.springs.j[m]] - s.X0[s.springs.i[m]]).T)
    s.springs.rest[m] = L0 * (1.0 - MUSCLE_AMPLITUDE * np.sin(2.0 * np.pi * f * t))


def widest_pair(s) -> tuple[int, int]:
    """Indices of the leftmost and rightmost points of the preferred bell."""
    return int(np.argmin(s.X0[:, 0])), int(np.argmax(s.X0[:, 0]))


def on_output(step, t, state):
    s, grid = state.structure, state.grid
    # the bell may straddle a periodic edge: unwrap every point about the first one, so the
    # centroid is continuous in time except for a jump of one box length when the whole
    # bell wraps, which analyze.py removes
    X = s.X[0] + grid.min_image(s.X - s.X[0])
    cx, cy = X.mean(axis=0)
    il, ir = widest_pair(s)
    d = grid.min_image(s.X[ir] - s.X[il])
    diameter = float(np.hypot(*d))
    force = 0.0
    if s.springs is not None:
        m = muscle_mask(s)
        if np.any(m):
            k0 = int(np.flatnonzero(m)[0])
            i, j = s.springs.i[k0], s.springs.j[k0]
            Lk = float(np.hypot(*grid.min_image(s.X[j] - s.X[i])))
            force = float(s.springs.k[k0] * (Lk - s.springs.rest[k0]))
    path = os.path.join(state.output_dir, "bell.csv")
    mode = "w" if step == 0 else "a"
    with open(path, mode, encoding="utf-8") as fh:
        if step == 0:
            fh.write("step,t,cx,cy,diameter,muscle_force\n")
        fh.write(f"{step},{t!r},{cx!r},{cy!r},{diameter!r},{force!r}\n")
