"""Centreline velocities of Ghia, Ghia & Shin 1982 for the lid-driven cavity.

U. Ghia, K. N. Ghia and C. T. Shin, "High-Re solutions for incompressible flow using the
Navier-Stokes equations and a multigrid method", J. Comput. Phys. 48 (1982) 387-411.
Table I (p. 398) lists u along the vertical line through the geometric centre of the cavity
(x = 0.5) and Table II (p. 399) lists v along the horizontal line through the centre
(y = 0.5), both from their 129 x 129 uniform grid, for Re = 100,
400, 1000, 3200, 5000, 7500 and 10000. The columns for Re = 100, 400 and 1000 are copied
here, at the 17 grid points they tabulate. The lid moves in +x with unit speed, the cavity is
the unit square, Re = U L/nu.

The comparison leaves out one tabulated point: v at x = 0.9063 for Re = 400, -0.23827,
which is out of line with its neighbours (-0.22847 at x = 0.9453, -0.44993 at x = 0.8594);
our 128^2 run gives -0.387 there. It is kept as tabulated, listed in LEFT_OUT, and left out
of the pass/fail check in analyze.py and the tests.
"""

from __future__ import annotations

import numpy as np

# Table I: y of the 17 tabulated points and u(0.5, y).
Y = np.array([1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172, 0.5000,
              0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0.0000])
U = {
    100: np.array([1.00000, 0.84123, 0.78871, 0.73722, 0.68717, 0.23151, 0.00332, -0.13641,
                   -0.20581, -0.21090, -0.15662, -0.10150, -0.06434, -0.04775, -0.04192,
                   -0.03717, 0.00000]),
    400: np.array([1.00000, 0.75837, 0.68439, 0.61756, 0.55892, 0.29093, 0.16256, 0.02135,
                   -0.11477, -0.17119, -0.32726, -0.24299, -0.14612, -0.10338, -0.09266,
                   -0.08186, 0.00000]),
    1000: np.array([1.00000, 0.65928, 0.57492, 0.51117, 0.46604, 0.33304, 0.18719, 0.05702,
                    -0.06080, -0.10648, -0.27805, -0.38289, -0.29730, -0.22220, -0.20196,
                    -0.18109, 0.00000]),
}

# Table II: x of the 17 tabulated points and v(x, 0.5).
X = np.array([1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047, 0.5000,
              0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625, 0.0000])
V = {
    100: np.array([0.00000, -0.05906, -0.07391, -0.08864, -0.10313, -0.16914, -0.22445,
                   -0.24533, 0.05454, 0.17527, 0.17507, 0.16077, 0.12317, 0.10890, 0.10091,
                   0.09233, 0.00000]),
    400: np.array([0.00000, -0.12146, -0.15663, -0.19254, -0.22847, -0.23827, -0.44993,
                   -0.38598, 0.05186, 0.30174, 0.30203, 0.28124, 0.22965, 0.20920, 0.19713,
                   0.18360, 0.00000]),
    1000: np.array([0.00000, -0.21388, -0.27669, -0.33714, -0.39188, -0.51550, -0.42665,
                    -0.31966, 0.02526, 0.32235, 0.33075, 0.37095, 0.32627, 0.30353, 0.29012,
                    0.27485, 0.00000]),
}

# (Re, "u" or "v", index into Y or X) of tabulated points left out of the comparison (see
# the docstring).
LEFT_OUT = {(400, "v", 5)}
REYNOLDS = (100, 400, 1000)


def mask(Re: int, component: str) -> np.ndarray:
    """True at the tabulated points that take part in the comparison."""
    n = len(Y) if component == "u" else len(X)
    m = np.ones(n, dtype=bool)
    for re_, c, i in LEFT_OUT:
        if re_ == Re and c == component:
            m[i] = False
    return m


def compare(y: np.ndarray, u: np.ndarray, x: np.ndarray, v: np.ndarray, Re: int) -> dict:
    """Interpolate a computed centreline pair (y, u at x = 0.5) and (x, v at y = 0.5) to the
    tabulated points and return the differences from Tables I and II.

    Returns a dict with ``u`` and ``v`` (computed values at Y and X), ``du``, ``dv``
    (computed minus Ghia), ``max_du``, ``max_dv`` (largest |difference| over the compared
    points) and ``max_dv_all`` (over every point, the left-out one included).
    """
    if Re not in U:
        raise ValueError(f"no Ghia column for Re = {Re}; have {sorted(U)}")
    y, u, x, v = (np.asarray(a, dtype=float) for a in (y, u, x, v))
    oy, ox = np.argsort(y), np.argsort(x)
    ui = np.interp(Y, y[oy], u[oy])
    vi = np.interp(X, x[ox], v[ox])
    du, dv = ui - U[Re], vi - V[Re]
    return {"u": ui, "v": vi, "du": du, "dv": dv,
            "max_du": float(np.max(np.abs(du[mask(Re, "u")]))),
            "max_dv": float(np.max(np.abs(dv[mask(Re, "v")]))),
            "max_dv_all": float(np.max(np.abs(dv)))}
