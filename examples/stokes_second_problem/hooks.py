"""The moving wall of Stokes' second problem: u = U cos(omega t), v = 0 on the bottom.

The MAC back end asks for the wall velocity at the coordinates x along the wall and the time
it needs (t + dt/2 for the half step, t + dt for the full step), so the motion is a function
of absolute time. The top wall has no hook and is a wall at rest. The constants come from
stokes_exact.py, which sits beside this file (the case directory is on sys.path while
hooks.py loads).
"""

import numpy as np

import stokes_exact as ex


def bc_bottom(x, t):
    x = np.asarray(x, dtype=float)
    return ex.U * np.cos(ex.OMEGA * t) * np.ones_like(x), np.zeros_like(x)
