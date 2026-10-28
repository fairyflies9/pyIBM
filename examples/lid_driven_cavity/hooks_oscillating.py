"""The oscillating lid: U(t) = U_LID sin(2 pi t / PERIOD) in +x on the top wall, as a
function of absolute time; the other walls are at rest.

input2d_oscillating names this file in its Hooks section (``Hooks { file =
"hooks_oscillating.py" }``), so the oscillating case shares the directory with the steady
ones, whose input files use the default hooks.py. The driver evaluates the hook at the time
levels the scheme needs (t for the advection data, t + dt/2 and t + dt for the half and full
steps), so nothing is accumulated here.
"""

import numpy as np

U_LID = 1.0     # m/s
PERIOD = 1.0    # s


def lid_speed(t):
    return U_LID * np.sin(2.0 * np.pi * t / PERIOD)


def bc_top(x, t):
    return lid_speed(t) * np.ones_like(x), np.zeros_like(x)
