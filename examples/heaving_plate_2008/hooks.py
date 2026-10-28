"""Prescribed motion of the heaving plate: the 2008 moveplate.m law.

Every target moves in x as x_T(t) = 0.2 + 0.05 sin(2 pi t/10) [m]; y stays where the point
started. The driver passes the time at which the forces are evaluated (t = n dt for the
pm1996 scheme, which is what the MATLAB loop did: move the plate at the current time, then
target forces, spreading, fluid solve, boundary move; t + dt/2 for the midpoint scheme).
The motion is a function of absolute time, never accumulated.
"""

import numpy as np

X0 = 0.2          # m
AMPLITUDE = 0.05  # m
PERIOD = 10.0     # s


def update_targets(t, dt, s):
    phase = 2.0 * np.pi * t / PERIOD
    s.targets.X[:, 0] = X0 + AMPLITUDE * np.sin(phase)
    s.targets.V[:, 0] = AMPLITUDE * (2.0 * np.pi / PERIOD) * np.cos(phase)
    s.targets.V[:, 1] = 0.0
