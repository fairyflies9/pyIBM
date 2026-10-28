"""Prescribed motion of the forced circle's targets (Fovargue 2012 section 5.4.0.1).

T(theta, t) = (0.5 + a(t) cos theta, 0.5 + b(t) sin theta) with a = c (1 + eps sin 2 pi t)
and b = c/(1 + eps sin 2 pi t), for the target points on the right half of the circle. The
angle of each target point is recovered from its initial position (s.X0), so the motion is a
function of absolute time only; the driver passes t + dt/2 for the midpoint scheme and calls
the hook again at output times for the target-error diagnostic.
"""

import numpy as np

C_AXIS = 0.2
EPS = 0.4
PERIOD = 1.0
CENTRE = (0.5, 0.5)


def update_targets(t, dt, s):
    tg = s.targets
    X0 = s.X0[tg.idx]
    th = np.arctan2(X0[:, 1] - CENTRE[1], X0[:, 0] - CENTRE[0])
    phase = 2.0 * np.pi * t / PERIOD
    g = 1.0 + EPS * np.sin(phase)
    dg = EPS * (2.0 * np.pi / PERIOD) * np.cos(phase)
    a, b = C_AXIS * g, C_AXIS / g
    tg.X[:, 0] = CENTRE[0] + a * np.cos(th)
    tg.X[:, 1] = CENTRE[1] + b * np.sin(th)
    tg.V[:, 0] = C_AXIS * dg * np.cos(th)
    tg.V[:, 1] = -C_AXIS * dg / g ** 2 * np.sin(th)
