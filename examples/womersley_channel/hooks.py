"""Boundary data and body force of the Womersley channel.

``bc_left`` prescribes the exact Womersley profile (womersley_exact.py eq. 1) as the inflow
on the left wall. The solver hands it the coordinates along the wall (the cell centres for
u, the faces for v) and the time it needs, so the profile is a function of absolute time;
the channel height comes off the coordinates (``channel_height``), which is what lets one
hooks.py serve both the alpha = 3 and the alpha = 10 inputs.

``external_force`` drives the periodic variants with the uniform body force
K cos(omega t) in x, the pressure gradient the inflow runs have to generate for themselves.
In the inflow/outflow channel it returns zero: the driver calls the hook whenever it is
defined, so the switch is the grid's periodicity. The constants come from
womersley_exact.py, which sits beside this file (the case directory is on sys.path while
hooks.py loads).
"""

import numpy as np

import womersley_exact as ex


def bc_left(y, t):
    y = np.asarray(y, dtype=float)
    return ex.velocity(y, t, ex.channel_height(y)), np.zeros_like(y)


def external_force(t, grid, u, v):
    if not grid.periodic_x:
        return np.zeros_like(u), np.zeros_like(v)
    return ex.pressure_gradient(t) * np.ones_like(u), np.zeros_like(v)
