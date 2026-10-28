"""The exact Taylor-Green field, and the hook that starts every run from it.

On the doubly periodic box [0, L)^2 the vortex

    u = -cos(2 pi x/L) sin(2 pi y/L) exp(-8 pi^2 nu t/L^2),
    v =  sin(2 pi x/L) cos(2 pi y/L) exp(-8 pi^2 nu t/L^2)

solves the Navier-Stokes equations. ``initial_velocity`` is called once when the Simulation
is built, with the places where u and v live: the nodes on the periodic_fft back end, the u
and v faces on the mac back end. It writes the exact field at t = 0 there, so
``python -m pyibm run examples/taylor_green`` starts from the vortex on either back end.
analyze.py uses ``exact_velocity`` from here for the error at the final time.
"""

import numpy as np


def exact_velocity(X, Y, t, nu, L=1.0):
    """(u, v) of the Taylor-Green vortex at time t on the box of side L."""
    a = np.exp(-8.0 * np.pi**2 * nu * t / L**2)
    k = 2.0 * np.pi / L
    return -a * np.cos(k * X) * np.sin(k * Y), a * np.sin(k * X) * np.cos(k * Y)


def initial_velocity(grid, layout_u, layout_v):
    """The exact field at t = 0 on the u and v layouts of the solver (the decay factor is 1
    at t = 0, so the viscosity does not enter)."""
    if grid.Lx != grid.Ly:
        raise ValueError(f"the Taylor-Green example needs a square box, got Lx = {grid.Lx}, "
                         f"Ly = {grid.Ly}")
    Xu, Yu = np.meshgrid(*layout_u.coordinates(), indexing="ij")
    Xv, Yv = np.meshgrid(*layout_v.coordinates(), indexing="ij")
    u = exact_velocity(Xu, Yu, 0.0, 0.0, grid.Lx)[0]
    v = exact_velocity(Xv, Yv, 0.0, 0.0, grid.Lx)[1]
    return u, v
