"""The lid of the cavity and the centreline output.

``bc_top`` gives the velocity on the top wall: the lid slides in +x at U_LID everywhere,
corners included (Ghia, Ghia & Shin 1982 use the same singular corners). The other three
walls have no hook and are no-slip walls at rest.

``on_output`` writes the two centreline profiles at every output, overwriting, so
analyze.py can compare the final ones with Tables I and II of the paper without averaging
the MAC faces to the cell centres: u lives on the vertical faces x = i dx, and face Nx/2 is
exactly x = Lx/2; v lives on the horizontal faces y = j dy, and face Ny/2 is y = Ly/2.
"""

import os

import numpy as np

U_LID = 1.0   # m/s


def bc_top(x, t):
    """(u, v) on the lid at the coordinates x and time t: uniform sliding, no normal flow."""
    return U_LID * np.ones_like(x), np.zeros_like(x)


def on_output(step, t, state):
    """<output_dir>/u_centreline.csv holds (y, u) at x = Lx/2 from the bottom wall (u = 0) to
    the lid (u = U_LID); v_centreline.csv holds (x, v) at y = Ly/2 from wall to wall (v = 0)."""
    g, s = state.grid, state.solver
    y = np.concatenate([[0.0], g.yc, [g.Ly]])
    u = np.concatenate([[0.0], s.u[g.Nx // 2], [bc_top(np.array([0.5 * g.Lx]), t)[0][0]]])
    x = np.concatenate([[0.0], g.xc, [g.Lx]])
    v = np.concatenate([[0.0], s.v[:, g.Ny // 2], [0.0]])
    os.makedirs(state.output_dir, exist_ok=True)
    np.savetxt(os.path.join(state.output_dir, "u_centreline.csv"), np.column_stack([y, u]),
               header="y,u", comments="", delimiter=",", fmt="%.17g")
    np.savetxt(os.path.join(state.output_dir, "v_centreline.csv"), np.column_stack([x, v]),
               header="x,v", comments="", delimiter=",", fmt="%.17g")
