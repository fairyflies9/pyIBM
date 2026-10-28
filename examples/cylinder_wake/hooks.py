"""Boundary data, the shedding trigger and the vorticity snapshots of the cylinder-wake case.

Free stream: the inflow (left) and the two far-field sides (bottom, top) carry (U(t), 0).
U(t) rises from 0 to U over T_RAMP with the quintic smoothstep, which has zero first and
second derivatives at both ends. A ramp with a kink (a linear ramp, or an impulsive start)
puts a jump into the added-mass force on the cylinder, and that jump rings the explicit
marker-tether system for the rest of the run: the drag then carries a step-to-step
oscillation several times the drag itself, which is what the stored histories of an earlier
build of this example showed. The right side is outflow and needs no data. The functions get
the driver's coordinate arrays and the time of the evaluation (t, t + dt/2 or t + dt) and
must be vectorised (docs/user-guide.md, hooks).

Trigger: a symmetric wake stays symmetric on a symmetric grid for a very long time, since
the symmetric flow is an exact solution of the discrete equations and roundoff is all that
breaks it. Lai & Peskin (J. Comput. Phys. 160 (2000) 705-719, section 4) seeded the asymmetry
with a random perturbation of the initial velocity. Here it is a smooth vertical body force
in the near wake, a Gaussian bump of width KICK_SIGMA centred KICK_X_OFFSET downstream of the
cylinder, switched on with a sin^2 envelope between KICK_T0 and KICK_T1 and never again. The
cylinder itself is not touched, so the target error measures only the fluid loading. The
kick only decides when the vortex street starts (README.md); the driver leaves it out of
fx_body, fy_body.
Everything is a function of absolute time, so a continued run sees the same forcing.

Snapshots: on_output saves a vorticity PNG every SNAPSHOT_INTERVAL time units (the inputs
keep save_vtk off because a history fine enough for the Strouhal number would mean
thousands of VTK dumps).
"""

from __future__ import annotations

import os

import numpy as np

U = 1.0                    # m/s, free-stream speed (the same U as generate.py)
D = 1.0                    # m, cylinder diameter (the same D as generate.py)
XC, YC = 4.0, 4.0          # m, cylinder centre (the same as generate.py)
T_RAMP = 2.0               # s, the free stream rises 0 -> U over this time (quintic smoothstep)

KICK_T0, KICK_T1 = 3.0, 6.0   # s, the trigger force acts between these times (sin^2 envelope)
KICK_AMPLITUDE = 0.5       # N/m^2, peak vertical force density, in units of rho U^2 / D
KICK_X_OFFSET = 1.5        # in diameters downstream of the centre
KICK_SIGMA = 0.5           # in diameters, Gaussian half-width of the force patch

SNAPSHOT_INTERVAL = 10.0   # s between vorticity PNGs; 0 disables them
SNAPSHOT_VMAX = 2.0        # 1/s, colour limits of the vorticity plots


def smoothstep5(x):
    """6x^5 - 15x^4 + 10x^3 clamped to [0, 1]: C^2 at both ends."""
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    return x * x * x * (x * (6.0 * x - 15.0) + 10.0)


def ramp(t: float) -> float:
    """Free-stream speed at time t: U smoothstep5(t / T_RAMP)."""
    return float(U * smoothstep5(t / T_RAMP))


def free_stream(coord, t):
    """(u, v) = (ramp(t), 0) at every point of a side."""
    c = np.asarray(coord, dtype=float)
    return np.full_like(c, ramp(t)), np.zeros_like(c)


bc_left = free_stream
bc_bottom = free_stream
bc_top = free_stream


def kick_envelope(t: float) -> float:
    """sin^2(pi (t - T0)/(T1 - T0)) inside the kick window, 0 outside."""
    if KICK_T0 < t < KICK_T1:
        return float(np.sin(np.pi * (t - KICK_T0) / (KICK_T1 - KICK_T0)) ** 2)
    return 0.0


def external_force(t, grid, u, v):
    """Vertical body force on the v faces: amplitude times the envelope times a Gaussian
    in the near wake; zero arrays outside the kick window. Shapes are those of the MAC
    u and v layouts ((Nx+1, Ny) and (Nx, Ny+1))."""
    Eu = np.zeros((len(grid.xu), len(grid.yu)))
    Ev = np.zeros((len(grid.xv), len(grid.yv)))
    a = kick_envelope(t)
    if a > 0.0:
        X, Y = grid.meshgrid("v")
        r2 = (X - (XC + KICK_X_OFFSET * D)) ** 2 + (Y - YC) ** 2
        Ev += KICK_AMPLITUDE * a * np.exp(-0.5 * r2 / (KICK_SIGMA * D) ** 2)
    return Eu, Ev


def on_output(step, t, state):
    """Vorticity PNG every SNAPSHOT_INTERVAL time units (not at t = 0)."""
    if SNAPSHOT_INTERVAL <= 0 or step == 0:
        return
    k = round(t / SNAPSHOT_INTERVAL)
    if abs(t - k * SNAPSHOT_INTERVAL) > 0.5 * state.dt:
        return
    from pyibm.io.plot import snapshot   # matplotlib, only when a picture is due

    path = os.path.join(state.output_dir, f"vorticity.{step:05d}.png")
    snapshot(state.grid, state.solver.vorticity(), X=state.structure.X, path=path,
             vmax=SNAPSHOT_VMAX, title=f"vorticity, t = {t:.1f} D/U", location="centres")
