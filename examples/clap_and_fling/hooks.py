"""Prescribed wing motion and per-wing force output for the clap-and-fling example.

update_targets moves the targets of the two wings along the fling kinematics of Miller &
Peskin 2005 (fling_kinematics.py) at the absolute time the driver passes (t + dt/2 for the
midpoint scheme; docs/numerics.md, time schemes), and sets their velocities, which the target
damping of the Re 128 inputs pulls the wing points towards (eq. 11 of the paper). The wall
targets, when present, stay where they were read, at rest. The translational speed V comes
from the structure name (fling_re128, fling_re8, ...), so one hooks.py serves every variant.

on_output appends one row per output to <output_dir>/wing_forces.csv with the force the
fluid exerts on each wing, -sum(f ds) over that wing's points (state.f is the Lagrangian
force density of the last step, N/m), and the lift and drag coefficients of eqs. 15-16.
That force belongs to the time state.t_force (half a step before the output time t for the
midpoint scheme, 0 at step 0), so the row records t_force too and tau and the fraction of
the stroke are those of t_force. The file is rewritten at step 0 so a rerun starts clean.
"""

from __future__ import annotations

import os

import numpy as np

import fling_kinematics as kin   # beside this file; the driver puts the case dir on sys.path

N = kin.N_PER_WING
COLUMNS = ("step", "t", "t_force", "tau", "fraction_of_stroke",
           "fx_left", "fy_left", "fx_right", "fy_right",
           "CL_left", "CD_left", "CL_right", "CD_right")


def update_targets(t, dt, s):
    V = kin.speed_of(s.name)
    left, right = kin.wing_positions(t, V)
    vl, vr = kin.wing_velocities(t, V)
    s.targets.X[:N] = left
    s.targets.X[N:2 * N] = right
    s.targets.V[:N] = vl
    s.targets.V[N:2 * N] = vr


def on_output(step, t, state):
    s = state.structure
    if state.f is None:
        return
    V = kin.speed_of(s.name)
    fds = state.f * s.ds[:, None]                   # force on the fluid per point [N]
    f_left = -fds[:N].sum(axis=0)                   # force on the wing = minus that
    f_right = -fds[N:2 * N].sum(axis=0)
    cl_l, cd_l = kin.force_coefficients(f_left[0], f_left[1], V, "left")
    cl_r, cd_r = kin.force_coefficients(f_right[0], f_right[1], V, "right")
    t_force = float(state.t_force)
    tau = float(kin.tau_of(t_force, V))
    row = (step, t, t_force, tau, tau / kin.TAU_END,
           f_left[0], f_left[1], f_right[0], f_right[1],
           float(cl_l), float(cd_l), float(cl_r), float(cd_r))
    os.makedirs(state.output_dir, exist_ok=True)
    path = os.path.join(state.output_dir, "wing_forces.csv")
    with open(path, "w" if step == 0 else "a", encoding="ascii") as fh:
        if step == 0:
            fh.write(",".join(COLUMNS) + "\n")
        fh.write(",".join(repr(float(v)) for v in row) + "\n")
