"""Prescribed motion of the rigid plate and vorticity snapshots.

update_targets puts every target at the pose of Jones et al. 2015 eqs. (2)-(3) at the
absolute time the driver passes (t + dt/2 for the midpoint scheme, the output time for the
target-error diagnostic; docs/numerics.md, time schemes). The plate frame is recovered from
the initial positions s.X0 (the positions read from plate.vertex), so the motion follows
whatever plate.vertex was read; nothing is accumulated step by step.

on_output saves a PNG of the vorticity with the plate every quarter cycle. The VTK output is
off in the input files because a history row every 1/800 cycle would mean thousands of
2 MB files; switch save_vtk on (and raise print_dump) in input2d if you want them.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_sibling(name: str):
    modname = f"flapping_plate_re64_{name}"
    if modname in sys.modules:
        return sys.modules[modname]
    spec = importlib.util.spec_from_file_location(modname, os.path.join(HERE, name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[modname] = module
    spec.loader.exec_module(module)
    return module


gen = _load_sibling("generate")

SNAPSHOTS_PER_CYCLE = 4
VORTICITY_SCALE = 8.0 * gen.UMAX / gen.CHORD     # colour limits of the snapshots, 1/s


def update_targets(t, dt, s):
    """Targets (and their velocities) at the prescribed pose at time t."""
    K = gen.stroke_from_initial_pose(s.X0)
    body = K.body_coordinates(s.X0, 0.0)
    idx = s.targets.idx
    s.targets.X[:] = K.positions(body, t)[idx]
    s.targets.V[:] = K.velocities(body, t)[idx]


def on_output(step, t, state):
    """vorticity.<step>.png every quarter cycle (only at output steps, so print_dump must
    divide STEPS_PER_CYCLE / SNAPSHOTS_PER_CYCLE, which generate.py arranges)."""
    n_cycle = int(round(gen.PERIOD / state.dt))
    every = max(n_cycle // SNAPSHOTS_PER_CYCLE, 1)
    if step % every:
        return
    from pyibm.io.plot import snapshot   # matplotlib only when needed

    os.makedirs(state.output_dir, exist_ok=True)
    s = state.structure
    phase = t / gen.PERIOD
    snapshot(state.grid, state.solver.vorticity(), X=s.X, targets=s.targets,
             path=os.path.join(state.output_dir, f"vorticity.{step:05d}.png"),
             vmax=VORTICITY_SCALE, title=f"t/T = {phase:.2f}, step {step}",
             location="nodes" if state.backend == "periodic_fft" else "centres")
