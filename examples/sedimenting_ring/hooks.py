"""Gravity ramp and per-output diagnostics of the sedimenting ring (no wall data: the four
walls are no-slip and at rest, which is the MAC back end's default for "dirichlet").

update_targets(t, dt, s) moves the targets of the rigid reference (input2d_rigid) down at
U_RIGID with the same ramp as gravity; the elastic ring has no targets and the hook returns.

update_masses(t, dt, s) sets s.gravity = (0, -g r(t)) with a sin^2 ramp r(t) from 0 to 1
over T_RAMP seconds. Switching the full weight on at t = 0 rings the ghost-tether mode
(frequency sqrt(k_mass/m), undamped in the symplectic Euler step of
pyibm.structures.advance_masses) with an amplitude of the order of the weight itself, which
then shows up aliased in fy_body; the ramp keeps that mode quiet. The ramp is a function of
absolute time.

on_output appends one row per output to <output_dir>/ring.csv: the centroid of the ring and
its velocity (the mean of the interpolated point velocities, which is what moves the
points), the enclosed area from the shoelace formula and the perimeter, the largest spring
stretch, the largest ghost-to-point distance in cells (how far the mass points lag their
tethers), and the distance from the lowest point to the bottom wall in cells (the stencil
raises if it reaches 2 cells, so Tfinal in generate.py keeps it above that).
"""

from __future__ import annotations

import importlib.util
import os

import numpy as np


def _load_generate():
    """generate.py beside this file, under its own module name so that other examples'
    generate.py (also importable as plain 'generate') cannot shadow it in one session."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generate.py")
    spec = importlib.util.spec_from_file_location("pyibm_example_sedimenting_ring_generate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_gen = _load_generate()
GRAVITY, T_RAMP, U_RIGID = _gen.GRAVITY, _gen.T_RAMP, _gen.U_RIGID


def ramp(t: float) -> float:
    """sin^2 ramp from 0 at t = 0 to 1 at t = T_RAMP, 1 afterwards."""
    tau = t / T_RAMP
    return 1.0 if tau >= 1.0 else float(np.sin(0.5 * np.pi * tau) ** 2)


def ramp_integral(t: float) -> float:
    """Integral of ramp from 0 to t: t/2 - (T_RAMP/2 pi) sin(pi t/T_RAMP) during the ramp."""
    if t >= T_RAMP:
        return 0.5 * T_RAMP + (t - T_RAMP)
    return 0.5 * t - (T_RAMP / (2.0 * np.pi)) * float(np.sin(np.pi * t / T_RAMP))


def update_targets(t: float, dt: float, s) -> None:
    """Rigid reference only: every target moves down at U_RIGID ramp(t) from where it started."""
    if s.targets is None:
        return
    s.targets.X[:, 0] = s.X0[s.targets.idx, 0]
    s.targets.X[:, 1] = s.X0[s.targets.idx, 1] - U_RIGID * ramp_integral(t)
    s.targets.V[:, 0] = 0.0
    s.targets.V[:, 1] = -U_RIGID * ramp(t)


def update_masses(t: float, dt: float, s) -> None:
    """Gravity on the mass points, ramped in over T_RAMP (a function of absolute time)."""
    s.gravity = (0.0, -GRAVITY * ramp(t))


RING_COLUMNS = ("step", "t", "xc", "yc", "uc", "vc", "area", "perimeter", "max_stretch",
                "ghost_gap_over_dx", "bottom_gap_over_dx")


def polygon_area(X: np.ndarray) -> float:
    """Shoelace area of the closed polygon through the points in order (positive counter-clockwise)."""
    x, y = X[:, 0], X[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def ring_diagnostics(state) -> dict[str, float]:
    s, g = state.structure, state.grid
    X = s.X
    U = state.coupling.interpolate_vector(state.solver.u, state.solver.v, X,
                                          state.solver.layout_u, state.solver.layout_v)
    seg = np.roll(X, -1, axis=0) - X
    seglen = np.hypot(seg[:, 0], seg[:, 1])
    if s.masses is not None:
        gap = s.masses.Y - X[s.masses.idx]
        ghost = float(np.max(np.hypot(gap[:, 0], gap[:, 1])) / g.dx)
        stretch = float(np.max(np.abs(seglen - s.springs.rest) / s.springs.rest))
    else:                                   # the rigid reference: targets, no springs
        ghost, stretch = 0.0, float(np.max(np.abs(seglen / seglen[0] - 1.0)))
    return {
        "step": float(state.n), "t": state.t,
        "xc": float(X[:, 0].mean()), "yc": float(X[:, 1].mean()),
        "uc": float(U[:, 0].mean()), "vc": float(U[:, 1].mean()),
        "area": polygon_area(X), "perimeter": float(seglen.sum()),
        "max_stretch": stretch,
        "ghost_gap_over_dx": ghost,
        "bottom_gap_over_dx": float(X[:, 1].min() / g.dy),
    }


def on_output(step: int, t: float, state) -> None:
    """Append the ring diagnostics to <output_dir>/ring.csv (a fresh file at step 0)."""
    from pyibm.io.history import HistoryWriter

    path = os.path.join(state.output_dir, "ring.csv")
    if step == 0 and os.path.exists(path):
        os.remove(path)
    with HistoryWriter(path, RING_COLUMNS) as w:
        w.write(ring_diagnostics(state))
