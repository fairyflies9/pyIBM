"""Travelling-wave curvature of the swimmer and its per-output diagnostics.

update_beams(t, dt, s) rewrites s.beams.curv so that the body's preferred shape at time t is
the Carling midline

    h(x, t) = r(t) A(x) sin(2 pi (x/lambda - t/T)),   A(x) = 0.125 L (x/L + 0.03125)/1.03125

(Carling, Williams & Bowtell 1998 eq. 2, with r(t) a smooth ramp over the first period so
the start is not impulsive), traced with arc length: node k sits at arc length s_k = k ds
along the graph of h, and the beam about it is given the cross product of two segments of
length ds that turn through the graph's curvature there,

    curv_k = -ds^2 sin(kappa(x(s_k), t) ds),   kappa = h_xx / (1 + h_x^2)^(3/2),

which is what pyibm's torsional beam compares (X_r - X_q) x (X_q - X_p) against
(pyibm.structures.beam_forces; for straight segments of length ds, c = ds^2 sin(theta_B -
theta_A)).
Everything is a function of absolute time, nothing is accumulated. The parameters come
from generate.py so they live in one place.

on_output appends one row per output to <output_dir>/swimmer.csv: the body centroid (head
image, so a body crossing the periodic edge is not torn), the centroid velocity (the mean
of the interpolated point velocities, which is exactly what moves the points), the heading
of the head-to-tail chord, the largest spring stretch, the body arc length and the RMS
curvature error against the prescribed curvature.
"""

from __future__ import annotations

import importlib.util
import os

import numpy as np


def _load_generate():
    """generate.py beside this file, under its own module name so that other examples'
    generate.py (also importable as plain 'generate') cannot shadow it in one session."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generate.py")
    spec = importlib.util.spec_from_file_location("pyibm_example_swimmer_generate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_gen = _load_generate()
L, WAVELENGTH, PERIOD = _gen.L, _gen.WAVELENGTH, _gen.PERIOD
AMPLITUDE_TAIL, RAMP_PERIODS = _gen.AMPLITUDE_TAIL, _gen.RAMP_PERIODS

SWIMMER_COLUMNS = ("step", "t", "xc", "yc", "uc", "vc", "heading", "max_stretch", "length",
                   "curv_rms_error", "curv_rms_prescribed")

_X_FINE = np.linspace(0.0, L, 2001)    # the x grid the arc length is tabulated on


def envelope(x: np.ndarray) -> np.ndarray:
    """A(x) = 0.125 L (x/L + 0.03125)/1.03125: linear from 0.0038 L at the head to 0.125 L at the tail."""
    return AMPLITUDE_TAIL * L * (x / L + 0.03125) / 1.03125


def ramp(t: float) -> float:
    """sin^2 ramp of the amplitude from 0 to 1 over RAMP_PERIODS periods."""
    tau = t / (RAMP_PERIODS * PERIOD)
    return 1.0 if tau >= 1.0 else float(np.sin(0.5 * np.pi * tau) ** 2)


def midline(x: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """h, h_x, h_xx of the ramped Carling midline at body coordinate x (0 = head) and time t."""
    k = 2.0 * np.pi / WAVELENGTH
    phase = k * x - 2.0 * np.pi * t / PERIOD
    A = envelope(x)
    dA = AMPLITUDE_TAIL / 1.03125
    r = ramp(t)
    sn, cs = np.sin(phase), np.cos(phase)
    h = r * A * sn
    h_x = r * (dA * sn + A * k * cs)
    h_xx = r * (2.0 * dA * k * cs - A * k * k * sn)
    return h, h_x, h_xx


def prescribed_curvature(s_nodes: np.ndarray, t: float) -> np.ndarray:
    """Curvature of the midline graph at the points of arc length s_nodes (measured from the head)."""
    _, h_x, h_xx = midline(_X_FINE, t)
    speed = np.sqrt(1.0 + h_x ** 2)
    arc = np.concatenate([[0.0], np.cumsum(0.5 * (speed[1:] + speed[:-1]) * np.diff(_X_FINE))])
    x_of_s = np.interp(s_nodes, arc, _X_FINE)        # arc is increasing; s beyond the end clamps
    _, h_x, h_xx = midline(x_of_s, t)
    return h_xx / (1.0 + h_x ** 2) ** 1.5


def prescribed_shape(s_nodes: np.ndarray, t: float) -> np.ndarray:
    """The preferred shape (N, 2) in body coordinates, head at the origin, traced with arc length."""
    _, h_x, _ = midline(_X_FINE, t)
    speed = np.sqrt(1.0 + h_x ** 2)
    arc = np.concatenate([[0.0], np.cumsum(0.5 * (speed[1:] + speed[:-1]) * np.diff(_X_FINE))])
    x_of_s = np.interp(s_nodes, arc, _X_FINE)
    h, _, _ = midline(x_of_s, t)
    return np.column_stack([x_of_s, h])


def update_beams(t: float, dt: float, s) -> None:
    """Set every beam's preferred cross product to that of the Carling midline at time t."""
    b = s.beams
    ds = float(s.ds[b.q[0]])
    s_nodes = b.q * ds                                 # the body is numbered head to tail
    kappa = prescribed_curvature(s_nodes, t)
    b.curv[:] = -ds * ds * np.sin(kappa * ds)


# ---------------------------------------------------------------------------- diagnostics

def unwrapped(X: np.ndarray, grid) -> np.ndarray:
    """Positions in the image of the head point, so a body straddling the box edge is whole."""
    return X[0] + grid.min_image(X - X[0])


def body_diagnostics(state) -> dict[str, float]:
    s, g = state.structure, state.grid
    X = unwrapped(s.X, g)
    U = state.coupling.interpolate_vector(state.solver.u, state.solver.v, s.X,
                                          state.solver.layout_u, state.solver.layout_v)
    seg = np.diff(X, axis=0)
    seglen = np.hypot(seg[:, 0], seg[:, 1])
    chord = X[-1] - X[0]
    b = s.beams
    A = X[b.r] - X[b.q]
    B = X[b.q] - X[b.p]
    c = A[:, 0] * B[:, 1] - A[:, 1] * B[:, 0]
    ds3 = float(s.ds[0]) ** 3
    err = (c - b.curv) / ds3
    return {
        "step": float(state.n), "t": state.t,
        "xc": float(X[:, 0].mean()), "yc": float(X[:, 1].mean()),
        "uc": float(U[:, 0].mean()), "vc": float(U[:, 1].mean()),
        "heading": float(np.arctan2(chord[1], chord[0])),
        "max_stretch": float(np.max(np.abs(seglen - s.springs.rest) / s.springs.rest)),
        "length": float(seglen.sum()),
        "curv_rms_error": float(np.sqrt(np.mean(err ** 2))),
        "curv_rms_prescribed": float(np.sqrt(np.mean((b.curv / ds3) ** 2))),
    }


def on_output(step: int, t: float, state) -> None:
    """Append the body diagnostics to <output_dir>/swimmer.csv (a fresh file at step 0)."""
    from pyibm.io.history import HistoryWriter

    path = os.path.join(state.output_dir, "swimmer.csv")
    if step == 0 and os.path.exists(path):
        os.remove(path)
    with HistoryWriter(path, SWIMMER_COLUMNS) as w:
        w.write(body_diagnostics(state))
