"""Target motion and flowmeter of the valveless-pumping case (Jung & Peskin 2001).

update_targets: the law of section 3.2. On the driven third of the flexible segment,
x_d <= x <= x_d + L_d, the target of a wall point starting at (x0, y0) is

    Z(x0, t) = (x0, y0 +/- A(x0, t)),   A = A0 sin(2 pi t / T) sin(pi (x0 - x_d) / L_d),

with the plus sign on the inner (upper) wall and the minus sign on the outer (lower) wall:
A > 0 expands the target tube and A < 0 pinches it. Every other target stays where its point
started. The motion is a function of absolute time; the driver passes t + dt/2 (midpoint) or
t (pm1996), and calls this again at output times for the target-error diagnostic.

The period T, the amplitude A0 and x_d come from the structure name, which generate.py
writes as tube[_coarse]_T<period>_A<amp>_x<xd>, so this file holds no parameters of its own.
The flexible segment is found as the targets with the smallest stiffness; its two walls are
told apart by their height, and L_d is a third of its extent in x, which is the paper's 8 cm
up to one point spacing (1.3e-5 relative).

on_output: the paper's flowmeter (section 5.1, Fig. 8). At every output it appends to
<output_dir>/flux.csv the flux through a vertical section at the middle of the top straight
segment (x = 8 cm, between the walls at y = 5.4 and 6.6 cm, positive = clockwise = +x) and
through a horizontal section at the middle of the right semicircle (y = 4 cm, x from 13.4 to
14.6 cm, positive = clockwise = -y), plus the area between the two walls (shoelace formula;
the paper's Fig. 4-5 volume check, 34.28 cm^2 at rest). The velocity is sampled bilinearly
from the periodic node field of the FFT back end.
"""

from __future__ import annotations

import os
import re

import numpy as np

_NAME = re.compile(r"_T([0-9.eE+-]+)_A([0-9.eE+-]+)_x([0-9.eE+-]+)$")


def drive_parameters(name: str) -> tuple[float, float, float]:
    """(period, amplitude, x_drive) parsed from the structure name."""
    m = _NAME.search(name)
    if m is None:
        raise ValueError(f"structure name {name!r} does not end in _T<period>_A<amp>_x<xd>")
    return float(m.group(1)), float(m.group(2)), float(m.group(3))


def flexible_mask(s) -> np.ndarray:
    """Targets on the flexible segment: those with the smallest tether stiffness."""
    return s.targets.k <= s.targets.k.min() * (1.0 + 1e-9)


def update_targets(t, dt, s):
    period, amplitude, x_drive = drive_parameters(s.name)
    X0 = s.X0[s.targets.idx]
    flex = flexible_mask(s)
    x_flex = X0[flex, 0]
    drive_length = (x_flex.max() - x_flex.min()) / 3.0
    y_centre = 0.5 * (X0[flex, 1].min() + X0[flex, 1].max())
    driven = flex & (X0[:, 0] >= x_drive - 1e-9) & (X0[:, 0] <= x_drive + drive_length + 1e-9)
    s.targets.X[:] = X0
    s.targets.V[:] = 0.0
    phase = 2.0 * np.pi * t / period
    shape = np.sin(np.pi * (X0[driven, 0] - x_drive) / drive_length)
    sign = np.where(X0[driven, 1] > y_centre, 1.0, -1.0)     # inner wall up, outer wall down
    s.targets.X[driven, 1] = X0[driven, 1] + sign * amplitude * np.sin(phase) * shape
    s.targets.V[driven, 1] = sign * amplitude * (2.0 * np.pi / period) * np.cos(phase) * shape


# ------------------------------------------------------------------------------ flowmeter

def sample_nodes(field: np.ndarray, x: np.ndarray, y: np.ndarray, h: float) -> np.ndarray:
    """Bilinear interpolation of a periodic node field (values at (i h, j h)) at points (x, y)."""
    nx, ny = field.shape
    fx, fy = x / h, y / h
    i0 = np.floor(fx).astype(int)
    j0 = np.floor(fy).astype(int)
    ax, ay = fx - i0, fy - j0
    i0 %= nx
    j0 %= ny
    i1, j1 = (i0 + 1) % nx, (j0 + 1) % ny
    return ((1 - ax) * (1 - ay) * field[i0, j0] + ax * (1 - ay) * field[i1, j0]
            + (1 - ax) * ay * field[i0, j1] + ax * ay * field[i1, j1])


def line_flux(field: np.ndarray, h: float, x: np.ndarray, y: np.ndarray) -> float:
    """Trapezoid integral of the normal velocity component along a straight section."""
    values = sample_nodes(field, x, y, h)
    length = np.hypot(x[-1] - x[0], y[-1] - y[0])
    dl = length / (len(x) - 1)
    return float(dl * (np.sum(values) - 0.5 * (values[0] + values[-1])))


def split_walls(X0: np.ndarray) -> int:
    """Index where the outer wall's points begin: the two closed curves are stored one after
    the other, and the only jump longer than a point spacing is from the end of the inner
    wall (back near its start at x = 4, y = 2.6 cm) to the outer wall's start at y = 1.4 cm."""
    steps = np.hypot(*np.diff(X0, axis=0).T)
    return int(np.argmax(steps)) + 1


def sections(s, h: float):
    """The two flowmeter sections from the rest geometry: the top straight lies between the
    largest y of the inner and outer walls at mid-box x, the right arc between their largest
    x at mid-box y. Returns ((x, y) of the top section, (x, y) of the right section, n_in)."""
    X0 = s.X0
    n_in = split_walls(X0)
    inner, outer = X0[:n_in], X0[n_in:]
    x_mid = 0.5 * (X0[:, 0].min() + X0[:, 0].max())
    y_mid = 0.5 * (X0[:, 1].min() + X0[:, 1].max())
    n = max(32, int(8 * (outer[:, 1].max() - inner[:, 1].max()) / h))
    top = (np.full(n, x_mid), np.linspace(inner[:, 1].max(), outer[:, 1].max(), n))
    right = (np.linspace(inner[:, 0].max(), outer[:, 0].max(), n), np.full(n, y_mid))
    return top, right, n_in


def polygon_area(X: np.ndarray) -> float:
    """Shoelace area of a closed polygon given by its vertices in order."""
    x, y = X[:, 0], X[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def on_output(step, t, state):
    s, grid, sol = state.structure, state.grid, state.solver
    h = grid.dx
    (xt, yt), (xr, yr), n_in = sections(s, h)
    flux_top = line_flux(sol.u, h, xt, yt)            # +x through the top section: clockwise
    flux_right = -line_flux(sol.v, h, xr, yr)         # -y through the right section: clockwise
    # unwrap the walls about the box centre before the shoelace formula (periodic images)
    X = grid.min_image(s.X - np.array([0.5 * grid.Lx, 0.5 * grid.Ly])) + np.array([0.5 * grid.Lx, 0.5 * grid.Ly])
    area = polygon_area(X[n_in:]) - polygon_area(X[:n_in])
    path = os.path.join(state.output_dir, "flux.csv")
    mode = "w" if step == 0 else "a"
    with open(path, mode, encoding="utf-8") as fh:
        if step == 0:
            fh.write("step,t,flux_top,flux_right,area\n")
        fh.write(f"{step},{t!r},{flux_top!r},{flux_right!r},{area!r}\n")
