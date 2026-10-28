"""Geometry and kinematics of the two-winged 'fling' of Miller & Peskin 2005.

Reference: L. A. Miller and C. S. Peskin, "A computational fluid dynamics of 'clap and fling'
in the smallest insects", J. Exp. Biol. 208, 195-212 (2005). Everything here is written from
the Materials and methods of that paper:

- dimensionless time tau = t V / c (eq. 3), V the translational speed of the wing during
  the constant-velocity part of the stroke, c the chord;
- translational speed during acceleration (eq. 2):
  v(tau) = (V/2) {1 + cos[pi + pi (tau - tau_accel)/dtau_accel]}, then v = V. A fling
  half-stroke lasts tau = 6 with tau_accel = 0.86 and dtau_accel = 1.3 and never
  decelerates, which is 4.49 chords of travel ("about 4.5");
- angular velocity during rotation (eq. 5): omega(tau) = (omega_rot/2) {1 - cos[2 pi
  (tau - tau_turn)/dtau_rot]} with omega_rot = 2 dtheta V/(dtau_rot c) (eq. 6),
  dtheta = pi/4, dtau_rot = 1.74. Rotation starts at tau_turn = 0 and translation starts
  halfway through it ("the translational phase of the downstroke also began halfway
  through the second rotational phase": 0.86 = 1.74/2 to the paper's two digits);
- the wings start parallel at an angle of attack of 90 degrees, 1/6 chord apart (10 mesh
  widths of the paper's 1230^2 grid on a 1 m box), and rotate apart about their trailing
  (lower) edges to 45 degrees; the right wing is the mirror image of the left wing;
- Re = rho V c/mu with the Dickinson & Goetz sucrose solution (mu = 0.0235 N s/m^2,
  rho = 1000 kg/m^3, c = 0.05 m): V = 0.00375 m/s gives Re = 8, V = 0.06 m/s gives 128;
- the box is 1 m x 1 m with four walls of target points 30 mesh widths inside its edges
  (0.0244 m; 0.025 m here) to keep the velocity near zero at the periodic edges.

Lift and drag per wing (eqs. 15-16): C_L = 2 F_L/(rho c V^2), C_D = 2 F_D/(rho c V^2), with
F the force the fluid exerts on the wing (minus the force the wing puts on the fluid).
Lift is the +y component. Drag opposes the motion: +x on the left wing, -x on the right.

All positions are functions of absolute time t (docs/user-guide.md, hooks). The left wing
is the first N_PER_WING points, the right wing the next N_PER_WING; wall points, when
present, follow. Shared by generate.py, hooks.py, analyze.py and the tests.
"""

from __future__ import annotations

import re

import numpy as np

# Fluid and wing (Materials and methods: the Dickinson & Goetz parameters).
RHO = 1000.0          # kg/m^2 (2D: per unit span)
MU = 0.0235           # N s/m^2
C = 0.05              # chord [m]
GAP = C / 6.0         # distance between the wings at the start of fling [m]
LX = LY = 1.0         # box [m]
WALL_INSET = 0.025    # walls this far inside the box edges [m] (paper: 30 mesh widths)
Y_TE = LY / 2.0       # height of the trailing edges (the paper's x-axis) [m]
SPEEDS = {8: 0.00375, 128: 0.06}   # V [m/s] per Reynolds number

# Kinematics (eqs. 2-6 and the text below them).
TAU_TURN = 0.0        # rotation starts at the start of the half-stroke
DTAU_ROT = 1.74       # duration of rotation
DTHETA = np.pi / 4.0  # total rotation: 90 -> 45 degrees angle of attack
TAU_ACCEL = 0.86      # translation starts halfway through rotation
DTAU_ACCEL = 1.3      # duration of the translational acceleration
TAU_END = 6.0         # duration of the fling half-stroke
THETA0 = np.pi / 2.0  # initial angle of the left wing with the +x axis

# Discretisation: ds = c/51 = 9.8e-4 m is dx/2 on the 512^2 grid of input2d (the paper:
# 120 points per wing, ds = dx/2 on its 1230^2 grid).
N_PER_WING = 51
DS = C / N_PER_WING


def reynolds(V: float) -> float:
    """Re = rho V c/mu (eq. 1)."""
    return RHO * V * C / MU


def speed_of(name: str) -> float:
    """V for a structure called fling_re<Re>...: 'fling_re128' -> 0.06, 'fling_re8_box' -> 0.00375."""
    m = re.search(r"re(\d+)", name)
    if m is None or int(m.group(1)) not in SPEEDS:
        raise ValueError(f"cannot read the Reynolds number from the structure name {name!r}")
    return SPEEDS[int(m.group(1))]


def tau_of(t, V: float):
    """Dimensionless time tau = t V/c (eq. 3)."""
    return np.asarray(t, dtype=float) * V / C


# ---------------------------------------------------------------- rotation (eqs. 5-6)

def rotation_angle(tau):
    """Angle rotated through since tau_turn [rad]: the integral of eq. 5 in dimensionless
    time, (dtheta/dtau_rot) [s - (dtau_rot/2 pi) sin(2 pi s/dtau_rot)] with s = tau - tau_turn
    clipped to [0, dtau_rot]. Equals dtheta after rotation."""
    s = np.clip(np.asarray(tau, dtype=float) - TAU_TURN, 0.0, DTAU_ROT)
    return (DTHETA / DTAU_ROT) * (s - (DTAU_ROT / (2.0 * np.pi)) * np.sin(2.0 * np.pi * s / DTAU_ROT))


def angular_rate(tau):
    """d(angle)/d(tau) = (dtheta/dtau_rot) {1 - cos[2 pi (tau - tau_turn)/dtau_rot]} inside
    the rotation window, 0 outside (eq. 5 divided by V/c)."""
    tau = np.asarray(tau, dtype=float)
    s = tau - TAU_TURN
    inside = (s >= 0.0) & (s <= DTAU_ROT)
    return np.where(inside, (DTHETA / DTAU_ROT) * (1.0 - np.cos(2.0 * np.pi * s / DTAU_ROT)), 0.0)


def angular_velocity(t, V: float):
    """omega(t) [rad/s] = (V/c) d(angle)/d(tau); its maximum is omega_rot of eq. 6."""
    return (V / C) * angular_rate(tau_of(t, V))


# ---------------------------------------------------------------- translation (eqs. 2-3)

def speed_fraction(tau):
    """v/V: (1/2){1 + cos[pi + pi (tau - tau_accel)/dtau_accel]} during acceleration
    (eq. 2), 0 before, 1 after (no deceleration in the fling half-stroke)."""
    tau = np.asarray(tau, dtype=float)
    s = tau - TAU_ACCEL
    accel = 0.5 * (1.0 + np.cos(np.pi + np.pi * s / DTAU_ACCEL))
    return np.where(s <= 0.0, 0.0, np.where(s >= DTAU_ACCEL, 1.0, accel))


def travel(tau):
    """Distance translated in chords: the integral of speed_fraction over tau,
    (1/2)[s - (dtau_accel/pi) sin(pi s/dtau_accel)] during acceleration and
    dtau_accel/2 + (s - dtau_accel) after. 4.49 chords at tau = 6."""
    tau = np.asarray(tau, dtype=float)
    s = tau - TAU_ACCEL
    sc = np.clip(s, 0.0, DTAU_ACCEL)
    d = 0.5 * (sc - (DTAU_ACCEL / np.pi) * np.sin(np.pi * sc / DTAU_ACCEL))
    return d + np.maximum(s - DTAU_ACCEL, 0.0)


# ---------------------------------------------------------------- wing points

def chord_stations() -> np.ndarray:
    """Arc length of each wing point from the trailing edge: (k + 1/2) ds, k = 0..N-1,
    so that the ds weights sum to the chord exactly."""
    return (np.arange(N_PER_WING) + 0.5) * DS


def wing_positions(t: float, V: float) -> tuple[np.ndarray, np.ndarray]:
    """Positions (N_PER_WING, 2) of the left and right wing at absolute time t.

    Left wing: trailing edge at (Lx/2 - gap/2 - c travel, Y_TE), direction
    (cos theta, sin theta) with theta = pi/2 + rotation_angle (90 -> 135 degrees, the
    leading edge tipping away from the other wing). Right wing: mirror image in x = Lx/2.
    """
    tau = tau_of(t, V)
    th = THETA0 + rotation_angle(tau)
    d = C * travel(tau)
    s = chord_stations()
    xl = LX / 2.0 - GAP / 2.0 - d
    xr = LX / 2.0 + GAP / 2.0 + d
    left = np.column_stack([xl + s * np.cos(th), Y_TE + s * np.sin(th)])
    right = np.column_stack([xr - s * np.cos(th), Y_TE + s * np.sin(th)])
    return left, right


def wing_velocities(t: float, V: float) -> tuple[np.ndarray, np.ndarray]:
    """Velocities (N_PER_WING, 2) of the left and right wing points at time t: the trailing
    edge speed -/+ V v(tau)/V in x plus the rotation s omega (-sin theta, cos theta)."""
    tau = tau_of(t, V)
    th = THETA0 + rotation_angle(tau)
    u_te = V * speed_fraction(tau)
    om = angular_velocity(t, V)
    s = chord_stations()
    left = np.column_stack([-u_te - s * om * np.sin(th), s * om * np.cos(th)])
    right = np.column_stack([u_te + s * om * np.sin(th), s * om * np.cos(th)])
    return left, right


def wall_points(ds: float = DS) -> np.ndarray:
    """Points (M, 2) along the square WALL_INSET inside the box, spaced ds along the
    perimeter (bottom, right, top, left; counter-clockwise from the bottom-left corner),
    at arc length (k + 1/2) ds so no point sits on a corner."""
    side = LX - 2.0 * WALL_INSET
    perimeter = 4.0 * side
    m = int(round(perimeter / ds))
    a = (np.arange(m) + 0.5) * (perimeter / m)
    lo, hi = WALL_INSET, LX - WALL_INSET
    x = np.where(a < side, lo + a,
        np.where(a < 2 * side, hi,
        np.where(a < 3 * side, hi - (a - 2 * side), lo)))
    y = np.where(a < side, lo,
        np.where(a < 2 * side, lo + (a - side),
        np.where(a < 3 * side, hi, hi - (a - 3 * side))))
    return np.column_stack([x, y])


def initial_points(walls: bool, ds: float = DS) -> np.ndarray:
    """The .vertex rows: left wing, right wing, then the walls when walls is True."""
    left, right = wing_positions(0.0, SPEEDS[128])   # t = 0 is the same shape for any V
    parts = [left, right]
    if walls:
        parts.append(wall_points(ds))
    return np.concatenate(parts, axis=0)


def force_coefficients(fx: np.ndarray, fy: np.ndarray, V: float, side: str
                       ) -> tuple[np.ndarray, np.ndarray]:
    """(C_L, C_D) from the force (fx, fy) the fluid exerts on one wing (eqs. 15-16):
    C = 2 F/(rho c V^2); lift is fy, drag is +fx for the left wing and -fx for the right."""
    q = 0.5 * RHO * C * V ** 2
    sign = 1.0 if side == "left" else -1.0
    return np.asarray(fy) / q, sign * np.asarray(fx) / q
