"""Rigid-plate kinematics of the idealised hovering strokes of Jones et al. 2015.

Jones, Laurenza, Hedrick, Griffith & Miller, "Lift vs. drag based mechanisms for vertical
force production in the smallest flying insects", J. Theor. Biol. 384 (2015) 105-120.
Section 2.1 prescribes the centre of the chord and the angle of attack as

    [x(t), y(t)] = (A0/2) (1 + cos(2 pi f t)) [cos(beta), sin(beta)]        eq. (2)
    alpha(t)     = alpha_0 + B sin(2 pi f t)                                 eq. (3)

with A0 the peak-to-peak translation amplitude, f the frequency, beta the stroke-plane
angle, alpha_0 the mean angle of attack and B the rotation amplitude. The chord moves back
and forth along a line tilted by beta; alpha is the angle between the chord and that line.
Here the same motion is written about the midpoint of the stroke, s(t) = (A0/2) cos(omega t)
with omega = 2 pi / T, so the midpoint can be put at the centre of the box. The plate
rotates about its midchord. At t = 0 the plate sits at the upper end of the stroke with
alpha = alpha_0 and starts the downstroke (0 < t < T/2); the upstroke is T/2 < t < T.

Alpha can be measured from either direction along the stroke plane, and we take the direction
from the paper's Fig. 2: the vector from the trailing edge to the leading edge makes the
angle alpha, counterclockwise, with the
stroke direction (cos beta, sin beta). In the laboratory frame the leading-edge-to-
trailing-edge vector then points at theta = beta + alpha + pi. Fig. 2a (the horizontal
stroke, beta = 0, alpha = 135 deg at mid-downstroke) settles it: the chord there runs from
the leading edge at the upper left to the trailing edge at the lower right, i.e. at 315 deg
= 0 + 135 + 180, and the leading edge leads both half strokes. For the tilted stroke of
Fig. 2c the same rule puts the plate nearly perpendicular to its path on the downstroke
(alpha = 105 deg at mid-stroke) and feathered to 15 deg on the upstroke, as that panel
shows, and the force histories have the paper's signatures (C_H peaks before C_V in the
downstroke, C_H < 0 through the upstroke). The mirror image, theta = beta - alpha, in which
the trailing edge leads the upstroke and the plate pushes fluid upward there, is available
as leading_edge_forward = False; in a coarse check (8 cells per chord, two cycles) it gave
about 0.4 of the cycle-mean vertical force of the paper's convention and a positive mean
C_H (see the README).

Derived quantities, eq. (10) and section 2.2: the peak translation speed is
U_max = pi f A0 = pi A0 / T, the Reynolds number is Re = rho U_max c / mu, and forces are
made dimensionless with 0.5 rho U_rms^2 c where U_rms = U_max / sqrt(2) is the root mean
square of the centre speed over a cycle.

Everything is a function of absolute time, so the driver can evaluate it at whatever time
level the scheme needs (docs/user-guide.md, hooks). Written from the equations above; nothing
here comes from another code.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class StrokeKinematics:
    """The stroke of eqs. (2)-(3) about a midpoint (x_mid, y_mid).

    stroke     : A0, peak-to-peak translation of the midchord along the stroke plane [m]
    period     : T = 1/f [s]
    beta       : stroke-plane angle from the x axis [rad]
    alpha_mean : alpha_0, mean angle of attack [rad]
    alpha_amp  : B, rotation amplitude [rad]
    x_mid, y_mid : midpoint of the stroke [m]
    phase      : phase of the rotation relative to the translation [rad]; 0 in the paper
    leading_edge_forward : True (the paper's Fig. 2c) puts the leading-edge-to-trailing-edge
                 vector at theta = beta + alpha + pi; False is the mirror image
                 theta = beta - alpha (see the module docstring)
    """

    stroke: float
    period: float
    beta: float
    alpha_mean: float
    alpha_amp: float
    x_mid: float = 0.0
    y_mid: float = 0.0
    phase: float = 0.0
    leading_edge_forward: bool = True

    # ------------------------------------------------------------------ scalars
    @property
    def omega(self) -> float:
        """2 pi f."""
        return 2.0 * np.pi / self.period

    @property
    def u_max(self) -> float:
        """Peak speed of the midchord, pi f A0 (eq. 10)."""
        return np.pi * self.stroke / self.period

    @property
    def u_rms(self) -> float:
        """Root mean square of the midchord speed over a cycle, U_max / sqrt(2)."""
        return self.u_max / np.sqrt(2.0)

    @property
    def direction(self) -> np.ndarray:
        """Unit vector along the stroke plane, (cos beta, sin beta)."""
        return np.array([np.cos(self.beta), np.sin(self.beta)])

    @property
    def normal_up(self) -> np.ndarray:
        """Unit normal to the stroke plane with a non-negative y component, (-sin beta, cos beta)."""
        n = np.array([-np.sin(self.beta), np.cos(self.beta)])
        return n if n[1] >= 0.0 else -n

    # ------------------------------------------------------------------ translation
    def s(self, t):
        """Position of the midchord along the stroke plane, (A0/2) cos(omega t) (eq. 2 shifted
        by A0/2 so the stroke is centred on the midpoint)."""
        return 0.5 * self.stroke * np.cos(self.omega * np.asarray(t, dtype=float))

    def s_dot(self, t):
        """d s / d t = -(A0/2) omega sin(omega t); its peak magnitude is u_max."""
        return -0.5 * self.stroke * self.omega * np.sin(self.omega * np.asarray(t, dtype=float))

    def centre(self, t) -> np.ndarray:
        """Midchord position: midpoint + s(t) (cos beta, sin beta). Shape (2,) for scalar t,
        (len(t), 2) for an array."""
        s = self.s(t)
        return np.stack([self.x_mid + s * np.cos(self.beta), self.y_mid + s * np.sin(self.beta)], axis=-1)

    def centre_velocity(self, t) -> np.ndarray:
        """Midchord velocity, s_dot(t) (cos beta, sin beta)."""
        sd = self.s_dot(t)
        return np.stack([sd * np.cos(self.beta), sd * np.sin(self.beta)], axis=-1)

    def motion_direction(self, t) -> np.ndarray:
        """Unit vector along the instantaneous motion, sign(s_dot) (cos beta, sin beta); zero
        at the stroke reversals where s_dot = 0."""
        sign = np.sign(self.s_dot(t))
        return np.stack([sign * np.cos(self.beta), sign * np.sin(self.beta)], axis=-1)

    def stroke_phase(self, t):
        """Fraction of the cycle, (t mod T) / T: the downstroke is [0, 1/2), the upstroke [1/2, 1)."""
        return np.mod(np.asarray(t, dtype=float) / self.period, 1.0)

    # ------------------------------------------------------------------ rotation
    def alpha(self, t):
        """Angle of attack alpha_0 + B sin(omega t + phase) (eq. 3)."""
        return self.alpha_mean + self.alpha_amp * np.sin(self.omega * np.asarray(t, dtype=float) + self.phase)

    def alpha_dot(self, t):
        return self.alpha_amp * self.omega * np.cos(self.omega * np.asarray(t, dtype=float) + self.phase)

    def theta(self, t):
        """Laboratory angle of the leading-edge-to-trailing-edge vector: beta + alpha(t) + pi
        (Fig. 2c), or beta - alpha(t) with leading_edge_forward = False."""
        if self.leading_edge_forward:
            return self.beta + self.alpha(t) + np.pi
        return self.beta - self.alpha(t)

    def theta_dot(self, t):
        return self.alpha_dot(t) if self.leading_edge_forward else -self.alpha_dot(t)

    def rotation(self, t: float) -> np.ndarray:
        """R(theta) = [[cos, -sin], [sin, cos]] for scalar t."""
        th = float(self.theta(t))
        c, s = np.cos(th), np.sin(th)
        return np.array([[c, -s], [s, c]])

    # ------------------------------------------------------------------ rigid body
    def positions(self, body: np.ndarray, t: float) -> np.ndarray:
        """Laboratory positions X = centre(t) + R(theta(t)) b of body points b (N, 2), whose
        first column runs along the chord from the leading edge (b_x < 0) to the trailing
        edge (b_x > 0) and whose origin is the midchord pivot."""
        body = np.asarray(body, dtype=float)
        return self.centre(t)[None, :] + body @ self.rotation(t).T

    def velocities(self, body: np.ndarray, t: float) -> np.ndarray:
        """V = centre_velocity + theta_dot * (-r_y, r_x) with r = R(theta) b: the rigid-body
        velocity of the body points at time t, shape (N, 2)."""
        body = np.asarray(body, dtype=float)
        r = body @ self.rotation(t).T
        w = float(self.theta_dot(t))
        V = np.empty_like(r)
        V[:, 0] = -w * r[:, 1]
        V[:, 1] = w * r[:, 0]
        return V + self.centre_velocity(t)[None, :]

    def body_coordinates(self, X: np.ndarray, t: float) -> np.ndarray:
        """Inverse of positions: b = R(theta(t))^T (X - centre(t))."""
        X = np.asarray(X, dtype=float)
        return (X - self.centre(t)[None, :]) @ self.rotation(t)

    def with_centre_at(self, centre: np.ndarray, t: float) -> "StrokeKinematics":
        """The same stroke with its midpoint moved so the midchord is at ``centre`` at time t."""
        c = np.asarray(centre, dtype=float)
        s = float(self.s(t))
        return replace(self, x_mid=float(c[0] - s * np.cos(self.beta)),
                       y_mid=float(c[1] - s * np.sin(self.beta)))


def reynolds_number(rho: float, u_max: float, chord: float, mu: float) -> float:
    """Re = rho U_max c / mu (eq. 10)."""
    return rho * u_max * chord / mu


def force_scale(rho: float, u_rms: float, chord: float) -> float:
    """0.5 rho U_rms^2 c, the force per unit span that makes C_V and C_H dimensionless
    (section 2.2). In 2D rho is a mass per unit area, so this is a force per unit span [N/m]."""
    return 0.5 * rho * u_rms**2 * chord
