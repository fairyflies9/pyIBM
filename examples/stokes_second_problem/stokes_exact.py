"""Stokes' second problem: the exact solution and the constants of this example.

A flat wall at y = 0 slides in its own plane with velocity U cos(omega t) under a fluid that
is at rest far away. The fully developed (time-periodic) response is the damped shear wave

    u(y, t) = U exp(-y/d) cos(omega t - y/d),      d = sqrt(2 nu / omega),          (1)

(Stokes 1851; a textbook case, e.g. Batchelor 1967 chapter 4, Schlichting & Gersten
chapter 5). It is the real part of U exp(-k y) exp(i omega t) with k = (1 + i)/d, and
k^2 = 2i/d^2 = i omega/nu, so u_t = nu u_yy holds exactly; the pressure is uniform and v = 0.
The wave travels away from the wall at speed omega d and its amplitude falls by a factor e
over the penetration depth d.

With a second wall at rest at y = Ly the periodic solution is

    u(y, t) = U Re[ sinh(k (Ly - y)) / sinh(k Ly) exp(i omega t) ],                    (2)

which equals (1) up to terms of size U exp(-Ly/d) (expand the sinh ratio: the correction
is -exp(-2 k Ly + k y) + exp(-k y - 2 k Ly), largest at the far wall). At Ly = 10 d that is
4.5e-5 U, far below the discretisation error of any grid used here, so the no-slip top wall
of the finite box stands in for "far away".

The constants below are the parameters of the example; generate.py, hooks.py, analyze.py and
the tests all import them from here so they are written once.
"""

from __future__ import annotations

import numpy as np

U = 1.0                     # m/s, wall velocity amplitude
OMEGA = 2.0 * np.pi         # rad/s, period T = 1 s
D = 0.1                     # m, penetration depth d = sqrt(2 nu/omega)
NU = 0.5 * OMEGA * D**2     # m^2/s, kinematic viscosity that gives d = D
RHO = 1.0                   # kg/m^2
LY = 10.0 * D               # m, box height: the top wall is 10 penetration depths away
PERIOD = 2.0 * np.pi / OMEGA


def penetration_depth(nu: float = NU, omega: float = OMEGA) -> float:
    """d = sqrt(2 nu / omega), the e-folding depth of the shear wave."""
    return float(np.sqrt(2.0 * nu / omega))


def velocity(y, t, U: float = U, omega: float = OMEGA, nu: float = NU):
    """Eq. (1): u(y, t) over a semi-infinite fluid."""
    y = np.asarray(y, dtype=float)
    d = penetration_depth(nu, omega)
    return U * np.exp(-y / d) * np.cos(omega * t - y / d)


def velocity_finite_depth(y, t, Ly: float = LY, U: float = U, omega: float = OMEGA,
                          nu: float = NU):
    """Eq. (2): u(y, t) with a wall at rest at y = Ly."""
    y = np.asarray(y, dtype=float)
    k = (1.0 + 1.0j) / penetration_depth(nu, omega)
    return (U * np.sinh(k * (Ly - y)) / np.sinh(k * Ly) * np.exp(1j * omega * t)).real


def pde_residual(y, t, U: float = U, omega: float = OMEGA, nu: float = NU,
                 eps_t: float = 1e-4, eps_y: float = 1e-4) -> float:
    """Largest |u_t - nu u_yy| / (omega U) of eq. (1) at the points y and time t, with the
    derivatives taken by central differences (independent of the formula's algebra).

    What is left is the truncation error of the differences, not of the formula:
    (eps_y/d)^2/6 from u_yy (the fourth derivative of exp(-y/d) cos(y/d) scales as 4/d^4)
    and (omega eps_t)^2/6 from u_t. With d = 0.1 the default steps give about 1.7e-7 and
    7e-8; eps_y = 1e-3 gives 1.7e-5, which is why the steps are this small. Below
    eps_y = 1e-5 round-off (1e-16 U/eps_y^2) takes over.
    """
    y = np.asarray(y, dtype=float)
    u = lambda yy, tt: velocity(yy, tt, U, omega, nu)
    u_t = (u(y, t + eps_t) - u(y, t - eps_t)) / (2.0 * eps_t)
    u_yy = (u(y + eps_y, t) - 2.0 * u(y, t) + u(y - eps_y, t)) / eps_y**2
    return float(np.max(np.abs(u_t - nu * u_yy)) / (omega * U))
