"""Womersley flow in a 2D channel: the exact solution and the constants of this example.

Flow between parallel walls at y = 0 and y = H driven by an oscillating pressure gradient

    -dp/dx = K cos(omega t) = Re[ K exp(i omega t) ].

With u = Re[ uhat(y) exp(i omega t) ], v = 0, the x momentum equation
rho u_t = -p_x + mu u_yy becomes  i omega uhat = K/rho + nu uhat'',  whose solution with
uhat(0) = uhat(H) = 0 is

    uhat(y) = (K / (i omega rho)) * (1 - cosh(k (y - H/2)) / cosh(k H/2)),
    k = sqrt(i omega / nu) = (1 + i)/d,   d = sqrt(2 nu / omega).                       (1)

Check: i omega uhat = (K/rho)(1 - C), nu uhat'' = (K/(i omega rho)) (-nu k^2) C = -(K/rho) C
with C the cosh ratio, so K/rho + nu uhat'' = (K/rho)(1 - C) = i omega uhat. This is the
plane-channel version of Womersley's pipe solution (Womersley 1955, J. Physiol. 127:553-563),
where the cosh ratio is replaced by J0(i^{3/2} alpha r/R)/J0(i^{3/2} alpha); the physics is the
same: the Womersley number

    alpha = (H/2) sqrt(omega / nu)                                                      (2)

compares the half-height with the Stokes layer thickness d (alpha = H/(sqrt(2) d)). For
alpha << 1 the profile is a quasi-steady parabola in phase with the pressure gradient,
u = (K/(2 mu)) y (H - y) cos(omega t); for alpha >> 1 the core moves as a plug
u = (K/(rho omega)) sin(omega t), lagging the gradient by 90 degrees, with Stokes layers of
thickness d at the walls in which the velocity overshoots the core (the annular effect).
The flow rate per unit depth is

    Q(t) = Re[ (K / (i omega rho)) (H - 2 tanh(k H/2) / k) exp(i omega t) ].             (3)

The constants below are the parameters of the example. The two Womersley numbers are reached
with the same fluid and frequency by changing the channel height, so one hooks.py serves
both inputs: it reads the height off the coordinates the solver hands it
(``channel_height``). generate.py, hooks.py, analyze.py and the tests import from here.
"""

from __future__ import annotations

import numpy as np

RHO = 1.0                   # kg/m^2
OMEGA = 2.0 * np.pi         # rad/s, period T = 1 s
NU = OMEGA / 100.0          # m^2/s: alpha = 10 at H = 2 and alpha = 3 at H = 0.6
U_CORE = 1.0                # m/s, the plug amplitude K/(rho omega) of the high-alpha core
K = RHO * OMEGA * U_CORE    # N/m^3, pressure-gradient amplitude: -dp/dx = K cos(omega t)
PERIOD = 2.0 * np.pi / OMEGA
HEIGHTS = {"alpha3": 0.6, "alpha10": 2.0}      # m, channel heights of the two cases


def stokes_layer(nu: float = NU, omega: float = OMEGA) -> float:
    """d = sqrt(2 nu / omega)."""
    return float(np.sqrt(2.0 * nu / omega))


def womersley_number(H: float, nu: float = NU, omega: float = OMEGA) -> float:
    """Eq. (2): alpha = (H/2) sqrt(omega/nu)."""
    return float(0.5 * H * np.sqrt(omega / nu))


def channel_height(y) -> float:
    """H from the coordinates a boundary hook is handed: y[0] + y[-1] is H both for the cell
    centres (dy/2 and H - dy/2) and for the faces (0 and H)."""
    y = np.asarray(y, dtype=float).ravel()
    return float(y[0] + y[-1])


def velocity_hat(y, H: float, nu: float = NU, omega: float = OMEGA, K: float = K,
                 rho: float = RHO):
    """Eq. (1): the complex amplitude uhat(y); u = Re[uhat exp(i omega t)]."""
    y = np.asarray(y, dtype=float)
    k = np.sqrt(1j * omega / nu)
    return (K / (1j * omega * rho)) * (1.0 - np.cosh(k * (y - 0.5 * H)) / np.cosh(0.5 * k * H))


def velocity(y, t, H: float, nu: float = NU, omega: float = OMEGA, K: float = K,
             rho: float = RHO):
    """u(y, t) of the Womersley channel flow, eq. (1)."""
    return (velocity_hat(y, H, nu, omega, K, rho) * np.exp(1j * omega * t)).real


def pressure_gradient(t, K: float = K, omega: float = OMEGA):
    """-dp/dx = K cos(omega t)."""
    return K * np.cos(omega * np.asarray(t, dtype=float))


def flow_rate(t, H: float, nu: float = NU, omega: float = OMEGA, K: float = K,
              rho: float = RHO):
    """Eq. (3): Q(t) = integral of u over the height."""
    k = np.sqrt(1j * omega / nu)
    qhat = (K / (1j * omega * rho)) * (H - 2.0 * np.tanh(0.5 * k * H) / k)
    return (qhat * np.exp(1j * omega * np.asarray(t, dtype=float))).real


def pde_residual(y, t, H: float, nu: float = NU, omega: float = OMEGA, K: float = K,
                 rho: float = RHO, eps_t: float = 1e-4, eps_y: float = 1e-4) -> float:
    """Largest |u_t - nu u_yy - (K/rho) cos(omega t)| / (K/rho) of eq. (1) at the points y
    and time t, with the derivatives taken by central differences (independent of the
    algebra of the formula).

    What is left is the truncation error of the differences, second order in the steps:
    the profile varies on the Stokes layer scale d = 0.14 m, so the u_yy part is about
    (eps_y/d)^2/6, i.e. 8e-6 for eps_y = 1e-3 and 8e-8 for the default 1e-4 (measured
    7.7e-6 and 7.6e-8 for both heights). Halve the steps and the residual falls by four.
    """
    y = np.asarray(y, dtype=float)
    u = lambda yy, tt: velocity(yy, tt, H, nu, omega, K, rho)
    u_t = (u(y, t + eps_t) - u(y, t - eps_t)) / (2.0 * eps_t)
    u_yy = (u(y + eps_y, t) - 2.0 * u(y, t) + u(y - eps_y, t)) / eps_y**2
    forcing = pressure_gradient(t, K, omega) / rho
    return float(np.max(np.abs(u_t - nu * u_yy - forcing)) / (K / rho))
