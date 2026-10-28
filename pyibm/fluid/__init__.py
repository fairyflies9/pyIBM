"""Fluid back ends: the interface every back end implements.

The concrete solvers live in their own modules (``pyibm.fluid.periodic_fft`` and
``pyibm.fluid.mac``) and are imported from there. This file holds only the ``FluidSolver``
protocol, so importing ``pyibm.fluid`` never pulls in a back end before it is needed.

See docs/numerics.md for the schemes behind each back end.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from pyibm.grid import FieldLayout, Grid


@runtime_checkable
class FluidSolver(Protocol):
    """What the driver expects from a fluid back end.

    ``Fu``, ``Fv`` are force densities [N/m^2] on ``layout_u`` and ``layout_v``. The state
    ``u``, ``v``, ``p`` is read directly; ``p`` lives on cell centres (MAC) or on the nodes
    (periodic). ``t`` is the time at the start of the step; back ends with boundary data
    evaluate it at ``t + dt/2`` and ``t + dt``.
    """

    grid: Grid
    layout_u: FieldLayout
    layout_v: FieldLayout
    u: np.ndarray
    v: np.ndarray
    p: np.ndarray

    def step(self, Fu: np.ndarray, Fv: np.ndarray, t: float) -> None:
        """First-order scheme (pm1996): u^n -> u^{n+1} with the force F at t^n."""
        ...

    def half_step(self, Fu: np.ndarray, Fv: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Midpoint scheme: return (u_h, v_h) at t + dt/2; the state is left unchanged."""
        ...

    def full_step(self, Fu: np.ndarray, Fv: np.ndarray, u_h: np.ndarray, v_h: np.ndarray,
                  t: float) -> None:
        """Midpoint scheme: u^n -> u^{n+1} using u_h for the advection; sets p."""
        ...

    def vorticity(self) -> np.ndarray:
        """Vorticity on cell centres (MAC) or nodes (periodic), shape (Nx, Ny)."""
        ...

    def cfl(self) -> float:
        """dt * max(|u|/dx + |v|/dy)."""
        ...

    def kinetic_energy(self) -> float:
        ...

    def mean_velocity(self) -> tuple[float, float]:
        ...


__all__ = ["FluidSolver"]
