"""Taylor-Green through both back ends via the driver: 32^2 cells, central advection, 20
midpoint steps, nu = 0.01, dt = 5e-3, on the doubly periodic unit box.

1. The plain vortex u = -cos(2 pi x) sin(2 pi y), v = sin(2 pi x) cos(2 pi y) is an exact
   discrete eigenmode of both discretisations: its advection is a discrete gradient that the
   projection removes, and the five-point Laplacian has the same eigenvalue on the nodes as
   on the faces. So each back end must return the Crank-Nicolson factor times its initial
   field, on its own points, to round-off, and the two energy decays must agree to round-off.
   This case checks the viscous step and the projection only, not the advection.

2. The same vortex carried by a uniform flow, U0 + TG(x - U0 t, y - V0 t), is not an
   eigenmode, so it exercises the advection. Each back end is compared with the exact solution
   on its own points (nodes, or u- and v-faces), never with the other back end, which would
   need interpolation. Measured max error: 1.85e-3 on both back ends; with the advection term
   set to zero it is 0.29, so a tolerance of 3e-3 tells the two apart.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyibm.driver import Simulation
from pyibm.inputfile import (BCParams, FluidParams, GridParams, OutputParams, Parameters,
                             SolverParams, StructureParams, TimeParams)

N, NU, DT, STEPS = 32, 0.01, 5e-3, 20
BACKENDS = ("periodic_fft", "mac")


def params_for(backend: str) -> Parameters:
    return Parameters(
        fluid=FluidParams(rho=1.0, mu=NU),
        time=TimeParams(Tfinal=STEPS * DT, dt=DT),
        grid=GridParams(Nx=N, Ny=N, Lx=1.0, Ly=1.0),
        solver=SolverParams(backend=backend, time_scheme="midpoint", advection="central"),
        bc=BCParams(),
        structure=StructureParams(),
        output=OutputParams(save_vtk=False, save_history=False))


def taylor_green(X, Y, t=0.0, U0=0.0, V0=0.0):
    """Taylor-Green carried by the uniform flow (U0, V0); an exact Navier-Stokes solution."""
    a = np.exp(-8 * np.pi**2 * NU * t)
    Xs, Ys = X - U0 * t, Y - V0 * t
    return (U0 - a * np.cos(2 * np.pi * Xs) * np.sin(2 * np.pi * Ys),
            V0 + a * np.sin(2 * np.pi * Xs) * np.cos(2 * np.pi * Ys))


def run(backend: str, out_dir: str, U0: float = 0.0, V0: float = 0.0):
    """Run 20 steps from Taylor-Green; return the simulation, the points of u and of v, the
    initial (u, v) and the initial kinetic energy."""
    sim = Simulation(params_for(backend), out_dir, quiet=True)
    g = sim.grid
    if backend == "periodic_fft":
        pts_u = pts_v = g.meshgrid("nodes")
    else:
        pts_u, pts_v = g.meshgrid("u"), g.meshgrid("v")
    sim.solver.u[:] = taylor_green(*pts_u, 0.0, U0, V0)[0]
    sim.solver.v[:] = taylor_green(*pts_v, 0.0, U0, V0)[1]
    u0, v0 = sim.solver.u.copy(), sim.solver.v.copy()
    ke0 = sim.solver.kinetic_energy()
    for _ in range(STEPS):
        sim.step()
    assert sim.n == STEPS and sim.t == pytest.approx(STEPS * DT)
    return sim, pts_u, pts_v, (u0, v0), ke0


def test_taylor_green_eigenmode_backends_agree(tmp_path):
    h = 1.0 / N
    z = NU * DT * (4 / h**2) * (np.sin(np.pi / N) ** 2 + np.sin(np.pi / N) ** 2)
    fac = ((1 - z / 2) / (1 + z / 2)) ** STEPS
    decay = {}
    for backend in BACKENDS:
        sim, _, _, (u0, v0), ke0 = run(backend, str(tmp_path))
        assert np.max(np.abs(sim.solver.u - fac * u0)) < 1e-12
        assert np.max(np.abs(sim.solver.v - fac * v0)) < 1e-12
        decay[backend] = sim.solver.kinetic_energy() / ke0
        if backend == "mac":
            assert np.max(np.abs(sim.solver.divergence())) < 1e-12
    assert abs(decay["periodic_fft"] - decay["mac"]) < 1e-12
    # and both follow the analytic decay of the energy, e^{-16 pi^2 nu t}, up to O(h^2 + dt^2)
    exact = np.exp(-16 * np.pi**2 * NU * STEPS * DT)
    for backend in BACKENDS:
        assert abs(decay[backend] - exact) < 1e-3


def test_translated_taylor_green_exercises_advection(tmp_path):
    U0, V0 = 0.5, 0.25
    for backend in BACKENDS:
        sim, pts_u, pts_v, _, _ = run(backend, str(tmp_path), U0, V0)
        ue = taylor_green(*pts_u, sim.t, U0, V0)[0]
        ve = taylor_green(*pts_v, sim.t, U0, V0)[1]
        err = max(np.max(np.abs(sim.solver.u - ue)), np.max(np.abs(sim.solver.v - ve)))
        assert err < 3e-3, f"{backend}: max error {err:.3e} against the exact solution"
