"""Write the structure and input files of the cylinder-wake case.

The case follows section 4 of M.-C. Lai and C. S. Peskin, "An immersed boundary method with
formal second-order accuracy and reduced numerical viscosity", J. Comput. Phys. 160 (2000)
705-719, doi:10.1006/jcph.2000.6483. A rigid circular cylinder is a ring of target points
held in place by stiff tethers (their eq. 41) in a uniform stream on the MAC back end:
free-stream velocity prescribed on the left (inflow) and on the top and bottom (far field),
outflow on the right. Everything is nondimensional but written in SI names: D = 1 m,
U = 1 m/s, rho = 1 kg/m^2, mu = rho U D / Re, so t is in units of D/U and the forces per unit
span are already C 0.5 rho U^2 D with C the force coefficient (their eqs. 44 and 51).

The marker spacing is ds = dx: n = round(pi D / dx) markers, so ds/dx is 0.99-1.01 on these
grids. We first built the ring at ds = dx/2, the usual rule of thumb for a boundary that
does not leak. On this ring of tethered target points that spacing grows a wrinkle with
wavelengths of 1 to 1.7 cells along it. The 4-point kernel barely sees a
pattern that fine, so the fluid does not hold it back, and the tether force in that pattern
barely reaches the fluid either. It grew to 0.7 cells and made the wake irregular, and a
stiffer tether does not cure it (README.md has the numbers).

At ds = dx the tether stiffness k matters as well. With a soft tether the ring settles into
a static grid-scale pattern (0.10 cells at k = 2000 on the D/dx = 12 grid), and the drag and
the lift depend on it. They converge as k grows, once the target error is below about 0.005
cells. The k below put C_D within about 1 % of the stiffest run on each grid; README.md has
the k sweeps.

The number of markers depends on the grid, so each resolution gets its own structure:

    cylinder            D/dx = 12 on a 192 x 96 grid of the 16 x 8 box     (input2d, input2d_re200)
    cylinder_fine       D/dx = 16 on a 256 x 128 grid of the same box      (input2d_fine)
    cylinder_coarse     D/dx = 8  on a 128 x 64 grid of the same box       (input2d_coarse)

The time step keeps the tether number k dt^2/(rho dx) at 7.5-7.8, below the driver's
rule-of-thumb value of 10 for targets, so the runs do not warn, and it divides the output
interval. The largest advective CFL number in the four runs is 0.09-0.14. Lai & Peskin's
Table I has kappa dt^2/h = 9.95 on both of their grids. The free-stream ramp, the kick that
starts the shedding and the snapshots live in hooks.py.

Run from anywhere: python examples/cylinder_wake/generate.py
"""

from __future__ import annotations

import os
import sys

# Let the script run from a plain clone (no pip install -e .).
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from pyibm.inputfile import (BCParams, FluidParams, GridParams, OutputParams, Parameters,
                             SolverParams, StructureParams, TimeParams, write_input_file)
from pyibm.structures import Structure, Targets, write_structure

HERE = os.path.dirname(os.path.abspath(__file__))

D = 1.0                 # m, cylinder diameter
U = 1.0                 # m/s, free-stream speed (also in hooks.py)
RHO = 1.0               # kg/m^2
LX, LY = 16.0, 8.0      # m, the box: 3.5 D upstream, 11.5 D downstream, 3.5 D to each far-field side
XC, YC = 4.0, 4.0       # m, cylinder centre (also in hooks.py)
T_FINAL = 100.0         # s = D/U; analyze.py uses the last 40 (README.md)
OUTPUT_INTERVAL = 0.1   # s between history rows (about 60 per shedding period)

# name -> (Nx, Ny, dt, k): grid, time step and target stiffness per unit length [N/m per m]
RESOLUTIONS = {
    "cylinder": (192, 96, 0.003125, 64000.0),         # D/dx = 12, k dt^2/(rho dx) = 7.50
    "cylinder_fine": (256, 128, 0.0025, 76800.0),     # D/dx = 16, k dt^2/(rho dx) = 7.68
    "cylinder_coarse": (128, 64, 0.1 / 14, 19200.0),  # D/dx = 8,  k dt^2/(rho dx) = 7.84
}

# input file -> (structure, Re, output_dir, description)
CASES = {
    "input2d": ("cylinder", 100.0, "viz/re100", "Re 100, D/dx = 12"),
    "input2d_re200": ("cylinder", 200.0, "viz/re200", "Re 200, D/dx = 12"),
    "input2d_fine": ("cylinder_fine", 100.0, "viz/re100_fine", "Re 100, D/dx = 16"),
    "input2d_coarse": ("cylinder_coarse", 100.0, "viz/re100_coarse", "Re 100, D/dx = 8 (quick look, smoke test)"),
}


def n_markers(Nx: int) -> int:
    """Number of markers for spacing ds = dx around the circumference pi D (module docstring)."""
    dx = LX / Nx
    return int(round(np.pi * D / dx))


def cylinder_points(Nx: int) -> np.ndarray:
    """Markers on the circle of diameter D about (XC, YC), evenly spaced in angle."""
    n = n_markers(Nx)
    th = 2.0 * np.pi * np.arange(n) / n
    return np.column_stack([XC + 0.5 * D * np.cos(th), YC + 0.5 * D * np.sin(th)])


def marker_ds(Nx: int) -> float:
    """Arc length per marker, pi D / n (the quadrature weight written to the input file)."""
    return np.pi * D / n_markers(Nx)


def tether_number(name: str) -> float:
    """k dt^2 / (rho dx) of a structure (pyibm.driver.TETHER_STABLE has the value it warns at)."""
    Nx, Ny, dt, k = RESOLUTIONS[name]
    return k * dt**2 / (RHO * LX / Nx)


def write_cylinder(name: str) -> list[str]:
    Nx, Ny, dt, k = RESOLUTIONS[name]
    X = cylinder_points(Nx)
    n = len(X)
    s = Structure(name=name, X=X, ds=np.full(n, marker_ds(Nx)),
                  targets=Targets(np.arange(n), np.full(n, k), X))
    return write_structure(HERE, s)


def parameters(structure: str, Re: float, output_dir: str) -> Parameters:
    Nx, Ny, dt, k = RESOLUTIONS[structure]
    return Parameters(
        fluid=FluidParams(rho=RHO, mu=RHO * U * D / Re),
        time=TimeParams(Tfinal=T_FINAL, dt=dt),
        grid=GridParams(Nx=Nx, Ny=Ny, Lx=LX, Ly=LY),
        solver=SolverParams(backend="mac", time_scheme="midpoint", coupling="delta",
                            kernel="peskin4", advection="central", viscous="implicit"),
        bc=BCParams(left="dirichlet", right="outflow", bottom="dirichlet", top="dirichlet"),
        structure=StructureParams(name=structure, ds=marker_ds(Nx)),
        output=OutputParams(print_dump=int(round(OUTPUT_INTERVAL / dt)), save_vtk=False,
                            save_history=True, plot=False, output_dir=output_dir),
    )


def main() -> None:
    written: list[str] = []
    for name in RESOLUTIONS:
        written += write_cylinder(name)
    for fname, (structure, Re, out, what) in CASES.items():
        Nx, Ny, dt, k = RESOLUTIONS[structure]
        header = (f"cylinder wake after Lai & Peskin, J. Comput. Phys. 160 (2000), sec. 4: {what}\n"
                  f"D = {D}, U = {U}, rho = {RHO}, mu = rho U D / Re; box {LX} x {LY}, centre ({XC}, {YC})\n"
                  f"{n_markers(Nx)} target markers at ds = dx, stiffness {k:g} (in {structure}.target), "
                  f"tether number {tether_number(structure):.2f}\n"
                  "generated by generate.py, do not edit by hand")
        path = os.path.join(HERE, fname)
        write_input_file(path, parameters(structure, Re, out), header=header)
        written.append(path)
    for p in written:
        print("wrote", os.path.relpath(p, HERE))


if __name__ == "__main__":
    main()
