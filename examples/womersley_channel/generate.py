"""Write the input files of the Womersley channel (no structure: a fluid-only case).

Oscillating flow between two walls, driven in two ways:

    input2d                   alpha = 3,  H = 0.6:  the exact profile is prescribed as a
                              time-varying Dirichlet inflow on the left (hooks.py), outflow
                              on the right, no-slip top and bottom; 48 x 24 cells
    input2d_alpha10           alpha = 10, H = 2.0:  the same with 80 x 80 cells
    input2d_incremental       input2d with the incremental pressure projection of the MAC
                              back end (Solver key mac_projection), which removes the O(dt)
                              wall slip of the default non-incremental one
    input2d_periodic          alpha = 3:  periodic in x, driven by the body force
                              K cos(omega t) (the external_force hook); 8 x 24 cells
    input2d_alpha10_periodic  alpha = 10: the same with 8 x 80 cells

The fluid and the frequency are the same in all four (womersley_exact.py); the Womersley
number is set by the channel height, as in an experiment. The cells are square with
h = 0.025 m (5.7 cells per Stokes layer d = 0.141 m), the time step is T/400 and output
comes every T/8 so analyze.py can compare eight phases of the last period with the exact
solution. The inflow runs last 4 periods (alpha 3) and 8 periods (alpha 10): with the flow
rate imposed by the inflow the start-up transient is small and gone within two periods
(alpha 3) and seven (alpha 10), measured as the period-to-period change of the kinetic
energy (1e-9 and 7e-7 of its peak). The periodic runs last 8 and 50 periods because the
flow rate is free there and the slowest transient mode, sin(pi y/H), decays on
H^2/(pi^2 nu) = 0.58 s (alpha 3) and 6.5 s (alpha 10); see the README for the numbers.
Run from anywhere: python examples/womersley_channel/generate.py (main(out_dir) writes the
same files into another folder; the tests compare them with the committed ones byte for byte).
"""

from __future__ import annotations

import os
import sys

# Let the script run from a plain clone (no pip install -e .).
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import womersley_exact as ex
from pyibm.inputfile import (BCParams, FluidParams, GridParams, OutputParams, Parameters,
                             SolverParams, StructureParams, TimeParams, write_input_file)

H_CELL = 0.025                  # m, square cells
STEPS_PER_PERIOD = 400
DT = ex.PERIOD / STEPS_PER_PERIOD
PRINT_DUMP = STEPS_PER_PERIOD // 8      # output every T/8: eight phases per period

# (input file, case, channel length Lx, periodic in x, periods, output directory, mac_projection)
VARIANTS = (
    ("input2d", "alpha3", 1.2, False, 4, "viz/alpha3", "nonincremental"),
    ("input2d_alpha10", "alpha10", 2.0, False, 8, "viz/alpha10", "nonincremental"),
    ("input2d_incremental", "alpha3", 1.2, False, 4, "viz/alpha3_incremental", "incremental"),
    ("input2d_periodic", "alpha3", 0.2, True, 8, "viz/alpha3_periodic", "nonincremental"),
    ("input2d_alpha10_periodic", "alpha10", 0.2, True, 50, "viz/alpha10_periodic", "nonincremental"),
)


def cells(length: float) -> int:
    n = int(round(length / H_CELL))
    if abs(n * H_CELL - length) > 1e-12:
        raise ValueError(f"{length} is not a multiple of the cell size {H_CELL}")
    return n


def parameters(case: str, Lx: float, periodic: bool, periods: int, output_dir: str,
               projection: str = "nonincremental") -> Parameters:
    H = ex.HEIGHTS[case]
    if periodic:
        bc = BCParams(left="periodic", right="periodic", bottom="dirichlet", top="dirichlet")
    else:
        bc = BCParams(left="dirichlet", right="outflow", bottom="dirichlet", top="dirichlet")
    return Parameters(
        fluid=FluidParams(rho=ex.RHO, mu=ex.RHO * ex.NU),
        time=TimeParams(Tfinal=periods * ex.PERIOD, dt=DT),
        grid=GridParams(Nx=cells(Lx), Ny=cells(H), Lx=Lx, Ly=H),
        solver=SolverParams(backend="mac", time_scheme="midpoint", coupling="delta",
                            kernel="peskin4", advection="blend", viscous="implicit",
                            mac_projection=projection),
        bc=bc,
        structure=StructureParams(),
        output=OutputParams(print_dump=PRINT_DUMP, save_vtk=True, save_history=True, plot=False,
                            output_dir=output_dir),
    )


def main(out_dir: str = HERE) -> list[str]:
    written = []
    for fname, case, Lx, periodic, periods, out, projection in VARIANTS:
        H = ex.HEIGHTS[case]
        drive = ("periodic in x, body force K cos(omega t)" if periodic
                 else "exact profile as the inflow on the left, outflow on the right")
        if projection != "nonincremental":
            drive += f", {projection} projection"
        header = (f"Womersley channel, alpha = {ex.womersley_number(H):g}: H = {H:g} m, {drive}\n"
                  f"nu = {ex.NU!r} m^2/s, omega = 2 pi rad/s, K = {ex.K!r} N/m^3 "
                  f"(Stokes layer d = {ex.stokes_layer():.4g} m, {ex.stokes_layer() / H_CELL:.3g} cells per d)\n"
                  f"dt = T/{STEPS_PER_PERIOD}, {periods} periods, output every T/{STEPS_PER_PERIOD // PRINT_DUMP}\n"
                  "generated by generate.py, do not edit by hand")
        path = os.path.join(out_dir, fname)
        write_input_file(path, parameters(case, Lx, periodic, periods, out, projection), header=header)
        written.append(path)
        print("wrote", os.path.relpath(path, out_dir))
    return written


if __name__ == "__main__":
    main()
