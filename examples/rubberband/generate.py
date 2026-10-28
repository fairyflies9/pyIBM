"""Write the structure and input files of the rubber band case.

An elliptical band of N points joined by Hookean springs (rest length shorter than the
initial spacing, so the band is stretched and wants to shrink) and torsional beams with
zero preferred curvature. The beams are a token bending stiffness that shows the element
in use: their force is about 1 % of the spring force, and without them the points end up
less than 0.001 dx from where they are with them (measured, see the README); the
stretched springs keep the band smooth on their own. The fluid inside cannot escape, so
the band cannot shrink: it trades its elliptical shape for the circle of the same area,
which has the shortest perimeter. The enclosed area is the quantity to watch; a perfect
method keeps it constant.

Geometry: semi-axes A = 0.2 and B = 0.1 about (0.5, 0.5) in a doubly periodic unit box of
64 x 64 cells. The N = 128 points are placed at equal arc length, spacing ds = 0.48 dx. The
final circle has radius sqrt(A B) = 0.1414.

Two inputs, same structure:

    input2d         delta-function coupling, peskin4 kernel, output in viz/delta
    input2d_blob    blob projection coupling, gaussian4 blob at delta = 3 dx, marker velocity
                    from the interpolated stream function (the default), output in viz/blob

Run from anywhere: python examples/rubberband/generate.py (main(out_dir) writes the same
files into another folder; the tests compare them with the committed ones byte for byte).
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
from pyibm.structures import Beams, Springs, Structure, write_structure

HERE = os.path.dirname(os.path.abspath(__file__))

NAME = "band"
N_POINTS = 128
A_AXIS = 0.2          # m, semi-axis along x
B_AXIS = 0.1          # m, semi-axis along y
CENTRE = (0.5, 0.5)   # m
NX = NY = 64
LX = LY = 1.0         # m
DX = LX / NX          # 0.015625 m
RHO = 1.0             # kg/m^2
MU = 0.01             # N s/m^2
K_SPRING = 400.0      # spring stiffness per unit length [N/m per m]
REST_FACTOR = 0.5     # rest length = REST_FACTOR * initial spacing: the band is stretched
K_BEAM = 5.0e5        # torsional beam stiffness per unit length; EI = K_BEAM ds^6 = 1e-7 N m^2
DT = 2.0e-3           # s
T_FINAL = 2.0         # s (1000 steps)
PRINT_DUMP = 20


def band_points(n: int = N_POINTS) -> np.ndarray:
    """N points on the ellipse at equal arc length (the arc length is tabulated on a fine
    parametrization and inverted by interpolation)."""
    fine = 20000
    phi = np.linspace(0.0, 2.0 * np.pi, fine + 1)
    x = CENTRE[0] + A_AXIS * np.cos(phi)
    y = CENTRE[1] + B_AXIS * np.sin(phi)
    seg = np.hypot(np.diff(x), np.diff(y))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    targets = s[-1] * np.arange(n) / n
    phi_n = np.interp(targets, s, phi)
    return np.column_stack([CENTRE[0] + A_AXIS * np.cos(phi_n), CENTRE[1] + B_AXIS * np.sin(phi_n)])


def band_structure() -> Structure:
    X = band_points()
    n = len(X)
    i = np.arange(n)
    j = (i + 1) % n
    L = np.hypot(*(X[j] - X[i]).T)
    ds = float(L.mean())
    springs = Springs(i, j, np.full(n, K_SPRING), REST_FACTOR * L)
    # one torsional beam about every point, between its two neighbours, straight preferred
    beams = Beams((i - 1) % n, i, j, np.full(n, K_BEAM), np.zeros(n))
    return Structure(name=NAME, X=X, ds=np.full(n, ds), springs=springs, beams=beams)


def parameters(coupling: str, output_dir: str) -> Parameters:
    s = band_structure()
    return Parameters(
        fluid=FluidParams(rho=RHO, mu=MU),
        time=TimeParams(Tfinal=T_FINAL, dt=DT),
        grid=GridParams(Nx=NX, Ny=NY, Lx=LX, Ly=LY),
        solver=SolverParams(backend="periodic_fft", time_scheme="midpoint", coupling=coupling,
                            kernel="peskin4", blob="gaussian4", blob_delta=3.0),
        bc=BCParams(),
        structure=StructureParams(name=NAME, ds=float(s.ds[0])),
        output=OutputParams(print_dump=PRINT_DUMP, save_vtk=True, save_history=True, plot=False,
                            output_dir=output_dir),
    )


def main(out_dir: str = HERE) -> list[str]:
    """Write the structure and input files into out_dir (default: this folder) and return
    their paths. The tests write them into a scratch folder and compare them with the
    committed files byte for byte."""
    s = band_structure()
    written = write_structure(out_dir, s)
    ds = float(s.ds[0])
    for fname, coupling, out, what in (
            ("input2d", "delta", "viz/delta", "delta-function coupling, peskin4 kernel"),
            ("input2d_blob", "blob", "viz/blob", "blob projection coupling, gaussian4 blob, delta = 3 dx")):
        header = (f"rubber band: ellipse {A_AXIS} x {B_AXIS} of {N_POINTS} points (ds = {ds:.6g} = "
                  f"{ds / DX:.3g} dx), springs k = {K_SPRING:g} with rest length {REST_FACTOR} x spacing, "
                  f"beams k = {K_BEAM:g}, {what}\n"
                  "generated by generate.py, do not edit by hand")
        path = os.path.join(out_dir, fname)
        write_input_file(path, parameters(coupling, out), header=header)
        written.append(path)
    for p in written:
        print("wrote", os.path.relpath(p, out_dir))
    return written


if __name__ == "__main__":
    main()
