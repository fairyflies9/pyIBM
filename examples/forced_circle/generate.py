"""Write the structure and input files of Fovargue's forced circle.

L. Fovargue, Addressing the computational cost of the immersed boundary method through
multi-implicit and multi-rate strategies with time parallelism, PhD thesis, University of
North Carolina at Chapel Hill (2012), section 5.4.0.1, "Non-stiff problem", printed
pp. 57-60, Figures 5.4 and 5.5: a circle of radius r = 0.2 about (0.5, 0.5) of 400 points
in a doubly periodic unit box, fluid at rest, nu = 1e-4. Neighbouring points are joined by
springs and the points on the right half, theta in [-pi/2, pi/2], are tethered to target
points (her eq. 5.39) that follow

    T(theta, t) = (0.5 + a(t) cos theta, 0.5 + b(t) sin theta),
    a(t) = c (1 + eps sin 2 pi t),   b(t) = c / (1 + eps sin 2 pi t),

with c = 0.2 and eps = 0.4 (a b = c^2, so the targets' half-ellipse keeps its area): the
right half is pulled into an ellipse 40 % wider over the first half period and pushed back
over the second; T = 1 is one period. The left half is free and follows through the
springs.

The thesis' membrane has stretching springs with the initial spacing as rest length
(eq. 5.38) and a bending term sigma kappa n (eq. 5.37). The membrane model and the
stiffnesses here are our own choice: a closed chain of zero-rest-length springs,
k_S = 1e4 per unit length, which carries the constant tension k_S ds L = 0.1 N: a
membrane under tension never goes slack, so it needs no bending term to stay smooth (with
the initial spacing as rest length and no bending term the chain ends the period with 44
kinked points on its targeted half and a wavy free half, against 4 kinked points for this
chain, delta coupling; measured, see the README). The targets are soft, k_T = 3200, so the
boundary lags them by about a cell; stiffer targets ring faster after the impulsive start
and the time step then has to resolve that ringing before the error falls at second order
(measured, see the README). The explicit tether numbers k dt^2/(rho dx) are 0.31 (springs)
and 0.1 (targets) at the default dt = dx/16.

Two inputs, same structure:

    input2d         delta-function coupling, peskin4 kernel, output in viz/delta
    input2d_blob    blob projection, gaussian4 blob at delta = 4 dx, output in viz/blob

The blob variant uses the Gaussian blob of Cortez & Minion (J. Comput. Phys. 161 (2000)
428-453) at 4 dx. On this membrane, which has no bending term, it gives a smoother chain at
nu = 1e-4 than the compact blob at 3 dx that Fovargue used, and it converges in dt
(measured, see the README). Both inputs keep the box-mean velocity at zero
(remove_mean_force), as her stream-function formulation did, and the blob input uses the
default stream-function interpolation of the marker velocity.

Run from anywhere: python examples/forced_circle/generate.py (main(out_dir) writes the same
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
from pyibm.structures import Springs, Structure, Targets, write_structure

HERE = os.path.dirname(os.path.abspath(__file__))

NAME = "circle"
N_POINTS = 400
RADIUS = 0.2          # m
CENTRE = (0.5, 0.5)   # m
C_AXIS = 0.2          # m, c in a(t), b(t)
EPS = 0.4             # forcing amplitude
PERIOD = 1.0          # s
NX = NY = 128
LX = LY = 1.0
DX = LX / NX          # 1/128
RHO = 1.0
NU = 1.0e-4           # m^2/s ("Re = 1000" in the thesis)
MU = NU * RHO
K_SPRING = 1.0e4      # per unit length [N/m per m], zero rest length: tension k ds L = 0.1 N
K_TARGET = 3200.0     # per unit length; tether numbers k dt^2/(rho dx) = 0.31 (springs), 0.1 (targets) at dx/16
BLOB = "gaussian4"    # the smoother chain at nu = 1e-4 on this setup (README, "Blob choices")
BLOB_DELTA = 4.0      # blob scale in cells
DT = DX / 16.0        # s, the default time step (2048 steps per period); analyze.py --study varies it
T_FINAL = PERIOD
PRINT_DUMP = 32


def circle_points(n: int = N_POINTS) -> np.ndarray:
    th = 2.0 * np.pi * np.arange(n) / n
    return np.column_stack([CENTRE[0] + RADIUS * np.cos(th), CENTRE[1] + RADIUS * np.sin(th)])


def right_half(n: int = N_POINTS) -> np.ndarray:
    """Indices of the points with theta in [-pi/2, pi/2] (cos theta >= 0)."""
    th = 2.0 * np.pi * np.arange(n) / n
    return np.nonzero(np.cos(th) > -1e-12)[0]


def circle_structure() -> Structure:
    X = circle_points()
    n = len(X)
    i = np.arange(n)
    j = (i + 1) % n
    L = np.hypot(*(X[j] - X[i]).T)
    ds = float(L.mean())
    idx = right_half(n)
    return Structure(name=NAME, X=X, ds=np.full(n, ds),
                     springs=Springs(i, j, np.full(n, K_SPRING), np.zeros(n)),   # rest length 0
                     targets=Targets(idx, np.full(len(idx), K_TARGET), X[idx]))


def parameters(coupling: str, output_dir: str, dt: float = DT) -> Parameters:
    s = circle_structure()
    return Parameters(
        fluid=FluidParams(rho=RHO, mu=MU),
        time=TimeParams(Tfinal=T_FINAL, dt=dt),
        grid=GridParams(Nx=NX, Ny=NY, Lx=LX, Ly=LY),
        solver=SolverParams(backend="periodic_fft", time_scheme="midpoint", coupling=coupling,
                            kernel="peskin4", remove_mean_force=True, blob=BLOB, blob_delta=BLOB_DELTA),
        bc=BCParams(),
        structure=StructureParams(name=NAME, ds=float(s.ds[0])),
        output=OutputParams(print_dump=PRINT_DUMP, save_vtk=True, save_history=True, plot=False,
                            output_dir=output_dir),
    )


def main(out_dir: str = HERE) -> list[str]:
    """Write the structure and input files into out_dir (default: this folder) and return
    their paths. The tests write them into a scratch folder and compare them with the
    committed files byte for byte."""
    s = circle_structure()
    written = write_structure(out_dir, s)
    ds = float(s.ds[0])
    for fname, coupling, out, what in (
            ("input2d", "delta", "viz/delta", "delta-function coupling, peskin4 kernel"),
            ("input2d_blob", "blob", "viz/blob", f"blob projection, {BLOB} blob, delta = {BLOB_DELTA:g} dx")):
        header = (f"Fovargue 2012 sec. 5.4 forced circle: r = {RADIUS}, {N_POINTS} points (ds = {ds:.6g} = "
                  f"{ds / DX:.3g} dx), targets on the right half with c = {C_AXIS}, eps = {EPS}, "
                  f"zero-rest-length springs k_S = {K_SPRING:g} (tension {K_SPRING * ds * ds:.3g} N), "
                  f"k_T = {K_TARGET:g}, nu = {NU}, dt = dx/16; {what}\n"
                  "generated by generate.py, do not edit by hand")
        path = os.path.join(out_dir, fname)
        write_input_file(path, parameters(coupling, out), header=header)
        written.append(path)
    for p in written:
        print("wrote", os.path.relpath(p, out_dir))
    return written


if __name__ == "__main__":
    main()
