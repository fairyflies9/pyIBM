"""Write the structure and input files of the Cortez & Minion perturbed ellipse.

The membrane is the curve of section 4.1 of R. Cortez and M. Minion, The blob projection
method for immersed boundary problems, J. Comput. Phys. 161 (2000) 428-453 (pp. 436-437), in
polar coordinates about (1/2, 1/2):

    z(theta) = (r cos(theta - theta0), r sin(theta - theta0)),
    r^2 = a^2 cos^2 theta + b^2 sin^2 theta
          + eps (-(4/3) exp(-3 (theta - pi)^2) - exp(-5 (theta - theta1)^2) + exp(-8 (theta - theta2)^2)),

with a = 0.2, b = 0.25, eps = 0.012, theta0 = pi/3, theta1 = 4 pi/5, theta2 = 2 pi/3. The
perturbation sets every point moving at once and removes the symmetries. 400 points at equal
arc length on a 128 x 128 grid (spacing 0.45 dx, below the h/2 the paper quotes as the
leak-free rule), fluid at rest, nu = 0.005, dt = 0.1 h, run to t = 1.

The force in the paper is the curvature force f = sigma kappa n (outward normal), sigma = 1/5,
the force of a membrane under the constant tension sigma. Here it is a closed ring of
zero-rest-length Hookean springs (as L. Fovargue discretised the same membrane in her 2012
UNC Chapel Hill PhD thesis) whose
stiffness hooks.py rescales at every force evaluation so that each link carries exactly the
tension sigma: the total force on a point is then sigma (tau_+ - tau_-), the exact integral of
sigma kappa n over the point's segment. The spring file holds the stiffness at the initial
spacing, sigma/(ds L0), so the structure stands on its own without the hook.

Two inputs, same structure:

    input2d           blob projection, gaussian4 blob (the paper's fourth-order cutoff), delta = 4 dx,
                      marker velocity from the interpolated stream function (divergence-free,
                      pyIBM's default), output in viz/blob
    input2d_delta     delta-function coupling, peskin4 kernel, output in viz/delta

analyze.py draws the membrane turned by half a turn about the box centre, the way round it
sits in the paper's Fig. 3; the box and grid are symmetric under the turn (see the README).

Run from anywhere: python examples/blob_ellipse/generate.py (main(out_dir) writes the same
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
from pyibm.structures import Springs, Structure, write_structure

HERE = os.path.dirname(os.path.abspath(__file__))

NAME = "ellipse"
N_POINTS = 400
A_AXIS = 0.2
B_AXIS = 0.25
EPS = 0.012
THETA0 = np.pi / 3.0
THETA1 = 4.0 * np.pi / 5.0
THETA2 = 2.0 * np.pi / 3.0
CENTRE = (0.5, 0.5)
SIGMA = 0.2           # N, membrane tension (the paper's stiffness constant sigma = 1/5)
NX = NY = 128
LX = LY = 1.0
DX = LX / NX          # 1/128
RHO = 1.0
NU = 0.005            # m^2/s
MU = NU * RHO
DT = 0.1 * DX         # the paper's time step, 7.8125e-4 s
T_FINAL = 1.0         # s (1280 steps)
PRINT_DUMP = 40       # every 0.03125 s; step 400 is t = 0.3125, the time of the paper's Fig. 3
BLOB_DELTA = 4.0      # blob scale in cells (the paper's main choice)

# (input file, coupling, blob interpolation, header text); the delta run ignores the interpolation
INPUTS = (
    ("input2d", "blob", "streamfunction",
     f"blob projection, gaussian4 blob, delta = {BLOB_DELTA:g} dx, stream-function interpolation"),
    ("input2d_delta", "delta", "streamfunction", "delta-function coupling, peskin4 kernel"),
)
OUTPUT_DIRS = {"input2d": "viz/blob", "input2d_delta": "viz/delta"}


def radius(theta: np.ndarray) -> np.ndarray:
    """r(theta) of the perturbed ellipse (Cortez & Minion 2000 section 4.1)."""
    r2 = (A_AXIS ** 2 * np.cos(theta) ** 2 + B_AXIS ** 2 * np.sin(theta) ** 2
          + EPS * (-(4.0 / 3.0) * np.exp(-3.0 * (theta - np.pi) ** 2)
                   - np.exp(-5.0 * (theta - THETA1) ** 2)
                   + np.exp(-8.0 * (theta - THETA2) ** 2)))
    return np.sqrt(r2)


def curve(theta: np.ndarray) -> np.ndarray:
    r = radius(theta)
    return np.column_stack([CENTRE[0] + r * np.cos(theta - THETA0),
                            CENTRE[1] + r * np.sin(theta - THETA0)])


def ellipse_points(n: int = N_POINTS) -> np.ndarray:
    """n points at equal arc length, starting at theta = 0."""
    fine = 40000
    theta = np.linspace(0.0, 2.0 * np.pi, fine + 1)
    Z = curve(theta)
    seg = np.hypot(np.diff(Z[:, 0]), np.diff(Z[:, 1]))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    theta_n = np.interp(s[-1] * np.arange(n) / n, s, theta)
    return curve(theta_n)


def ellipse_structure() -> Structure:
    X = ellipse_points()
    n = len(X)
    i = np.arange(n)
    j = (i + 1) % n
    L = np.hypot(*(X[j] - X[i]).T)
    ds = float(L.mean())
    # zero rest length, tension k ds L = sigma at the initial spacing
    k = SIGMA / (ds * L)
    return Structure(name=NAME, X=X, ds=np.full(n, ds), springs=Springs(i, j, k, np.zeros(n)))


def parameters(coupling: str, output_dir: str, interpolation: str = "streamfunction") -> Parameters:
    s = ellipse_structure()
    return Parameters(
        fluid=FluidParams(rho=RHO, mu=MU),
        time=TimeParams(Tfinal=T_FINAL, dt=DT),
        grid=GridParams(Nx=NX, Ny=NY, Lx=LX, Ly=LY),
        solver=SolverParams(backend="periodic_fft", time_scheme="midpoint", coupling=coupling,
                            kernel="peskin4", blob="gaussian4", blob_delta=BLOB_DELTA,
                            blob_interpolation=interpolation),
        bc=BCParams(),
        structure=StructureParams(name=NAME, ds=float(s.ds[0])),
        output=OutputParams(print_dump=PRINT_DUMP, save_vtk=True, save_history=True, plot=False,
                            output_dir=output_dir),
    )


def main(out_dir: str = HERE) -> list[str]:
    """Write the structure and input files into out_dir (default: this folder) and return
    their paths. The tests write them into a scratch folder and compare them with the
    committed files byte for byte."""
    s = ellipse_structure()
    written = write_structure(out_dir, s)
    ds = float(s.ds[0])
    for fname, coupling, interpolation, what in INPUTS:
        header = (f"Cortez & Minion 2000 sec. 4.1 perturbed ellipse: a = {A_AXIS}, b = {B_AXIS}, "
                  f"eps = {EPS}, sigma = {SIGMA}, {N_POINTS} points (ds = {ds:.6g} = {ds / DX:.3g} dx), "
                  f"nu = {NU}, dt = 0.1 dx; {what}\n"
                  "generated by generate.py, do not edit by hand")
        path = os.path.join(out_dir, fname)
        write_input_file(path, parameters(coupling, OUTPUT_DIRS[fname], interpolation), header=header)
        written.append(path)
    for p in written:
        print("wrote", os.path.relpath(p, out_dir))
    return written


if __name__ == "__main__":
    main()
