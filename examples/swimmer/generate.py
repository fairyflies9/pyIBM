"""Write the structure and input files of the self-propelled swimmer.

The swimmer is a straight filament of N points along x, joined by Hookean springs (rest
length ds, which keeps the body inextensible) and torsional beams (one per interior point).
hooks.py rewrites the beams' preferred curvature every step so that the body wants to take
the shape of a travelling wave with the amplitude envelope of Carling, Williams & Bowtell
1998 (Self-propelled anguilliform swimming: simultaneous solution of the two-dimensional
Navier-Stokes equations and Newton's laws of motion, J. Exp. Biol. 201, 3143-3166, their
eq. 2): the lateral deflection

    h(x, t) = A(x) sin(2 pi (x/lambda - t/T)),   A(x) = 0.125 L (x/L + 0.03125)/1.03125,

so the amplitude grows linearly from 0.0038 L at the head (x = 0) to 0.125 L at the tail
(x = L), the wavelength is one body length and the wave runs from head to tail. The body
swims head first, towards -x. Nothing holds the body: no targets, no mass points, and the
spring and beam forces sum to zero, so the box mean velocity stays zero on its own.

Two structures and two inputs:

    swimmer         L = 1 m in a 3 m x 1.5 m box, 192 x 96 cells (dx = 1/64 m), ds = dx/2
                    (129 points), 4 tail-beat periods
    swimmer_coarse  the same body in the same box, 96 x 48 cells (dx = 1/32 m), ds = dx/2
                    (65 points), 6 periods; the smoke test and a quick look

The stretching stiffness EA and the bending rigidity EI are the physical constants; the
per-unit-length spring and beam coefficients that pyibm wants follow from ds (k_spring =
EA/ds^2, k_beam = EI/ds^6, see k_spring and k_beam below), so both resolutions model the
same body. The README gives the stability limits the time steps were chosen against.

Run from anywhere: python examples/swimmer/generate.py
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

# ---------------------------------------------------------------- body and kinematics
L = 1.0                 # m, body length (arc length)
WAVELENGTH = 1.0        # m, lambda = L in Carling et al. 1998
PERIOD = 1.0            # s, tail-beat period T; wave speed V = lambda/T = 1 m/s
AMPLITUDE_TAIL = 0.125  # tail amplitude / L (Carling et al. 1998)
RAMP_PERIODS = 1.0      # the amplitude is ramped in smoothly over this many periods

# ---------------------------------------------------------------- fluid
RHO = 1.0               # kg/m^2
MU = 0.005              # N s/m^2: Re = V L / nu = 200 on the wave speed

# ---------------------------------------------------------------- elasticity
EA = 250.0              # N per unit depth, stretching stiffness = k_spring ds^2
EI = 0.2                # N m^2 per unit depth, bending rigidity = k_beam ds^6

# ---------------------------------------------------------------- the two cases
# box, cells, time step, length in tail-beat periods, output cadence, head position
CASES = {
    "swimmer": dict(Lx=3.0, Ly=1.5, Nx=192, Ny=96, dt=1.5e-4, periods=4.0, print_dump=200,
                    x_head=1.6),
    "swimmer_coarse": dict(Lx=3.0, Ly=1.5, Nx=96, Ny=48, dt=4.0e-4, periods=6.0, print_dump=100,
                           x_head=1.6),
}


def body_points(name: str) -> tuple[np.ndarray, float]:
    """Straight body along x at mid-height, head at x_head, spaced ds = dx/2. Returns (X, ds)."""
    c = CASES[name]
    dx = c["Lx"] / c["Nx"]
    ds = 0.5 * dx
    n = int(round(L / ds)) + 1
    s = np.arange(n) * ds
    X = np.column_stack([c["x_head"] + s, np.full(n, 0.5 * c["Ly"])])
    return X, ds


def k_spring(ds: float) -> float:
    """Spring stiffness that gives the stretching stiffness EA: the element energy per unit
    length k/2 (L - ds)^2 = k ds^2/2 eps^2 with the strain eps = (L - ds)/ds, times ds of
    fibre, is (EA/2) eps^2 ds with EA = k ds^2."""
    return EA / ds ** 2


def k_beam(ds: float) -> float:
    """Torsional beam stiffness that gives the bending rigidity EI.

    pyibm's beam energy k/2 (c - curv)^2 is an energy per unit length (its gradient is a
    force per unit length; pyibm.structures.beam_energy, docs/numerics.md), so the
    element's energy is ds k/2 (c - curv)^2. With c = ds^2 sin(d theta) = kappa ds^3 for
    small angles that is ds (k ds^6/2)(kappa - kappa0)^2, the Euler-Bernoulli energy
    (EI/2)(kappa - kappa0)^2 ds with EI = k ds^6.
    Check: the force per unit length on a straight body is ds D2(g) with g = k (c - curv),
    i.e. k ds^6 kappa0'' = EI kappa0'', the continuum beam force.
    """
    return EI / ds ** 6


def structure(name: str) -> Structure:
    X, ds = body_points(name)
    n = len(X)
    i = np.arange(n - 1)
    springs = Springs(i, i + 1, np.full(n - 1, k_spring(ds)), np.full(n - 1, ds))
    q = np.arange(1, n - 1)
    beams = Beams(q - 1, q, q + 1, np.full(n - 2, k_beam(ds)), np.zeros(n - 2))
    return Structure(name=name, X=X, ds=np.full(n, ds), springs=springs, beams=beams)


def parameters(name: str) -> Parameters:
    c = CASES[name]
    _, ds = body_points(name)
    return Parameters(
        fluid=FluidParams(rho=RHO, mu=MU),
        time=TimeParams(Tfinal=c["periods"] * PERIOD, dt=c["dt"]),
        grid=GridParams(Nx=c["Nx"], Ny=c["Ny"], Lx=c["Lx"], Ly=c["Ly"]),
        solver=SolverParams(backend="periodic_fft", time_scheme="midpoint", coupling="delta",
                            kernel="peskin4", remove_mean_force=False),
        bc=BCParams(),
        structure=StructureParams(name=name, ds=ds),
        output=OutputParams(print_dump=c["print_dump"], save_vtk=True, save_history=True,
                            plot=False, output_dir="viz/" + name),
    )


def main() -> None:
    written = []
    for name, input_name in (("swimmer", "input2d"), ("swimmer_coarse", "input2d_coarse")):
        written += write_structure(HERE, structure(name))
        _, ds = body_points(name)
        header = (f"self-propelled swimmer ({name}): Carling-type travelling-wave curvature set "
                  "through hooks.update_beams\n"
                  f"L = {L} m, lambda = {WAVELENGTH} m, T = {PERIOD} s, tail amplitude {AMPLITUDE_TAIL} L, "
                  f"Re = V L/nu = {RHO * (WAVELENGTH / PERIOD) * L / MU:g}\n"
                  f"EA = {EA:g} (k_spring = {k_spring(ds):.6g}), EI = {EI:g} (k_beam = {k_beam(ds):.6g}), ds = {ds:g}\n"
                  "generated by generate.py, do not edit by hand")
        path = os.path.join(HERE, input_name)
        write_input_file(path, parameters(name), header=header)
        written.append(path)
    for p in written:
        print("wrote", os.path.relpath(p, HERE))


if __name__ == "__main__":
    main()
