"""Write the structure and input files of the sedimenting elastic ring.

A closed ring of N points (springs between neighbours with rest length ds, torsional beams
that prefer the circle's curvature, and a mass point on every node) is released at rest
near the top of a closed box of still fluid. Gravity acts on the mass points only
(pyibm.structures.advance_masses: the ghost of mass m per unit length obeys
m dV/dt = -k (Y - X) + m g), so the ring is a heavy membrane around neutrally buoyant
fluid: it falls, flattens at the bottom and relaxes, and the fluid it encloses falls with
it. The box has four no-slip walls at rest
(MAC back end, all sides "dirichlet", no hooks needed for the walls).

Three structures and three inputs:

    ring         64 x 128 cells on 0.5 m x 1.0 m, dx = 1/128 m, ring radius 0.08 m (10 cells),
                 ds = dx/2 (129 points), 3 s
    ring_coarse  32 x 64 cells on the same box, dx = 1/64 m, ds = dx/2 (65 points), 3 s;
                 the smoke test and a quick look
    ring_rigid   the reference: the same circle made of target points, dragged down the same
                 box at the settling speed U_RIGID the elastic ring reaches (hooks.py), so
                 that the drag of a rigid circle in this box can be set against the weight

The stretching stiffness EA and the bending rigidity EI are the physical constants; the
spring and beam coefficients per unit length follow from ds (k_spring = EA/ds^2, k_beam =
EI/ds^6), so both resolutions model the same ring.

What to compare (README): the vertical force the fluid puts on the ring (fy_body in
history.csv) carries the ring's weight M g = m 2 pi R g once gravity is ramped in (a check
that the weight reaches the fluid, not of terminal speed: the ghost update makes it
M (g r(t) - a_ghost)), the enclosed area stays within a fraction of a percent of its initial
value, the slope of the speed over the last 0.5 s says how far the ring still is from its
terminal speed, and the speed gives a drag coefficient that is set against the rigid circle
in the same box.

Run from anywhere: python examples/sedimenting_ring/generate.py
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
from pyibm.structures import Beams, Masses, Springs, Structure, Targets, write_structure

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- box and fluid
LX, LY = 0.5, 1.0        # m
RHO = 1000.0             # kg/m^2
MU = 1.0                 # N s/m^2 (nu = 1e-3 m^2/s)
GRAVITY = 9.81           # m/s^2, acting in -y on the mass points
T_RAMP = 0.2             # s, gravity is ramped in over this time (hooks.py)

# ---------------------------------------------------------------- ring
RADIUS = 0.08            # m
CENTRE = (0.25, 0.78)    # m, released here at rest
MASS = 2.0               # kg/m, mass per unit length of the ring (ghost mass of each point)
K_MASS = 5.0e4           # N/m^2 per unit length, tether of each point to its ghost
EA = 750.0               # N per unit depth, stretching stiffness = k_spring ds^2
EI = 1.0e-3              # N m^2 per unit depth, bending rigidity = k_beam ds^6
K_TARGET = 1.0e7         # N/m^2 per unit length, the rigid reference's target stiffness
U_RIGID = 0.137          # m/s, the elastic ring's speed at t = 3 s (measured, README; still
                         # rising 4.6 % per second then, so not its terminal speed)

# ---------------------------------------------------------------- the three cases
CASES = {
    "ring": dict(Nx=64, Ny=128, dt=2.5e-4, Tfinal=3.0, print_dump=200),
    "ring_coarse": dict(Nx=32, Ny=64, dt=5.0e-4, Tfinal=3.0, print_dump=100),
    "ring_rigid": dict(Nx=64, Ny=128, dt=5.0e-4, Tfinal=3.0, print_dump=100),
}


def ring_points(name: str) -> tuple[np.ndarray, float]:
    """N points counter-clockwise on the circle, spaced ds = dx/2 (rounded so the ring closes)."""
    c = CASES[name]
    dx = LX / c["Nx"]
    n = int(round(2.0 * np.pi * RADIUS / (0.5 * dx)))
    ds = 2.0 * np.pi * RADIUS / n
    th = 2.0 * np.pi * np.arange(n) / n
    X = np.column_stack([CENTRE[0] + RADIUS * np.cos(th), CENTRE[1] + RADIUS * np.sin(th)])
    return X, ds


def k_spring(ds: float) -> float:
    """Spring stiffness for the stretching stiffness EA = k ds^2 (the element energy per unit
    length k/2 (L - ds)^2 times ds of fibre is (EA/2) eps^2 ds with eps = (L - ds)/ds)."""
    return EA / ds ** 2


def k_beam(ds: float) -> float:
    """Torsional beam stiffness for the bending rigidity EI: the element energy per unit
    length k/2 (c - curv)^2 with c = ds^2 sin(d theta) = kappa ds^3 gives EI = k ds^6
    (same derivation as examples/swimmer/generate.py)."""
    return EI / ds ** 6


def structure(name: str) -> Structure:
    X, ds = ring_points(name)
    n = len(X)
    i = np.arange(n)
    if name == "ring_rigid":
        return Structure(name=name, X=X, ds=np.full(n, ds),
                         targets=Targets(i, np.full(n, K_TARGET), X))
    j = (i + 1) % n
    seg = X[j] - X[i]
    rest = np.hypot(seg[:, 0], seg[:, 1])            # the chord length, what the springs hold
    springs = Springs(i, j, np.full(n, k_spring(ds)), rest)
    p, q, r = (i - 1) % n, i, (i + 1) % n
    A = X[r] - X[q]
    B = X[q] - X[p]
    curv = A[:, 0] * B[:, 1] - A[:, 1] * B[:, 0]      # the circle's cross product, as structures.beam_curvature forms it
    beams = Beams(p, q, r, np.full(n, k_beam(ds)), curv)
    masses = Masses(i, np.full(n, K_MASS), np.full(n, MASS), X)
    return Structure(name=name, X=X, ds=np.full(n, ds), springs=springs, beams=beams, masses=masses,
                     gravity=(0.0, -GRAVITY))


def parameters(name: str) -> Parameters:
    c = CASES[name]
    _, ds = ring_points(name)
    return Parameters(
        fluid=FluidParams(rho=RHO, mu=MU),
        time=TimeParams(Tfinal=c["Tfinal"], dt=c["dt"]),
        grid=GridParams(Nx=c["Nx"], Ny=c["Ny"], Lx=LX, Ly=LY),
        solver=SolverParams(backend="mac", time_scheme="midpoint", coupling="delta",
                            kernel="peskin4", advection="blend", viscous="implicit"),
        bc=BCParams(left="dirichlet", right="dirichlet", bottom="dirichlet", top="dirichlet"),
        structure=StructureParams(name=name, ds=ds, gravity_x=0.0, gravity_y=-GRAVITY),
        output=OutputParams(print_dump=c["print_dump"], save_vtk=True, save_history=True,
                            plot=False, output_dir="viz/" + name),
    )


def main() -> None:
    written = []
    for name, input_name in (("ring", "input2d"), ("ring_coarse", "input2d_coarse"),
                             ("ring_rigid", "input2d_rigid")):
        written += write_structure(HERE, structure(name))
        X, ds = ring_points(name)
        if name == "ring_rigid":
            header = (f"rigid reference ({name}): the same circle as target points dragged down the box "
                      f"at U_RIGID = {U_RIGID} m/s (hooks.update_targets)\n"
                      f"R = {RADIUS} m, {len(X)} points, ds = {ds:.6g} m, k_target = {K_TARGET:g}\n"
                      "generated by generate.py, do not edit by hand")
        else:
            header = (f"sedimenting elastic ring ({name}): springs, beams and mass points under gravity "
                      "in a closed no-slip box\n"
                      f"R = {RADIUS} m, {len(X)} points, ds = {ds:.6g} m, m = {MASS} kg/m, weight M g = "
                      f"{MASS * 2 * np.pi * RADIUS * GRAVITY:.5g} N\n"
                      f"EA = {EA:g} (k_spring = {k_spring(ds):.6g}), EI = {EI:g} (k_beam = {k_beam(ds):.6g}), "
                      f"k_mass = {K_MASS:g}\n"
                      "generated by generate.py, do not edit by hand")
        path = os.path.join(HERE, input_name)
        write_input_file(path, parameters(name), header=header)
        written.append(path)
    for p in written:
        print("wrote", os.path.relpath(p, HERE))


if __name__ == "__main__":
    main()
