"""Write the structure and input files of the flapping-filament case of Huang, Shin & Sung 2007.

A flexible, heavy, inextensible filament is pinned at one end in a uniform stream and
flaps (Huang, Shin & Sung 2007, section 4.2, Fig. 13; reference below). Their
problem is set in nondimensional form with the filament length L_r, the free-stream speed
U_inf and the fluid density rho_0 as scales (their eqs. 5-6):

    Re = rho_0 U_inf L_r / mu = 200       mass ratio rho = rho_1 / (rho_0 L_r) = 1.5
    gamma = 0.001  (bending rigidity in units of rho_1 U_inf^2 L_r^2)
    Fr = g L_r / U_inf^2 = 0.5            L = 1, initial angle k = 0.1 pi (eq. 43)
    simply supported ("BC1", eq. 9) at the fixed end

rho_1 is the filament's line density in excess of the fluid it displaces [kg/m], so the
neutrally buoyant filament has rho = 0 and gravity acts on that excess mass only (their
eq. 3). We pick L_r = 1 m, U_inf = 1 m/s, rho_0 = 1 kg/m^2, so every nondimensional number
in the paper is also its SI value here, and build the filament from pyIBM elements:

    springs between neighbours       stretching stiffness K_S [N]:  k_spring = K_S / ds^2
    torsional beams at inner nodes   bending rigidity gamma [N m^2]: k_beam = gamma / ds^6
    mass points on every free node   line density rho_1 [kg/m]:    m = rho_1, tether k_m
    one target point (pinned end)    k_target = K_T / ds with K_T [N/m] the physical tether

The mapping is derived in README.md (the force density on a point is multiplied by ds when
it is spread, so a per-unit-length stiffness k gives the physical stiffness k ds). The
paper's filament is exactly inextensible; ours is a stiff spring chain with K_S = 100 in
units of rho_1 U_inf^2 (150 N), a strain of about 1 % under the loads of this case.

Three inputs, each with its own structure because the stiffnesses depend on ds:

    input2d           192 x 128 cells on 6 x 4 (dx = 1/32), 32 segments, pinned, dt = 2.5e-4
    input2d_coarse     96 x  64 cells on 6 x 4 (dx = 1/16), 16 segments, pinned, dt = 8e-4
    input2d_clamped   as input2d but clamped (two target points, the paper's BC2, eq. 10)
                      and started from the paper's clamp-compatible shape (eq. 47)

Run from anywhere: python examples/flapping_filament/generate.py

Huang, W.-X., Shin, S. J. & Sung, H. J. (2007). Simulation of flexible filaments in a uniform
flow by the immersed boundary method. J. Comput. Phys. 226, 2206-2228.
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

# ---------------------------------------------------------------- the paper's case (sec. 4.2)
L = 1.0                 # m, filament length L_r = L = 1
U_INF = 1.0             # m/s, free-stream speed
RHO0 = 1.0              # kg/m^2, fluid density
RE = 200.0
MASS_RATIO = 1.5        # rho = rho_1 / (rho_0 L_r)
GAMMA_ND = 1.0e-3       # bending rigidity in units of rho_1 U_inf^2 L_r^2
FROUDE = 0.5            # g L_r / U_inf^2, gravity along the stream (+x, their Fig. 1)
ANGLE = 0.1 * np.pi     # k in eq. 43, the initial tilt of the straight filament
KS_ND = 100.0           # stretching stiffness in units of rho_1 U_inf^2 (ours; theirs is inextensible)

# ---------------------------------------------------------------- the same in SI
MU = RHO0 * U_INF * L / RE                  # 0.005 N s/m^2
RHO1 = MASS_RATIO * RHO0 * L                # 1.5 kg/m, excess line density
GAMMA = GAMMA_ND * RHO1 * U_INF**2 * L**2   # 0.0015 N m^2
GRAVITY = FROUDE * U_INF**2 / L             # 0.5 m/s^2
KS = KS_ND * RHO1 * U_INF**2                # 150 N
K_TARGET_PHYS = 1500.0                      # N/m, physical tether of the pinned end (k_target = this / ds)
K_MASS = 1.0e4                              # N/m per m, mass-point tether per unit length

# ---------------------------------------------------------------- domain and run
LX, LY = 6.0, 4.0                           # m; the paper uses 8 x 8 (-2 < x < 6, -4 < y < 4)
X_FIXED = (1.5, 2.0)                        # m, the pinned end (their origin)
T_FINAL = 30.0                              # s; Fig. 13 runs to 60, the amplitude has settled by 25
CLAMP_TURN = 0.64                           # total turn of the clamped initial shape (eq. 47, a = 0.01 at N = 64)

VARIANTS = {
    # name: (structure, Nx, Ny, n_seg, dt, print_dump, save_vtk, clamped, output_dir)
    "input2d": ("filament", 192, 128, 32, 2.5e-4, 200, False, False, "viz/fine"),
    "input2d_coarse": ("filament_coarse", 96, 64, 16, 8.0e-4, 250, True, False, "viz/coarse"),
    "input2d_clamped": ("filament_clamped", 192, 128, 32, 2.5e-4, 200, False, True, "viz/clamped"),
}


def stiffnesses(ds: float) -> dict[str, float]:
    """Per-unit-length element stiffnesses for a point spacing ds (README, 'mapping')."""
    return {"spring": KS / ds**2, "beam": GAMMA / ds**6, "target": K_TARGET_PHYS / ds,
            "mass": K_MASS}


def pinned_points(n_seg: int) -> np.ndarray:
    """Eq. 43: a straight filament from the fixed end X_O at angle k, node 0 the free end
    (s = 0) and node n_seg the fixed end (s = L), as in the paper's Fig. 2."""
    ds = L / n_seg
    s = ds * np.arange(n_seg + 1)
    x0, y0 = X_FIXED
    return np.column_stack([x0 + (L - s) * np.cos(ANGLE), y0 + (L - s) * np.sin(ANGLE)])


def clamped_points(n_seg: int) -> np.ndarray:
    """Eq. 47: segment j from the fixed end turns by j*a, so the first one is horizontal
    (compatible with the clamp, eq. 10) and the free end sits near (0.93, 0.31) relative
    to X_O for a total turn of 0.64 rad (their a = 0.01 with 64 segments)."""
    ds = L / n_seg
    a = CLAMP_TURN / n_seg
    j = np.arange(n_seg)
    turn = j * a
    X = np.empty((n_seg + 1, 2))
    X[n_seg] = X_FIXED
    # node n_seg-1-j = node n_seg-j + ds (cos(j a), sin(j a)); walk from the fixed end
    dx = np.cumsum(ds * np.cos(turn))
    dy = np.cumsum(ds * np.sin(turn))
    X[n_seg - 1 - j, 0] = X_FIXED[0] + dx
    X[n_seg - 1 - j, 1] = X_FIXED[1] + dy
    return X


def build_structure(name: str, n_seg: int, clamped: bool) -> Structure:
    ds = L / n_seg
    k = stiffnesses(ds)
    X = clamped_points(n_seg) if clamped else pinned_points(n_seg)
    n = n_seg + 1
    i = np.arange(n_seg)
    springs = Springs(i, i + 1, np.full(n_seg, k["spring"]), np.full(n_seg, ds))
    q = np.arange(1, n_seg)                                   # inner nodes carry a beam
    beams = Beams(q - 1, q, q + 1, np.full(n_seg - 1, k["beam"]), np.zeros(n_seg - 1))
    free = np.arange(n_seg)                                   # every node but the fixed one
    masses = Masses(free, np.full(n_seg, k["mass"]), np.full(n_seg, RHO1), X[free])
    held = np.array([n_seg, n_seg - 1]) if clamped else np.array([n_seg])
    targets = Targets(held, np.full(len(held), k["target"]), X[held])
    return Structure(name=name, X=X, ds=np.full(n, ds), springs=springs, beams=beams,
                     targets=targets, masses=masses)


def parameters(input_name: str) -> Parameters:
    structure, nx, ny, n_seg, dt, dump, vtk, _clamped, out = VARIANTS[input_name]
    return Parameters(
        fluid=FluidParams(rho=RHO0, mu=MU),
        time=TimeParams(Tfinal=T_FINAL, dt=dt),
        grid=GridParams(Nx=nx, Ny=ny, Lx=LX, Ly=LY),
        solver=SolverParams(backend="mac", time_scheme="midpoint", coupling="delta",
                            kernel="peskin4", advection="blend", viscous="implicit"),
        bc=BCParams(left="dirichlet", right="outflow", bottom="dirichlet", top="dirichlet"),
        structure=StructureParams(name=structure, ds=L / n_seg, gravity_x=GRAVITY, gravity_y=0.0),
        output=OutputParams(print_dump=dump, save_vtk=vtk, save_history=True, plot=False,
                            output_dir=out),
    )


def main() -> None:
    written: list[str] = []
    for input_name, (structure, nx, ny, n_seg, dt, dump, vtk, clamped, out) in VARIANTS.items():
        written += write_structure(HERE, build_structure(structure, n_seg, clamped))
        ds = L / n_seg
        k = stiffnesses(ds)
        header = (f"flapping filament, Huang, Shin & Sung 2007 sec. 4.2: Re {RE:g}, mass ratio "
                  f"{MASS_RATIO:g}, gamma {GAMMA_ND:g}, Fr {FROUDE:g}, L {L:g}, "
                  f"{'clamped (BC2)' if clamped else 'pinned (BC1)'}\n"
                  f"{nx} x {ny} cells, dx = {LX / nx:g}, {n_seg} segments, ds = {ds:g}; "
                  f"k_spring {k['spring']:.6g}, k_beam {k['beam']:.6g}, k_target {k['target']:.6g}, "
                  f"k_mass {k['mass']:g}, m {RHO1:g} per unit length\n"
                  "generated by generate.py, do not edit by hand")
        path = os.path.join(HERE, input_name)
        write_input_file(path, parameters(input_name), header=header)
        written.append(path)
    for p in written:
        print("wrote", os.path.relpath(p, HERE))


if __name__ == "__main__":
    main()
