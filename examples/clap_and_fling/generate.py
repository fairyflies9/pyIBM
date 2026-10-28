"""Write the structure and input files of the clap-and-fling example.

Two rigid wings of target points perform the 'fling' half-stroke of Miller & Peskin 2005
(fling_kinematics.py has the kinematics and the citations). Three inputs:

    input2d        Re 128, periodic FFT back end, 512^2 on the paper's 1 m box, four walls of
                   target points 0.025 m inside the box edges as in the paper
    input2d_re8    Re 8, the same box, grid and walls
    input2d_mac    Re 128, MAC back end with no-slip walls on the box edges, 512^2

The stiffness lives in the .target file, so each input has its own structure: fling_re128
and fling_re8 (wings + walls) and fling_re128_box (wings only; the MAC box has real walls).
Every point is a target point. ds = c/51 for every point (dx/2 on the 512^2 grid).

Target stiffness, damping and time step. The wings ring on their tether springs against the
added mass of the fluid around them (period 2 pi sqrt(m_added/k): 0.115 s at Re 128 with
k = 2e5, so m_added is about 67 kg/m, per unit length of wing; 0.36 s with k = 2e4), and
the ringing shows in the force traces. Eq. 11 of the paper has a damping term
c_targ (dY/dt - dX/dt) for it; the Re 128 inputs damp the wing points with our own choice
c = 1500 N s/m^2, about 0.2 of the critical damping 2 sqrt(k m_added) (7300 N s/m^2).
On the first 0.45 of the stroke at Re 128 this cut the ringing at its period from 0.070 to
0.011 in C_L, moved the filtered landmarks by 1.5 % or less and the largest lag from 0.59 to
0.57 cells. The damping is explicit: c dt/(rho dx) is 1.54 here; 3700 (3.8) also ran, 7300
(7.5) blew up in 67 steps. At Re 8 the ringing is weaker (about 0.03 in C_L at its period,
0.021 of the stroke); damping it the same way (c = 94, damping number 1.54) moved the
landmarks by 3 % or less in a trial over the first 0.36 of the stroke, so the Re 8 input
keeps c = 0 (README).

k = 2e5 N/m per m at Re 128 gives a lag of 0.57 cells near the lift rotation peak (0.59 at
the end of the stroke); at Re 8 the forces are 256 times smaller and k = 900 holds the wing
to 0.9 cells (k = 350 at 1600 steps let it lag 1.5 cells). The tether numbers
k dt^2/(rho dx) are 0.41 and 0.47, far below the driver's rule of thumb for targets
(driver.TETHER_STABLE). A ten times stiffer wing at Re 128 (k = 2e6, tether number 4.1,
undamped) ran and lagged only 0.11 cells, but rang at a period of 0.0075 of the stroke with
an amplitude of 0.2 in C_L, three times the ringing of k = 2e5, so the example keeps 2e5
with damping. I checked at 256^2 that halving dt at fixed k does not change the traces.
Run from anywhere: python examples/clap_and_fling/generate.py
"""

from __future__ import annotations

import importlib.util
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


def load_kinematics():
    """fling_kinematics.py beside this file, under a name no other example uses."""
    spec = importlib.util.spec_from_file_location("fling_kinematics", os.path.join(HERE, "fling_kinematics.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


kin = load_kinematics()

# One row per input file: structure name, Re, back end, grid, steps per half-stroke,
# target stiffness k [N/m per m], target damping c of the wing points [N s/m per m; the
# walls are not damped], walls of target points, output directory.
VARIANTS = {
    "input2d": dict(name="fling_re128", re=128, backend="periodic_fft", N=512, steps=2500,
                    k=2.0e5, c=1500.0, walls=True, out="viz/re128",
                    what="Re 128, periodic FFT back end, walls of target points"),
    "input2d_re8": dict(name="fling_re8", re=8, backend="periodic_fft", N=512, steps=2500,
                        k=900.0, c=0.0, walls=True, out="viz/re8",
                        what="Re 8, periodic FFT back end, walls of target points"),
    "input2d_mac": dict(name="fling_re128_box", re=128, backend="mac", N=512, steps=2500,
                        k=2.0e5, c=1500.0, walls=False, out="viz/re128_mac",
                        what="Re 128, MAC back end, no-slip box"),
}
PRINT_DUMP = 5


def tfinal(re: int) -> float:
    """The fling half-stroke lasts tau = 6, i.e. 6 c/V seconds (5 s at Re 128, 80 s at Re 8)."""
    return kin.TAU_END * kin.C / kin.SPEEDS[re]


def tether_number(v: dict) -> float:
    """k dt^2/(rho dx), the explicit tether number (the driver warns above
    driver.TETHER_STABLE, a rule of thumb)."""
    dt = tfinal(v["re"]) / v["steps"]
    return v["k"] * dt ** 2 / (kin.RHO * kin.LX / v["N"])


def damping_number(v: dict) -> float:
    """c dt/(rho dx) of the target damping, the explicit damping's counterpart of the tether
    number: c = 1500 at Re 128 gives 1.54; 3.8 also ran and 7.5 blew up (module docstring)."""
    dt = tfinal(v["re"]) / v["steps"]
    return v["c"] * dt / (kin.RHO * kin.LX / v["N"])


def write_wings(name: str, k: float, c: float, walls: bool) -> list[str]:
    """The .vertex and .target files: every point a target of stiffness k, the two wings
    (the first 2 N_PER_WING points) damped with c, the walls undamped."""
    X = kin.initial_points(walls)
    n = len(X)
    damping = np.zeros(n)
    damping[:2 * kin.N_PER_WING] = c
    s = Structure(name=name, X=X, ds=np.full(n, kin.DS),
                  targets=Targets(np.arange(n), np.full(n, k), X.copy(), c=damping))
    return write_structure(HERE, s)


def parameters(v: dict) -> Parameters:
    periodic = v["backend"] == "periodic_fft"
    side = "periodic" if periodic else "dirichlet"
    T = tfinal(v["re"])
    return Parameters(
        fluid=FluidParams(rho=kin.RHO, mu=kin.MU),
        time=TimeParams(Tfinal=T, dt=T / v["steps"]),
        grid=GridParams(Nx=v["N"], Ny=v["N"], Lx=kin.LX, Ly=kin.LY),
        solver=SolverParams(backend=v["backend"], time_scheme="midpoint", coupling="delta",
                            kernel="peskin4", remove_mean_force=False),
        bc=BCParams(left=side, right=side, bottom=side, top=side),
        structure=StructureParams(name=v["name"], ds=kin.DS),
        output=OutputParams(print_dump=PRINT_DUMP, save_vtk=False, save_history=True, plot=False,
                            output_dir=v["out"]),
    )


def main() -> None:
    written = []
    for v in VARIANTS.values():
        written += write_wings(v["name"], v["k"], v["c"], v["walls"])
    for fname, v in VARIANTS.items():
        p = parameters(v)
        header = (f"clap and fling (Miller & Peskin 2005), fling half-stroke: {v['what']}\n"
                  f"V = {kin.SPEEDS[v['re']]} m/s, Re = {kin.reynolds(kin.SPEEDS[v['re']]):.1f}, "
                  f"tau = t V/c runs 0..6; {v['steps']} steps; target stiffness {v['k']:g} "
                  f"(tether number {tether_number(v):.2f}), wing damping {v['c']:g} "
                  f"(damping number {damping_number(v):.2f})\n"
                  "VTK output is off (16 MB per dump at 512^2); set save_vtk = true to get it\n"
                  "generated by generate.py, do not edit by hand")
        path = os.path.join(HERE, fname)
        write_input_file(path, p, header=header)
        written.append(path)
    for p in written:
        print("wrote", os.path.relpath(p, HERE))


if __name__ == "__main__":
    main()
