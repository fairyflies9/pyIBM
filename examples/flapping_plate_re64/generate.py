"""Write the structure and input files of the flapping plate at Re 64.

The case is the tilted hybrid stroke of Jones et al. 2015 (section 2.1 and Fig. 2c): a rigid
flat plate of chord c that translates back and forth along a stroke plane tilted 62.8
degrees from the horizontal, with a peak-to-peak amplitude of 2.5 chords, while it rotates
about its midchord with an angle of attack alpha = 60 deg + 45 deg sin(2 pi f t). The
paper's dimensionless choices (c = 0.1, f = 1, Table 1) are replaced by dimensional ones of
our own: rho = 10, peak speed U_max = 24, so the period is T = pi A0 / U_max and the
viscosity follows from Re = rho U_max c / mu = 64 (eq. 10). Forces are made dimensionless
with 0.5 rho U_rms^2 c, U_rms = U_max / sqrt(2) (section 2.2). The plate is a line of target
points, one every half cell, pulled toward the prescribed pose by hooks.update_targets.

Two inputs, same structure and grid:

    input2d       doubly periodic box, FFT back end, remove_mean_force = 1 (the net force on
                  the plate would otherwise accelerate the whole box), midpoint, peskin4
    input2d_mac   the same box with no-slip walls at rest (the paper's boundary condition),
                  MAC back end, midpoint, peskin4

Every parameter lives here; run  python examples/flapping_plate_re64/generate.py  after
changing one. The generated files are committed so the example runs without this script.
hooks.py, analyze.py and the tests load this module by path and use its constants and
functions (write_case for cases at other resolutions in temporary directories).

Jones, S. K., Laurenza, R., Hedrick, T. L., Griffith, B. E. & Miller, L. A. (2015). Lift vs.
drag based mechanisms for vertical force production in the smallest flying insects. J. Theor.
Biol. 384, 105-120.
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


def load_sibling(name: str):
    """Import <name>.py from this directory under a name no other example can collide with."""
    modname = f"flapping_plate_re64_{name}"
    if modname in sys.modules:
        return sys.modules[modname]
    spec = importlib.util.spec_from_file_location(modname, os.path.join(HERE, name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[modname] = module
    spec.loader.exec_module(module)
    return module


kin = load_sibling("kinematics")

# ---------------------------------------------------------------- the stroke (Jones et al. 2015)
CHORD = 0.1                      # m, c
STROKE_OVER_CHORD = 2.5          # A0 / c
BETA_DEG = 62.8                  # stroke-plane angle, the tilted stroke of Fig. 2c
ALPHA_MEAN_DEG = 60.0            # alpha_0
ALPHA_AMP_DEG = 45.0             # B = pi/4

# ---------------------------------------------------------------- dimensional choices (ours)
RHO = 10.0                       # kg/m^2
UMAX = 24.0                      # m/s, peak speed of the midchord
RE = 64.0

STROKE = STROKE_OVER_CHORD * CHORD            # 0.25 m
PERIOD = np.pi * STROKE / UMAX                # T = pi A0 / U_max = 0.0327249 s
MU = RHO * UMAX * CHORD / RE                  # 0.375 N s/m^2 from eq. (10)
URMS = UMAX / np.sqrt(2.0)
Q = kin.force_scale(RHO, URMS, CHORD)         # 144 N/m

# ---------------------------------------------------------------- discretisation
BOX_CHORDS = 12                  # square box, 12 chords a side (the paper used 20)
CELLS_PER_CHORD = 16             # dx = c / 16; the paper's finest level had 25.6
TARGET_STIFFNESS = 1.0e9         # N/m per unit length of plate; see README for the lag it gives
STEPS_PER_CYCLE = 6400           # dt = T / STEPS_PER_CYCLE
CYCLES = 3                       # the paper averages its 3rd and 4th cycles
ROWS_PER_CYCLE = 800             # history rows per cycle (print_dump = STEPS_PER_CYCLE / 800 = 8):
                                 # fine enough to resolve the force oscillation as the plate
                                 # crosses grid lines (one cell per 0.008 T at mid-stroke)

LX = LY = round(BOX_CHORDS * CHORD, 12)     # 1.2 m (rounded so 12 * 0.1 does not print as 1.2000000000000002)

VARIANTS = (
    # input file, backend, output dir, description
    ("input2d", "periodic_fft", "viz/periodic", "doubly periodic box, FFT back end, remove_mean_force"),
    ("input2d_mac", "mac", "viz/mac", "no-slip box at rest, MAC back end"),
)


# ---------------------------------------------------------------- geometry
def grid_spacing(cells_per_chord: int = CELLS_PER_CHORD) -> float:
    return CHORD / cells_per_chord


def marker_spacing(cells_per_chord: int = CELLS_PER_CHORD) -> float:
    """ds = dx / 2."""
    return 0.5 * grid_spacing(cells_per_chord)


def stroke() -> "kin.StrokeKinematics":
    """The paper's tilted stroke with its midpoint at the centre of the box."""
    return kin.StrokeKinematics(stroke=STROKE, period=PERIOD, beta=np.radians(BETA_DEG),
                                alpha_mean=np.radians(ALPHA_MEAN_DEG),
                                alpha_amp=np.radians(ALPHA_AMP_DEG),
                                x_mid=0.5 * LX, y_mid=0.5 * LY)


def stroke_from_initial_pose(X0: np.ndarray) -> "kin.StrokeKinematics":
    """The stroke whose t = 0 pose has its midchord at the mean of the initial marker
    positions X0 (the markers are symmetric about the pivot, so that mean is the midchord).
    hooks.py uses this so the motion follows whatever plate.vertex was read."""
    return stroke().with_centre_at(np.asarray(X0, dtype=float).mean(axis=0), 0.0)


def body_points(ds: float) -> np.ndarray:
    """Marker positions in the plate frame: N = round(c/ds) points at b_x = (i + 1/2) ds - c/2,
    b_y = 0, so every marker carries the same length ds and none sits on a tip."""
    n = int(round(CHORD / ds))
    i = np.arange(n)
    return np.column_stack([(i + 0.5) * ds - 0.5 * CHORD, np.zeros(n)])


def initial_positions(cells_per_chord: int = CELLS_PER_CHORD) -> np.ndarray:
    """The plate at t = 0: upper end of the stroke, alpha = alpha_0."""
    return stroke().positions(body_points(marker_spacing(cells_per_chord)), 0.0)


def plate_structure(cells_per_chord: int = CELLS_PER_CHORD, stiffness: float = TARGET_STIFFNESS,
                    name: str = "plate") -> Structure:
    ds = marker_spacing(cells_per_chord)
    X = initial_positions(cells_per_chord)
    n = len(X)
    return Structure(name=name, X=X, ds=np.full(n, ds),
                     targets=Targets(np.arange(n), np.full(n, stiffness), X))


# ---------------------------------------------------------------- parameters
def time_step(steps_per_cycle: int = STEPS_PER_CYCLE) -> float:
    return PERIOD / steps_per_cycle


def tether_number(stiffness: float = TARGET_STIFFNESS, steps_per_cycle: int = STEPS_PER_CYCLE,
                  cells_per_chord: int = CELLS_PER_CHORD) -> float:
    """k dt^2 / (rho dx), the tether number (the driver warns above driver.TETHER_STABLE, a
    rule of thumb; README, 'What to expect', for what stiffer plates do here)."""
    return stiffness * time_step(steps_per_cycle) ** 2 / (RHO * grid_spacing(cells_per_chord))


def parameters(backend: str, output_dir: str, cells_per_chord: int = CELLS_PER_CHORD,
               steps_per_cycle: int = STEPS_PER_CYCLE, cycles: int = CYCLES,
               name: str = "plate") -> Parameters:
    nx = BOX_CHORDS * cells_per_chord
    dt = time_step(steps_per_cycle)
    periodic = backend == "periodic_fft"
    side = "periodic" if periodic else "dirichlet"
    return Parameters(
        fluid=FluidParams(rho=RHO, mu=MU),
        time=TimeParams(Tfinal=cycles * steps_per_cycle * dt, dt=dt),
        grid=GridParams(Nx=nx, Ny=nx, Lx=LX, Ly=LY),
        solver=SolverParams(backend=backend, time_scheme="midpoint", coupling="delta",
                            kernel="peskin4", remove_mean_force=periodic),
        bc=BCParams(side, side, side, side),
        structure=StructureParams(name=name, ds=marker_spacing(cells_per_chord)),
        output=OutputParams(print_dump=max(steps_per_cycle // ROWS_PER_CYCLE, 1),
                            save_vtk=False, save_history=True, plot=False, output_dir=output_dir),
    )


def write_case(case_dir: str, cells_per_chord: int = CELLS_PER_CHORD,
               stiffness: float = TARGET_STIFFNESS, steps_per_cycle: int = STEPS_PER_CYCLE,
               cycles: int = CYCLES, variants=VARIANTS) -> list[str]:
    """Write plate.vertex, plate.target and the input files into case_dir; return the paths."""
    os.makedirs(case_dir, exist_ok=True)
    written = write_structure(case_dir, plate_structure(cells_per_chord, stiffness))
    dx = grid_spacing(cells_per_chord)
    for fname, backend, out, what in variants:
        header = (f"flapping plate, Jones et al. 2015 tilted stroke at Re {RE:g}: {what}\n"
                  f"c = {CHORD} m, A0 = {STROKE_OVER_CHORD} c, beta = {BETA_DEG} deg, "
                  f"alpha = {ALPHA_MEAN_DEG} + {ALPHA_AMP_DEG} sin(2 pi t/T) deg, "
                  f"T = pi A0/Umax = {PERIOD!r} s, Umax = {UMAX} m/s\n"
                  f"{cells_per_chord} cells per chord (dx = {dx:g}), ds = dx/2, "
                  f"target stiffness {stiffness:g}, {steps_per_cycle} steps per cycle, {cycles} cycles, "
                  f"tether number k dt^2/(rho dx) = {tether_number(stiffness, steps_per_cycle, cells_per_chord):.3g}\n"
                  "generated by generate.py, do not edit by hand")
        path = os.path.join(case_dir, fname)
        write_input_file(path, parameters(backend, out, cells_per_chord, steps_per_cycle, cycles),
                         header=header)
        written.append(path)
    return written


def main() -> None:
    for p in write_case(HERE):
        print("wrote", os.path.relpath(p, HERE))
    print(f"Re = {kin.reynolds_number(RHO, UMAX, CHORD, MU):g}, T = {PERIOD:.6g} s, "
          f"q = 0.5 rho Urms^2 c = {Q:g} N/m, dt = {time_step():.4g} s, "
          f"tether number = {tether_number():.3g}")


if __name__ == "__main__":
    main()
