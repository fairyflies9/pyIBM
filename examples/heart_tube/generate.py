"""Write the structure and input files of the valveless-pumping (heart tube) case.

Jung, E. & Peskin, C. S. (2001). Two-dimensional simulations of valveless pumping using the
immersed boundary method. SIAM J. Sci. Comput. 23, 19-45. Section 3 and Tables 1-2: a
racetrack loop of tubing in a 16 cm x 8 cm doubly periodic box. Each wall of the tube is a
closed curve of immersed boundary points tethered to target points; the bottom straight
segment (x from 4 to 12 cm) is flexible (soft tethers), the rest is almost rigid (stiff
tethers). Pumping is driven by moving the targets of the left third of the flexible segment
up and down in time (hooks.py, the law of section 3.2):

    A(x, t) = A0 sin(2 pi t / T) sin(pi (x - x_d) / L_d),  x_d <= x <= x_d + L_d,

so that the target tube expands (A > 0) and is pinched (A < 0) once per period. The paper's
main result (Fig. 3) is that the time-averaged flux around the loop depends on the period T,
and that it changes sign: clockwise at T = 0.325 s, counterclockwise at T = 0.21 s, about
zero at T = 1.34 s (all at A0 = 0.6 cm). Driving the centre third of the flexible segment
instead gives almost no net flow (section 5.2, -0.0236 cm^2/s).

Geometry (Fig. 1): the centreline of the loop is two straight segments of length 8 cm at
y = 2 and y = 6 cm joined by semicircles of radius 2 cm about (4, 4) and (12, 4) cm, so the
circumference is 16 + 4 pi = 28.57 cm (Table 1). The walls sit 0.6 cm either side of the
centreline (Fig. 1 and Fig. 9 put them at y = 1.4 and 2.6 cm on the bottom; the initial area
between the walls, 34.28 cm^2, is the value quoted under Fig. 4).

Elastic model (eq. 2.6-2.7): f = -kt (X - Z) + kc (X_{l+1} - 2 X_l + X_{l-1}) / ds^2, i.e.
target tethers plus Hookean springs of zero rest length between neighbours with stiffness
kc / ds^2 (the energy is the sum of the squared segment lengths). kt = 26000 g/(s^2 cm) on
the rigid part, 900 on the flexible part, kc = 120 g cm/s^2. CGS units throughout: rho = 1,
mu = 0.01.

Resolution (Table 2): 256 x 128 cells, h = 1/16 cm, ds = h/4, dt = 0.5 h^2 = 0.00195 s;
section 4.1 keeps dt/h = 0.0312 s/cm when the grid changes. The paper uses the first-order
scheme of Peskin & McQueen 1996 (our pm1996, upwind advection); input2d_pm1996 reproduces it
with the paper's dt. The other inputs use the second-order midpoint scheme with the 4-point
kernel at half the paper's dt: the pinch pushes fluid out fast (up to 39-49 cm/s in the
midpoint runs at T = 0.325 and 0.21 s), which puts the centred advection of the midpoint
scheme above CFL 1 at the paper's dt (up to 1.91 at T = 0.325 s); at half of it the runs
stay below CFL 1 (at most 0.93, README). Every input sets remove_mean_force: the targets
pass a net force to the fluid, which in a periodic box would accelerate the box mean. The
solver then holds the box-mean velocity at zero in both schemes; for pm1996 it removes the
mean of the upwind advection term too, since that term is not conservative
(docs/numerics.md).

The drive parameters are encoded in the structure name, tube[_coarse]_T<period>_A<amp>_x<xd>,
so hooks.py (which only sees the structure) knows the period, the amplitude and where the
driven third starts, and generate.py stays the single place where parameters are set.

Inputs written (all at A0 = 0.6 cm):

    input2d           T = 0.325 s, the maximum clockwise case, paper resolution, 20 cycles
    input2d_ccw       T = 0.21 s, the maximum counterclockwise case, 20 cycles
    input2d_zero      T = 1.34 s, the almost-zero-flow case, 10 cycles (13.4 s)
    input2d_symmetric T = 0.21 s with the drive on the centre third of the flexible segment
    input2d_pm1996    like input2d with the paper's first-order scheme and the paper's dt
    input2d_ccw_pm1996, input2d_zero_pm1996   the same for T = 0.21 s and T = 1.34 s
    input2d_coarse    T = 0.325 s on 64 x 32 cells (h = 0.25 cm), for the smoke test

Run from anywhere: python examples/heart_tube/generate.py
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

# --- Table 1 (CGS) ---------------------------------------------------------------------
XSCALE, YSCALE = 16.0, 8.0     # cm, computational domain
RHO = 1.0                      # g/cm^3 (per unit depth in 2D)
MU = 0.01                      # g/(cm s)
HALF_WIDTH = 0.6               # cm, wall offset from the centreline (Fig. 1; Table 1's "d")
KT_RIGID = 26000.0             # g/(s^2 cm), almost rigid tethers
KT_FLEX = 900.0                # g/(s^2 cm), flexible segment tethers
KC = 120.0                     # g cm/s^2, kc of eq. 2.6-2.7 (zero-rest-length neighbour springs)
AMPLITUDE = 0.6                # cm, A0 of the target motion
PERIOD_CW = 0.325              # s, maximum clockwise flow (section 5)
PERIOD_CCW = 0.21              # s, maximum counterclockwise flow
PERIOD_ZERO = 1.34             # s, almost zero flow

# --- geometry (Fig. 1, section 3.2) -----------------------------------------------------
X_LEFT, X_RIGHT = 0.25 * XSCALE, 0.75 * XSCALE     # 4, 12 cm: ends of the straight segments
Y_BOTTOM, Y_TOP = 0.25 * YSCALE, 0.75 * YSCALE     # 2, 6 cm: centrelines of the straights
R_CENTRE = 0.5 * (Y_TOP - Y_BOTTOM)                # 2 cm: radius of the centreline arcs
L_STRAIGHT = X_RIGHT - X_LEFT                      # 8 cm
DRIVE_LENGTH = L_STRAIGHT / 3.0                    # the driven third, 8/3 cm
X_DRIVE_LEFT = X_LEFT                              # the paper's drive: left third
X_DRIVE_CENTRE = X_LEFT + DRIVE_LENGTH             # section 5.2: centre third
CIRCUMFERENCE = 2.0 * L_STRAIGHT + 2.0 * np.pi * R_CENTRE   # 28.57 cm, Table 1

# --- Table 2 ---------------------------------------------------------------------------
DS_OVER_H = 0.25               # ds = h/4
DT_OVER_H_PAPER = 0.03125      # s/cm: dt = 0.5 h^2 at h = 1/16 (section 4.1 keeps dt/dx fixed)
DT_OVER_H = 0.5 * DT_OVER_H_PAPER   # the midpoint inputs: at T = 0.325 and 0.21 s the
                                    # pinch drives speeds of up to 39-49 cm/s, and the
                                    # paper's dt puts the explicit advection at CFL up to 1.91
NX_PAPER, NY_PAPER = 256, 128
NX_COARSE, NY_COARSE = 64, 32
N_CYCLES = 20                  # cycles per run here (the paper ran 150 s)
N_CYCLES_SLOW_DRIVE = 10       # the T = 1.34 s case: 10 cycles are already 13.4 s
OUTPUTS_PER_CYCLE = 40         # history and flux rows per cycle (the flowmeter needs many)


def loop_points(r: float, ds: float) -> tuple[np.ndarray, np.ndarray]:
    """Points of one wall, a closed racetrack of arc radius r (1.4 cm inner, 2.6 cm outer),
    spaced equally in arclength (spacing = perimeter / N, within rounding of ds).

    The curve starts at the left end of the bottom straight (x = 4 cm) and runs
    counterclockwise: bottom straight to the right, right semicircle up, top straight to
    the left, left semicircle down. Returns (X (N, 2), flexible (N,) bool) with flexible
    marking the bottom straight, 4 <= x <= 12 cm.
    """
    perimeter = 2.0 * L_STRAIGHT + 2.0 * np.pi * r
    n = int(round(perimeter / ds))
    s = np.arange(n) * (perimeter / n)
    yc = 0.5 * (Y_BOTTOM + Y_TOP)
    X = np.empty((n, 2))
    half_arc = np.pi * r
    seg1 = s < L_STRAIGHT
    seg2 = (s >= L_STRAIGHT) & (s < L_STRAIGHT + half_arc)
    seg3 = (s >= L_STRAIGHT + half_arc) & (s < 2.0 * L_STRAIGHT + half_arc)
    seg4 = s >= 2.0 * L_STRAIGHT + half_arc
    X[seg1] = np.column_stack([X_LEFT + s[seg1], np.full(seg1.sum(), yc - r)])
    phi = -0.5 * np.pi + (s[seg2] - L_STRAIGHT) / r
    X[seg2] = np.column_stack([X_RIGHT + r * np.cos(phi), yc + r * np.sin(phi)])
    X[seg3] = np.column_stack([X_RIGHT - (s[seg3] - L_STRAIGHT - half_arc), np.full(seg3.sum(), yc + r)])
    phi = 0.5 * np.pi + (s[seg4] - 2.0 * L_STRAIGHT - half_arc) / r
    X[seg4] = np.column_stack([X_LEFT + r * np.cos(phi), yc + r * np.sin(phi)])
    flexible = s <= L_STRAIGHT + 1e-12
    return X, flexible


def tube_structure(name: str, h: float) -> Structure:
    """Both walls as one structure: springs of zero rest length and stiffness kc/ds^2 between
    neighbours within each wall (eq. 2.7), a target on every point (kt flexible on the bottom
    straight, kt rigid elsewhere). Point ds = h/4 up to the rounding of each perimeter."""
    ds = DS_OVER_H * h
    X_in, flex_in = loop_points(R_CENTRE - HALF_WIDTH, ds)
    X_out, flex_out = loop_points(R_CENTRE + HALF_WIDTH, ds)
    X = np.vstack([X_in, X_out])
    flexible = np.concatenate([flex_in, flex_out])
    n_in, n_out = len(X_in), len(X_out)
    i_in = np.arange(n_in)
    i_out = n_in + np.arange(n_out)
    i = np.concatenate([i_in, i_out])
    j = np.concatenate([np.roll(i_in, -1), np.roll(i_out, -1)])
    k_spring = KC / ds ** 2
    springs = Springs(i, j, np.full(len(i), k_spring), np.zeros(len(i)))
    kt = np.where(flexible, KT_FLEX, KT_RIGID)
    targets = Targets(np.arange(len(X)), kt, X)
    # one quadrature weight per wall: its actual spacing
    ds_pts = np.concatenate([np.full(n_in, (2 * L_STRAIGHT + 2 * np.pi * (R_CENTRE - HALF_WIDTH)) / n_in),
                             np.full(n_out, (2 * L_STRAIGHT + 2 * np.pi * (R_CENTRE + HALF_WIDTH)) / n_out)])
    return Structure(name=name, X=X, ds=ds_pts, springs=springs, targets=targets)


def structure_name(period: float, amplitude: float, x_drive: float, coarse: bool = False) -> str:
    """tube[_coarse]_T<period>_A<amplitude>_x<x_drive>; hooks.py parses it."""
    tag = "_coarse" if coarse else ""
    return f"tube{tag}_T{period:.10g}_A{amplitude:.10g}_x{x_drive:.10g}"


def time_step(nx: int, period: float, time_scheme: str) -> float:
    """dt/h of the paper for its first-order scheme, half of it for the midpoint scheme,
    rounded so that a cycle is a whole number of steps."""
    h = XSCALE / nx
    dt = (DT_OVER_H_PAPER if time_scheme == "pm1996" else DT_OVER_H) * h
    return period / int(round(period / dt))


def n_cycles(period: float) -> int:
    """20 cycles, or 10 for the slow drive (T = 1.34 s)."""
    return N_CYCLES_SLOW_DRIVE if period >= 1.0 else N_CYCLES


def parameters(name: str, nx: int, ny: int, period: float, time_scheme: str, kernel: str,
               output_dir: str) -> Parameters:
    dt = time_step(nx, period, time_scheme)
    steps_per_cycle = int(round(period / dt))
    print_dump = max(1, steps_per_cycle // OUTPUTS_PER_CYCLE)
    return Parameters(
        fluid=FluidParams(rho=RHO, mu=MU),
        time=TimeParams(Tfinal=n_cycles(period) * period, dt=dt),
        grid=GridParams(Nx=nx, Ny=ny, Lx=XSCALE, Ly=YSCALE),
        solver=SolverParams(backend="periodic_fft", time_scheme=time_scheme, coupling="delta",
                            kernel=kernel, remove_mean_force=True),
        bc=BCParams(),
        structure=StructureParams(name=name),
        output=OutputParams(print_dump=print_dump, save_vtk=False, save_history=True, plot=False,
                            output_dir=output_dir),
    )


CASES = (
    # input file, period, x of the driven third, coarse?, scheme, kernel, output dir, description
    ("input2d", PERIOD_CW, X_DRIVE_LEFT, False, "midpoint", "peskin4", "viz/cw",
     "T = 0.325 s, maximum clockwise flow (Fig. 3, section 5)"),
    ("input2d_ccw", PERIOD_CCW, X_DRIVE_LEFT, False, "midpoint", "peskin4", "viz/ccw",
     "T = 0.21 s, maximum counterclockwise flow (Fig. 3, section 5)"),
    ("input2d_zero", PERIOD_ZERO, X_DRIVE_LEFT, False, "midpoint", "peskin4", "viz/zero",
     "T = 1.34 s, almost zero net flow (Fig. 3, section 5)"),
    ("input2d_symmetric", PERIOD_CCW, X_DRIVE_CENTRE, False, "midpoint", "peskin4", "viz/symmetric",
     "T = 0.21 s with the drive on the centre third: almost zero net flow (section 5.2)"),
    ("input2d_pm1996", PERIOD_CW, X_DRIVE_LEFT, False, "pm1996", "peskin4", "viz/pm1996",
     "T = 0.325 s with the paper's first-order scheme (Peskin & McQueen 1996) and dt"),
    ("input2d_ccw_pm1996", PERIOD_CCW, X_DRIVE_LEFT, False, "pm1996", "peskin4", "viz/ccw_pm1996",
     "T = 0.21 s with the paper's first-order scheme and dt"),
    ("input2d_zero_pm1996", PERIOD_ZERO, X_DRIVE_LEFT, False, "pm1996", "peskin4", "viz/zero_pm1996",
     "T = 1.34 s with the paper's first-order scheme and dt"),
    ("input2d_coarse", PERIOD_CW, X_DRIVE_LEFT, True, "midpoint", "peskin4", "viz/coarse",
     "T = 0.325 s on 64 x 32 cells, h = 0.25 cm: the smoke test"),
)


def main() -> None:
    written: list[str] = []
    names_done: set[str] = set()
    for fname, period, x_drive, coarse, scheme, kernel, out, what in CASES:
        name = structure_name(period, AMPLITUDE, x_drive, coarse)
        nx, ny = (NX_COARSE, NY_COARSE) if coarse else (NX_PAPER, NY_PAPER)
        if name not in names_done:
            written += write_structure(HERE, tube_structure(name, XSCALE / nx))
            names_done.add(name)
        p = parameters(name, nx, ny, period, scheme, kernel, out)
        header = (f"valveless pumping, Jung & Peskin 2001: {what}\n"
                  f"A0 = {AMPLITUDE} cm, driven third from x = {x_drive:.4g} cm, "
                  f"{n_cycles(period)} cycles, dt = {p.time.dt!r} s\n"
                  "generated by generate.py, do not edit by hand")
        path = os.path.join(HERE, fname)
        write_input_file(path, p, header=header)
        written.append(path)
    for path in written:
        print("wrote", os.path.relpath(path, HERE))


if __name__ == "__main__":
    main()
