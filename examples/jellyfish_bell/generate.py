"""Write the structure and input files of the jellyfish-bell case.

Hoover, A. & Miller, L. (2015). A numerical study of the benefits of driving jellyfish bells
at their natural frequency. J. Theor. Biol. 374, 13-25. A two-dimensional hemielliptical
bell that resists stretching and bending, driven by a sinusoidal force toward the centreline
in a band at the margin, swims forward; the free-vibration frequency of the bell was
f_ref = 0.86 s^-1 and the bell swam fastest when driven near it.

Geometry (eq. 1, 10, Table 1, L = 1): the preferred shape is the arc of the ellipse
(x - xc)^2/a^2 + (y - yc)^2/b^2 = 1 for y >= yc - d with a = 0.5 L, b = 0.75 L, d = 0.25 L,
i.e. theta from -theta_lim to pi + theta_lim with theta_lim = asin(d/b). Points are spaced
equally in arclength with ds = h/2 (Table 2, "so that the mesh is not allowed to leak"). The
bell opens downward and swims in +y.

Elastic model (eq. 6): stretching energy ks/2 int (|dX/dq| - 1)^2 dq and bending energy
kb/2 int |d^2 (X - X_b)/dq^2|^2 dq. Discretised with the structure elements of pyIBM:
Hookean springs between neighbours of stiffness ks/ds^2 (per unit length) and rest length
equal to the preferred segment length, and torsional beams about every interior point of
stiffness kb/ds^6 with the preferred cross product of the hemiellipse (the cross product of
two segments of length ds turned by an angle ds*kappa is ds^3 kappa, so k_beam ds^6 (kappa -
kappa_0)^2 ds matches kb (kappa - kappa_0)^2 ds; the beam energy is k/2 (c - curv)^2 per
unit length, pyibm.structures.beam_energy and docs/numerics.md, conventions). The discrete
bending energy of the squeezed bell of section 3.1 is within 5 % of the continuum integral.

Drive: the paper applies a force F_D(t) = F_Mag sin(2 pi f t) along the vector joining a
point to its mirror image on the other half of the bell, on the lower quarter of the bell
(eq. 2, 7, Fig. 5). Here the same band is driven through muscle springs: every point of the
lower quarter is joined to its mirror image by a spring whose rest length is changed in time
by hooks.update_springs,

    rest(t) = L0 (1 - A_m sin(2 pi f t)),

so that its force k_m (L - rest) = k_m L0 (A_m sin(2 pi f t) + strain). With k_m L0 A_m =
F_Mag the nominal force is the paper's; the actual force is smaller by the strain of the
band. A_m = 1 keeps that feedback as small as a spring allows. The contraction phase (sin > 0,
rest shorter) pulls the sides together, the expansion phase pushes them apart, as in the
paper.

Stiffness and force scale. Table 1 gives ks, kb and F_Mag as multiples of rho U^2 L,
rho U^2 L^3 and rho U^2 with U = f_ref L = 0.86 (eq. 16-18). pyIBM's spring and beam
constants and forces are per unit length, and the mapping from these values to them is our
own. Used directly in that convention, the values give a squeezed bell in the free-vibration
study that creeps from a diameter of 0.90 to 0.92 in 3 s instead of swinging to 1.07 in
0.5 s (Fig. 2), and a drive that moves the margin by a hundredth of a diameter instead of a
quarter (Fig. 6). So the paper's ratio ks : kb is kept and two factors are calibrated against
the paper's measurements:

    STIFFNESS_SCALE on ks and kb, so that the free vibration of the squeezed bell has the
        recorded frequency 0.86 s^-1 (the paper's definition of the reference bell; measured
        here as the inverse of the time between the first two maxima of the diameter; it goes
        as the square root of the scale: 0.66 s^-1 at 256, 0.94 at 512, hence 430);
    FORCE_SCALE on F_Mag, so that the quasi-static response, the peak-to-peak swing of the
        diameter when driven slowly at f = 0.25 s^-1, is the 0.186 of Fig. 6. The response
        is not linear in the force (in cycle 2 the swing per unit of force scale is 1.58e-3
        at 118, 1.56e-3 at 150 and 1.26e-3 at 430), so the factor was found by running that
        case itself (write_calibration_case, or python generate.py --calibration <dir>): the
        swing in cycles 2 and 3 is 0.182-0.183 at 115, 0.187-0.188 at 118, 0.190-0.191 at
        120 and 0.192-0.194 at 122 (0.232-0.233 at 150, 0.54 at 430), hence 118.

The resonant amplitude (Fig. 6) and the swimming speed (Fig. 7) are then predictions.

Fluid: rho = 1, mu = rho U L/Re with Re = 172 (Table 1), so mu = 0.005 at f_ref.

Resolution: the paper used h = L/64, ds = L/128 and dt = 1e-5 (Table 2). The inputs here use
h = L/16 (ds = L/32) and dt = 2e-4 (the free vibration at this stiffness is the same at
dt = 2e-4, 1.5e-4, 1e-4 and 5e-5), and input2d_fine at h = L/32 (ds = L/64) with dt = 1e-4
(stable over the two cycles that were run; the whole input takes about 40 minutes). The box
is 4 L wide and 12 L tall, doubly periodic; the bell starts at y = 2 L and swims up.

The driving frequency is encoded in the structure name, bell[_fine]_f<frequency>, so hooks.py
reads it from the structure it is given; bell_free (no muscle springs, no drive) is the
free-vibration study of section 3.1 (initial diameter 90 % of the preferred, the same number
of points at the same spacing along the narrower ellipse, so the springs start unloaded and
the arc reaches a little further down; rest lengths and curvatures of the preferred bell).

Inputs written:

    input2d           f = 0.86 (the resonant case), h = L/16, 10 cycles
    input2d_f0.5, input2d_f0.75, input2d_f1, input2d_f1.25   other points of Fig. 6-7, 5 cycles
    input2d_freevib   free vibration from the squeezed bell, h = L/16, 3 time units
    input2d_fine      f = 0.86 at h = L/32, 10 cycles (two cycles run, see README)

Run from anywhere: python examples/jellyfish_bell/generate.py
The calibration case: python examples/jellyfish_bell/generate.py --calibration <dir> [force_scale],
then python -m pyibm run <dir> and python examples/jellyfish_bell/analyze.py --calibration <dir>.
"""

from __future__ import annotations

import os
import shutil
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

# --- Table 1 (nondimensional, L = 1) ---------------------------------------------------
L = 1.0
A_BELL = 0.5 * L               # half width
B_BELL = 0.75 * L              # height above the ellipse centre
D_BELL = 0.25 * L              # depth of the margin below the ellipse centre
F_REF = 0.86                   # s^-1, the recorded free-vibration frequency
RE_REF = 172.0
RHO = 1.0
U_BODY = F_REF * L             # characteristic speed (section 2.4)
MU = RHO * U_BODY * L / RE_REF  # 0.005
KS_PAPER = 1.0563 * RHO * U_BODY ** 2 * L        # stretching stiffness, eq. 17
KB_PAPER = 0.0528 * RHO * U_BODY ** 2 * L ** 3   # bending stiffness, eq. 18
F_MAG_PAPER = 0.0406 * RHO * U_BODY ** 2         # driving force per unit length, Table 1

# --- calibration (see the module docstring) --------------------------------------------
STIFFNESS_SCALE = 430.0        # free vibration at 0.86 s^-1 (Fig. 2-3)
FORCE_SCALE = 118.0            # quasi-static diameter swing at f = 0.25 s^-1: 0.187-0.188 in
                               # cycles 2-3 of write_calibration_case runs; Fig. 6: 0.186
KS = STIFFNESS_SCALE * KS_PAPER
KB = STIFFNESS_SCALE * KB_PAPER
F_MAG = FORCE_SCALE * F_MAG_PAPER
MUSCLE_AMPLITUDE = 1.0         # A_m: relative rest-length swing of the muscle springs
MUSCLE_BAND = 0.25 * (B_BELL + D_BELL)     # the lower quarter of the bell (section 3.2)
SQUEEZE = 0.9                  # free vibration starts at 90 % of the preferred width (3.1)

# --- grid and time -----------------------------------------------------------------------
LX, LY = 4.0 * L, 12.0 * L
X_CENTRE, Y_CENTRE = 0.5 * LX, 2.0 * L     # ellipse centre at the start
RESOLUTIONS = {"": (16, 2e-4), "_fine": (32, 1e-4)}   # cells per L, dt
DS_OVER_H = 0.5
N_CYCLES = 10                   # the resonant case
N_CYCLES_SWEEP = 5              # the other frequencies: the P = 5 points of Fig. 6-7
OUTPUTS_PER_CYCLE = 40
T_FREE = 3.0                    # free vibration: Fig. 2 runs to t = 3
FREQUENCIES = (0.5, 0.75, 0.86, 1.0, 1.25)
CALIBRATION_FREQUENCY = 0.25   # s^-1: the slow drive FORCE_SCALE is calibrated on (Fig. 6)
N_CYCLES_CALIBRATION = 3


def half_arc(a: float, b: float, n: int, length: float | None = None
             ) -> tuple[np.ndarray, np.ndarray]:
    """n + 1 points from the apex (theta = pi/2) down the right side of the ellipse
    x = a cos(theta), y = b sin(theta), spaced equally in arclength. By default the arc
    ends at the margin theta = -theta_lim (eq. 10); with ``length`` it ends where the arclength
    from the apex reaches that value instead (the squeezed bell of section 3.1 keeps the
    preferred spacing, so its arc runs further down a narrower ellipse). Returns (x, y)
    relative to the ellipse centre."""
    theta_lim = np.arcsin(D_BELL / b)
    th = np.linspace(0.5 * np.pi, -0.5 * np.pi, 40001)
    dl = np.hypot(a * np.sin(th), b * np.cos(th))                  # |dX/dtheta|
    s = np.concatenate([[0.0], np.cumsum(0.5 * (dl[1:] + dl[:-1]) * (-np.diff(th)))])
    if length is None:
        length = float(np.interp(-theta_lim, th[::-1], s[::-1]))
    if length > s[-1]:
        raise ValueError("the requested arc is longer than the half ellipse")
    targets = np.linspace(0.0, length, n + 1)
    th_pts = np.interp(targets, s, th)
    return a * np.cos(th_pts), b * np.sin(th_pts)


def half_arclength(a: float, b: float) -> float:
    theta_lim = np.arcsin(D_BELL / b)
    th = np.linspace(-theta_lim, 0.5 * np.pi, 20001)
    dl = np.hypot(a * np.sin(th), b * np.cos(th))
    return float(np.sum(0.5 * (dl[1:] + dl[:-1]) * np.diff(th)))


def bell_points(h: float, squeeze: float = 1.0) -> np.ndarray:
    """The whole bell, left margin -> apex -> right margin, mirror symmetric about x = X_CENTRE,
    with n_half = round(half arclength / (h/2)) segments on each side. squeeze < 1 narrows
    the ellipse to squeeze * a while keeping the number of points and their spacing (the
    free-vibration start of section 3.1, Fig. 1: the same ds everywhere, so the springs
    carry no load and the bending forces drive the motion)."""
    length = half_arclength(A_BELL, B_BELL)
    n_half = int(round(length / (DS_OVER_H * h)))
    xr, yr = half_arc(squeeze * A_BELL, B_BELL, n_half, length=None if squeeze == 1.0 else length)
    x = np.concatenate([-xr[:0:-1], xr])      # left side mirrored (apex not duplicated)
    y = np.concatenate([yr[:0:-1], yr])
    return np.column_stack([X_CENTRE + x, Y_CENTRE + y])


def preferred_elements(X_pref: np.ndarray, ds: float):
    """Springs (consecutive points, rest = preferred length, k = ks/ds^2) and torsional beams
    (every interior point, curv = preferred cross product, k = kb/ds^6) of the bell."""
    n = len(X_pref)
    i = np.arange(n - 1)
    j = i + 1
    rest = np.hypot(*(X_pref[j] - X_pref[i]).T)
    springs = Springs(i, j, np.full(n - 1, KS / ds ** 2), rest)
    p = np.arange(n - 2)
    q = p + 1
    r = p + 2
    A = X_pref[r] - X_pref[q]
    B = X_pref[q] - X_pref[p]
    curv = A[:, 0] * B[:, 1] - A[:, 1] * B[:, 0]
    beams = Beams(p, q, r, np.full(n - 2, KB / ds ** 6), curv)
    return springs, beams


def muscle_springs(X_pref: np.ndarray, f_mag: float = F_MAG) -> Springs:
    """One spring per mirror pair in the lower quarter of the bell (y below yc - d + band),
    stiffness f_mag / (A_m L0) per unit length so that the nominal force is f_mag (F_Mag by
    default)."""
    n = len(X_pref)
    centre = (n - 1) // 2
    right = np.arange(centre + 1, n)
    left = n - 1 - right
    band = X_pref[right, 1] < Y_CENTRE - D_BELL + MUSCLE_BAND
    i, j = left[band], right[band]
    L0 = np.hypot(*(X_pref[j] - X_pref[i]).T)
    return Springs(i, j, f_mag / (MUSCLE_AMPLITUDE * L0), L0)


def bell_structure(name: str, h: float, driven: bool, squeeze: float = 1.0,
                   f_mag: float = F_MAG) -> Structure:
    ds = DS_OVER_H * h
    X_pref = bell_points(h)
    springs, beams = preferred_elements(X_pref, ds)
    if driven:
        m = muscle_springs(X_pref, f_mag)
        springs = Springs(np.concatenate([springs.i, m.i]), np.concatenate([springs.j, m.j]),
                          np.concatenate([springs.k, m.k]), np.concatenate([springs.rest, m.rest]))
    X = X_pref if squeeze == 1.0 else bell_points(h, squeeze)
    n_half = (len(X_pref) - 1) // 2
    ds_actual = half_arclength(A_BELL, B_BELL) / n_half
    return Structure(name=name, X=X, ds=np.full(len(X), ds_actual), springs=springs, beams=beams)


def structure_name(frequency: float | None, tag: str = "") -> str:
    """bell[_fine]_f<frequency>, or bell[_fine]_free for the free vibration; hooks.py parses it."""
    return f"bell{tag}_free" if frequency is None else f"bell{tag}_f{frequency:.10g}"


def parameters(s: Structure, cells_per_L: int, dt: float, t_final: float, print_dump: int,
               output_dir: str) -> Parameters:
    return Parameters(
        fluid=FluidParams(rho=RHO, mu=MU),
        time=TimeParams(Tfinal=t_final, dt=dt),
        grid=GridParams(Nx=int(round(LX / L)) * cells_per_L, Ny=int(round(LY / L)) * cells_per_L,
                        Lx=LX, Ly=LY),
        solver=SolverParams(backend="periodic_fft", time_scheme="midpoint", coupling="delta",
                            kernel="peskin4"),
        bc=BCParams(),
        structure=StructureParams(name=s.name, ds=float(s.ds[0])),
        output=OutputParams(print_dump=print_dump, save_vtk=False, save_history=True, plot=False,
                            output_dir=output_dir),
    )


def n_cycles(frequency: float) -> int:
    """10 cycles at the resonant frequency, 5 (the P = 5 points of Fig. 6-7) elsewhere."""
    return N_CYCLES if frequency == F_REF else N_CYCLES_SWEEP


def cases() -> list[tuple[str, float | None, str, str, str]]:
    """(input file, frequency or None, resolution tag, output dir, description)."""
    out = []
    for f in FREQUENCIES:
        fname = "input2d" if f == F_REF else f"input2d_f{f:g}"
        out.append((fname, f, "", f"viz/f{f:g}",
                    f"driven at f = {f:g} s^-1 (Fig. 6-7), h = L/16, {n_cycles(f)} cycles"))
    out.append(("input2d_freevib", None, "", "viz/freevib",
                "free vibration from 90 % width (section 3.1, Fig. 2), h = L/16"))
    out.append(("input2d_fine", F_REF, "_fine", "viz/fine",
                f"driven at f = 0.86 s^-1, h = L/32, {N_CYCLES} cycles (slow)"))
    return out


def write_calibration_case(case_dir: str, force_scale: float = FORCE_SCALE,
                           cycles: int = N_CYCLES_CALIBRATION) -> str:
    """Write the slow-drive case FORCE_SCALE was calibrated on into case_dir: the bell driven
    at f = CALIBRATION_FREQUENCY (0.25 s^-1) with the force F_Mag^paper * force_scale, h = L/16,
    for the given number of cycles, plus a copy of hooks.py. Returns the input file's path.
    Run it with pyibm.run(case_dir) and read bell.csv with analyze.per_cycle. case_dir must
    not be the example folder itself, whose input2d and structure files it would overwrite."""
    if os.path.normcase(os.path.realpath(case_dir)) == os.path.normcase(os.path.realpath(HERE)):
        raise ValueError("the calibration case needs its own folder, not the example folder "
                         f"{HERE}")
    os.makedirs(case_dir, exist_ok=True)
    cells, dt = RESOLUTIONS[""]
    f = CALIBRATION_FREQUENCY
    s = bell_structure(structure_name(f), L / cells, driven=True, f_mag=force_scale * F_MAG_PAPER)
    write_structure(case_dir, s)
    shutil.copyfile(os.path.join(HERE, "hooks.py"), os.path.join(case_dir, "hooks.py"))
    steps_per_cycle = int(round(1.0 / (f * dt)))
    dt = 1.0 / (f * steps_per_cycle)
    p = parameters(s, cells, dt, cycles / f, max(1, steps_per_cycle // OUTPUTS_PER_CYCLE),
                   f"viz/f{f:g}")
    path = os.path.join(case_dir, "input2d")
    header = (f"jellyfish bell, calibration of the force scale: driven at f = {f:g} s^-1, "
              f"force x {force_scale:g}, {cycles} cycles\n"
              "written by generate.write_calibration_case")
    write_input_file(path, p, header=header)
    return path


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "--calibration":
        if len(argv) not in (2, 3):
            raise SystemExit("usage: python generate.py --calibration <dir> [force_scale]")
        scale = float(argv[2]) if len(argv) == 3 else FORCE_SCALE
        path = write_calibration_case(argv[1], force_scale=scale)
        print("wrote", path, f"(force x {scale:g}); run it with: python -m pyibm run", argv[1])
        return
    written: list[str] = []
    for fname, f, tag, out, what in cases():
        cells, dt = RESOLUTIONS[tag]
        h = L / cells
        name = structure_name(f, tag)
        if f is None:
            s = bell_structure(name, h, driven=False, squeeze=SQUEEZE)
            t_final = T_FREE
            print_dump = max(1, int(round(0.01 / dt)))
        else:
            s = bell_structure(name, h, driven=True)
            steps_per_cycle = int(round(1.0 / (f * dt)))
            dt = 1.0 / (f * steps_per_cycle)         # a whole number of steps per cycle
            t_final = n_cycles(f) / f
            print_dump = max(1, steps_per_cycle // OUTPUTS_PER_CYCLE)
        written += write_structure(HERE, s)
        p = parameters(s, cells, dt, t_final, print_dump, out)
        header = (f"jellyfish bell, Hoover & Miller 2015: {what}\n"
                  f"{s.n} points, ds = {p.structure.ds:.6g}, dt = {dt!r}; stiffness x {STIFFNESS_SCALE:g}, "
                  f"force x {FORCE_SCALE:g} (see generate.py)\n"
                  "generated by generate.py, do not edit by hand")
        path = os.path.join(HERE, fname)
        write_input_file(path, p, header=header)
        written.append(path)
    for path in written:
        print("wrote", os.path.relpath(path, HERE))


if __name__ == "__main__":
    main()
