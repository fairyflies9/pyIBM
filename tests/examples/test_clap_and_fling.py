"""examples/clap_and_fling: smoke runs of the three inputs on coarsened grids, the committed
files against generate.py, the kinematics against the formulas of Miller & Peskin 2005,
the reference values against the paper's own refinement figure, the ringing filter of
analyze.py on a synthetic trace and its row count, the reader of old force files, and
(slow) the rotation and acceleration phase of the fling at Re 128 and Re 8 at full
resolution against the paper's Figs. 9 and 11."""

from __future__ import annotations

import dataclasses
import importlib.util
import os

import numpy as np
import pytest

from pyibm.driver import TETHER_STABLE, Simulation, run
from pyibm.inputfile import read_input_file
from pyibm.io.history import read_history
from pyibm.structures import load_hooks, read_structure, read_target, read_vertex

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CASE = os.path.join(ROOT, "examples", "clap_and_fling")
INPUTS = ("input2d", "input2d_re8", "input2d_mac")
N_WALL = 3876            # wall points of the periodic variants (4 x 0.95 m at ds = c/51)
STEPS_TO_045 = 1125      # 0.45 of the half-stroke: rotation peak, trough and acceleration peak


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(CASE, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_kinematics():
    return load_module("fling_kinematics", "fling_kinematics.py")


def load_generate():
    return load_module("clap_and_fling_generate", "generate.py")


def load_analyze():
    return load_module("clap_and_fling_analyze", "analyze.py")


def coarse_params(input_name: str, N: int):
    """The committed input with the grid coarsened to N^2 (same box, dt and files)."""
    p = read_input_file(os.path.join(CASE, input_name))
    return dataclasses.replace(p, grid=dataclasses.replace(p.grid, Nx=N, Ny=N))


@pytest.mark.parametrize("input_name,N", [("input2d", 128), ("input2d_re8", 128), ("input2d_mac", 64)])
def test_smoke_three_steps_coarse(tmp_path, input_name, N):
    p = coarse_params(input_name, N)
    sim = Simulation(p, CASE, quiet=True, output_dir=str(tmp_path))
    res = sim.run(steps=3)
    assert res.steps == 3 and res.history.step.tolist() == [0.0, 3.0]   # print_dump = 5
    assert np.all(np.isfinite(res.history.view(float)))
    assert res.history.cfl.max() < 1.0
    assert res.history.max_target_error_over_dx.max() < 0.5
    assert os.path.isfile(tmp_path / "history.csv")
    assert len(read_history(tmp_path / "history.csv")) == 2
    # the hook's per-wing force file: a header and two rows, mirror-symmetric wings
    an = load_analyze()
    d = an.read_wing_forces(str(tmp_path / "wing_forces.csv"))
    assert len(d) == 2 and d.step.tolist() == [0.0, 3.0]
    assert np.all(np.isfinite(np.column_stack([d.CL_left, d.CD_left, d.CL_right, d.CD_right])))
    assert np.allclose(d.CL_left, d.CL_right, atol=1e-6) and np.allclose(d.CD_left, d.CD_right, atol=1e-6)
    # the force of a row is that of the last step, at the history's t_force (t - dt/2 for the
    # midpoint scheme, 0 at step 0), and tau is that of t_force
    assert np.array_equal(d.t_force, res.history.t_force)
    assert d.t_force[0] == 0.0 and np.isclose(d.t_force[1], d.t[1] - 0.5 * p.time.dt)
    assert np.allclose(d.fraction_of_stroke, d.tau / 6.0)
    assert np.allclose(d.tau, d.t_force * (0.06 if "re8" not in input_name else 0.00375) / 0.05)


def test_smoke_default_input_at_full_resolution(tmp_path):
    """Three steps of the committed input2d exactly as it is (512^2, 3978 points)."""
    res = run(CASE, steps=3, quiet=True, output_dir=str(tmp_path))
    assert res.steps == 3 and res.structure.n == 2 * 51 + N_WALL
    assert res.grid.Nx == 512 and res.parameters.solver.backend == "periodic_fft"
    assert np.all(np.isfinite(res.history.view(float)))
    assert res.history.max_target_error_over_dx.max() < 0.5
    assert os.path.isfile(tmp_path / "wing_forces.csv")


def test_kinematics_match_the_paper():
    kin = load_kinematics()
    # Re = rho V c/mu (eq. 1) for the two speeds of the paper
    assert abs(kin.reynolds(kin.SPEEDS[128]) - 128) < 1.0 and abs(kin.reynolds(kin.SPEEDS[8]) - 8) < 0.1
    # rotation through pi/4 over dtau_rot = 1.74 (eqs. 5-6), then constant
    assert np.isclose(kin.rotation_angle(kin.DTAU_ROT), np.pi / 4)
    assert np.isclose(kin.rotation_angle(6.0), np.pi / 4) and kin.rotation_angle(-1.0) == 0.0
    V = kin.SPEEDS[128]
    om = kin.angular_velocity(np.linspace(0.0, 2.0, 4001) * kin.C / V, V)
    assert np.isclose(om.max(), 2 * kin.DTHETA * V / (kin.DTAU_ROT * kin.C), rtol=1e-6)   # omega_rot, eq. 6
    # translation: starts at tau = 0.86, accelerates over 1.3, 4.49 chords by tau = 6 (eq. 2)
    assert kin.speed_fraction(kin.TAU_ACCEL) == 0.0 and kin.speed_fraction(kin.TAU_ACCEL + kin.DTAU_ACCEL) == 1.0
    assert np.isclose(kin.travel(kin.TAU_ACCEL + kin.DTAU_ACCEL), kin.DTAU_ACCEL / 2)
    assert np.isclose(kin.travel(kin.TAU_END), 4.49)
    # the paper's "halfway through rotation" is 1.74/2 = 0.87 to two digits of 0.86
    assert abs(kin.TAU_ACCEL - kin.DTAU_ROT / 2) < 0.02
    # geometry: vertical wings c/6 apart at t = 0, right wing the mirror image of the left
    L, R = kin.wing_positions(0.0, V)
    assert np.allclose(L[:, 0], kin.LX / 2 - kin.GAP / 2) and np.allclose(R[:, 0], kin.LX / 2 + kin.GAP / 2)
    assert np.isclose(L[-1, 1] - L[0, 1], kin.C - kin.DS)
    for t in (0.3, 0.9, 2.0, 4.9):
        L, R = kin.wing_positions(t, V)
        assert np.allclose(R[:, 0], kin.LX - L[:, 0]) and np.allclose(R[:, 1], L[:, 1])
        # velocities are the time derivative of the positions
        h = 1e-6
        Lp, Rp = kin.wing_positions(t + h, V)
        Lm, Rm = kin.wing_positions(t - h, V)
        vl, vr = kin.wing_velocities(t, V)
        assert np.allclose((Lp - Lm) / (2 * h), vl, atol=1e-7) and np.allclose((Rp - Rm) / (2 * h), vr, atol=1e-7)
    # final angle of attack 45 degrees: the left wing leans left at 135 degrees from +x
    L, R = kin.wing_positions(5.0, V)
    d = L[-1] - L[0]
    assert np.isclose(np.degrees(np.arctan2(d[1], d[0])), 135.0)
    # walls: inside the box by WALL_INSET, spaced ds along the perimeter
    W = kin.wall_points()
    assert len(W) == N_WALL
    assert np.isclose(W.min(), kin.WALL_INSET) and np.isclose(W.max(), kin.LX - kin.WALL_INSET)
    assert abs(len(W) * kin.DS - 4 * (kin.LX - 2 * kin.WALL_INSET)) < kin.DS
    # force coefficients: eqs. 15-16, drag opposes the motion of each wing
    cl, cd = kin.force_coefficients(np.array([1.0]), np.array([2.0]), V, "left")
    q = 0.5 * kin.RHO * kin.C * V ** 2
    assert np.isclose(cl[0], 2.0 / q) and np.isclose(cd[0], 1.0 / q)
    assert np.isclose(kin.force_coefficients(np.array([1.0]), np.array([2.0]), V, "right")[1][0], -1.0 / q)


def test_committed_files_match_generate():
    gen = load_generate()
    kin = gen.kin
    for input_name, v in gen.VARIANTS.items():
        X = read_vertex(os.path.join(CASE, v["name"] + ".vertex"))
        assert np.allclose(X, kin.initial_points(v["walls"]))
        T = read_target(os.path.join(CASE, v["name"] + ".target"), X)
        assert np.all(T.k == v["k"]) and T.idx.tolist() == list(range(len(X)))
        # the wing points are damped with c (eq. 11 of the paper), the walls are not
        n_wing = 2 * kin.N_PER_WING
        assert np.all(T.c[:n_wing] == v["c"]) and np.all(T.c[n_wing:] == 0.0)
        p = read_input_file(os.path.join(CASE, input_name))
        assert p.structure.name == v["name"] and p.structure.ds == kin.DS
        assert (p.grid.Nx, p.grid.Ny, p.grid.Lx, p.grid.Ly) == (v["N"], v["N"], kin.LX, kin.LY)
        assert (p.fluid.rho, p.fluid.mu) == (kin.RHO, kin.MU)
        assert np.isclose(p.time.Tfinal, kin.TAU_END * kin.C / kin.SPEEDS[v["re"]]) and p.time.n_steps == v["steps"]
        assert p.solver.backend == v["backend"] and p.solver.time_scheme == "midpoint"
        assert p.solver.coupling == "delta" and p.solver.kernel == "peskin4"
        assert p.output.output_dir == v["out"] and p.output.print_dump == gen.PRINT_DUMP
        # explicit tether number below the driver's rule of thumb for targets, and the
        # damping number c dt/(rho dx) below 2 (1.54 at Re 128 ran, 7.5 blew up)
        assert gen.tether_number(v) < TETHER_STABLE[("target", "midpoint")]
        assert np.isclose(gen.tether_number(v), v["k"] * p.time.dt ** 2 / (kin.RHO * p.grid.dx))
        assert gen.damping_number(v) < 2.0
        assert np.isclose(gen.damping_number(v), v["c"] * p.time.dt / (kin.RHO * p.grid.dx))
    p = read_input_file(os.path.join(CASE, "input2d_mac"))
    assert (p.bc.left, p.bc.right, p.bc.bottom, p.bc.top) == ("dirichlet",) * 4
    # every point is tethered and the walls, when present, are 2 cells or more from nothing
    # (periodic box) while the MAC wings start 0.49 m from the Dirichlet walls
    X = read_vertex(os.path.join(CASE, "fling_re128_box.vertex"))
    assert len(X) == 2 * kin.N_PER_WING and X.min() > 0.4 and X.max() < 0.6


def test_hook_moves_the_wings_only():
    kin = load_kinematics()
    hooks = load_hooks(CASE)
    s = read_structure(CASE, "fling_re128", ds=kin.DS)
    n = kin.N_PER_WING
    for t in (0.0, 0.5, 1.2, 3.0):
        hooks.update_targets(t, 0.002, s)
        L, R = kin.wing_positions(t, kin.SPEEDS[128])
        assert np.allclose(s.targets.X[:n], L) and np.allclose(s.targets.X[n:2 * n], R)
        assert np.array_equal(s.targets.X[2 * n:], s.X0[2 * n:])
        # the damping pulls the wings towards these velocities; the walls stay at rest
        vl, vr = kin.wing_velocities(t, kin.SPEEDS[128])
        assert np.allclose(s.targets.V[:n], vl) and np.allclose(s.targets.V[n:2 * n], vr)
        assert np.all(s.targets.V[2 * n:] == 0.0)
    s8 = read_structure(CASE, "fling_re8", ds=kin.DS)
    hooks.update_targets(16.0, 0.032, s8)       # tau = 1.2 at Re 8 ...
    hooks.update_targets(1.0, 0.002, s)         # ... and at Re 128: the same pose
    assert np.allclose(s8.targets.X[:2 * n], s.targets.X[:2 * n])


def test_reference_values_agree_with_the_papers_refinement_figure():
    """Figs. 9 and 11 (Re 128) and Fig. 3 (the paper's refinement study, Re 128, on the same
    mesh) show the same fling, so their rotation peaks, troughs and acceleration peaks must
    agree to within how well the figures were read (0.1 in C_L, 0.2 in C_D). A misread of the
    drag trough as 1.6 (Fig. 3 has 0.98 on that mesh) fails here."""
    an = load_analyze()
    for which, tol in (("lift", 0.1), ("drag", 0.2)):
        for name in ("rotation peak", "trough", "acceleration peak"):
            fig3 = an.FIG3["coarse"][which][name]
            assert abs(an.REFERENCE[128][which][name][1] - fig3) < tol, (which, name)
    # the refinement lowers the rotation peaks (5.30 -> 4.89 and 1.85 -> 1.69)
    for which in ("lift", "drag"):
        assert an.FIG3["fine"][which]["rotation peak"] < an.FIG3["coarse"][which]["rotation peak"]
    # every reference landmark lies inside its own search window
    for re in (8, 128):
        for which in ("lift", "drag"):
            for name, (fr, _) in an.REFERENCE[re][which].items():
                lo, hi, _kind = an.WINDOWS[name]
                assert lo <= fr <= hi, (re, which, name)


def test_analyze_reads_force_files_without_t_force(tmp_path):
    """A wing_forces.csv written before the t_force column existed (tau of the output time
    t) is read with t_force = t - dt/2, row 0 at 0, and tau recomputed from it."""
    an = load_analyze()
    old_columns = ["step", "t", "tau", "fraction_of_stroke", "fx_left", "fy_left", "fx_right",
                   "fy_right", "CL_left", "CD_left", "CL_right", "CD_right"]
    dt, V, c = 0.002, 0.06, 0.05
    rows = []
    for step in (0, 5, 10):
        t = step * dt
        rows.append([float(step), t, t * V / c, t * V / c / 6.0] + [0.1 * step] * 8)
    path = tmp_path / "wing_forces.csv"
    with open(path, "w", encoding="ascii") as fh:
        fh.write(",".join(old_columns) + "\n")
        for r in rows:
            fh.write(",".join(repr(v) for v in r) + "\n")
    d = an.read_wing_forces(str(path))
    assert list(d.dtype.names[:5]) == ["step", "t", "t_force", "tau", "fraction_of_stroke"]
    assert np.allclose(d.t_force, [0.0, 0.009, 0.019])
    assert np.allclose(d.tau, d.t_force * V / c) and np.allclose(d.fraction_of_stroke, d.tau / 6.0)
    assert np.allclose(d.CL_left, [0.0, 0.5, 1.0])


def test_analyze_filter_recovers_a_ringing_trace():
    """A smooth two-peak curve plus a 0.038-period sinusoid of amplitude 0.4: the
    ringing_amplitude at that period is within 15 %, the one-period moving average brings
    the landmarks back to those of the smooth curve to within its own smoothing of the
    peaks (3 % of a Gaussian peak of width 0.06; the raw ones are off by the amplitude),
    landmarks beyond the run are None, and a clean curve has an amplitude below 0.05."""
    an = load_analyze()
    frac = np.linspace(0.0, 0.45, 1126)
    smooth_curve = 1.8 * np.exp(-((frac - 0.14) / 0.06) ** 2) + 2.6 * np.exp(-((frac - 0.33) / 0.06) ** 2)
    period, amp = an.FILTER_WINDOW["input2d"], 0.4
    noisy = smooth_curve + amp * np.sin(2 * np.pi * frac / period)
    names = ["fraction_of_stroke", "t", "CL_left", "CD_left", "CL_right", "CD_right"]
    d = np.rec.fromarrays([frac, frac * 5.0, noisy, noisy, noisy, noisy], names=names)
    assert abs(an.ringing_amplitude(d, period) - amp) < 0.15 * amp
    flt = an.landmarks(d, 128, period)
    raw = an.landmarks(d, 128)
    assert abs(flt["lift"]["rotation peak"][1] - 1.8) < 0.08 and abs(raw["lift"]["rotation peak"][1] - 1.8) > 0.25
    assert abs(flt["lift"]["trough"][1] - smooth_curve[(frac > 0.15) & (frac < 0.3)].min()) < 0.08
    assert abs(flt["lift"]["acceleration peak"][1] - 2.6) < 0.08 and flt["lift"]["end"] is None   # run stops at 0.45
    assert flt["lift"]["minimum during translation"] is None
    clean = np.rec.fromarrays([frac, frac * 5.0] + [smooth_curve] * 4, names=names)
    assert an.ringing_amplitude(clean, period) < 0.05
    assert an.ringing_amplitude(clean, None) == 0.0 and np.array_equal(an.smooth(frac, noisy, None), noisy)


def test_filter_box_has_the_documented_row_count():
    """The moving average of analyze.py is the odd row count nearest to window / row
    spacing. At the 0.002 spacing of the committed inputs (output every 5 of 2500 steps) the
    0.023 window gives 11.5, which floating point returns as 11.4999...; the box is 11 rows
    (0.022 of the stroke) and must stay 11 when the same ratio comes out as 11.5000...1
    (rounding that ratio gave 12, then 13 rows). Through 11 rows a sinusoid of period 0.023
    keeps 4.6 % of its amplitude (13 rows would keep 11 %)."""
    an = load_analyze()
    p = read_input_file(os.path.join(CASE, "input2d"))
    dt = p.time.dt
    t = np.arange(0, p.time.n_steps + 1, p.output.print_dump) * dt
    t_force = np.maximum(t - 0.5 * dt, 0.0)                    # the rows' time level
    frac = t_force / p.time.Tfinal
    window = an.FILTER_WINDOW["input2d"]
    assert window == 0.023 and np.isclose(np.median(np.diff(frac)), 0.002)
    assert an.box_rows(frac, window) == 11
    assert an.box_rows(frac, 3 * window) == 35                  # ringing_amplitude's detrend
    # just above the tie: still 11
    tie = np.arange(400) * (window / 11.5 * (1.0 - 1e-12))
    assert window / float(np.median(np.diff(tie))) > 11.5 and an.box_rows(tie, window) == 11
    # smooth is exactly that 11-row centred average in the interior
    rng = np.random.default_rng(0)
    y = rng.standard_normal(len(frac))
    ref = np.convolve(y, np.ones(11) / 11, mode="same")
    assert np.allclose(an.smooth(frac, y, window)[5:-5], ref[5:-5], rtol=0, atol=1e-13)
    # what it leaves of the ringing
    ring = np.sin(2 * np.pi * frac / window)
    left = np.abs(an.smooth(frac, ring, window)[20:-20]).max()
    assert abs(left - 0.046) < 0.002


# Bands for the slow tests: landmark value / paper value and |fraction - paper fraction|.
# They are what the 512^2 grid gives (wing gap 4.3 cells against the paper's 10 on 1230^2;
# the paper's own Fig. 3 shows the rotation peaks falling with refinement), measured in
# the README's runs: Re 128 lift 1.42 / 1.19 / 1.20, drag 1.36 / 0.91 / 1.13; Re 8 lift
# 1.07 / 0.88 / 0.99, drag 1.44 / 1.03 / 1.04 for rotation peak / trough / acceleration peak.
BANDS = {
    128: {"lift": {"rotation peak": (1.0, 1.7), "trough": (0.7, 1.7), "acceleration peak": (0.9, 1.5)},
          "drag": {"rotation peak": (1.0, 1.7), "trough": (0.6, 1.2), "acceleration peak": (0.9, 1.5)}},
    8: {"lift": {"rotation peak": (0.8, 1.35), "trough": (0.7, 1.3), "acceleration peak": (0.8, 1.25)},
        "drag": {"rotation peak": (1.0, 1.7), "trough": (0.8, 1.3), "acceleration peak": (0.8, 1.3)}},
}
# Largest amplitude of the tether ringing (C_L at the filter period) the slow runs accept:
# the damped Re 128 wings measured 0.011 (0.070 undamped); Re 8 is not filtered.
MAX_RINGING = 0.03
# Largest |fraction - paper fraction| of a landmark. The README's runs are off by at most
# 0.026 (Re 128 lift rotation peak, 0.134 against 0.16) and 0.024 (Re 8 drag acceleration
# peak, 0.296 against 0.32); the others by 0.020 or less. The README's "within 0.03 of the
# stroke" describes these; 0.04 leaves room for a shift of a few output rows (0.002 each).
MAX_FRACTION_OFFSET = 0.04


def check_rotation_phase(tmp_path, input_name: str, re: int, max_lag: float) -> dict:
    """Run the first 0.45 of the half-stroke at full resolution and compare the three
    landmarks the paper's figures show there, after the ringing filter of analyze.py."""
    res = run(CASE, input_file=input_name, steps=STEPS_TO_045, quiet=True, output_dir=str(tmp_path))
    h = res.history
    assert res.steps == STEPS_TO_045 and np.all(np.isfinite(h.view(float))) and h.cfl.max() < 1.0
    assert h.max_target_error_over_dx.max() < max_lag
    an = load_analyze()
    d = an.read_wing_forces(str(tmp_path / "wing_forces.csv"))
    assert abs(d.fraction_of_stroke[-1] - 0.45) < 1e-3 and len(d) == STEPS_TO_045 // 5 + 1
    assert np.array_equal(d.t_force, h.t_force)
    assert np.allclose(d.CL_left, d.CL_right, atol=1e-6 * np.abs(d.CL_left).max())
    window = an.FILTER_WINDOW[input_name]
    amp = an.ringing_amplitude(d, window)
    assert amp < MAX_RINGING
    got = an.landmarks(d, re, window)
    ref = an.REFERENCE[re]
    for which in ("lift", "drag"):
        for name, (lo, hi) in BANDS[re][which].items():
            fr_ref, val_ref = ref[which][name]
            fr, val = got[which][name]
            assert lo < val / val_ref < hi, (which, name, val, val_ref)
            assert abs(fr - fr_ref) < MAX_FRACTION_OFFSET, (which, name, fr, fr_ref)
        # the two peaks stand above the trough between them
        assert got[which]["trough"][1] < min(got[which]["rotation peak"][1], got[which]["acceleration peak"][1])
    print(f"\n{input_name}: {res.wall_time:.0f} s, max target error {h.max_target_error_over_dx.max():.3f} cells, "
          f"filter window {window}, ringing amplitude {amp:.2f}, landmarks {got}")
    return got


@pytest.mark.slow
def test_re128_fling_rotation_against_the_paper(tmp_path):
    """input2d (512^2, periodic) to 0.45 of the half-stroke: 48-463 s on a laptop shared with
    other jobs."""
    check_rotation_phase(tmp_path, "input2d", 128, max_lag=1.0)


@pytest.mark.slow
def test_re8_fling_rotation_against_the_paper(tmp_path):
    """input2d_re8 (512^2, periodic) to 0.45 of the half-stroke: 48-344 s on a laptop shared with
    other jobs."""
    check_rotation_phase(tmp_path, "input2d_re8", 8, max_lag=1.2)
