"""examples/flapping_plate_re64: smoke runs of both inputs, the committed files against
generate.py, the kinematics against the paper's definitions, the hooks as functions of
absolute time, the analysis (on synthetic histories, with and without the t_force column),
and (slow) cycles of the periodic run against Jones et al. 2015."""

from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pytest

from pyibm.driver import TETHER_STABLE, Simulation, run
from pyibm.inputfile import read_input_file
from pyibm.io.history import read_history
from pyibm.structures import load_hooks, read_structure, read_target, read_vertex

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CASE = os.path.join(ROOT, "examples", "flapping_plate_re64")
INPUTS = ("input2d", "input2d_mac")


def load_sibling(name: str):
    modname = f"flapping_plate_re64_{name}"
    if modname in sys.modules:
        return sys.modules[modname]
    spec = importlib.util.spec_from_file_location(modname, os.path.join(CASE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("input_name", INPUTS)
def test_smoke_three_steps(tmp_path, input_name):
    res = run(CASE, input_file=input_name, steps=3, quiet=True, output_dir=str(tmp_path))
    assert res.steps == 3 and res.history.step.tolist() == [0.0, 3.0]   # print_dump = 8
    assert np.all(np.isfinite(res.history.view(float)))
    assert res.structure.n == 32 and res.structure.targets is not None
    assert res.history.max_target_error_over_dx[-1] < 0.1
    # the plate starts pitching nose-down at its peak rate, so the fluid pushes back at once
    assert res.history.fy_body[-1] != 0.0 and res.history.fx_body[-1] != 0.0
    assert res.solver.kinetic_energy() > 0.0
    assert os.path.isfile(tmp_path / "history.csv")
    assert len(read_history(tmp_path / "history.csv")) == 2
    assert os.path.isfile(tmp_path / "vorticity.00000.png")   # on_output at step 0
    assert not os.path.isfile(tmp_path / "vorticity.00003.png")


def test_committed_files_match_generate():
    gen = load_sibling("generate")
    X = gen.initial_positions()
    assert X.shape == (32, 2)
    V = read_vertex(os.path.join(CASE, "plate.vertex"))
    assert np.array_equal(V, X)
    T = read_target(os.path.join(CASE, "plate.target"), V)
    assert T.idx.tolist() == list(range(32)) and np.all(T.k == gen.TARGET_STIFFNESS)
    # the paper's stroke and the dimensional choices of generate.py
    assert (gen.CHORD, gen.STROKE_OVER_CHORD, gen.BETA_DEG) == (0.1, 2.5, 62.8)
    assert (gen.ALPHA_MEAN_DEG, gen.ALPHA_AMP_DEG) == (60.0, 45.0)
    assert (gen.RHO, gen.UMAX, gen.RE) == (10.0, 24.0, 64.0)
    assert gen.MU == pytest.approx(0.375) and gen.Q == pytest.approx(144.0)
    assert gen.PERIOD == pytest.approx(np.pi * 0.25 / 24.0)
    for input_name, backend in zip(INPUTS, ("periodic_fft", "mac")):
        p = read_input_file(os.path.join(CASE, input_name))
        assert p.solver.backend == backend
        assert (p.solver.time_scheme, p.solver.coupling, p.solver.kernel) == ("midpoint", "delta", "peskin4")
        assert p.solver.remove_mean_force == (backend == "periodic_fft")
        assert p.time.dt == gen.time_step() and p.time.n_steps == gen.CYCLES * gen.STEPS_PER_CYCLE
        assert (p.grid.Nx, p.grid.Ny) == (gen.BOX_CHORDS * gen.CELLS_PER_CHORD,) * 2
        assert (p.grid.Lx, p.grid.Ly) == (gen.LX, gen.LY)
        assert (p.fluid.rho, p.fluid.mu) == (gen.RHO, gen.MU)
        assert p.structure.name == "plate" and p.structure.ds == gen.marker_spacing()
        assert p.output.print_dump == gen.STEPS_PER_CYCLE // gen.ROWS_PER_CYCLE
        assert not p.output.save_vtk and p.output.save_history
        side = "periodic" if backend == "periodic_fft" else "dirichlet"
        assert (p.bc.left, p.bc.right, p.bc.bottom, p.bc.top) == (side,) * 4
    assert gen.tether_number() < TETHER_STABLE[("target", "midpoint")]
    assert gen.tether_number() == pytest.approx(0.418, abs=1e-3)       # the README's value


def test_kinematics_match_the_paper():
    """Eqs. (2)-(3) and (10) of Jones et al. 2015 in the centred form of kinematics.py."""
    gen = load_sibling("generate")
    K = gen.stroke()
    b = gen.body_points(gen.marker_spacing())
    assert len(b) == 32 and np.allclose(b[:, 1], 0.0) and np.isclose(b[:, 0].mean(), 0.0)
    assert np.isclose(b[-1, 0] - b[0, 0], gen.CHORD - gen.marker_spacing())
    # peak speed, rms speed, Reynolds number, force scale
    t = np.linspace(0.0, K.period, 20001)
    assert np.max(np.abs(K.s_dot(t))) == pytest.approx(gen.UMAX, rel=1e-8)
    assert np.sqrt(np.mean(K.s_dot(t) ** 2)) == pytest.approx(gen.UMAX / np.sqrt(2), rel=1e-4)
    assert gen.kin.reynolds_number(gen.RHO, K.u_max, gen.CHORD, gen.MU) == pytest.approx(64.0)
    assert gen.kin.force_scale(gen.RHO, K.u_rms, gen.CHORD) == pytest.approx(0.5 * 10 * 288 * 0.1)
    # t = 0: upper end of the stroke, alpha = alpha_0, at rest in translation; then the downstroke
    beta = np.radians(62.8)
    assert np.allclose(K.centre(0.0), [0.6 + 0.125 * np.cos(beta), 0.6 + 0.125 * np.sin(beta)])
    assert K.alpha(0.0) == pytest.approx(np.radians(60.0))
    assert K.theta(0.0) == pytest.approx(beta + np.radians(60.0) + np.pi)
    assert np.allclose(K.centre_velocity(0.0), 0.0, atol=1e-12)
    # Fig. 2c: the leading edge leads both half strokes. The leading-edge-to-trailing-edge
    # vector has a negative component along the motion at mid-downstroke (plate 75 deg to
    # its path, leading edge slightly ahead) and at mid-upstroke (feathered, 15 deg).
    for tt in (K.period / 4, 3 * K.period / 4):
        le_to_te = np.array([np.cos(K.theta(tt)), np.sin(K.theta(tt))])
        assert le_to_te @ K.motion_direction(tt) < 0.0
    # alpha is the counterclockwise angle from the stroke direction (cos beta, sin beta) to the
    # trailing-edge-to-leading-edge vector: 105 deg at mid-downstroke, 15 deg at mid-upstroke
    for tt, expected in ((K.period / 4, 105.0), (3 * K.period / 4, 15.0)):
        te_to_le = -np.array([np.cos(K.theta(tt)), np.sin(K.theta(tt))])
        d = K.direction
        angle = np.arctan2(d[0] * te_to_le[1] - d[1] * te_to_le[0], d @ te_to_le)
        assert np.degrees(angle) == pytest.approx(expected)
    mirror = gen.kin.StrokeKinematics(K.stroke, K.period, K.beta, K.alpha_mean, K.alpha_amp,
                                      K.x_mid, K.y_mid, leading_edge_forward=False)
    assert mirror.theta(0.0) == pytest.approx(beta - np.radians(60.0))
    le_to_te = np.array([np.cos(mirror.theta(3 * K.period / 4)), np.sin(mirror.theta(3 * K.period / 4))])
    assert le_to_te @ mirror.motion_direction(3 * K.period / 4) > 0.0     # trailing edge leads
    assert np.allclose(K.centre(K.period / 4), [0.6, 0.6])
    assert np.allclose(K.motion_direction(K.period / 4), [-np.cos(beta), -np.sin(beta)])
    assert K.alpha(K.period / 4) == pytest.approx(np.radians(105.0))
    assert K.alpha(3 * K.period / 4) == pytest.approx(np.radians(15.0))
    assert np.allclose(K.motion_direction(3 * K.period / 4), [np.cos(beta), np.sin(beta)])
    # rigid-body velocities against a finite difference of the positions; inverse map
    eps = 1e-7
    for tt in np.linspace(0.0, K.period, 9):
        V = K.velocities(b, tt)
        Vfd = (K.positions(b, tt + eps) - K.positions(b, tt - eps)) / (2 * eps)
        assert np.allclose(V, Vfd, rtol=1e-6, atol=1e-6 * gen.UMAX)
        assert np.allclose(K.body_coordinates(K.positions(b, tt), tt), b, atol=1e-12)
    # the midpoint recovered from the initial pose is the box centre
    K2 = gen.stroke_from_initial_pose(gen.initial_positions())
    assert (K2.x_mid, K2.y_mid) == pytest.approx((0.6, 0.6))


def test_hooks_are_functions_of_absolute_time():
    gen = load_sibling("generate")
    hooks = load_hooks(CASE)
    s = read_structure(CASE, "plate", ds=gen.marker_spacing())
    K = gen.stroke()
    b = K.body_coordinates(s.X0, 0.0)
    dt = gen.time_step()
    hooks.update_targets(0.0, dt, s)
    assert np.allclose(s.targets.X, s.X0, atol=1e-12)
    for t in (0.3 * K.period, 1.7 * K.period, 0.3 * K.period):     # back and forth in time
        hooks.update_targets(t, dt, s)
        assert np.allclose(s.targets.X, K.positions(b, t), atol=1e-12)
        assert np.allclose(s.targets.V, K.velocities(b, t), atol=1e-9)
    once = s.targets.X.copy()
    hooks.update_targets(0.3 * K.period, dt, s)
    assert np.array_equal(s.targets.X, once)                         # idempotent
    # every marker stays well inside the box over a cycle (2 dx from the walls for the MAC run)
    for t in np.linspace(0.0, K.period, 50):
        hooks.update_targets(t, dt, s)
        assert np.all(s.targets.X > 2 * gen.grid_spacing()) and np.all(s.targets.X < gen.LX - 2 * gen.grid_spacing())


def synthetic_history(gen, cycles: int = 2, t_force: bool = True):
    """A made-up history with 800 rows per cycle, F_y = q (1 + 2 sin(2 pi t_f/T)) and
    F_x = q cos(2 pi t_f/T) at the force time t_f = t - dt/2 (0 in row 0), with or without
    the t_force column."""
    K = gen.stroke()
    n = cycles * gen.ROWS_PER_CYCLE
    t = np.arange(n + 1) * (K.period / gen.ROWS_PER_CYCLE)
    tf = np.maximum(t - 0.5 * gen.time_step(), 0.0)
    fy = gen.Q * (1.0 + 2.0 * np.sin(2 * np.pi * tf / K.period))
    fx = gen.Q * np.cos(2 * np.pi * tf / K.period)
    lag = 0.01 * np.arange(n + 1) / n
    if t_force:
        return np.rec.fromarrays([t, tf, fx, fy, lag],
                                 names=["t", "t_force", "fx_body", "fy_body", "max_target_error_over_dx"])
    return np.rec.fromarrays([t, fx, fy, lag], names=["t", "fx_body", "fy_body", "max_target_error_over_dx"])


def test_analyze_on_a_synthetic_history():
    """analyze.coefficients / cycle_means / running_mean on a made-up two-cycle history
    (synthetic_history), so the cycle means, extrema, phases and the motion-frame lift and
    drag are known."""
    gen = load_sibling("generate")
    analyze = load_sibling("analyze")
    K = gen.stroke()
    h = synthetic_history(gen)
    n = len(h) - 1
    tf = h.t_force
    fx, fy = h.fx_body, h.fy_body
    c = analyze.coefficients(h)
    assert c["smooth_rows"] == round(analyze.SMOOTH_CYCLES * gen.ROWS_PER_CYCLE)
    assert np.allclose(c["CV"], 1.0 + 2.0 * np.sin(2 * np.pi * c["tau"]))
    # lift along the upward normal and drag against the motion at the force time, NaN where
    # the plate does not translate (row 0, the start of the downstroke)
    nrm, d = K.normal_up, K.motion_direction(tf)
    assert np.allclose(c["CL"], (fx * nrm[0] + fy * nrm[1]) / gen.Q)
    moving = np.isfinite(c["CD"])
    assert not moving[0] and moving[1] and moving[gen.ROWS_PER_CYCLE // 2]
    assert np.allclose(c["CD"][moving], -(fx * d[:, 0] + fy * d[:, 1])[moving] / gen.Q)
    rows = analyze.cycle_means(c)
    assert [r["cycle"] for r in rows] == [1, 2] and all(r["rows"] == gen.ROWS_PER_CYCLE for r in rows)
    for r in rows:
        # 800 rows a cycle, uniform in the force time apart from row 0
        assert r["CV"] == pytest.approx(1.0, abs=1e-5) and r["CH"] == pytest.approx(0.0, abs=1e-5)
        assert r["CV_max_raw"] == pytest.approx(3.0) and r["CV_max_phase_raw"] == pytest.approx(0.25, abs=2e-4)
        assert r["CV_min_raw"] == pytest.approx(-1.0) and r["CV_min_phase_raw"] == pytest.approx(0.75, abs=2e-4)
        # the 17-row running mean trims a sinusoid's peak by 2 * (pi * 0.02)^2 / 6 = 0.0013
        assert r["CV_max"] == pytest.approx(3.0, abs=5e-3) and r["CV_max_phase"] == pytest.approx(0.25, abs=0.01)
        assert r["CH_max"] == pytest.approx(1.0, abs=5e-3) and r["CH_min"] == pytest.approx(-1.0, abs=5e-3)
        assert r["CV_over_CT"] == pytest.approx(r["CV"] / r["CT"])
    assert rows[1]["lag"] == pytest.approx(0.01 * (n - 1) / n)
    # the running mean keeps a constant, averages a sawtooth away and leaves one row alone
    assert np.allclose(analyze.running_mean(np.full(10, 2.0), 5), 2.0)
    saw = np.where(np.arange(100) % 2 == 0, 1.0, -1.0)
    assert np.all(np.abs(analyze.running_mean(saw, 3)[2:-2]) < 0.34)
    assert np.array_equal(analyze.running_mean(saw, 1), saw)


def test_analyze_uses_the_force_time():
    """The coefficients and their phases are those of the force time t_force (dt/2 before the
    row's t), not of t; a row stays in the cycle of its own t, so the first row of a cycle
    has phase -dt/(2T). A history without t_force is read with t - dt/2 (0 in row 0)."""
    gen = load_sibling("generate")
    analyze = load_sibling("analyze")
    K = gen.stroke()
    h = synthetic_history(gen)
    c = analyze.coefficients(h)
    assert np.array_equal(c["t"], h.t_force) and np.allclose(c["tau"], h.t_force / K.period)
    first = gen.ROWS_PER_CYCLE                                  # t = T, t_force = T - dt/2
    assert c["cycle"][first] == 1
    assert c["phase"][first] == pytest.approx(-0.5 * gen.time_step() / K.period, rel=1e-9)
    old = synthetic_history(gen, t_force=False)
    assert np.allclose(analyze.force_time(old), h.t_force, rtol=0.0, atol=1e-15)
    c_old = analyze.coefficients(old)
    assert np.allclose(c_old["tau"], c["tau"]) and np.allclose(c_old["CD"], c["CD"], equal_nan=True)


@pytest.mark.slow
def test_periodic_run_against_jones2015(tmp_path):
    """Two cycles of the committed periodic input (16 cells per chord, 12800 steps). Cycle 2
    is compared with the rigid-plate values read off Jones et al. 2015 Fig. 11A, Fig. 12A
    and Fig. 21(b.ii): C_V positive and peaking during the downstroke, negative around the
    reversal, cycle-mean C_V near 1 (paper: 1.00 at Re 60, 1.10 at Re 80, so about 1.02 at
    Re 64), C_H positive early in the downstroke and negative through the upstroke. The
    tolerances are wide because of the coarse grid, the shorter run and the smaller periodic
    box; the README has the numbers. Measured: mean C_V 1.12, C_V max 7.3 at phase 0.16, min
    -3.0 at 0.47, C_H max 4.1, min -3.7, max lag 0.126 cells; 97-539 s on a laptop shared with
    other jobs (7.6-42 ms per step)."""
    gen = load_sibling("generate")
    analyze = load_sibling("analyze")
    res = run(CASE, input_file="input2d", steps=2 * gen.STEPS_PER_CYCLE, quiet=True,
              output_dir=str(tmp_path))
    h = res.history
    assert np.all(np.isfinite(h.view(float)))
    rows = analyze.cycle_means(analyze.coefficients(h))
    assert len(rows) == 2
    r = rows[1]
    print(f"\ncycle 2: mean CV {r['CV']:.3f} (paper ~{analyze.REFERENCE['mean_CV_re64']:.2f}), "
          f"CV max {r['CV_max']:.2f} at phase {r['CV_max_phase']:.2f}, min {r['CV_min']:.2f} at "
          f"{r['CV_min_phase']:.2f}, CH max {r['CH_max']:.2f}, min {r['CH_min']:.2f}, "
          f"max lag {h.max_target_error_over_dx.max():.4f} cells, {res.wall_time:.0f} s")
    assert 0.5 < r["CV"] < 1.8
    assert 4.0 < r["CV_max"] < 12.0 and 0.1 < r["CV_max_phase"] < 0.35
    assert -8.0 < r["CV_min"] < -1.0 and 0.45 < r["CV_min_phase"] < 0.8
    assert r["CH_max"] > 1.5 and r["CH_max_phase"] < 0.3
    assert r["CH_min"] < -1.5
    assert h.max_target_error_over_dx.max() < 0.3
    assert np.all(h.cfl < 0.5)
    assert abs(h.mean_u[-1]) < 1e-9 and abs(h.mean_v[-1]) < 1e-9     # remove_mean_force


@pytest.mark.slow
def test_mac_run_one_cycle_coarse(tmp_path):
    """One cycle of the no-slip-box variant at 8 cells per chord (96^2), generated into a
    temporary directory with generate.write_case: the first downstroke gives the same
    C_V peak as the periodic run within 20 % (the far walls hardly matter). 46-172 s on a
    laptop shared with other jobs (both back ends together)."""
    gen = load_sibling("generate")
    analyze = load_sibling("analyze")
    hooks = load_hooks(CASE)
    hooks.on_output = None
    P, spc = 8, 3200
    gen.write_case(str(tmp_path), cells_per_chord=P, stiffness=gen.TARGET_STIFFNESS / 2,
                   steps_per_cycle=spc, cycles=1)
    peaks = {}
    for backend in ("periodic_fft", "mac"):
        params = gen.parameters(backend, "viz_" + backend, P, spc, 1)
        sim = Simulation(params, str(tmp_path), hooks=hooks, quiet=True)
        res = sim.run()
        h = res.history
        assert np.all(np.isfinite(h.view(float)))
        rows = analyze.cycle_means(analyze.coefficients(h))
        peaks[backend] = rows[0]["CV_max"]
        assert h.max_target_error_over_dx.max() < 0.3
    assert abs(peaks["mac"] - peaks["periodic_fft"]) < 0.2 * abs(peaks["periodic_fft"])
