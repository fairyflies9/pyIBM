"""examples/sedimenting_ring: smoke runs of the coarse and rigid inputs, the committed files
against generate.py, the gravity ramp of hooks.py, the analysis helpers, the weight transfer
while the ring is still slow, and (slow) the fine run to the end with the weight transfer,
area conservation, the speed and its slope checked against the README."""

from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest

from pyibm.driver import run
from pyibm.inputfile import read_input_file
from pyibm.io.history import read_history
from pyibm.structures import (load_hooks, read_beam, read_mass, read_spring, read_structure,
                              read_target, read_vertex)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CASE = os.path.join(ROOT, "examples", "sedimenting_ring")


def load_generate():
    spec = importlib.util.spec_from_file_location("sedimenting_ring_generate_test",
                                                  os.path.join(CASE, "generate.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_analyze():
    spec = importlib.util.spec_from_file_location("sedimenting_ring_analyze_test",
                                                  os.path.join(CASE, "analyze.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_smoke_three_steps(tmp_path):
    """Three midpoint steps of the coarse ring in the closed box: history rows [0, 3], finite
    diagnostics, the ring still a circle (gravity is ramped in from zero), ring.csv written by
    the on_output hook, mass points present and gravity pointing down."""
    res = run(CASE, input_file="input2d_coarse", steps=3, quiet=True, output_dir=str(tmp_path))
    assert res.steps == 3 and res.history.step.tolist() == [0.0, 3.0]   # print_dump = 100
    assert np.all(np.isfinite(res.history.view(float)))
    s = res.structure
    assert s.n == 64 and len(s.springs) == 64 and len(s.beams) == 64 and len(s.masses) == 64
    assert s.targets is None
    assert s.gravity[0] == 0.0 and -9.81 < s.gravity[1] <= 0.0     # ramped, set by the hook
    assert res.grid.periodic_x is False and res.grid.periodic_y is False
    assert os.path.isfile(tmp_path / "history.csv") and os.path.isfile(tmp_path / "lag.00003.vtk")
    r = read_history(tmp_path / "ring.csv")
    assert len(r) == 2 and r.step.tolist() == [0.0, 3.0]
    assert abs(r.area[-1] / r.area[0] - 1.0) < 1e-6
    assert r.max_stretch[-1] < 1e-4 and r.ghost_gap_over_dx[-1] < 1e-3
    assert r.bottom_gap_over_dx[-1] > 2.0
    # the fluid starts to carry a little of the (ramped) weight: an upward force on the ring
    assert res.history.fy_body[-1] >= 0.0


def test_smoke_rigid_reference(tmp_path):
    res = run(CASE, input_file="input2d_rigid", steps=3, quiet=True, output_dir=str(tmp_path))
    assert res.steps == 3 and res.history.step.tolist() == [0.0, 3.0]
    s = res.structure
    assert s.targets is not None and len(s.targets) == 129 and s.masses is None and s.springs is None
    assert res.history.max_target_error_over_dx[-1] < 1e-3   # the targets have barely moved
    assert np.all(np.isfinite(res.history.view(float)))
    r = read_history(tmp_path / "ring.csv")
    assert len(r) == 2 and r.ghost_gap_over_dx[-1] == 0.0


def test_committed_files_match_generate():
    gen = load_generate()
    assert gen.RADIUS == 0.08 and gen.LX == 0.5 and gen.LY == 1.0 and gen.GRAVITY == 9.81
    for name, input_name, n in (("ring", "input2d", 129), ("ring_coarse", "input2d_coarse", 64),
                                ("ring_rigid", "input2d_rigid", 129)):
        X, ds = gen.ring_points(name)
        assert X.shape == (n, 2)
        rad = np.hypot(X[:, 0] - gen.CENTRE[0], X[:, 1] - gen.CENTRE[1])
        assert np.allclose(rad, gen.RADIUS) and abs(ds - 2 * np.pi * gen.RADIUS / n) < 1e-15
        V = read_vertex(os.path.join(CASE, name + ".vertex"))
        assert np.array_equal(V, X)
        p = read_input_file(os.path.join(CASE, input_name))
        c = gen.CASES[name]
        assert (p.grid.Nx, p.grid.Ny, p.grid.Lx, p.grid.Ly) == (c["Nx"], c["Ny"], gen.LX, gen.LY)
        assert p.time.dt == c["dt"] and p.time.Tfinal == c["Tfinal"]
        assert (p.fluid.rho, p.fluid.mu, p.structure.ds, p.structure.name) == (gen.RHO, gen.MU, ds, name)
        assert p.solver.backend == "mac" and p.solver.time_scheme == "midpoint"
        assert (p.bc.left, p.bc.right, p.bc.bottom, p.bc.top) == ("dirichlet",) * 4
        assert p.structure.gravity == (0.0, -gen.GRAVITY)
        assert abs(ds - 0.5 * p.grid.dx) < 0.02 * ds       # ds = dx/2 up to the rounding that closes the ring
        if name == "ring_rigid":
            T = read_target(os.path.join(CASE, name + ".target"), V)
            assert np.all(T.k == gen.K_TARGET) and T.idx.tolist() == list(range(n))
            continue
        sp = read_spring(os.path.join(CASE, name + ".spring"), n)
        assert len(sp) == n and np.allclose(sp.k, gen.k_spring(ds)) and np.allclose(sp.k * ds ** 2, gen.EA)
        chord = 2 * gen.RADIUS * np.sin(np.pi / n)
        assert np.allclose(sp.rest, chord)
        bm = read_beam(os.path.join(CASE, name + ".beam"), n)
        assert len(bm) == n and np.allclose(bm.k, gen.k_beam(ds)) and np.allclose(bm.k * ds ** 6, gen.EI)
        # preferred cross product = the circle's: chord^2 sin(2 pi / n) up to sign (counter-clockwise)
        assert np.allclose(np.abs(bm.curv), chord ** 2 * np.sin(2 * np.pi / n))
        assert np.allclose(bm.curv, bm.curv[0], rtol=1e-12)
        ms = read_mass(os.path.join(CASE, name + ".mass"), V)
        assert np.all(ms.k == gen.K_MASS) and np.all(ms.m == gen.MASS)


def test_gravity_ramp_and_rigid_motion():
    gen = load_generate()
    hooks = load_hooks(CASE)
    m = hooks.module
    assert m.ramp(0.0) == 0.0 and m.ramp(gen.T_RAMP) == 1.0 and m.ramp(0.5 * gen.T_RAMP) == pytest.approx(0.5)
    s = read_structure(CASE, "ring", ds=gen.ring_points("ring")[1], gravity=(0.0, -gen.GRAVITY))
    hooks.update_masses(0.0, 1e-3, s)
    assert s.gravity == (0.0, 0.0)
    hooks.update_masses(10.0, 1e-3, s)
    assert s.gravity == (0.0, -gen.GRAVITY)
    hooks.update_targets(1.0, 1e-3, s)       # no targets on the elastic ring: a no-op
    # the integral of the ramp by quadrature
    t = np.linspace(0.0, gen.T_RAMP, 20001)
    f = np.array([m.ramp(v) for v in t])
    quad = float(np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(t)))
    assert abs(m.ramp_integral(gen.T_RAMP) - quad) < 1e-8
    assert m.ramp_integral(2.0) == pytest.approx(0.5 * gen.T_RAMP + 2.0 - gen.T_RAMP)
    r = read_structure(CASE, "ring_rigid", ds=gen.ring_points("ring_rigid")[1])
    hooks.update_targets(2.0, 1e-3, r)
    assert np.allclose(r.targets.X[:, 0], r.X0[:, 0])
    assert np.allclose(r.X0[:, 1] - r.targets.X[:, 1], gen.U_RIGID * m.ramp_integral(2.0))
    assert np.allclose(r.targets.V[:, 1], -gen.U_RIGID)


def test_analysis_helpers():
    """analyze.speed_trend recovers the slope of a noisy straight line over the last 0.5 s,
    and analyze.gravity_ramp is the ramp of hooks.py."""
    ana = load_analyze()
    gen = load_generate()
    t = np.arange(0.0, 3.0 + 1e-9, 0.05)
    rng = np.random.default_rng(1)
    U = 0.1 + 0.006 * t + 1e-4 * rng.standard_normal(t.size)
    U[t < 2.5] += 0.05                                    # only the last 0.5 s counts
    slope, mean = ana.speed_trend(t, U)
    assert abs(slope - 0.006) < 1e-3 and abs(mean - (0.1 + 0.006 * 2.75)) < 2e-4
    m = load_hooks(CASE).module
    tt = np.linspace(0.0, 0.5, 101)
    assert np.allclose(ana.gravity_ramp(tt, gen.T_RAMP), [m.ramp(v) for v in tt], atol=1e-15)


def test_weight_reaches_the_fluid_long_before_terminal_speed(tmp_path):
    """fy_body / (M g) is not a read-out of terminal speed: the ghost update makes fy_body =
    M (g r(t) - a_ghost), and the ring's mass is small next to the fluid it carries, so the
    ratio follows the gravity ramp r(t) closely while the ring has barely started to move.
    The coarse ring to t = 0.25 s (500 steps): at every output the ratio is within 3 % of the
    ramp at the force's time level, and at the end it is within 3 % of 1 while the ring is
    at less than a fifth of its speed at t = 3 s (0.12 m/s on this grid, README)."""
    ana = load_analyze()
    gen = load_generate()
    res = run(CASE, input_file="input2d_coarse", steps=500, quiet=True, output_dir=str(tmp_path))
    h = res.history
    Mg = gen.MASS * 2 * np.pi * gen.RADIUS * gen.GRAVITY
    ratio = h.fy_body / Mg
    assert np.max(np.abs(ratio - ana.gravity_ramp(h.t_force, gen.T_RAMP))) < 0.03
    assert abs(ratio[-1] - 1.0) < 0.03
    r = read_history(tmp_path / "ring.csv")
    assert r.t[-1] == pytest.approx(0.25) and -r.vc[-1] < 0.2 * 0.12


@pytest.mark.slow
def test_fine_run_settles(tmp_path):
    """The 64 x 128 ring for 3 s (12000 steps, about 2 minutes): the weight reaches the fluid
    (fy_body / M g within 2 % of 1 over the last 0.5 s; a check of the tethers, not of
    terminal speed), the enclosed area drifts by less than 0.3 %, the ring stretches less
    than 5 %, the speed over the last 0.5 s is 0.11 to 0.16 m/s (measured 0.135) and is still
    rising, slowly (a straight-line fit gives dU/dt / U = 4.6 % per second; we accept 0 to
    10 %), and the ring stays well above the floor."""
    gen = load_generate()
    ana = load_analyze()
    res = run(CASE, quiet=True, output_dir=str(tmp_path))
    assert res.steps == 12000
    r = read_history(tmp_path / "ring.csv")
    h = res.history
    Mg = gen.MASS * 2 * np.pi * gen.RADIUS * gen.GRAVITY
    last = h.t >= h.t[-1] - 0.5
    assert abs(np.mean(h.fy_body[last]) / Mg - 1.0) < 0.02
    assert np.max(np.abs(r.area / r.area[0] - 1.0)) < 3e-3
    assert r.max_stretch.max() < 0.05
    U = float(np.mean(-r.vc[r.t >= r.t[-1] - 0.5]))
    assert 0.11 < U < 0.16
    dUdt, _ = ana.speed_trend(r.t, -r.vc)
    assert 0.0 < dUdt / U < 0.10
    assert r.bottom_gap_over_dx.min() > 20.0
    assert np.all(h.cfl < 0.5)
    print(f"\nfine ring: {res.wall_time:.1f} s, speed {U:.4f} m/s, dU/dt / U {dUdt / U:.4f} /s, fy/Mg "
          f"{np.mean(h.fy_body[last]) / Mg:.4f}, area drift {np.max(np.abs(r.area / r.area[0] - 1.0)):.2e}")
