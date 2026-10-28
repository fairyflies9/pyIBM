"""examples/forced_circle: smoke runs of both couplings, the committed files against
generate.py, the target motion of hooks.py, analyze.py (the warning notes and the figure on
short runs), and (slow) the time-step study of analyze.py with the second-order trend of the
midpoint scheme."""

from __future__ import annotations

import importlib.util
import os
import re
import warnings

import numpy as np
import pytest

from pyibm.driver import TETHER_STABLE, run
from pyibm.inputfile import read_input_file
from pyibm.io.history import read_history
from pyibm.structures import load_hooks, read_spring, read_structure, read_target, read_vertex

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CASE = os.path.join(ROOT, "examples", "forced_circle")
INPUTS = (("input2d", "delta"), ("input2d_blob", "blob"))


def load(name: str):
    spec = importlib.util.spec_from_file_location(f"forced_circle_{name}", os.path.join(CASE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("input_name, coupling", INPUTS)
def test_smoke_three_steps(tmp_path, input_name, coupling):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = run(CASE, input_file=input_name, steps=3, quiet=True, output_dir=str(tmp_path))
    assert not [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
    assert res.parameters.solver.coupling == coupling
    assert res.steps == 3 and res.history.step.tolist() == [0.0, 3.0]   # print_dump = 32
    assert np.all(np.isfinite(res.history.view(float)))
    assert res.structure.n == 400 and len(res.structure.targets) == 201 and res.structure.springs is not None
    assert len(read_history(tmp_path / "history.csv")) == 2
    assert os.path.isfile(tmp_path / "lag.00003.vtk")
    # the targets have moved outward (a(t) grows) and the fluid pushes back in -x
    assert 0.0 < res.history.max_target_error_over_dx[-1] < 1.0
    assert res.history.fx_body[-1] < 0.0
    assert abs(res.history.fy_body[-1]) < 1e-6 * abs(res.history.fx_body[-1])     # symmetric about y = 1/2
    assert res.history.kinetic_energy[-1] > 0.0 and res.history.cfl[-1] < 0.5
    # the box mean is held at zero (remove_mean_force) although the net force is not zero
    assert abs(res.history.mean_u[-1]) < 1e-12 and abs(res.history.mean_v[-1]) < 1e-12


def test_committed_files_match_generate(tmp_path):
    gen = load("generate")
    written = gen.main(str(tmp_path))
    assert sorted(os.path.basename(p) for p in written) == sorted(
        ["circle.vertex", "circle.spring", "circle.target", "input2d", "input2d_blob"])
    for path in written:   # byte for byte, headers included
        with open(path, "rb") as a, open(os.path.join(CASE, os.path.basename(path)), "rb") as b:
            assert a.read() == b.read(), os.path.basename(path)
    X = gen.circle_points()
    assert X.shape == (400, 2)
    V = read_vertex(os.path.join(CASE, "circle.vertex"))
    assert np.allclose(V, X, rtol=0, atol=1e-15)
    assert np.allclose(np.hypot(X[:, 0] - 0.5, X[:, 1] - 0.5), 0.2, rtol=1e-12)
    seg = np.hypot(*(np.roll(X, -1, axis=0) - X).T)
    assert np.allclose(seg, 2 * 0.2 * np.sin(np.pi / 400), rtol=1e-10)      # the chord of 2 pi/400
    sp = read_spring(os.path.join(CASE, "circle.spring"), 400)
    assert sp.i.tolist() == list(range(400)) and sp.j.tolist() == [(i + 1) % 400 for i in range(400)]
    assert np.all(sp.k == gen.K_SPRING) and np.all(sp.rest == 0.0)       # a membrane under tension k ds L
    assert gen.K_SPRING * seg.mean() ** 2 == pytest.approx(0.0987, rel=1e-3)   # the tension k ds L, about 0.1 N
    tg = read_target(os.path.join(CASE, "circle.target"), V)
    idx = gen.right_half()
    assert tg.idx.tolist() == idx.tolist() and np.all(tg.k == gen.K_TARGET)
    # the right half: cos theta >= 0, which is 201 of the 400 points (theta = +-pi/2 included)
    assert len(idx) == 201 and np.all(X[idx, 0] >= 0.5 - 1e-12)
    assert np.all(X[np.setdiff1d(np.arange(400), idx), 0] < 0.5)
    for input_name, coupling in INPUTS:
        p = read_input_file(os.path.join(CASE, input_name))
        assert p == gen.parameters(coupling, "viz/" + coupling)
        assert p.solver.coupling == coupling and p.solver.backend == "periodic_fft"
        assert p.solver.remove_mean_force is True and p.solver.time_scheme == "midpoint"
        assert (p.grid.Nx, p.fluid.mu, p.fluid.rho, p.time.Tfinal) == (128, 1e-4, 1.0, 1.0)
        assert p.time.dt == 1 / 2048 and p.time.n_steps == 2048
    pb = read_input_file(os.path.join(CASE, "input2d_blob"))
    assert pb.solver.blob == "gaussian4" and pb.solver.blob_delta == 4.0
    assert pb.solver.blob_interpolation == "streamfunction" and pb.hooks.file == "hooks.py"
    # the tether numbers k dt^2/(rho dx) quoted in generate.py and the README
    assert gen.K_SPRING * gen.DT ** 2 / (gen.RHO * gen.DX) == pytest.approx(0.305, abs=1e-3)
    assert gen.K_TARGET * gen.DT ** 2 / (gen.RHO * gen.DX) == pytest.approx(0.0977, abs=1e-4)
    # the driver's rule of thumb (targets only) is not reached at any level of the study,
    # the coarsest dt = dx/2 included (6.25 against 10)
    an = load("analyze")
    for factor in an.FACTORS + (an.REFERENCE,):
        dt = gen.DX / factor
        assert gen.K_TARGET * dt ** 2 / (gen.RHO * gen.DX) < TETHER_STABLE[("target", "midpoint")]


def test_hook_is_fovargues_motion():
    """T(theta, t) = (0.5 + a cos theta, 0.5 + b sin theta), a = c(1 + eps sin 2 pi t),
    b = c/(1 + eps sin 2 pi t), c = 0.2, eps = 0.4 (thesis section 5.4.0.1)."""
    gen = load("generate")
    hooks = load_hooks(CASE)
    s = read_structure(CASE, "circle", ds=read_input_file(os.path.join(CASE, "input2d")).structure.ds)
    th = 2 * np.pi * s.targets.idx / 400
    for t in (0.0, 0.1, 0.25, 0.6, 1.0):
        hooks.update_targets(t, 1e-3, s)
        g = 1 + 0.4 * np.sin(2 * np.pi * t)
        a, b = 0.2 * g, 0.2 / g
        assert np.allclose(s.targets.X[:, 0], 0.5 + a * np.cos(th), atol=1e-14)
        assert np.allclose(s.targets.X[:, 1], 0.5 + b * np.sin(th), atol=1e-14)
        assert a * b == pytest.approx(0.04)                       # the half-ellipse keeps its area
        # the target velocity is the time derivative of the target position
        eps = 1e-6
        hooks.update_targets(t + eps, 1e-3, s)
        Xp = s.targets.X.copy()
        hooks.update_targets(t - eps, 1e-3, s)
        Xm = s.targets.X.copy()
        hooks.update_targets(t, 1e-3, s)
        assert np.allclose(s.targets.V, (Xp - Xm) / (2 * eps), atol=1e-6)
    hooks.update_targets(0.0, 1e-3, s)
    assert np.allclose(s.targets.X, s.X0[s.targets.idx], atol=1e-14)   # a circle at t = 0
    hooks.update_targets(0.25, 1e-3, s)
    assert np.max(s.targets.X[:, 0]) == pytest.approx(0.5 + 0.28)      # 40 % wider at t = T/4


def test_study_notes_the_drivers_warnings():
    """run_level keeps a short note of each driver warning for the study table. At dt = dx the
    target tether number is 3200 (1/128)^2 / (1/128) = 25, above the rule of thumb (10), and
    the note is "tether number 25" (two steps of the delta run, which also go past CFL 1 and
    get a "CFL x.xx" note)."""
    an = load("analyze")
    lev = an.run_level("delta", 1, tfinal=2.0 / 128)
    notes = lev["notes"].split(", ")
    assert lev["steps"] == 2 and notes[0] == "tether number 25"
    assert all(re.fullmatch(r"CFL \d+(\.\d+)?", n) for n in notes[1:])
    assert an.warning_note("CFL number 1.23 > 1 at step 7 (t = 0.1); ...") == "CFL 1.23"
    other = "something else entirely, a message longer than forty characters"
    assert an.warning_note(other) == other[:40]


def test_study_samples_the_lag_as_the_default_runs_do():
    """Every level of the study records 65 history rows, enough to catch the peaks of the
    target lag, so its max lag column at dx/16 is the default run's."""
    an = load("analyze")
    gen = load("generate")
    lev = an.run_level("delta", 64, tfinal=64 * gen.DX / 64)
    assert lev["steps"] == 64 and len(lev["history"]) == 65
    assert gen.parameters("delta", "viz/delta").output.print_dump == 2048 // 64


def test_analyze_on_short_runs(tmp_path, monkeypatch, capsys):
    """analyze.py without --study reads both history files and the lag files (with
    pyibm.io.vtk.read_lagrangian_points) and writes results.png; here on 4-step runs."""
    for input_name, coupling in INPUTS:
        run(CASE, input_file=input_name, steps=4, quiet=True, output_dir=str(tmp_path / "viz" / coupling))
    an = load("analyze")
    monkeypatch.setattr(an, "HERE", str(tmp_path))
    monkeypatch.setattr(an, "read_vertex", lambda path: read_vertex(os.path.join(CASE, "circle.vertex")))
    monkeypatch.setattr(an, "read_target", lambda path, X: read_target(os.path.join(CASE, "circle.target"), X))
    png = an.main([])
    assert png == str(tmp_path / "results.png") and os.path.getsize(png) > 10000
    text = capsys.readouterr().out
    assert "input2d (delta" in text and "input2d_blob (blob" in text and "target stiffness 3200" in text


@pytest.mark.slow
@pytest.mark.parametrize("coupling, factors, reference", (("delta", (8, 16), 32), ("blob", (8, 16), 32)))
def test_error_falls_with_the_time_step(coupling, factors, reference):
    """The time-step study of analyze.py over one period, dt = dx/8 and dx/16 against dx/32:
    E_u and E_X fall at the second-order rate of the midpoint scheme (measured orders 2.6 and
    2.1 for the delta coupling, 2.6 and 2.0 for the blob; the full study in the README goes to
    dx/64). The boundary lags its soft targets by about a cell (measured 0.93 and 1.17 cells).
    About 3 min for both."""
    an = load("analyze")
    table = an.study(coupling, factors, reference, tfinal=1.0, verbose=False)
    an.print_study(table, coupling, reference)
    assert np.all(np.isfinite(table.E_u)) and np.all(np.isfinite(table.E_X))
    assert np.all(np.diff(table.E_u) < 0) and np.all(np.diff(table.E_X) < 0)
    assert np.all(table.order_u[1:] > 1.5) and np.all(table.order_X[1:] > 1.5)
    assert np.all(table.max_lag_over_dx < 1.5)
    assert table.E_u[-1] < 0.1 and table.E_X_over_dx[-1] < 0.1
