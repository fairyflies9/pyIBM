"""examples/heaving_plate_2008: smoke runs of the three inputs, the committed files against
generate.py, the full runs against the numbers in the README (100 steps on 16 x 16 take
about 0.2 s each, so they are always on), the two stiff runs compared at the time level of
their force, and (slow) the second-order run (input2d_midpoint) to the end. The bit-level
comparison with the MATLAB checkpoints is tests/test_matlab_regression.py."""

from __future__ import annotations

import dataclasses
import importlib.util
import os

import numpy as np
import pytest

from pyibm.driver import Simulation, run
from pyibm.inputfile import read_input_file
from pyibm.io.history import read_history
from pyibm.structures import read_target, read_vertex

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CASE = os.path.join(ROOT, "examples", "heaving_plate_2008")
INPUTS = ("input2d", "input2d_stiff1e4", "input2d_midpoint")


def load_generate():
    spec = importlib.util.spec_from_file_location("heaving_plate_generate", os.path.join(CASE, "generate.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("input_name", INPUTS)
def test_smoke_three_steps(tmp_path, input_name):
    res = run(CASE, input_file=input_name, steps=3, quiet=True, output_dir=str(tmp_path))
    assert res.steps == 3 and res.history.step.tolist() == [0.0, 3.0]   # print_dump = 5
    assert np.all(np.isfinite(res.history.view(float)))
    assert res.history.max_target_error_over_dx[-1] < 0.1
    assert res.structure.n == 16 and res.structure.targets is not None
    assert os.path.isfile(tmp_path / "history.csv")
    assert os.path.isfile(tmp_path / "lag.00003.vtk")
    assert len(read_history(tmp_path / "history.csv")) == 2
    # the plate is pulled in +x by its targets, so the fluid pushes back in -x
    assert res.history.fx_body[-1] < 0.0
    assert res.solver.kinetic_energy() > 0.0


def test_committed_files_match_generate(tmp_path):
    gen = load_generate()
    written = gen.main(str(tmp_path))
    names = sorted(os.path.basename(p) for p in written)
    assert names == sorted(["plate.vertex", "plate.target", "plate_stiff1e4.vertex",
                            "plate_stiff1e4.target"] + list(INPUTS))
    for path in written:   # byte for byte, headers included
        with open(path, "rb") as a, open(os.path.join(CASE, os.path.basename(path)), "rb") as b:
            assert a.read() == b.read(), os.path.basename(path)
    X = gen.plate_points()
    assert X.shape == (16, 2)
    assert np.allclose(X[:, 0], 0.2) and np.allclose(X[:, 1], 0.1 + 0.0125 * np.arange(16))
    for name, k in (("plate", 1.0), ("plate_stiff1e4", 1e4)):
        V = read_vertex(os.path.join(CASE, name + ".vertex"))
        assert np.array_equal(V, X)
        T = read_target(os.path.join(CASE, name + ".target"), V)
        assert np.all(T.k == k) and T.idx.tolist() == list(range(16))
    assert gen.DT == 0.025 * 0.025 / 1.5
    for input_name in INPUTS:
        p = read_input_file(os.path.join(CASE, input_name))
        assert p.time.dt == gen.DT and p.time.n_steps == 100
        assert (p.grid.Nx, p.grid.Ny, p.grid.Lx, p.grid.Ly) == (16, 16, 0.4, 0.4)
        assert (p.fluid.mu, p.fluid.rho, p.structure.ds) == (3.0, 2.0, 0.0125)
        assert p.solver.backend == "periodic_fft" and p.solver.coupling == "delta"
    p = read_input_file(os.path.join(CASE, "input2d"))
    assert (p.solver.time_scheme, p.solver.kernel, p.structure.name) == ("pm1996", "cosine4", "plate")
    p = read_input_file(os.path.join(CASE, "input2d_stiff1e4"))
    assert (p.solver.time_scheme, p.solver.kernel, p.structure.name) == ("pm1996", "cosine4", "plate_stiff1e4")
    p = read_input_file(os.path.join(CASE, "input2d_midpoint"))
    assert (p.solver.time_scheme, p.solver.kernel, p.structure.name) == ("midpoint", "peskin4", "plate_stiff1e4")


def test_hook_is_the_2008_law():
    gen = load_generate()
    from pyibm.structures import load_hooks, read_structure
    hooks = load_hooks(CASE)
    s = read_structure(CASE, "plate_stiff1e4", ds=gen.DS)
    for t in (0.0, 1.0, 2.5, 7.0):
        hooks.update_targets(t, gen.DT, s)
        assert np.allclose(s.targets.X[:, 0], 0.2 + 0.05 * np.sin(2 * np.pi * t / 10))
        assert np.array_equal(s.targets.X[:, 1], s.X0[:, 1])


# (largest target error [cells], fx_body at the row t = 0.01875 s, at the last row) as the
# README table gives them, with the tolerance that the rounding there allows
README_TABLE = {
    "input2d": (0.052, -1.15e-4, -2.6e-4),
    "input2d_stiff1e4": (0.016, -0.68, +0.079),
    "input2d_midpoint": (0.016, -0.68, +0.085),
}


@pytest.mark.parametrize("input_name", INPUTS)
def test_full_runs_match_the_readme_table(tmp_path, input_name):
    """The README table: the force at the row t = 0.01875 s (step 45) and at the end, the
    largest target error, and for k = 1 the fluid speed at the end, 2.07e-5 m/s (so the
    README says 2.1e-5, not below 2e-5), and the linear growth of the force from zero
    (-1.15e-4 N at step 45)."""
    res = run(CASE, input_file=input_name, quiet=True, output_dir=str(tmp_path))
    h = res.history
    assert res.steps == 100 and len(h) == 21
    i45 = int(np.flatnonzero(h.step == 45)[0])
    assert h.t[i45] == pytest.approx(0.01875)
    err, fx45, fx_end = README_TABLE[input_name]
    assert h.max_target_error_over_dx.max() == pytest.approx(err, abs=5e-4)
    assert h.fx_body[i45] == pytest.approx(fx45, rel=0.01)
    assert h.fx_body[-1] == pytest.approx(fx_end, rel=0.03)
    # the force is labelled with its own time level: t - dt for pm1996, t - dt/2 for midpoint
    dt = res.parameters.time.dt
    lag = dt if res.parameters.solver.time_scheme == "pm1996" else 0.5 * dt
    assert np.allclose(h.t_force[1:], h.t[1:] - lag)
    assert np.all(h.cfl < 1e-3)
    if input_name == "input2d":
        assert h.max_speed[-1] == pytest.approx(2.1e-5, rel=0.02) and h.max_speed[-1] > 2e-5
        # the force grows linearly from zero: fx / t_force is the same at every row
        slope = h.fx_body[1:] / h.t_force[1:]
        assert np.max(np.abs(slope / slope[-1] - 1.0)) < 0.05


def _history_every_step(tmp_path, input_name, dt_factor=1, **solver):
    """History of the full run of input_name with a row every step (print_dump = 1, no VTK
    files), the time step divided by dt_factor and the Solver keys in solver replaced."""
    p = read_input_file(os.path.join(CASE, input_name))
    p = dataclasses.replace(
        p, solver=dataclasses.replace(p.solver, **solver),
        time=dataclasses.replace(p.time, dt=p.time.dt / dt_factor),
        output=dataclasses.replace(p.output, print_dump=1, save_vtk=False))
    out = tmp_path / "_".join([input_name, str(dt_factor)] + list(solver.values()))
    h = Simulation(p, CASE, output_dir=str(out), quiet=True).run().history
    assert len(h) == 100 * dt_factor + 1
    return h


def _force_at(h, t_force):
    """fx_body of history h at the force time levels t_force (linear in between)."""
    return np.interp(t_force, h.t_force[1:], h.fx_body[1:])


def test_stiff_runs_agree_at_the_force_time_level(tmp_path):
    """The README paragraph on the two stiff runs. Row for row they differ by up to 0.013 N,
    1.9 % of the 0.678 N peak, at step 1: a pm1996 row holds the force at t - dt and a
    midpoint row the force at t - dt/2, and the force grows by 0.026 N per step. At the same
    t_force they differ by 0.0033 N, 0.49 % of the peak. A quarter of the time step changes
    the force at its own time level by 0.38 % of the peak with pm1996 and 0.014 % with
    midpoint, and swapping only the kernel changes it by 1.6e-6 of the peak."""
    pm = _history_every_step(tmp_path, "input2d_stiff1e4")
    mid = _history_every_step(tmp_path, "input2d_midpoint")
    dt = read_input_file(os.path.join(CASE, "input2d_stiff1e4")).time.dt
    assert np.allclose(pm.t_force[1:], pm.t[1:] - dt)
    assert np.allclose(mid.t_force[1:], mid.t[1:] - 0.5 * dt)
    peak = np.max(np.abs(pm.fx_body))
    assert peak == pytest.approx(0.678, abs=5e-4)

    # row for row: mostly the half step between the two time levels
    rows = np.abs(pm.fx_body - mid.fx_body)
    assert int(pm.step[np.argmax(rows)]) == 1
    assert rows.max() == pytest.approx(0.0131, abs=1e-4)
    assert rows.max() / peak == pytest.approx(0.019, abs=5e-4)
    assert pm.fx_body[1] == 0.0 and mid.fx_body[1] == pytest.approx(-0.0131, abs=1e-4)
    assert pm.fx_body[2] - pm.fx_body[1] == pytest.approx(-0.026, abs=5e-4)

    # at the same t_force (the midpoint history put on the pm1996 force times)
    tf = pm.t_force[1:]
    tf = tf[tf >= mid.t_force[1]]
    gap = np.abs(_force_at(pm, tf) - _force_at(mid, tf))
    assert gap.max() / peak < 0.01
    assert gap.max() / peak == pytest.approx(0.0049, abs=3e-4)
    assert tf[np.argmax(gap)] == pytest.approx(pm.t_force[-1])        # largest at the end

    # a quarter of the time step, compared at the force times of the dt run
    for base, name, bound, measured in ((pm, "input2d_stiff1e4", 0.01, 0.0038),
                                        (mid, "input2d_midpoint", 0.001, 0.00014)):
        fine = _history_every_step(tmp_path, name, dt_factor=4)
        change = np.max(np.abs(_force_at(fine, base.t_force[1:]) - base.fx_body[1:])) / peak
        assert change < bound
        assert change == pytest.approx(measured, abs=0.3 * measured)

    # the kernel alone
    k = _history_every_step(tmp_path, "input2d_stiff1e4", kernel="peskin4")
    assert np.max(np.abs(k.fx_body - pm.fx_body)) / peak < 1e-5


@pytest.mark.slow
def test_second_order_run_follows_the_targets(tmp_path):
    """100 steps of the midpoint/peskin4 run at stiffness 1e4: the plate lags its targets by
    less than 0.05 cells (measured 0.016) and the history has the 21 expected rows."""
    res = run(CASE, input_file="input2d_midpoint", quiet=True, output_dir=str(tmp_path))
    assert res.steps == 100
    assert len(res.history) == 21
    assert res.history.max_target_error_over_dx.max() < 0.05
    assert np.all(res.history.cfl < 0.1)
    print(f"\nsecond-order run:{res.wall_time:.2f} s, max target error "
          f"{res.history.max_target_error_over_dx.max():.4f} cells")
