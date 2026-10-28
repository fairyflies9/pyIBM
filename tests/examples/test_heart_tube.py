"""examples/heart_tube (Jung & Peskin 2001, valveless pumping): smoke runs on the coarse and
the paper grid, the box mean held at zero under the paper's scheme, the committed files
against generate.py, the target law and the flowmeter of hooks.py, the area measures of
analyze.py, and (slow) ten-cycle runs whose net flow goes clockwise at T = 0.325 s and
counterclockwise at T = 0.21 s, plus the paper's own scheme against the paper's number."""

from __future__ import annotations

import importlib.util
import os
import warnings

import numpy as np
import pytest

from pyibm.driver import run
from pyibm.inputfile import read_input_file
from pyibm.io.history import read_history
from pyibm.structures import load_hooks, read_structure, read_target, read_vertex

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CASE = os.path.join(ROOT, "examples", "heart_tube")
INPUTS = ("input2d", "input2d_ccw", "input2d_zero", "input2d_symmetric", "input2d_pm1996",
          "input2d_coarse")
AREA_PAPER = 34.2796    # cm^2, the initial area between the walls quoted under Fig. 4


def load_generate():
    spec = importlib.util.spec_from_file_location("heart_tube_generate", os.path.join(CASE, "generate.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_analyze():
    spec = importlib.util.spec_from_file_location("heart_tube_analyze", os.path.join(CASE, "analyze.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def recorded_run(*args, **kwargs):
    """run() that also returns the warnings it issued. The neighbour springs, kc/ds^2 =
    4.9e5 g/(s^2 cm), are at k dt^2/(rho dx) = 7.5 on the paper grid, but the driver checks
    only targets and mass tethers against its rule of thumb, and the targets are at 0.4
    (1.6 with the paper's dt), so no tether warning is expected."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = run(*args, **kwargs)
    return res, [str(w.message) for w in caught]


def read_flux(path):
    return np.genfromtxt(path, delimiter=",", names=True).view(np.recarray)


# ------------------------------------------------------------------ smoke

def test_smoke_coarse_three_steps(tmp_path):
    res, caught = recorded_run(CASE, input_file="input2d_coarse", steps=3, quiet=True,
                               output_dir=str(tmp_path))
    assert not any("tether number" in m for m in caught)
    assert res.steps == 3 and res.history.step.tolist() == [0.0, 2.0, 3.0]   # print_dump = 2
    assert np.all(np.isfinite(res.history.view(float)))
    assert res.structure.n == 914 and res.structure.targets is not None
    assert res.grid.Nx == 64 and res.grid.Ny == 32
    assert os.path.isfile(tmp_path / "history.csv")
    assert len(read_history(tmp_path / "history.csv")) == 3
    flux = read_flux(tmp_path / "flux.csv")
    assert len(flux) == 3 and flux.step.tolist() == [0.0, 2.0, 3.0]
    assert abs(flux.area[0] - AREA_PAPER) < 0.01          # the paper's initial area
    assert abs(flux.area[-1] - flux.area[0]) < 0.01
    # the targets start moving up (expansion) at t = 0+, so the walls are being pulled apart
    assert res.history.max_target_error_over_dx[-1] > 0.0
    assert res.solver.kinetic_energy() > 0.0


def test_smoke_paper_grid_three_steps(tmp_path):
    res, caught = recorded_run(CASE, steps=3, quiet=True, output_dir=str(tmp_path))
    assert not any("tether number" in m for m in caught)
    assert res.steps == 3 and res.history.step.tolist() == [0.0, 3.0]        # print_dump = 8
    assert np.all(np.isfinite(res.history.view(float)))
    assert res.structure.n == 3657 and res.grid.Nx == 256 and res.grid.Ny == 128
    assert res.parameters.solver.remove_mean_force is True
    assert abs(res.history.mean_u[-1]) < 1e-12 and abs(res.history.mean_v[-1]) < 1e-12
    assert len(read_flux(tmp_path / "flux.csv")) == 2


def test_paper_scheme_holds_the_box_mean(tmp_path):
    """remove_mean_force under pm1996 (input2d_pm1996): the first-order upwind advection term
    is not conservative, so subtracting the mean force alone let the box mean drift (by
    5.8e-4 cm/s in these 40 steps, and up to 1.8 cm/s over the 20 cycles at T = 0.21 s). The
    solver now holds it at zero in both schemes, which the README relies on."""
    res, _ = recorded_run(CASE, input_file="input2d_pm1996", steps=40, quiet=True,
                          output_dir=str(tmp_path))
    assert res.parameters.solver.time_scheme == "pm1996"
    assert res.parameters.solver.remove_mean_force is True
    assert res.history.max_speed[-1] > 0.1                 # the drive is moving fluid
    assert np.max(np.abs(res.history.mean_u)) < 1e-12
    assert np.max(np.abs(res.history.mean_v)) < 1e-12


# ------------------------------------------------------------------ files

def test_committed_files_match_generate():
    gen = load_generate()
    assert abs(gen.CIRCUMFERENCE - 28.57) < 0.01                       # Table 1
    for fname, period, x_drive, coarse, scheme, kernel, out, _ in gen.CASES:
        name = gen.structure_name(period, gen.AMPLITUDE, x_drive, coarse)
        nx, ny = (gen.NX_COARSE, gen.NY_COARSE) if coarse else (gen.NX_PAPER, gen.NY_PAPER)
        s = gen.tube_structure(name, gen.XSCALE / nx)
        V = read_vertex(os.path.join(CASE, name + ".vertex"))
        assert np.allclose(V, s.X, atol=1e-12)
        T = read_target(os.path.join(CASE, name + ".target"), V)
        assert T.idx.tolist() == list(range(len(V)))
        assert set(np.unique(T.k).tolist()) == {gen.KT_FLEX, gen.KT_RIGID}
        p = read_input_file(os.path.join(CASE, fname))
        assert p.structure.name == name
        assert (p.grid.Nx, p.grid.Ny, p.grid.Lx, p.grid.Ly) == (nx, ny, 16.0, 8.0)
        assert (p.fluid.rho, p.fluid.mu) == (1.0, 0.01)
        assert p.solver.time_scheme == scheme and p.solver.kernel == kernel
        assert p.solver.backend == "periodic_fft" and p.solver.remove_mean_force is True
        assert p.output.output_dir == out
        steps_per_cycle = round(period / p.time.dt)
        assert abs(steps_per_cycle * p.time.dt - period) < 1e-12       # whole cycles
        assert p.time.n_steps == gen.n_cycles(period) * steps_per_cycle
        h = gen.XSCALE / nx
        dt_over_h = gen.DT_OVER_H_PAPER if scheme == "pm1996" else gen.DT_OVER_H
        assert abs(p.time.dt - dt_over_h * h) / p.time.dt < 0.02
    # the paper grid: 3654 points in the paper (Table 2), 3657 here from rounding each wall
    V = read_vertex(os.path.join(CASE, "tube_T0.325_A0.6_x4.vertex"))
    assert len(V) == 3657
    # walls 0.6 cm either side of the centreline: bottom straight at y = 2.6 (inner) and 1.4 (outer)
    assert np.isclose(V[0, 1], 2.6) and np.isclose(V[0, 0], 4.0)
    assert np.isclose(V[:, 1].min(), 1.4) and np.isclose(V[:, 1].max(), 6.6)


# ------------------------------------------------------------------ hooks

def test_target_law_of_section_3_2():
    gen = load_generate()
    hooks = load_hooks(CASE)
    s = read_structure(CASE, "tube_T0.325_A0.6_x4")
    X0 = s.X0
    T, A0, xd = hooks.module.drive_parameters(s.name)
    assert (T, A0, xd) == (0.325, 0.6, 4.0)
    Ld = gen.DRIVE_LENGTH
    driven = (s.targets.k == gen.KT_FLEX) & (X0[:, 0] >= xd - 1e-9) & (X0[:, 0] <= xd + Ld + 1e-9)
    assert driven.sum() > 0
    inner = driven & (X0[:, 1] > 2.0)
    outer = driven & (X0[:, 1] < 2.0)
    for t in (0.0, 0.25 * T, 0.75 * T, 1.3 * T):
        hooks.update_targets(t, 1e-3, s)
        shape = np.sin(np.pi * (X0[:, 0] - xd) / Ld)
        A = A0 * np.sin(2 * np.pi * t / T) * shape
        assert np.allclose(s.targets.X[~driven], X0[~driven])
        assert np.allclose(s.targets.X[driven, 0], X0[driven, 0])
        # L_d in the hook is a third of the extent of the flexible targets, 8 cm up to one
        # point spacing (1.3e-5 relative, 3e-5 cm at the far end of the driven third)
        assert np.allclose(s.targets.X[inner, 1], X0[inner, 1] + A[inner], rtol=0, atol=1e-4)
        assert np.allclose(s.targets.X[outer, 1], X0[outer, 1] - A[outer], rtol=0, atol=1e-4)
    # at t = T/4 the target tube is fully open at the middle of the driven third (y = 3.2 / 0.8),
    # at 3T/4 it is pinched shut there (both walls at y = 2.0)
    mid = driven & (np.abs(X0[:, 0] - (xd + 0.5 * Ld)) < 0.02)
    hooks.update_targets(0.75 * T, 1e-3, s)
    assert np.allclose(s.targets.X[mid, 1], 2.0, atol=1e-3)
    hooks.update_targets(0.25 * T, 1e-3, s)
    assert np.allclose(np.abs(s.targets.X[mid, 1] - 2.0), 1.2, atol=1e-3)
    # the same law in the symmetric case starts at the centre third
    s2 = read_structure(CASE, "tube_T0.21_A0.6_x6.666666667")
    assert hooks.module.drive_parameters(s2.name)[2] == pytest.approx(gen.X_DRIVE_CENTRE)


def test_flowmeter_sections_and_area():
    hooks = load_hooks(CASE).module
    s = read_structure(CASE, "tube_T0.325_A0.6_x4")
    h = 1.0 / 16
    (xt, yt), (xr, yr), n_in = hooks.sections(s, h)
    assert n_in == 1587                                        # inner wall first
    assert np.allclose(xt, 8.0) and np.isclose(yt[0], 5.4) and np.isclose(yt[-1], 6.6)
    assert np.allclose(yr, 4.0) and np.isclose(xr[0], 13.4) and np.isclose(xr[-1], 14.6)
    # a uniform u = 1 through the top section is a flux of 1.2 cm^2/s (the wall gap)
    field = np.ones((256, 128))
    assert hooks.line_flux(field, h, xt, yt) == pytest.approx(1.2)
    # a linear field is sampled exactly by the bilinear interpolation
    x = np.arange(256) * h
    field = np.repeat(x[:, None], 128, axis=1)
    assert np.allclose(hooks.sample_nodes(field, xt, yt, h), 8.0)
    # the area between the walls is the paper's 34.2796 cm^2
    area = hooks.polygon_area(s.X0[n_in:]) - hooks.polygon_area(s.X0[:n_in])
    assert abs(area - AREA_PAPER) < 1e-3


def test_summarize_reports_the_area_two_ways():
    """analyze.summarize gives the largest instantaneous change of the area and the change of
    the time-averaged area, the paper's Fig. 4 measure; the two differ (the README quotes
    both). A made-up record, 40 rows per cycle over 4 cycles, with the area swinging by 2 %
    about a level 0.5 % above the start: 2.5 % and +0.5 %."""
    ana = load_analyze()
    T, n = 0.21, 4 * 40
    t = np.arange(n + 1) * (T / 40)
    area = AREA_PAPER * (1.0 + (0.005 + 0.02 * np.sin(2 * np.pi * t / T)) * (t > 0))
    q = np.where(t > 0, -1.0, 0.0)
    flux = np.rec.fromarrays([np.arange(n + 1.0), t, q, 1.1 * q, area],
                             names=["step", "t", "flux_top", "flux_right", "area"])
    s = ana.summarize(flux, T)
    assert s["cycles"] == 4 and s["n_last"] == 4
    assert s["area_change"] == pytest.approx(0.025, rel=1e-3)
    assert s["mean_area_change"] == pytest.approx(0.005, rel=1e-9)
    assert s["mean_top"] == pytest.approx(-1.0) and s["mean_right"] == pytest.approx(-1.1)


# ------------------------------------------------------------------ slow

def _mean_flux_last_cycles(out_dir, period, n_last=5):
    flux = read_flux(os.path.join(out_dir, "flux.csv"))
    n = int(np.floor(flux.t[-1] / period + 1e-9))
    m = flux.t > (n - n_last) * period + 1e-9
    return float(flux.flux_top[m].mean()), float(flux.flux_right[m].mean()), flux


@pytest.mark.slow
def test_clockwise_net_flow_at_T0325(tmp_path):
    """Ten cycles at T = 0.325 s (the paper's maximum clockwise case, Fig. 3): the mean
    flux over cycles 6-10 is positive at both sections, the sections agree, and the largest
    instantaneous change of the area is below 1.5 % (the paper's Fig. 4 measure, the
    time-averaged area, has a worst case of 1.22 %). About 1.5 minutes."""
    p = read_input_file(os.path.join(CASE, "input2d"))
    spc = round(0.325 / p.time.dt)
    res, _ = recorded_run(CASE, steps=10 * spc, quiet=True, output_dir=str(tmp_path))
    top, right, flux = _mean_flux_last_cycles(str(tmp_path), 0.325)
    print(f"\nT = 0.325 s, midpoint: mean flux cycles 6-10 top {top:+.3f} right {right:+.3f} cm^2/s "
          f"(paper +0.6 over 150 s), {res.wall_time:.0f} s")
    assert top > 0.2 and right > 0.2
    assert abs(top - right) < 0.25 * max(abs(top), abs(right)) + 0.1
    assert np.max(np.abs(flux.area - flux.area[0])) / flux.area[0] < 0.015
    assert np.all(res.history.cfl < 1.5)


@pytest.mark.slow
def test_counterclockwise_net_flow_at_T021(tmp_path):
    """Ten cycles at T = 0.21 s (the paper's maximum counterclockwise case): the mean flux
    over cycles 6-10 is negative at both sections. About a minute."""
    p = read_input_file(os.path.join(CASE, "input2d_ccw"))
    spc = round(0.21 / p.time.dt)
    res, _ = recorded_run(CASE, input_file="input2d_ccw", steps=10 * spc, quiet=True,
                          output_dir=str(tmp_path))
    top, right, flux = _mean_flux_last_cycles(str(tmp_path), 0.21)
    print(f"\nT = 0.21 s, midpoint: mean flux cycles 6-10 top {top:+.3f} right {right:+.3f} cm^2/s "
          f"(paper -1.6 over 150 s), {res.wall_time:.0f} s")
    assert top < -0.2 and right < -0.2
    assert np.max(np.abs(flux.area - flux.area[0])) / flux.area[0] < 0.015


@pytest.mark.slow
def test_paper_scheme_reproduces_the_paper_flux(tmp_path):
    """Ten cycles with the paper's first-order scheme and time step at T = 0.325 s: the mean
    flux over cycles 6-10 lies between 0.3 and 1.2 cm^2/s (measured 0.80; the paper's
    150 s average is 0.6) and the flowmeter swings between about -1 and +3 cm^2/s as in
    Fig. 8. About 30 s."""
    p = read_input_file(os.path.join(CASE, "input2d_pm1996"))
    spc = round(0.325 / p.time.dt)
    res, _ = recorded_run(CASE, input_file="input2d_pm1996", steps=10 * spc, quiet=True,
                          output_dir=str(tmp_path))
    top, right, flux = _mean_flux_last_cycles(str(tmp_path), 0.325)
    m = flux.t > 5 * 0.325
    print(f"\nT = 0.325 s, pm1996: mean flux cycles 6-10 top {top:+.3f} right {right:+.3f} cm^2/s "
          f"(paper +0.6), range [{flux.flux_top[m].min():.2f}, {flux.flux_top[m].max():.2f}], "
          f"{res.wall_time:.0f} s")
    assert 0.3 < top < 1.2 and 0.3 < right < 1.2
    assert -2.0 < flux.flux_top[m].min() < 0.0 and 1.5 < flux.flux_top[m].max() < 4.5
    assert np.all(res.history.cfl < 1.0)
