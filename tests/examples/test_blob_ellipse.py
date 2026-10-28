"""examples/blob_ellipse: smoke runs of the two inputs, the committed files against
generate.py, the curve and the constant-tension force against the paper's sigma kappa n,
analyze.py on short runs, and (slow) the two runs to t = 1 with the area error against
Cortez & Minion 2000 (J. Comput. Phys. 161, 428-453, Figs. 4, 7 and 8)."""

from __future__ import annotations

import importlib.util
import os
import warnings

import numpy as np
import pytest

from pyibm.driver import Simulation, run
from pyibm.grid import Grid
from pyibm.inputfile import read_input_file
from pyibm.io.history import read_history
from pyibm.structures import compute_forces, read_spring, read_structure, read_vertex

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CASE = os.path.join(ROOT, "examples", "blob_ellipse")
# (input file, coupling, blob interpolation, output folder)
INPUTS = (("input2d", "blob", "streamfunction", "viz/blob"),
          ("input2d_delta", "delta", "streamfunction", "viz/delta"))


def load(name: str):
    spec = importlib.util.spec_from_file_location(f"blob_ellipse_{name}", os.path.join(CASE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_area(path):
    return np.genfromtxt(path, delimiter=",", names=True).view(np.recarray)


def checked_run(**kw):
    """run() that fails on any warning from the driver: the case has springs only (the tether
    rule of thumb checks targets and mass tethers) and the CFL number stays near 0.06."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = run(CASE, **kw)
    assert not [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
    return res


@pytest.mark.parametrize("input_name, coupling, interpolation, out", INPUTS)
def test_smoke_three_steps(tmp_path, input_name, coupling, interpolation, out):
    res = checked_run(input_file=input_name, steps=3, quiet=True, output_dir=str(tmp_path))
    assert res.parameters.solver.coupling == coupling
    if coupling == "blob":
        assert res.simulation.coupling.interpolation == interpolation
    assert res.steps == 3 and res.history.step.tolist() == [0.0, 3.0]   # print_dump = 40
    assert np.all(np.isfinite(res.history.view(float)))
    assert res.structure.n == 400 and res.structure.springs is not None and res.structure.targets is None
    assert len(read_history(tmp_path / "history.csv")) == 2
    assert os.path.isfile(tmp_path / "lag.00003.vtk")
    # a closed membrane puts no net force on the fluid
    assert np.max(np.abs(res.history.fx_body)) < 1e-12 and np.max(np.abs(res.history.fy_body)) < 1e-12
    assert res.history.kinetic_energy[-1] > 0.0 and res.history.cfl[-1] < 0.1
    a = read_area(tmp_path / "area.csv")
    assert a.step.tolist() == [0.0, 3.0] and a.rel_area_error[0] == 0.0
    # measured 1.1e-8 (blob, stream function), 3.2e-6 (delta)
    assert abs(a.rel_area_error[1]) < (1e-6 if coupling == "blob" else 1e-4)
    # the constant-tension hook set every link to the tension sigma at the half-step positions
    # where the forces were taken; at the end-of-step positions the links are within 1e-3 of it
    sp = res.structure.springs
    d = res.structure.X[sp.j] - res.structure.X[sp.i]
    assert np.allclose(sp.k * res.structure.ds[sp.i] * np.hypot(d[:, 0], d[:, 1]), 0.2, rtol=1e-3)
    load("hooks").update_springs(res.t, res.parameters.time.dt, res.structure)
    d = res.structure.X[sp.j] - res.structure.X[sp.i]
    assert np.allclose(sp.k * res.structure.ds[sp.i] * np.hypot(d[:, 0], d[:, 1]), 0.2, rtol=1e-12)


def test_committed_files_match_generate(tmp_path):
    gen = load("generate")
    written = gen.main(str(tmp_path))
    assert sorted(os.path.basename(p) for p in written) == sorted(
        ["ellipse.vertex", "ellipse.spring"] + [i[0] for i in INPUTS])
    for path in written:   # byte for byte, headers included
        with open(path, "rb") as a, open(os.path.join(CASE, os.path.basename(path)), "rb") as b:
            assert a.read() == b.read(), os.path.basename(path)
    X = gen.ellipse_points()
    assert X.shape == (400, 2)
    V = read_vertex(os.path.join(CASE, "ellipse.vertex"))
    assert np.allclose(V, X, rtol=0, atol=1e-15)
    # the points lie on the paper's curve, at (nearly) equal arc length
    th = np.mod(np.arctan2(X[:, 1] - 0.5, X[:, 0] - 0.5) + gen.THETA0, 2 * np.pi)
    assert np.allclose(np.hypot(X[:, 0] - 0.5, X[:, 1] - 0.5), gen.radius(th), rtol=1e-10)
    seg = np.hypot(*(np.roll(X, -1, axis=0) - X).T)
    assert seg.max() / seg.min() < 1.001
    assert seg.mean() < 0.5 / 128                         # below h/2, the paper's leak-free rule
    sp = read_spring(os.path.join(CASE, "ellipse.spring"), 400)
    assert sp.i.tolist() == list(range(400)) and sp.j.tolist() == [(i + 1) % 400 for i in range(400)]
    assert np.all(sp.rest == 0.0) and np.allclose(sp.k * seg.mean() * seg, gen.SIGMA, rtol=1e-12)
    assert [(i[0], i[1], i[2]) for i in gen.INPUTS] == [(i[0], i[1], i[2]) for i in INPUTS]
    for input_name, coupling, interpolation, out in INPUTS:
        p = read_input_file(os.path.join(CASE, input_name))
        assert p == gen.parameters(coupling, out, interpolation) and gen.OUTPUT_DIRS[input_name] == out
        assert p.output.output_dir == out and p.hooks.file == "hooks.py"
        assert p.solver.coupling == coupling and p.solver.backend == "periodic_fft"
        assert p.solver.blob_interpolation == interpolation
        assert (p.grid.Nx, p.time.dt, p.fluid.mu, p.fluid.rho) == (128, 0.1 / 128, 0.005, 1.0)
        assert p.time.n_steps == 1280 and p.structure.ds == pytest.approx(seg.mean())
    p = read_input_file(os.path.join(CASE, "input2d"))
    assert (p.solver.blob, p.solver.blob_delta) == ("gaussian4", 4.0)


def test_curve_is_the_papers():
    """r^2 = a^2 cos^2 + b^2 sin^2 + eps(...) with the paper's constants (sec. 4.1)."""
    gen = load("generate")
    assert (gen.A_AXIS, gen.B_AXIS, gen.EPS, gen.SIGMA) == (0.2, 0.25, 0.012, 0.2)
    assert (gen.THETA0, gen.THETA1, gen.THETA2) == (np.pi / 3, 4 * np.pi / 5, 2 * np.pi / 3)
    th = np.array([0.0, 0.7, np.pi, 4 * np.pi / 5, 2 * np.pi / 3, 5.0])
    r2 = (0.04 * np.cos(th) ** 2 + 0.0625 * np.sin(th) ** 2
          + 0.012 * (-(4 / 3) * np.exp(-3 * (th - np.pi) ** 2) - np.exp(-5 * (th - 4 * np.pi / 5) ** 2)
                     + np.exp(-8 * (th - 2 * np.pi / 3) ** 2)))
    assert np.allclose(gen.radius(th) ** 2, r2, rtol=1e-14)
    Z = gen.curve(np.array([0.0]))
    assert np.allclose(Z[0], [0.5 + 0.2 * np.cos(-np.pi / 3), 0.5 + 0.2 * np.sin(-np.pi / 3)])


def test_constant_tension_chain_is_sigma_kappa_n():
    """With every link at the tension sigma, the force per unit length on a point is
    sigma (tau+ - tau-)/ds, which is the paper's sigma kappa n (pointing inward) to O(ds^2):
    compare with the exact curvature of the polar curve at every point."""
    gen = load("generate")
    hooks = load("hooks")
    grid = Grid.periodic_box(1.0, 1.0, 128, 128)
    s = read_structure(CASE, "ellipse", ds=float(read_input_file(os.path.join(CASE, "input2d")).structure.ds))
    hooks.update_springs(0.0, 1e-3, s)
    f = compute_forces(s, grid, 0.0, 1e-3)
    X = s.X
    th = np.mod(np.arctan2(X[:, 1] - 0.5, X[:, 0] - 0.5) + gen.THETA0, 2 * np.pi)
    eps = 1e-5
    r = gen.radius(th)
    rp = (gen.radius(th + eps) - gen.radius(th - eps)) / (2 * eps)
    rpp = (gen.radius(th + eps) - 2 * r + gen.radius(th - eps)) / eps ** 2
    kappa = (r ** 2 + 2 * rp ** 2 - r * rpp) / (r ** 2 + rp ** 2) ** 1.5
    assert kappa.min() < 0 < kappa.max()        # the dent at theta = pi makes the curve locally concave
    # outward normal from the exact tangent of the counter-clockwise parametrization
    tx = rp * np.cos(th - gen.THETA0) - r * np.sin(th - gen.THETA0)
    ty = rp * np.sin(th - gen.THETA0) + r * np.cos(th - gen.THETA0)
    norm = np.hypot(tx, ty)
    tau = np.column_stack([tx, ty]) / norm[:, None]
    n_out = np.column_stack([tau[:, 1], -tau[:, 0]])
    f_n = np.sum(f * n_out, axis=1)
    f_t = np.sum(f * tau, axis=1)
    scale = gen.SIGMA * np.max(np.abs(kappa))
    # a membrane under tension pulls inward where it is convex: f . n_out = -sigma kappa
    assert np.allclose(f_n, -gen.SIGMA * kappa, rtol=0, atol=2e-2 * scale)
    assert np.max(np.abs(f_t)) < 2e-2 * scale
    # and exactly sigma (tau+ - tau-)/ds
    d = np.roll(X, -1, axis=0) - X
    tau = d / np.hypot(d[:, 0], d[:, 1])[:, None]
    assert np.allclose(f, gen.SIGMA * (tau - np.roll(tau, 1, axis=0)) / s.ds[:, None], rtol=1e-10, atol=1e-10)


def test_area_csv_without_any_driver_output(tmp_path):
    """With save_history, save_vtk and plot all off the driver writes nothing and makes no
    output folder, but on_output still writes area.csv (it makes the folder itself)."""
    p = read_input_file(os.path.join(CASE, "input2d_delta"))
    p.output.save_history = False
    p.output.save_vtk = False
    p.output.print_dump = 2
    out = tmp_path / "not" / "yet" / "there"
    Simulation(p, CASE, output_dir=str(out), quiet=True).run(steps=2)
    a = read_area(out / "area.csv")
    assert a.step.tolist() == [0.0, 2.0] and sorted(os.listdir(out)) == ["area.csv"]


@pytest.mark.parametrize("input_name", [i[0] for i in INPUTS])
def test_half_turn_is_a_symmetry_of_the_run(tmp_path, input_name):
    """analyze.py draws the membrane turned by half a turn about the box centre, the way round
    it sits in the paper's Fig. 3. That is only fair if the turn changes nothing: started from
    the turned curve, the run is the turned run (to round-off), with the same area error."""
    p = read_input_file(os.path.join(CASE, input_name))
    p.output.print_dump = 4
    p.output.save_vtk = False
    runs = {}
    for name, turn in (("plain", False), ("turned", True)):
        s = read_structure(CASE, "ellipse", ds=p.structure.ds)
        if turn:
            s.X = 1.0 - s.X
            s.X0 = s.X.copy()
        sim = Simulation(p, CASE, structure=s, output_dir=str(tmp_path / name), quiet=True)
        sim.run(steps=4)
        runs[name] = (sim.structure.X.copy(), read_area(tmp_path / name / "area.csv"))
    (X, a), (Xt, at) = runs["plain"], runs["turned"]
    assert np.max(np.abs(Xt - (1.0 - X))) < 1e-13
    assert a.rel_area_error[-1] != 0.0
    assert at.rel_area_error[-1] == pytest.approx(a.rel_area_error[-1], rel=1e-6, abs=1e-14)
    an = load("analyze")
    assert np.array_equal(an.half_turn(X), 1.0 - X)


def test_analyze_on_short_runs(tmp_path, monkeypatch, capsys):
    """analyze.py reads the two output folders (area.csv, history.csv and the lag files,
    with pyibm.io.vtk.read_lagrangian_points) and writes results.png; here on 4-step runs in
    a copy of the folder layout."""
    for input_name, coupling, interpolation, out in INPUTS:
        checked_run(input_file=input_name, steps=4, quiet=True, output_dir=str(tmp_path / out))
    an = load("analyze")
    monkeypatch.setattr(an, "HERE", str(tmp_path))
    png = an.main()
    assert png == str(tmp_path / "results.png") and os.path.getsize(png) > 10000
    text = capsys.readouterr().out
    for input_name, _, _, _ in INPUTS:
        assert f"{input_name} (" in text
    assert "delta over input2d:" in text


@pytest.mark.slow
@pytest.mark.parametrize("input_name, low, high", (
        ("input2d", 0.0, 1e-5), ("input2d_delta", 1e-3, 1.5e-2)))
def test_area_error_at_t_1(tmp_path, input_name, low, high):
    """1280 steps to t = 1 at the paper's parameters. Measured relative area error (polygon):
    blob with stream-function interpolation +2.5e-6 at t = 1, between -5.7e-6 and +2.9e-6 over
    the run, inside the paper's +-1e-5 (Figs. 7 and 8); the delta function -6.9e-3 (the paper's
    Fig. 4: -1.4e-2 with the decoupled centred projection this back end uses). The bounds pin
    that the blob run is inside the paper's band and the delta run loses area steadily."""
    res = checked_run(input_file=input_name, quiet=True, output_dir=str(tmp_path))
    assert res.steps == 1280 and len(res.history) == 33
    assert np.all(np.isfinite(res.history.view(float)))
    assert res.history.cfl.max() < 0.1
    a = read_area(tmp_path / "area.csv")
    err = a.rel_area_error
    assert np.max(np.abs(err)) < high
    if low > 0.0:
        assert -high < err[-1] < -low                 # a steady loss
        assert np.all(np.diff(err[::8]) < 0.0)        # falling over every quarter of the run
    print(f"\n{input_name}: {res.wall_time:.1f} s, relative area error at t = 1: {err[-1]:+.3e} "
          f"(max |{np.max(np.abs(err)):.3e}|)")
