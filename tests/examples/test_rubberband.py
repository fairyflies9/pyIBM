"""examples/rubberband: smoke runs of both couplings, the committed files against
generate.py, the area diagnostic on the initial ellipse, analyze.py on short runs, and
(slow) both runs to the end with the area held and the band rounded up."""

from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest

from pyibm.driver import Simulation, run
from pyibm.grid import Grid
from pyibm.inputfile import read_input_file
from pyibm.io.history import read_history
from pyibm.structures import read_beam, read_spring, read_vertex

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CASE = os.path.join(ROOT, "examples", "rubberband")
INPUTS = (("input2d", "delta"), ("input2d_blob", "blob"))


def load(name: str):
    spec = importlib.util.spec_from_file_location(f"rubberband_{name}", os.path.join(CASE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_area(path):
    return np.genfromtxt(path, delimiter=",", names=True).view(np.recarray)


@pytest.mark.parametrize("input_name, coupling", INPUTS)
def test_smoke_three_steps(tmp_path, input_name, coupling):
    res = run(CASE, input_file=input_name, steps=3, quiet=True, output_dir=str(tmp_path))
    assert res.parameters.solver.coupling == coupling
    assert res.steps == 3 and res.history.step.tolist() == [0.0, 3.0]   # print_dump = 20
    assert np.all(np.isfinite(res.history.view(float)))
    assert res.structure.n == 128 and res.structure.springs is not None and res.structure.beams is not None
    assert res.structure.targets is None
    assert len(read_history(tmp_path / "history.csv")) == 2
    assert os.path.isfile(tmp_path / "lag.00003.vtk")
    # the band is an isolated elastic structure: no net force on the fluid, to round-off
    assert np.max(np.abs(res.history.fx_body)) < 1e-12
    assert np.max(np.abs(res.history.fy_body)) < 1e-12
    # it starts moving (the springs are stretched) but is still an ellipse
    assert res.history.kinetic_energy[-1] > 0.0
    a = read_area(tmp_path / "area.csv")
    assert a.step.tolist() == [0.0, 3.0]
    assert abs(a.area[1] / a.area[0] - 1.0) < 1e-5            # measured -2.0e-6 delta, +2.0e-8 blob
    assert 1.99 < a.rmax_over_rmin[1] <= 2.0 + 1e-9
    if coupling == "blob":
        assert res.simulation.coupling.interpolation == "streamfunction"


def test_committed_files_match_generate(tmp_path):
    gen = load("generate")
    written = gen.main(str(tmp_path))
    assert sorted(os.path.basename(p) for p in written) == sorted(
        ["band.vertex", "band.spring", "band.beam", "input2d", "input2d_blob"])
    for path in written:   # byte for byte, headers included
        with open(path, "rb") as a, open(os.path.join(CASE, os.path.basename(path)), "rb") as b:
            assert a.read() == b.read(), os.path.basename(path)
    X = gen.band_points()
    assert X.shape == (128, 2)
    V = read_vertex(os.path.join(CASE, "band.vertex"))
    assert np.allclose(V, X, rtol=0, atol=1e-15)
    # on the ellipse, at (nearly) equal arc length
    assert np.allclose(((X[:, 0] - 0.5) / 0.2) ** 2 + ((X[:, 1] - 0.5) / 0.1) ** 2, 1.0, atol=1e-12)
    seg = np.hypot(*(np.roll(X, -1, axis=0) - X).T)
    assert seg.max() / seg.min() < 1.001
    sp = read_spring(os.path.join(CASE, "band.spring"), 128)
    assert sp.i.tolist() == list(range(128)) and sp.j.tolist() == [(i + 1) % 128 for i in range(128)]
    assert np.all(sp.k == gen.K_SPRING) and np.allclose(sp.rest, gen.REST_FACTOR * seg)
    bm = read_beam(os.path.join(CASE, "band.beam"), 128)
    assert bm.q.tolist() == list(range(128)) and np.all(bm.k == gen.K_BEAM) and np.all(bm.curv == 0.0)
    assert bm.p.tolist() == [(i - 1) % 128 for i in range(128)] and bm.r.tolist() == [(i + 1) % 128 for i in range(128)]
    for input_name, coupling in INPUTS:
        p = read_input_file(os.path.join(CASE, input_name))
        assert p == gen.parameters(coupling, "viz/" + coupling)
        assert p.solver.coupling == coupling and p.solver.backend == "periodic_fft"
        assert p.solver.blob_interpolation == "streamfunction" and p.hooks.file == "hooks.py"
        assert p.structure.ds == pytest.approx(seg.mean())


def test_area_diagnostic_on_the_initial_ellipse():
    hooks = load("hooks")
    gen = load("generate")
    grid = Grid.periodic_box(1.0, 1.0, 64, 64)
    X0 = gen.band_points()
    area, perimeter, ratio = hooks.polygon_geometry(X0, grid)
    # the polygon misses pi a b by the sum of its segments' caps, (ds^2/12) times the total
    # turning 2 pi: 3.0e-5 m^2, or 4.8e-4 of 0.0628
    ds = np.mean(np.hypot(*(np.roll(X0, -1, axis=0) - X0).T))
    assert 1.0 - area / (np.pi * 0.2 * 0.1) == pytest.approx(2 * np.pi * ds ** 2 / 12 / (np.pi * 0.2 * 0.1), rel=1e-3)
    assert perimeter == pytest.approx(0.9688, rel=1e-3)     # Ramanujan: 0.96884 for a = 0.2, b = 0.1
    assert ratio == pytest.approx(2.0, abs=1e-9)
    # the unwrapping: the same band shifted across the periodic edge gives the same numbers
    X = grid.wrap(gen.band_points() + np.array([0.45, 0.0]))
    area2, perimeter2, ratio2 = hooks.polygon_geometry(X, grid)
    assert area2 == pytest.approx(area, rel=1e-12) and perimeter2 == pytest.approx(perimeter, rel=1e-12)


def test_area_csv_without_any_driver_output(tmp_path):
    """With save_history, save_vtk and plot all off the driver writes nothing and makes no
    output folder, but on_output still writes area.csv (it makes the folder itself)."""
    p = read_input_file(os.path.join(CASE, "input2d"))
    p.output.save_history = False
    p.output.save_vtk = False
    p.output.print_dump = 2
    out = tmp_path / "not" / "yet" / "there"
    Simulation(p, CASE, output_dir=str(out), quiet=True).run(steps=2)
    a = read_area(out / "area.csv")
    assert a.step.tolist() == [0.0, 2.0] and sorted(os.listdir(out)) == ["area.csv"]


def test_analyze_on_short_runs(tmp_path, monkeypatch, capsys):
    """analyze.py reads both output folders (area.csv, history.csv and the lag files, with
    pyibm.io.vtk.read_lagrangian_points) and writes results.png; here on 5-step runs in a copy
    of the folder layout."""
    for input_name, coupling in INPUTS:
        run(CASE, input_file=input_name, steps=5, quiet=True, output_dir=str(tmp_path / "viz" / coupling))
    an = load("analyze")
    monkeypatch.setattr(an, "HERE", str(tmp_path))
    got = an.shape_at(str(tmp_path / "viz" / "delta"), read_area(tmp_path / "viz" / "delta" / "area.csv"), 0.0)
    assert got is not None and got[0] == 0.0
    assert np.allclose(got[1], read_vertex(os.path.join(CASE, "band.vertex")), rtol=0, atol=1e-15)
    png = an.main()
    assert png == str(tmp_path / "results.png") and os.path.getsize(png) > 10000
    text = capsys.readouterr().out
    assert "input2d (delta" in text and "input2d_blob (blob" in text


@pytest.mark.slow
@pytest.mark.parametrize("input_name, coupling, area_bound", (("input2d", "delta", 1e-3),
                                                              ("input2d_blob", "blob", 1e-4)))
def test_full_run_holds_the_area_and_rounds_up(tmp_path, input_name, coupling, area_bound):
    """1000 steps to t = 2 s: the area is held (measured largest change 5.7e-4 for the delta
    function, 3.2e-5 for the blob with its divergence-free stream-function interpolation), the
    band passes within 2 % of a circle (measured rmax/rmin 1.0097 and 1.0095) and rings back
    to no worse than 1.2 (1.111 at t = 2 s), with no net force on it."""
    res = run(CASE, input_file=input_name, quiet=True, output_dir=str(tmp_path))
    assert res.steps == 1000 and len(res.history) == 51
    assert np.all(np.isfinite(res.history.view(float)))
    assert np.max(np.abs(res.history.fx_body)) < 1e-12 and np.max(np.abs(res.history.fy_body)) < 1e-12
    assert res.history.cfl.max() < 0.05
    a = read_area(tmp_path / "area.csv")
    rel = a.area / a.area[0] - 1.0
    assert np.max(np.abs(rel)) < area_bound
    if coupling == "delta":
        assert rel[-1] < -3e-4                    # the steady loss once the band is round
    assert a.rmax_over_rmin.min() < 1.02
    assert a.rmax_over_rmin[-1] < 1.2
    assert a.perimeter[-1] == pytest.approx(2 * np.pi * np.sqrt(0.2 * 0.1), rel=5e-3)
    print(f"\n{input_name}: {res.wall_time:.1f} s, area change {rel[-1]:+.2e} (max |{np.max(np.abs(rel)):.2e}|), "
          f"rmax/rmin min {a.rmax_over_rmin.min():.4f} final {a.rmax_over_rmin[-1]:.4f}")
