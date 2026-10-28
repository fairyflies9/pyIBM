"""examples/lid_driven_cavity: smoke runs of the Re = 100 input and of the oscillating lid
(input2d_oscillating, which names hooks_oscillating.py in its Hooks section), the committed
inputs against generate.py, the hooks, the Ghia tables, and (slow) the three
steady runs against Tables I and II of Ghia, Ghia & Shin 1982 (each run is checked to be
steady at the end, |dKE/dt|/KE < 1e-4, so Tfinal matters; generate.py explains the choice)."""

from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest

from pyibm.driver import run
from pyibm.inputfile import read_input_file
from pyibm.io.history import read_history
from pyibm.structures import load_hooks

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CASE = os.path.join(ROOT, "examples", "lid_driven_cavity")
STEADY_INPUTS = ("input2d", "input2d_re400", "input2d_re1000")
OSC_INPUT = "input2d_oscillating"
OSC_HOOKS = "hooks_oscillating.py"


def load(name: str):
    spec = importlib.util.spec_from_file_location(f"lid_driven_cavity_{name}", os.path.join(CASE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_smoke_three_steps_re100(tmp_path):
    res = run(CASE, steps=3, quiet=True, output_dir=str(tmp_path))
    assert res.steps == 3 and res.history.step.tolist() == [0.0, 3.0]   # print_dump = 80
    assert np.all(np.isfinite(res.history.view(float)))
    assert res.structure is None
    assert res.history.fx_body.tolist() == [0.0, 0.0] and res.history.kinetic_energy[0] == 0.0
    assert res.history.kinetic_energy[1] > 0.0 and res.history.max_speed[1] > 0.0
    g, s = res.grid, res.solver
    # the lid drags the top row only; the wall faces carry u = 0
    assert np.max(s.u[:, -1]) > 10 * np.max(np.abs(s.u[:, 0]))
    assert np.all(s.u[0] == 0.0) and np.all(s.u[-1] == 0.0)
    assert np.max(np.abs(s.divergence())) < 1e-12
    assert len(read_history(tmp_path / "history.csv")) == 2
    assert os.path.isfile(tmp_path / "velocity.00003.vtk")
    # the centreline files of the hook: wall to wall, u = 1 on the lid, v = 0 on the side walls
    an = load("analyze")
    y, u, x, v = an.read_centrelines(str(tmp_path))
    assert len(y) == g.Ny + 2 and y[0] == 0.0 and y[-1] == g.Ly and u[0] == 0.0 and u[-1] == 1.0
    assert len(x) == g.Nx + 2 and x[0] == 0.0 and x[-1] == g.Lx and v[0] == 0.0 and v[-1] == 0.0
    assert np.array_equal(u[1:-1], s.u[g.Nx // 2]) and np.array_equal(v[1:-1], s.v[:, g.Ny // 2])


def test_smoke_three_steps_oscillating(tmp_path):
    res = run(CASE, input_file=OSC_INPUT, steps=3, quiet=True, output_dir=str(tmp_path))
    assert res.steps == 3 and res.history.step.tolist() == [0.0, 3.0]   # print_dump = 8
    assert np.all(np.isfinite(res.history.view(float)))
    assert res.history.kinetic_energy[1] > 0.0
    assert not os.path.isfile(tmp_path / "velocity.00003.vtk")               # save_vtk = 0
    # the run used hooks_oscillating.py, not the steady hooks.py beside it: no on_output
    # centreline files, and the lid at t = 3 dt moves at sin(2 pi t), not at 1
    assert res.parameters.hooks.file == OSC_HOOKS
    assert not os.path.isfile(tmp_path / "u_centreline.csv")
    g = res.grid
    hooks = load_hooks(CASE, OSC_HOOKS)
    assert hooks.on_output is None and hooks.bc_left is None
    u_lid = hooks.bc_top(g.xc, res.t)[0]
    assert u_lid == pytest.approx(np.sin(2 * np.pi * res.t) * np.ones(g.Nx))
    assert 0.0 < np.max(res.solver.u[:, -1]) < u_lid[0] < 0.1   # a lid at U = 1 gives about 0.6
    xs = np.linspace(0.0, 1.0, 5)
    assert hooks.bc_top(xs, 0.25)[0] == pytest.approx(np.ones(5))
    assert hooks.bc_top(xs, 0.5)[0] == pytest.approx(np.zeros(5), abs=1e-15)
    assert hooks.bc_top(xs, 0.75)[0] == pytest.approx(-np.ones(5))
    assert np.all(hooks.bc_top(xs, 0.3)[1] == 0.0)
    assert np.max(np.abs(res.solver.divergence())) < 1e-12


def test_committed_inputs_match_generate(tmp_path):
    gen = load("generate")
    written = gen.main(str(tmp_path))
    assert [os.path.basename(p) for p in written] == list(STEADY_INPUTS) + [OSC_INPUT]
    for path in written:   # byte for byte, header included
        with open(path, "rb") as a, open(os.path.join(CASE, os.path.basename(path)), "rb") as b:
            assert a.read() == b.read(), os.path.basename(path)
    for name in STEADY_INPUTS:
        p = read_input_file(os.path.join(CASE, name))
        assert p == gen.steady_parameters(name)
        assert p.solver.backend == "mac" and p.structure.name is None
        assert all(s == "dirichlet" for s in (p.bc.left, p.bc.right, p.bc.bottom, p.bc.top))
        assert p.time.dt == pytest.approx(gen.CFL / p.grid.Nx)
        assert p.fluid.mu == pytest.approx(1.0 / {"input2d": 100, "input2d_re400": 400, "input2d_re1000": 1000}[name])
        assert p.hooks.file == "hooks.py"
    p = read_input_file(os.path.join(CASE, OSC_INPUT))
    assert p == gen.oscillating_parameters()
    assert p.time.Tfinal == pytest.approx(4.0) and p.output.save_vtk is False
    assert p.hooks.file == OSC_HOOKS and p.output.output_dir == "viz/oscillating"
    # the oscillating case lives beside the steady ones now, not in a subfolder of its own
    assert not os.path.exists(os.path.join(CASE, "oscillating"))


def test_steady_lid_hook():
    hooks = load_hooks(CASE)
    xs = np.linspace(0.0, 1.0, 7)
    for t in (0.0, 1.0, 12.5):
        u, v = hooks.bc_top(xs, t)
        assert np.all(u == 1.0) and np.all(v == 0.0) and u.shape == xs.shape
    assert hooks.bc_left is None and hooks.bc_right is None and hooks.bc_bottom is None


def test_ghia_tables():
    ghia = load("ghia1982")
    assert len(ghia.Y) == len(ghia.X) == 17
    assert np.all(np.diff(ghia.Y) < 0) and np.all(np.diff(ghia.X) < 0)
    for Re in ghia.REYNOLDS:
        assert ghia.U[Re][0] == 1.0 and ghia.U[Re][-1] == 0.0
        assert ghia.V[Re][0] == 0.0 and ghia.V[Re][-1] == 0.0
        # the tables compared with themselves give zero error
        c = ghia.compare(ghia.Y, ghia.U[Re], ghia.X, ghia.V[Re], Re)
        assert c["max_du"] == 0.0 and c["max_dv"] == 0.0 and c["max_dv_all"] == 0.0
    assert not ghia.mask(400, "v")[5] and np.all(ghia.mask(100, "v")) and np.all(ghia.mask(400, "u"))


def _steady_run(tmp_path, input_name, Re):
    an = load("analyze")
    res = run(CASE, input_file=input_name, quiet=True, output_dir=str(tmp_path))
    c = an.report(Re, str(tmp_path), verbose=False)
    print(f"\n{input_name}: {res.steps} steps in {res.wall_time:.0f} s, max|u - Ghia| {c['max_du']:.4f}, "
          f"max|v - Ghia| {c['max_dv']:.4f}, steadiness {c['steadiness']:.1e}")
    assert np.all(res.history.cfl < 1.0)
    assert c["steadiness"] < 1e-4
    return c


@pytest.mark.slow
def test_re100_matches_ghia(tmp_path):
    """64^2 to t = 18: measured max |u - Ghia| = 0.0038, max |v - Ghia| = 0.0087 (lid speed 1),
    steadiness 2.2e-5. About 15 s."""
    c = _steady_run(tmp_path, "input2d", 100)
    assert c["max_du"] < 0.012 and c["max_dv"] < 0.012


@pytest.mark.slow
def test_re400_matches_ghia(tmp_path):
    """128^2 to t = 40: measured max |u - Ghia| = 0.0021, max |v - Ghia| = 0.0048, leaving out
    the tabulated v at x = 0.9063 (we get -0.387 there), steadiness 2.7e-5. About 4 minutes
    on an idle laptop (18 ms per step at 128^2)."""
    c = _steady_run(tmp_path, "input2d_re400", 400)
    assert c["max_du"] < 0.01 and c["max_dv"] < 0.01


@pytest.mark.slow
def test_re1000_matches_ghia(tmp_path):
    """128^2 to t = 70: measured max |u - Ghia| = 0.0033, max |v - Ghia| = 0.0119,
    steadiness 5.7e-5. About 7 minutes on an idle laptop (18 ms per step at 128^2)."""
    c = _steady_run(tmp_path, "input2d_re1000", 1000)
    assert c["max_du"] < 0.02 and c["max_dv"] < 0.02
