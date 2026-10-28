"""examples/swimmer: smoke run of the coarse input, the committed files against generate.py,
the Carling kinematics of hooks.py, and (slow) the coarse run to the end with its swimming
speed, curvature following and stretch checked against the numbers in the README."""

from __future__ import annotations

import importlib.util
import os
import warnings

import numpy as np
import pytest

from pyibm.driver import run
from pyibm.inputfile import read_input_file
from pyibm.io.history import read_history
from pyibm.structures import load_hooks, read_beam, read_spring, read_structure, read_vertex

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CASE = os.path.join(ROOT, "examples", "swimmer")


def load_generate():
    spec = importlib.util.spec_from_file_location("swimmer_generate_test", os.path.join(CASE, "generate.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_smoke_three_steps(tmp_path):
    """Three midpoint steps of the coarse case: history rows [0, 3], finite diagnostics, the
    body still straight and unstretched (the amplitude ramp starts from zero), swimmer.csv
    written by the on_output hook. No tether warning: the driver's rule of thumb covers only
    target points and mass tethers. The springs sit at k dt^2/(rho dx) = 5.2, well inside the
    limit of 20 to 80 measured for this case (README)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = run(CASE, input_file="input2d_coarse", steps=3, quiet=True, output_dir=str(tmp_path))
    assert not any("tether number" in str(w.message) for w in caught)
    assert res.steps == 3 and res.history.step.tolist() == [0.0, 3.0]   # print_dump = 100
    assert np.all(np.isfinite(res.history.view(float)))
    s = res.structure
    assert s.n == 65 and s.springs is not None and s.beams is not None
    assert s.targets is None and s.masses is None
    assert len(s.springs) == 64 and len(s.beams) == 63
    # internal forces only: no net force on the fluid
    assert abs(res.history.fx_body[-1]) < 1e-9 and abs(res.history.fy_body[-1]) < 1e-9
    assert os.path.isfile(tmp_path / "history.csv")
    assert os.path.isfile(tmp_path / "lag.00003.vtk")
    b = read_history(tmp_path / "swimmer.csv")
    assert len(b) == 2 and b.step.tolist() == [0.0, 3.0]
    assert b.max_stretch[-1] < 1e-4 and abs(b.length[-1] - 1.0) < 1e-6
    assert abs(b.heading[-1]) < 1e-3
    assert b.curv_rms_prescribed[-1] > 0.0     # the hook has set the beams at t = 3.5 dt


def test_committed_files_match_generate():
    gen = load_generate()
    assert gen.L == 1.0 and gen.WAVELENGTH == 1.0 and gen.PERIOD == 1.0 and gen.AMPLITUDE_TAIL == 0.125
    for name, input_name, n in (("swimmer", "input2d", 129), ("swimmer_coarse", "input2d_coarse", 65)):
        X, ds = gen.body_points(name)
        assert X.shape == (n, 2)
        assert np.allclose(np.diff(X[:, 0]), ds) and np.allclose(X[:, 1], X[0, 1])
        assert abs(X[-1, 0] - X[0, 0] - gen.L) < 1e-12
        V = read_vertex(os.path.join(CASE, name + ".vertex"))
        assert np.array_equal(V, X)
        sp = read_spring(os.path.join(CASE, name + ".spring"), n)
        assert len(sp) == n - 1 and np.allclose(sp.rest, ds) and np.allclose(sp.k, gen.k_spring(ds))
        assert np.allclose(sp.k * ds ** 2, gen.EA)
        bm = read_beam(os.path.join(CASE, name + ".beam"), n)
        assert len(bm) == n - 2 and np.all(bm.curv == 0.0) and np.allclose(bm.k, gen.k_beam(ds))
        assert np.allclose(bm.k * ds ** 6, gen.EI)
        assert bm.q.tolist() == list(range(1, n - 1))
        p = read_input_file(os.path.join(CASE, input_name))
        c = gen.CASES[name]
        assert (p.grid.Nx, p.grid.Ny, p.grid.Lx, p.grid.Ly) == (c["Nx"], c["Ny"], c["Lx"], c["Ly"])
        assert p.time.dt == c["dt"] and abs(p.time.Tfinal - c["periods"] * gen.PERIOD) < 1e-12
        assert (p.fluid.rho, p.fluid.mu, p.structure.ds, p.structure.name) == (gen.RHO, gen.MU, ds, name)
        assert p.solver.backend == "periodic_fft" and p.solver.time_scheme == "midpoint"
        assert p.solver.coupling == "delta" and p.solver.kernel == "peskin4"
        assert abs(ds - 0.5 * p.grid.dx) < 1e-15


def test_carling_kinematics():
    """The hook's midline is Carling et al. 1998 eq. 2: linear envelope from 0.0038 L to
    0.125 L, one wavelength on the body, the wave travelling from the head (x = 0) to the
    tail; the ramp goes from 0 to 1 over one period."""
    gen = load_generate()
    hooks = load_hooks(CASE)
    m = hooks.module
    x = np.linspace(0.0, 1.0, 11)
    assert np.allclose(m.envelope(x), 0.125 * (x + 0.03125) / 1.03125)
    assert m.ramp(0.0) == 0.0 and m.ramp(0.5) == pytest.approx(0.5) and m.ramp(1.0) == 1.0 and m.ramp(7.0) == 1.0
    h1, hx, hxx = m.midline(x, 2.0)
    assert np.allclose(h1, m.envelope(x) * np.sin(2 * np.pi * x))
    # finite-difference check of the derivatives
    eps = 1e-6
    hp = m.midline(x + eps, 2.0)[0]
    hm = m.midline(x - eps, 2.0)[0]
    assert np.allclose((hp - hm) / (2 * eps), hx, atol=1e-7)
    assert np.allclose((hp - 2 * h1 + hm) / eps ** 2, hxx, atol=1e-3)
    # the crest moves towards the tail (+x) as time goes on
    xf = np.linspace(0.0, 1.0, 2001)
    crest = [xf[np.argmax(m.midline(xf, t)[0] / np.maximum(m.envelope(xf), 1e-9))] for t in (2.0, 2.1, 2.2)]
    assert crest[0] < crest[1] < crest[2]
    # the prescribed shape is traced with arc length and reaches the full tail amplitude
    s_nodes = np.arange(129) / 128.0
    P = m.prescribed_shape(s_nodes, 2.25)
    seg = np.diff(P, axis=0)
    assert np.allclose(np.hypot(seg[:, 0], seg[:, 1]), 1.0 / 128, rtol=2e-3)
    ratio = []
    for t in np.linspace(2.0, 3.0, 101):
        P = m.prescribed_shape(s_nodes, t)
        ratio.append(abs(P[-1, 1]) / m.envelope(P[-1, 0]))
    assert 0.995 < max(ratio) <= 1.0 + 1e-9     # the tail sweeps the full envelope once per period


def test_update_beams_matches_prescribed_shape():
    """update_beams gives every beam the cross product (X_r - X_q) x (X_q - X_p) of the
    prescribed shape sampled at the same arc lengths (to a few 1e-4 relative, the
    difference between sin(kappa ds) of the graph curvature and the sampled chords)."""
    gen = load_generate()
    hooks = load_hooks(CASE)
    m = hooks.module
    s = read_structure(CASE, "swimmer", ds=gen.body_points("swimmer")[1])
    ds = float(s.ds[0])
    for t in (0.4, 1.3, 2.75):
        hooks.update_beams(t, 1e-4, s)
        P = m.prescribed_shape(np.arange(s.n) * ds, t)
        b = s.beams
        A = P[b.r] - P[b.q]
        B = P[b.q] - P[b.p]
        c = A[:, 0] * B[:, 1] - A[:, 1] * B[:, 0]
        assert np.max(np.abs(c - b.curv)) < 1e-3 * np.max(np.abs(b.curv))
        assert np.max(np.abs(b.curv)) > 0.0
    hooks.update_beams(0.0, 1e-4, s)
    assert np.all(s.beams.curv == 0.0)     # the ramp starts from a straight body


@pytest.mark.slow
def test_coarse_run_swims(tmp_path):
    """The coarse case to the end (6 periods): the body swims head first (towards -x) at a
    period-averaged speed between 0.02 and 0.12 of the wave speed once the ramp is over
    (measured 0.03 on this grid, 0.09 on the fine one), follows the prescribed curvature to
    within 25 % RMS, stretches less than 3 %, and keeps its length to 0.1 %."""
    gen = load_generate()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = run(CASE, input_file="input2d_coarse", quiet=True, output_dir=str(tmp_path))
    assert res.steps == gen.CASES["swimmer_coarse"]["periods"] * gen.PERIOD / gen.CASES["swimmer_coarse"]["dt"]
    b = read_history(tmp_path / "swimmer.csv")
    V = gen.WAVELENGTH / gen.PERIOD
    settled = b.t > gen.RAMP_PERIODS * gen.PERIOD
    U = float(np.mean(-b.uc[settled]))
    assert 0.02 * V < U < 0.12 * V
    err = np.mean(b.curv_rms_error[settled] / b.curv_rms_prescribed[settled])
    assert err < 0.25
    assert b.max_stretch.max() < 0.03
    assert np.max(np.abs(b.length - gen.L)) < 1e-3 * gen.L
    assert np.all(res.history.cfl < 0.5)
    print(f"\ncoarse swimmer: {res.wall_time:.1f} s, U/V = {U / V:.4f}, curvature error {err:.3f}, "
          f"max stretch {b.max_stretch.max():.2e}")
