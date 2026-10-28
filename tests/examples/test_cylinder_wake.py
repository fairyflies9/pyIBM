"""examples/cylinder_wake: smoke runs of the four inputs, the committed files against
generate.py, the hooks (free-stream ramp, wake kick, snapshots), the analysis functions on
synthetic lift and drag signals, and (slow) the coarse input to the end and the first 30 time
units of the default input, against the values in README.md and the Strouhal number of Lai &
Peskin, J. Comput. Phys. 160 (2000) 705-719."""

from __future__ import annotations

import dataclasses
import importlib.util
import os
import warnings

import numpy as np
import pytest

from pyibm.driver import TETHER_STABLE, run
from pyibm.grid import Grid
from pyibm.inputfile import read_input_file, write_input_file
from pyibm.io.history import read_history
from pyibm.structures import load_hooks, read_target, read_vertex

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CASE = os.path.join(ROOT, "examples", "cylinder_wake")
INPUTS = ("input2d", "input2d_re200", "input2d_fine", "input2d_coarse")

# Measured over t = 60-100 of the committed runs (README.md, "What to expect").
ST_RE100, CD_RE100, CL_AMP_RE100 = 0.1765, 1.5705, 0.3603      # input2d
ST_COARSE, CD_COARSE, CL_AMP_COARSE = 0.1704, 1.6070, 0.4025   # input2d_coarse
# input2d over t = 15-30 (the k = 64000 row of the k sweep in README.md): the lift is still
# 1.5 % above its value over t = 60-100 there.
ST_RE100_EARLY, CD_RE100_EARLY, CL_AMP_RE100_EARLY = 0.1767, 1.5730, 0.3658

# Messages of pyIBM's own warnings about a set-up (pyibm/driver.py, pyibm/structures.py,
# pyibm/fluid/mac.py). The example must raise none of them. Warnings from other libraries (a
# numpy deprecation, say) are not the example's business and must not fail these tests.
SETUP_WARNINGS = ("tether number", "CFL number", "nearest-neighbour distances", "explicit viscosity")


def load_module(name: str):
    spec = importlib.util.spec_from_file_location(f"cylinder_wake_{name}", os.path.join(CASE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def setup_warnings(messages) -> list[str]:
    """The messages among ``messages`` that are pyIBM set-up warnings (SETUP_WARNINGS)."""
    return [m for m in messages if any(key in m for key in SETUP_WARNINGS)]


def run_quietly(input_name: str, output_dir, steps=None):
    """Run an input (a name in CASE or an absolute path) and return (RunResult, the messages
    of the pyIBM set-up warnings it issued)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = run(CASE, input_file=input_name, steps=steps, quiet=True, output_dir=str(output_dir))
    return res, setup_warnings(str(w.message) for w in caught)


def test_only_setup_warnings_count(tmp_path, monkeypatch):
    """A warning from another library during a run does not fail the example's checks; a
    tether warning does. The stand-in is a DeprecationWarning issued inside the run; the
    tether warning comes from the coarse input with dt doubled (tether number 31)."""
    real_run = run

    def noisy_run(*args, **kwargs):
        warnings.warn("some_function is deprecated", DeprecationWarning)
        return real_run(*args, **kwargs)

    monkeypatch.setitem(globals(), "run", noisy_run)
    res, messages = run_quietly("input2d_coarse", tmp_path / "quiet", steps=1)
    assert res.steps == 1 and messages == []
    p = read_input_file(os.path.join(CASE, "input2d_coarse"))
    fast = tmp_path / "input2d_coarse_2dt"
    write_input_file(str(fast), dataclasses.replace(p, time=dataclasses.replace(p.time, dt=2.0 * p.time.dt)))
    res, messages = run_quietly(str(fast), tmp_path / "loud", steps=1)
    assert len(messages) == 1 and messages[0].startswith("tether number")


@pytest.mark.parametrize("input_name", INPUTS)
def test_smoke_three_steps(tmp_path, input_name):
    """Three steps of every input, with no pyIBM set-up warning: the tether number 7.5-7.8 is
    below the driver's rule-of-thumb value for targets."""
    res, messages = run_quietly(input_name, tmp_path, steps=3)
    assert messages == []
    assert res.steps == 3 and res.history.step.tolist() == [0.0, 3.0]   # print_dump >= 3
    assert np.all(np.isfinite(res.history.view(float)))
    assert res.structure.targets is not None and res.structure.n == len(res.structure.targets)
    assert os.path.isfile(tmp_path / "history.csv")
    assert len(read_history(tmp_path / "history.csv")) == 2
    assert not os.path.isfile(tmp_path / "lag.00003.vtk")             # save_vtk is off
    # the force of the last row is the midpoint step's, half a step before t
    assert res.history.t_force[-1] == pytest.approx(res.history.t[-1] - 0.5 * res.parameters.time.dt)
    # the free stream has barely started (quintic ramp, t = 3 dt << T_RAMP): a tiny
    # added-mass force in +x, no lift on the symmetric start, markers still on the circle
    assert 0.0 <= res.history.fx_body[-1] < 0.5
    assert abs(res.history.fy_body[-1]) < 1e-8
    assert res.history.max_target_error_over_dx[-1] < 0.01
    assert res.history.cfl[-1] < 0.05
    assert res.history.max_speed[-1] < 0.05


def test_committed_files_match_generate():
    gen = load_module("generate")
    for name, (Nx, Ny, dt, k) in gen.RESOLUTIONS.items():
        dx = gen.LX / Nx
        X = gen.cylinder_points(Nx)
        n = gen.n_markers(Nx)
        assert X.shape == (n, 2)
        # spacing ds = dx along the circumference (a ring at ds = dx/2 grows a grid-scale
        # wrinkle that makes the wake irregular, README.md), markers on the circle, 2 cells
        # off the walls
        assert abs(gen.marker_ds(Nx) - dx) < 0.02 * dx
        assert np.allclose(np.hypot(X[:, 0] - gen.XC, X[:, 1] - gen.YC), 0.5 * gen.D)
        assert np.allclose(np.hypot(*np.diff(np.vstack([X, X[:1]]), axis=0).T), 2 * np.sin(np.pi / n) * 0.5 * gen.D)
        assert X[:, 1].min() > 2 * gen.LY / Ny and X[:, 1].max() < gen.LY - 2 * gen.LY / Ny
        assert X[:, 0].min() > 2 * dx
        # a stiff tether: in the k sweeps of README.md the forces settle from k dx of a few
        # thousand rho U^2 on; the soft tethers of the earlier builds (k dx = 150-170 rho U^2)
        # gave C_D 20-42 % and the lift amplitude 45-155 % above the stiffest runs
        assert k * dx >= 2400.0 * gen.RHO * gen.U**2
        # dt keeps the tether number below the driver's rule-of-thumb value for targets and
        # divides the output interval
        assert 7.0 < gen.tether_number(name) < 8.0 < TETHER_STABLE[("target", "midpoint")]
        assert gen.OUTPUT_INTERVAL / dt == pytest.approx(round(gen.OUTPUT_INTERVAL / dt), abs=1e-9)
        V = read_vertex(os.path.join(CASE, name + ".vertex"))
        assert np.array_equal(V, X)
        T = read_target(os.path.join(CASE, name + ".target"), V)
        assert np.all(T.k == k) and T.idx.tolist() == list(range(n)) and np.all(T.c == 0.0)
    for input_name, (structure, Re, out, _) in gen.CASES.items():
        p = read_input_file(os.path.join(CASE, input_name))
        Nx, Ny, dt, k = gen.RESOLUTIONS[structure]
        assert (p.grid.Nx, p.grid.Ny, p.grid.Lx, p.grid.Ly) == (Nx, Ny, gen.LX, gen.LY)
        assert p.time.dt == dt and p.time.Tfinal == gen.T_FINAL
        assert p.fluid.rho == gen.RHO and p.fluid.mu == pytest.approx(gen.RHO * gen.U * gen.D / Re)
        assert p.structure.name == structure and p.structure.ds == pytest.approx(gen.marker_ds(Nx))
        assert p.solver.backend == "mac" and p.solver.time_scheme == "midpoint"
        assert p.solver.coupling == "delta" and p.solver.kernel == "peskin4"
        assert p.solver.advection == "central" and p.solver.viscous == "implicit"
        assert (p.bc.left, p.bc.right, p.bc.bottom, p.bc.top) == ("dirichlet", "outflow", "dirichlet", "dirichlet")
        assert p.output.output_dir == out and not p.output.save_vtk and p.output.save_history
        assert p.output.print_dump == round(gen.OUTPUT_INTERVAL / dt)
    assert set(gen.CASES) == set(INPUTS)


def test_hooks_free_stream_ramp_and_kick():
    gen = load_module("generate")
    hooks = load_hooks(CASE)
    m = hooks.module
    assert (m.U, m.D, m.XC, m.YC) == (gen.U, gen.D, gen.XC, gen.YC)
    # quintic ramp: 0 at the start, U/2 half way, U from T_RAMP on, C^2 at both ends
    y = np.linspace(0.0, gen.LY, 7)
    for side in ("bc_left", "bc_bottom", "bc_top"):
        fn = getattr(hooks, side)
        u, v = fn(y, 0.0)
        assert np.all(u == 0.0) and np.all(v == 0.0) and u.shape == y.shape
        u, v = fn(y, 0.5 * m.T_RAMP)
        assert np.allclose(u, 0.5 * gen.U) and np.all(v == 0.0)
        u, v = fn(y, 50.0)
        assert np.allclose(u, gen.U)
    s = m.smoothstep5(np.array([0.0, 0.5, 1.0, 1.5, -0.2]))
    assert np.allclose(s, [0.0, 0.5, 1.0, 1.0, 0.0])
    eps = 1e-3
    for x0 in (0.0, 1.0):
        d1 = (m.smoothstep5(x0 + eps) - m.smoothstep5(x0 - eps)) / (2 * eps)
        d2 = (m.smoothstep5(x0 + eps) - 2 * m.smoothstep5(x0) + m.smoothstep5(x0 - eps)) / eps**2
        assert abs(d1) < 1e-5 and abs(d2) < 1e-2
    assert hooks.bc_right is None and hooks.update_targets is None
    # the kick: zero outside its window, a vertical Gaussian bump downstream inside it
    g = Grid(gen.LX, gen.LY, 64, 32, periodic_x=False, periodic_y=False)
    u = np.zeros(g.shape_u())
    v = np.zeros(g.shape_v())
    for t in (0.0, m.KICK_T0, m.KICK_T1, 50.0):
        Eu, Ev = hooks.external_force(t, g, u, v)
        assert Eu.shape == g.shape_u() and Ev.shape == g.shape_v()
        assert np.all(Eu == 0.0) and np.all(Ev == 0.0)
    t_mid = 0.5 * (m.KICK_T0 + m.KICK_T1)
    Eu, Ev = hooks.external_force(t_mid, g, u, v)
    assert np.all(Eu == 0.0) and np.all(Ev >= 0.0)
    assert Ev.max() == pytest.approx(m.KICK_AMPLITUDE, rel=0.05)   # the peak sits between faces
    X, Y = g.meshgrid("v")
    i, j = np.unravel_index(np.argmax(Ev), Ev.shape)
    assert abs(X[i, j] - (gen.XC + m.KICK_X_OFFSET * gen.D)) <= g.dx
    assert abs(Y[i, j] - gen.YC) <= g.dy
    assert Ev[0, :].max() < 1e-12 and Ev[-1, :].max() < 1e-12       # confined to the wake


def test_analysis_recovers_a_synthetic_strouhal_number():
    ana = load_module("analyze")
    t = np.arange(0.0, 100.0, 0.1)
    St, cd0, amp = 0.165, 1.4, 0.33
    cl = amp * np.sin(2.0 * np.pi * St * t) + 0.02
    cd = cd0 + 0.05 * np.cos(4.0 * np.pi * St * t)
    r = ana.analyse(t, cd, cl, t_start=60.0)
    assert r["n_crossings"] >= 6 and r["periods"] == len(r["St_cycles"]) == len(r["CL_amp_cycles"])
    assert abs(r["St"] - St) < 2e-3
    assert abs(r["St_fft"] - St) < 4e-3               # zero-padded bins 1/320 apart
    assert r["St_spread"] < 2e-3                      # a pure sine: every cycle the same
    assert abs(r["CD"] - cd0) < 5e-3                  # whole periods: the cos averages out
    assert np.allclose(r["CD_cycles"], cd0, atol=5e-3)
    assert abs(r["CD_osc"] - 0.1) < 2e-3 and abs(r["CD_drift"]) < 1e-4
    assert abs(r["CL_amp"] - amp) < 5e-3
    assert abs(r["CL_rms"] - amp / np.sqrt(2.0)) < 5e-3 and abs(r["CL_rms_amp"] - amp) < 5e-3
    r = ana.analyse(t, cd, cl, t_start=99.0)           # too short: no crossings
    assert np.isnan(r["St"]) and r["n_crossings"] < 2 and len(r["St_cycles"]) == 0


def test_analysis_of_a_modulated_wake():
    """A lift whose amplitude swings by 20 % over the window and whose frequency wanders
    by 3 %, and a drag with a 2f oscillation on a slow drift. The lift amplitude is the mean
    of the per-cycle half peak-to-peak, not half the max - min over the window (that is the
    largest cycle, 20 % high here); the per-cycle St shows the wander; the 2f peak-to-peak
    is not mixed up with the drift."""
    ana = load_module("analyze")
    t = np.arange(0.0, 100.0, 0.1)
    St, amp = 0.165, 0.5
    phase = 2.0 * np.pi * St * (t - 0.03 * 25.0 / (2.0 * np.pi) * np.cos(2.0 * np.pi * t / 25.0))
    env = 1.0 + 0.2 * np.sin(2.0 * np.pi * (t - 60.0) / 40.0)       # zero mean over t = 60-100
    cl = amp * env * np.sin(phase)
    cd = 1.9 + 0.003 * (t - 80.0) + 0.015 * np.cos(2.0 * phase)
    r = ana.analyse(t, cd, cl, t_start=60.0)
    window = 0.5 * np.ptp(cl[t >= 60.0])
    assert window > 1.15 * amp
    assert abs(r["CL_amp"] - amp) < 0.05 * amp
    assert np.all(r["CL_amp_cycles"] < window + 1e-12) and np.ptp(r["CL_amp_cycles"]) > 0.2 * amp
    assert abs(r["CL_rms_amp"] - amp) < 0.05 * amp
    assert 0.03 < r["St_spread"] < 0.08 and abs(r["St"] - St) < 0.02 * St
    assert abs(r["CD_osc"] - 0.03) < 0.005 and abs(r["CD_drift"] - 0.003) < 3e-4
    assert np.ptp(r["CD_cycles"]) > 0.08                              # the drift, cycle by cycle


def test_coefficients_use_the_time_of_the_force():
    """C_D, C_L are placed at t_force (the time level of the step force, t - dt/2 for the
    midpoint scheme), and at t for an old history without that column."""
    ana = load_module("analyze")
    t = np.array([0.0, 0.1, 0.2])
    tf = np.array([0.0, 0.09375, 0.19375])
    f = np.array([0.0, 0.5, 1.0])
    h = np.rec.fromarrays([t, tf, f, -f], names=["t", "t_force", "fx_body", "fy_body"])
    tt, cd, cl = ana.coefficients(h, 1.0, 1.0, 1.0)
    assert np.array_equal(tt, tf) and np.allclose(cd, 2 * f) and np.allclose(cl, -2 * f)
    old = np.rec.fromarrays([t, f, -f], names=["t", "fx_body", "fy_body"])
    tt, cd, cl = ana.coefficients(old, 2.0, 1.0, 1.0)
    assert np.array_equal(tt, t) and np.allclose(cd, f)


def wake_statistics(res, t_start=None):
    """analyze.analyse of a run from t_start (default: the last ANALYSIS_WINDOW time units),
    with a one-line summary printed for the log."""
    gen = load_module("generate")
    ana = load_module("analyze")
    h = res.history
    t, cd, cl = ana.coefficients(h, gen.RHO, gen.U, gen.D)
    r = ana.analyse(t, cd, cl, t[-1] - ana.ANALYSIS_WINDOW if t_start is None else t_start, gen.U, gen.D)
    print(f"\n{res.steps} steps in {res.wall_time:.0f} s: t = {r['t_start']:g}-{r['t_end']:.0f}, "
          f"St {r['St']:.4f} (fft {r['St_fft']:.4f}, spread {100 * r['St_spread']:.3f} %), "
          f"C_D {r['CD']:.4f}, C_L amp {r['CL_amp']:.4f}, periods {r['periods']:g}, "
          f"max target error {h.max_target_error_over_dx.max():.4f} cells, max cfl {h.cfl.max():.3f}")
    return r


@pytest.mark.slow
def test_coarse_input_to_the_end(tmp_path):
    """input2d_coarse (Re 100, D/dx = 8) to T_FINAL, 14000 steps, 77 s on a laptop.
    Over the last ANALYSIS_WINDOW time units the wake is periodic (the St of every shedding
    cycle within 1 % of the others) and the ring stays within 0.01 cells of its targets. With
    the softer tether of the earlier builds (k = 1200) a static grid-scale pattern of 0.15
    cells stayed on the ring here, and the ds = dx/2 ring of the first build grew a wrinkle
    of 0.76 cells. St is within 5 % of Lai & Peskin's 0.165 (Table II, Scheme 2), and St,
    C_D and the lift amplitude are within 2 % of the committed run's values (the table in
    README.md, "What to expect")."""
    gen = load_module("generate")
    res, messages = run_quietly("input2d_coarse", tmp_path)
    assert messages == []
    h = res.history
    r = wake_statistics(res)
    assert res.steps == round(gen.T_FINAL / gen.RESOLUTIONS["cylinder_coarse"][2])
    assert r["periods"] >= 5
    assert r["St_spread"] < 0.01
    assert h.max_target_error_over_dx.max() < 0.01
    assert abs(r["St"] - 0.165) < 0.05 * 0.165
    assert r["St"] == pytest.approx(ST_COARSE, rel=0.02)
    assert r["CD"] == pytest.approx(CD_COARSE, rel=0.02)
    assert r["CL_amp"] == pytest.approx(CL_AMP_COARSE, rel=0.02)
    assert np.all(h.cfl < 1.0)


@pytest.mark.slow
def test_re100_default_input_first_30_time_units(tmp_path):
    """The default input (Re 100, D/dx = 12, 16 x 8 box) for 30 time units, 9600 steps, 143 s
    on a laptop (the whole run is 32000 steps). The lift reaches 95 % of its final
    amplitude by t = 6.1, and over t = 15-30 every shedding cycle has the same St to 1 %. The
    ring stays within 0.01 cells of its targets, St is within 10 % of Lai & Peskin's 0.165,
    St, C_D and the lift amplitude are within 1 % of the committed run's values over the same
    window (the k sweep in README.md), and St and C_D within 2 % of its values over t = 60-100
    (the table in README.md, "What to expect")."""
    gen = load_module("generate")
    dt = gen.RESOLUTIONS["cylinder"][2]
    res, messages = run_quietly("input2d", tmp_path, steps=round(30.0 / dt))
    assert messages == []
    h = res.history
    r = wake_statistics(res, t_start=15.0)
    assert h.t[-1] == pytest.approx(30.0)
    assert r["periods"] >= 2
    assert r["St_spread"] < 0.01
    assert h.max_target_error_over_dx.max() < 0.01
    assert abs(r["St"] - 0.165) < 0.10 * 0.165
    assert r["St"] == pytest.approx(ST_RE100_EARLY, rel=0.01)
    assert r["CD"] == pytest.approx(CD_RE100_EARLY, rel=0.01)
    assert r["CL_amp"] == pytest.approx(CL_AMP_RE100_EARLY, rel=0.01)
    assert r["St"] == pytest.approx(ST_RE100, rel=0.02)
    assert r["CD"] == pytest.approx(CD_RE100, rel=0.02)
    assert np.all(h.cfl < 1.0)
