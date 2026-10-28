"""examples/stokes_second_problem: the exact solution, the committed inputs against
generate.py, the wall hook, smoke runs of the three inputs, and (slow) the full runs
compared with the damped shear wave at eight phases of the last period."""

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
CASE = os.path.join(ROOT, "examples", "stokes_second_problem")
INPUTS = ("input2d", "input2d_ny32", "input2d_ny128")
NY = {"input2d": 64, "input2d_ny32": 32, "input2d_ny128": 128}


def load(name: str):
    """Import <CASE>/<name>.py under a private module name (the heaving-plate pattern)."""
    spec = importlib.util.spec_from_file_location(f"stokes_second_problem_{name}",
                                                  os.path.join(CASE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ the exact solution
def test_exact_solution():
    ex = load("stokes_exact")
    assert ex.penetration_depth() == pytest.approx(ex.D)
    y = np.linspace(0.0, ex.LY, 257)
    for t in (0.0, 0.137, 0.5, 0.81):
        # eq. (1) solves u_t = nu u_yy (derivatives by central differences, so the check
        # does not reuse the algebra of the formula). What remains is the truncation error
        # of the differences, (eps_y/d)^2/6 + (omega eps_t)^2/6: 1.6e-7 at the default
        # steps of 1e-4 and a hundred times larger at 1e-3. A residual that falls by 100
        # when both steps fall by 10 says the formula is exact and only the check is
        # approximate (measured ratio 100.0 at every t; a wrong formula would give 1).
        assert ex.pde_residual(y[1:-1], t) < 1e-6
        coarse = ex.pde_residual(y[1:-1], t, eps_t=1e-3, eps_y=1e-3)
        fine = ex.pde_residual(y[1:-1], t, eps_t=1e-4, eps_y=1e-4)
        assert 80.0 < coarse / fine < 125.0
        # the wall moves as prescribed, the far field is at rest
        assert ex.velocity(0.0, t) == pytest.approx(ex.U * np.cos(ex.OMEGA * t))
        assert abs(ex.velocity(ex.LY, t)) < ex.U * np.exp(-9.0)
        # eq. (2): same wall motion, exactly at rest on the top wall, and within
        # U exp(-Ly/d) of eq. (1) everywhere
        assert ex.velocity_finite_depth(0.0, t) == pytest.approx(ex.U * np.cos(ex.OMEGA * t))
        assert abs(ex.velocity_finite_depth(ex.LY, t)) < 1e-12
        assert np.max(np.abs(ex.velocity_finite_depth(y, t) - ex.velocity(y, t))) < 2 * ex.U * np.exp(-ex.LY / ex.D)
    # the amplitude falls by e over one penetration depth and the phase lags by one radian
    t = np.linspace(0.0, ex.PERIOD, 401)
    assert np.max(ex.velocity(ex.D, t)) == pytest.approx(ex.U * np.exp(-1.0), rel=1e-4)
    assert t[np.argmax(ex.velocity(ex.D, t))] == pytest.approx(1.0 / ex.OMEGA, abs=ex.PERIOD / 400)


def test_wall_shear_and_wall_acceleration_peak_at_different_phases():
    """What the README says about where the error peaks: the wall shear u_y(0, t) =
    (U/d)(sin omega t - cos omega t) is largest in size at omega t = 135 and 315 degrees,
    while the wall acceleration |dU_w/dt| = U omega |sin omega t| (and with it the wall
    curvature u_yy = u_t/nu) is largest at 90 and 270 degrees, where the wall is at rest."""
    ex = load("stokes_exact")
    phase = np.arange(0.0, 360.0, 0.5)
    t = np.radians(phase) / ex.OMEGA
    eps = 1e-4                       # central differences in y, as in pde_residual
    shear = (ex.velocity(eps, t) - ex.velocity(-eps, t)) / (2 * eps)
    assert np.allclose(shear, ex.U / ex.D * (np.sin(ex.OMEGA * t) - np.cos(ex.OMEGA * t)),
                       rtol=1e-5, atol=1e-5 * ex.U / ex.D)
    curvature = (ex.velocity(eps, t) - 2 * ex.velocity(0.0, t) + ex.velocity(-eps, t)) / eps**2

    def peaks(f):
        """Phase of the largest |f| in each half period."""
        half = len(phase) // 2
        return [float(phase[np.argmax(np.abs(f[:half]))]),
                float(phase[half + np.argmax(np.abs(f[half:]))])]

    assert peaks(shear) == [135.0, 315.0]
    assert peaks(curvature) == [90.0, 270.0]
    assert np.allclose(ex.velocity(0.0, t)[np.isin(phase, (90.0, 270.0))], 0.0, atol=1e-12)


# ------------------------------------------------------------------ files and hooks
def test_committed_files_match_generate(tmp_path):
    gen = load("generate")
    ex = load("stokes_exact")
    written = gen.main(str(tmp_path))
    assert [os.path.basename(p) for p in written] == list(INPUTS)
    for path in written:   # byte for byte, header included
        with open(path, "rb") as a, open(os.path.join(CASE, os.path.basename(path)), "rb") as b:
            assert a.read() == b.read(), os.path.basename(path)
    assert gen.DT == ex.PERIOD / 400 and gen.PERIODS == 5 and gen.PRINT_DUMP == 50
    assert [v[0] for v in gen.VARIANTS] == list(INPUTS)
    for input_name in INPUTS:
        p = read_input_file(os.path.join(CASE, input_name))
        Ny = NY[input_name]
        assert (p.grid.Nx, p.grid.Ny) == (gen.NX, Ny) and p.grid.Ly == ex.LY
        assert p.grid.Lx == pytest.approx(gen.NX * ex.LY / Ny)       # square cells
        assert p.grid.dx == pytest.approx(p.grid.dy)
        assert (p.fluid.rho, p.fluid.mu) == (ex.RHO, ex.RHO * ex.NU)
        assert p.time.dt == gen.DT and p.time.n_steps == 2000
        assert (p.solver.backend, p.solver.time_scheme) == ("mac", "midpoint")
        assert (p.bc.left, p.bc.right, p.bc.bottom, p.bc.top) == ("periodic", "periodic", "dirichlet", "dirichlet")
        assert p.structure.name is None                               # fluid only
        assert p.output.print_dump == gen.PRINT_DUMP and p.output.output_dir == f"viz/ny{Ny}"
        # the written parameters equal what generate.py builds
        assert p == gen.parameters(Ny, f"viz/ny{Ny}")


def test_hook_moves_the_bottom_wall():
    ex = load("stokes_exact")
    hooks = load_hooks(CASE)
    assert hooks.bc_bottom is not None and hooks.bc_top is None and hooks.bc_left is None
    x = np.linspace(0.0, 0.125, 9)
    for t in (0.0, 0.3, 0.75, 2.2):
        u, v = hooks.bc_bottom(x, t)
        assert u.shape == x.shape and v.shape == x.shape
        assert np.allclose(u, ex.U * np.cos(ex.OMEGA * t)) and np.all(v == 0.0)


# ------------------------------------------------------------------ smoke runs
@pytest.mark.parametrize("input_name", INPUTS)
def test_smoke_three_steps(tmp_path, input_name):
    res = run(CASE, input_file=input_name, steps=3, quiet=True, output_dir=str(tmp_path))
    assert res.steps == 3 and res.history.step.tolist() == [0.0, 3.0]    # print_dump = 50
    assert np.all(np.isfinite(res.history.view(float)))
    assert res.structure is None
    assert res.history.max_target_error_over_dx[-1] == 0.0 and res.history.fx_body[-1] == 0.0
    assert res.history.max_speed[-1] > 0.0 and res.solver.kinetic_energy() > 0.0
    # one-dimensional: u is the same on every x face, v and p vanish
    u = res.solver.u
    assert u.shape == (8, NY[input_name])
    assert np.max(np.abs(u - u[:1])) < 1e-12
    assert np.max(np.abs(res.solver.v)) < 1e-12 and np.max(np.abs(res.solver.p)) < 1e-10
    # the first cell has been dragged along by the wall and the far field is still at rest
    assert 0.0 < u[0, 0] < 1.0 and abs(u[0, -1]) < 1e-12
    assert len(read_history(tmp_path / "history.csv")) == 2
    assert os.path.isfile(tmp_path / "velocity.00003.vtk")


# ------------------------------------------------------------------ the comparison
@pytest.mark.slow
def test_full_runs_match_the_exact_solution(tmp_path):
    """Five periods of each input, eight phases of the last period against eq. (1).
    Measured: 1.92e-2, 5.48e-3 and 1.45e-3 U for Ny = 32, 64, 128, i.e. second order. The
    error is largest at omega t = 90 or 270 degrees, where the wall is at rest and its
    acceleration is largest, not at the shear peaks (135, 315). With a wall velocity that
    changes in time and implicit viscosity, this test also guards the time levels of the
    boundary data in the MAC Helmholtz solves."""
    an = load("analyze")
    worst = {}
    for input_name in INPUTS:
        out = tmp_path / input_name
        res = run(CASE, input_file=input_name, quiet=True, output_dir=str(out))
        assert res.steps == 2000 and len(res.history) == 41
        assert np.all(res.history.cfl < 0.5)
        rows = an.last_period_profiles(CASE, input_name, str(out))
        table = an.compare(rows)
        assert len(rows) == 8 and rows[0]["Ny"] == NY[input_name]
        assert max(r["x_spread"] for r in rows) < 1e-12 and max(r["v_max"] for r in rows) < 1e-12
        worst[input_name] = max(r["err"] for r in table)
        peak = max(table, key=lambda r: r["err"])
        assert round(peak["phase_deg"]) % 360 in (90, 270)
        by_phase = {round(r["phase_deg"]) % 360: r["err"] for r in table}
        assert max(by_phase[90], by_phase[270]) > 1.1 * max(by_phase[135], by_phase[315])
        # the finite-depth solution is indistinguishable at Ly = 10 d
        assert max(abs(r["err"] - r["err_finite_depth"]) for r in table) < 1e-4
        print(f"\n{input_name}: {res.wall_time:.2f} s, max error {worst[input_name]:.3e} U")
    assert worst["input2d_ny32"] < 2.5e-2
    assert worst["input2d"] < 7e-3
    assert worst["input2d_ny128"] < 2e-3
    assert worst["input2d_ny32"] / worst["input2d"] > 3.0          # second order in dy
    assert worst["input2d"] / worst["input2d_ny128"] > 3.0
