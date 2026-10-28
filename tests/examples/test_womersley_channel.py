"""examples/womersley_channel: the exact solution (it satisfies the momentum equation and
its limits), the committed inputs against generate.py, the hooks, the time level at which
the pressure is compared, smoke runs of the five inputs, and (slow) the full runs compared
with the exact profile at eight phases of the last period, plus the pressure's time level
in pm1996 runs of input2d."""

from __future__ import annotations

import dataclasses
import importlib.util
import os

import numpy as np
import pytest
from scipy.integrate import quad

from pyibm.driver import Simulation, run
from pyibm.grid import Grid
from pyibm.inputfile import read_input_file
from pyibm.io.history import read_history
from pyibm.structures import load_hooks

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CASE = os.path.join(ROOT, "examples", "womersley_channel")
INPUTS = ("input2d", "input2d_alpha10", "input2d_incremental", "input2d_periodic",
          "input2d_alpha10_periodic")
ALPHA = {"input2d": 3.0, "input2d_alpha10": 10.0, "input2d_incremental": 3.0,
         "input2d_periodic": 3.0, "input2d_alpha10_periodic": 10.0}
PERIODIC = {"input2d": False, "input2d_alpha10": False, "input2d_incremental": False,
            "input2d_periodic": True, "input2d_alpha10_periodic": True}


def load(name: str):
    """Import <CASE>/<name>.py under a private module name (the heaving-plate pattern)."""
    spec = importlib.util.spec_from_file_location(f"womersley_channel_{name}",
                                                  os.path.join(CASE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ the exact solution
def test_exact_solution_satisfies_the_momentum_equation():
    ex = load("womersley_exact")
    assert ex.womersley_number(0.6) == pytest.approx(3.0) and ex.womersley_number(2.0) == pytest.approx(10.0)
    assert ex.stokes_layer() == pytest.approx(np.sqrt(2 * ex.NU / ex.OMEGA))
    for H in ex.HEIGHTS.values():
        y = np.linspace(0.0, H, 401)
        for t in (0.0, 0.137, 0.5, 0.81):
            # u_t = (K/rho) cos(omega t) + nu u_yy with the derivatives by central
            # differences, so the check does not reuse the algebra of eq. (1). The
            # residual is the truncation error of the differences (6e-8 to 8e-8 at the
            # default steps of 1e-4) and falls by a hundred when both steps fall by ten
            # (measured 98.2 to 100.0; a wrong formula would give a ratio of 1).
            assert ex.pde_residual(y[1:-1], t, H) < 1e-6
            coarse = ex.pde_residual(y[1:-1], t, H, eps_t=1e-3, eps_y=1e-3)
            fine = ex.pde_residual(y[1:-1], t, H, eps_t=1e-4, eps_y=1e-4)
            assert 80.0 < coarse / fine < 125.0
            # no slip on both walls
            assert abs(ex.velocity(0.0, t, H)) < 1e-12 and abs(ex.velocity(H, t, H)) < 1e-12
            # eq. (3) is the integral of eq. (1): adaptive quadrature of the profile
            # (the trapezoidal rule on 401 points is only good to 4e-5, the size of its
            # own truncation error, which is not a test of the formula)
            q, q_err = quad(lambda yy: ex.velocity(yy, t, H), 0.0, H, epsabs=1e-13, epsrel=1e-13)
            assert q_err < 1e-10
            assert q == pytest.approx(ex.flow_rate(t, H), abs=1e-10 * H * ex.U_CORE)
    # the height a boundary hook reads off the coordinates it is handed
    g = Grid(1.2, 0.6, 48, 24, periodic_x=False, periodic_y=False)
    assert ex.channel_height(g.yc) == pytest.approx(0.6) and ex.channel_height(g.yv) == pytest.approx(0.6)


def test_exact_solution_limits():
    """alpha -> 0: the quasi-steady parabola in phase with the pressure gradient;
    alpha -> infinity: a plug K/(rho omega) sin(omega t) lagging it by 90 degrees."""
    ex = load("womersley_exact")
    H = 0.02                                                    # alpha = 0.1
    y = np.linspace(0.0, H, 101)
    for t in (0.0, 0.2, 0.7):
        parabola = ex.K / (2 * ex.RHO * ex.NU) * y * (H - y) * np.cos(ex.OMEGA * t)
        assert np.max(np.abs(ex.velocity(y, t, H) - parabola)) < 1e-2 * ex.K * H**2 / (8 * ex.RHO * ex.NU)
    H = 20.0                                                    # alpha = 100
    for t in (0.0, 0.2, 0.7):
        assert ex.velocity(0.5 * H, t, H) == pytest.approx(ex.U_CORE * np.sin(ex.OMEGA * t), abs=1e-6)
    # the pressure gradient amplitude and the core amplitude are tied by K = rho omega U_core
    assert ex.pressure_gradient(0.0) == pytest.approx(ex.RHO * ex.OMEGA * ex.U_CORE)


# ------------------------------------------------------------------ files and hooks
def test_committed_files_match_generate(tmp_path):
    gen = load("generate")
    ex = load("womersley_exact")
    assert gen.DT == ex.PERIOD / 400 and gen.PRINT_DUMP == 50 and gen.H_CELL == 0.025
    assert [v[0] for v in gen.VARIANTS] == list(INPUTS)
    written = gen.main(str(tmp_path))
    assert [os.path.basename(p) for p in written] == list(INPUTS)
    for path in written:   # byte for byte, header included
        with open(path, "rb") as a, open(os.path.join(CASE, os.path.basename(path)), "rb") as b:
            assert a.read() == b.read(), os.path.basename(path)
    for fname, case, Lx, periodic, periods, out, projection in gen.VARIANTS:
        p = read_input_file(os.path.join(CASE, fname))
        H = ex.HEIGHTS[case]
        assert p.grid.Ly == H and p.grid.Lx == Lx
        assert p.grid.dx == pytest.approx(gen.H_CELL) and p.grid.dy == pytest.approx(gen.H_CELL)
        assert (p.fluid.rho, p.fluid.mu) == (ex.RHO, ex.RHO * ex.NU)
        assert ex.womersley_number(p.grid.Ly, p.fluid.mu / p.fluid.rho) == pytest.approx(ALPHA[fname])
        assert p.time.dt == gen.DT and p.time.n_steps == 400 * periods
        assert (p.solver.backend, p.solver.time_scheme) == ("mac", "midpoint")
        assert p.solver.mac_projection == ("incremental" if fname == "input2d_incremental"
                                           else "nonincremental")
        assert periodic == PERIODIC[fname]
        if periodic:
            assert (p.bc.left, p.bc.right) == ("periodic", "periodic") and p.grid.Nx == 8
        else:
            assert (p.bc.left, p.bc.right) == ("dirichlet", "outflow")
        assert (p.bc.bottom, p.bc.top) == ("dirichlet", "dirichlet")
        assert p.structure.name is None                               # fluid only
        assert p.output.print_dump == gen.PRINT_DUMP and p.output.output_dir == out
        assert p == gen.parameters(case, Lx, periodic, periods, out, projection)


def test_hooks_prescribe_the_exact_profile_and_the_body_force():
    ex = load("womersley_exact")
    hooks = load_hooks(CASE)
    assert hooks.bc_left is not None and hooks.external_force is not None
    assert hooks.bc_right is None and hooks.bc_bottom is None and hooks.bc_top is None
    for H, Nx, Ny in ((0.6, 48, 24), (2.0, 80, 80)):
        g = Grid(1.2, H, Nx, Ny, periodic_x=False, periodic_y=False)
        for t in (0.0, 0.3, 0.75, 2.2):
            for y in (g.yc, g.yv):
                u, v = hooks.bc_left(y, t)
                assert u.shape == y.shape and np.all(v == 0.0)
                assert np.allclose(u, ex.velocity(y, t, H), atol=1e-14)
            Fu, Fv = hooks.external_force(t, g, np.zeros(g.shape_u()), np.zeros(g.shape_v()))
            assert Fu.shape == g.shape_u() and Fv.shape == g.shape_v()
            assert np.all(Fu == 0.0) and np.all(Fv == 0.0)             # the inflow drives
        gp = Grid(0.2, H, 8, Ny, periodic_x=True, periodic_y=False)
        for t in (0.0, 0.3, 0.75):
            Fu, Fv = hooks.external_force(t, gp, np.zeros(gp.shape_u()), np.zeros(gp.shape_v()))
            assert Fu.shape == gp.shape_u() and Fv.shape == gp.shape_v()
            assert np.allclose(Fu, ex.K * np.cos(ex.OMEGA * t)) and np.all(Fv == 0.0)


# ------------------------------------------------------------------ the pressure's time level
def test_pressure_is_compared_at_its_own_time_level():
    """The pressure written at the output time t is the projection pressure of the step that
    ends at t, which belongs to the middle of the step, t - dt/2, for both MAC time schemes.
    compare() must hold -dp/dx against K cos(omega (t - dt/2)): a row whose pressure gradient
    is exactly that must come out with no error. Comparing at t instead would add up to
    K sin(omega dt/2) = 0.049."""
    an = load("analyze")
    ex = load("womersley_exact")
    dt = 0.0025
    assert an.pressure_time(1.0, dt, "midpoint") == 1.0 - 0.5 * dt
    assert an.pressure_time(1.0, dt, "pm1996") == 1.0 - 0.5 * dt
    with pytest.raises(ValueError, match="time_scheme"):
        an.pressure_time(1.0, dt, "rk4")
    H = ex.HEIGHTS["alpha3"]
    y = (np.arange(24) + 0.5) * (H / 24)
    rows = []
    for t in (3.125, 3.25, 3.375):          # omega t = 45, 90, 135 degrees
        t_p = an.pressure_time(t, dt, "midpoint")
        rows.append({"t": t, "y": y, "dy": H / 24, "H": H, "dt": dt, "t_pressure": t_p,
                     "periodic": False, "u": ex.velocity(y, t, H), "v_mid": 0.0,
                     "u_spread_mid": 0.0, "dpdx": -float(ex.pressure_gradient(t_p))})
    table = an.compare(rows)
    for r in table:
        assert abs(r["dpdx"] - r["dpdx_exact"]) < 1e-12
        assert r["err"] < 1e-12
    # at omega t = 90 degrees the forcing at t is zero, at t - dt/2 it is K sin(omega dt/2)
    assert table[1]["dpdx"] == pytest.approx(ex.K * np.sin(0.5 * ex.OMEGA * dt), rel=1e-12)


def test_smoke_rows_carry_the_pressure_time(tmp_path):
    an = load("analyze")
    p = read_input_file(os.path.join(CASE, "input2d"))
    run(CASE, input_file="input2d", steps=3, quiet=True, output_dir=str(tmp_path))
    rows = an.last_period_profiles(CASE, "input2d", str(tmp_path))
    assert [r["t"] for r in rows] == pytest.approx([0.0, 3 * p.time.dt])
    for r in rows:
        assert r["dt"] == p.time.dt and r["t_pressure"] == pytest.approx(r["t"] - 0.5 * p.time.dt)


# ------------------------------------------------------------------ smoke runs
@pytest.mark.parametrize("input_name", INPUTS)
def test_smoke_three_steps(tmp_path, input_name):
    res = run(CASE, input_file=input_name, steps=3, quiet=True, output_dir=str(tmp_path))
    assert res.steps == 3 and res.history.step.tolist() == [0.0, 3.0]    # print_dump = 50
    assert np.all(np.isfinite(res.history.view(float)))
    assert res.structure is None
    assert res.history.max_target_error_over_dx[-1] == 0.0 and res.history.fx_body[-1] == 0.0
    assert res.history.max_speed[-1] > 0.0 and res.solver.kinetic_energy() > 0.0
    assert np.max(np.abs(res.solver.divergence())) < 1e-10
    g = res.grid
    if PERIODIC[input_name]:
        # the body force accelerates the whole channel, uniformly in x
        assert res.solver.mean_velocity()[0] > 0.0
        assert np.max(np.abs(res.solver.u - res.solver.u[:1])) < 1e-12
    else:
        # what flows in on the left flows out on the right
        assert abs(res.solver.u[0].sum() - res.solver.u[-1].sum()) * g.dy < 1e-10
        assert np.max(np.abs(res.solver.u[0])) > 0.0
    assert len(read_history(tmp_path / "history.csv")) == 2
    assert os.path.isfile(tmp_path / "velocity.00003.vtk") and os.path.isfile(tmp_path / "pressure.00003.vtk")


# ------------------------------------------------------------------ the comparison
def _pressure_fit(rows, dt):
    """Least-squares fit of the computed -dp/dx to A cos(omega (t - lag)) over the rows, with
    t the output time. Returns A, lag/dt and the largest |-dp/dx - K cos(omega (t - s))| for
    s = 0, dt/2 and dt, keyed by s/dt."""
    ex = load("womersley_exact")
    t = np.array([r["t"] for r in rows])
    dp = -np.array([r["dpdx"] for r in rows])
    a_cos, a_sin = np.linalg.lstsq(np.column_stack([np.cos(ex.OMEGA * t), np.sin(ex.OMEGA * t)]),
                                   dp, rcond=None)[0]
    misfit = {shift: float(np.max(np.abs(dp - ex.K * np.cos(ex.OMEGA * (t - shift * dt)))))
              for shift in (0.0, 0.5, 1.0)}
    return float(np.hypot(a_cos, a_sin)), float(np.arctan2(a_sin, a_cos) / ex.OMEGA / dt), misfit


def _full_run(tmp_path, input_name):
    an = load("analyze")
    out = tmp_path / input_name
    res = run(CASE, input_file=input_name, quiet=True, output_dir=str(out))
    p = res.parameters
    assert res.steps == p.time.n_steps and np.all(res.history.cfl < 0.5)
    rows = an.last_period_profiles(CASE, input_name, str(out))
    table = an.compare(rows)
    assert len(rows) == 8 and rows[0]["periodic"] == PERIODIC[input_name]
    worst = max(r["err"] for r in table)
    worst_int = max(r["err_interior"] for r in table)
    print(f"\n{input_name}: {res.wall_time:.1f} s, max error {worst:.3e}, interior {worst_int:.3e}")
    return res, rows, table, worst, worst_int


@pytest.mark.slow
@pytest.mark.parametrize("input_name", ("input2d", "input2d_alpha10"))
def test_inflow_outflow_runs_match_the_exact_profile(tmp_path, input_name):
    """The exact profile prescribed at the inlet, outflow at the right. The profile at
    x = Lx/2 agrees with eq. (1) to 1.34e-2 (alpha 3) and 1.38e-2 (alpha 10) of the core
    amplitude at the wall cells and 6.7e-3 and 8.4e-3 in the interior (measured; see the
    README): the wall part is the O(dt) tangential slip of the non-incremental projection,
    dt K/rho = 0.016 here, on top of the spatial error of 5.7 cells per Stokes layer. The
    pressure gradient between Lx/4 and 3 Lx/4 follows K cos(omega (t - dt/2)), the forcing
    at the pressure's own time level, to 0.50 % (alpha 3) and 0.14 % (alpha 10) of K. At
    alpha 3 a fit of -dp/dx to A cos(omega (t - lag)) gives A = 6.253 and a lag of 0.64 dt,
    and t - dt/2 matches better than t or t - dt."""
    ex = load("womersley_exact")
    res, rows, table, worst, worst_int = _full_run(tmp_path, input_name)
    assert worst < 2e-2
    assert worst_int < 1e-2
    assert max(abs(r["flux_err"]) for r in table) < 2e-3
    assert max(abs(r["dpdx"] - r["dpdx_exact"]) for r in table) < 1e-2 * ex.K
    if input_name == "input2d":
        amp, lag, misfit = _pressure_fit(rows, res.parameters.time.dt)
        assert amp == pytest.approx(6.253, abs=2e-3) and lag == pytest.approx(0.644, abs=0.01)
        assert misfit[0.5] < min(misfit[0.0], misfit[1.0])
    # one-dimensional away from the inlet and the outlet
    assert max(r["v_mid"] for r in rows) < 2e-3 and max(r["u_spread_mid"] for r in rows) < 2e-3


@pytest.mark.slow
def test_incremental_projection_removes_the_wall_slip(tmp_path):
    """input2d with mac_projection = "incremental": the wall slip is gone, so the profile
    error drops to the level of the periodic body-force runs (measured 6.7e-3 at the wall
    cells, 3.4e-3 inside, against 1.34e-2 and 6.7e-3 for input2d), and -dp/dx follows
    K cos(omega (t - dt/2)) to 0.19 % of K (0.50 % with the default projection). The
    predictor carries the previous pressure here, and the stored pressure still belongs to
    the middle of the step: the fitted lag is 0.56 dt."""
    ex = load("womersley_exact")
    res, rows, table, worst, worst_int = _full_run(tmp_path, "input2d_incremental")
    assert res.parameters.solver.mac_projection == "incremental"
    assert worst < 1e-2
    assert worst_int < 5e-3
    assert max(abs(r["flux_err"]) for r in table) < 2e-3
    assert max(abs(r["dpdx"] - r["dpdx_exact"]) for r in table) < 5e-3 * ex.K
    assert max(r["v_mid"] for r in rows) < 1e-3 and max(r["u_spread_mid"] for r in rows) < 1e-3
    amp, lag, misfit = _pressure_fit(rows, res.parameters.time.dt)
    assert lag == pytest.approx(0.559, abs=0.01)
    assert misfit[0.5] < min(misfit[0.0], misfit[1.0])


# measured fitted lag of -dp/dx in units of dt, input2d run with time_scheme = "pm1996"
PM1996_LAG = {("nonincremental", 1): 0.683, ("nonincremental", 2): 0.740,
              ("incremental", 1): 0.513, ("incremental", 2): 0.566}


@pytest.mark.slow
@pytest.mark.parametrize("projection, dt_factor", sorted(PM1996_LAG))
def test_pm1996_pressure_is_nearest_the_middle_of_the_step(tmp_path, projection, dt_factor):
    """input2d with time_scheme = "pm1996" (no committed input uses it) at dt = T/400 and
    T/800: the fitted lag of -dp/dx is 0.68 dt and 0.74 dt with the default projection and
    0.51 dt and 0.57 dt with the incremental one. That is not a clean half step, but t - dt/2
    matches better than t or t - dt in all four runs, so pressure_time uses it for pm1996
    too."""
    an = load("analyze")
    p = read_input_file(os.path.join(CASE, "input2d"))
    p = dataclasses.replace(
        p, solver=dataclasses.replace(p.solver, time_scheme="pm1996", mac_projection=projection),
        time=dataclasses.replace(p.time, dt=p.time.dt / dt_factor),
        output=dataclasses.replace(p.output, print_dump=p.output.print_dump * dt_factor))
    out = tmp_path / "pm1996"
    Simulation(p, CASE, output_dir=str(out), quiet=True).run()
    rows = an.last_period_profiles(CASE, "input2d", str(out))
    assert len(rows) == 8
    amp, lag, misfit = _pressure_fit(rows, p.time.dt)
    print(f"\npm1996 {projection} dt = T/{400 * dt_factor}: lag {lag:.3f} dt, misfit at "
          f"t, t - dt/2, t - dt: {misfit[0.0]:.4f} {misfit[0.5]:.4f} {misfit[1.0]:.4f}")
    assert lag == pytest.approx(PM1996_LAG[projection, dt_factor], abs=0.01)
    assert misfit[0.5] < min(misfit[0.0], misfit[1.0])


@pytest.mark.slow
@pytest.mark.parametrize("input_name", ("input2d_periodic", "input2d_alpha10_periodic"))
def test_periodic_body_force_runs_match_the_exact_profile(tmp_path, input_name):
    """The same flow driven by the body force in a periodic channel: no inflow, no
    projection slip, so this is the discretisation error alone (measured 6.9e-3 at the
    wall cell at 5.7 cells per Stokes layer and 4.2e-3 in the interior, for both alpha)."""
    res, rows, table, worst, worst_int = _full_run(tmp_path, input_name)
    assert worst < 1e-2
    assert worst_int < 5e-3
    assert max(abs(r["flux_err"]) for r in table) < 5e-3
    assert max(r["v_mid"] for r in rows) < 1e-12 and max(r["u_spread_mid"] for r in rows) < 1e-12
