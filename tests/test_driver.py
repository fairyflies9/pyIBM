"""Two-step runs of tiny cases through the driver (pyibm.driver, docs/user-guide.md).

Every back end and coupling (periodic + delta, periodic + blob, MAC + delta with Dirichlet
walls, MAC with periodic x and Dirichlet y) with both time schemes, mass points under
gravity, the hooks (external_force, lagrangian_force, initial_velocity, bc_top, on_output, a
hooks file named in the input file), the hook time levels, the time level of the history
force, the blob membrane pressure, the output files (fresh, continued, cleaned, guarded), the
checks (parameters, non-finite state, wall distance), the warnings and the command line.
"""

from __future__ import annotations

import os
import subprocess
import sys
import warnings

import numpy as np
import pytest

from pyibm import __version__
from pyibm.driver import HISTORY_COLUMNS, TETHER_STABLE, RunResult, Simulation, run
from pyibm.inputfile import (BCParams, FluidParams, GridParams, HooksParams, OutputParams,
                             Parameters, SolverParams, StructureParams, TimeParams,
                             read_input_file, write_input_file)
from pyibm.io.history import HistoryWriter, read_history
from pyibm.io.vtk import (read_lagrangian_file, read_structured_scalar,
                          read_structured_vector)
from pyibm.structures import Hooks, Masses, Springs, Structure, Targets, write_structure

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
DT = 1e-3
NX = 16
PERIODIC = ("periodic",) * 4
CLOSED = ("dirichlet",) * 4
CHANNEL = ("periodic", "periodic", "dirichlet", "dirichlet")


# ------------------------------------------------------------------ case builders
def ring_points(n: int = 24, cx: float = 0.5, cy: float = 0.5, r: float = 0.2) -> np.ndarray:
    th = 2.0 * np.pi * np.arange(n) / n
    return np.column_stack([cx + r * np.cos(th), cy + r * np.sin(th)])


def ring_structure(n: int = 24, k: float = 1.0, rest_factor: float = 0.8, targets: bool = False,
                   masses: bool = False, gravity=(0.0, 0.0)) -> Structure:
    """A closed ring of springs; rest lengths shorter than the initial spacing so it contracts
    and drives the fluid from the first step. Optional target or mass points on every node."""
    X = ring_points(n)
    i = np.arange(n)
    j = (i + 1) % n
    L = np.hypot(*(X[j] - X[i]).T)
    return Structure(
        name="ring", X=X, ds=np.full(n, float(L.mean())),
        springs=Springs(i, j, np.full(n, k), rest_factor * L),
        targets=Targets(i, np.full(n, k), X) if targets else None,
        masses=Masses(i, np.full(n, k), np.full(n, 1.0), X) if masses else None,
        gravity=gravity)


def make_params(backend="periodic_fft", scheme="midpoint", coupling="delta", bc=PERIODIC,
                name="ring", dt=DT, nx=NX, rho=1.0, mu=0.01, print_dump=100, plot=False,
                gravity=(0.0, 0.0), tfinal_steps=2, **solver_kw) -> Parameters:
    solver = SolverParams(backend=backend, time_scheme=scheme, coupling=coupling, **solver_kw)
    if coupling == "blob" and "blob" not in solver_kw:
        solver.blob = "compact"
    return Parameters(
        fluid=FluidParams(rho=rho, mu=mu),
        time=TimeParams(Tfinal=tfinal_steps * dt, dt=dt),
        grid=GridParams(Nx=nx, Ny=nx, Lx=1.0, Ly=1.0),
        solver=solver,
        bc=BCParams(*bc),
        structure=StructureParams(name=name, ds=None, gravity_x=gravity[0], gravity_y=gravity[1]),
        output=OutputParams(print_dump=print_dump, save_vtk=True, save_history=True, plot=plot,
                            output_dir="viz"))


def write_case(tmp_path, params: Parameters, structure: Structure | None = None,
               hooks_src: str | None = None, input_name: str = "input2d") -> str:
    case = str(tmp_path)
    os.makedirs(case, exist_ok=True)
    write_input_file(os.path.join(case, input_name), params)
    if structure is not None:
        write_structure(case, structure)
    if hooks_src is not None:
        with open(os.path.join(case, "hooks.py"), "w", encoding="utf-8") as fh:
            fh.write(hooks_src)
    return case


# ------------------------------------------------------------------ back ends and couplings
@pytest.mark.parametrize("scheme", ["midpoint", "pm1996"])
@pytest.mark.parametrize("backend, coupling, bc", [
    ("periodic_fft", "delta", PERIODIC),
    ("periodic_fft", "blob", PERIODIC),
    ("mac", "delta", CLOSED),
    ("mac", "delta", CHANNEL),
])
def test_two_steps_each_backend_and_coupling(tmp_path, backend, coupling, bc, scheme):
    # k = 1e3: the inward pull of a contracting ring is mostly a gradient, which the
    # projection hands to the pressure, so a soft ring barely moves in two steps
    s = ring_structure(k=1e3)
    params = make_params(backend, scheme, coupling, bc)
    case = write_case(tmp_path, params, s)
    res = run(case, steps=2, quiet=True)

    assert isinstance(res, RunResult)
    assert res.steps == 2 and res.t == pytest.approx(2 * DT)
    assert res.wall_time >= 0.0
    assert res.history.dtype.names == HISTORY_COLUMNS
    assert res.history.step.tolist() == [0.0, 2.0]              # step 0 and the last step
    assert res.history.t.tolist() == pytest.approx([0.0, 2 * DT])
    assert np.all(np.isfinite(res.history.view(float)))
    assert res.grid.periodic_x == (bc[0] == "periodic") and res.grid.periodic_y == (bc[2] == "periodic")
    assert res.solver.layout_u.shape == res.simulation.Fu.shape
    assert res.solver.layout_v.shape == res.simulation.Fv.shape

    # the contracting ring drives the fluid and moves
    assert res.solver.kinetic_energy() > 0.0
    assert res.history.kinetic_energy[-1] > 0.0 and res.history.max_speed[-1] > 0.0
    assert np.max(np.abs(res.structure.X - ring_points())) > 1e-9
    assert np.all(np.isfinite(res.simulation.f))
    # a closed ring of springs puts no net force on the fluid: both couplings conserve the
    # total, so the body force is round-off relative to the force scale
    fscale = float(np.sum(np.abs(res.simulation.f) * s.ds[:, None]))
    assert abs(res.history.fx_body[-1]) < 1e-12 * fscale
    assert abs(res.history.fy_body[-1]) < 1e-12 * fscale
    assert res.history.max_target_error_over_dx[-1] == 0.0        # no targets

    # files
    out = os.path.join(case, "viz")
    assert os.path.normpath(res.output_dir) == os.path.normpath(out)
    h = read_history(os.path.join(out, "history.csv"))
    assert h.dtype.names == HISTORY_COLUMNS and len(h) == 2
    assert np.allclose(h.kinetic_energy, res.history.kinetic_energy)
    for step in (0, 2):
        for field in ("velocity", "vorticity", "pressure", "lag"):
            assert os.path.isfile(os.path.join(out, f"{field}.{step:05d}.vtk"))
    assert not os.path.isfile(os.path.join(out, "vorticity.00001.vtk"))
    vort = read_structured_scalar(os.path.join(out, "vorticity.00002.vtk"))
    vel = read_structured_vector(os.path.join(out, "velocity.00002.vtk"))
    h_ = 1.0 / NX
    origin = (0.0, 0.0) if backend == "periodic_fft" else (0.5 * h_, 0.5 * h_)
    assert vort.origin == pytest.approx(origin) and vel.origin == pytest.approx(origin)
    assert vort.spacing == pytest.approx((h_, h_))
    assert vort.field.shape == (NX, NX) and vel.fx.shape == (NX, NX)
    assert np.allclose(vort.field, res.solver.vorticity())


def test_steps_default_to_tfinal_over_dt(tmp_path):
    params = make_params(tfinal_steps=3, print_dump=1)
    case = write_case(tmp_path, params, ring_structure())
    res = run(case, quiet=True)
    assert res.steps == 3 and res.history.step.tolist() == [0.0, 1.0, 2.0, 3.0]


def test_output_dir_override_and_fresh_history(tmp_path):
    params = make_params(print_dump=1)
    case = write_case(tmp_path, params, ring_structure())
    out = tmp_path / "elsewhere"
    res = run(case, steps=1, quiet=True, output_dir=str(out))
    assert os.path.normpath(res.output_dir) == os.path.normpath(str(out))
    assert len(read_history(out / "history.csv")) == 2
    run(case, steps=2, quiet=True, output_dir=str(out))              # rerun overwrites
    assert len(read_history(out / "history.csv")) == 3
    assert not os.path.exists(tmp_path / "viz")


# ------------------------------------------------------------------ masses and gravity
@pytest.mark.parametrize("scheme", ["midpoint", "pm1996"])
def test_masses_fall_under_gravity(tmp_path, scheme):
    g = -1.0
    s = ring_structure(k=10.0, rest_factor=1.0, masses=True, gravity=(0.0, g))
    params = make_params(scheme=scheme, gravity=(0.0, g))
    case = write_case(tmp_path, params, s)
    res = run(case, steps=2, quiet=True)
    m = res.structure.masses
    assert np.all(m.V[:, 1] < 0.0) and np.all(m.Y[:, 1] < ring_points()[:, 1])
    assert np.allclose(m.V[:, 0], 0.0, atol=1e-12)
    # the tether force is tiny after two steps, so the ghosts are nearly in free fall
    assert m.V[:, 1] == pytest.approx(2 * DT * g, rel=1e-3)
    # symplectic Euler: Y_2 = (1 + 2) dt^2 g; midpoint: Y_2 = (1/2 + 3/2) dt^2 g
    assert m.Y[:, 1] - ring_points()[:, 1] == pytest.approx(
        (3.0 if scheme == "pm1996" else 2.0) * DT**2 * g, rel=1e-3)
    # the ghosts pull the fluid points down: the fluid holds the ring up (fy_body > 0) and
    # the box as a whole is pushed down (no remove_mean_force on the periodic box)
    assert res.history.fy_body[-1] > 0.0
    assert res.solver.mean_velocity()[1] < 0.0
    assert np.all(res.structure.X[:, 1] < ring_points()[:, 1] + 1e-15)


# ------------------------------------------------------------------ hooks
@pytest.mark.parametrize("backend, bc", [("periodic_fft", PERIODIC), ("mac", PERIODIC)])
def test_external_force_hook_accelerates_the_box(tmp_path, backend, bc):
    F, rho = 3.0, 2.0
    hooks = ("import numpy as np\n"
             f"F = {F}\n"
             "def external_force(t, grid, u, v):\n"
             "    return F * np.ones_like(u), np.zeros_like(v)\n")
    params = make_params(backend=backend, bc=bc, name=None, rho=rho)
    case = write_case(tmp_path, params, hooks_src=hooks)
    res = run(case, steps=2, quiet=True)
    assert res.structure is None
    assert res.solver.mean_velocity()[0] == pytest.approx(2 * DT * F / rho, rel=1e-12)
    assert res.solver.mean_velocity()[1] == pytest.approx(0.0, abs=1e-14)
    # the external force is not part of the force on the body
    assert res.history.fx_body[-1] == 0.0 and res.history.fy_body[-1] == 0.0
    assert res.history.mean_u[-1] == pytest.approx(2 * DT * F / rho, rel=1e-12)


def test_external_force_shape_is_checked(tmp_path):
    hooks = ("import numpy as np\n"
             "def external_force(t, grid, u, v):\n"
             "    return np.ones(3), np.ones(3)\n")
    case = write_case(tmp_path, make_params(name=None), hooks_src=hooks)
    with pytest.raises(ValueError, match="external_force"):
        run(case, steps=1, quiet=True)


def test_fluid_only_lid_driven_box(tmp_path):
    hooks = ("import numpy as np\n"
             "calls = []\n"
             "def bc_top(x, t):\n"
             "    return np.ones_like(x), np.zeros_like(x)\n"
             "def on_output(step, t, state):\n"
             "    calls.append((step, t, float(state.solver.kinetic_energy())))\n")
    params = make_params(backend="mac", bc=CLOSED, name=None, print_dump=1)
    case = write_case(tmp_path, params, hooks_src=hooks)
    res = run(case, steps=2, quiet=True)
    assert res.structure is None
    assert res.solver.kinetic_energy() > 0.0 and res.history.max_speed[-1] > 0.0
    assert res.history.max_target_error_over_dx.tolist() == [0.0, 0.0, 0.0]
    calls = res.simulation.hooks.module.calls
    assert [c[0] for c in calls] == [0, 1, 2]
    assert [c[1] for c in calls] == pytest.approx([0.0, DT, 2 * DT])
    assert calls[0][2] == 0.0 and calls[2][2] > calls[1][2] > 0.0
    # the lid moves the fluid near the top only
    assert np.max(res.solver.u[:, -1]) > np.max(np.abs(res.solver.u[:, 0]))
    assert np.max(np.abs(res.solver.divergence())) < 1e-12


@pytest.mark.parametrize("scheme, expected", [
    ("midpoint", [0.0, 0.0, 0.5 * DT, 1.5 * DT, 2 * DT]),
    ("pm1996", [0.0, 0.0, 0.0, DT, 2 * DT]),
])
def test_update_hook_time_levels(tmp_path, scheme, expected):
    """The forces are evaluated at t + dt/2 (midpoint) or t (pm1996); the targets are also
    refreshed at the output time (step 0: initial force, then the diagnostic; step 2: the
    diagnostic) so the target error compares positions at the same instant."""
    hooks = ("times = []\n"
             "def update_targets(t, dt, s):\n"
             "    times.append(t)\n")
    s = ring_structure(rest_factor=1.0, targets=True)
    params = make_params(scheme=scheme)
    case = write_case(tmp_path, params, s, hooks_src=hooks)
    res = run(case, steps=2, quiet=True)
    assert res.simulation.hooks.module.times == pytest.approx(expected)


def test_prescribed_targets_drag_the_ring(tmp_path):
    hooks = ("import numpy as np\n"
             "def update_targets(t, dt, s):\n"
             "    s.targets.X[:, 0] = s.X0[s.targets.idx, 0] + 0.5 * t\n")
    s = ring_structure(k=100.0, rest_factor=1.0, targets=True)
    params = make_params(print_dump=1)
    case = write_case(tmp_path, params, s, hooks_src=hooks)
    res = run(case, steps=2, quiet=True)
    err = res.history.max_target_error_over_dx
    assert err[0] == 0.0 and err[1] > 0.0 and err[2] > 0.0
    assert err[2] == pytest.approx(0.5 * 2 * DT / res.grid.dx, rel=0.2)   # the ring lags
    assert res.history.fx_body[-1] < 0.0        # the fluid resists the motion in +x


# ------------------------------------------------------------------ plot and warnings
def test_plot_writes_snapshots(tmp_path):
    params = make_params(plot=True, print_dump=1)
    case = write_case(tmp_path, params, ring_structure())
    res = run(case, steps=1, quiet=True)
    for step in (0, 1):
        assert os.path.getsize(os.path.join(res.output_dir, f"snapshot.{step:05d}.png")) > 0


def test_tether_number_warning(tmp_path):
    s = ring_structure(k=1e6, rest_factor=1.0, targets=True)   # k dt^2/(rho dx) = 16
    params = make_params()
    case = write_case(tmp_path, params, s)
    with pytest.warns(RuntimeWarning, match="tether number") as caught:
        run(case, steps=1, quiet=True)
    assert "rule of thumb" in str(caught[0].message)


@pytest.mark.parametrize("scheme", ["midpoint", "pm1996"])
@pytest.mark.parametrize("element", ["target", "mass tether"])
def test_tether_rule_of_thumb_per_scheme_and_element(element, scheme):
    """The warning fires just above the scheme's measured stable tether number for targets
    and mass tethers, not just below it, and never for springs however stiff."""
    limit = TETHER_STABLE[(element, scheme)]
    dx = 1.0 / NX
    for factor, expect in ((0.95, False), (1.05, True)):
        k = factor * limit * 1.0 * dx / DT**2
        s = ring_structure(k=1.0, rest_factor=1.0, targets=(element == "target"),
                           masses=(element == "mass tether"))
        el = s.targets if element == "target" else s.masses
        el.k[:] = k
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Simulation(make_params(scheme=scheme), ".", hooks=Hooks(), structure=s, quiet=True)
        hits = [w for w in caught if "tether number" in str(w.message)]
        assert bool(hits) == expect, (element, scheme, factor)
        if expect:
            assert element in str(hits[0].message) and "rule of thumb" in str(hits[0].message)
    s = ring_structure(k=1e8, rest_factor=1.0)                    # springs only: number 1600
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Simulation(make_params(scheme=scheme), ".", hooks=Hooks(), structure=s, quiet=True)


def test_tether_thresholds_are_read_only():
    """No module-level mutable state: a caller cannot change the thresholds for the process."""
    with pytest.raises(TypeError):
        TETHER_STABLE[("target", "midpoint")] = 1e9
    assert TETHER_STABLE[("target", "midpoint")] == 10.0


def test_cfl_warning_once(tmp_path):
    params = make_params(name=None)
    case = write_case(tmp_path, params)
    sim = Simulation(params, case, quiet=True)
    sim.solver.u[:] = 1e3                                        # cfl = dt u / dx = 16
    with pytest.warns(RuntimeWarning, match="CFL"):
        d = sim.output()
    assert d["cfl"] > 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sim.output()                                             # warned once only
    sim.close()


def test_history_off_and_vtk_off(tmp_path):
    params = make_params(print_dump=1)
    params.output.save_vtk = False
    params.output.save_history = False
    case = write_case(tmp_path, params, ring_structure())
    res = run(case, steps=2, quiet=True)
    assert len(res.history) == 3                                 # still returned in memory
    assert not os.path.exists(os.path.join(case, "viz"))


def test_quiet_false_prints_one_line_per_output(tmp_path, capsys):
    params = make_params(print_dump=1)
    case = write_case(tmp_path, params, ring_structure())
    run(case, steps=2, quiet=False)
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("step ")]
    assert len(lines) == 3 and "err/dx" in lines[0] and "cfl" in lines[-1]


# ------------------------------------------------------------------ command line
def _cli(*args, cwd=ROOT):
    env = dict(os.environ)
    env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run([PY, "-m", "pyibm", *args], capture_output=True, text=True, env=env, cwd=cwd)


def test_cli_version():
    r = _cli("version")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == f"pyibm {__version__}"


def test_cli_run(tmp_path):
    params = make_params(print_dump=1)
    case = write_case(tmp_path, params, ring_structure())
    write_input_file(os.path.join(case, "input2d_alt"), make_params(print_dump=100))

    out = tmp_path / "cli_out"
    r = _cli("run", case, "--steps", "2", "--quiet", "--output-dir", str(out))
    assert r.returncode == 0, r.stderr
    assert r.stdout == ""
    assert len(read_history(out / "history.csv")) == 3

    r = _cli("run", case, "--input", "input2d_alt", "--steps", "3")
    assert r.returncode == 0, r.stderr
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("step ")]
    assert len(lines) == 2 and lines[0].startswith("step       0") and lines[1].startswith("step       3")
    assert "history: 2 rows" in r.stdout
    assert len(read_history(os.path.join(case, "viz", "history.csv"))) == 2

    r = _cli("run", str(tmp_path / "missing"))
    assert r.returncode != 0 and "input2d" in r.stderr


# ------------------------------------------------------------------ hooks wired by the driver
def test_lagrangian_force_hook_reaches_the_fluid(tmp_path):
    """lagrangian_force(t, dt, s, X) adds a force per unit length at the force's positions and
    time level; it is part of the body force (fx_body = -sum f ds)."""
    hooks = ("import numpy as np\n"
             "calls = []\n"
             "def lagrangian_force(t, dt, s, X):\n"
             "    calls.append((t, X.copy()))\n"
             "    f = np.zeros_like(X)\n"
             "    f[:, 0] = 2.0\n"
             "    return f\n")
    s = ring_structure(k=0.0, rest_factor=1.0)
    params = make_params(print_dump=1)
    case = write_case(tmp_path, params, s, hooks_src=hooks)
    res = run(case, steps=2, quiet=True)
    total = 2.0 * float(np.sum(s.ds))
    assert res.history.fx_body == pytest.approx([-total] * 3, rel=1e-12)
    assert res.history.fy_body == pytest.approx([0.0] * 3, abs=1e-12 * total)
    calls = res.simulation.hooks.module.calls
    assert [c[0] for c in calls] == pytest.approx([0.0, 0.5 * DT, 1.5 * DT])
    assert np.array_equal(calls[-1][1], res.simulation.X_prev)        # X_h of the last step

    bad = "import numpy as np\ndef lagrangian_force(t, dt, s, X):\n    return np.zeros(3)\n"
    case2 = write_case(tmp_path / "bad", params, s, hooks_src=bad)
    with pytest.raises(ValueError, match="lagrangian_force"):
        run(case2, steps=1, quiet=True)


@pytest.mark.parametrize("backend, bc", [("periodic_fft", PERIODIC), ("mac", CHANNEL)])
def test_initial_velocity_hook(tmp_path, backend, bc):
    hooks = ("import numpy as np\n"
             "def initial_velocity(grid, layout_u, layout_v):\n"
             "    return np.full(layout_u.shape, 0.25), np.zeros(layout_v.shape)\n")
    params = make_params(backend=backend, bc=bc, name=None, print_dump=1)
    case = write_case(tmp_path, params, hooks_src=hooks)
    sim = Simulation(read_input_file(os.path.join(case, "input2d")), case, quiet=True)
    assert np.all(sim.solver.u == 0.25) and np.all(sim.solver.v == 0.0)
    res = sim.run(1)
    assert res.history.mean_u[0] == pytest.approx(0.25, rel=1e-12)
    assert res.history.max_speed[0] == pytest.approx(0.25, rel=1e-12)

    bad = ("import numpy as np\n"
           "def initial_velocity(grid, lu, lv):\n"
           "    return np.zeros(3), np.zeros(3)\n")
    case2 = write_case(tmp_path / "bad", params, hooks_src=bad)
    with pytest.raises(ValueError, match="initial_velocity"):
        run(case2, steps=1, quiet=True)


def test_hooks_file_named_in_the_input_file(tmp_path):
    params = make_params(name=None)
    params.hooks = HooksParams(file="hooks_alt.py")
    case = write_case(tmp_path, params,
                      hooks_src="def on_output(step, t, state):\n    raise AssertionError\n")
    with open(os.path.join(case, "hooks_alt.py"), "w", encoding="utf-8") as fh:
        fh.write("seen = []\ndef on_output(step, t, state):\n    seen.append(step)\n")
    res = run(case, steps=1, quiet=True)
    assert os.path.basename(res.simulation.hooks.path) == "hooks_alt.py"
    assert res.simulation.hooks.module.seen == [0, 1]
    os.remove(os.path.join(case, "hooks_alt.py"))
    with pytest.raises(FileNotFoundError):
        run(case, steps=1, quiet=True)


def test_solver_keys_reach_the_coupling_and_the_solver(tmp_path):
    case = str(tmp_path)
    sim = Simulation(make_params(coupling="blob", blob_interpolation="bicubic", name=None),
                     case, hooks=Hooks(), quiet=True)
    assert sim.coupling.interpolation == "bicubic"
    sim = Simulation(make_params(coupling="blob", name=None), case, hooks=Hooks(), quiet=True)
    assert sim.coupling.interpolation == "streamfunction"
    sim = Simulation(make_params(backend="mac", bc=CLOSED, mac_projection="incremental",
                                 name=None), case, hooks=Hooks(), quiet=True)
    assert sim.solver.projection == "incremental"
    sim = Simulation(make_params(backend="mac", bc=CLOSED, name=None), case, hooks=Hooks(),
                     quiet=True)
    assert sim.solver.projection == "nonincremental"


# ------------------------------------------------------------------ the time level of the force
@pytest.mark.parametrize("scheme", ["midpoint", "pm1996"])
def test_history_force_carries_its_time_level(tmp_path, scheme):
    """Each row's fx_body is the force the fluid received in the last step; t_force says when
    the scheme evaluated it (t - dt/2 midpoint, t - dt pm1996, 0 in row 0), with the
    prescribed targets at that time. Before, the rows had no such column and pm1996 rows 0
    and 1 silently held the same force."""
    hooks = ("import numpy as np\n"
             "times = []\n"
             "def update_targets(t, dt, s):\n"
             "    times.append(t)\n"
             "    s.targets.X[:, 0] = s.X0[s.targets.idx, 0] + 0.5 * t\n")
    s = ring_structure(k=100.0, rest_factor=1.0, targets=True)
    params = make_params(scheme=scheme, print_dump=1)
    case = write_case(tmp_path, params, s, hooks_src=hooks)
    sim = Simulation(read_input_file(os.path.join(case, "input2d")), case, quiet=True)
    res = sim.run(3)
    h = res.history
    assert h.dtype.names[:3] == ("step", "t", "t_force")
    lag = 0.5 * DT if scheme == "midpoint" else DT
    assert h.t_force[0] == 0.0
    assert h.t_force[1:] == pytest.approx(h.t[1:] - lag, abs=1e-15)
    if scheme == "pm1996":
        assert h.t_force[1] == h.t_force[0] and h.fx_body[1] == h.fx_body[0]
    assert sim.t_force == h.t_force[-1]
    # the force of the last row is the force at t_force: -sum f ds of the recorded f
    assert h.fx_body[-1] == pytest.approx(-float(np.sum(sim.f[:, 0] * s.ds)), rel=1e-12)
    # every row's force time is one of the times the hooks were called with
    times = np.array(sim.hooks.module.times)
    for tf in h.t_force:
        assert np.min(np.abs(times - tf)) < 1e-15
    # the lag file carries the same force, the history file the same column
    lag_file = read_lagrangian_file(os.path.join(res.output_dir, "lag.00003.vtk"))
    assert np.array_equal(lag_file.point_data["force"], sim.f)
    assert np.array_equal(lag_file.X, sim.structure.X)
    on_disk = read_history(os.path.join(res.output_dir, "history.csv"))
    assert np.array_equal(on_disk.t_force, h.t_force)


@pytest.mark.parametrize("coupling", ["delta", "blob"])
def test_output_at_step_zero_carries_the_initial_force(tmp_path, coupling):
    """Row 0 holds the force at the initial positions whether it comes from run() or from
    sim.output() called by hand. Before, only run() evaluated it: sim.output() wrote
    fx_body = 0, Simulation.f stayed None, and a run() after it never reported the force."""
    def structure():
        s = ring_structure(k=100.0, rest_factor=1.0, targets=True)
        s.targets.X = s.X[s.targets.idx] + np.array([0.01, 0.0])   # net pull in +x
        return s

    params = make_params(coupling=coupling, print_dump=1)
    ref = Simulation(params, str(tmp_path / "ref"), hooks=Hooks(), structure=structure(),
                     quiet=True).run(1).history
    assert ref.fx_body[0] < 0.0 and ref.t_force[0] == 0.0       # the fluid pulls back in -x

    sim = Simulation(params, str(tmp_path / "by_hand"), hooks=Hooks(), structure=structure(),
                     quiet=True)
    d = sim.output()
    assert d["fx_body"] == ref.fx_body[0] and d["t_force"] == 0.0
    assert d["fx_body"] == pytest.approx(-float(np.sum(sim.f[:, 0] * sim.structure.ds)),
                                         rel=1e-12)
    lag_file = read_lagrangian_file(os.path.join(sim.output_dir, "lag.00000.vtk"))
    assert np.array_equal(lag_file.point_data["force"], sim.f)
    res = sim.run(1)                                     # continues: row 0 is not repeated
    assert res.history.step.tolist() == [0.0, 1.0]
    assert np.array_equal(res.history.view(float), ref.view(float))
    on_disk = read_history(os.path.join(sim.output_dir, "history.csv"))
    assert np.array_equal(on_disk.fx_body, ref.fx_body)


# ------------------------------------------------------------------ blob membrane pressure
def pressurised_ring(n=160, R=0.25, T=1.0) -> Structure:
    """Zero-rest-length springs with tension T: the discrete Laplace jump is about T/R."""
    X = ring_points(n, r=R)
    i = np.arange(n)
    L = 2 * R * np.sin(np.pi / n)
    ds = 2 * np.pi * R / n
    return Structure("ring", X, np.full(n, ds),
                     springs=Springs(i, (i + 1) % n, np.full(n, T / (L * ds)), np.zeros(n)))


@pytest.mark.parametrize("scheme", ["midpoint", "pm1996"])
def test_blob_pressure_holds_the_membrane_jump(tmp_path, scheme):
    """With the blob coupling the fluid gets P F~ only; the driver adds p_F, so solver.p and
    pressure.NNNNN.vtk show the Laplace jump T/R = 4 (it was 0). 32^2, gaussian4 at 4h: the
    regularised jump is about 9 % above T/R on this coarse grid (tests/test_invariants.py
    checks it on finer grids)."""
    params = make_params(scheme=scheme, coupling="blob", blob="gaussian4", blob_delta=4.0,
                         nx=32, mu=1.0, print_dump=1)
    case = write_case(tmp_path, params, pressurised_ring())
    res = run(case, steps=2, quiet=True)
    sim = res.simulation
    Xg, Yg = res.grid.meshgrid("nodes")
    r = np.hypot(Xg - 0.5, Yg - 0.5)

    def jump(p):
        return float(p[r < 0.125].mean() - p[(r > 0.35) & (r < 0.45)].mean())

    assert sim.p_F is not None
    assert jump(res.solver.p) == pytest.approx(4.0, rel=0.15)
    assert abs(jump(res.solver.p - sim.p_F)) < 1e-6          # the solver's own part has none
    vtk_p = read_structured_scalar(os.path.join(res.output_dir, "pressure.00002.vtk")).field
    assert np.array_equal(vtk_p, res.solver.p)
    # p_F belongs to the force the step used, at the positions it used (X_h or X^n)
    fresh = sim.coupling.force_pressure(sim.f, sim.X_prev, sim.structure.ds)
    assert np.allclose(fresh, sim.p_F, rtol=0.0, atol=1e-12 * np.max(np.abs(sim.p_F)))


def test_adjoint_blob_interpolation_from_the_input_file(tmp_path):
    """blob_interpolation = "adjoint" is accepted by the input file, reaches BlobProjection, and
    gives the exact power identity on a stretched ring: the power the fluid receives,
    sum F . u h^2, equals the power the structure gives, sum f . U ds (docs/numerics.md)."""
    params = make_params(coupling="blob", blob="gaussian4", blob_delta=4.0, nx=32,
                         blob_interpolation="adjoint")
    case = write_case(tmp_path, params, ring_structure())
    text = open(os.path.join(case, "input2d"), encoding="utf-8").read()
    assert 'blob_interpolation = "adjoint"' in text
    res = run(case, steps=2, quiet=True)
    sim = res.simulation
    assert sim.coupling.interpolation == "adjoint"
    assert np.all(np.isfinite(res.solver.u)) and np.all(np.isfinite(sim.structure.X))
    h2 = res.grid.dx * res.grid.dy
    Fx, Fy = sim.coupling.projected_force(sim.f, sim.X_prev, sim.structure.ds)
    U = sim.coupling.interpolate(res.solver.u, res.solver.v, sim.X_prev)
    fluid = float(np.sum(Fx * res.solver.u + Fy * res.solver.v) * h2)
    structure = float(np.sum(np.sum(sim.f * U, axis=1) * sim.structure.ds))
    assert fluid == pytest.approx(structure, rel=1e-10, abs=1e-14)


# ------------------------------------------------------------------ run lifecycle and files
def test_history_is_closed_when_the_run_fails(tmp_path):
    """A hook that raises mid-run: history.csv is closed and holds the rows written before the
    failure, so the case can be rerun in the same process on Windows even while the failed
    Simulation is still referenced (as IPython and pytest.raises keep it)."""
    failing = ("def on_output(step, t, state):\n"
               "    if step >= 2:\n"
               "        raise RuntimeError('hook failed')\n")
    params = make_params(print_dump=1)
    case = write_case(tmp_path, params, ring_structure(), hooks_src=failing)
    sim = Simulation(read_input_file(os.path.join(case, "input2d")), case, quiet=True)
    with pytest.raises(RuntimeError, match="hook failed") as excinfo:
        sim.run(4)
    assert excinfo.value is not None and sim._writer is None
    path = os.path.join(case, "viz", "history.csv")
    assert read_history(path).step.tolist() == [0.0, 1.0, 2.0]
    with open(os.path.join(case, "hooks.py"), "w", encoding="utf-8") as fh:
        fh.write("def on_output(step, t, state):\n    pass\n")
    res = run(case, steps=4, quiet=True)                   # the old Simulation is still alive
    assert read_history(path).step.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert len(res.history) == 5


def test_a_failed_output_is_not_repeated_by_the_next_run(tmp_path):
    """on_output fails once at step 2, after the history row of step 2 was written. A second
    run() on the same Simulation continues from step 2 without writing that row again (it
    used to: rows 0, 1, 2, 2, 3, 4 in memory and on disk)."""
    failing = ("failed = []\n"
               "def on_output(step, t, state):\n"
               "    if step == 2 and not failed:\n"
               "        failed.append(step)\n"
               "        raise RuntimeError('hook failed once')\n")
    params = make_params(print_dump=1)
    case = write_case(tmp_path, params, ring_structure(), hooks_src=failing)
    sim = Simulation(read_input_file(os.path.join(case, "input2d")), case, quiet=True)
    with pytest.raises(RuntimeError, match="hook failed once"):
        sim.run(4)
    assert sim.n == 2
    res = sim.run(2)
    assert res.history.step.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    on_disk = read_history(os.path.join(case, "viz", "history.csv"))
    assert on_disk.step.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert np.array_equal(on_disk.kinetic_energy, res.history.kinetic_energy)


@pytest.mark.parametrize("save_history", [True, False])
def test_second_run_continues(tmp_path, save_history):
    params = make_params(print_dump=1)
    params.output.save_history = save_history
    case = write_case(tmp_path, params, ring_structure())
    sim = Simulation(read_input_file(os.path.join(case, "input2d")), case, quiet=True)
    first = sim.run(2)
    second = sim.run(2)
    assert first.history.step.tolist() == [0.0, 1.0, 2.0]
    assert second.history.step.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert second.steps == 2 and second.t == pytest.approx(4 * DT)
    path = os.path.join(case, "viz", "history.csv")
    if save_history:
        on_disk = read_history(path)
        assert on_disk.step.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
        assert np.array_equal(on_disk.kinetic_energy, second.history.kinetic_energy)
    else:
        assert not os.path.exists(path)
    assert sim.run(0).history.step.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]


def _names(folder):
    return sorted(os.listdir(folder))


def test_rerun_removes_stale_output_of_an_earlier_run(tmp_path):
    params = make_params(print_dump=2, plot=True)
    case = write_case(tmp_path, params, ring_structure())
    out = os.path.join(case, "viz")
    run(case, steps=6, quiet=True)
    assert "lag.00006.vtk" in _names(out) and "snapshot.00006.png" in _names(out)
    for name in ("notes.txt", "vorticity.00004.png", "lag.final.vtk"):    # not ours: kept
        open(os.path.join(out, name), "w").close()
    run(case, steps=2, quiet=True)
    vtk = [n for n in _names(out) if n.endswith(".vtk") and n != "lag.final.vtk"]
    assert vtk == [f"{f}.{s:05d}.vtk" for f in ("lag", "pressure", "velocity", "vorticity")
                   for s in (0, 2)]
    assert [n for n in _names(out) if n.startswith("snapshot")] == ["snapshot.00000.png",
                                                                    "snapshot.00002.png"]
    assert read_history(os.path.join(out, "history.csv")).step.tolist() == [0.0, 2.0]
    # a rerun with every output off leaves nothing of the earlier runs behind
    params.output.save_vtk = params.output.plot = params.output.save_history = False
    write_input_file(os.path.join(case, "input2d"), params)
    run(case, steps=2, quiet=True)
    assert _names(out) == ["lag.final.vtk", "notes.txt", "vorticity.00004.png"]


def test_output_folder_guards(tmp_path):
    params = make_params(print_dump=1)
    case = write_case(tmp_path, params, ring_structure())
    out = os.path.join(case, "viz")
    os.makedirs(out)
    # a history.csv that is not ours is neither deleted nor appended to
    path = os.path.join(out, "history.csv")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("experiment,value\n1,2\n")
    open(os.path.join(out, "lag.00000.vtk"), "w").close()
    with pytest.raises(ValueError, match="not a pyIBM history"):
        run(case, steps=1, quiet=True)
    with open(path, encoding="utf-8") as fh:
        assert fh.read() == "experiment,value\n1,2\n"
    assert os.path.isfile(os.path.join(out, "lag.00000.vtk"))       # nothing deleted either
    # a history in the layout before t_force is ours and is replaced
    old = [c for c in HISTORY_COLUMNS if c != "t_force"]
    os.remove(path)
    with HistoryWriter(path, old) as w:
        w.write([0.0] * len(old))
    res = run(case, steps=1, quiet=True)
    assert read_history(path).dtype.names == HISTORY_COLUMNS and len(res.history) == 2
    # the input file may not name the case directory itself as the output folder
    for bad in ("", ".", "viz/.."):
        params.output.output_dir = bad
        with pytest.raises(ValueError, match="case directory"):
            Simulation(params, case, quiet=True)
    sim = Simulation(params, case, output_dir=case, quiet=True)     # an explicit choice is kept
    assert os.path.normpath(sim.output_dir) == os.path.normpath(case)


def test_a_retry_after_the_history_guard_still_cleans(tmp_path):
    """The guard raised for a history.csv that is not ours; the user moves it and calls run()
    again on the same Simulation. The cleanup must run then (it used to be marked done before
    the guard raised, so a stale lag.00007.vtk stayed beside the new files)."""
    params = make_params(print_dump=1)
    case = write_case(tmp_path, params, ring_structure())
    out = os.path.join(case, "viz")
    os.makedirs(out)
    foreign = os.path.join(out, "history.csv")
    with open(foreign, "w", encoding="utf-8") as fh:
        fh.write("experiment,value\n1,2\n")
    open(os.path.join(out, "lag.00007.vtk"), "w").close()
    sim = Simulation(read_input_file(os.path.join(case, "input2d")), case, quiet=True)
    with pytest.raises(ValueError, match="not a pyIBM history"):
        sim.run(1)
    assert os.path.isfile(os.path.join(out, "lag.00007.vtk"))
    os.replace(foreign, os.path.join(case, "experiment.csv"))
    res = sim.run(1)
    assert "lag.00007.vtk" not in _names(out)
    assert res.history.step.tolist() == [0.0, 1.0]
    assert read_history(os.path.join(out, "history.csv")).step.tolist() == [0.0, 1.0]


# ------------------------------------------------------------------ checks
def test_parameters_built_in_code_are_validated(tmp_path):
    params = make_params(bc=CLOSED, name=None)                       # walls on the FFT back end
    with pytest.raises(ValueError, match="periodic_fft"):
        Simulation(params, str(tmp_path), hooks=Hooks(), quiet=True)
    params = make_params(print_dump=0, name=None)
    with pytest.raises(ValueError, match="print_dump"):
        Simulation(params, str(tmp_path), hooks=Hooks(), quiet=True)


@pytest.mark.parametrize("backend, coupling, bc", [("periodic_fft", "delta", PERIODIC),
                                                   ("periodic_fft", "blob", PERIODIC),
                                                   ("mac", "delta", CLOSED)])
def test_non_finite_state_raises(tmp_path, backend, coupling, bc):
    """A NaN in the velocity stops the run at the next output or step with
    FloatingPointError, for both couplings and for fluid-only runs."""
    for structure in (ring_structure(), None):
        params = make_params(backend=backend, coupling=coupling, bc=bc,
                             name=None if structure is None else "ring")
        sim = Simulation(params, str(tmp_path), hooks=Hooks(), structure=structure, quiet=True)
        sim.solver.u[3, 3] = np.nan
        with pytest.raises(FloatingPointError, match="non-finite"):
            sim.output()
        with pytest.raises(FloatingPointError, match="non-finite"):
            sim.step()
        sim.close()


def test_blow_up_fails_the_command_line(tmp_path):
    """The blob-coupled ring that blows up (k = 4e5 on 16^2, dt = 4e-3) used to run to Tfinal
    writing NaN rows and exit with code 0; now the run stops at that step and exits with 1."""
    params = make_params(coupling="blob", dt=4e-3, tfinal_steps=200, print_dump=25)
    case = write_case(tmp_path, params, ring_structure(k=4e5, rest_factor=0.5))
    r = _cli("run", case, "--quiet")
    assert r.returncode == 1, (r.stdout, r.stderr)
    assert "non-finite solution at step" in r.stderr
    h = read_history(os.path.join(case, "viz", "history.csv"))
    assert len(h) >= 1 and np.all(np.isfinite(h.view(float)))


def test_markers_must_keep_off_the_walls_on_mac(tmp_path):
    """MAC back end: a point closer than the kernel's half-width to a non-periodic side would
    put part of its force on a wall face, which the solver overwrites; we refuse it when the
    case is built and when a step brings a point there."""
    dx = 1.0 / NX
    s = ring_structure()
    s.X[:, 0] += 1.5 * dx - s.X[:, 0].min()                   # leftmost point 1.5 cells off x = 0
    with pytest.raises(ValueError, match="cells from the left side"):
        Simulation(make_params(backend="mac", bc=CLOSED), str(tmp_path), hooks=Hooks(),
                   structure=s, quiet=True)
    # x periodic: no wall there
    Simulation(make_params(backend="mac", bc=CHANNEL), str(tmp_path), hooks=Hooks(),
               structure=s, quiet=True)
    # roma3 reaches 1.5 cells, so 1.5 cells is allowed ...
    sim = Simulation(make_params(backend="mac", bc=CLOSED, kernel="roma3"), str(tmp_path),
                     hooks=Hooks(), structure=s, quiet=True)
    # ... and 1.2 cells is not, also when the point gets there during the run
    sim.structure.X[:, 1] += 1.2 * dx - sim.structure.X[:, 1].min()
    with pytest.raises(ValueError, match="cells from the bottom side at step 0"):
        sim.step()
