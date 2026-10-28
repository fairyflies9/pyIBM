"""examples/taylor_green: the initial_velocity hook, smoke runs of both inputs from it
(through analyze.run_case and through pyibm.run, as the command line runs it), the committed
inputs against generate.py, the decay rate on a coarse grid (always on) and the spatial and
temporal convergence tables (slow)."""

from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest

from pyibm.driver import run
from pyibm.grid import Grid, layout_collocated, layout_u, layout_v
from pyibm.inputfile import read_input_file
from pyibm.io.history import read_history
from pyibm.structures import load_hooks

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CASE = os.path.join(ROOT, "examples", "taylor_green")
INPUTS = (("input2d", "periodic_fft"), ("input2d_mac", "mac"))


def load(name: str):
    spec = importlib.util.spec_from_file_location(f"taylor_green_{name}", os.path.join(CASE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("input_name, backend", INPUTS)
def test_smoke_three_steps(tmp_path, input_name, backend):
    an = load("analyze")
    params = read_input_file(os.path.join(CASE, input_name))
    assert params.solver.backend == backend and params.structure.name is None
    params = an.with_settings(params, N=16, files=True)
    res = an.run_case(params, steps=3, output_dir=str(tmp_path))
    assert res.steps == 3 and res.history.step.tolist() == [0.0, 3.0]   # print_dump = 25
    assert np.all(np.isfinite(res.history.view(float)))
    assert res.structure is None
    # KE(0) = (rho/2) * 2 * (1/4) * L^2 = 0.25 exactly on the grid (trig sums are exact)
    assert res.history.kinetic_energy[0] == pytest.approx(0.25, rel=1e-12)
    assert res.history.kinetic_energy[1] < res.history.kinetic_energy[0]
    if backend == "periodic_fft":
        assert res.history.max_speed[0] == pytest.approx(1.0, rel=1e-12)
    else:   # mac: the max speed is measured on the cell centres after face averaging
        assert 0.9 < res.history.max_speed[0] < 1.0
    nu = params.fluid.mu / params.fluid.rho
    ratio = res.history.max_speed[1] / res.history.max_speed[0]
    assert ratio == pytest.approx(np.exp(-8 * np.pi**2 * nu * res.t), rel=5e-3)
    assert an.max_error(res) < 1e-3
    assert os.path.isfile(tmp_path / "history.csv") and os.path.isfile(tmp_path / "velocity.00003.vtk")
    assert len(read_history(tmp_path / "history.csv")) == 2
    if backend == "mac":
        assert np.max(np.abs(res.solver.divergence())) < 1e-12


def test_hook_gives_the_exact_field_on_nodes_and_faces():
    hooks = load_hooks(CASE)
    assert hooks.defined() == ["initial_velocity"]
    an = load("analyze")
    g = Grid(1.0, 1.0, 16, 16)
    for lu, lv in ((layout_collocated(g), layout_collocated(g)), (layout_u(g), layout_v(g))):
        u, v = hooks.initial_velocity(g, lu, lv)
        assert u.shape == lu.shape and v.shape == lv.shape
        Xu, Yu = np.meshgrid(*lu.coordinates(), indexing="ij")
        Xv, Yv = np.meshgrid(*lv.coordinates(), indexing="ij")
        assert np.array_equal(u, -np.cos(2 * np.pi * Xu) * np.sin(2 * np.pi * Yu))
        assert np.array_equal(v, np.sin(2 * np.pi * Xv) * np.cos(2 * np.pi * Yv))
        assert np.array_equal(u, an.exact_velocity(Xu, Yu, 0.0, 0.01)[0])
    with pytest.raises(ValueError, match="square box"):
        g2 = Grid(2.0, 1.0, 32, 16)
        hooks.initial_velocity(g2, layout_collocated(g2), layout_collocated(g2))


@pytest.mark.parametrize("input_name, backend", INPUTS)
def test_command_line_run_starts_from_the_vortex(tmp_path, input_name, backend):
    """What python -m pyibm run examples/taylor_green does: the hook sets the field, so the
    run starts from the vortex (KE = 0.25), not from a fluid at rest, and decays."""
    res = run(CASE, input_file=input_name, steps=3, quiet=True, output_dir=str(tmp_path))
    assert res.history.kinetic_energy[0] == pytest.approx(0.25, rel=1e-12)
    assert res.history.kinetic_energy[1] < res.history.kinetic_energy[0]
    assert load("analyze").max_error(res) < 1e-3


def test_committed_inputs_match_generate(tmp_path):
    gen = load("generate")
    written = gen.main(str(tmp_path))
    assert [os.path.basename(p) for p in written] == list(gen.FILES)
    for path in written:   # byte for byte, header included
        with open(path, "rb") as a, open(os.path.join(CASE, os.path.basename(path)), "rb") as b:
            assert a.read() == b.read(), os.path.basename(path)
    for input_name, backend in INPUTS:
        p = read_input_file(os.path.join(CASE, input_name))
        q = gen.parameters(backend)
        assert p == q
        assert (p.fluid.rho, p.fluid.mu) == (gen.RHO, gen.MU)
        assert (p.grid.Nx, p.grid.Ny, p.grid.Lx, p.grid.Ly) == (gen.N, gen.N, gen.L, gen.L)
        assert (p.time.dt, p.time.Tfinal, p.time.n_steps) == (gen.DT, gen.TFINAL, 250)
        assert p.solver.time_scheme == "midpoint" and p.solver.advection == "central"
        assert all(s == "periodic" for s in (p.bc.left, p.bc.right, p.bc.bottom, p.bc.top))


@pytest.mark.parametrize("input_name, backend", INPUTS)
def test_decay_rate_on_a_coarse_grid(input_name, backend):
    """32^2, 50 midpoint steps: the kinetic energy decays at the grid rate 2 nu lambda_h
    (to 0.1 %), which is 0.32 % below the exact 16 pi^2 nu at this resolution."""
    an = load("analyze")
    params = an.with_settings(read_input_file(os.path.join(CASE, input_name)), N=32, Tfinal=0.1)
    res = an.run_case(params)
    assert res.steps == 50
    nu = params.fluid.mu / params.fluid.rho
    # the gap the docstring quotes: 1 - sin^2(pi h)/(pi h)^2 = 0.321 % at h = 1/32
    gap = 1.0 - an.discrete_decay_rate(32, nu) / an.exact_decay_rate(nu)
    assert gap == pytest.approx(3.21e-3, rel=1e-3)
    rate = an.measured_decay_rate(res.history)
    assert rate == pytest.approx(an.discrete_decay_rate(32, nu), rel=1e-3)
    assert rate == pytest.approx(an.exact_decay_rate(nu), rel=2e-2)
    assert an.max_error(res) < 1e-3


@pytest.mark.slow
def test_convergence_tables():
    """Spatial order 2 on both back ends (central advection), temporal order 2 for the
    midpoint scheme and 1 for pm1996; the mac blend advection does not converge in h at
    fixed dt (its artificial viscosity is 0.6 dt U^2). Measured: spatial orders 2.00 / 1.98,
    temporal 2.00 / 1.1 (periodic_fft) and 2.00 / 1.04 (mac). About 10 s.

    Taylor-Green is an eigenmode of every step with central advection, so the midpoint
    columns check the Crank-Nicolson factor of the viscous term only, and mac/pm1996 the
    backward-Euler factor; we assert that too, so the docstring cannot claim more. Only
    periodic_fft/pm1996 (upwind advection) feels the nonlinear term."""
    an = load("analyze")
    base = {backend: read_input_file(os.path.join(CASE, fname)) for fname, backend in INPUTS}
    spatial = an.spatial_table(base, Ns=(16, 32, 64))
    for label in ("periodic_fft", "mac central"):
        errs = [r[1] for r in spatial[label]]
        assert errs[0] < 5e-3
        for a, b in zip(errs[:-1], errs[1:]):
            assert np.log2(a / b) > 1.9
    assert spatial["mac blend"][-1][1] > 5 * spatial["mac central"][-1][1]
    temporal = an.temporal_table(base, N=32, dts=(0.01, 0.005, 0.0025))
    for backend in ("periodic_fft", "mac"):
        errs = [r[1] for r in temporal[f"{backend}/midpoint"]]
        for a, b in zip(errs[:-1], errs[1:]):
            assert np.log2(a / b) > 1.9
        errs = [r[1] for r in temporal[f"{backend}/pm1996"]]
        for a, b in zip(errs[:-1], errs[1:]):
            assert 0.9 < np.log2(a / b) < 1.4
    # the scalar factors: z = nu lambda_h dt, n = T/dt, against the reference at dt_min/8
    nu = base["mac"].fluid.mu / base["mac"].fluid.rho
    T, lam = base["mac"].time.Tfinal, an.discrete_decay_rate(32, nu) / (2 * nu)
    ref_dt = 0.0025 / an.REF_FACTOR

    def amplitude(dt, scheme):
        z, n = nu * lam * dt, int(round(T / dt))
        return ((1 - z / 2) / (1 + z / 2)) ** n if scheme == "midpoint" else (1 + z) ** -n

    u_max = {"periodic_fft": 1.0, "mac": np.sin(2 * np.pi * 7.5 / 32)}   # largest |u0| on the points
    for label, scheme in (("periodic_fft/midpoint", "midpoint"), ("mac/midpoint", "midpoint"),
                          ("mac/pm1996", "pm1996")):
        for dt, err, _ in temporal[label]:
            expected = abs(amplitude(dt, scheme) - amplitude(ref_dt, scheme)) * u_max[label.split("/")[0]]
            assert err == pytest.approx(expected, rel=1e-2), (label, dt)
