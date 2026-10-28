"""examples/flapping_filament: smoke runs of the three inputs, the committed files against
generate.py, the nondimensional-to-SI mapping of the element stiffnesses, the hooks, the
paper's Fig. 13 reference tables of analyze.py, and (slow) the coarse run to t = 20 against
the flapping amplitude and period of Huang, Shin & Sung 2007, Fig. 13."""

from __future__ import annotations

import importlib.util
import os
import warnings

import numpy as np
import pytest

from pyibm.driver import TETHER_STABLE, run
from pyibm.grid import Grid
from pyibm.inputfile import read_input_file
from pyibm.structures import (beam_energy, load_hooks, read_beam, read_mass, read_spring,
                              read_structure, read_target, read_vertex, spring_energy)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CASE = os.path.join(ROOT, "examples", "flapping_filament")
INPUTS = ("input2d", "input2d_coarse", "input2d_clamped")


def load_module(name: str):
    spec = importlib.util.spec_from_file_location(f"flapping_filament_{name}", os.path.join(CASE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("input_name", INPUTS)
def test_smoke_three_steps(tmp_path, input_name):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)       # no tether-number or CFL warning
        res = run(CASE, input_file=input_name, steps=3, quiet=True, output_dir=str(tmp_path))
    assert res.steps == 3 and res.history.step.tolist() == [0.0, 3.0]   # print_dump >= 3
    assert np.all(np.isfinite(res.history.view(float)))
    s = res.structure
    n_seg = s.n - 1
    assert s.springs is not None and len(s.springs) == n_seg
    assert s.beams is not None and len(s.beams) == n_seg - 1
    assert s.masses is not None and len(s.masses) == n_seg
    assert s.targets is not None and len(s.targets) == (2 if input_name == "input2d_clamped" else 1)
    assert res.history.max_target_error_over_dx[-1] < 0.1
    assert os.path.isfile(tmp_path / "history.csv")
    tip = np.genfromtxt(tmp_path / "tip.csv", delimiter=",", names=True)
    assert tip.shape == (2,) and tip["step"].tolist() == [0.0, 3.0]
    assert abs(tip["length"][0] - 1.0) < 1e-12 and abs(tip["length"][-1] - 1.0) < 1e-3
    # the free stream is ramped, so after three steps the fluid barely moves
    assert res.history.max_speed[-1] < 0.05
    assert res.parameters.solver.backend == "mac" and res.parameters.bc.right == "outflow"


def test_committed_files_match_generate():
    gen = load_module("generate")
    for input_name, (structure, nx, ny, n_seg, dt, dump, vtk, clamped, out) in gen.VARIANTS.items():
        ds = gen.L / n_seg
        k = gen.stiffnesses(ds)
        X = gen.clamped_points(n_seg) if clamped else gen.pinned_points(n_seg)
        assert X.shape == (n_seg + 1, 2)
        assert np.allclose(np.hypot(*(X[1:] - X[:-1]).T), ds)            # unit spacing
        assert np.allclose(X[n_seg], gen.X_FIXED)                         # last node is the fixed end
        if clamped:
            assert np.allclose(X[n_seg - 1] - X[n_seg], (ds, 0.0))        # first segment horizontal
        else:
            assert np.allclose(X[0] - X[n_seg], gen.L * np.array([np.cos(gen.ANGLE), np.sin(gen.ANGLE)]))
        V = read_vertex(os.path.join(CASE, structure + ".vertex"))
        assert np.allclose(V, X, atol=1e-15)
        sp = read_spring(os.path.join(CASE, structure + ".spring"), len(V))
        assert sp.i.tolist() == list(range(n_seg)) and sp.j.tolist() == list(range(1, n_seg + 1))
        assert np.allclose(sp.k, k["spring"]) and np.allclose(sp.rest, ds)
        be = read_beam(os.path.join(CASE, structure + ".beam"), len(V))
        assert be.q.tolist() == list(range(1, n_seg)) and np.all(be.p == be.q - 1) and np.all(be.r == be.q + 1)
        assert np.allclose(be.k, k["beam"]) and np.all(be.curv == 0.0)
        ma = read_mass(os.path.join(CASE, structure + ".mass"), V)
        assert ma.idx.tolist() == list(range(n_seg)) and np.allclose(ma.m, gen.RHO1) and np.allclose(ma.k, k["mass"])
        ta = read_target(os.path.join(CASE, structure + ".target"), V)
        assert ta.idx.tolist() == ([n_seg, n_seg - 1] if clamped else [n_seg]) and np.allclose(ta.k, k["target"])
        p = read_input_file(os.path.join(CASE, input_name))
        assert (p.grid.Nx, p.grid.Ny, p.grid.Lx, p.grid.Ly) == (nx, ny, gen.LX, gen.LY)
        assert p.time.dt == dt and p.time.Tfinal == gen.T_FINAL
        assert (p.fluid.rho, p.fluid.mu) == (gen.RHO0, gen.MU)
        assert p.structure.name == structure and p.structure.ds == ds
        assert (p.structure.gravity_x, p.structure.gravity_y) == (gen.GRAVITY, 0.0)
        assert (p.solver.backend, p.solver.time_scheme, p.solver.kernel) == ("mac", "midpoint", "peskin4")
        assert (p.bc.left, p.bc.right, p.bc.bottom, p.bc.top) == ("dirichlet", "outflow", "dirichlet", "dirichlet")
        assert p.output.print_dump == dump and p.output.save_vtk == vtk and p.output.output_dir == out
        # the tether numbers k dt^2 / (rho dx): targets and mass tethers below the driver's rule
        # of thumb, and all three below 0.5, where the README puts them (springs are not
        # checked by the driver)
        dx = gen.LX / nx
        for name, element in (("target", "target"), ("mass", "mass tether")):
            assert k[name] * dt**2 / (gen.RHO0 * dx) < TETHER_STABLE[(element, "midpoint")]
        for name in ("spring", "target", "mass"):
            assert k[name] * dt**2 / (gen.RHO0 * dx) < 0.5, (input_name, name)


def test_nondimensional_numbers_of_the_paper():
    """The SI constants reproduce the paper's Re, mass ratio, gamma and Fr (their eqs. 5-6)."""
    gen = load_module("generate")
    assert gen.RHO0 * gen.U_INF * gen.L / gen.MU == pytest.approx(200.0)
    assert gen.RHO1 / (gen.RHO0 * gen.L) == pytest.approx(1.5)
    assert gen.GAMMA / (gen.RHO1 * gen.U_INF**2 * gen.L**2) == pytest.approx(1e-3)
    assert gen.GRAVITY * gen.L / gen.U_INF**2 == pytest.approx(0.5)
    assert gen.KS / (gen.RHO1 * gen.U_INF**2) == pytest.approx(100.0)


def test_element_stiffness_mapping():
    """k_beam = gamma/ds^6 and k_spring = K_S/ds^2 give the continuum energies.

    Bend the filament into a circular arc of radius R: the physical bending energy
    (beam_energy times ds, since force densities are multiplied by ds when spread) must be
    gamma/2 kappa^2 per unit length over the arc the inner nodes cover. Stretch it uniformly
    by a strain eps: the spring energy must be K_S/2 eps^2 L.
    """
    gen = load_module("generate")
    grid = Grid(gen.LX, gen.LY, 192, 128, periodic_x=False, periodic_y=False)
    s = read_structure(CASE, "filament", ds=gen.L / 32)
    ds = float(s.ds[0])
    n_seg = s.n - 1
    R = 2.0
    # nodes spaced by chord ds (so the springs are at rest); the arc length per node is
    # ds (1 + O(ds^2/R^2)), well inside the 1e-3 tolerance on the bending energy
    dtheta = 2.0 * np.arcsin(0.5 * ds / R)
    theta = dtheta * np.arange(n_seg + 1)
    arc = np.column_stack([2.0 + R * np.sin(theta), 2.0 - R * np.cos(theta)])
    e_bend = beam_energy(arc, s.beams, grid) * ds
    expected = 0.5 * gen.GAMMA / R**2 * (n_seg - 1) * ds
    assert e_bend == pytest.approx(expected, rel=1e-3)
    assert spring_energy(arc, s.springs, grid) * ds == pytest.approx(0.0, abs=1e-12)
    eps = 0.01
    stretched = s.X0[n_seg] + (1.0 + eps) * (s.X0 - s.X0[n_seg])
    assert spring_energy(stretched, s.springs, grid) * ds == pytest.approx(0.5 * gen.KS * eps**2 * gen.L, rel=1e-12)
    assert beam_energy(stretched, s.beams, grid) == pytest.approx(0.0, abs=1e-12)


def test_hooks_ramp_and_recorder(tmp_path):
    hooks = load_hooks(CASE)
    mod = load_module("hooks")
    y = np.linspace(0.0, 4.0, 9)
    for fn in (hooks.bc_left, hooks.bc_bottom, hooks.bc_top):
        u0, v0 = fn(y, 0.0)
        u1, v1 = fn(y, 5.0)
        uh, _ = fn(y, 0.5 * mod.T_RAMP)
        assert np.all(u0 == 0.0) and np.all(v0 == 0.0) and np.all(u1 == mod.U_INF) and np.all(v1 == 0.0)
        assert np.allclose(uh, 0.5 * mod.U_INF) and u0.shape == y.shape
    assert hooks.bc_right is None and hooks.update_targets is None
    assert mod.free_stream_speed(0.25 * mod.T_RAMP) == pytest.approx(0.5 * (1 - np.cos(np.pi / 4)))


def test_paper_reference_tables():
    """The extrema of Fig. 13 in analyze.py: maxima and minima alternate in time, the swing
    grows to t = 20 and then holds, and paper_stats gives the amplitude and period the
    README quotes (first trough -0.25 at t = 2.0, first maximum 0.24 at t = 3.7; 0.29 over
    10 < t < 15, 0.31 over 10 < t < 20, 0.33 settled; period 3.27 early, 3.22 overall).
    The maxima are read from the top edge of the drawn line less half its width: the middle
    of the dark run in a crest column takes in both flanks and reads low (0.297 and 0.310
    for the fourth and fifth maxima, where the line is at 0.302 and 0.317)."""
    an = load_module("analyze")
    mx, mn = np.array(an.REF_MAXIMA), np.array(an.REF_MINIMA)
    assert np.all(mx[:, 1] > 0) and np.all(mn[:, 1] < 0)
    times = np.sort(np.concatenate([mx[:, 0], mn[:, 0]]))
    kind = np.isin(times, mx[:, 0])
    assert np.all(kind[1:] != kind[:-1])                     # max, min, max, ... alternate
    # after the release from the initial tilt (first trough -0.25) the swing grows to t = 20
    assert np.all(np.diff(mx[mx[:, 0] < 20, 1]) > 0) and np.all(np.diff(-mn[1:][mn[1:, 0] < 20, 1]) > 0)
    assert mn[0, 0] == pytest.approx(2.0, abs=0.05) and mn[0, 1] == pytest.approx(-0.25, abs=0.005)
    assert mx[0, 0] == pytest.approx(3.7, abs=0.05) and mx[0, 1] == pytest.approx(0.236, abs=0.003)
    assert mx[3, 0] == pytest.approx(13.5, abs=0.05) and mx[3, 1] == pytest.approx(0.302, abs=0.003)
    assert mx[4, 0] == pytest.approx(16.7, abs=0.05) and mx[4, 1] == pytest.approx(0.317, abs=0.003)
    assert an.paper_stats(10.0, 15.0)["amplitude"] == pytest.approx(0.29, abs=0.005)
    assert an.paper_stats(10.0, 20.0)["amplitude"] == pytest.approx(0.31, abs=0.005)
    assert an.REF_AMPLITUDE == pytest.approx(0.33, abs=0.005)
    assert an.paper_stats(0.0, 17.0)["period"] == pytest.approx(3.27, abs=0.01)
    assert an.REF_PERIOD == pytest.approx(3.22, abs=0.01)
    # the same formulas as flapping_stats: a sampled sine of amplitude 0.3 and period 3.2
    t = np.linspace(0.0, 20.0, 4001)
    st = an.flapping_stats(t, 0.3 * np.sin(2 * np.pi * t / 3.2), 5.0)
    assert st["amplitude"] == pytest.approx(0.3, abs=1e-4) and st["period"] == pytest.approx(3.2, abs=1e-3)


def test_force_time_of_old_and_new_histories():
    """analyze.force_time: the t_force column when the history has one, else t - dt/2 (0 in
    row 0)."""
    an = load_module("analyze")
    t = np.array([0.0, 0.05, 0.1])
    old = np.rec.fromarrays([t, np.ones(3)], names=["t", "fx_body"])
    assert np.allclose(an.force_time(old, 2.5e-4), [0.0, 0.05 - 1.25e-4, 0.1 - 1.25e-4])
    new = np.rec.fromarrays([t, t - 1e-3, np.ones(3)], names=["t", "t_force", "fx_body"])
    assert np.array_equal(an.force_time(new, 2.5e-4), t - 1e-3)


@pytest.mark.slow
def test_coarse_run_sustains_flapping(tmp_path):
    """input2d_coarse to t = 20 (25000 steps, 128-958 s on this laptop depending on its load): the free
    end settles into a sustained flapping state by t = 10. Measured over 10 < t < 20:
    amplitude 0.46, period 3.55-3.6, three maxima spread by 0.04, strain 2.5 %, target
    error 0.03 cells. Huang, Shin & Sung 2007 Fig. 13 has 0.31 and 3.21 over the same window
    (0.33 settled): the coarse filament, 16 cells long under a 4-cell kernel, bends far more
    than theirs and overshoots the amplitude by half (README, 'What to expect'; the fine
    input is the comparison, at 2 h it is not a test). The bands below hold the coarse run
    to its own state while admitting the paper's values, and catch a run that stops flapping,
    blows up or stretches."""
    an = load_module("analyze")
    res = run(CASE, input_file="input2d_coarse", steps=25000, quiet=True, output_dir=str(tmp_path))
    assert res.steps == 25000 and res.t == pytest.approx(20.0)
    st = an.analyze_variant(str(tmp_path), t_from=10.0)
    ref = an.paper_stats(10.0, 20.0)
    print(f"\ncoarse run: {res.wall_time:.0f} s; amplitude {st['amplitude']:.3f} (measured 0.46; paper "
          f"{ref['amplitude']:.2f} at t = 10-20, {an.REF_AMPLITUDE:.2f} settled), period {st['period']:.2f} "
          f"(measured 3.6; paper {ref['period']:.2f}), "
          f"{st['n_max']} maxima spread {st['max_spread']:.3f}, max strain {st['max_strain']:.2%}, "
          f"max target error {st['max_target_error_over_dx']:.3f} cells")
    assert st["n_max"] >= 2
    assert 0.25 < st["amplitude"] < 0.6
    assert 2.6 < st["period"] < 4.4
    assert st["max_spread"] < 0.1               # settled: successive maxima agree
    assert st["max_strain"] < 0.04
    assert st["max_target_error_over_dx"] < 0.1
    assert np.all(res.history.cfl < 1.0)
    assert np.all(np.isfinite(res.history.view(float)))
