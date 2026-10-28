"""examples/jellyfish_bell (Hoover & Miller 2015): smoke runs of the driven, the
free-vibration and the calibration inputs, the committed files against generate.py, the
muscle law of hooks.py, and (slow) the free vibration against Fig. 2, the calibration case
against the quasi-static swing of Fig. 6, and three pulse cycles at the resonant frequency in
which the bell oscillates with the expected amplitude and does not swim at this resolution."""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys

import numpy as np
import pytest

from pyibm.driver import run
from pyibm.inputfile import read_input_file
from pyibm.io.history import read_history
from pyibm.structures import load_hooks, read_beam, read_spring, read_structure, read_vertex

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CASE = os.path.join(ROOT, "examples", "jellyfish_bell")


def load_module(name):
    spec = importlib.util.spec_from_file_location(f"jellyfish_{name}", os.path.join(CASE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_bell(path):
    return np.genfromtxt(path, delimiter=",", names=True).view(np.recarray)


# ------------------------------------------------------------------ smoke

def test_smoke_driven_three_steps(tmp_path):
    res = run(CASE, steps=3, quiet=True, output_dir=str(tmp_path))
    assert res.steps == 3 and res.history.step.tolist() == [0.0, 3.0]
    assert np.all(np.isfinite(res.history.view(float)))
    s = res.structure
    assert s.n == 81 and s.beams is not None and len(s.beams) == 79
    assert len(s.springs) == 80 + 9                       # neighbours + muscle springs
    assert s.targets is None
    assert os.path.isfile(tmp_path / "history.csv") and len(read_history(tmp_path / "history.csv")) == 2
    b = read_bell(tmp_path / "bell.csv")
    assert len(b) == 2 and abs(b.diameter[0] - 1.0) < 1e-3
    assert b.muscle_force[-1] != 0.0                      # the drive has started (sin > 0)
    assert res.solver.kinetic_energy() > 0.0


def test_smoke_free_vibration_three_steps(tmp_path):
    res = run(CASE, input_file="input2d_freevib", steps=3, quiet=True, output_dir=str(tmp_path))
    assert res.history.step.tolist() == [0.0, 3.0]
    s = res.structure
    assert s.n == 81 and len(s.springs) == 80 and len(s.beams) == 79
    b = read_bell(tmp_path / "bell.csv")
    assert abs(b.diameter[0] - 0.9) < 1e-3                # squeezed to 90 % of the width
    assert np.all(b.muscle_force == 0.0)
    # the bending forces are pushing the bell open: fluid is moving after three steps
    assert res.history.max_speed[-1] > 0.0


def test_smoke_calibration_case(tmp_path):
    """generate.write_calibration_case writes the slow-drive case FORCE_SCALE is calibrated
    on: the bell at f = 0.25 s^-1 with the muscle force F_Mag^paper * force_scale, a copy of
    hooks.py, whole cycles; three steps of it run."""
    gen = load_module("generate")
    case = tmp_path / "case"
    path = gen.write_calibration_case(str(case), force_scale=gen.FORCE_SCALE, cycles=2)
    p = read_input_file(path)
    assert p.structure.name == "bell_f0.25" and gen.CALIBRATION_FREQUENCY == 0.25
    assert p.time.n_steps == 2 * round(1.0 / (0.25 * p.time.dt))
    s = read_structure(str(case), "bell_f0.25", ds=p.structure.ds)
    m = np.abs(s.springs.i - s.springs.j) > 1
    L0 = np.hypot(*(s.X0[s.springs.j[m]] - s.X0[s.springs.i[m]]).T)
    assert m.sum() == 9
    assert np.allclose(s.springs.k[m] * L0, gen.FORCE_SCALE * gen.F_MAG_PAPER)    # A_m = 1
    assert gen.F_MAG == gen.FORCE_SCALE * gen.F_MAG_PAPER
    res = run(str(case), steps=3, quiet=True, output_dir=str(tmp_path / "out"))
    assert res.steps == 3 and np.all(np.isfinite(res.history.view(float)))
    assert len(read_bell(tmp_path / "out" / "bell.csv")) == 2


def test_calibration_case_refuses_the_example_folder(tmp_path, monkeypatch):
    """`python generate.py --calibration .` run from the example folder must stop before it
    writes anything: the calibration case would leave bell_f0.25.* among the committed files
    and its input2d would replace the committed one. Checked on a copy of the scripts (the
    copy's generate.py takes the copy as its example folder), so the committed files are
    never at risk."""
    copy = tmp_path / "jellyfish_bell"
    copy.mkdir()
    for name in ("generate.py", "hooks.py"):
        shutil.copyfile(os.path.join(CASE, name), copy / name)
    (copy / "input2d").write_text("the committed input\n")
    monkeypatch.setattr(sys, "path", list(sys.path))      # generate.py edits sys.path
    spec = importlib.util.spec_from_file_location("jellyfish_generate_copy", copy / "generate.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    monkeypatch.chdir(copy)
    with pytest.raises(ValueError, match="own folder"):
        gen.main(["--calibration", "."])
    spellings = [str(copy), str(copy) + os.sep + "."]
    if os.name == "nt":
        spellings.append(str(copy).swapcase())             # the same folder on Windows
    for spelling in spellings:
        with pytest.raises(ValueError, match="own folder"):
            gen.write_calibration_case(spelling, cycles=1)
    assert sorted(os.listdir(copy)) == ["generate.py", "hooks.py", "input2d"]
    assert (copy / "input2d").read_text() == "the committed input\n"
    # a folder of its own is fine, also next to the example
    path = gen.write_calibration_case(str(tmp_path / "calibration"), cycles=1)
    assert read_input_file(path).structure.name == "bell_f0.25"


def test_contraction_rises_on_a_synthetic_record():
    """analyze.contraction_rises measures the climb of the centroid from each trough to the
    next peak (the README's rise per contraction), across the periodic edge of the box and
    without the drift from cycle to cycle; a rise cut off by the end of the record is left
    out."""
    ana = load_module("analyze")
    f = 0.5
    t = np.arange(0.0, 3.25 / f + 1e-12, 1.0 / (400 * f))
    y = 0.05 * (1.0 - np.cos(2 * np.pi * f * t))        # troughs at whole cycles, rise 0.1
    cy = np.mod(11.95 + y, ana.LY)                       # crosses y = 12 every cycle
    P, rise = ana.contraction_rises(np.rec.fromarrays([t, cy], names="t,cy"), f)
    assert P.tolist() == [1, 2, 3] and np.allclose(rise, 0.1, atol=1e-9)
    # with a backward drift of c per cycle the climb is 0.1 cos(phi) - c (1/2 - phi/pi),
    # sin(phi) = c/(0.1 pi), less than max - min over a cycle
    c = 0.03
    phi = np.arcsin(c / (0.1 * np.pi))
    cy = np.mod(11.95 + y - c * f * t, ana.LY)
    b = np.rec.fromarrays([t, cy], names="t,cy")
    P, rise = ana.contraction_rises(b, f)
    assert P.tolist() == [1, 2, 3]
    assert np.allclose(rise, 0.1 * np.cos(phi) - c * (0.5 - phi / np.pi), atol=1e-5)
    yu = ana.unwrap_height(b.cy)
    first = t * f <= 1.0
    assert rise[0] < yu[first].max() - yu[first].min() - 0.01


# ------------------------------------------------------------------ files

def test_committed_files_match_generate():
    gen = load_module("generate")
    assert (gen.A_BELL, gen.B_BELL, gen.D_BELL, gen.F_REF, gen.RE_REF) == (0.5, 0.75, 0.25, 0.86, 172.0)
    assert np.isclose(gen.MU, 0.86 / 172)
    for fname, f, tag, out, _ in gen.cases():
        cells, dt = gen.RESOLUTIONS[tag]
        h = gen.L / cells
        name = gen.structure_name(f, tag)
        if f is None:
            s = gen.bell_structure(name, h, driven=False, squeeze=gen.SQUEEZE)
        else:
            s = gen.bell_structure(name, h, driven=True)
        V = read_vertex(os.path.join(CASE, name + ".vertex"))
        assert np.allclose(V, s.X, atol=1e-12)
        S = read_spring(os.path.join(CASE, name + ".spring"), len(V))
        assert np.array_equal(S.i, s.springs.i) and np.array_equal(S.j, s.springs.j)
        assert np.allclose(S.k, s.springs.k) and np.allclose(S.rest, s.springs.rest)
        B = read_beam(os.path.join(CASE, name + ".beam"), len(V))
        assert np.allclose(B.k, s.beams.k) and np.allclose(B.curv, s.beams.curv)
        p = read_input_file(os.path.join(CASE, fname))
        assert p.structure.name == name and np.isclose(p.structure.ds, s.ds[0])
        assert (p.grid.Nx, p.grid.Ny) == (4 * cells, 12 * cells)
        assert p.solver.backend == "periodic_fft" and p.solver.time_scheme == "midpoint"
        assert p.output.output_dir == out
        if f is None:
            assert p.time.Tfinal == gen.T_FREE
        else:
            steps_per_cycle = round(1.0 / (f * p.time.dt))
            assert abs(steps_per_cycle * p.time.dt * f - 1.0) < 1e-12     # whole cycles
            assert p.time.n_steps == gen.n_cycles(f) * steps_per_cycle
            assert abs(p.time.dt - dt) / dt < 0.01


def test_bell_geometry():
    """Points equally spaced in arclength at ds = h/2, mirror symmetric, margin at
    y = yc - d (eq. 1 and 10, Table 1); the squeezed bell keeps the spacing (section 3.1)."""
    gen = load_module("generate")
    X = read_vertex(os.path.join(CASE, "bell_f0.86.vertex"))
    d = np.hypot(*np.diff(X, axis=0).T)
    assert d.max() / d.min() - 1 < 1e-3 and abs(d.mean() - 0.5 / 16) < 1e-3
    assert np.allclose(X[:, 0] + X[::-1, 0], 2 * gen.X_CENTRE) and np.allclose(X[:, 1], X[::-1, 1])
    assert np.isclose(X[:, 0].max() - X[:, 0].min(), 1.0, atol=2e-4)       # diameter L
    assert np.isclose(X[:, 1].max(), gen.Y_CENTRE + gen.B_BELL)             # apex
    assert np.isclose(X[0, 1], gen.Y_CENTRE - gen.D_BELL)                    # margin
    Xs = read_vertex(os.path.join(CASE, "bell_free.vertex"))
    ds = np.hypot(*np.diff(Xs, axis=0).T)
    assert len(Xs) == len(X) and ds.max() / ds.min() - 1 < 1e-3 and abs(ds.mean() - d.mean()) < 1e-4
    assert np.isclose(Xs[:, 0].max() - Xs[:, 0].min(), 0.9, atol=2e-4)
    assert Xs[0, 1] < X[0, 1]                                                # the arc runs further down


# ------------------------------------------------------------------ hooks

def test_muscle_law():
    gen = load_module("generate")
    hooks = load_hooks(CASE)
    assert hooks.module.MUSCLE_AMPLITUDE == gen.MUSCLE_AMPLITUDE
    assert hooks.module.frequency("bell_f0.86") == 0.86
    assert hooks.module.frequency("bell_fine_f1.25") == 1.25
    assert hooks.module.frequency("bell_free") is None
    s = read_structure(CASE, "bell_f0.86", ds=0.5 / 16)
    m = hooks.module.muscle_mask(s)
    assert m.sum() == 9 and np.all(np.abs(s.springs.i[m] - s.springs.j[m]) > 1)
    # the muscle springs join mirror images in the lower quarter of the bell
    assert np.allclose(s.X0[s.springs.i[m], 1], s.X0[s.springs.j[m], 1])
    assert np.all(s.X0[s.springs.i[m], 1] < gen.Y_CENTRE + 1e-12)
    L0 = np.hypot(*(s.X0[s.springs.j[m]] - s.X0[s.springs.i[m]]).T)
    rest_other = s.springs.rest[~m].copy()
    f, A = 0.86, gen.MUSCLE_AMPLITUDE
    for t in (0.0, 0.25 / f, 0.75 / f, 1.3 / f):
        hooks.update_springs(t, 1e-4, s)
        assert np.allclose(s.springs.rest[m], L0 * (1 - A * np.sin(2 * np.pi * f * t)))
        assert np.array_equal(s.springs.rest[~m], rest_other)
    # the nominal muscle force at full contraction is the paper's F_Mag (scaled like the bell)
    hooks.update_springs(0.25 / f, 1e-4, s)
    assert np.allclose(s.springs.k[m] * (L0 - s.springs.rest[m]), gen.F_MAG)
    # the free bell is left alone
    s_free = read_structure(CASE, "bell_free", ds=0.5 / 16)
    rest = s_free.springs.rest.copy()
    hooks.update_springs(0.3, 1e-4, s_free)
    assert np.array_equal(s_free.springs.rest, rest)


# ------------------------------------------------------------------ slow

@pytest.mark.slow
def test_free_vibration_matches_fig_2(tmp_path):
    """input2d_freevib to t = 3: the bell squeezed to 90 % of its width swings to a diameter
    of 1.07, back to 0.95 and vibrates at 0.86 s^-1 (Table 1, Fig. 2; measured 1.072, 0.945
    and 0.84 s^-1). The frequency is what the stiffness scale was calibrated to; the
    overshoot and the second swing are not. About a minute."""
    ana = load_module("analyze")
    res = run(CASE, input_file="input2d_freevib", quiet=True, output_dir=str(tmp_path))
    b = read_bell(tmp_path / "bell.csv")
    freq, d_max, d_min = ana.free_vibration_frequency(b)
    print(f"\nfree vibration: first maximum {d_max:.3f}, next minimum {d_min:.3f}, "
          f"frequency {freq:.3f} s^-1 (paper 1.07, 0.95, 0.86), {res.wall_time:.0f} s")
    assert np.all(np.isfinite(res.history.view(float)))
    assert 1.04 < d_max < 1.10 and 0.92 < d_min < 0.97
    assert 0.75 < freq < 0.95


@pytest.mark.slow
def test_force_scale_gives_the_quasi_static_swing_of_fig_6(tmp_path):
    """Two cycles of the calibration case (f = 0.25 s^-1, force x FORCE_SCALE): the swing of
    the diameter in the second cycle is Fig. 6's quasi-static 0.186 to within 5 % (measured
    0.187 at force x 118). The old FORCE_SCALE of 150, the linear extrapolation from a run at
    force x 430, gave 0.233. About 2.5 minutes."""
    gen = load_module("generate")
    ana = load_module("analyze")
    gen.write_calibration_case(str(tmp_path / "case"), cycles=2)
    res = run(str(tmp_path / "case"), quiet=True, output_dir=str(tmp_path / "out"))
    b = read_bell(tmp_path / "out" / "bell.csv")
    P, V, A = ana.per_cycle(b, gen.CALIBRATION_FREQUENCY)
    print(f"\ncalibration, force x {gen.FORCE_SCALE:g}: swing per cycle {np.round(A, 4).tolist()} "
          f"(paper {ana.PAPER_AMPLITUDE[0.25]}), {res.wall_time:.0f} s")
    assert len(P) == 2 and np.all(np.isfinite(res.history.view(float)))
    assert abs(A[1] / ana.PAPER_AMPLITUDE[0.25] - 1.0) < 0.05
    assert np.all(np.abs(V) < 0.02)                        # a slow squeeze does not swim


@pytest.mark.slow
def test_driven_bell_oscillates_at_the_resonant_frequency(tmp_path):
    """Three pulse cycles of input2d (f = 0.86 s^-1): the diameter swings by 0.3 to 0.55
    per cycle (measured 0.39 in cycles 2 and 3; the paper's Fig. 6: 0.245 at P = 5, 0.285 at
    P = 25) and the bell does not get away: at h = L/16 it moves back and forth by a few
    hundredths of a diameter per cycle (measured -0.04, +0.03, 0.00 L/s) where the paper's
    bell swims forward at 0.31 L/s by the fifth cycle (see the README). Under a minute."""
    gen = load_module("generate")
    ana = load_module("analyze")
    p = read_input_file(os.path.join(CASE, "input2d"))
    f = gen.F_REF
    spc = round(1.0 / (f * p.time.dt))
    res = run(CASE, steps=3 * spc, quiet=True, output_dir=str(tmp_path))
    b = read_bell(tmp_path / "bell.csv")
    P, V, A = ana.per_cycle(b, f)
    print(f"\n3 cycles at f = {f}: V per cycle {np.round(V, 3).tolist()}, "
          f"amplitude {np.round(A, 3).tolist()}, {res.wall_time:.0f} s")
    assert len(P) == 3
    assert np.all(np.isfinite(res.history.view(float)))
    assert 0.3 < A[-1] < 0.55
    assert np.all(np.abs(V) < 0.2)
