"""Regression against the 2008 MATLAB code (tests/data/README.md describes the data).

The reference files hold the state of the original moving-plate code after steps 1, 2, 3, 5,
10, 25, 50 and 100 for target stiffness 1 and 1e4. We run examples/heaving_plate_2008 with
the matching input file (pm1996 scheme, cosine4 kernel, 16 x 16, dt = dx^2/nu) through the
driver and compare, at every checkpoint, u, v, X, the pressure, the spread force and the
Lagrangian force with a relative tolerance of 1e-10 (SciPy's pocketfft and MATLAB's FFTW
round differently, so bit equality is not demanded; the measured differences are below
1e-14). The MATLAB target update uses the accumulated time, ours n*dt; that difference is
below 1e-14 relative over 100 steps and is checked too.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import scipy.io as sio

from pyibm.driver import Simulation
from pyibm.inputfile import read_input_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASE = os.path.join(ROOT, "examples", "heaving_plate_2008")
DATA = os.path.join(ROOT, "tests", "data")
RTOL = 1e-10
FIELDS = ("u", "v", "X", "press", "f1", "f2", "fb")


def rel_diff(ours, ref) -> float:
    """max |ours - ref| / max |ref|; for an all-zero reference, max |ours|."""
    ours = np.asarray(ours, dtype=float)
    ref = np.asarray(ref, dtype=float)
    assert ours.shape == ref.shape, (ours.shape, ref.shape)
    scale = float(np.max(np.abs(ref)))
    err = float(np.max(np.abs(ours - ref)))
    return err / scale if scale > 0.0 else err


def load(mat_name: str) -> dict:
    return sio.loadmat(os.path.join(DATA, f"heaving_plate_2008_{mat_name}.mat"), simplify_cells=True)


@pytest.mark.parametrize("input_name, mat_name", [("input2d", "stiff1"),
                                                  ("input2d_stiff1e4", "stiff1e4")])
def test_matches_matlab_checkpoints(tmp_path, input_name, mat_name):
    d = load(mat_name)
    p = d["params"]
    params = read_input_file(os.path.join(CASE, input_name))

    # The input file reproduces the MATLAB parameters exactly (dt bit for bit).
    assert params.time.dt == float(p["dt"])
    assert (params.grid.Nx, params.grid.Ny) == (int(p["M"]), int(p["N"]))
    assert params.grid.Lx == float(p["length"]) and params.grid.Ly == float(p["width"])
    assert params.fluid.mu == float(p["mu"]) and params.fluid.rho == float(p["rho"])
    assert params.structure.ds == float(p["ds"])
    assert params.solver.time_scheme == "pm1996" and params.solver.kernel == "cosine4"
    nsteps = int(p["nsteps"])
    assert params.time.n_steps == nsteps

    sim = Simulation(params, CASE, output_dir=str(tmp_path), quiet=True)
    assert np.array_equal(sim.structure.X, d["X0"])
    assert float(np.max(sim.structure.targets.k)) == float(p["stiff"])

    # Fourier symbols: MATLAB's a, b, c (projection) and d (1/A) arrays, moving_plate.m.
    sym = sim.solver.symbols
    assert rel_diff(sym.s1 * sym.s1 * sym.inv_s_sq, d["a"]) < 1e-14
    assert rel_diff(sym.s1 * sym.s2 * sym.inv_s_sq, d["b"]) < 1e-14
    assert rel_diff(sym.s2 * sym.s2 * sym.inv_s_sq, d["c"]) < 1e-14
    assert rel_diff(sim.solver.inv_A_pm1996, d["d"]) < 1e-14

    save_steps = [int(k) for k in np.atleast_1d(p["save_steps"])]
    assert all(f"step_{n:04d}" in d for n in save_steps)
    worst = {k: 0.0 for k in FIELDS}
    worst_time = 0.0
    for n in range(1, nsteps + 1):
        sim.step()
        key = f"step_{n:04d}"
        if key not in d:
            continue
        s = d[key]
        assert int(s["n"]) == n == sim.n
        checks = {
            "u": (sim.solver.u, s["u"]),
            "v": (sim.solver.v, s["v"]),
            "X": (sim.structure.X, s["X"]),
            "press": (sim.solver.p, s["press"]),
            "f1": (sim.Fu, s["f1"]),          # spread force used in this step
            "f2": (sim.Fv, s["f2"]),
            "fb": (sim.f, s["fb"]),           # target force per unit length
        }
        for name, (ours, ref) in checks.items():
            e = rel_diff(ours, ref)
            worst[name] = max(worst[name], e)
            assert e <= RTOL, f"{mat_name}: {name} at step {n} differs by {e:.3e} relative (> {RTOL})"
        # n*dt against MATLAB's accumulated time
        e_t = abs(sim.t - float(s["time"])) / float(s["time"])
        worst_time = max(worst_time, e_t)
        assert e_t < 1e-14
    summary = ", ".join(f"{k} {v:.2e}" for k, v in worst.items())
    print(f"\n{mat_name}: max relative difference over the checkpoints: {summary}; "
          f"time {worst_time:.2e}")


def test_reference_files_are_complete():
    """Both files hold the eight checkpoints the generator script saves."""
    for mat_name in ("stiff1", "stiff1e4"):
        d = load(mat_name)
        steps = [int(k) for k in np.atleast_1d(d["params"]["save_steps"])]
        assert steps == [1, 2, 3, 5, 10, 25, 50, 100]
        for n in steps:
            s = d[f"step_{n:04d}"]
            for name in ("u", "v", "press", "f1", "f2"):
                assert s[name].shape == (16, 16)
            assert s["X"].shape == (16, 2)
