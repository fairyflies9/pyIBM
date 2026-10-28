"""Run the Taylor-Green vortex through both back ends and check it against the exact solution.

The exact solution on the doubly periodic box [0, L)^2 is

    u = -cos(2 pi x/L) sin(2 pi y/L) exp(-8 pi^2 nu t/L^2),
    v =  sin(2 pi x/L) cos(2 pi y/L) exp(-8 pi^2 nu t/L^2),
    p = -(rho/4) (cos(4 pi x/L) + cos(4 pi y/L)) exp(-16 pi^2 nu t/L^2),

so the kinetic energy decays as exp(-16 pi^2 nu t/L^2). On the grid the viscous term sees
the symbol of the five-point Laplacian instead of 8 pi^2/L^2:

    lambda_h = (8/h^2) sin^2(pi h/L)  =  8 pi^2/L^2 (1 - (pi h/L)^2/3 + ...),

and the discrete nonlinear term (centred differences) is still a discrete gradient that the
projection removes, on both back ends, so the grid solution is u_exact(x_i)
exp(-nu lambda_h t) up to the time-stepping error. The spatial error of both back ends is
therefore the O(h^2) symbol error. For the same reason the field is an exact eigenmode of
the midpoint step: each step multiplies it by the Crank-Nicolson factor of the viscous
term, so the midpoint temporal error measured here is that of the Crank-Nicolson factor
alone, and the predictor and the advection play no part in it.

The initial field is the initial_velocity hook in hooks.py, which the driver calls when the
Simulation is built, so `python -m pyibm run examples/taylor_green` starts from the vortex
too. This script

1. runs the two default inputs (input2d: periodic_fft; input2d_mac: mac, all sides
   periodic), writes viz/<backend>/ and measures the decay rate of the kinetic energy from
   history.csv;
2. prints a spatial convergence table (N = 16, 32, 64, 128 at a fixed small dt) with the
   max-norm velocity error against the exact solution, for periodic_fft, mac with central
   advection and mac with the default blend advection (robust, but at fixed dt it adds an
   artificial viscosity of about 0.6 dt U^2, so it does not converge in h);
3. prints a temporal convergence table (N = 32, dt halved four times) with the error against
   a run of the same scheme at dt/8, for the midpoint and pm1996 schemes on both back ends;
4. saves results.png.

    python examples/taylor_green/analyze.py            # full tables, about a minute
    python examples/taylor_green/analyze.py --quick    # three resolutions, three steps
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import os
import sys
import time

# Let the script run from a plain clone (no pip install -e .).
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from pyibm.driver import RunResult, Simulation
from pyibm.inputfile import OutputParams, Parameters, read_input_file

HERE = os.path.dirname(os.path.abspath(__file__))
INPUTS = (("input2d", "periodic_fft"), ("input2d_mac", "mac"))


def _load_hooks():
    """hooks.py of this directory under a private module name."""
    spec = importlib.util.spec_from_file_location("taylor_green_hooks", os.path.join(HERE, "hooks.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


exact_velocity = _load_hooks().exact_velocity    # (X, Y, t, nu, L) -> (u, v)

SPATIAL_N = (16, 32, 64, 128)
SPATIAL_DT = 1e-3
TEMPORAL_N = 32
TEMPORAL_DT = (0.01, 0.005, 0.0025, 0.00125)
REF_FACTOR = 8                    # the temporal reference run uses dt_min / REF_FACTOR


# ------------------------------------------------------------------ exact solution
def discrete_decay_rate(N: int, nu: float, L: float = 1.0) -> float:
    """Decay rate of the kinetic energy on the grid, 2 nu lambda_h with
    lambda_h = (8/h^2) sin^2(pi h/L) the five-point Laplacian symbol of the mode (1, 1)."""
    h = L / N
    return 2.0 * nu * (8.0 / h**2) * np.sin(np.pi * h / L) ** 2


def exact_decay_rate(nu: float, L: float = 1.0) -> float:
    """Decay rate of the kinetic energy, 16 pi^2 nu/L^2."""
    return 16.0 * np.pi**2 * nu / L**2


# ------------------------------------------------------------------ running a case
def nu_of(params: Parameters) -> float:
    return params.fluid.mu / params.fluid.rho


def run_case(params: Parameters, steps: int | None = None, output_dir: str | None = None,
             quiet: bool = True) -> RunResult:
    """Build the Simulation in this directory and run it. The Simulation loads hooks.py from
    here, whose initial_velocity hook writes the exact field at t = 0."""
    sim = Simulation(params, HERE, output_dir=output_dir, quiet=quiet)
    return sim.run(steps)


def with_settings(params: Parameters, N: int | None = None, dt: float | None = None,
                  Tfinal: float | None = None, scheme: str | None = None,
                  advection: str | None = None, files: bool = False) -> Parameters:
    """A copy of params with the resolution, step, length, scheme or advection replaced.
    With files = False nothing is written to disk (no history.csv, no VTK)."""
    grid = params.grid if N is None else dataclasses.replace(params.grid, Nx=N, Ny=N)
    tm = dataclasses.replace(params.time, dt=params.time.dt if dt is None else dt,
                             Tfinal=params.time.Tfinal if Tfinal is None else Tfinal)
    solver = dataclasses.replace(params.solver,
                                 time_scheme=params.solver.time_scheme if scheme is None else scheme,
                                 advection=params.solver.advection if advection is None else advection)
    output = params.output if files else OutputParams(print_dump=10**9, save_vtk=False,
                                                      save_history=False, plot=False)
    return dataclasses.replace(params, grid=grid, time=tm, solver=solver, output=output)


def max_error(res: RunResult) -> float:
    """Max-norm velocity error against the exact solution at the final time of the run."""
    g, s, nu, L = res.grid, res.solver, nu_of(res.parameters), res.parameters.grid.Lx
    if res.parameters.solver.backend == "periodic_fft":
        X, Y = g.meshgrid("nodes")
        ue, ve = exact_velocity(X, Y, res.t, nu, L)
        return float(max(np.max(np.abs(s.u - ue)), np.max(np.abs(s.v - ve))))
    Xu, Yu = g.meshgrid("u")
    Xv, Yv = g.meshgrid("v")
    ue = exact_velocity(Xu, Yu, res.t, nu, L)[0]
    ve = exact_velocity(Xv, Yv, res.t, nu, L)[1]
    return float(max(np.max(np.abs(s.u - ue)), np.max(np.abs(s.v - ve))))


def max_difference(a: RunResult, b: RunResult) -> float:
    """Max-norm difference of the velocities of two runs on the same grid and back end."""
    return float(max(np.max(np.abs(a.solver.u - b.solver.u)), np.max(np.abs(a.solver.v - b.solver.v))))


def measured_decay_rate(history) -> float:
    """-d(log KE)/dt from a least-squares line through the history rows."""
    return float(-np.polyfit(history.t, np.log(history.kinetic_energy), 1)[0])


# ------------------------------------------------------------------ tables
def orders(errors) -> list[str]:
    out = [""]
    for a, b in zip(errors[:-1], errors[1:]):
        out.append(f"{np.log2(a / b):.2f}" if a > 0 and b > 0 else "")
    return out


def spatial_table(base: dict[str, Parameters], Ns=SPATIAL_N, dt: float = SPATIAL_DT):
    """Max-norm error at Tfinal against the exact solution for every N, at a fixed dt small
    enough that the time error is negligible (midpoint scheme). Columns: periodic_fft, mac
    with central advection, mac with the blend. Returns {label: [(N, error, seconds), ...]}."""
    columns = (("periodic_fft", base["periodic_fft"], None),
               ("mac central", base["mac"], "central"),
               ("mac blend", base["mac"], "blend"))
    table = {}
    for label, params, adv in columns:
        rows = []
        for N in Ns:
            p = with_settings(params, N=N, dt=dt, scheme="midpoint", advection=adv)
            t0 = time.perf_counter()
            res = run_case(p)
            rows.append((N, max_error(res), time.perf_counter() - t0))
        table[label] = rows
    return table


def temporal_table(base: dict[str, Parameters], N: int = TEMPORAL_N, dts=TEMPORAL_DT,
                   ref_factor: int = REF_FACTOR):
    """Max-norm difference at Tfinal from a run of the same scheme and grid at
    dt_min/ref_factor, for midpoint and pm1996 on both back ends (mac with central
    advection). Taylor-Green is an eigenmode of the midpoint step, so the midpoint columns
    measure the Crank-Nicolson factor of the viscous term (module docstring).
    Returns {label: [(dt, error, seconds), ...]}."""
    table = {}
    for backend in ("periodic_fft", "mac"):
        for scheme in ("midpoint", "pm1996"):
            ref = run_case(with_settings(base[backend], N=N, dt=min(dts) / ref_factor,
                                         scheme=scheme, advection="central"))
            rows = []
            for dt in dts:
                p = with_settings(base[backend], N=N, dt=dt, scheme=scheme, advection="central")
                t0 = time.perf_counter()
                res = run_case(p)
                rows.append((dt, max_difference(res, ref), time.perf_counter() - t0))
            table[f"{backend}/{scheme}"] = rows
    return table


def print_table(title: str, table: dict, key_name: str, key_fmt: str) -> None:
    labels = list(table)
    keys = [r[0] for r in table[labels[0]]]
    print(title)
    print(f"  {key_name:>8s}  " + "  ".join(f"{lab:>22s}" for lab in labels))
    ords = {lab: orders([r[1] for r in table[lab]]) for lab in labels}
    for i, k in enumerate(keys):
        cells = []
        for lab in labels:
            e = table[lab][i][1]
            o = ords[lab][i]
            cells.append(f"{e:12.3e} {('(' + o + ')') if o else '':>8s}")
        print(f"  {key_fmt.format(k):>8s}  " + "  ".join(f"{c:>22s}" for c in cells))
    secs = sum(r[2] for lab in labels for r in table[lab])
    print(f"  (error = max-norm velocity error; order in parentheses; {secs:.1f} s of runs)")


# ------------------------------------------------------------------ figure
def save_figure(base_runs: dict[str, RunResult], spatial: dict, temporal: dict, path: str) -> None:
    fig = Figure(figsize=(12.0, 3.8), layout="constrained")
    FigureCanvasAgg(fig)
    ax_ke, ax_h, ax_dt = fig.subplots(1, 3)
    for backend, res in base_runs.items():
        h = res.history
        ax_ke.semilogy(h.t, h.kinetic_energy / h.kinetic_energy[0], "o", label=backend, mfc="none")
    if base_runs:
        res = next(iter(base_runs.values()))
        t = np.linspace(0.0, res.t, 100)
        ax_ke.semilogy(t, np.exp(-exact_decay_rate(nu_of(res.parameters), res.parameters.grid.Lx) * t),
                       "k-", label="exact")
    ax_ke.set_xlabel("t [s]")
    ax_ke.set_ylabel("KE(t) / KE(0)")
    ax_ke.set_title("kinetic energy decay")
    ax_ke.legend()
    for label, rows in spatial.items():
        Ns = np.array([r[0] for r in rows], float)
        ax_h.loglog(Ns, [r[1] for r in rows], "o-", label=label)
    if spatial:
        rows = next(iter(spatial.values()))
        Ns = np.array([r[0] for r in rows], float)
        ax_h.loglog(Ns, rows[0][1] * (Ns[0] / Ns) ** 2, "k:", label="h^2")
    ax_h.set_xlabel("N (cells per side)")
    ax_h.set_ylabel("max |u - u_exact|")
    ax_h.set_title(f"spatial error, dt = {SPATIAL_DT:g}")
    ax_h.legend(fontsize=8)
    for label, rows in temporal.items():
        dts = np.array([r[0] for r in rows])
        ax_dt.loglog(dts, [r[1] for r in rows], "o-", label=label)
    if temporal:
        rows = next(iter(temporal.values()))
        dts = np.array([r[0] for r in rows])
        ax_dt.loglog(dts, rows[0][1] * (dts / dts[0]) ** 2, "k:", label="dt^2")
        ax_dt.loglog(dts, rows[0][1] * (dts / dts[0]), "k--", label="dt")
    ax_dt.set_xlabel("dt [s]")
    ax_dt.set_ylabel("max |u - u_ref|")
    ax_dt.set_title(f"temporal error, N = {TEMPORAL_N}")
    ax_dt.legend(fontsize=8)
    fig.savefig(path, dpi=150)


# ------------------------------------------------------------------ main
def main(quick: bool = False, plot: bool = True) -> dict:
    base = {backend: read_input_file(os.path.join(HERE, fname)) for fname, backend in INPUTS}

    print("default runs (exact initial condition, output to viz/<backend>/):")
    base_runs = {}
    for backend, params in base.items():
        t0 = time.perf_counter()
        res = run_case(params, quiet=True)
        wall = time.perf_counter() - t0
        base_runs[backend] = res
        nu, L, N = nu_of(params), params.grid.Lx, params.grid.Nx
        rate = measured_decay_rate(res.history)
        print(f"  {backend:12s} {N}^2, dt {params.time.dt:g}, {res.steps} steps in {wall:.2f} s: "
              f"max error {max_error(res):.3e}, KE decay rate {rate:.5f} "
              f"(exact 16 pi^2 nu = {exact_decay_rate(nu, L):.5f}, "
              f"grid 2 nu lambda_h = {discrete_decay_rate(N, nu, L):.5f})")

    Ns = SPATIAL_N[:-1] if quick else SPATIAL_N
    dts = TEMPORAL_DT[:-1] if quick else TEMPORAL_DT
    spatial = spatial_table(base, Ns)
    print_table(f"spatial convergence at dt = {SPATIAL_DT:g}, T = {base['periodic_fft'].time.Tfinal:g}, midpoint:",
                spatial, "N", "{:d}")
    temporal = temporal_table(base, TEMPORAL_N, dts)
    print_table(f"temporal convergence at N = {TEMPORAL_N}, T = {base['periodic_fft'].time.Tfinal:g}, "
                f"error against the same scheme at dt_min/{REF_FACTOR}:", temporal, "dt", "{:g}")
    out = None
    if plot:
        out = os.path.join(HERE, "results.png")
        save_figure(base_runs, spatial, temporal, out)
        print("saved", out)
    return {"base": base_runs, "spatial": spatial, "temporal": temporal, "figure": out}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true", help="three resolutions and three time steps")
    ap.add_argument("--no-plot", action="store_true", help="do not write results.png")
    args = ap.parse_args()
    main(quick=args.quick, plot=not args.no_plot)
