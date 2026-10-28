"""The forced circle: histories of the two couplings and the error trend with the time step.

Without options, reads history.csv of the variants that have been run (viz/delta, viz/blob),
prints the largest target lag in cells next to the stiffness and the force on the circle,
reads any saved time-step study (viz/dt_study_<coupling>.csv) and saves results.png.

With --study, repeats the run with dt = dx/2, dx/4, ... dx/32 and a reference at dx/64
(Fovargue 2012, section 5.4.0.1, varies dt the same way, from h/2 to h/100), measures at
t = T the errors of her eq. 5.40,

    E_u = max |u - u_ref| / max |u_ref|   (over both components and all nodes),
    E_X = max_k |X_k - X_ref,k|           (the boundary positions, in metres),

against the finest run of the same coupling, prints them with the observed orders next to
her E_u values (read off her Fig. 5.6: 4.0e-3, 3.1e-5, 2.8e-7, 9.0e-8 at dt = h/2, h/10,
h/50, h/100, standard algorithm, about +-15 %), saves viz/dt_study_<coupling>.csv and the
plot. The midpoint scheme is second order, so E_u and E_X should fall by 4 per halving of
dt; her IMEX spectral deferred corrections were fourth order. The fourth panel of
results.png draws the boundary of both default runs at t = T/4 and t = T from the
lag.NNNNN.vtk files.

L. Fovargue, Addressing the computational cost of the immersed boundary method through
multi-implicit and multi-rate strategies with time parallelism, PhD thesis, University of
North Carolina at Chapel Hill (2012).

    python -m pyibm run examples/forced_circle
    python -m pyibm run examples/forced_circle --input input2d_blob
    python examples/forced_circle/analyze.py
    python examples/forced_circle/analyze.py --study                      # both couplings
    python examples/forced_circle/analyze.py --study --coupling delta --factors 4,8,16 --reference 32
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys
import warnings

# Let the script run from a plain clone (no pip install -e .).
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from pyibm.driver import Simulation
from pyibm.io.history import read_history
from pyibm.io.vtk import read_lagrangian_points
from pyibm.structures import read_target, read_vertex

HERE = os.path.dirname(os.path.abspath(__file__))

VARIANTS = (
    ("input2d", "delta", "viz/delta", "delta (peskin4)"),
    ("input2d_blob", "blob", "viz/blob", "blob (gaussian4, 4 dx)"),
)
FACTORS = (2, 4, 8, 16, 32)       # dt = dx / factor
REFERENCE = 64
FOVARGUE_E_U = {2: 4.0e-3, 10: 3.1e-5, 50: 2.8e-7, 100: 9.0e-8}   # thesis Fig. 5.6, standard algorithm
STUDY_COLUMNS = ("factor", "dt", "E_u", "E_X", "E_X_over_dx", "order_u", "order_X", "max_lag_over_dx", "wall")
SHAPE_TIMES = (0.25, 1.0)         # s, the boundary is drawn at the output nearest each


def load_generate():
    spec = importlib.util.spec_from_file_location("forced_circle_generate", os.path.join(HERE, "generate.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def warning_note(message: str) -> str:
    """A short note for the study table from one of the driver's warnings: "tether number 25"
    from the tether rule of thumb, "CFL 1.2" from the CFL check, else the first 40 characters."""
    m = re.match(r"tether number k dt\^2/\(rho dx\) = ([^ ]+)", message)
    if m:
        return "tether number " + m.group(1)
    m = re.match(r"CFL number ([^ ]+)", message)
    if m:
        return "CFL " + m.group(1)
    return message[:40]


def run_level(coupling: str, factor: int, tfinal: float = 1.0) -> dict:
    """One run with dt = dx/factor to t = tfinal, no files written. Returns u, v, X at the
    end, the history (65 rows, so the largest target lag is sampled as often in time at
    every level, and at dx/16 exactly as in the default runs), the wall time and the
    driver's warnings (tether number, CFL)."""
    gen = load_generate()
    p = gen.parameters(coupling, "viz/study", dt=gen.DX / factor)
    p.time.Tfinal = tfinal
    p.output.save_vtk = False
    p.output.save_history = False
    p.output.print_dump = max(1, p.time.n_steps // 64)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sim = Simulation(p, HERE, quiet=True)
        res = sim.run()
    notes = [warning_note(str(w.message)) for w in caught if issubclass(w.category, RuntimeWarning)]
    return {"factor": factor, "dt": p.time.dt, "steps": res.steps, "u": res.solver.u.copy(),
            "v": res.solver.v.copy(), "X": res.structure.X.copy(), "history": res.history,
            "wall": res.wall_time, "notes": ", ".join(notes)}


def errors(level: dict, ref: dict) -> tuple[float, float]:
    """(E_u, E_X) of a level against the reference run (Fovargue 2012 eq. 5.40)."""
    du = max(np.max(np.abs(level["u"] - ref["u"])), np.max(np.abs(level["v"] - ref["v"])))
    umax = max(np.max(np.abs(ref["u"])), np.max(np.abs(ref["v"])))
    dX = level["X"] - ref["X"]
    return float(du / umax), float(np.max(np.hypot(dX[:, 0], dX[:, 1])))


def study(coupling: str, factors=FACTORS, reference: int = REFERENCE, tfinal: float = 1.0,
          verbose: bool = True) -> np.recarray:
    """Run the levels and the reference; return a record array with the STUDY_COLUMNS:
    factor, dt, E_u, E_X, E_X/dx, the observed orders between consecutive levels (nan on the
    first), the largest target lag in cells and the wall time in seconds."""
    gen = load_generate()
    levels = []
    for f in list(factors) + [reference]:
        lev = run_level(coupling, f, tfinal)
        levels.append(lev)
        if verbose:
            print(f"  {coupling}: dt = dx/{f:<3d} {lev['steps']:6d} steps  {lev['wall']:6.1f} s"
                  f"  max lag {lev['history'].max_target_error_over_dx.max():.3f} dx  max CFL "
                  f"{lev['history'].cfl.max():.3f}  {lev['notes']}", flush=True)
    ref = levels.pop()
    rows = []
    for lev in levels:
        e_u, e_x = errors(lev, ref)
        rows.append((float(lev["factor"]), lev["dt"], e_u, e_x, e_x / gen.DX, np.nan, np.nan,
                     float(lev["history"].max_target_error_over_dx.max()), lev["wall"]))
    data = np.array(rows, dtype=float)
    out = np.rec.fromarrays([data[:, j] for j in range(len(STUDY_COLUMNS))], names=list(STUDY_COLUMNS))
    for k in range(1, len(out)):
        ratio = out.dt[k - 1] / out.dt[k]
        out.order_u[k] = np.log(out.E_u[k - 1] / out.E_u[k]) / np.log(ratio)
        out.order_X[k] = np.log(out.E_X[k - 1] / out.E_X[k]) / np.log(ratio)
    return out


def save_study(path: str, table: np.recarray, coupling: str, reference: int, tfinal: float) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"# forced circle time-step study, {coupling} coupling, reference dt = dx/{reference}, T = {tfinal}\n")
        fh.write(",".join(STUDY_COLUMNS) + "\n")
        for row in table:
            fh.write(",".join(repr(float(v)) for v in row) + "\n")


def read_study(path: str) -> np.recarray:
    return np.genfromtxt(path, delimiter=",", names=True, skip_header=1).view(np.recarray)


def print_study(table: np.recarray, coupling: str, reference: int) -> None:
    print(f"{coupling} coupling, errors at the end against the run with dt = dx/{reference}:")
    print("   dt      steps/period     E_u        order     E_X [m]     E_X/dx    order   max lag/dx")
    for r in table:
        o_u = "   -  " if np.isnan(r.order_u) else f"{r.order_u:6.2f}"
        o_x = "   -  " if np.isnan(r.order_X) else f"{r.order_X:6.2f}"
        print(f"  dx/{int(r.factor):<3d} {1.0 / r.dt:10.0f}     {r.E_u:9.3e}  {o_u}   {r.E_X:9.3e}  "
              f"{r.E_X_over_dx:8.4f}  {o_x}   {r.max_lag_over_dx:8.3f}")
    print("  Fovargue 2012 Fig. 5.6 (E_u, standard IMEX SDC algorithm, her stiffnesses, read off the plot):")
    print("   " + ",  ".join(f"dx/{f}: {e:.1e}" for f, e in FOVARGUE_E_U.items()))


def plot(ax_lag, ax_force, ax_study, ax_shape, studies: dict) -> bool:
    found = False
    for input_name, coupling, out, label in VARIANTS:
        path = os.path.join(HERE, out, "history.csv")
        if not os.path.isfile(path):
            print(f"{input_name}: no {out}/history.csv (run it first)")
            continue
        found = True
        h = read_history(path)
        X = read_vertex(os.path.join(HERE, "circle.vertex"))
        k = float(np.max(read_target(os.path.join(HERE, "circle.target"), X).k))
        print(f"{input_name} ({label}): target stiffness {k:g}, max target lag {h.max_target_error_over_dx.max():.3f} "
              f"cells, max CFL {h.cfl.max():.3f}, max speed {h.max_speed.max():.3f} m/s, fx on the circle in "
              f"[{h.fx_body.min():+.3f}, {h.fx_body.max():+.3f}] N, {len(h)} rows to t = {h.t[-1]:.4g} s")
        color = "C0" if coupling == "delta" else "C1"
        ax_lag.plot(h.t, h.max_target_error_over_dx, color=color, label=label)
        ax_force.plot(h.t, h.fx_body, color=color, label=label)
        for t_want, style in zip(SHAPE_TIMES, (":", "-")):
            k = int(np.argmin(np.abs(h.t - t_want)))
            vtk = os.path.join(HERE, out, f"lag.{int(h.step[k]):05d}.vtk")
            if os.path.isfile(vtk):
                Xs = read_lagrangian_points(vtk)
                Xc = np.vstack([Xs, Xs[:1]])
                ax_shape.plot(Xc[:, 0], Xc[:, 1], style, color=color, lw=0.8,
                              label=f"{coupling}, t = {h.t[k]:.3g} s")
    for coupling, table in studies.items():
        found = True
        color = "C0" if coupling == "delta" else "C1"      # the same colours as the other panels
        ax_study.loglog(table.dt, table.E_u, "o-", color=color, label=f"E_u, {coupling}")
        ax_study.loglog(table.dt, table.E_X_over_dx, "s--", color=color, mfc="none",
                        label=f"E_X / dx, {coupling}")
    if studies:
        dts = np.array(sorted({float(d) for t in studies.values() for d in t.dt}))
        anchor = max(float(np.max(t.E_u)) for t in studies.values())
        ax_study.loglog(dts, anchor * (dts / dts.max()) ** 2, "k:", label="slope 2")
        f = np.array(list(FOVARGUE_E_U))
        ax_study.loglog((1.0 / 128) / f, list(FOVARGUE_E_U.values()), "r^", label="Fovargue 2012 Fig. 5.6, E_u")
    return found


def main(argv=None) -> str | None:
    ap = argparse.ArgumentParser(description="forced circle: histories and the time-step study")
    ap.add_argument("--study", action="store_true", help="run the time-step study")
    ap.add_argument("--coupling", default="both", choices=("delta", "blob", "both"))
    ap.add_argument("--factors", default=",".join(str(f) for f in FACTORS), help="dt = dx/factor, comma separated")
    ap.add_argument("--reference", type=int, default=REFERENCE, help="factor of the reference run")
    ap.add_argument("--tfinal", type=float, default=1.0)
    args = ap.parse_args(argv)
    studies = {}
    if args.study:
        factors = tuple(int(f) for f in args.factors.split(","))
        for coupling in (("delta", "blob") if args.coupling == "both" else (args.coupling,)):
            print(f"time-step study, {coupling} coupling, T = {args.tfinal}")
            table = study(coupling, factors, args.reference, args.tfinal)
            path = os.path.join(HERE, "viz", f"dt_study_{coupling}.csv")
            save_study(path, table, coupling, args.reference, args.tfinal)
            print_study(table, coupling, args.reference)
            print("saved", path)
            studies[coupling] = table
    else:
        for coupling in ("delta", "blob"):
            path = os.path.join(HERE, "viz", f"dt_study_{coupling}.csv")
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as fh:
                    header = fh.readline().strip("# \n")
                print(header)
                studies[coupling] = read_study(path)
                reference = int(header.split("dx/")[1].split(",")[0])
                print_study(studies[coupling], coupling, reference)
    fig = Figure(figsize=(10.0, 7.5), layout="constrained")
    FigureCanvasAgg(fig)
    (ax_lag, ax_force), (ax_study, ax_shape) = fig.subplots(2, 2)
    if not plot(ax_lag, ax_force, ax_study, ax_shape, studies):
        return None
    ax_shape.set_aspect("equal")
    ax_shape.set_xlabel("x [m]")
    ax_shape.set_ylabel("y [m]")
    ax_shape.set_title("the boundary (default runs, dt = dx/16)")
    if ax_shape.lines:
        ax_shape.legend(fontsize="small", loc="center left", bbox_to_anchor=(1.02, 0.5))   # off the curves
    ax_lag.set_ylabel("max target lag / dx")
    ax_lag.set_xlabel("t [s]")
    ax_lag.legend(fontsize="small", loc="upper left")
    ax_force.set_ylabel("x force on the circle [N]")
    ax_force.set_xlabel("t [s]")
    ax_force.legend(fontsize="small", loc="upper right")
    ax_study.set_xlabel("dt [s]")
    ax_study.set_ylabel("error at t = T against the finest run")
    ax_study.set_title("time-step study")
    if studies:
        ax_study.set_ylim(bottom=1e-10)                      # room for the legend below the data
        ax_study.legend(fontsize="small", loc="lower right")
    out = os.path.join(HERE, "results.png")
    fig.savefig(out, dpi=150)
    print("saved", out)
    return out


if __name__ == "__main__":
    main()
