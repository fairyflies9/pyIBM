"""Compare the computed velocity profiles with the exact solution of Stokes' second problem.

For every variant that has been run (viz/ny64, viz/ny32, viz/ny128) this reads
history.csv for the output steps, the velocity.NNNNN.vtk dumps that fall in the last period
(eight phases, T/8 apart), and compares u(y) with the exact damped shear wave
u = U exp(-y/d) cos(omega t - y/d) (stokes_exact.py eq. 1) and with the finite-depth
solution that has the top wall at rest (eq. 2). It prints one line per phase, the largest
error per variant, and saves results.png: the profiles at four phases next to the exact
curves, and the largest error against Ny.

    python -m pyibm run examples/stokes_second_problem
    python -m pyibm run examples/stokes_second_problem --input input2d_ny32
    python -m pyibm run examples/stokes_second_problem --input input2d_ny128
    python examples/stokes_second_problem/analyze.py
"""

from __future__ import annotations

import os
import sys

# Let the script run from a plain clone (no pip install -e .).
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np

import stokes_exact as ex
from pyibm.inputfile import read_input_file
from pyibm.io.history import read_history
from pyibm.io.vtk import read_structured_vector, vtk_filename

VARIANTS = (
    ("input2d", "viz/ny64"),
    ("input2d_ny32", "viz/ny32"),
    ("input2d_ny128", "viz/ny128"),
)


def last_period_profiles(case_dir: str, input_name: str, output_dir: str | None = None) -> list[dict]:
    """The velocity dumps of the last period of a run: one dict per output time with
    ``t``, ``y`` (cell centres), ``u`` (the profile, averaged over the periodic direction),
    ``x_spread`` (largest variation of u along x: zero up to round-off for this problem),
    ``v_max`` and ``Ny``. ``output_dir`` defaults to the input file's Output_Info."""
    params = read_input_file(os.path.join(case_dir, input_name))
    out = params.output.output_dir if output_dir is None else output_dir
    if not os.path.isabs(out):
        out = os.path.join(case_dir, out)
    h = read_history(os.path.join(out, "history.csv"))
    sel = h.t > h.t[-1] - ex.PERIOD + 1e-9 * ex.PERIOD
    rows = []
    for step, t in zip(h.step[sel], h.t[sel]):
        vel = read_structured_vector(vtk_filename(out, "velocity", int(round(step))))
        y = vel.origin[1] + vel.spacing[1] * np.arange(vel.fx.shape[1])
        rows.append({"t": float(t), "y": y, "u": vel.fx.mean(axis=0),
                     "x_spread": float(np.max(np.ptp(vel.fx, axis=0))),
                     "v_max": float(np.max(np.abs(vel.fy))), "Ny": vel.fx.shape[1]})
    return rows


def compare(rows: list[dict]) -> list[dict]:
    """Per phase: the largest |u - u_exact| / U against the semi-infinite solution (eq. 1)
    and against the finite-depth one (eq. 2)."""
    table = []
    for r in rows:
        ue = ex.velocity(r["y"], r["t"])
        uf = ex.velocity_finite_depth(r["y"], r["t"])
        table.append({"t": r["t"], "phase_deg": (ex.OMEGA * r["t"] * 180.0 / np.pi) % 360.0,
                      "err": float(np.max(np.abs(r["u"] - ue)) / ex.U),
                      "err_finite_depth": float(np.max(np.abs(r["u"] - uf)) / ex.U),
                      "x_spread": r["x_spread"], "v_max": r["v_max"]})
    return table


def main() -> str | None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import NullFormatter

    results = {}
    for input_name, out in VARIANTS:
        path = os.path.join(HERE, out, "history.csv")
        if not os.path.isfile(path):
            print(f"{input_name}: no {out}/history.csv (run it first)")
            continue
        rows = last_period_profiles(HERE, input_name)
        table = compare(rows)
        Ny = rows[0]["Ny"]
        print(f"{input_name}: Ny = {Ny} ({Ny * ex.D / ex.LY:.1f} cells per penetration depth), "
              f"{len(rows)} phases in the last period")
        print("      t      phase   max|u-u_exact|/U   vs finite depth   x-variation   max|v|")
        for row in table:
            print(f"   {row['t']:7.4f}  {row['phase_deg']:5.0f}      {row['err']:.3e}         "
                  f"{row['err_finite_depth']:.3e}       {row['x_spread']:.1e}      {row['v_max']:.1e}")
        worst = max(row["err"] for row in table)
        print(f"   largest error over the period: {worst:.3e} U")
        results[input_name] = (Ny, rows, worst)
    if not results:
        return None

    fig = Figure(figsize=(10.0, 4.5), layout="constrained")
    FigureCanvasAgg(fig)
    ax_prof, ax_conv = fig.subplots(1, 2)
    # profiles of the finest run at four phases: dots are the computation, lines the exact wave
    name = max(results, key=lambda k: results[k][0])
    Ny, rows, _ = results[name]
    yy = np.linspace(0.0, ex.LY, 400)
    wanted = (0.0, 90.0, 180.0, 270.0)
    colours = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd")
    for phase, colour in zip(wanted, colours):
        r = min(rows, key=lambda r: abs(((ex.OMEGA * r["t"] * 180 / np.pi) % 360) - phase))
        ax_prof.plot(ex.velocity(yy, r["t"]) / ex.U, yy / ex.D, "-", color=colour, lw=1.5,
                     label=f"omega t = {phase:.0f} deg")
        ax_prof.plot(r["u"] / ex.U, r["y"] / ex.D, "o", color=colour, ms=3.5, mfc="none")
    ax_prof.set_xlabel("u / U")
    ax_prof.set_ylabel("y / d")
    ax_prof.set_ylim(0.0, 5.0)
    ax_prof.set_title(f"{name}: Ny = {Ny} (dots), exact (lines)", fontsize=10)
    ax_prof.legend(fontsize=8)
    # convergence: largest error in the last period against Ny
    Nys = sorted(results[k][0] for k in results)
    errs = [results[k][2] for k in sorted(results, key=lambda k: results[k][0])]
    ax_conv.loglog(Nys, errs, "o-", color="#1f77b4", label="max |u - u_exact| / U")
    ref = errs[0] * (np.asarray(Nys, dtype=float) / Nys[0]) ** -2
    ax_conv.loglog(Nys, ref, "--", color="#7f7f7f", label="second order")
    ax_conv.set_xlabel("Ny")
    ax_conv.set_ylabel("error over the last period")
    ax_conv.set_xticks(Nys)
    ax_conv.set_xticklabels([str(n) for n in Nys])
    ax_conv.xaxis.set_minor_formatter(NullFormatter())     # only the three Ny labels
    ax_conv.legend(fontsize=8)
    out = os.path.join(HERE, "results.png")
    fig.savefig(out, dpi=150)
    print("saved", out)
    return out


if __name__ == "__main__":
    main()
