"""Compare the computed profiles of the Womersley channel with the exact solution.

For every variant that has been run (viz/alpha3, viz/alpha10, viz/alpha3_incremental,
viz/alpha3_periodic, viz/alpha10_periodic) this reads history.csv for the output steps, the
velocity and pressure dumps that fall in the last period (eight phases, T/8 apart), takes
the profile u(y) on the column of cells at x = Lx/2 (away from the inlet and the outlet) and
compares it with womersley_exact.py eq. (1). Per phase it prints the largest error over the
profile, the largest error with the two cells next to each wall left out, the flow rate
error, and for the inflow runs the pressure gradient between x = Lx/4 and 3 Lx/4 next to
K cos(omega t_p). It saves results.png: the alpha = 3 and alpha = 10 inflow profiles at four
phases next to the exact curves.

The time level of the pressure. The velocity in velocity.NNNNN.vtk is u^{n+1} at the output
time t, but the pressure in pressure.NNNNN.vtk comes from the projection of the full step
from t - dt to t, and it is the pressure that balances the momentum over that whole step.
With the default non-incremental projection the midpoint predictor has no pressure term,
so the projection's phi is the whole pressure, averaged over the step. With
mac_projection = "incremental" the predictor carries the previous step's pressure and the
projection adds the change, so the stored pressure is again the one for the whole step.
Either way it belongs to the middle of the step, t_p = t - dt/2: a least-squares fit of
-dp/dx to A cos(omega (t - lag)) gives a lag of 0.64 dt for input2d and 0.56 dt for
input2d_incremental. Comparing with K cos(omega t) instead would add a phase error of up to
K sin(omega dt/2) = 0.049, 0.8 % of K. All the inputs here use time_scheme = "midpoint". The
pm1996 step also gets its pressure from one projection over the whole step. On input2d its
fitted lag is 0.68 dt at dt = T/400 and 0.74 dt at T/800 (0.51 dt and 0.57 dt with the
incremental projection). That is not a clean half step, but t - dt/2 matches better than t
or t - dt in all four runs, so we compare at t - dt/2 there too.

    python -m pyibm run examples/womersley_channel
    python -m pyibm run examples/womersley_channel --input input2d_alpha10
    python -m pyibm run examples/womersley_channel --input input2d_periodic
    python -m pyibm run examples/womersley_channel --input input2d_alpha10_periodic
    python examples/womersley_channel/analyze.py
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

import womersley_exact as ex
from pyibm.inputfile import read_input_file
from pyibm.io.history import read_history
from pyibm.io.vtk import read_structured_scalar, read_structured_vector, vtk_filename

VARIANTS = (
    ("input2d", "viz/alpha3", "alpha = 3, inflow/outflow"),
    ("input2d_alpha10", "viz/alpha10", "alpha = 10, inflow/outflow"),
    ("input2d_incremental", "viz/alpha3_incremental", "alpha = 3, inflow/outflow, incremental"),
    ("input2d_periodic", "viz/alpha3_periodic", "alpha = 3, periodic, body force"),
    ("input2d_alpha10_periodic", "viz/alpha10_periodic", "alpha = 10, periodic, body force"),
)
WALL_CELLS = 2      # cells next to each wall left out of the "interior" error


def pressure_time(t: float, dt: float, time_scheme: str) -> float:
    """Time level of the pressure written at the output time t: the middle of the step,
    t - dt/2, for both MAC time schemes (module docstring: measured for midpoint, and the
    best of t, t - dt/2 and t - dt for pm1996). Any other scheme raises ValueError."""
    if time_scheme not in ("midpoint", "pm1996"):
        raise ValueError(f"the pressure's time level is known only for time_scheme "
                         f"'midpoint' and 'pm1996', got {time_scheme!r}")
    return t - 0.5 * dt


def last_period_profiles(case_dir: str, input_name: str, output_dir: str | None = None) -> list[dict]:
    """The dumps of the last period of a run: one dict per output time with ``t``, ``y``
    (cell centres), ``u`` (the profile at x = Lx/2), ``dy``, ``H``, ``periodic``,
    ``v_mid`` (largest |v| over the middle half of the channel), ``u_spread_mid``
    (largest variation of u along x over the middle half), ``dpdx`` (mean over y of the
    pressure gradient between x = Lx/4 and 3 Lx/4; None for the periodic runs, where the
    body force stands in for it), ``dt`` and ``t_pressure``, the time level of that pressure
    (``pressure_time``). ``output_dir`` defaults to the input file's Output_Info."""
    params = read_input_file(os.path.join(case_dir, input_name))
    out = params.output.output_dir if output_dir is None else output_dir
    if not os.path.isabs(out):
        out = os.path.join(case_dir, out)
    periodic = params.bc.left == "periodic"
    h = read_history(os.path.join(out, "history.csv"))
    sel = h.t > h.t[-1] - ex.PERIOD + 1e-9 * ex.PERIOD
    rows = []
    for step, t in zip(h.step[sel], h.t[sel]):
        step = int(round(step))
        vel = read_structured_vector(vtk_filename(out, "velocity", step))
        Nx, Ny = vel.fx.shape
        y = vel.origin[1] + vel.spacing[1] * np.arange(Ny)
        mid = slice(Nx // 4, 3 * Nx // 4)
        row = {"t": float(t), "y": y, "dy": vel.spacing[1], "H": params.grid.Ly,
               "dt": params.time.dt,
               "t_pressure": pressure_time(float(t), params.time.dt, params.solver.time_scheme),
               "periodic": periodic, "u": vel.fx[Nx // 2].copy(),
               "v_mid": float(np.max(np.abs(vel.fy[mid]))),
               "u_spread_mid": float(np.max(np.ptp(vel.fx[mid], axis=0))), "dpdx": None}
        if not periodic:
            p = read_structured_scalar(vtk_filename(out, "pressure", step)).field
            x = vel.origin[0] + vel.spacing[0] * np.arange(Nx)
            i0, i1 = Nx // 4, 3 * Nx // 4
            row["dpdx"] = float(np.mean(p[i1] - p[i0]) / (x[i1] - x[i0]))
        rows.append(row)
    return rows


def compare(rows: list[dict]) -> list[dict]:
    """Per phase: the largest |u - u_exact| over the profile and over its interior (the
    WALL_CELLS cells next to each wall left out), both in units of the core amplitude
    K/(rho omega); the flow-rate error relative to the exact amplitude; and the computed
    -dp/dx (None for the periodic runs) next to K cos(omega t_p) at the pressure's own time
    level t_p = row["t_pressure"]."""
    table = []
    for r in rows:
        H, t = r["H"], r["t"]
        ue = ex.velocity(r["y"], t, H)
        e = np.abs(r["u"] - ue) / ex.U_CORE
        q_amp = float(np.max(np.abs(ex.flow_rate(np.linspace(0.0, ex.PERIOD, 721), H))))
        table.append({"t": t, "phase_deg": (ex.OMEGA * t * 180.0 / np.pi) % 360.0,
                      "err": float(e.max()), "err_interior": float(e[WALL_CELLS:-WALL_CELLS].max()),
                      "flux_err": float((np.sum(r["u"]) * r["dy"] - ex.flow_rate(t, H)) / q_amp),
                      "dpdx": None if r["dpdx"] is None else -r["dpdx"],
                      "dpdx_exact": float(ex.pressure_gradient(r["t_pressure"])),
                      "v_mid": r["v_mid"], "u_spread_mid": r["u_spread_mid"]})
    return table


def print_table(label: str, rows: list[dict], table: list[dict]) -> tuple[float, float]:
    H = rows[0]["H"]
    print(f"{label}: H = {H:g} m, alpha = {ex.womersley_number(H):.3g}, "
          f"{H / rows[0]['dy']:.0f} cells across, {len(rows)} phases in the last period")
    print("      t      phase   max|u-u_ex|   interior    flux err    -dp/dx   exact    max|v| mid   u x-spread mid")
    for row in table:
        dp = "    -   " if row["dpdx"] is None else f"{row['dpdx']:+8.4f}"
        print(f"   {row['t']:7.4f}  {row['phase_deg']:5.0f}    {row['err']:.3e}   {row['err_interior']:.3e}"
              f"   {row['flux_err']:+.1e}   {dp}  {row['dpdx_exact']:+8.4f}   {row['v_mid']:.1e}      {row['u_spread_mid']:.1e}")
    worst = max(row["err"] for row in table)
    worst_int = max(row["err_interior"] for row in table)
    print(f"   largest error over the period: {worst:.3e}, interior {worst_int:.3e} "
          f"(units of K/(rho omega) = {ex.U_CORE:g} m/s)")
    if table[0]["dpdx"] is not None:
        lag = rows[0]["t"] - rows[0]["t_pressure"]
        dp_err = max(abs(row["dpdx"] - row["dpdx_exact"]) for row in table)
        print(f"   -dp/dx against K cos(omega (t - {lag:g})), the pressure's own time level: "
              f"largest difference {dp_err:.4f} = {100 * dp_err / ex.K:.2f} % of K")
    return worst, worst_int


def main() -> str | None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    results = {}
    for input_name, out, label in VARIANTS:
        path = os.path.join(HERE, out, "history.csv")
        if not os.path.isfile(path):
            print(f"{input_name}: no {out}/history.csv (run it first)")
            continue
        rows = last_period_profiles(HERE, input_name)
        table = compare(rows)
        results[input_name] = (rows, table, label) + print_table(label, rows, table)
    plotted = [k for k in ("input2d", "input2d_alpha10") if k in results]
    if not plotted:
        return None

    fig = Figure(figsize=(5.0 * len(plotted), 4.5), layout="constrained")
    FigureCanvasAgg(fig)
    axes = np.atleast_1d(fig.subplots(1, len(plotted)))
    wanted = (0.0, 90.0, 180.0, 270.0)
    colours = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd")
    for ax, name in zip(axes, plotted):
        rows, table, label, worst, worst_int = results[name]
        H = rows[0]["H"]
        yy = np.linspace(0.0, H, 400)
        for phase, colour in zip(wanted, colours):
            r = min(rows, key=lambda r: abs(((ex.OMEGA * r["t"] * 180 / np.pi) % 360) - phase))
            ax.plot(ex.velocity(yy, r["t"], H) / ex.U_CORE, yy / H, "-", color=colour, lw=1.5,
                    label=f"omega t = {phase:.0f} deg")
            ax.plot(r["u"] / ex.U_CORE, r["y"] / H, "o", color=colour, ms=3.0, mfc="none")
        ax.set_xlabel("u / (K / rho omega)")
        ax.set_ylabel("y / H")
        ax.set_title(f"{label} (dots), exact (lines); max error {worst:.1e}", fontsize=9)
        ax.legend(fontsize=8)
    out = os.path.join(HERE, "results.png")
    fig.savefig(out, dpi=150)
    print("saved", out)
    return out


if __name__ == "__main__":
    main()
