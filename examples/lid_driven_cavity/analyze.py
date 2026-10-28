"""Compare the cavity centrelines with Ghia, Ghia & Shin 1982 and plot the oscillating lid.

For every steady case that has been run (viz/re100, viz/re400, viz/re1000) this reads the
centreline profiles the on_output hook wrote (u_centreline.csv: u at x = 0.5 against y;
v_centreline.csv: v at y = 0.5 against x), interpolates them to the 17 points of Tables I
and II of the paper (ghia1982.py), prints the largest differences and the per-point table,
and checks from history.csv that the run was steady at the end (relative change of the
kinetic energy over the last output interval). results.png shows the profiles against the
tabulated points. For the oscillating lid (viz/oscillating/history.csv) it prints the
kinetic energy maxima of each period and saves results_oscillating.png.

    python -m pyibm run examples/lid_driven_cavity
    python -m pyibm run examples/lid_driven_cavity --input input2d_re400
    python -m pyibm run examples/lid_driven_cavity --input input2d_re1000
    python -m pyibm run examples/lid_driven_cavity --input input2d_oscillating
    python examples/lid_driven_cavity/analyze.py
"""

from __future__ import annotations

import importlib.util
import os
import sys

# Let the script run from a plain clone (no pip install -e .).
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from pyibm.io.history import read_history

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"lid_driven_cavity_{name}", os.path.join(HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ghia = _load("ghia1982")
gen = _load("generate")

STEADY = tuple((name, Re, out) for name, Re, _N, _T, _dump, out in gen.STEADY)


def read_centrelines(out_dir: str):
    """(y, u, x, v) from the two files the hook wrote in out_dir."""
    yu = np.loadtxt(os.path.join(out_dir, "u_centreline.csv"), delimiter=",", skiprows=1)
    xv = np.loadtxt(os.path.join(out_dir, "v_centreline.csv"), delimiter=",", skiprows=1)
    return yu[:, 0], yu[:, 1], xv[:, 0], xv[:, 1]


def steadiness(history) -> float:
    """|d(KE)/dt| / KE over the last output interval, a measure of how steady the end is."""
    if len(history) < 2:
        return np.nan
    dke = history.kinetic_energy[-1] - history.kinetic_energy[-2]
    dt = history.t[-1] - history.t[-2]
    return float(abs(dke / dt) / history.kinetic_energy[-1])


def report(Re: int, out_dir: str, verbose: bool = True) -> dict | None:
    """Comparison of one run with the paper; None when the run is missing."""
    path = os.path.join(out_dir, "u_centreline.csv")
    if not os.path.isfile(path):
        return None
    y, u, x, v = read_centrelines(out_dir)
    c = ghia.compare(y, u, x, v, Re)
    h = read_history(os.path.join(out_dir, "history.csv"))
    c["steadiness"] = steadiness(h)
    c["t_end"] = float(h.t[-1])
    c["history"] = h
    c["profiles"] = (y, u, x, v)
    if verbose:
        print(f"Re = {Re}: t = {h.t[-1]:g}, max |u - Ghia| = {c['max_du']:.4f} on the centreline x = 0.5, "
              f"max |v - Ghia| = {c['max_dv']:.4f} on y = 0.5 "
              f"(every point: {c['max_dv_all']:.4f}), |dKE/dt|/KE at the end = {c['steadiness']:.1e} 1/s")
        print("      y    u_pyibm   u_Ghia     diff   |       x    v_pyibm   v_Ghia     diff")
        mv = ghia.mask(Re, "v")
        for i in range(len(ghia.Y)):
            flag = "" if mv[i] else "  (left out of the comparison)"
            print(f"  {ghia.Y[i]:6.4f}  {c['u'][i]:+8.5f}  {ghia.U[Re][i]:+8.5f}  {c['du'][i]:+7.4f}   |  "
                  f"{ghia.X[i]:6.4f}  {c['v'][i]:+8.5f}  {ghia.V[Re][i]:+8.5f}  {c['dv'][i]:+7.4f}{flag}")
    return c


def save_steady_figure(results: dict, path: str) -> None:
    fig = Figure(figsize=(4.0 * len(results), 7.0), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, len(results), squeeze=False)
    for k, (Re, c) in enumerate(results.items()):
        y, u, x, v = c["profiles"]
        ax = axes[0, k]
        ax.plot(u, y, "-", label=f"pyibm, t = {c['t_end']:g}")
        ax.plot(ghia.U[Re], ghia.Y, "ko", mfc="none", label="Ghia et al. 1982, Table I")
        ax.set_xlabel("u on x = 0.5")
        ax.set_ylabel("y")
        ax.set_title(f"Re = {Re}")
        ax.legend(fontsize=8)
        ax = axes[1, k]
        ax.plot(x, v, "-", label=f"pyibm, t = {c['t_end']:g}")
        m = ghia.mask(Re, "v")
        ax.plot(ghia.X[m], ghia.V[Re][m], "ko", mfc="none", label="Ghia et al. 1982, Table II")
        if not np.all(m):
            ax.plot(ghia.X[~m], ghia.V[Re][~m], "rx", label="Table II, left out of the comparison")
        ax.set_xlabel("x")
        ax.set_ylabel("v on y = 0.5")
        ax.legend(fontsize=8)
    fig.savefig(path, dpi=150)


def oscillating_report(verbose: bool = True, out_dir: str | None = None) -> dict | None:
    """Kinetic energy maxima of each lid period from <out_dir>/history.csv (default
    viz/oscillating, the output folder of input2d_oscillating)."""
    out_dir = os.path.join(HERE, gen.OSC_OUTPUT) if out_dir is None else out_dir
    path = os.path.join(out_dir, "history.csv")
    if not os.path.isfile(path):
        return None
    h = read_history(path)
    hooks = _load(os.path.splitext(gen.OSC_HOOKS)[0])
    period = hooks.PERIOD
    n_periods = int(np.floor(h.t[-1] / period + 1e-9))
    peaks = []
    for k in range(n_periods):
        sel = (h.t >= k * period) & (h.t <= (k + 1) * period + 1e-12)
        peaks.append(float(np.max(h.kinetic_energy[sel])))
    if verbose:
        print(f"oscillating lid: {len(h)} history rows to t = {h.t[-1]:g} s, "
              "kinetic energy maximum per period: " + ", ".join(f"{p:.5f}" for p in peaks))
        if len(peaks) >= 2:
            print(f"  change of the maximum over the last period: {abs(peaks[-1] - peaks[-2]) / peaks[-1]:.2e} (relative)")
    return {"history": h, "peaks": peaks, "period": period, "lid_speed": hooks.lid_speed}


def save_oscillating_figure(r: dict, path: str) -> None:
    h = r["history"]
    fig = Figure(figsize=(8.0, 4.0), layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    ax.plot(h.t, h.kinetic_energy, "-", label="kinetic energy [J]")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("kinetic energy [J]")
    ax2 = ax.twinx()
    t = np.linspace(0.0, h.t[-1], 400)
    ax2.plot(t, r["lid_speed"](t), "k--", lw=0.8, label="lid speed U(t)")
    ax2.set_ylabel("lid speed [m/s]")
    ax.set_title("oscillating lid, U(t) = sin(2 pi t), Re = 100 on the peak speed")
    lines = ax.get_legend_handles_labels()
    lines2 = ax2.get_legend_handles_labels()
    ax.legend(lines[0] + lines2[0], lines[1] + lines2[1], loc="upper left", fontsize=8)
    fig.savefig(path, dpi=150)


def main() -> dict:
    results = {}
    for input_name, Re, out in STEADY:
        c = report(Re, os.path.join(HERE, out))
        if c is None:
            print(f"{input_name} (Re = {Re}): no {out}/u_centreline.csv (run it first)")
        else:
            results[Re] = c
    figs = []
    if results:
        p = os.path.join(HERE, "results.png")
        save_steady_figure(results, p)
        print("saved", p)
        figs.append(p)
    osc = oscillating_report()
    if osc is None:
        print(f"{gen.OSC_INPUT}: no {gen.OSC_OUTPUT}/history.csv (run it first)")
    else:
        p = os.path.join(HERE, "results_oscillating.png")
        save_oscillating_figure(osc, p)
        print("saved", p)
        figs.append(p)
    return {"steady": results, "oscillating": osc, "figures": figs}


if __name__ == "__main__":
    main()
