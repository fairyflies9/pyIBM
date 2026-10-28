"""Swimming speed of the swimmer against the wave speed, from viz/<name>/swimmer.csv.

For every variant that has been run (viz/swimmer, viz/swimmer_coarse) this prints the
period-by-period mean swimming speed U (minus the centroid's x velocity, which is the mean of
the interpolated point velocities), the ratio U/V to the wave speed V = lambda/T, the RMS
curvature error against the prescribed curvature, the largest spring stretch and the body
length, and saves results.png: the centroid track (unwrapped across the periodic box), the
instantaneous and period-averaged speed, the heading, and the curvature error.

    python -m pyibm run examples/swimmer
    python -m pyibm run examples/swimmer --input input2d_coarse
    python examples/swimmer/analyze.py
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

from pyibm.inputfile import read_input_file
from pyibm.io.history import read_history

HERE = os.path.dirname(os.path.abspath(__file__))

VARIANTS = (
    ("input2d", "swimmer", "viz/swimmer", "192 x 96, dx = 1/64"),
    ("input2d_coarse", "swimmer_coarse", "viz/swimmer_coarse", "96 x 48, dx = 1/32"),
)


def load_generate():
    spec = importlib.util.spec_from_file_location("pyibm_example_swimmer_generate",
                                                  os.path.join(HERE, "generate.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def unwrap_track(x: np.ndarray, Lx: float) -> np.ndarray:
    """Centroid x positions recorded in a periodic box, made continuous (minimum image on the
    increments; the body moves much less than Lx/2 between outputs)."""
    d = np.diff(x)
    d -= Lx * np.round(d / Lx)
    return x[0] + np.concatenate([[0.0], np.cumsum(d)])


def period_means(t: np.ndarray, u: np.ndarray, T: float) -> list[tuple[int, float]]:
    """(period number, mean of u over that period) for every complete period in t."""
    out = []
    for k in range(int(np.floor(t[-1] / T + 1e-9))):
        sel = (t > k * T + 1e-12) & (t <= (k + 1) * T + 1e-12)
        if sel.sum() >= 2:
            out.append((k + 1, float(np.mean(u[sel]))))
    return out


def summarize(name: str, out: str, gen) -> dict | None:
    path = os.path.join(HERE, out, "swimmer.csv")
    if not os.path.isfile(path):
        return None
    h = read_history(path)
    p = read_input_file(os.path.join(HERE, "input2d" if name == "swimmer" else "input2d_coarse"))
    V = gen.WAVELENGTH / gen.PERIOD
    track = unwrap_track(h.xc, p.grid.Lx)
    means = period_means(h.t, -h.uc, gen.PERIOD)
    settled = h.t >= gen.RAMP_PERIODS * gen.PERIOD
    err = h.curv_rms_error[settled] / np.maximum(h.curv_rms_prescribed[settled], 1e-300) if settled.any() else np.array([np.nan])
    return dict(h=h, track=track, means=means, V=V, p=p,
                U_last=means[-1][1] if means else np.nan,
                curv_err=float(np.mean(err)), max_stretch=float(h.max_stretch.max()),
                length=(float(h.length.min()), float(h.length.max())),
                heading=float(np.degrees(np.max(np.abs(h.heading)))))


def main() -> str | None:
    gen = load_generate()
    fig = Figure(figsize=(9.0, 8.0), layout="constrained")
    FigureCanvasAgg(fig)
    ax_x, ax_u, ax_h, ax_c = fig.subplots(4, 1, sharex=True)
    found = False
    for input_name, name, out, label in VARIANTS:
        r = summarize(name, out, gen)
        if r is None:
            print(f"{input_name}: no {out}/swimmer.csv (run it first)")
            continue
        found = True
        h = r["h"]
        print(f"{input_name} ({label}): {len(h)} rows to t = {h.t[-1]:.4g} s, "
              f"wave speed V = {r['V']:g} m/s")
        for k, U in r["means"]:
            print(f"    period {k}: U = {U:+.4f} m/s, U/V = {U / r['V']:+.4f}")
        print(f"    curvature RMS error / prescribed after the ramp: {r['curv_err']:.3f}; "
              f"max spring stretch {r['max_stretch']:.2e}; body length {r['length'][0]:.5f} .. "
              f"{r['length'][1]:.5f} m; max |heading| {r['heading']:.2f} deg")
        ax_x.plot(h.t, r["track"] - r["track"][0], label=label)
        ax_u.plot(h.t, -h.uc, lw=0.8, label=label + " (instantaneous)")
        if r["means"]:
            ks = np.array([k for k, _ in r["means"]], dtype=float)
            Us = np.array([U for _, U in r["means"]])
            ax_u.step(ks * gen.PERIOD, Us, where="pre", lw=2.0, label=label + " (period mean)")
        ax_h.plot(h.t, np.degrees(h.heading), label=label)
        ax_c.plot(h.t, h.curv_rms_error / np.maximum(h.curv_rms_prescribed, 1e-300), label=label)
    if not found:
        return None
    ax_x.set_ylabel("centroid x - x(0) [m]")
    ax_x.legend()
    ax_u.set_ylabel("swimming speed -u_c [m/s]")
    ax_u.axhline(gen.WAVELENGTH / gen.PERIOD, color="k", lw=0.5, ls="--")
    ax_u.legend(fontsize=8)
    ax_h.set_ylabel("heading [deg]")
    ax_c.set_ylabel("curvature error / prescribed")
    ax_c.set_ylim(0.0, 1.0)
    ax_c.set_xlabel("t [s]")
    out = os.path.join(HERE, "results.png")
    fig.savefig(out, dpi=150)
    print("saved", out)
    return out


if __name__ == "__main__":
    main()
