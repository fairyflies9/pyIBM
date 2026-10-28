"""Relative area error of the perturbed ellipse against time: blob projection against delta.

Reads <output_dir>/area.csv (written by hooks.py at every output) and history.csv of the
runs that have been done (viz/blob, viz/delta), prints the relative area error at a few
times next to the values read off the figures of Cortez & Minion 2000 (Figs. 4, 7 and 8,
128 x 128 grid with 400 points), and saves results.png: the area error of both runs on one
axis, the blob run on its own scale with the paper's band, the membrane at t = 0
and t = 0.3125 (the times of the paper's Fig. 3, and turned by half a turn about the box
centre the way that figure draws it), read from the lag.NNNNN.vtk files, and the perimeter.

R. Cortez and M. Minion, The blob projection method for immersed boundary problems,
J. Comput. Phys. 161 (2000) 428-453.

    python -m pyibm run examples/blob_ellipse
    python -m pyibm run examples/blob_ellipse --input input2d_delta
    python examples/blob_ellipse/analyze.py
"""

from __future__ import annotations

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
from pyibm.io.vtk import read_lagrangian_points

HERE = os.path.dirname(os.path.abspath(__file__))

VARIANTS = (
    ("input2d", "viz/blob", "blob, stream-function interpolation"),
    ("input2d_delta", "viz/delta", "delta function (peskin4)"),
)
BLOB_RUNS = ("input2d",)
REPORT_TIMES = (0.25, 0.5, 0.75, 1.0)
FIG3_TIME = 0.3125

# Relative volume error at t = 1 read off the paper's figures (128 x 128 grid, 400 points,
# dt = 0.1 h, nu = 0.005; about +-10 % of the printed scale).
PAPER = {
    "blob projection, fourth-order blob, delta = 4h (Fig. 7 top, Fig. 8)": "within +-1e-5 (mostly +-0.5e-5)",
    "blob projection, fourth-order blob, delta = 2h (Fig. 8)": "within +-1.2e-5",
    "immersed boundary, decoupled centred projection (Fig. 4)": "-1.4e-2",
    "immersed boundary, spectral projection (Fig. 4)": "-0.8e-3",
    "immersed boundary, Peskin-Printz projection (Fig. 7 top)": "-1.6e-4",
}


def read_area(path: str) -> np.recarray:
    """area.csv -> record array with step, t, area, rel_area_error, perimeter."""
    return np.genfromtxt(path, delimiter=",", names=True).view(np.recarray)


def error_at(a: np.recarray, t: float) -> float:
    """Relative area error at the output nearest to t."""
    return float(a.rel_area_error[int(np.argmin(np.abs(a.t - t)))])


def half_turn(X: np.ndarray) -> np.ndarray:
    """Positions turned by half a turn about the centre of the unit box, (x, y) -> (1 - x, 1 - y).

    The paper's Fig. 3 shows the membrane this way round (the dent at the lower right). The
    periodic box and the grid are symmetric under the turn, so the run is the same either
    way; we turn the drawing so it can be set beside the figure."""
    return 1.0 - np.asarray(X, dtype=float)


def main() -> str | None:
    fig = Figure(figsize=(10.0, 7.5), layout="constrained")
    FigureCanvasAgg(fig)
    (ax_all, ax_blob), (ax_shape, ax_perim) = fig.subplots(2, 2)
    results = {}
    for input_name, out, label in VARIANTS:
        out_dir = os.path.join(HERE, out)
        path = os.path.join(out_dir, "area.csv")
        if not os.path.isfile(path):
            print(f"{input_name}: no {out}/area.csv (run it first)")
            continue
        a = read_area(path)
        h = read_history(os.path.join(out_dir, "history.csv"))
        results[input_name] = a
        print(f"{input_name} ({label}): {len(a)} outputs to t = {a.t[-1]:.4g} s; "
              f"max speed {h.max_speed.max():.3g} m/s, max CFL {h.cfl.max():.3g}, "
              f"net force |f| <= {max(np.max(np.abs(h.fx_body)), np.max(np.abs(h.fy_body))):.1e} N")
        print("    relative area error at t = " + ", ".join(f"{t:g}: {error_at(a, t):+.3e}" for t in REPORT_TIMES)
              + f"; largest |error| {np.max(np.abs(a.rel_area_error)):.3e}")
        color = f"C{[v[0] for v in VARIANTS].index(input_name)}"
        ax_all.plot(a.t, a.rel_area_error, color=color, label=label)
        ax_perim.plot(a.t, a.perimeter, color=color, label=label)
        if input_name in BLOB_RUNS:
            ax_blob.plot(a.t, a.rel_area_error, "o-", color=color, ms=3, label=label)
        if input_name == "input2d":
            for t_want, style in ((0.0, "k-"), (FIG3_TIME, "C3-")):
                k = int(np.argmin(np.abs(a.t - t_want)))
                vtk = os.path.join(out_dir, f"lag.{int(a.step[k]):05d}.vtk")
                if os.path.isfile(vtk):
                    X = half_turn(read_lagrangian_points(vtk))
                    Xc = np.vstack([X, X[:1]])
                    ax_shape.plot(Xc[:, 0], Xc[:, 1], style, label=f"t = {a.t[k]:.4g} s")
    if not results:
        return None
    print("Cortez & Minion 2000, relative volume error at t = 1 (read off the figures):")
    for what, value in PAPER.items():
        print(f"    {value:>28s}   {what}")
    if "input2d_delta" in results:
        d = results["input2d_delta"]
        for name in BLOB_RUNS:
            if name in results:
                b = results[name]
                print(f"delta over {name}: |error| ratio {abs(error_at(d, 1.0)) / abs(error_at(b, 1.0)):.0f} at t = 1, "
                      f"{np.max(np.abs(d.rel_area_error)) / np.max(np.abs(b.rel_area_error)):.0f} for the largest "
                      "|error| over the run")
        print("    (the paper: blob at least 10 times better than its best IB run, Fig. 7)")
    ax_all.set_ylabel("area / area(0) - 1")
    ax_all.set_xlabel("t [s]")
    ax_all.set_title("both runs")
    ax_all.legend(fontsize="small")
    ax_blob.set_ylabel("area / area(0) - 1")
    ax_blob.set_xlabel("t [s]")
    ax_blob.set_title("the blob run (paper, Figs. 7-8: within +-1e-5)")
    ax_blob.axhspan(-1e-5, 1e-5, color="0.9", label="paper, blob, +-1e-5")
    ax_blob.axhline(0.0, color="0.5", lw=0.5)
    ax_blob.legend(fontsize="small")
    ax_shape.set_aspect("equal")
    ax_shape.set_xlabel("x [m]")
    ax_shape.set_ylabel("y [m]")
    ax_shape.set_title("the membrane (blob run), turned half a turn as in the paper's Fig. 3",
                       fontsize="medium")
    ax_shape.legend(fontsize="small")
    ax_perim.set_ylabel("perimeter [m]")
    ax_perim.set_xlabel("t [s]")
    ax_perim.legend(fontsize="small")
    out = os.path.join(HERE, "results.png")
    fig.savefig(out, dpi=150)
    print("saved", out)
    return out


if __name__ == "__main__":
    main()
