"""Enclosed area, perimeter and shape of the rubber band against time, delta against blob.

Reads <output_dir>/area.csv (written by hooks.py at every output) and history.csv of the two
variants that have been run (viz/delta, viz/blob), prints the relative area change and the
shape ratio against time for both, and saves results.png: the relative area change, the
shape ratio rmax/rmin, the perimeter, and the band itself at a few times (positions read from
the lag.NNNNN.vtk files).

    python -m pyibm run examples/rubberband
    python -m pyibm run examples/rubberband --input input2d_blob
    python examples/rubberband/analyze.py
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
    ("input2d", "viz/delta", "delta (peskin4)"),
    ("input2d_blob", "viz/blob", "blob (gaussian4, 3 dx)"),
)
SHAPE_TIMES = (0.0, 0.5, 1.0, 2.0)     # s, the band is drawn at the output nearest each


def read_area(path: str) -> np.recarray:
    """area.csv -> record array with step, t, area, perimeter, rmax_over_rmin."""
    return np.genfromtxt(path, delimiter=",", names=True).view(np.recarray)


def shape_at(out_dir: str, a: np.recarray, t: float) -> tuple[float, np.ndarray] | None:
    """(t_output, X) of the output nearest to t, or None if the VTK file is missing."""
    k = int(np.argmin(np.abs(a.t - t)))
    path = os.path.join(out_dir, f"lag.{int(a.step[k]):05d}.vtk")
    if not os.path.isfile(path):
        return None
    return float(a.t[k]), read_lagrangian_points(path)


def main() -> str | None:
    fig = Figure(figsize=(10.0, 7.5), layout="constrained")
    FigureCanvasAgg(fig)
    (ax_area, ax_ratio), (ax_perim, ax_shape) = fig.subplots(2, 2)
    found = False
    for input_name, out, label in VARIANTS:
        out_dir = os.path.join(HERE, out)
        path = os.path.join(out_dir, "area.csv")
        if not os.path.isfile(path):
            print(f"{input_name}: no {out}/area.csv (run it first)")
            continue
        found = True
        a = read_area(path)
        h = read_history(os.path.join(out_dir, "history.csv"))
        rel = a.area / a.area[0] - 1.0
        print(f"{input_name} ({label}): {len(a)} outputs to t = {a.t[-1]:.4g} s")
        print("    t [s]   area/area0 - 1   perimeter [m]   rmax/rmin")
        for k in range(0, len(a), max(1, (len(a) - 1) // 10)):
            print(f"  {a.t[k]:7.3f}   {rel[k]:+13.3e}   {a.perimeter[k]:13.5f}   {a.rmax_over_rmin[k]:9.4f}")
        if (len(a) - 1) % max(1, (len(a) - 1) // 10):
            print(f"  {a.t[-1]:7.3f}   {rel[-1]:+13.3e}   {a.perimeter[-1]:13.5f}   {a.rmax_over_rmin[-1]:9.4f}")
        print(f"  largest |area change| {np.max(np.abs(rel)):.3e}, final {rel[-1]:+.3e}; "
              f"rmax/rmin: 2 -> {a.rmax_over_rmin.min():.4f} at t = {a.t[np.argmin(a.rmax_over_rmin)]:.3g} s "
              f"-> {a.rmax_over_rmin[-1]:.4f} at the end; max speed {h.max_speed.max():.4g} m/s, "
              f"max CFL {h.cfl.max():.3g}, net force on the band |fx| <= {np.max(np.abs(h.fx_body)):.1e} N")
        ax_area.plot(a.t, rel, label=label)
        ax_ratio.plot(a.t, a.rmax_over_rmin, label=label)
        ax_perim.plot(a.t, a.perimeter, label=label)
        if out == "viz/delta":
            for t_want in SHAPE_TIMES:
                got = shape_at(out_dir, a, t_want)
                if got is not None:
                    t_got, X = got
                    Xc = np.vstack([X, X[:1]])
                    ax_shape.plot(Xc[:, 0], Xc[:, 1], label=f"t = {t_got:.3g} s")
    if not found:
        return None
    ax_area.set_ylabel("area / area(0) - 1")
    ax_area.set_xlabel("t [s]")
    ax_area.legend()
    ax_ratio.set_ylabel("rmax / rmin (1 = circle)")
    ax_ratio.set_xlabel("t [s]")
    ax_ratio.legend()
    ax_perim.set_ylabel("perimeter [m]")
    ax_perim.set_xlabel("t [s]")
    ax_perim.legend()
    ax_shape.set_aspect("equal")
    ax_shape.set_xlabel("x [m]")
    ax_shape.set_ylabel("y [m]")
    ax_shape.set_title("the band (delta run)")
    ax_shape.legend(fontsize="small")
    out = os.path.join(HERE, "results.png")
    fig.savefig(out, dpi=150)
    print("saved", out)
    return out


if __name__ == "__main__":
    main()
