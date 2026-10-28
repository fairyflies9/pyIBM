"""Plot the target error and the force history of the heaving-plate runs.

Reads <output_dir>/history.csv of every variant that has been run (viz/stiff1,
viz/stiff1e4, viz/midpoint), prints the largest target error in cells next to the stiffness, and saves
results.png with the target error and the x force on the plate against time. The force is
plotted at its own time level, the t_force column of history.csv (t - dt for pm1996,
t - dt/2 for midpoint); the target error belongs to the row's t.

    python -m pyibm run examples/heaving_plate_2008
    python -m pyibm run examples/heaving_plate_2008 --input input2d_stiff1e4
    python -m pyibm run examples/heaving_plate_2008 --input input2d_midpoint
    python examples/heaving_plate_2008/analyze.py
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
from pyibm.structures import read_target, read_vertex

HERE = os.path.dirname(os.path.abspath(__file__))

VARIANTS = (
    ("input2d", "plate", "viz/stiff1", "k = 1, pm1996, cosine4"),
    ("input2d_stiff1e4", "plate_stiff1e4", "viz/stiff1e4", "k = 1e4, pm1996, cosine4"),
    ("input2d_midpoint", "plate_stiff1e4", "viz/midpoint", "k = 1e4, midpoint, peskin4"),
)


def main() -> str | None:
    fig = Figure(figsize=(8.0, 6.0), layout="constrained")
    FigureCanvasAgg(fig)
    ax_err, ax_fx = fig.subplots(2, 1, sharex=True)
    found = False
    for input_name, structure, out, label in VARIANTS:
        path = os.path.join(HERE, out, "history.csv")
        if not os.path.isfile(path):
            print(f"{input_name}: no {out}/history.csv (run it first)")
            continue
        h = read_history(path)
        if "t_force" not in h.dtype.names:
            print(f"{input_name}: {out}/history.csv has no t_force column (written by an older "
                  "pyIBM); run it again")
            continue
        found = True
        X = read_vertex(os.path.join(HERE, structure + ".vertex"))
        k = float(np.max(read_target(os.path.join(HERE, structure + ".target"), X).k))
        print(f"{input_name}: stiffness {k:g}, max target error {np.max(h.max_target_error_over_dx):.4g} "
              f"cells, final fx {h.fx_body[-1]:+.4e} N, {len(h)} history rows to t = {h.t[-1]:.5g} s")
        ax_err.plot(h.t, h.max_target_error_over_dx, label=label)
        ax_fx.plot(h.t_force, h.fx_body, label=label)
    if not found:
        return None
    ax_err.set_ylabel("max target error / dx")
    ax_err.legend()
    ax_fx.set_ylabel("x force on the plate [N]")
    ax_fx.set_xlabel("t [s] (the force at its time level t_force)")
    out = os.path.join(HERE, "results.png")
    fig.savefig(out, dpi=150)
    print("saved", out)
    return out


if __name__ == "__main__":
    main()
