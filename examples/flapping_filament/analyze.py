"""Flapping amplitude and period of the free end against Huang, Shin & Sung 2007, Fig. 13.

Reads <output_dir>/tip.csv (written by hooks.on_output) and history.csv of every variant
that has been run (viz/fine, viz/coarse, viz/clamped), finds the maxima and minima of the
transverse free-end position Y(0, t) = y_tip - y_fixed over the last WINDOW time units
(or from --from T0), and prints the amplitude (half the mean peak-to-trough distance) and
the period (mean spacing of successive maxima and of successive minima) next to the values
read off their Fig. 13. results.png shows Y(0, t) with the paper's settled amplitude as
dotted lines, the filament length error and the streamwise force on the filament.

    python -m pyibm run examples/flapping_filament
    python -m pyibm run examples/flapping_filament --input input2d_coarse
    python examples/flapping_filament/analyze.py [--from T0]

Reference values: the maxima and minima of Y(0, t) in Fig. 13 of the paper (pinned end,
Re 200, mass ratio 1.5, gamma 0.001, Fr 0.5, L 1), digitised from the figure up to t = 38,
where its five curves still lie within a pixel or two of one another (REF_MAXIMA,
REF_MINIMA). paper_stats applies the same amplitude and period formulas to them over the
run's own window, so the run and the paper are compared over the same stretch of time. From
the tables: the first trough is -0.25 at t = 2.0 and the first maximum 0.24 at t = 3.7; the
amplitude is 0.29 over 10 < t < 15 and 0.31 over 10 < t < 20, and settles at 0.33 for
t > 20; the period is 3.27 over t < 17 and 3.22 over the whole table.

How the extrema were read: in the journal's PDF the figure is an 866 x 404 raster with 333
pixels per unit of Y and 12.5 per unit of t, so each crest is only a few pixels wide and the
dark run in the crest column takes in both flanks of the line; the middle of that run sits
inside the curve. Each maximum is the top edge of the 2-pixel line at its crest less half
the line width, each minimum the bottom edge plus half the width, and t is the middle of the
crest. That is good to about 0.005 in Y and 0.08 in t (one pixel). Past t = 20 the five
curves part by a pixel or two at the extrema, and this reading takes the outermost one.

Huang, W.-X., Shin, S. J. & Sung, H. J. (2007). Simulation of flexible filaments in a uniform
flow by the immersed boundary method. J. Comput. Phys. 226, 2206-2228.

The force columns of history.csv are the force of the last step, at the time t_force of the
same row (half a step before t for the midpoint scheme); histories written before that column
existed fall back to t - dt/2. The free-end positions in tip.csv are at t.
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

HERE = os.path.dirname(os.path.abspath(__file__))

# (t, Y(0, t)) of the maxima and minima of the free end in Fig. 13 of the paper, digitised
# from the figure: the outer edge of the drawn line at each crest less half its 2-pixel width
# (the module docstring says why); about 0.005 in Y, 0.08 in t.
REF_MAXIMA = ((3.71, 0.236), (7.02, 0.260), (10.29, 0.284), (13.52, 0.302), (16.71, 0.317),
              (19.94, 0.329), (23.17, 0.332), (26.36, 0.332), (29.56, 0.332), (32.75, 0.332),
              (35.90, 0.332))
REF_MINIMA = ((2.03, -0.254), (5.38, -0.251), (8.70, -0.272), (11.93, -0.293), (15.16, -0.311),
              (18.35, -0.323), (21.58, -0.329), (24.77, -0.329), (27.92, -0.332), (31.15, -0.335),
              (34.30, -0.338), (37.49, -0.338))
WINDOW = 10.0             # time units at the end of the run used for the statistics

VARIANTS = (
    ("input2d", "viz/fine", "192 x 128, 32 segments, pinned"),
    ("input2d_coarse", "viz/coarse", "96 x 64, 16 segments, pinned"),
    ("input2d_clamped", "viz/clamped", "192 x 128, 32 segments, clamped"),
)
VARIANT_DT = {"input2d": 2.5e-4, "input2d_coarse": 8.0e-4, "input2d_clamped": 2.5e-4}   # generate.py


def read_tip(path: str) -> np.recarray:
    """tip.csv -> recarray with the columns of hooks.TIP_COLUMNS."""
    return np.genfromtxt(path, delimiter=",", names=True).view(np.recarray)


def peaks(t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Local maxima of y(t), each refined with the parabola through its three samples.

    Returns (t_peak, y_peak). The end samples are never peaks.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(y) < 3:
        return np.array([]), np.array([])
    i = np.flatnonzero((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:])) + 1
    a, b, c = y[i - 1], y[i], y[i + 1]
    denom = a - 2.0 * b + c
    shift = np.where(denom < 0.0, 0.5 * (a - c) / np.where(denom < 0.0, denom, 1.0), 0.0)
    h = t[i + 1] - t[i]
    t_peak = t[i] + shift * h
    y_peak = b - 0.25 * (a - c) * shift
    return t_peak, y_peak


def flapping_stats(t: np.ndarray, y: np.ndarray, t_from: float) -> dict[str, float]:
    """Amplitude and period of the oscillation y(t) for t >= t_from.

    amplitude = (mean of the maxima - mean of the minima) / 2; period = mean spacing of
    successive maxima and successive minima; also the number of maxima used and the
    spread (max - min) of the maxima as a settling indicator.
    """
    m = t >= t_from
    t, y = t[m], y[m]
    t_max, y_max = peaks(t, y)
    t_min, y_min = peaks(t, -y)
    st = extrema_stats(t_max, y_max, t_min, -y_min)
    st.update({
        "max_spread": float(y_max.max() - y_max.min()) if len(y_max) else np.nan,
        "t_from": float(t_from),
        "t_to": float(t[-1]) if len(t) else np.nan,
    })
    return st


def extrema_stats(t_max, y_max, t_min, y_min) -> dict[str, float]:
    """amplitude = (mean of the maxima - mean of the minima) / 2 and period = mean spacing of
    successive maxima and of successive minima (the formulas of flapping_stats), from lists of
    extrema; NaN where a list is too short."""
    t_max, y_max, t_min, y_min = (np.asarray(a, dtype=float) for a in (t_max, y_max, t_min, y_min))
    gaps = np.concatenate([np.diff(t_max), np.diff(t_min)])
    return {
        "amplitude": 0.5 * (y_max.mean() - y_min.mean()) if len(y_max) and len(y_min) else np.nan,
        "period": float(gaps.mean()) if len(gaps) else np.nan,
        "n_max": int(len(y_max)),
    }


def paper_stats(t_from: float, t_to: float) -> dict[str, float]:
    """Amplitude and period of the paper's Fig. 13 over t_from <= t <= t_to, from the
    digitised extrema (REF_MAXIMA, REF_MINIMA) with the formulas of flapping_stats."""
    mx = np.array([e for e in REF_MAXIMA if t_from <= e[0] <= t_to]).reshape(-1, 2)
    mn = np.array([e for e in REF_MINIMA if t_from <= e[0] <= t_to]).reshape(-1, 2)
    return extrema_stats(mx[:, 0], mx[:, 1], mn[:, 0], mn[:, 1])


REF_PERIOD = paper_stats(0.0, 40.0)["period"]        # 3.22 over the digitised table
REF_AMPLITUDE = paper_stats(20.0, 40.0)["amplitude"]  # 0.33, settled (t > 20)


def force_time(h, dt: float) -> np.ndarray:
    """The time level of the force columns of a history: its t_force column, or for a
    history written before that column existed, t - dt/2 (midpoint scheme; 0 in row 0)."""
    names = h.dtype.names or ()
    if "t_force" in names:
        return np.asarray(h.t_force, dtype=float)
    return np.maximum(np.asarray(h.t, dtype=float) - 0.5 * dt, 0.0)


def analyze_variant(out_dir: str, t_from: float | None = None, dt: float = 2.5e-4
                    ) -> dict[str, float] | None:
    """Statistics of one run, or None when it has not been run. dt is used only for a history
    without a t_force column (force_time)."""
    tip_path = os.path.join(out_dir, "tip.csv")
    if not os.path.isfile(tip_path):
        return None
    tip = read_tip(tip_path)
    t = tip.t
    y = tip.y_tip - tip.y_fixed
    if t_from is None:
        t_from = max(t[-1] - WINDOW, 0.5 * t[-1])
    st = flapping_stats(t, y, t_from)
    st["max_strain"] = float(np.max(tip.max_strain))
    st["length_error"] = float(np.max(np.abs(tip.length - 1.0)))
    hist_path = os.path.join(out_dir, "history.csv")
    if os.path.isfile(hist_path):
        h = read_history(hist_path)
        st["max_target_error_over_dx"] = float(np.max(h.max_target_error_over_dx))
        st["mean_fx"] = float(np.mean(h.fx_body[force_time(h, dt) >= t_from]))
    return st


def main(argv: list[str] | None = None) -> str | None:
    argv = sys.argv[1:] if argv is None else argv
    t_from = None
    if len(argv) == 2 and argv[0] == "--from":
        t_from = float(argv[1])
    elif argv:
        print(__doc__)
        return None

    fig = Figure(figsize=(8.0, 8.0), layout="constrained")
    FigureCanvasAgg(fig)
    ax_y, ax_len, ax_fx = fig.subplots(3, 1, sharex=True)
    found = False
    t_end = 0.0
    print(f"reference (Huang, Shin & Sung 2007, Fig. 13): amplitude {REF_AMPLITUDE:.2f} "
          f"(settled, t > 20), period {REF_PERIOD:.2f}")
    for input_name, out, label in VARIANTS:
        out_dir = os.path.join(HERE, out)
        st = analyze_variant(out_dir, t_from, VARIANT_DT[input_name])
        if st is None:
            print(f"{input_name}: no {out}/tip.csv (run it first)")
            continue
        found = True
        ref = paper_stats(st["t_from"], st["t_to"])
        print(f"{input_name} ({label}): window {st['t_from']:.1f} < t < {st['t_to']:.1f}, "
              f"{st['n_max']} maxima; amplitude {st['amplitude']:.3f} (paper {ref['amplitude']:.3f} over "
              f"the same window), period {st['period']:.2f} (paper {ref['period']:.2f}), "
              f"maxima spread {st['max_spread']:.3f}; "
              f"max strain {st['max_strain']:.3%}, max length error {st['length_error']:.3%}, "
              f"max target error {st.get('max_target_error_over_dx', float('nan')):.3f} cells, "
              f"mean fx {st.get('mean_fx', float('nan')):+.3f} N")
        tip = read_tip(os.path.join(out_dir, "tip.csv"))
        t_end = max(t_end, float(tip.t[-1]))
        ax_y.plot(tip.t, tip.y_tip - tip.y_fixed, label=label)
        ax_len.plot(tip.t, 100.0 * (tip.length - 1.0), label=label)
        hist_path = os.path.join(out_dir, "history.csv")
        if os.path.isfile(hist_path):
            h = read_history(hist_path)
            ax_fx.plot(force_time(h, VARIANT_DT[input_name]), h.fx_body, label=label)
    if not found:
        return None
    ref = np.array(REF_MAXIMA + REF_MINIMA)
    ref = ref[ref[:, 0] <= t_end]
    ax_y.plot(ref[:, 0], ref[:, 1], "ks", mfc="none", ms=4, label="Huang et al. 2007 Fig. 13, extrema")
    ax_y.set_xlim(0.0, t_end)
    for sgn in (1.0, -1.0):
        ax_y.axhline(sgn * REF_AMPLITUDE, color="k", linestyle=":", linewidth=0.8,
                     label=f"+/- {REF_AMPLITUDE:.2f}, their settled amplitude" if sgn > 0 else None)
    ax_y.set_ylabel("Y(0, t), free end")
    ax_y.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7, frameon=False)
    ax_len.set_ylabel("length error [%]")
    ax_fx.set_ylabel("streamwise force on the filament [N]")
    ax_fx.set_xlabel("t")
    path = os.path.join(HERE, "results.png")
    fig.savefig(path, dpi=150)
    print("saved", path)
    return path


if __name__ == "__main__":
    main()
