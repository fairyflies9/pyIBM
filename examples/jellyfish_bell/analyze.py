"""Forward speed and bell oscillation of the jellyfish runs against Hoover & Miller 2015.

Reads <output_dir>/bell.csv (written by hooks.on_output: the centroid of the bell, the
diameter at the widest cross-section and the force in the first muscle spring) of every
variant that has been run.

Driven runs (input2d, input2d_f*, input2d_fine): for every pulse cycle P (one period 1/f
of the drive) the forward speed is the rise of the centroid over the cycle divided by the
period, V_P = (y(P/f) - y((P-1)/f)) f, in bell diameters per second (L = 1), and the
amplitude of the diameter oscillation is max(D) - min(D) within the cycle. These are the
quantities of the paper's Fig. 7 (V_forward) and Fig. 6 (A_max) at the pulse cycles
P = 5, 10, 15, 20, 25. The reference values below are those figures' values for the
reference force F_Mag^ref (the lower-left panels), read from the plotted lines in the vector
data of the published PDF and mapped through the axis ticks, so they are good to about 0.01.
It also prints the rise of the centroid in each contraction: from each local minimum of the
centroid height to the next local maximum. That is the up-and-down motion within a cycle,
without the drift from cycle to cycle that max - min over a cycle would include.

Free vibration (input2d_freevib): the frequency of free vibration is the inverse of the
time between the first two maxima of the diameter (the paper's definition under Fig. 3,
"the inverse of the first period of contraction and expansion"); the paper recorded 0.86 s^-1
for the reference bell, and Fig. 2 shows the diameter going from 0.9 to about 1.07 and
back to about 0.95 in the first period.

Saves results.png: the centroid height and the diameter against time for every driven run,
the per-cycle forward speed against the driving frequency next to the paper's Fig. 7
values, and the free-vibration diameter trace.

With --calibration <dir> it reads instead the case written by generate.py --calibration
(the bell driven slowly at f = 0.25 s^-1, which FORCE_SCALE is calibrated on) and prints the
swing of the diameter in every cycle next to Fig. 6's 0.186.

    python -m pyibm run examples/jellyfish_bell
    python -m pyibm run examples/jellyfish_bell --input input2d_freevib
    python examples/jellyfish_bell/analyze.py
    python examples/jellyfish_bell/analyze.py --calibration <dir>
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

HERE = os.path.dirname(os.path.abspath(__file__))
F_REF = 0.86                 # s^-1, the paper's recorded free-vibration frequency (Table 1)
LY = 12.0                    # box height, for unwrapping the centroid (generate.LY)

# Fig. 7, F_Mag^ref panel: average forward speed at pulse cycles P = 5 and P = 25,
# and Fig. 6: amplitude of the diameter oscillation at P = 25 (f = 0.25 is the quasi-static
# swing FORCE_SCALE is calibrated on); read from the vector data of the figures.
PAPER_SPEED = {0.5: (0.12, 0.28), 0.75: (0.33, 0.72), 0.86: (0.31, 0.80), 1.0: (0.19, 0.78),
               1.25: (0.06, 0.08)}
PAPER_AMPLITUDE = {0.25: 0.186, 0.5: 0.237, 0.75: 0.289, 0.86: 0.285, 1.0: 0.252, 1.25: 0.139}
CALIBRATION_FREQUENCY = 0.25   # s^-1, generate.CALIBRATION_FREQUENCY

# input, output dir, frequency (None = free vibration), label
VARIANTS = (
    ("input2d", "viz/f0.86", 0.86, "f = 0.86 (resonant), h = L/16"),
    ("input2d_f0.5", "viz/f0.5", 0.5, "f = 0.5"),
    ("input2d_f0.75", "viz/f0.75", 0.75, "f = 0.75"),
    ("input2d_f1", "viz/f1", 1.0, "f = 1.0"),
    ("input2d_f1.25", "viz/f1.25", 1.25, "f = 1.25"),
    ("input2d_fine", "viz/fine", 0.86, "f = 0.86, h = L/32"),
    ("input2d_freevib", "viz/freevib", None, "free vibration"),
)


def read_bell(path: str) -> np.recarray:
    """bell.csv as a record array: step, t, cx, cy, diameter, muscle_force."""
    return np.genfromtxt(path, delimiter=",", names=True).view(np.recarray)


def unwrap_height(cy: np.ndarray, Ly: float = LY) -> np.ndarray:
    """Remove the jumps of one box height when the whole bell crosses the periodic edge."""
    d = np.diff(cy)
    d -= Ly * np.round(d / Ly)
    return cy[0] + np.concatenate([[0.0], np.cumsum(d)])


def per_cycle(b: np.recarray, f: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(P, V_P, A_P): pulse cycle number, forward speed over the cycle and amplitude of
    the diameter oscillation within it, for every complete cycle in the record."""
    T = 1.0 / f
    y = unwrap_height(b.cy)
    n = int(np.floor(b.t[-1] * f + 1e-9))
    P = np.arange(1, n + 1)
    V = np.empty(n)
    A = np.empty(n)
    for k in range(n):
        y0 = np.interp(k * T, b.t, y)
        y1 = np.interp((k + 1) * T, b.t, y)
        V[k] = (y1 - y0) / T
        m = (b.t >= k * T - 1e-9) & (b.t <= (k + 1) * T + 1e-9)
        A[k] = b.diameter[m].max() - b.diameter[m].min()
    return P, V, A


def contraction_rises(b: np.recarray, f: float) -> tuple[np.ndarray, np.ndarray]:
    """(P, rise): the rise of the unwrapped centroid height from each local minimum to the
    next local maximum, and the pulse cycle P in which that minimum falls (the nearest cycle
    start, so a minimum just before t = (P - 1)/f counts for cycle P). The extrema are taken
    over a window of a quarter period either side. A maximum at the last row is left out,
    since the record may stop part way through a rise."""
    y = unwrap_height(b.cy)
    t = b.t
    n = len(y)
    w = max(1, int(round(0.25 / (f * (t[1] - t[0])))))
    is_max = np.array([y[i] >= y[max(0, i - w):i + w + 1].max() for i in range(n)])
    is_min = np.array([y[i] <= y[max(0, i - w):i + w + 1].min() for i in range(n)])
    imax = np.flatnonzero(is_max & (np.arange(n) < n - 1))
    P, rise = [], []
    for i in np.flatnonzero(is_min):
        later = imax[imax > i]
        if len(later):
            P.append(int(round(t[i] * f)) + 1)
            rise.append(y[later[0]] - y[i])
    return np.array(P, dtype=int), np.array(rise)


def free_vibration_frequency(b: np.recarray) -> tuple[float, float, float]:
    """(frequency, first maximum, first minimum after it) of the diameter trace: the
    frequency is 1 / (time between the first two maxima)."""
    d = b.diameter
    t = b.t
    # local maxima and minima over a window of 0.1 s either side
    w = max(1, int(round(0.1 / (t[1] - t[0]))))
    is_max = np.array([d[i] >= d[max(0, i - w):i + w + 1].max() for i in range(len(d))])
    is_min = np.array([d[i] <= d[max(0, i - w):i + w + 1].min() for i in range(len(d))])
    imax = np.flatnonzero(is_max & (np.arange(len(d)) > w))
    imin = np.flatnonzero(is_min & (np.arange(len(d)) > w))
    if len(imax) < 2:
        return float("nan"), float(d.max()), float(d.min())
    i1, i2 = imax[0], imax[1]
    mins = imin[(imin > i1) & (imin < i2)]
    d_min = float(d[mins].min()) if len(mins) else float("nan")
    return float(1.0 / (t[i2] - t[i1])), float(d[i1]), d_min


def calibration(case_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Print and return (P, swing per cycle) of the calibration case in case_dir (written by
    generate.py --calibration and run with python -m pyibm run <case_dir>)."""
    b = read_bell(os.path.join(case_dir, "viz", f"f{CALIBRATION_FREQUENCY:g}", "bell.csv"))
    P, _, A = per_cycle(b, CALIBRATION_FREQUENCY)
    print(f"calibration case {case_dir}: driven at f = {CALIBRATION_FREQUENCY:g} s^-1, "
          f"paper Fig. 6: {PAPER_AMPLITUDE[CALIBRATION_FREQUENCY]:.3f}")
    for p, a in zip(P, A):
        print(f"    cycle {p}: swing of the diameter {a:.4f}")
    return P, A


def main(argv: list[str] | None = None) -> str | None:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "--calibration":
        if len(argv) != 2:
            raise SystemExit("usage: python analyze.py --calibration <dir>")
        calibration(argv[1])
        return None
    fig = Figure(figsize=(10.0, 9.0), layout="constrained")
    FigureCanvasAgg(fig)
    (ax_y, ax_d), (ax_v, ax_free) = fig.subplots(2, 2)
    found = False
    all_V = [np.zeros(1)]
    for input_name, out, f, label in VARIANTS:
        path = os.path.join(HERE, out, "bell.csv")
        if not os.path.isfile(path):
            print(f"{input_name}: no {out}/bell.csv (run it first)")
            continue
        found = True
        b = read_bell(path)
        if f is None:
            freq, d_max, d_min = free_vibration_frequency(b)
            print(f"{input_name}: free vibration from diameter {b.diameter[0]:.3f}: first maximum "
                  f"{d_max:.3f}, next minimum {d_min:.3f}, frequency {freq:.3f} s^-1 "
                  f"(paper {F_REF}, Fig. 2: 0.9 -> 1.07 -> 0.95)")
            ax_free.plot(b.t, b.diameter, label=label)
            continue
        P, V, A = per_cycle(b, f)
        y = unwrap_height(b.cy) - b.cy[0]
        ref = PAPER_SPEED.get(f)
        ref_a = PAPER_AMPLITUDE.get(f)
        print(f"{input_name}: {label}, {len(P)} cycles, swam {y[-1]:+.3f} L in {b.t[-1]:.2f} s")
        print("    cycle:     " + " ".join(f"{p:6d}" for p in P))
        print("    V_forward: " + " ".join(f"{v:6.3f}" for v in V)
              + (f"   paper Fig. 7 at P = 5: {ref[0]:.2f}, P = 25: {ref[1]:.2f}" if ref else ""))
        print("    amplitude: " + " ".join(f"{a:6.3f}" for a in A)
              + (f"   paper Fig. 6 at P = 25: {ref_a:.3f}" if ref_a else ""))
        rise = np.full(len(P), np.nan)
        for p, r in zip(*contraction_rises(b, f)):
            if 1 <= p <= len(P):
                rise[p - 1] = r
        print("    rise:      " + " ".join(f"{r:6.3f}" for r in rise)
              + "   (of the centroid in the contraction, trough to next peak)")
        all_V.append(V)
        ax_y.plot(b.t * f, y, label=label)
        ax_d.plot(b.t * f, b.diameter, label=label)
        ax_v.plot(np.full(len(P), f), V, ".", color="0.6", zorder=2)
        ax_v.plot(f, V[-1], "o", label=f"f = {f:g}, cycle {P[-1]}", zorder=3)
        if ref:
            ax_v.plot([f, f], ref, "k+", ms=10, mew=2, zorder=4)
    if not found:
        return None
    ax_y.set_xlabel("pulse cycles, t f")
    ax_y.set_ylabel("rise of the centroid [L]")
    ax_y.legend(fontsize=7)
    ax_d.set_xlabel("pulse cycles, t f")
    ax_d.set_ylabel("diameter [L]")
    ax_v.axvline(F_REF, color="r", ls=":", label="f_ref = 0.86", zorder=1)
    ax_v.set_xlabel("driving frequency f [s^-1]")
    ax_v.set_ylabel("forward speed per cycle [L/s] (dots: earlier cycles)")
    ax_v.plot([], [], "k+", ms=10, mew=2, label="Hoover & Miller 2015 Fig. 7, P = 5 and 25")
    # Room above the paper's marks (the highest is 0.80) for the legend, so that it does
    # not hide the P = 25 marks at f = 0.75 and 0.86.
    ax_v.set_ylim(min(-0.15, float(np.concatenate(all_V).min()) - 0.05), 1.3)
    ax_v.legend(fontsize=6, loc="upper left")
    ax_free.set_xlabel("t [s]")
    ax_free.set_ylabel("diameter [L], free vibration")
    ax_free.axhline(1.0, color="k", lw=0.5)
    out = os.path.join(HERE, "results.png")
    fig.savefig(out, dpi=150)
    print("saved", out)
    return out


if __name__ == "__main__":
    main()
