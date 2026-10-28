"""Lift and drag coefficients per wing during fling against Miller & Peskin 2005.

Reads <output_dir>/wing_forces.csv (written by hooks.on_output) and history.csv of every
variant that has been run (viz/re128, viz/re8, viz/re128_mac), prints the landmark values
of the force curves next to the values read off the paper's figures, reports the target
stiffness with the largest target error in cells, and saves results.png: C_L and C_D per
wing against the fraction of the half-stroke (tau/6), with the paper's values as markers.

    python -m pyibm run examples/clap_and_fling
    python -m pyibm run examples/clap_and_fling --input input2d_re8
    python -m pyibm run examples/clap_and_fling --input input2d_mac
    python examples/clap_and_fling/analyze.py

The reference values are the landmarks of the curves of Fig. 9 (lift per wing, two-winged
fling, Re 8-128) and Fig. 11 (drag per wing), taken from the vector paths of those curves in
the journal's PDF with the axes calibrated on their tick marks, so they are the plotted
values to better than 0.01 in C and 0.005 in the fraction of the stroke. The Re 128 lift
curve oscillates by about +-0.05 around the rotation peak and from 0.92 to 1.10 over the
last 0.05 of the stroke, so the landmarks read there carry an uncertainty of that size.
"end" is the mean over the last 0.025 of the stroke. Fig. 3 (the paper's refinement study on
a 0.5 m box at Re 128, on a different time axis) gives the same peaks and trough on the mesh
of Figs. 9 and 11 (FIG3, from its vector paths too), and halving the mesh width lowered the
rotation peaks from 5.30 to 4.89 (drag) and 1.85 to 1.69 (lift), so a coarser grid reads
high there.

Miller, L. A. & Peskin, C. S. (2005). A computational fluid dynamics of 'clap and fling' in the
smallest insects. J. Exp. Biol. 208, 195-212.
Miller, L. A. & Peskin, C. S. (2009). Flexible clap and fling in tiny insect flight. J. Exp.
Biol. 212, 3076-3090.

Filtering. The wings sit on their tether springs and ring against the added mass of the
fluid between and around them, with a period 2 pi sqrt(m_added/k): 0.115 s (0.023 of the
half-stroke) at Re 128 with k = 2e5 N/m per m on both back ends (0.36 s with the ten times
softer tethers I started with). Undamped, that ringing had an amplitude of 0.07 in C_L on the
512^2 grid; the target damping of the Re 128 inputs (generate.py) brings it down to about
0.01. The 2009 paper filtered its force coefficients "to remove high frequency noise from the
vibrations of the elastic boundary", and so does this script: a centred moving average over
one ringing period, which also smooths what is left of the wiggle as the wings cross grid
lines. The box is the odd number of rows nearest to the period (box_rows: 11 rows, 0.022 of
the stroke, at the 0.002 row spacing of the inputs), which leaves 4.6 % of a sinusoid of
that period. On curves shaped like the paper's it raises a trough by 2-3 % and lowers a
peak by 0.5-1.6 %: applied to the paper's own curves sampled at the same 0.002 spacing it
takes the Re 128 drag trough from 0.97 to 0.995, the lift trough from 0.838 to 0.858 and
the drag rotation peak from 5.30 to 5.25 (the narrow Re 8 lift trough rises 2.9 %). The
period is set per variant in VARIANTS (as a fraction of the stroke), and
``ringing_amplitude`` reports the amplitude of the line at that period, so the printout
says how much there was to filter. The Re 8 input is not damped and its traces are left
unfiltered; they ring more weakly (about 0.03 in C_L at their tether period, 0.021 of the
stroke), and the README says what the same box would do to them. Raw and filtered
landmarks are both printed; the comparison and the test use the filtered ones.

Time. wing_forces.csv records the force of the last step with its own time t_force (half a
step before the output time for the midpoint scheme), and tau and the fraction of the stroke
are those of t_force. A file written before that column existed is read with t - dt/2.
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
TAU_END = 6.0       # tau at the end of the fling half-stroke (fling_kinematics.TAU_END)

# input, structure, Re, output directory, label, filter window (one tether-ringing period
# as a fraction of the half-stroke, measured from the run; None = no filter).
VARIANTS = (
    ("input2d", "fling_re128", 128, "viz/re128", "Re 128, periodic FFT 512^2", 0.023),
    ("input2d_re8", "fling_re8", 8, "viz/re8", "Re 8, periodic FFT 512^2", None),
    ("input2d_mac", "fling_re128_box", 128, "viz/re128_mac", "Re 128, MAC 512^2", 0.023),
)
FILTER_WINDOW = {v[0]: v[5] for v in VARIANTS}

# Landmarks of the two-winged fling curves, per wing, from the vector paths of the paper's
# figures: (fraction of stroke, value). "rotation peak" is the first maximum (wing rotation),
# "trough" the minimum between the peaks, "acceleration peak" the second maximum
# (translational acceleration), "end" the mean over the last 0.025 of the half-stroke (4.49
# chords of travel).
REFERENCE = {
    128: {
        "lift": {"rotation peak": (0.16, 1.83), "trough": (0.24, 0.84), "acceleration peak": (0.34, 2.57),
                 "minimum during translation": (0.80, 0.78), "end": (1.0, 1.03)},
        "drag": {"rotation peak": (0.08, 5.30), "trough": (0.23, 0.97), "acceleration peak": (0.33, 2.71),
                 "end": (1.0, 1.40)},
        "source": "Fig. 9 (lift) and Fig. 11 (drag), Re = 128 curves",
    },
    8: {
        "lift": {"rotation peak": (0.13, 3.00), "trough": (0.24, 1.02), "acceleration peak": (0.33, 2.99),
                 "end": (1.0, 1.67)},
        "drag": {"rotation peak": (0.076, 24.5), "trough": (0.22, 2.39), "acceleration peak": (0.32, 5.70),
                 "end": (1.0, 3.03)},
        "source": "Fig. 9 (lift) and Fig. 11 (drag), Re = 8 curves",
    },
}
# The paper's refinement study, Fig. 3 (Re 128 on a 0.5 m box; its own time axis), from the
# vector paths: the peak and trough values per wing on the mesh of Figs. 9 and 11 ("coarse",
# 8.33e-4 m) and on half that mesh width ("fine").
FIG3 = {
    "coarse": {"lift": {"rotation peak": 1.85, "trough": 0.85, "acceleration peak": 2.62},
               "drag": {"rotation peak": 5.30, "trough": 0.98, "acceleration peak": 2.73}},
    "fine": {"lift": {"rotation peak": 1.69, "trough": 0.83, "acceleration peak": 2.54},
             "drag": {"rotation peak": 4.89, "trough": 0.99, "acceleration peak": 2.68}},
}
# Search windows (fraction of stroke) for each landmark and whether it is a max or a min.
WINDOWS = {
    "rotation peak": (0.0, 0.2, "max"),
    "trough": (0.15, 0.3, "min"),
    "acceleration peak": (0.25, 0.45, "max"),
    "minimum during translation": (0.45, 1.0, "min"),
    "end": (0.975, 1.0, "end"),     # the mean over the last 0.025 of the stroke
}
# The force rows sit at t_force, half a step before the output times, so a run that reaches the
# end of the stroke has its last row 0.0002 of the stroke short of it; a window counts as
# reached when the run gets within this of its upper end.
REACH = 1e-3


def read_wing_forces(path: str) -> np.recarray:
    """The CSV hooks.on_output writes, as a record array. A file written before the t_force
    column existed (where tau and fraction_of_stroke are those of the output time t) gets a
    t_force of t - dt/2 (midpoint scheme; 0 in row 0, dt from its rows), and tau and
    fraction_of_stroke are recomputed from it."""
    data = np.genfromtxt(path, delimiter=",", names=True).view(np.recarray)
    if "t_force" in data.dtype.names:
        return data
    t, step = np.asarray(data.t, dtype=float), np.asarray(data.step, dtype=float)
    later = step > 0
    dt = float(np.median(t[later] / step[later])) if np.any(later) else 0.0
    t_force = np.maximum(t - 0.5 * dt, 0.0)
    v_over_c = float(np.median(data.tau[later] / t[later])) if np.any(later) else 0.0
    cols = {name: np.asarray(data[name], dtype=float) for name in data.dtype.names}
    cols["t_force"] = t_force
    cols["tau"] = t_force * v_over_c
    cols["fraction_of_stroke"] = cols["tau"] / TAU_END
    names = ["step", "t", "t_force"] + [n for n in data.dtype.names if n not in ("step", "t")]
    return np.rec.fromarrays([cols[n] for n in names], names=names)


def box_rows(frac: np.ndarray, window: float) -> int:
    """Number of rows of the moving average of ``smooth``: the odd count nearest to
    window / (row spacing), 2 floor(x/2) + 1. Its ties sit at even x, so the 0.023 window
    at the inputs' 0.002 spacing (x = 11.5, which floating point gives as 11.4999...)
    lands squarely on 11 rows."""
    step = float(np.median(np.diff(frac)))
    return 2 * int(np.floor(0.5 * window / step)) + 1


def smooth(frac: np.ndarray, y: np.ndarray, window: float | None) -> np.ndarray:
    """Centred moving average of y over box_rows(frac, window) rows (rows are uniform in
    time), with the end values held beyond the ends. None or 0 returns y unchanged. A box of
    exactly one period cancels a sinusoid of that period; at the 0.002 row spacing the
    0.023 window is 11 rows (0.022 of the stroke) and leaves 4.6 % of it."""
    if not window:
        return np.asarray(y, dtype=float)
    y = np.asarray(y, dtype=float)
    n = box_rows(frac, window)
    pad = n // 2
    yp = np.concatenate([np.full(pad, y[0]), y, np.full(pad, y[-1])])
    return np.convolve(yp, np.ones(n) / n, mode="valid")


def ringing_amplitude(d: np.recarray, period: float | None, lo: float = 0.05, hi: float = 0.45) -> float:
    """Amplitude in C_L of the tether ringing at the given period (fraction of stroke):
    the Hann-windowed Fourier line of the left wing's C_L at that period, between lo and
    hi of the stroke, after removing a moving average three periods wide (so the
    stroke-scale features leak little into the line). 0 when period is None or the run is
    too short."""
    if not period:
        return 0.0
    frac, y = d.fraction_of_stroke, np.asarray(d.CL_left, dtype=float)
    m = (frac >= lo) & (frac <= min(hi, float(frac[-1])))
    if np.count_nonzero(m) < 16:
        return 0.0
    r = (y - smooth(frac, y, 3.0 * period))[m]
    w = np.hanning(len(r))
    line = np.sum(r * w * np.exp(-2j * np.pi * frac[m] / period))
    return 2.0 * float(np.abs(line)) / float(np.sum(w))


def landmark(frac: np.ndarray, y: np.ndarray, name: str) -> tuple[float, float] | None:
    """(fraction of stroke, value) of one landmark of a curve sampled at frac, or None when
    the run does not reach the end of the landmark's search window (to within REACH). The
    "end" landmark is (1, the mean of y over its window)."""
    lo, hi, kind = WINDOWS[name]
    if frac[-1] < hi - REACH:
        return None
    if kind == "end":
        return 1.0, float(np.mean(y[frac >= lo]))
    m = (frac >= lo) & (frac <= hi)
    idx = np.flatnonzero(m)
    i = idx[np.argmax(y[m])] if kind == "max" else idx[np.argmin(y[m])]
    return float(frac[i]), float(y[i])


def landmarks(d: np.recarray, re: int, window: float | None = None
              ) -> dict[str, dict[str, tuple[float, float] | None]]:
    """Landmarks of the left-wing C_L and C_D curves (the right wing mirrors them) after
    the moving average of ``smooth`` with the given window (None: raw)."""
    out = {}
    for which, col in (("lift", "CL_left"), ("drag", "CD_left")):
        y = smooth(d.fraction_of_stroke, d[col], window)
        out[which] = {name: landmark(d.fraction_of_stroke, y, name) for name in REFERENCE[re][which]}
    return out


def print_comparison(label: str, d: np.recarray, re: int, window: float | None) -> None:
    """Print the filter window with the ringing amplitude at that period, then the raw and
    filtered landmarks next to the paper's."""
    amp = ringing_amplitude(d, window)
    raw, flt = landmarks(d, re), landmarks(d, re, window)
    if window is None:
        print(f"{label}: no filter (this variant's traces are left raw; README); 'filtered' = raw")
    else:
        print(f"{label}: filter = moving average over one tether-ringing period, {window:.3f} of the stroke "
              f"({window * d.t_force[-1] / d.fraction_of_stroke[-1]:.3f} s); ringing amplitude at that period {amp:.2f} in C_L")
    print(f"  landmarks of the left wing against the paper ({REFERENCE[re]['source']})")
    print(f"  {'quantity':<36s} {'paper (frac, value)':>20s} {'raw (frac, value)':>20s} {'filtered (frac, value)':>24s} {'ratio':>6s}")
    for which in ("lift", "drag"):
        for name, (fr, val) in REFERENCE[re][which].items():
            r, f = raw[which][name], flt[which][name]
            rs = f"{r[0]:8.3f} {r[1]:9.2f}" if r else f"{'n/a':>18s}"
            if f:
                fs, ratio = f"{f[0]:8.3f} {f[1]:9.2f}", f"{f[1] / val:6.2f}"
            else:
                fs, ratio = f"{'n/a':>18s}", "   n/a"
            print(f"  {which + ' ' + name:<36s} {fr:>9.3f} {val:>8.2f}   {rs:>18s}   {fs:>18s} {ratio}")
    asym = np.max(np.abs(d.CL_left - d.CL_right)) + np.max(np.abs(d.CD_left - d.CD_right))
    print(f"  left/right asymmetry (max |dCL| + max |dCD|): {asym:.2e}")


def main() -> str | None:
    fig = Figure(figsize=(11.0, 7.5), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 2)
    found = False
    for input_name, structure, re, out, label, window in VARIANTS:
        fpath = os.path.join(HERE, out, "wing_forces.csv")
        hpath = os.path.join(HERE, out, "history.csv")
        if not (os.path.isfile(fpath) and os.path.isfile(hpath)):
            print(f"{input_name}: no {out}/wing_forces.csv (run it first)")
            continue
        found = True
        d = read_wing_forces(fpath)
        h = read_history(hpath)
        X = read_vertex(os.path.join(HERE, structure + ".vertex"))
        k = float(np.max(read_target(os.path.join(HERE, structure + ".target"), X).k))
        print(f"{input_name} ({label}): stiffness {k:g}, max target error "
              f"{np.max(h.max_target_error_over_dx):.3f} cells, max CFL {np.max(h.cfl):.3f}, "
              f"{len(d)} rows to tau/6 = {d.fraction_of_stroke[-1]:.3f}")
        print_comparison(input_name, d, re, window)
        col = 0 if re == 128 else 1
        ax_l, ax_d = axes[0, col], axes[1, col]
        ls = "--" if "mac" in input_name else "-"
        fr = d.fraction_of_stroke
        tag = "filtered" if window else "raw"
        for ax, cl, cr in ((ax_l, d.CL_left, d.CL_right), (ax_d, d.CD_left, d.CD_right)):
            line, = ax.plot(fr, smooth(fr, cl, window), ls, lw=1.6, label=f"{label}, left wing, {tag}")
            if window:
                ax.plot(fr, cl, "-", lw=0.4, alpha=0.5, color=line.get_color(), label=f"{label}, left wing, raw")
            ax.plot(fr, smooth(fr, cr, window), ":", lw=1.2, color=line.get_color(), label=f"{label}, right wing, {tag}")
    if not found:
        return None
    for col, re in ((0, 128), (1, 8)):
        for row, which in ((0, "lift"), (1, "drag")):
            ax = axes[row, col]
            pts = np.array(list(REFERENCE[re][which].values()))
            ax.plot(pts[:, 0], pts[:, 1], "ks", mfc="none", ms=7,
                    label=f"Miller & Peskin 2005, Fig. {9 if which == 'lift' else 11} (landmarks)")
            ax.set_title(f"Re = {re}: {which} coefficient per wing")
            ax.set_xlabel("fraction of the fling half-stroke (tau/6)")
            ax.set_ylabel(f"C_{'L' if which == 'lift' else 'D'}")
            ax.set_xlim(0.0, 1.0)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=6)
    out = os.path.join(HERE, "results.png")
    fig.savefig(out, dpi=150)
    print("saved", out)
    return out


if __name__ == "__main__":
    main()
