"""Net flow around the loop: the valveless-pumping runs against Jung & Peskin 2001.

Reads <output_dir>/flux.csv (written by hooks.on_output: the flux through the vertical
section at the middle of the top straight segment and through the horizontal section at
the middle of the right arc, positive = clockwise, plus the area between the walls) and
<output_dir>/history.csv of every variant that has been run, and prints for each

- the time-averaged flux over the last five cycles at both sections (the paper's
  flowmeter, Fig. 3 and Fig. 8; the two sections agree when the area is conserved),
- the running cycle average, so one can see the spin-up (the paper averaged over 150 s,
  we run 20 cycles),
- the range of the flux in the last five cycles (Fig. 8 shows -1.1 to +2.5 cm^2/s at
  T = 0.325 s, -1.1 to +1.35 at 1.34 s, -5.1 to +2.8 at 0.21 s),
- the area between the walls two ways: the largest instantaneous change relative to the
  initial 34.28 cm^2, and the change of the time-averaged area, which is the paper's measure
  (Fig. 4: the time-averaged area minus the initial area, worst case 0.4196 cm^2 = 1.22 % at
  T = 0.2 s over 150 s; Fig. 5 shows the area against time for that case),
- from history.csv: the largest CFL number, speed and target error, and the largest box-mean
  velocity |mean_u|, |mean_v| (remove_mean_force holds it at its starting value, zero),

next to the paper's time-averaged flux: read from the plotted points of Fig. 3 (A0 = 0.6 cm;
+0.60 (right) and +0.61 (top) cm^2/s at T = 0.325 s and -1.61 (right) and -1.59 (top) at
T = 0.21 s, zero crossing between T = 1.325 and 1.34 s), and -0.0236 cm^2/s for the
symmetric drive (section 5.2, quoted exactly). The figure values come from the vector data
of the published PDF, mapped through the axis ticks, so they are good to about 0.01. Saves
results.png: the flowmeter trace over the last five cycles of every variant (like Fig. 8),
the mean flux against the period next to the paper's points (like Fig. 3), and the area
against time.

    python -m pyibm run examples/heart_tube
    python -m pyibm run examples/heart_tube --input input2d_ccw
    python examples/heart_tube/analyze.py
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
AREA_PAPER = 34.2796        # cm^2, the initial area between the walls (under Fig. 4)
N_LAST = 5                  # cycles averaged, as in the paper's Fig. 8

# input, output dir, period [s], label, paper's time-averaged flux [cm^2/s], where it comes from
VARIANTS = (
    ("input2d", "viz/cw", 0.325, "T = 0.325 s, midpoint", +0.6, "Fig. 3"),
    ("input2d_pm1996", "viz/pm1996", 0.325, "T = 0.325 s, pm1996 (the paper's scheme)", +0.6,
     "Fig. 3"),
    ("input2d_ccw", "viz/ccw", 0.21, "T = 0.21 s, midpoint", -1.6, "Fig. 3"),
    ("input2d_ccw_pm1996", "viz/ccw_pm1996", 0.21, "T = 0.21 s, pm1996 (the paper's scheme)", -1.6,
     "Fig. 3"),
    ("input2d_zero", "viz/zero", 1.34, "T = 1.34 s, midpoint", 0.0, "Fig. 3 (zero crossing)"),
    ("input2d_zero_pm1996", "viz/zero_pm1996", 1.34, "T = 1.34 s, pm1996 (the paper's scheme)", 0.0,
     "Fig. 3 (zero crossing)"),
    ("input2d_symmetric", "viz/symmetric", 0.21, "T = 0.21 s, centre third driven", -0.0236,
     "section 5.2"),
    ("input2d_coarse", "viz/coarse", 0.325, "T = 0.325 s, 64 x 32 cells", +0.6,
     "Fig. 3"),
)


def read_flux(path: str) -> np.recarray:
    """flux.csv as a record array with fields step, t, flux_top, flux_right, area."""
    data = np.genfromtxt(path, delimiter=",", names=True)
    return data.view(np.recarray)


def cycle_means(t: np.ndarray, q: np.ndarray, period: float) -> np.ndarray:
    """Mean of q over each whole cycle (n T, (n+1) T]; the row at t = 0 is left out."""
    n_cycles = int(np.floor(t[-1] / period + 1e-9))
    out = np.empty(n_cycles)
    for n in range(n_cycles):
        m = (t > n * period + 1e-9) & (t <= (n + 1) * period + 1e-9)
        out[n] = q[m].mean() if np.any(m) else np.nan
    return out


def summarize(flux: np.recarray, period: float) -> dict[str, float]:
    """The numbers printed for one run (see the module docstring)."""
    t = flux.t
    n_cycles = int(np.floor(t[-1] / period + 1e-9))
    n_last = min(N_LAST, n_cycles)
    m = t > (n_cycles - n_last) * period + 1e-9
    return {
        "cycles": n_cycles,
        "n_last": n_last,
        "mean_top": float(flux.flux_top[m].mean()),
        "mean_right": float(flux.flux_right[m].mean()),
        "flux_min": float(flux.flux_top[m].min()),
        "flux_max": float(flux.flux_top[m].max()),
        "area_change": float(np.max(np.abs(flux.area - flux.area[0])) / flux.area[0]),
        # rows are equally spaced in time, so their mean is the time average (row 0 left out)
        "mean_area_change": float((flux.area[1:].mean() - flux.area[0]) / flux.area[0]),
        "area0": float(flux.area[0]),
    }


def main() -> str | None:
    fig = Figure(figsize=(9.0, 9.0), layout="constrained")
    FigureCanvasAgg(fig)
    ax_meter, ax_period, ax_area = fig.subplots(3, 1)
    found = []
    for input_name, out, period, label, ref, where in VARIANTS:
        path = os.path.join(HERE, out, "flux.csv")
        if not os.path.isfile(path):
            print(f"{input_name}: no {out}/flux.csv (run it first)")
            continue
        flux = read_flux(path)
        hist = read_history(os.path.join(HERE, out, "history.csv"))
        s = summarize(flux, period)
        found.append((period, s["mean_top"], label))
        per_cycle = cycle_means(flux.t, flux.flux_top, period)
        print(f"{input_name}: {label}")
        print(f"    mean flux over the last {s['n_last']} of {s['cycles']} cycles: "
              f"{s['mean_top']:+.3f} cm^2/s (top section), {s['mean_right']:+.3f} (right section); "
              f"paper {ref:+.3f} ({where})")
        print(f"    flux range in those cycles [{s['flux_min']:.2f}, {s['flux_max']:.2f}] cm^2/s; "
              f"cycle means: " + " ".join(f"{v:+.2f}" for v in per_cycle))
        print(f"    area {s['area0']:.3f} cm^2 at rest (paper {AREA_PAPER}), largest instantaneous "
              f"change {100 * s['area_change']:.2f} %, time-averaged area {100 * s['mean_area_change']:+.2f} % "
              f"(paper, Fig. 4: at most 1.22 %)")
        print(f"    max CFL {hist.cfl.max():.2f}, max speed {hist.max_speed.max():.1f} cm/s, max target "
              f"error {hist.max_target_error_over_dx.max():.1f} cells, box mean max |mean_u| "
              f"{np.abs(hist.mean_u).max():.1e}, max |mean_v| {np.abs(hist.mean_v).max():.1e} cm/s")
        m = flux.t > (s["cycles"] - s["n_last"]) * period
        ax_meter.plot((flux.t[m] - flux.t[m][0]) / period, flux.flux_top[m], label=label)
        ax_area.plot(flux.t / period, flux.area, label=label)
    if not found:
        return None
    ax_meter.axhline(0.0, color="k", lw=0.5)
    ax_meter.set_xlabel(f"last {N_LAST} cycles, t / T")
    ax_meter.set_ylabel("flux through the top section [cm^2/s]")
    ax_meter.legend(fontsize=8)
    for period, mean, label in found:
        ax_period.plot(period, mean, "o", label=label)
    # the paper's Fig. 3 values (left-third drive) and, apart, its section 5.2 value for the
    # centre third
    refs = sorted({(p, r) for _, _, p, label, r, _ in VARIANTS if "centre" not in label})
    ax_period.plot([p for p, _ in refs], [r for _, r in refs], "k+", ms=10, mew=2,
                   label="Jung & Peskin 2001, Fig. 3 (A0 = 0.6 cm)")
    sym = [(p, r) for _, _, p, label, r, _ in VARIANTS if "centre" in label]
    ax_period.plot([p for p, _ in sym], [r for _, r in sym], "kx", ms=8, mew=1.5,
                   label="Jung & Peskin 2001, section 5.2 (centre third driven)")
    ax_period.axhline(0.0, color="k", lw=0.5)
    ax_period.set_xlabel("period T [s]")
    ax_period.set_ylabel(f"mean flux, last {N_LAST} cycles [cm^2/s]")
    ax_period.legend(fontsize=8)
    ax_area.axhline(AREA_PAPER, color="k", lw=0.5)
    ax_area.set_xlabel("t / T")
    ax_area.set_ylabel("area between the walls [cm^2]")
    out = os.path.join(HERE, "results.png")
    fig.savefig(out, dpi=150)
    print("saved", out)
    return out


if __name__ == "__main__":
    main()
