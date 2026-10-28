"""Force coefficients of the flapping-plate runs against Jones et al. 2015.

Reads <output_dir>/history.csv of every variant that has been run (viz/periodic, viz/mac),
turns the force the fluid puts on the plate into the paper's coefficients and prints, per
cycle, their means next to the values read off the paper's figures, plus the largest target
lag in cells. Saves results.png (C_V, C_H, C_L, C_D and the lag against t/T).

    python -m pyibm run examples/flapping_plate_re64
    python -m pyibm run examples/flapping_plate_re64 --input input2d_mac
    python examples/flapping_plate_re64/analyze.py

Coefficients (paper section 2.2 and Fig. 1b; q = 0.5 rho U_rms^2 c):

    C_V = F_y / q            vertical force, positive up
    C_H = F_x / q            horizontal force, positive in +x
    C_D = -(F . t_hat) / q   drag: the force component against the instantaneous motion of the
                             midchord, t_hat = sign(s_dot) (cos beta, sin beta); undefined (NaN)
                             at the two stroke reversals where the plate does not translate
    C_L = (F . n_hat) / q    lift: the component normal to the stroke plane, n_hat =
                             (-sin beta, cos beta), the normal that points up
    C_T = mean of |F| / q    the "net total force" used in the paper's C_V/C_T (my reading)

F = (fx_body, fy_body) is the force per unit span the fluid exerts on the plate (history.csv).
A history row carries the force of the last step, evaluated at the half step: its time is the
t_force column, dt/2 = T/12800 before the row's t. Every coefficient, its phase in the cycle
and the direction of motion used for C_D are taken at t_force (for a history written before
that column existed, at t - dt/2). The target lag is measured at t.

The raw force oscillates as the plate crosses grid lines (one cell per 0.008 T at mid-stroke;
the usual immersed-boundary artefact of a stiff penalty body), which the 800 rows per cycle
resolve. The cycle means are taken from the raw rows. The extrema and the plotted curves
come from a centred running mean over SMOOTH_CYCLES of a cycle (the raw extrema are reported
next to them), because the paper's curves are smooth and a raw peak is inflated by the
oscillation.
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


def _load_sibling(name: str):
    modname = f"flapping_plate_re64_{name}"
    if modname in sys.modules:
        return sys.modules[modname]
    spec = importlib.util.spec_from_file_location(modname, os.path.join(HERE, name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[modname] = module
    spec.loader.exec_module(module)
    return module


gen = _load_sibling("generate")

VARIANTS = (
    ("input2d", "viz/periodic", "periodic box, FFT"),
    ("input2d_mac", "viz/mac", "no-slip box, MAC"),
)

# Values read off the paper's plots for the RIGID plate in the tilted stroke. This example
# runs at Re 64, so the reference is interpolated between its neighbours: the markers at Re 60
# and 80 of Fig. 11A (cycle-mean C_V against Re, averaged over cycles 3-4) and Fig. 12A
# (C_V / C_T), with the Re 50 and 100 points of Fig. 22B / Fig. 23B (rigid against flexible)
# as a cross-check. Figs. 11-12 are the rigid wing: their tilted-stroke values at Re 10, 50-60
# and 100 (0.75, 1.0, 1.17) sit on the rigid curve of Fig. 22B (0.73, 0.98, 1.15), not the
# flexible one (0.92, 1.22, 1.48). The reading error is about 0.03 on the means (0.01 on the
# ratios) and 0.3 on the peaks; the plots are small. The time series are Fig. 21(b.ii)
# (rigid, black) and the Re 100 curve of Fig. 6 at Re 100, Fig. 21(b.i) at Re 10; the phases
# are fractions of the cycle (downstroke 0-0.5).
REFERENCE = {
    "fig11A_mean_CV": {40: 0.88, 60: 1.00, 80: 1.10, 100: 1.17, 200: 1.32},
    "fig12A_CV_over_CT": {40: 0.232, 60: 0.284, 80: 0.318, 100: 0.344, 200: 0.441},
    "fig22B_mean_CV": {5: 0.92, 10: 0.73, 50: 0.98, 100: 1.15},
    "fig23B_CV_over_CT": {5: 0.113, 10: 0.126, 50: 0.289, 100: 0.353},
    "re100": {"CV_max": 6.2, "CV_max_phase": 0.19, "CV_min": -2.2, "CV_min_phase": 0.53,
              "CV_zero_phase": 0.44, "CV_at_start": 1.0,
              "CH_max": 3.4, "CH_max_phase": 0.10, "CH_min": -3.1, "CH_min_phase": 0.85,
              "CH_at_reversal": -2.9, "CH_rebound": -1.0, "CH_rebound_phase": 0.65},
    "re10": {"CV_max": 10.0, "CV_max_phase": 0.20, "CV_min": -6.5, "CV_min_phase": 0.68,
             "CH_max": 5.2, "CH_max_phase": 0.10, "CH_min": -7.5, "CH_min_phase": 0.56},
}


def interpolate_reference(re: float, table: dict[float, float]) -> float:
    """Linear interpolation in Re between the plotted markers of one of the REFERENCE tables."""
    keys = np.array(sorted(table), dtype=float)
    return float(np.interp(re, keys, [table[k] for k in sorted(table)]))


REFERENCE["mean_CV_re64"] = interpolate_reference(gen.RE, REFERENCE["fig11A_mean_CV"])
REFERENCE["CV_over_CT_re64"] = interpolate_reference(gen.RE, REFERENCE["fig12A_CV_over_CT"])
REFERENCE["mean_CV_re64_fig22B"] = interpolate_reference(gen.RE, REFERENCE["fig22B_mean_CV"])
REFERENCE["CV_over_CT_re64_fig23B"] = interpolate_reference(gen.RE, REFERENCE["fig23B_CV_over_CT"])


SMOOTH_CYCLES = 0.02     # running-mean window for the extrema and the plotted curves, in cycles


def running_mean(a: np.ndarray, n: int) -> np.ndarray:
    """Centred running mean over n rows (n made odd), the ends padded with the end values."""
    n = max(int(n), 1)
    n += 1 - n % 2
    if n == 1:
        return np.array(a, dtype=float)
    pad = n // 2
    ap = np.concatenate([np.full(pad, a[0]), a, np.full(pad, a[-1])])
    return np.convolve(ap, np.ones(n) / n, mode="valid")


def force_time(h, dt: float | None = None) -> np.ndarray:
    """The time level of the force columns of a history: its t_force column, or for a history
    written before that column existed, t - dt/2 (the midpoint scheme of both inputs; row 0,
    the force at the initial positions, stays at 0). dt defaults to the committed inputs'."""
    names = h.dtype.names or ()
    if "t_force" in names:
        return np.asarray(h.t_force, dtype=float)
    dt = gen.time_step() if dt is None else dt
    return np.maximum(np.asarray(h.t, dtype=float) - 0.5 * dt, 0.0)


def coefficients(h, K=None, q: float | None = None, smooth_cycles: float = SMOOTH_CYCLES,
                 dt: float | None = None) -> dict[str, np.ndarray]:
    """The dimensionless coefficients of a history record array (see the module docstring),
    at the force times of force_time(h, dt). CV_smooth and CH_smooth are the running means over
    smooth_cycles of a cycle (the window is rounded to whole rows from the row spacing; one row
    means no smoothing)."""
    K = gen.stroke() if K is None else K
    q = gen.Q if q is None else q
    t = force_time(h, dt)
    fx = np.asarray(h.fx_body, dtype=float)
    fy = np.asarray(h.fy_body, dtype=float)
    tau = t / K.period
    # A row belongs to the cycle of its own time t, so every cycle is the same 800 rows as the
    # output steps; its phase in that cycle is that of its force time (the first row of a cycle
    # sits dt/2 before the cycle starts, phase -1/12800).
    cycle = np.floor(np.asarray(h.t, dtype=float) / K.period + 1e-9).astype(int)
    d = K.motion_direction(t)                       # (len(t), 2), zero at the reversals
    moving = np.abs(K.s_dot(t)) > 1e-9 * K.u_max
    n = K.normal_up
    CD = np.where(moving, -(fx * d[:, 0] + fy * d[:, 1]) / q, np.nan)
    CL = (fx * n[0] + fy * n[1]) / q
    row_spacing = float(np.median(np.diff(tau))) if len(tau) > 1 else 1.0
    window = int(round(smooth_cycles / row_spacing)) if row_spacing > 0 else 1
    return {"t": t, "tau": tau, "phase": tau - cycle, "cycle": cycle,
            "tau_row": np.asarray(h.t, dtype=float) / K.period,
            "CV": fy / q, "CH": fx / q, "CL": CL, "CD": CD,
            "CV_smooth": running_mean(fy / q, window), "CH_smooth": running_mean(fx / q, window),
            "smooth_rows": window,
            "CT": np.hypot(fx, fy) / q, "lag": np.asarray(h.max_target_error_over_dx, dtype=float)}


def cycle_means(c: dict[str, np.ndarray]) -> list[dict[str, float]]:
    """One dict per complete cycle k (rows with k <= t/T < k+1 by the row's own time, a uniform
    sample of the cycle): means of C_V, C_H, C_L, C_D, C_T from the raw rows, the ratio mean
    C_V / mean C_T, the extrema of the smoothed C_V and C_H with the phase of their force time
    (the raw extrema under *_raw), and the largest lag in cells."""
    out = []
    n_cycles = int(np.floor(c["tau_row"][-1] + 1e-9))
    for k in range(n_cycles):
        m = c["cycle"] == k
        if np.count_nonzero(m) < 8:
            continue
        ph = c["phase"][m]
        row = {"cycle": k + 1, "rows": int(np.count_nonzero(m)),
               "CV": float(np.mean(c["CV"][m])), "CH": float(np.mean(c["CH"][m])),
               "CL": float(np.mean(c["CL"][m])), "CD": float(np.nanmean(c["CD"][m])),
               "CT": float(np.mean(c["CT"][m])), "lag": float(np.max(c["lag"][m]))}
        row["CV_over_CT"] = row["CV"] / row["CT"] if row["CT"] > 0 else float("nan")
        for name, key, suffix in (("CV", "CV_smooth", ""), ("CH", "CH_smooth", ""),
                                  ("CV", "CV", "_raw"), ("CH", "CH", "_raw")):
            a = c[key][m]
            i, j = int(np.argmax(a)), int(np.argmin(a))
            row[f"{name}_max{suffix}"], row[f"{name}_max_phase{suffix}"] = float(a[i]), float(ph[i])
            row[f"{name}_min{suffix}"], row[f"{name}_min_phase{suffix}"] = float(a[j]), float(ph[j])
        out.append(row)
    return out


def report(label: str, h, stiffness: float | None = None) -> list[dict[str, float]]:
    """Print the per-cycle table of a run and return the rows."""
    c = coefficients(h)
    rows = cycle_means(c)
    print(f"\n{label}: {len(h)} history rows to t/T = {c['tau_row'][-1]:.3f}"
          + (f", target stiffness {stiffness:g}" if stiffness is not None else "")
          + f", max target lag {c['lag'].max():.4f} cells over the run")
    print(f"  extrema from the running mean over {c['smooth_rows']} rows ({SMOOTH_CYCLES:g} of a cycle); "
          "the raw extrema are in the second line of each cycle")
    print("  cycle   mean CV  mean CH  mean CL  mean CD  mean CT  CV/CT   CV max (phase)   CV min (phase)"
          "   CH max (phase)   CH min (phase)   max lag")
    for r in rows:
        print(f"  {r['cycle']:5d}  {r['CV']:8.3f} {r['CH']:8.3f} {r['CL']:8.3f} {r['CD']:8.3f} {r['CT']:8.3f}"
              f" {r['CV_over_CT']:6.3f}   {r['CV_max']:5.2f} ({r['CV_max_phase']:.2f})   "
              f"{r['CV_min']:6.2f} ({r['CV_min_phase']:.2f})   {r['CH_max']:5.2f} ({r['CH_max_phase']:.2f})   "
              f"{r['CH_min']:6.2f} ({r['CH_min_phase']:.2f})   {r['lag']:.4f}")
        print(f"  {'raw':>5s}  {'':8s} {'':8s} {'':8s} {'':8s} {'':8s} {'':6s}   "
              f"{r['CV_max_raw']:5.2f} ({r['CV_max_phase_raw']:.2f})   "
              f"{r['CV_min_raw']:6.2f} ({r['CV_min_phase_raw']:.2f})   {r['CH_max_raw']:5.2f} ({r['CH_max_phase_raw']:.2f})   "
              f"{r['CH_min_raw']:6.2f} ({r['CH_min_phase_raw']:.2f})")
    R = REFERENCE
    f11, f12 = R["fig11A_mean_CV"], R["fig12A_CV_over_CT"]
    print(f"  paper, rigid plate, values read off the plots and interpolated in Re: cycle-mean "
          f"CV {f11[60]:.2f} at Re 60 and {f11[80]:.2f} at Re 80 (Fig. 11A), so about {R['mean_CV_re64']:.2f} "
          f"at Re {gen.RE:g} (Fig. 22B gives {R['mean_CV_re64_fig22B']:.2f}); CV/CT {f12[60]:.3f} / {f12[80]:.3f} "
          f"(Fig. 12A), about {R['CV_over_CT_re64']:.3f} (Fig. 23B gives {R['CV_over_CT_re64_fig23B']:.3f})")
    r1, r2 = R["re100"], R["re10"]
    print(f"  paper, Fig. 21(b.ii) Re 100: CV max {r1['CV_max']:.1f} at phase {r1['CV_max_phase']:.2f}, "
          f"zero at {r1['CV_zero_phase']:.2f}, min {r1['CV_min']:.1f} at {r1['CV_min_phase']:.2f}; "
          f"CH max {r1['CH_max']:.1f} at {r1['CH_max_phase']:.2f}, {r1['CH_at_reversal']:.1f} at the reversal, "
          f"rebound to {r1['CH_rebound']:.1f} at {r1['CH_rebound_phase']:.2f}, min {r1['CH_min']:.1f} at "
          f"{r1['CH_min_phase']:.2f}")
    print(f"  paper, Fig. 21(b.i)  Re 10:  CV max {r2['CV_max']:.1f} at phase {r2['CV_max_phase']:.2f}, "
          f"min {r2['CV_min']:.1f} at {r2['CV_min_phase']:.2f}; CH max {r2['CH_max']:.1f} at {r2['CH_max_phase']:.2f}, "
          f"min {r2['CH_min']:.1f} at {r2['CH_min_phase']:.2f}")
    return rows


def main() -> str | None:
    from pyibm.structures import read_target, read_vertex

    fig = Figure(figsize=(9.0, 10.0), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(4, 1, sharex=True)
    found = False
    t_end = 0.0
    for input_name, out, label in VARIANTS:
        path = os.path.join(HERE, out, "history.csv")
        if not os.path.isfile(path):
            print(f"{input_name}: no {out}/history.csv (run it first)")
            continue
        found = True
        h = read_history(path)
        X = read_vertex(os.path.join(HERE, "plate.vertex"))
        k = float(np.max(read_target(os.path.join(HERE, "plate.target"), X).k))
        report(f"{input_name} ({label})", h, k)
        c = coefficients(h)
        t_end = max(t_end, c["tau_row"][-1])
        w = c["smooth_rows"]
        axes[0].plot(c["tau"], c["CV"], color="0.8", linewidth=0.5)
        axes[0].plot(c["tau"], c["CV_smooth"], label=f"{label} (running mean over {w} rows; raw in grey)")
        axes[1].plot(c["tau"], c["CH"], color="0.8", linewidth=0.5)
        axes[1].plot(c["tau"], c["CH_smooth"], label=label)
        axes[2].plot(c["tau"], running_mean(c["CL"], w), label=f"C_L, {label}")
        CD = np.where(np.isfinite(c["CD"]), c["CD"], 0.0)
        axes[2].plot(c["tau"], np.where(np.isfinite(c["CD"]), running_mean(CD, w), np.nan), "--",
                     label=f"C_D, {label}")
        axes[3].plot(c["tau_row"], c["lag"], label=label)          # the lag is measured at t
    if not found:
        return None
    R = REFERENCE["re100"]
    axes[0].axhline(R["CV_max"], color="0.6", linestyle=":", label="paper Fig. 21(b.ii), Re 100 rigid: CV max/min")
    axes[0].axhline(R["CV_min"], color="0.6", linestyle=":")
    axes[0].axhline(REFERENCE["mean_CV_re64"], color="0.3", linestyle="-.",
                    label=f"paper Fig. 11A, cycle-mean CV at Re 64 = {REFERENCE['mean_CV_re64']:.2f} (interpolated)")
    axes[1].axhline(R["CH_max"], color="0.6", linestyle=":", label="paper Fig. 21(b.ii), Re 100 rigid: CH max/min")
    axes[1].axhline(R["CH_min"], color="0.6", linestyle=":")
    for ax in axes:
        for x in np.arange(0.0, t_end + 0.5, 0.5):
            ax.axvline(x, color="0.8", linestyle="--", linewidth=0.8)
        ax.axhline(0.0, color="0.5", linewidth=0.8)
        ax.legend(loc="upper right", fontsize=8)
    axes[0].set_ylabel("C_V (vertical)")
    axes[1].set_ylabel("C_H (horizontal)")
    axes[2].set_ylabel("C_L, C_D (motion frame)")
    axes[3].set_ylabel("max target lag [cells]")
    axes[3].set_xlabel("t / T  (downstroke 0-0.5, upstroke 0.5-1 of every cycle)")
    out = os.path.join(HERE, "results.png")
    fig.savefig(out, dpi=150)
    print("saved", out)
    return out


if __name__ == "__main__":
    main()
