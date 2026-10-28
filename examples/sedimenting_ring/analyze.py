"""Settling speed, weight transfer and area conservation of the sedimenting ring.

Reads viz/<name>/ring.csv and history.csv of every variant that has been run (viz/ring,
viz/ring_coarse, and viz/ring_rigid for the rigid reference), prints

- the speed of the fall (mean of -v_c over the last 0.5 s), the Reynolds number on the
  diameter and the drag coefficient the weight implies, C_D = M g / (1/2 rho U^2 2R),
- how far the ring still is from terminal speed: the slope dU/dt of a straight-line fit of
  -v_c over the last 0.5 s (a fit, since point-to-point differences at the 0.05 s output
  spacing are noisy), as dU/dt / U and as (M + rho pi R^2) dU/dt / (M g), the share of the
  weight still spent on accelerating the ring and the fluid inside it (a lower bound: the
  added mass of the fluid outside is left out),
- the vertical force the fluid puts on the ring over its weight, fy_body / (M g), over the
  last 0.5 s. This is not a measure of terminal speed. fy_body is minus the spread tether
  force, and the ghost update makes it M (g r(t) - a_ghost) exactly, with r(t) the gravity
  ramp and a_ghost the acceleration of the ghost masses. The ring's own mass is small next to
  the fluid it carries, so a_ghost stays small and the ratio follows the ramp to within about
  2 % from t = 0.2 s on, while the ring is at 11 % of its speed at t = 3 s. It checks that
  the weight reaches the fluid through the tethers,
- the enclosed area relative to its initial value (largest drift), the perimeter, the
  largest spring stretch and the largest ghost-tether gap in cells,
- for the rigid reference: the drag of a rigid circle of the same radius dragged down the
  same box at the ring's speed at t = 3 s, next to the ring's weight,

and saves results.png: centroid height and speed, force over weight next to the gravity ramp
r(t), area, and the ring's shape at several times (read from the lag.NNNNN.vtk files).

    python -m pyibm run examples/sedimenting_ring
    python -m pyibm run examples/sedimenting_ring --input input2d_rigid
    python examples/sedimenting_ring/analyze.py
"""

from __future__ import annotations

import glob
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
from pyibm.io.vtk import read_lagrangian_points

HERE = os.path.dirname(os.path.abspath(__file__))
WINDOW = 0.5     # s, the speed and its slope are taken over the last WINDOW seconds

VARIANTS = (
    ("input2d", "ring", "viz/ring", "64 x 128"),
    ("input2d_coarse", "ring_coarse", "viz/ring_coarse", "32 x 64"),
)


def load_generate():
    spec = importlib.util.spec_from_file_location("pyibm_example_sedimenting_ring_generate",
                                                  os.path.join(HERE, "generate.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def speed_trend(t: np.ndarray, U: np.ndarray, window: float = WINDOW) -> tuple[float, float]:
    """(slope dU/dt, mean U) of a least-squares straight line through U(t) over the last
    `window` seconds of the record."""
    last = t >= t[-1] - window
    slope = np.polyfit(t[last], U[last], 1)[0]
    return float(slope), float(np.mean(U[last]))


def gravity_ramp(t: np.ndarray, t_ramp: float) -> np.ndarray:
    """The sin^2 ramp of hooks.ramp, vectorised: 0 at t = 0, 1 from t = t_ramp on."""
    tau = np.clip(np.asarray(t, dtype=float) / t_ramp, 0.0, 1.0)
    return np.sin(0.5 * np.pi * tau) ** 2


def summarize(name: str, out: str, gen) -> dict | None:
    rpath = os.path.join(HERE, out, "ring.csv")
    hpath = os.path.join(HERE, out, "history.csv")
    if not (os.path.isfile(rpath) and os.path.isfile(hpath)):
        return None
    r = read_history(rpath)
    h = read_history(hpath)
    M = gen.MASS * 2.0 * np.pi * gen.RADIUS
    Mg = M * gen.GRAVITY
    dUdt, U = speed_trend(r.t, -r.vc)
    D = 2.0 * gen.RADIUS
    M_carried = M + gen.RHO * np.pi * gen.RADIUS ** 2       # the ring plus the fluid inside it
    return dict(r=r, h=h, Mg=Mg, U=U, Re=gen.RHO * U * D / gen.MU,
                dUdt=dUdt, dUdt_over_U=dUdt / U if U > 0 else np.nan,
                accel_over_weight=M_carried * dUdt / Mg,
                CD=Mg / (0.5 * gen.RHO * U * U * D) if U > 0 else np.nan,
                fy_over_Mg=float(np.mean(h.fy_body[h.t >= h.t[-1] - WINDOW])) / Mg,
                area_drift=float(np.max(np.abs(r.area / r.area[0] - 1.0))),
                perimeter_drift=float(np.max(np.abs(r.perimeter / r.perimeter[0] - 1.0))),
                max_stretch=float(r.max_stretch.max()), ghost=float(r.ghost_gap_over_dx.max()),
                bottom=float(r.bottom_gap_over_dx.min()))


def main() -> str | None:
    gen = load_generate()
    fig = Figure(figsize=(11.0, 7.5), layout="constrained")
    FigureCanvasAgg(fig)
    gs = fig.add_gridspec(4, 2, width_ratios=[1.6, 1.0])
    ax_y = fig.add_subplot(gs[0, 0])
    ax_v = fig.add_subplot(gs[1, 0], sharex=ax_y)
    ax_f = fig.add_subplot(gs[2, 0], sharex=ax_y)
    ax_a = fig.add_subplot(gs[3, 0], sharex=ax_y)
    ax_s = fig.add_subplot(gs[:, 1])
    found = False
    for input_name, name, out, label in VARIANTS:
        s = summarize(name, out, gen)
        if s is None:
            print(f"{input_name}: no {out}/ring.csv and history.csv (run it first)")
            continue
        found = True
        r, h = s["r"], s["h"]
        print(f"{input_name} ({label}): {len(r)} rows to t = {r.t[-1]:.4g} s; weight M g = {s['Mg']:.4f} N")
        print(f"    mean speed over the last {WINDOW:g} s: {s['U']:.4f} m/s, Re_D = {s['Re']:.1f}, "
              f"C_D = M g/(rho U^2 R) = {s['CD']:.2f}")
        print(f"    still accelerating: dU/dt = {s['dUdt']:.2e} m/s^2 over the last {WINDOW:g} s "
              f"(straight-line fit), {100 * s['dUdt_over_U']:.1f} % of U per second; "
              f"(M + rho pi R^2) dU/dt = {100 * s['accel_over_weight']:.1f} % of M g")
        print(f"    fy_body / (M g) over the last {WINDOW:g} s: {s['fy_over_Mg']:.4f} (the weight "
              "reaching the fluid through the tethers, not a terminal-speed measure)")
        print(f"    area drift {s['area_drift']:.2e}, perimeter drift {s['perimeter_drift']:.2e}, "
              f"max spring stretch {s['max_stretch']:.2e}, max ghost gap {s['ghost']:.3f} cells, "
              f"lowest point {s['bottom']:.1f} cells above the floor")
        ax_y.plot(r.t, r.yc, label=label)
        ax_v.plot(r.t, -r.vc, label=label)
        t_force = h.t_force if "t_force" in h.dtype.names else h.t   # the force's time level
        ax_f.plot(t_force, h.fy_body / s["Mg"], label=label)
        ax_a.plot(r.t, r.area / r.area[0] - 1.0, label=label)
        if name == "ring":
            files = sorted(glob.glob(os.path.join(HERE, out, "lag.*.vtk")))
            picks = files[:: max(1, len(files) // 6)]
            if files and files[-1] not in picks:
                picks.append(files[-1])
            for f in picks:
                X = read_lagrangian_points(f)
                step = int(os.path.basename(f).split(".")[1])
                ax_s.plot(np.append(X[:, 0], X[0, 0]), np.append(X[:, 1], X[0, 1]), lw=1.0,
                          label=f"step {step}")
    rigid = os.path.join(HERE, "viz/ring_rigid/history.csv")
    if os.path.isfile(rigid):
        hr = read_history(rigid)
        Mg = gen.MASS * 2.0 * np.pi * gen.RADIUS * gen.GRAVITY
        fy = float(np.mean(hr.fy_body[hr.t >= hr.t[-1] - WINDOW]))
        U = gen.U_RIGID
        print(f"input2d_rigid: rigid circle dragged down the same box at {U:g} m/s: fluid force "
              f"{fy:.4f} N over the last {WINDOW:g} s = {fy / Mg:.3f} of the ring's weight, "
              f"C_D = {fy / (0.5 * gen.RHO * U * U * 2 * gen.RADIUS):.2f}; max target error "
              f"{hr.max_target_error_over_dx.max():.2e} cells")
        ax_f.axhline(fy / Mg, color="k", ls="--", lw=0.8, label="rigid circle at U_RIGID")
    if not found:
        return None
    ax_y.set_ylabel("centroid y [m]")
    ax_y.legend()
    ax_v.set_ylabel("speed of the fall -v_c [m/s]")
    tt = np.linspace(0.0, 1.0, 201)
    ax_f.plot(tt, gravity_ramp(tt, gen.T_RAMP), "k:", lw=0.8, label="gravity ramp r(t)")
    ax_f.set_ylabel("fy_body / (M g)")
    ax_f.set_ylim(0.0, 1.5)
    ax_f.legend(fontsize=8)
    ax_a.set_ylabel("area / area(0) - 1")
    ax_a.set_xlabel("t [s]")
    ax_s.set_aspect("equal")
    ax_s.set_xlim(0.0, gen.LX)
    ax_s.set_ylim(0.0, gen.LY)
    ax_s.set_xlabel("x [m]")
    ax_s.set_ylabel("y [m]")
    ax_s.set_title("ring shape (fine run)")
    ax_s.legend(fontsize=7)
    out = os.path.join(HERE, "results.png")
    fig.savefig(out, dpi=150)
    print("saved", out)
    return out


if __name__ == "__main__":
    main()
