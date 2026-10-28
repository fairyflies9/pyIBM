"""Make the pictures in docs/images: the banner, the social preview and one card per example.

    python docs/images/make_images.py                  # run the cases, then draw
    python docs/images/make_images.py --jobs 8         # run the cases in parallel
    python docs/images/make_images.py --draw-only      # draw from frames saved earlier

Step 1 runs each example from its committed input file with ``pyibm.Simulation`` (nothing in
the input is changed; the history and any VTK output go to a scratch folder) and saves the
vorticity, the velocity and the structure at a few chosen times as .npz frames in the work
folder. Step 2 draws the PNG files from those frames. We use one look throughout: a dark
background, a diverging colour map whose zero is the background colour (blue for clockwise,
red to gold for counterclockwise vorticity), and the structure drawn as a thin light line.
The colour limits are fixed per picture and saturate the strongest vorticity on purpose.
README.md in this folder lists what each picture shows.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# ====================================================================== the runs

_T_PLATE = np.pi * 0.25 / 24.0      # flapping plate period, pi A0 / U_max (its generate.py)
_T_HEART = 0.325                    # heart tube drive period of input2d
_F_BELL = 0.86                      # jellyfish drive frequency of input2d_fine

# tag: (example, input file, times at which we save a frame)
CASES = {
    "cyl": ("cylinder_wake", "input2d", [24.0, 26.0, 28.0, 30.0, 32.0]),
    "plate": ("flapping_plate_re64", "input2d", [2 * _T_PLATE + k * _T_PLATE / 8 for k in range(1, 9)]),
    "fling": ("clap_and_fling", "input2d", [tau * 0.05 / 0.06 for tau in (0.5, 1, 1.5, 2, 3, 4, 5, 6)]),
    "jellyfine": ("jellyfish_bell", "input2d_fine", [c / _F_BELL for c in (0.5, 1.0, 1.25, 1.5, 1.75, 2.0)]),
    "fil": ("flapping_filament", "input2d_coarse", [20.0, 22.0, 23.0, 24.0, 25.0, 26.0]),
    "heart": ("heart_tube", "input2d", [6.5 - _T_HEART + k * _T_HEART / 8 for k in range(1, 9)]),
    "swim": ("swimmer", "input2d", [0.0, 1.0, 2.0, 3.0, 3.5, 4.0]),
    "ring": ("sedimenting_ring", "input2d", [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]),
    "cav100": ("lid_driven_cavity", "input2d", [18.0]),
    "cav1000": ("lid_driven_cavity", "input2d_re1000", [70.0]),
    "circle": ("forced_circle", "input2d", [0.0, 0.25, 0.5, 0.75, 1.0]),
    "ellipse": ("blob_ellipse", "input2d", [0.1 * k for k in range(11)]),
    "band": ("rubberband", "input2d", [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]),
    "stokes": ("stokes_second_problem", "input2d", [4.0 + k / 16 for k in range(17)]),
    "wom": ("womersley_channel", "input2d", [3.0 + k / 16 for k in range(17)]),
    "tg": ("taylor_green", "input2d", [0.0, 0.25, 0.5]),
    "heave": ("heaving_plate_2008", "input2d", [0.0, 0.041666666666666675]),
}


def frame_path(work: str, tag: str, k: int) -> str:
    return os.path.join(work, f"{tag}_{k:02d}.npz")


def capture(tag: str, work: str) -> str:
    """Run one case from its committed input and save a frame at each of its times: the
    vorticity as the solver reports it (nodes for the FFT back end, cell centres for MAC),
    the velocity at the same points, and the structure positions."""
    from pyibm.driver import Simulation
    from pyibm.inputfile import read_input_file
    from pyibm.io import vtk as vtkio

    example, input_file, times = CASES[tag]
    case = os.path.join(ROOT, "examples", example)
    params = read_input_file(os.path.join(case, input_file))
    sim = Simulation(params, case, output_dir=os.path.join(work, "runs", tag), quiet=True)
    for k, tc in enumerate(times):
        steps = int(round(tc / sim.dt)) - sim.n
        if steps > 0:
            sim.run(steps)
        elif sim.n == 0 and k == 0:
            sim.output()
        g, sol = sim.grid, sim.solver
        if sim.backend == "periodic_fft":
            loc, (u, v) = "nodes", (np.array(sol.u), np.array(sol.v))
        else:
            loc, (u, v) = "centres", vtkio.mac_to_centres(sol.u, sol.v, g)
        d = dict(t=sim.t, n=sim.n, loc=loc, Lx=g.Lx, Ly=g.Ly, Nx=g.Nx, Ny=g.Ny,
                 omega=np.array(sol.vorticity()), u=np.array(u), v=np.array(v),
                 backend=sim.backend)
        if sim.structure is not None:
            d["X"] = np.array(sim.structure.X)
        np.savez_compressed(frame_path(work, tag, k), **d)
    return tag


# ====================================================================== the look

BG = "#0b0f17"         # background, also the colour of zero vorticity
FG = "#e6edf3"         # text
MUTED = "#9aa4b2"      # secondary text and axes
STRUCT = "#f4f6fa"     # structures
FRAME = "#2a3140"      # panel borders
ACCENT = "#f59a52"     # computed values in the profile plots
TAU = "τ"

# Lightness rises monotonically from the background at zero to pale blue (negative) and pale
# gold (positive).
CMAP = LinearSegmentedColormap.from_list("pyibm_dark", [
    (0.00, "#d6f3ff"), (0.12, "#7cc8f2"), (0.27, "#2f86c9"), (0.40, "#174a7a"), (0.50, BG),
    (0.60, "#7a1f2c"), (0.73, "#c9433a"), (0.88, "#f59a52"), (1.00, "#ffe7b3")], N=512)
# Its warm half, for speeds.
WARM = LinearSegmentedColormap.from_list("pyibm_warm", [
    (0.0, BG), (0.2, "#7a1f2c"), (0.46, "#c9433a"), (0.76, "#f59a52"), (1.0, "#ffe7b3")], N=512)

plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": BG,
                     "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": FG})


class Frames:
    """Loads the saved frames of the work folder."""

    def __init__(self, work: str):
        self.work = work

    def __call__(self, tag: str, k: int) -> dict:
        with np.load(frame_path(self.work, tag, k)) as z:
            return {key: z[key] for key in z.files}


def coords(d):
    """Positions of the field values: nodes i dx (FFT) or centres (i + 1/2) dx (MAC)."""
    Nx, Ny, Lx, Ly = int(d["Nx"]), int(d["Ny"]), float(d["Lx"]), float(d["Ly"])
    off = 0.0 if str(d["loc"]) == "nodes" else 0.5
    return (np.arange(Nx) + off) * Lx / Nx, (np.arange(Ny) + off) * Ly / Ny


def periodic(d) -> bool:
    return str(d["backend"]) == "periodic_fft"


def chains(X, Lx, Ly, wrap, gap=2.5):
    """Split the points into runs of consecutive indices closer than gap times the median
    neighbour distance (so separate bodies and long muscle links are not drawn), unwrap each
    run (minimum image when periodic) and close it when its two ends are that close too."""
    X = np.asarray(X, float)
    L = np.array([Lx, Ly])
    dX = np.diff(X, axis=0)
    if wrap:
        dX -= L * np.round(dX / L)
    dist = np.hypot(dX[:, 0], dX[:, 1])
    h = np.median(dist)
    cut = np.where(dist > gap * h)[0]
    out = []
    for a, b in zip(np.r_[0, cut + 1], np.r_[cut + 1, len(X)]):
        P = X[a] + np.vstack([np.zeros(2), np.cumsum(dX[a:b - 1], axis=0)])
        if b - a > 2:
            e = X[a] - X[b - 1]
            if wrap:
                e -= L * np.round(e / L)
            if np.hypot(*e) < gap * h:
                P = np.vstack([P, P[-1] + e])
        out.append(P)
    return out


def draw_structure(ax, d, lw=1.0, alpha=0.95, dots=0.0, zorder=5):
    """The structure as thin lines (and its periodic images, clipped by the axes)."""
    Lx, Ly, wrap = float(d["Lx"]), float(d["Ly"]), periodic(d)
    shifts = [(i * Lx, j * Ly) for i in (-1, 0, 1) for j in (-1, 0, 1)] if wrap else [(0, 0)]
    segs = [P + np.array(s) for P in chains(d["X"], Lx, Ly, wrap) for s in shifts]
    ax.add_collection(LineCollection(segs, colors=STRUCT, linewidths=lw, alpha=alpha,
                                     zorder=zorder, capstyle="round", joinstyle="round"))
    if dots > 0:
        Q = np.vstack(segs)
        ax.plot(Q[:, 0], Q[:, 1], "o", ms=dots, color=STRUCT, alpha=alpha, mew=0, zorder=zorder)


def draw_field(ax, d, F, vmin, vmax, crop, cmap=CMAP, interpolation="bilinear"):
    """F (Nx, Ny) with each value centred on its point, tiled 3 x 3 when periodic so a crop
    may cross the box edge; values between points are interpolated bilinearly."""
    Lx, Ly = float(d["Lx"]), float(d["Ly"])
    x, y = coords(d)
    dx, dy = x[1] - x[0], y[1] - y[0]
    ext = [x[0] - dx / 2, x[-1] + dx / 2, y[0] - dy / 2, y[-1] + dy / 2]
    if periodic(d):
        F = np.tile(F, (3, 3))
        ext = [ext[0] - Lx, ext[1] + Lx, ext[2] - Ly, ext[3] + Ly]
    ax.imshow(F.T, origin="lower", extent=ext, cmap=cmap, vmin=vmin, vmax=vmax,
              interpolation=interpolation, interpolation_stage="data", aspect="equal")
    ax.set_xlim(crop[0], crop[1])
    ax.set_ylim(crop[2], crop[3])


def vorticity(ax, d, vmax, crop, lw=1.0, structure=True, interpolation="bilinear", dots=0.0):
    draw_field(ax, d, d["omega"], -vmax, vmax, crop, interpolation=interpolation)
    if structure and "X" in d:
        draw_structure(ax, d, lw=lw, dots=dots)


def ghost(ax, d, lw=0.8, alpha=0.32):
    """A faint outline of the structure at another time."""
    draw_structure(ax, d, lw=lw, alpha=alpha, zorder=4)


def bare_axes(fig, rect, W, H):
    l, b, w, h = rect
    ax = fig.add_axes([l / W, b / H, w / W, h / H])
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(FRAME)
        sp.set_linewidth(0.8)
    return ax


def label(ax, text, loc="lower left", size=8.5, pad=(0.03, 0.035)):
    left, lower = "left" in loc, "lower" in loc
    ax.text(pad[0] if left else 1 - pad[0], pad[1] if lower else 1 - pad[1], text,
            transform=ax.transAxes, ha="left" if left else "right",
            va="bottom" if lower else "top", fontsize=size, color=FG, zorder=30,
            bbox=dict(boxstyle="round,pad=0.28", fc=BG, ec="none", alpha=0.7))


def fmt(v):
    a = abs(v)
    if a == 0:
        return "0"
    if a >= 10:
        s = f"{a:.0f}"
    elif a >= 1:
        s = f"{a:.1f}".rstrip("0").rstrip(".")
    elif a >= 0.01:
        s = f"{a:.2g}"
    else:
        m, e = f"{a:.1e}".split("e")
        s = f"{m}e{int(e)}"
    return ("-" if v < 0 else "") + s


def aspect(crop):
    return (crop[1] - crop[0]) / (crop[3] - crop[2])


def crop_around(center, w_px, h_px, span_y):
    """A crop window with the aspect of a w_px x h_px panel, span_y tall, around center."""
    span_x = span_y * w_px / h_px
    return (center[0] - span_x / 2, center[0] + span_x / 2,
            center[1] - span_y / 2, center[1] + span_y / 2)


# ====================================================================== example cards

W_CARD, H_CARD = 640, 400


def card(title, subtitle, rows, vmax, vmin=None, cmap=CMAP, cbar_label="vorticity (1/s)",
         gap=8):
    """A 640 x 400 card: title, subtitle, a block of panels (rows of crop aspects, all the
    same height, centred) and a slim colour bar. Returns the figure and the axes by row."""
    W, H = W_CARD, H_CARD
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    fig.text(14 / W, 1 - 12 / H, title, ha="left", va="top", fontsize=12.5, color=FG,
             fontweight="bold")
    fig.text(14 / W, 1 - 32 / H, subtitle, ha="left", va="top", fontsize=8.6, color=MUTED)
    bl, bb, bw, bh = 14, 12, W - 14 - 62, H - 12 - 54
    hr = min((bw - gap * (len(r) - 1)) / sum(r) for r in rows)
    hr = min(hr, (bh - gap * (len(rows) - 1)) / len(rows))
    width = max(sum(a * hr for a in r) + gap * (len(r) - 1) for r in rows)
    height = len(rows) * hr + gap * (len(rows) - 1)
    x0, y0 = bl + (bw - width) / 2, bb + (bh - height) / 2
    axes = []
    for i, r in enumerate(rows):
        x = x0 + (width - sum(a * hr for a in r) - gap * (len(r) - 1)) / 2
        y = y0 + (len(rows) - 1 - i) * (hr + gap)
        row = []
        for a in r:
            row.append(bare_axes(fig, (x, y, a * hr, hr), W, H))
            x += a * hr + gap
        axes.append(row)
    lo = -vmax if vmin is None else vmin
    cax = fig.add_axes([(x0 + width + 10) / W, (y0 + 0.1 * height) / H, 7 / W, 0.8 * height / H])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(lo, vmax)), cax=cax)
    cb.outline.set_visible(False)
    ticks = [lo, 0.5 * (lo + vmax), vmax]
    cb.set_ticks(ticks)
    cb.set_ticklabels([fmt(t) for t in ticks])
    cb.ax.tick_params(labelsize=7.5, length=2, color=MUTED, labelcolor=MUTED)
    cb.set_label(cbar_label, fontsize=7.5, color=MUTED, labelpad=2)
    return fig, axes


def card_cylinder_wake(F):
    crop, vm = (1.6, 15.6, 0.0, 8.0), 3.0
    fig, axs = card("Flow past a cylinder, Re 100",
                    "von Karman street behind a ring of target points (Lai & Peskin 2000), MAC, input2d",
                    [[aspect(crop)]], vm)
    d = F("cyl", 3)
    vorticity(axs[0][0], d, vm, crop, lw=1.2)
    label(axs[0][0], f"t = {float(d['t']):.0f} D/U")
    return fig


def card_flapping_filament(F):
    crop, vm = (0.0, 6.0, 0.0, 4.0), 6.0
    fig, axs = card("Flapping filament in a uniform stream",
                    "Huang, Shin & Sung 2007, Re 200, coarse grid 96 x 64 (input2d_coarse)",
                    [[aspect(crop)]], vm)
    ax = axs[0][0]
    for k in (1, 2, 3, 4):
        ghost(ax, F("fil", k), lw=0.9, alpha=0.35)
    d = F("fil", 5)
    vorticity(ax, d, vm, crop, lw=1.6)
    label(ax, f"t = {float(d['t']):.0f}; faint: t = 22 to 25")
    return fig


def card_heart_tube(F):
    crop, vm = (0.4, 15.6, 0.4, 7.6), 12.0
    fig, axs = card("Valveless pumping in a heart tube",
                    "Jung & Peskin 2001, pinched near one end of the soft segment, T = 0.325 s (input2d)",
                    [[aspect(crop)]], vm, vmin=0.0, cmap=WARM, cbar_label="speed (cm/s)")
    ax = axs[0][0]
    d = F("heart", 7)
    draw_field(ax, d, np.hypot(d["u"], d["v"]), 0.0, vm, crop, cmap=WARM)
    draw_structure(ax, d, lw=0.9)
    label(ax, f"speed at t = {float(d['t']):.2f} s, the end of cycle 20")
    return fig


def card_jellyfish_bell(F):
    crop, vm = (0.9, 3.1, 0.95, 3.15), 4.0
    fig, axs = card("Jellyfish bell driven at its natural frequency",
                    "Hoover & Miller 2015, f = 0.86 1/s, h = L/32, second cycle (input2d_fine)",
                    [[aspect(crop)] * 2], vm)
    for ax, k, what in zip(axs[0], [2, 4], ["contracted", "expanded"]):
        d = F("jellyfine", k)
        vorticity(ax, d, vm, crop, lw=1.3)
        label(ax, f"{what}, t = {float(d['t']):.2f} s")
    return fig


def card_sedimenting_ring(F):
    crop, vm = (0.0, 0.5, 0.0, 1.0), 6.0
    fig, axs = card("Sedimenting elastic ring",
                    "mass points under gravity in a closed no-slip box, MAC back end (input2d)",
                    [[aspect(crop)] * 3], vm)
    for ax, k in zip(axs[0], [2, 4, 6]):
        d = F("ring", k)
        vorticity(ax, d, vm, crop, lw=1.1)
        ghost(ax, F("ring", 0), alpha=0.3)
        label(ax, f"t = {float(d['t']):.0f} s")
    return fig


def card_flapping_plate_re64(F):
    crop, vm = (0.33, 0.93, 0.08, 0.86), 600.0
    fig, axs = card("Flapping plate, Re 64",
                    "tilted hybrid stroke of Jones et al. 2015, third cycle, periodic box (input2d)",
                    [[aspect(crop)] * 2], vm)
    a, b = F("plate", 3)["X"].mean(axis=0), F("plate", 7)["X"].mean(axis=0)   # stroke ends
    for ax, k, what in zip(axs[0], [1, 5], ["mid-downstroke", "mid-upstroke"]):
        vorticity(ax, F("plate", k), vm, crop, lw=1.6)
        ax.plot([a[0], b[0]], [a[1], b[1]], ls=(0, (2, 3)), color=STRUCT, lw=0.8, alpha=0.5,
                zorder=4)
        label(ax, what)
    return fig


def card_clap_and_fling(F):
    crop, vm = (0.2, 0.8, 0.43, 0.61), 6.0
    fig, axs = card("Clap and fling, Re 128",
                    "two wings fling apart (Miller & Peskin 2005), 512 x 512 periodic box (input2d)",
                    [[aspect(crop)], [aspect(crop)]], vm, gap=6)
    for row, k in zip(axs, [3, 7]):
        d = F("fling", k)
        vorticity(row[0], d, vm, crop, lw=1.4)
        label(row[0], f"{TAU} = {float(d['t']) * 0.06 / 0.05:.0f}", loc="upper left")
    return fig


def card_swimmer(F):
    crop, vm = (0.85, 2.95, 0.3, 1.2), 8.0
    fig, axs = card("Self-propelled swimmer",
                    "travelling-wave curvature through update_beams, periodic box (input2d)",
                    [[aspect(crop)]], vm)
    vorticity(axs[0][0], F("swim", 5), vm, crop, lw=1.6)
    label(axs[0][0], "t = 4 s, swimming to the left")
    return fig


def _two_times(F, tag, ks, title, subtitle, crop, vm, lw=1.3):
    fig, axs = card(title, subtitle, [[aspect(crop)] * 2], vm)
    for ax, k in zip(axs[0], ks):
        d = F(tag, k)
        vorticity(ax, d, vm, crop, lw=lw)
        ghost(ax, F(tag, 0), alpha=0.35)
        label(ax, f"t = {float(d['t']):g} s")
    return fig


def card_rubberband(F):
    return _two_times(F, "band", [2, 6], "Rubber band",
                      "an elliptical band relaxes towards a circle, delta coupling (input2d)",
                      (0.15, 0.85, 0.15, 0.85), 1.0)


def card_blob_ellipse(F):
    return _two_times(F, "ellipse", [2, 6], "Perturbed ellipse, blob projection",
                      "Cortez & Minion 2000 sec. 4.1, gaussian4 blob, 128 x 128 (input2d)",
                      (0.15, 0.85, 0.15, 0.85), 6.0)


def card_forced_circle(F):
    return _two_times(F, "circle", [2, 4], "Forced circle",
                      "Fovargue 2012 sec. 5.4, delta coupling, 128 x 128, nu = 1e-4 (input2d)",
                      (0.12, 0.88, 0.12, 0.88), 40.0, lw=1.1)


def _streamlines(ax, d, density, lw):
    x, y = coords(d)
    ax.streamplot(x, y, d["u"].T, d["v"].T, color=STRUCT, linewidth=lw, density=density,
                  arrowsize=0.55)
    ax.set_xlim(0, float(d["Lx"]))
    ax.set_ylim(0, float(d["Ly"]))


def card_taylor_green(F):
    d0, d1 = F("tg", 0), F("tg", 2)
    vm = float(np.abs(d0["omega"]).max())
    fig, axs = card("Taylor-Green vortex",
                    "decaying vortex array, 64 x 64 periodic box, FFT back end (input2d)",
                    [[1.0, 1.0]], vm)
    for ax, d in zip(axs[0], [d0, d1]):
        vorticity(ax, d, vm, (0, 1, 0, 1), structure=False)
        _streamlines(ax, d, 0.9, 0.5)
        label(ax, f"t = {float(d['t']):g} s")
    return fig


def card_heaving_plate_2008(F):
    d = F("heave", 1)
    vm = float(np.abs(d["omega"]).max())
    fig, axs = card("Heaving plate, the 2008 MATLAB case",
                    "16 target points on a 16 x 16 periodic grid, step 100 (input2d)", [[1.0]], vm)
    vorticity(axs[0][0], d, vm, (0, 0.4, 0, 0.4), lw=1.4, dots=3.0, interpolation="nearest")
    label(axs[0][0], f"t = {float(d['t']):.4f} s, one square per grid node")
    return fig


def card_lid_driven_cavity(F):
    vm = 5.0
    fig, axs = card("Lid-driven cavity",
                    "vorticity and streamlines, MAC back end, t = 18 s (input2d) and 70 s (input2d_re1000)",
                    [[1.0, 1.0]], vm)
    for ax, tag, what in zip(axs[0], ["cav100", "cav1000"], ["Re 100, 64 x 64", "Re 1000, 128 x 128"]):
        d = F(tag, 0)
        vorticity(ax, d, vm, (0, 1, 0, 1), structure=False)
        _streamlines(ax, d, 1.1, 0.45)
        for sp in ax.spines.values():
            sp.set_color(MUTED)
            sp.set_linewidth(1.0)
        ax.spines["top"].set_color("#ffd38a")     # the moving lid
        ax.spines["top"].set_linewidth(2.0)
        label(ax, what)
    return fig


def _profile_card(title, subtitle):
    W, H = W_CARD, H_CARD
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    fig.text(14 / W, 1 - 12 / H, title, ha="left", va="top", fontsize=12.5, color=FG,
             fontweight="bold")
    fig.text(14 / W, 1 - 32 / H, subtitle, ha="left", va="top", fontsize=8.6, color=MUTED)
    ax = fig.add_axes([62 / W, 46 / H, (W - 62 - 168) / W, (H - 46 - 64) / H])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)
        ax.spines[sp].set_linewidth(0.8)
    ax.tick_params(labelsize=8, color=MUTED, labelcolor=MUTED, length=3)
    ax.axvline(0, color=FRAME, lw=0.8, zorder=0)
    return fig, ax


def _profile_legend(fig, exact, computed):
    W, H = W_CARD, H_CARD
    h1, = plt.plot([], [], "-", color=STRUCT, lw=1.0, alpha=0.8)
    h2, = plt.plot([], [], "o", color=ACCENT, ms=3.6, mew=0)
    fig.legend([h1, h2], [exact, computed], loc="upper left",
               bbox_to_anchor=((W - 158) / W, (H - 70) / H), frameon=False, fontsize=8,
               labelcolor=FG, handlelength=1.6)


def _example_module(example, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "examples", example,
                                                                     name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def card_stokes_second_problem(F):
    ex = _example_module("stokes_second_problem", "stokes_exact")
    fig, ax = _profile_card("Stokes' second problem",
                            "oscillating wall, 64 cells over 10 d, eight phases of the fifth period (input2d)")
    yy = np.linspace(0, ex.LY, 400)
    for k in range(0, 16, 2):                   # t = 4, 4 1/8, ..., 4 7/8
        d = F("stokes", k)
        t = float(d["t"])
        _, y = coords(d)
        u = d["u"].mean(axis=0)                 # the 8 columns of the periodic direction
        ax.plot(ex.velocity_finite_depth(yy, t) / ex.U, yy / ex.D, "-", color=STRUCT, lw=0.9,
                alpha=0.75)
        m = y <= 5.0 * ex.D
        ax.plot(u[m] / ex.U, y[m] / ex.D, "o", color=ACCENT, ms=3.2, mew=0, zorder=5)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(0, 5)
    ax.set_xlabel("u / U", fontsize=9, color=FG)
    ax.set_ylabel("y / d", fontsize=9, color=FG)
    _profile_legend(fig, "exact, wall at rest\nat y = 10 d", "pyIBM, cell\ncentres")
    return fig


def card_womersley_channel(F):
    ex = _example_module("womersley_channel", "womersley_exact")
    fig, ax = _profile_card("Womersley flow in a channel",
                            "oscillating pressure gradient, alpha = 3, 24 cells across, inflow and outflow (input2d)")
    H = float(F("wom", 0)["Ly"])
    yy = np.linspace(0, H, 300)
    for k in range(0, 16, 2):                   # t = 3, 3 1/8, ..., 3 7/8
        d = F("wom", k)
        _, y = coords(d)
        i = int(d["Nx"]) // 2
        u = 0.5 * (d["u"][i - 1] + d["u"][i])   # the two cell columns either side of Lx/2
        ax.plot(ex.velocity(yy, float(d["t"]), H), yy / H, "-", color=STRUCT, lw=0.9, alpha=0.75)
        ax.plot(u, y / H, "o", color=ACCENT, ms=3.2, mew=0, zorder=5)
    ax.set_ylim(0, 1)
    ax.set_xlabel("u (m/s)", fontsize=9, color=FG)
    ax.set_ylabel("y / H", fontsize=9, color=FG)
    _profile_legend(fig, "exact solution", "pyIBM, cells\nnext to x = Lx/2")
    return fig


# examples/<name> -> its card, saved as examples/<name>.png
CARDS = {
    "blob_ellipse": card_blob_ellipse,
    "clap_and_fling": card_clap_and_fling,
    "cylinder_wake": card_cylinder_wake,
    "flapping_filament": card_flapping_filament,
    "flapping_plate_re64": card_flapping_plate_re64,
    "forced_circle": card_forced_circle,
    "heart_tube": card_heart_tube,
    "heaving_plate_2008": card_heaving_plate_2008,
    "jellyfish_bell": card_jellyfish_bell,
    "lid_driven_cavity": card_lid_driven_cavity,
    "rubberband": card_rubberband,
    "sedimenting_ring": card_sedimenting_ring,
    "stokes_second_problem": card_stokes_second_problem,
    "swimmer": card_swimmer,
    "taylor_green": card_taylor_green,
    "womersley_channel": card_womersley_channel,
}


# ====================================================================== banner and preview

# key: (tag, frame, vmax, caption, line width)
PANELS = {
    "cyl": ("cyl", 3, 3.0, "Cylinder wake, Re 100", 3.0),
    "fling": ("fling", 7, 6.0, "Clap and fling, Re 128", 3.2),
    "plate": ("plate", 5, 900.0, "Flapping plate, Re 64", 3.6),
    "jelly": ("jellyfine", 4, 9.0, "Jellyfish bell", 3.0),
}


def banner_panel(F, fig, rect, W, H, key, crop, size=12):
    tag, k, vm, caption, lw = PANELS[key]
    ax = bare_axes(fig, rect, W, H)
    vorticity(ax, F(tag, k), vm, crop, lw=lw)
    label(ax, caption, size=size, pad=(10 / rect[2], 10 / rect[3]))
    return ax


def hero(F, path, W=1800, H=640, m=22, gap=10, left_w=1010):
    """Cylinder wake over clap and fling on the left, the plate and the bell on the right."""
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    hh = H - 2 * m
    ph = (hh - gap) / 2
    banner_panel(F, fig, (m, m + ph + gap, left_w, ph), W, H, "cyl",
                 crop_around((8.9, 4.0), left_w, ph, 4.1))
    banner_panel(F, fig, (m, m, left_w, ph), W, H, "fling",
                 crop_around((0.5, 0.52), left_w, ph, 0.62 * ph / left_w))
    rw = (W - 2 * m - left_w - 2 * gap) / 2
    x = m + left_w + gap
    banner_panel(F, fig, (x, m, rw, hh), W, H, "plate", crop_around((0.62, 0.47), rw, hh, 0.80))
    banner_panel(F, fig, (x + rw + gap, m, rw, hh), W, H, "jelly",
                 crop_around((2.0, 2.0), rw, hh, 2.45))
    fig.savefig(path, dpi=100)
    plt.close(fig)


def social_preview(F, path, W=1280, H=640, m=40, gap=10, top=200, rw=240):
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    fig.text(m / W, 1 - 52 / H, "pyIBM", ha="left", va="top", fontsize=64, color=FG,
             fontweight="bold")
    fig.text((m + 4) / W, 1 - 150 / H, "2D immersed boundary methods in Python", ha="left",
             va="top", fontsize=25, color=MUTED)
    hh = H - top - m
    lw_ = W - 2 * m - 2 * rw - 2 * gap
    banner_panel(F, fig, (m, m, lw_, hh), W, H, "cyl", crop_around((6.9, 4.0), lw_, hh, 5.0),
                 size=14)
    banner_panel(F, fig, (m + lw_ + gap, m, rw, hh), W, H, "plate",
                 crop_around((0.62, 0.47), rw, hh, 0.80), size=14)
    banner_panel(F, fig, (m + lw_ + rw + 2 * gap, m, rw, hh), W, H, "jelly",
                 crop_around((2.0, 2.0), rw, hh, 2.6), size=14)
    fig.savefig(path, dpi=100)
    plt.close(fig)


# ====================================================================== main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--work", default=os.path.join(tempfile.gettempdir(), "pyibm-images"),
                    help="folder for the saved frames and the run output")
    ap.add_argument("--out", default=HERE, help="folder for the PNG files (default: this one)")
    ap.add_argument("--jobs", type=int, default=1, help="cases to run at the same time")
    ap.add_argument("--draw-only", action="store_true", help="skip the runs, draw from --work")
    ap.add_argument("--only", nargs="*", help="run only these case tags (see CASES)")
    args = ap.parse_args(argv)
    os.makedirs(args.work, exist_ok=True)
    if not args.draw_only:
        tags = args.only or list(CASES)
        if args.jobs > 1:
            with concurrent.futures.ProcessPoolExecutor(args.jobs) as pool:
                for tag in pool.map(capture, tags, [args.work] * len(tags)):
                    print("ran", tag, flush=True)
        else:
            for tag in tags:
                capture(tag, args.work)
                print("ran", tag, flush=True)
        if args.only:
            return
    F = Frames(args.work)
    os.makedirs(os.path.join(args.out, "examples"), exist_ok=True)
    hero(F, os.path.join(args.out, "hero.png"))
    social_preview(F, os.path.join(args.out, "social-preview.png"))
    for name, fn in CARDS.items():
        fig = fn(F)
        fig.savefig(os.path.join(args.out, "examples", f"{name}.png"), dpi=100)
        plt.close(fig)
    print("wrote", 2 + len(CARDS), "pictures to", args.out)


if __name__ == "__main__":
    main()
