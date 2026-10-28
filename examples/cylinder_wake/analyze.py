"""Strouhal number, mean drag and lift amplitude of the cylinder-wake runs.

Reads <output_dir>/history.csv of every input that has been run (generate.CASES), turns
fx_body, fy_body into C_D, C_L = F / (0.5 rho U^2 D) at their own time level t_force
(eqs. 44 and 51 of Lai & Peskin 2000; the force per unit span is the force the fluid puts on
the cylinder, their eqs. 45 and 52), and over the last ANALYSIS_WINDOW time units of the
record measures

- St = f D / U from the upward zero crossings of C_L (their eq. 53), cross-checked with the
  peak of the lift spectrum (Hann window, zero-padded rfft), and the St of each shedding
  cycle, whose spread says whether the wake is periodic,
- the mean C_D over the whole cycles between the first and the last crossing, and the mean
  C_D of each cycle,
- the lift amplitude, the mean over the cycles of half the peak-to-peak of C_L in each
  cycle, and sqrt(2) times the rms of C_L (the same number for a pure sine),
- the peak-to-peak of the drag oscillation at twice the shedding frequency, from a least
  squares fit of a constant, a linear drift and the 2f cosine and sine over the whole cycles.

The numbers are printed next to the paper's. Table I (Re 100, Scheme 2): h = 1/128, C_D
1.4473, C_L 0.3299, St 0.165, maximum displacement 2.57 % of the radius; h = 1/64, 1.5167,
0.2904, 0.155, 4.54 %. Their D is 0.3, so D/h = 38.4 and 19.2. Table II gives St of Scheme
2 and of the experiments of Williamson and Roshko: Re 100 0.165 / 0.166 / 0.164, Re 200
0.190 / 0.197 / 0.190. The paper calls its C_D and C_L time-averaged; the mean lift is zero
by symmetry, so we compare its C_L with our lift amplitude. results.png shows C_D and C_L
against time with the analysis window shaded, and the lift spectrum.

M.-C. Lai and C. S. Peskin, "An immersed boundary method with formal second-order accuracy
and reduced numerical viscosity", J. Comput. Phys. 160 (2000) 705-719, section 4.

    python -m pyibm run examples/cylinder_wake
    python -m pyibm run examples/cylinder_wake --input input2d_re200
    python examples/cylinder_wake/analyze.py
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

from pyibm.io.history import read_history

HERE = os.path.dirname(os.path.abspath(__file__))

ANALYSIS_WINDOW = 40.0    # s (= D/U) at the end of the record used for St, C_D, C_L
SPECTRUM_PADDING = 8      # zero-padding factor of the lift spectrum (bins 1/(8 window) apart)

# Lai & Peskin 2000: Re -> reference values (Table I for Re 100, Table II for St).
REFERENCE = {
    100.0: {"St": 0.165, "St_exp": (0.166, 0.164), "CD": 1.4473, "CL": 0.3299,
            "CD_coarse": 1.5167, "CL_coarse": 0.2904, "St_coarse": 0.155,
            "disp": 2.57, "disp_coarse": 4.54,
            "note": "Table I Scheme 2, h = 1/128 (D/h = 38.4); h = 1/64 (D/h = 19.2) in brackets; "
                    "Table II St (Williamson, Roshko)"},
    200.0: {"St": 0.190, "St_exp": (0.197, 0.190), "CD": None, "CL": None,
            "note": "Table II Scheme 2 St (Williamson, Roshko); St only at Re 200"},
}


def load_generate():
    spec = importlib.util.spec_from_file_location("cylinder_wake_generate", os.path.join(HERE, "generate.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def coefficients(h, rho: float, U: float, D: float):
    """(t, C_D, C_L) from a history record array, with t the time level of the force
    (the t_force column; t for a history written before that column existed)."""
    q = 0.5 * rho * U**2 * D
    t = h.t_force if "t_force" in h.dtype.names else h.t
    return t, h.fx_body / q, h.fy_body / q


def zero_crossings(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Times of the upward zero crossings of y (linear interpolation between samples)."""
    up = np.where((y[:-1] < 0.0) & (y[1:] >= 0.0))[0]
    return t[up] - y[up] * (t[up + 1] - t[up]) / (y[up + 1] - y[up])


def spectrum(t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(frequency, amplitude) of the Hann-windowed, zero-padded spectrum of y - mean(y)."""
    dt = float(np.median(np.diff(t)))
    y0 = (y - y.mean()) * np.hanning(len(y))
    n = SPECTRUM_PADDING * len(y0)
    return np.fft.rfftfreq(n, dt), np.abs(np.fft.rfft(y0, n))


def spectrum_peak(t: np.ndarray, y: np.ndarray) -> float:
    """Frequency of the largest line of the spectrum of y (0 if too short)."""
    if len(t) < 8:
        return 0.0
    freq, amp = spectrum(t, y)
    return float(freq[np.argmax(amp[1:]) + 1])


def drag_oscillation(t: np.ndarray, cd: np.ndarray, f: float) -> tuple[float, float]:
    """(peak-to-peak of the 2f component, drift per unit time) of C_D: least squares fit of
    a + b (t - mean t) + c cos(4 pi f t) + d sin(4 pi f t), peak-to-peak 2 sqrt(c^2 + d^2)."""
    w = 4.0 * np.pi * f
    A = np.column_stack([np.ones_like(t), t - t.mean(), np.cos(w * t), np.sin(w * t)])
    a, b, c, d = np.linalg.lstsq(A, cd, rcond=None)[0]
    return float(2.0 * np.hypot(c, d)), float(b)


def analyse(t, cd, cl, t_start: float, U: float = 1.0, D: float = 1.0) -> dict:
    """Wake statistics for t >= t_start (module docstring). St and the per-cycle values
    are nan or empty when fewer than two upward crossings of C_L fall in the window.

    Keys: t_start, t_end, n_crossings, periods (whole cycles used), St (from the crossings),
    St_fft, St_cycles (St of each cycle), St_spread ((max - min)/St of St_cycles), CD (mean
    over whole cycles), CD_cycles, CD_osc (2f peak-to-peak), CD_drift (per unit time),
    CL_amp (mean of CL_amp_cycles), CL_amp_cycles (half the peak-to-peak in each cycle),
    CL_rms, CL_rms_amp (sqrt(2) CL_rms)."""
    m = t >= t_start
    tt, cc, ll = t[m], cd[m], cl[m]
    empty = np.zeros(0)
    out = {"t_start": float(t_start), "t_end": float(t[-1]), "n_crossings": 0, "periods": 0.0,
           "St": np.nan, "St_fft": np.nan, "St_cycles": empty, "St_spread": np.nan,
           "CD": float(cc.mean()) if len(cc) else np.nan, "CD_cycles": empty,
           "CD_osc": np.nan, "CD_drift": np.nan,
           "CL_amp": np.nan, "CL_amp_cycles": empty, "CL_rms": np.nan, "CL_rms_amp": np.nan}
    if len(tt) < 4:
        return out
    tc = zero_crossings(tt, ll - ll.mean())
    out["n_crossings"] = int(len(tc))
    out["St_fft"] = spectrum_peak(tt, ll) * D / U
    if len(tc) < 2:
        return out
    out["periods"] = float(len(tc) - 1)
    St = (len(tc) - 1) / (tc[-1] - tc[0])
    out["St"] = St * D / U
    out["St_cycles"] = D / (U * np.diff(tc))
    out["St_spread"] = float(np.ptp(out["St_cycles"]) / out["St"])
    w = (tt >= tc[0]) & (tt <= tc[-1])
    out["CD"] = float(cc[w].mean())
    cycles = [(tt >= a) & (tt <= b) for a, b in zip(tc[:-1], tc[1:])]
    out["CD_cycles"] = np.array([cc[c].mean() for c in cycles])
    out["CL_amp_cycles"] = np.array([0.5 * np.ptp(ll[c]) for c in cycles])
    out["CL_amp"] = float(out["CL_amp_cycles"].mean())
    out["CL_rms"] = float(np.sqrt(np.mean((ll[w] - ll[w].mean())**2)))
    out["CL_rms_amp"] = float(np.sqrt(2.0) * out["CL_rms"])
    out["CD_osc"], out["CD_drift"] = drag_oscillation(tt[w], cc[w], St)
    return out


def _span(a: np.ndarray) -> str:
    """'min..max' of an array to 4 decimals, or 'none' when it is empty."""
    return f"{a.min():.4f}..{a.max():.4f}" if len(a) else "none"


def report(name: str, Re: float, r: dict, k: float, max_err: float, R_over_dx: float) -> None:
    ref = REFERENCE.get(Re)
    print(f"{name} (Re {Re:g}): window t = {r['t_start']:g}..{r['t_end']:g}, {r['periods']:g} cycles")
    print(f"   St = {r['St']:.4f} (zero crossings), {r['St_fft']:.4f} (spectrum peak); "
          f"per cycle {_span(r['St_cycles'])} (spread {100.0 * r['St_spread']:.2f} %)")
    print(f"   mean C_D = {r['CD']:.4f}; per cycle {_span(r['CD_cycles'])}; "
          f"2f oscillation peak-to-peak {r['CD_osc']:.4f}")
    print(f"   C_L amplitude {r['CL_amp']:.4f} (per cycle {_span(r['CL_amp_cycles'])}), "
          f"sqrt(2) rms {r['CL_rms_amp']:.4f}")
    print(f"   target stiffness {k:g}, max target error {max_err:.4f} cells "
          f"({100.0 * max_err / R_over_dx:.2f} % of the radius)")
    if ref is not None:
        exp = ref["St_exp"]
        dev = 100.0 * (r["St"] / ref["St"] - 1.0)
        print(f"   Lai & Peskin: St {ref['St']} (experiments {exp[0]}, {exp[1]}); this run {dev:+.1f} %")
        if ref["CD"] is not None:
            print(f"   Lai & Peskin: C_D {ref['CD']} ({ref['CD_coarse']}), C_L {ref['CL']} ({ref['CL_coarse']}), "
                  f"St {ref['St']} ({ref['St_coarse']}), max displacement {ref['disp']} % ({ref['disp_coarse']} %) "
                  f"of the radius; this run C_D {100.0 * (r['CD'] / ref['CD'] - 1.0):+.1f} %, "
                  f"C_L {100.0 * (r['CL_amp'] / ref['CL'] - 1.0):+.1f} %")
        print(f"   ({ref['note']})")


def main() -> str | None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    gen = load_generate()
    from pyibm.structures import read_target, read_vertex

    fig = Figure(figsize=(9.0, 7.0), layout="constrained")
    FigureCanvasAgg(fig)
    ax_cd, ax_cl, ax_sp = fig.subplots(3, 1)
    found = False
    for input_name, (structure, Re, out, what) in gen.CASES.items():
        path = os.path.join(HERE, out, "history.csv")
        if not os.path.isfile(path):
            print(f"{input_name}: no {out}/history.csv (run it first)")
            continue
        found = True
        h = read_history(path)
        t, cd, cl = coefficients(h, gen.RHO, gen.U, gen.D)
        r = analyse(t, cd, cl, max(t[-1] - ANALYSIS_WINDOW, 0.0), gen.U, gen.D)
        X = read_vertex(os.path.join(HERE, structure + ".vertex"))
        k = float(np.max(read_target(os.path.join(HERE, structure + ".target"), X).k))
        Nx = gen.RESOLUTIONS[structure][0]
        report(input_name, Re, r, k, float(np.max(h.max_target_error_over_dx)), 0.5 * gen.D * Nx / gen.LX)
        label = f"{input_name}: {what}"
        ax_cd.plot(t, cd, label=label)
        ax_cl.plot(t, cl, label=label)
        m = t >= r["t_start"]
        if m.sum() >= 8:
            freq, amp = spectrum(t[m], cl[m])
            ax_sp.plot(freq * gen.D / gen.U, amp / amp.max(), label=label)
        for ax in (ax_cd, ax_cl):
            ax.axvspan(r["t_start"], r["t_end"], color="0.9", zorder=0)
    if not found:
        return None
    for Re, ref in REFERENCE.items():
        ax_sp.axvline(ref["St"], color="k", linestyle=":", linewidth=0.8)
    ax_cd.axhline(REFERENCE[100.0]["CD"], color="k", linestyle=":", linewidth=0.8)
    ax_cd.set_ylabel("C_D")
    ax_cd.set_ylim(0.0, 3.0)
    ax_cd.legend(fontsize=8)
    ax_cd.set_title("drag (dotted: Lai & Peskin, Re 100, h = 1/128) and lift; "
                    "shaded: analysis window", fontsize=9)
    ax_cl.set_ylabel("C_L")
    ax_cl.set_xlabel("t U / D")
    ax_sp.set_xlim(0.0, 0.6)
    ax_sp.set_xlabel("f D / U")
    ax_sp.set_ylabel("|C_L spectrum| (normalised)")
    ax_sp.set_title("lift spectrum over the analysis window (dotted: Lai & Peskin St at Re 100, 200)", fontsize=9)
    out = os.path.join(HERE, "results.png")
    fig.savefig(out, dpi=150)
    print("saved", out)
    return out


if __name__ == "__main__":
    main()
