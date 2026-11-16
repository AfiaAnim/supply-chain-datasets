"""
Sliding-window TDA figure for the SEC data, three panels stacked:
    1. the SEC time series with the current window highlighted (MTS displacement dashed, for reference)
    2. the window's delay-embedded point cloud (x(t), x(t + tau)) with the fitted FastTDA ellipse
    3. the TDA feature over time (FastTDA minor axis 2b; Max H1 too if ripser / giotto-tda is installed)
A still image at SNAPSHOT_S and an animated GIF of the window sliding through the trial.
Paste into ONE Jupyter cell and run (or: python sec_tda_window_figure.py).

Output (in DATA_DIR/tda_results/window_figure):
    trial_NN_SEC_TDA_window.png     still image at SNAPSHOT_S
    trial_NN_SEC_TDA_window.gif     animation (if MAKE_GIF)

Requirements:
    pip install numpy pandas scipy matplotlib pillow   (+ ripser for Max H1)
"""
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal

try:                                    # make plots appear inside Jupyter
    from IPython import get_ipython
    if get_ipython() is not None:
        get_ipython().run_line_magic("matplotlib", "inline")
except Exception:
    pass

# === Parameters (same names as the DROPBEAR / H1 code) ===
DATA_DIR = "."                # folder with the SEC and MTS csv files (or sec\ and mts\ subfolders)
OUTPUT_DIR = None             # None = DATA_DIR/tda_results/window_figure
trial_ids = None              # trials to draw (one still + GIF each); None = every trial found
sec_col = "C1_raw_pF"

freq = 20                     # analysis rate (Hz) after low-pass + resampling
pass_value = 0.25             # low-pass cutoff (Hz)
window_s = 20.0               # window length (s): 20 s = half a loading cycle, so a window holding a
                              # turnaround traces a real loop; 8 s mostly gives lines; 46.6 s = the
                              # DROPBEAR rule (1/min_f + 0.5/max_f) -> one whole cycle per window
tau_s = 2.5                   # Takens delay (s)
s = 1                         # Takens stride (points per cloud = window / s); use 3 for windows over
                              # ~30 s, or Vietoris-Rips gets slow
s_W = 10                      # window step in samples (10 = 0.5 s)
SNAPSHOT_S = None             # still image: window ENDING at this time (s); None = the window with the
                              # largest loop (largest Max H1, or 2b without H1) after the first cycle
MAKE_GIF = True
GIF_STEP_S = 1.0              # GIF: one frame every GIF_STEP_S of window movement
GIF_FPS = 10
COMPUTE_H1 = True             # also Max H1 (Vietoris-Rips) if ripser or giotto-tda is installed

GOOD_OFFSET_CORR = 0.7
TARE_S = 2.0
C = {"signal": "#2a78d6", "window": "#f2c744", "ellipse": "#e34948", "fasttda": "#eb6834",
     "h1": "#6250d6", "target": "#8a8984"}


# === Loading and alignment (as in the other SEC scripts) ===
def read_numeric_csv(path):
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    return df.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)


def find_pairs(base):
    files = set(glob.glob(os.path.join(base, "*.csv")) + glob.glob(os.path.join(base, "sec", "*.csv"))
                + glob.glob(os.path.join(base, "mts", "*.csv")))
    sec, mts = {}, {}
    for p in sorted(files):
        m = re.search(r"trial[_\- ]?(\d+)", os.path.basename(p), re.I)
        if not m:
            continue
        with open(p, "r", encoding="utf-8-sig", errors="replace") as f:
            header = f.readline()
        if sec_col in header and "time_s" in header:
            sec.setdefault(int(m.group(1)), p)
        elif "RunningTime_s" in header and "Displacement_m" in header:
            mts.setdefault(int(m.group(1)), p)
    return [(n, sec[n], mts[n]) for n in sorted(set(sec) & set(mts)) if not trial_ids or n in trial_ids]


def load_trial(num, sec_path, mts_path):
    sec, mts = read_numeric_csv(sec_path), read_numeric_csv(mts_path)
    t_s = sec["time_s"].to_numpy(float)
    t_s = t_s - t_s[0]
    fs = (len(t_s) - 1) / t_s[-1]
    t_u = np.arange(0, t_s[-1], 1 / fs)                                 # uneven -> uniform
    sos = signal.butter(4, pass_value, btype="lowpass", fs=fs, output="sos")
    x = signal.sosfiltfilt(sos, np.interp(t_u, t_s, sec[sec_col].to_numpy(float)))
    t_m = mts["RunningTime_s"].to_numpy(float)
    t_m = t_m - t_m[0]
    d = mts["Displacement_m"].to_numpy(float)
    best = (10.0, 0.0)                                                  # SEC starts ~10 s early
    tt = np.arange(0, t_m[-1], 0.1)
    for lag in np.arange(5.0, 15.0, 0.1):
        if tt[-1] + lag <= t_u[-1]:
            r = abs(np.corrcoef(np.interp(tt + lag, t_u, x), np.interp(tt, t_m, d))[0, 1])
            best = max(best, (float(lag), float(r)), key=lambda b: b[1])
    offset, r = best
    t = np.arange(0, min(t_m[-1], t_u[-1] - offset), 1 / freq)
    x = np.interp(t + offset, t_u, x)
    print(f"  trial_{num:02d}: offset {offset:.1f} s (|corr| {r:.2f})" + ("" if r >= GOOD_OFFSET_CORR else "  <-- alignment unreliable"))
    return {"name": f"trial_{num:02d}", "t": t, "x": x - np.median(x[t < TARE_S]),
            "disp": 1000 * np.interp(t, t_m, d)}


# === FastTDA (ellipse fit) and Max H1 ===
def fasttda_fit(pc):
    """SVD-LSE ellipse fit (A + C = 1) -> (minor axis 2b, ellipse curve (x, y)) or (NaN, None)."""
    mx, my = pc[:, 0].mean(), pc[:, 1].mean()
    X, Y = pc[:, 0] - mx, pc[:, 1] - my
    rng = max(np.abs(X).max(), np.abs(Y).max())
    if not np.isfinite(rng) or rng == 0:
        return np.nan, None
    k = 1.5 / rng
    X, Y = X * k, Y * k
    G = np.column_stack([X**2 - Y**2, X * Y, X, Y, np.ones_like(X)])
    U, S, Vt = np.linalg.svd(G, full_matrices=False)
    Sinv = np.where(S > np.finfo(float).eps * max(G.shape) * S[0], 1.0 / S, 0.0)
    A, B, D, E, F = (Vt.T * Sinv) @ (U.T @ -(Y**2))
    Cc = 1.0 - A
    if 4 * A * Cc - B**2 <= 0:
        return np.nan, None
    th = 0.5 * np.arctan2(B, A - Cc)
    c, sn = np.cos(th), np.sin(th)
    Ap, Cp = A*c*c + B*c*sn + Cc*sn*sn, A*sn*sn - B*c*sn + Cc*c*c
    Dp, Ep = D*c + E*sn, -D*sn + E*c
    if Ap <= 0 or Cp <= 0:
        return np.nan, None
    Fn = F - Dp**2 / (4 * Ap) - Ep**2 / (4 * Cp)
    if Fn >= 0:
        return np.nan, None
    a_u, a_v = np.sqrt(-Fn / Ap), np.sqrt(-Fn / Cp)
    u0, v0 = -Dp / (2 * Ap), -Ep / (2 * Cp)
    ang = np.linspace(0, 2 * np.pi, 200)
    u, v = u0 + a_u * np.cos(ang), v0 + a_v * np.sin(ang)
    return 2.0 * min(a_u, a_v), (mx + (c * u - sn * v) / k, my + (sn * u + c * v) / k)


def h1_backend():
    if not COMPUTE_H1:
        return None
    try:
        from ripser import ripser
        return lambda pc: ripser(pc, maxdim=1)["dgms"][1]
    except ImportError:
        pass
    try:
        from gtda.homology import VietorisRipsPersistence
        vr = VietorisRipsPersistence(homology_dimensions=(1,))
        return lambda pc: (lambda d: d[d[:, 2] == 1][:, :2])(vr.fit_transform(pc[None])[0])
    except ImportError:
        print("  Max H1 skipped: install ripser (pip install ripser) or giotto-tda")
        return None


def windows_of(tr, h1):
    """Every window (step s_W): end time, scaled window, point cloud, 2b, ellipse, Max H1."""
    w, d = int(round(window_s * freq)), max(1, int(round(tau_s * freq)))
    W = sliding_window_view(tr["x"], w)[::s_W]
    lo, hi = W.min(axis=1, keepdims=True), W.max(axis=1, keepdims=True)
    Z = 2 * (W - lo) / np.where(hi > lo, hi - lo, 1.0) - 1               # each window scaled to [-1, 1]
    starts = np.arange(len(tr["x"]) - w + 1)[::s_W]
    out = {"start": starts, "end_t": tr["t"][starts + w - 1], "clouds": [], "two_b": [], "ellipse": [], "h1": []}
    for z in Z:
        pc = np.column_stack([z[:-d:s], z[d::s]])
        b, ell = fasttda_fit(pc)
        out["clouds"].append(pc)
        out["two_b"].append(b)
        out["ellipse"].append(ell)
        if h1 is not None:
            dg = h1(pc)
            out["h1"].append(float(np.max(dg[:, 1] - dg[:, 0])) if len(dg) else 0.0)
    out["two_b"] = np.array(out["two_b"])
    out["h1"] = np.array(out["h1"]) if out["h1"] else None
    out["w"], out["d"] = w, d
    return out


# === Figure ===
def make_figure(tr, wd):
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(7.5, 12), gridspec_kw={"height_ratios": [1, 1.25, 1]})
    a1.plot(tr["t"], tr["x"], color=C["signal"], lw=1.2)
    a1.set_ylabel("Capacitance change (pF)")
    a1.set_xlabel("Time (s)")
    a1b = a1.twinx()
    a1b.plot(tr["t"], tr["disp"], color=C["target"], lw=0.9, ls="--")
    a1b.set_ylabel("MTS displacement (mm), dashed", color=C["target"])
    span = [a1.axvspan(0, 1, color=C["window"], alpha=0.45, lw=0)]   # redrawn every frame
    pts, = a2.plot([], [], ".", color=C["signal"], ms=3)
    ell, = a2.plot([], [], color=C["ellipse"], lw=2)
    a2.set_xlim(-1.35, 1.35)
    a2.set_ylim(-1.35, 1.35)
    a2.set_aspect("equal")
    a2.set_xlabel("x(t)  (window scaled to [-1, 1])")
    a2.set_ylabel(f"x(t + {tau_s:g} s)")
    a2.grid(color="#e4e3df", lw=0.6)
    a3.plot(wd["end_t"], wd["two_b"], color=C["fasttda"], lw=0.8, alpha=0.25)
    line_b, = a3.plot([], [], color=C["fasttda"], lw=1.6, label="FastTDA 2b")
    dot_b, = a3.plot([], [], "o", color=C["fasttda"], ms=7)
    a3.set_ylabel("FastTDA minor axis 2b", color=C["fasttda"])
    a3.set_xlabel("Time (s, end of window)")
    a3.set_xlim(tr["t"][0], tr["t"][-1])
    top = np.nanpercentile(wd["two_b"], 99) * 1.15
    a3.set_ylim(0, top if np.isfinite(top) and top > 0 else 1)
    handles = [line_b]
    if wd["h1"] is not None:
        a3h = a3.twinx()
        a3h.plot(wd["end_t"], wd["h1"], color=C["h1"], lw=0.8, alpha=0.25)
        line_h, = a3h.plot([], [], color=C["h1"], lw=1.4, label="Max H1 (Vietoris-Rips)")
        a3h.set_ylabel("Max H1", color=C["h1"])
        handles.append(line_h)
    else:
        line_h = None
    a3d = a3.twinx()
    a3d.plot(tr["t"], tr["disp"], color=C["target"], lw=0.8, ls="--")
    a3d.set_yticks([])
    a3.legend(handles=handles, frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2),
              ncol=len(handles))                            # below the panel: never covers the data
    for a in (a1, a3):
        a.grid(color="#e4e3df", lw=0.6)
    fig.suptitle(f"{tr['name']}: sliding-window TDA of the SEC signal\n"
                 f"(window {window_s:g} s, tau {tau_s:g} s, low-pass {pass_value:g} Hz)", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.5)       # room for the main title and panel titles

    def draw(i):
        t0, t1 = tr["t"][wd["start"][i]], wd["end_t"][i]
        span[0].remove()                                     # works whatever axvspan returns
        span[0] = a1.axvspan(t0, t1, color=C["window"], alpha=0.45, lw=0)
        pc, e = wd["clouds"][i], wd["ellipse"][i]
        pts.set_data(pc[:, 0], pc[:, 1])
        ell.set_data(*(e if e is not None else ([], [])))
        b = wd["two_b"][i]
        a1.set_title(f"SEC time series - window {t0:.1f}-{t1:.1f} s (yellow)", fontsize=10)
        a2.set_title(f"Takens embedding of the window: "
                     + (f"ellipse 2b = {b:.2f}" if np.isfinite(b) else "no ellipse (points on a line)")
                     + (f", Max H1 = {wd['h1'][i]:.3f}" if wd["h1"] is not None else ""), fontsize=10)
        line_b.set_data(wd["end_t"][: i + 1], wd["two_b"][: i + 1])
        dot_b.set_data([t1], [b if np.isfinite(b) else np.nan])
        if line_h is not None:
            line_h.set_data(wd["end_t"][: i + 1], wd["h1"][: i + 1])
        a3.set_title(f"TDA feature over time (now: {t1:.1f} s)", fontsize=10)
        return span[0], pts, ell, line_b, dot_b

    return fig, draw


# === Run ===
base = DATA_DIR
out_dir = OUTPUT_DIR or os.path.join(base, "tda_results", "window_figure")
os.makedirs(out_dir, exist_ok=True)
h1 = h1_backend()
print(f"window {window_s:g} s, tau {tau_s:g} s, stride {s}, step {s_W} samples ({s_W / freq:g} s)")
for p in find_pairs(base):
    tr = load_trial(*p)
    wd = windows_of(tr, h1)
    fig, draw = make_figure(tr, wd)
    if SNAPSHOT_S is None:
        score = wd["h1"] if wd["h1"] is not None else np.nan_to_num(wd["two_b"])
        later = wd["end_t"] >= 40.0 + window_s                     # window after the first loading cycle
        i = int(np.argmax(np.where(later, score, -np.inf))) if later.any() else int(np.argmax(score))
    else:
        i = int(np.argmin(np.abs(wd["end_t"] - SNAPSHOT_S)))
    draw(i)
    png = os.path.join(out_dir, f"{tr['name']}_SEC_TDA_window.png")
    fig.savefig(png, dpi=150)
    print(f"Saved: {png}")
    plt.show()
    if MAKE_GIF:
        frames = range(0, len(wd["end_t"]), max(1, int(round(GIF_STEP_S * freq / s_W))))
        anim = FuncAnimation(fig, draw, frames=frames, blit=False)
        gif = os.path.join(out_dir, f"{tr['name']}_SEC_TDA_window.gif")
        anim.save(gif, writer=PillowWriter(fps=GIF_FPS), dpi=80)
        print(f"Saved: {gif}")
    plt.close(fig)
