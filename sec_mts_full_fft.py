"""
Full-length Fourier transform of every SEC and MTS file (trials 1-9) - Python version of the
MATLAB reference:

    dt   = time(2);                              % sample spacing
    facc = fft(acc);                             % FFT of the WHOLE record (no window)
    N    = length(facc);
    w    = linspace(0,1,N/2)./(2*dt);            % frequency axis 0 .. Nyquist, in Hz
    plot(w, abs(facc(1:N/2)))                    % |fx| vs f

Each file is transformed over its full time length, so the frequency resolution is
df = 1 / record length (e.g. 160 s -> 0.00625 Hz) and the peaks are exact FFT bins.
Paste into ONE Jupyter cell and run (or: python sec_mts_full_fft.py).

Output (in DATA_DIR/tda_results/full_fft):
    <trial>_full_fft.png              |fx| vs f (0 .. Nyquist, MATLAB style) for SEC, MTS displacement
                                      and MTS force, with a zoom inset on the dominant peak
    <trial>_full_fft_buildup.gif      the same FFT computed on a growing record (0 -> full length)
    All_trials_full_fft.png           all trials side by side, dominant peaks marked
    Full_FFT_peaks.xlsx               dominant frequency / period and next peaks per file

Requirements: pip install numpy pandas scipy matplotlib openpyxl pillow
"""

import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation
from scipy import signal

try:                                    # make plots appear inside Jupyter
    from IPython import get_ipython
    from IPython.display import Image, display
    IN_NOTEBOOK = get_ipython() is not None
    if IN_NOTEBOOK:
        get_ipython().run_line_magic("matplotlib", "inline")
except Exception:
    IN_NOTEBOOK = False


# ============================================================
# SETTINGS
# ============================================================

DATA_DIR = "."                          # folder with the SEC and MTS csv files (or sec\ and mts\ subfolders)
SEC_SUBDIR, MTS_SUBDIR = "sec", "mts"
OUTPUT_DIR = None                       # None = DATA_DIR/tda_results/full_fft
TRIALS = None                           # None = every trial found; or e.g. ["trial_04", "trial_08"]

SEC_TIME_COL, SEC_COL = "time_s", "C1_filtered_pF"
MTS_TIME_COL = "RunningTime_s"
TARGET_COLS = ["Displacement_m", "Force_N"]

REMOVE_MEAN = True                      # True: subtract the mean before the FFT. The MATLAB code does
                                        # not; then the 0 Hz bin (SEC ~45 pF offset) dwarfs every peak
MATLAB_AXIS = False                     # True: w = linspace(0,1,N/2)/(2*dt) exactly as in MATLAB;
                                        # False: exact bin frequencies k/(N*dt) (differ by < 1 bin)
MATLAB_LOOK = True                      # per-trial plots styled like the MATLAB figure (Times, bold,
                                        # boxed axes, blue line, full axis 0 .. Nyquist)
USE_RAW = False                         # True: unfiltered columns (C1_raw_pF, Displacement_raw_m,
                                        # Force_raw_N) - they keep the high-frequency content
F_MAX_PLOT = None                       # x-axis limit in Hz; None = up to Nyquist (as in MATLAB)
LOG_Y = False                           # True: logarithmic |fx| axis - shows the small high-frequency
                                        # peaks (e.g. 24 / 41 Hz in Force_raw_N) next to the 0.025 Hz one
ZOOM_HZ = 0.2                           # inset zoom 0 .. ZOOM_HZ on the dominant peak; None = no inset
F_MAX_OVERVIEW = 0.5                    # x-axis limit of the all-trials overview and the animation
MIN_PEAK_HZ = 0.01                      # peaks below this (drift) are not reported
N_PEAKS = 4                             # peaks listed per signal (dominant + next strongest)
MIN_CYCLES = 3                          # a peak's period must fit this many times into the record
MIN_POWER_PCT = 20.0                    # dominant peak must hold >= this % of the spectrum's power,
                                        # otherwise the signal is reported as "no dominant peak"

BUILDUP = True                          # animation: FFT of a growing record 0 -> full length
BUILDUP_FRAMES = 40
BUILDUP_FPS = 6
BUILDUP_TRIALS = None                   # None = all trials; or e.g. ["trial_05"]

RAW_COLS = {"SEC": "C1_raw_pF", "Displacement_m": "Displacement_raw_m", "Force_N": "Force_raw_N"}
MATLAB_BLUE = "#0072BD"
COLORS = {"SEC": "#52514e", "Displacement_m": "#6250d6", "Force_N": "#e87ba4"}
UNITS = {"SEC": "pF", "Displacement_m": "m", "Force_N": "N"}
GRID, INK = "#e4e3df", "#0b0b0b"


# ============================================================
# FILES
# ============================================================

def to_local_path(path):
    path = str(path).strip().strip('"')
    if os.name != "nt" and len(path) > 2 and path[1] == ":" and path[2] in "\\/":
        return f"/mnt/{path[0].lower()}/{path[3:].replace(chr(92), '/')}"
    return path


def read_numeric_csv(path):
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, [c for c in df.columns if not c.startswith("Unnamed")]]
    return df.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)


def trial_key(path):
    m = re.search(r"trial[_\- ]?(\d+)", os.path.basename(path), re.I)
    return (f"trial_{int(m.group(1)):02d}", int(m.group(1))) if m else (None, None)


def file_kind(path):
    try:
        with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
            header = f.readline()
    except OSError:
        return None
    if SEC_COL in header:
        return "sec"
    if MTS_TIME_COL in header and all(c in header for c in TARGET_COLS):
        return "mts"
    return None


def find_pairs():
    base = to_local_path(DATA_DIR)
    files = glob.glob(os.path.join(base, "*.csv"))
    for sub in (SEC_SUBDIR, MTS_SUBDIR):
        files += glob.glob(os.path.join(base, sub, "*.csv"))
    sec, mts = {}, {}
    for p in sorted(files):
        name, num = trial_key(p)
        kind = file_kind(p) if name else None
        if not name:
            print(f"  ignored {os.path.basename(p)}: no 'trial_NN' in the file name")
        elif not kind:
            print(f"  ignored {os.path.basename(p)}: columns not recognised as SEC ({SEC_COL}) or "
                  f"MTS ({MTS_TIME_COL}, {', '.join(TARGET_COLS)})")
        elif name in (sec if kind == "sec" else mts):
            print(f"  ignored {os.path.basename(p)}: second {kind.upper()} file for {name}")
        else:
            (sec if kind == "sec" else mts)[name] = p
    if not sec:
        raise FileNotFoundError(f"No SEC csv files (with a {SEC_COL!r} column) found in {os.path.abspath(base)}")
    for name in sorted(set(mts) - set(sec)):
        print(f"  {name}: MTS file found but no SEC file - skipped")
    pairs = []
    for name in sorted(sec):
        if TRIALS and name not in TRIALS:
            continue
        if name not in mts:
            print(f"  {name}: SEC file found but no MTS file - skipped")
            continue
        pairs.append((name, sec[name], mts[name]))
        print(f"  {name}: SEC {os.path.basename(sec[name])}  +  MTS {os.path.basename(mts[name])}")
    print(f"Found {len(pairs)} SEC/MTS pairs in {os.path.abspath(base)}")
    return pairs


def uniform(t, y):
    """FFT needs equally spaced samples: put the record on a uniform grid with the median dt."""
    t = t - t[0]
    dt = float(np.median(np.diff(t)))
    tu = np.arange(0, t[-1] + dt / 2, dt)
    return tu, np.interp(tu, t, y), dt


def load_trial(name, sec_path, mts_path):
    sec, mts = read_numeric_csv(sec_path), read_numeric_csv(mts_path)
    sigs = {}
    col = lambda name: RAW_COLS.get(name, name) if USE_RAW else (SEC_COL if name == "SEC" else name)
    t, y, dt = uniform(sec[SEC_TIME_COL].to_numpy(float), sec[col("SEC")].to_numpy(float))
    sigs["SEC"] = (t, y, dt)
    for c in TARGET_COLS:
        t, y, dt = uniform(mts[MTS_TIME_COL].to_numpy(float), mts[col(c)].to_numpy(float))
        sigs[c] = (t, y, dt)
    print(f"  {name}: SEC {sigs['SEC'][0][-1]:.1f} s at {1 / sigs['SEC'][2]:.0f} Hz, "
          f"MTS {sigs[TARGET_COLS[0]][0][-1]:.1f} s at {1 / sigs[TARGET_COLS[0]][2]:.0f} Hz")
    return {"name": name, "sigs": sigs}


# ============================================================
# FULL-LENGTH FFT (the MATLAB code)
# ============================================================

def full_fft(y, dt):
    """|fx| over the whole record, frequencies 0 .. Nyquist (first N/2 bins), as in the MATLAB code."""
    y = np.asarray(y, float)
    if REMOVE_MEAN:
        y = y - y.mean()
    fx = np.fft.fft(y)                                   # facc = fft(acc)
    N = len(fx)                                          # N = length(facc)
    half = N // 2
    if MATLAB_AXIS:
        w = np.linspace(0, 1, half) / (2 * dt)           # w = linspace(0,1,N/2)./(2*dt)
    else:
        w = np.arange(half) / (N * dt)                   # exact bin frequencies
    return w, np.abs(fx[:half])                          # abs(facc(1:N/2))


def peaks_of(w, amp, record_s):
    """Strongest local maxima (period fits >= MIN_CYCLES times): list of (f, |fx|, % of total |fx|^2)."""
    pk, _ = signal.find_peaks(amp)                       # local maxima over the whole spectrum,
    pk = pk[w[pk] >= max(MIN_PEAK_HZ, MIN_CYCLES / record_s)]   # then keep those slow enough
    if not pk.size:
        return []
    pk = pk[np.argsort(amp[pk])[::-1][:N_PEAKS]]
    total = np.sum(amp[1:] ** 2)
    return [(float(w[i]), float(amp[i]), float(100 * amp[i] ** 2 / total)) for i in pk]


def is_dominant(pk):
    return bool(pk) and pk[0][2] >= MIN_POWER_PCT


def visible(w, *arrays, fmax=None):
    """Only the plotted frequency range (drawing every bin up to Nyquist is slow)."""
    m = w <= (fmax or w[-1])
    return (w[m],) + tuple(a[m] for a in arrays)


def thin(t, y, n=4000):
    k = max(1, len(t) // n)
    return t[::k], y[::k]


MATLAB_RC = {"font.family": "serif",
             "font.serif": ["Times New Roman", "Times", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
             "font.weight": "bold", "axes.labelweight": "bold", "axes.titleweight": "bold",
             "font.size": 14, "axes.linewidth": 1.0, "xtick.direction": "in", "ytick.direction": "in",
             "xtick.top": True, "ytick.right": True, "axes.grid": False}


def plot_trial(tr, out_dir):
    """|fx| vs f of the full record for SEC, MTS displacement and MTS force (MATLAB-style figure)."""
    names = ["SEC"] + TARGET_COLS
    rows = []
    with plt.rc_context(MATLAB_RC if MATLAB_LOOK else {}):
        fig, axes = plt.subplots(len(names), 1, figsize=(11, 3.6 * len(names)), squeeze=False)
        for ax, name in zip(axes[:, 0], names):
            t, y, dt = tr["sigs"][name]
            w, amp = full_fft(y, dt)
            pk = peaks_of(w, amp, t[-1])
            dominant = is_dominant(pk)
            colr = MATLAB_BLUE if MATLAB_LOOK else COLORS[name]
            lab = ("SEC " if name == "SEC" else "MTS ") + (RAW_COLS.get(name, name) if USE_RAW else
                                                          (SEC_COL if name == "SEC" else name))
            fmax = F_MAX_PLOT or w[-1]
            ax.plot(*visible(w, amp, fmax=fmax), color=colr, lw=1.0)
            ax.set_xlim(0, fmax)
            if LOG_Y:
                ax.set_yscale("log")
                ax.set_ylim(np.percentile(amp[1:], 1), 3 * amp[1:].max())
            else:
                ax.set_ylim(0, 1.08 * amp[1:].max())
            ax.set_xlabel("f (Hz)")
            ax.set_ylabel("|fx|")
            ax.set_title(f"{tr['name']} - {lab}: full record {t[-1]:.1f} s, N = {len(t)}", fontsize=12)
            for k, (f, a, share) in enumerate(pk):
                rows.append({"Trial": tr["name"], "Signal": name, "Rank": k + 1, "Freq_Hz": f,
                             "Period_s": 1 / f, "abs_fx": a, "Power_share_pct": share,
                             "Dominant": k == 0 and dominant, "Record_s": t[-1],
                             "Resolution_Hz": 1 / (len(t) * dt)})
            if ZOOM_HZ and ZOOM_HZ < fmax:                  # inset: zoom on the slow loading peaks
                ins = ax.inset_axes([0.42, 0.30, 0.55, 0.62])
                ins.plot(*visible(w, amp, fmax=ZOOM_HZ), color=colr, lw=1.2)
                ins.set_xlim(0, ZOOM_HZ)
                zm = w <= ZOOM_HZ
                ins.set_ylim(0, 1.25 * amp[zm][1:].max())
                ins.tick_params(labelsize=10)
                ins.set_title(f"zoom 0-{ZOOM_HZ:g} Hz", fontsize=10)
                target = ins
            else:
                target = ax
            for k, (f, a, share) in enumerate(pk):
                if f > (ZOOM_HZ if target is not ax else fmax):
                    continue
                main = k == 0 and dominant
                target.plot(f, a, "o" if main else "^", color="#D95319" if main else "#7E2F8E",
                            ms=7 if main else 5, zorder=5)
                target.annotate(f"{f:.4f} Hz (T = {1 / f:.1f} s)" if main else f"{f:.3f} Hz",
                                (f, a), xytext=(6, 2 if main else 10 * (k % 2) - 3),
                                textcoords="offset points", fontsize=9 if main else 8)
            if not dominant:
                ax.text(0.98, 0.95, "no dominant peak (not periodic)", transform=ax.transAxes,
                        ha="right", va="top", fontsize=10, color="#A2142F")
        fig.tight_layout()
        png = os.path.join(out_dir, f"{tr['name']}_full_fft.png")
        fig.savefig(png, dpi=150)
    print(f"Saved: {png}")
    plt.show()
    plt.close(fig)
    return rows


def buildup_animation(tr, out_dir):
    """FFT of the record from 0 to t, t growing to the full length: the peaks sharpen as the
    record grows; the last frame is the full-length FFT."""
    names = ["SEC"] + TARGET_COLS
    fig, axes = plt.subplots(len(names), 2, figsize=(14, 2.6 * len(names)),
                             gridspec_kw={"width_ratios": [1.4, 1]}, squeeze=False)
    arts = []
    for (ax_t, ax_f), name in zip(axes, names):
        t, y, dt = tr["sigs"][name]
        ax_t.plot(*thin(t, y), color="#cfcdc6", lw=0.8)
        used, = ax_t.plot([], [], color=COLORS[name], lw=1.0)
        ax_t.set_xlim(0, t[-1])
        ax_t.set_ylabel(f"{'SEC' if name == 'SEC' else 'MTS ' + name.split('_')[0]}\n({UNITS[name]})", fontsize=8)
        w_full, a_full = full_fft(y, dt)
        line, = ax_f.plot([], [], color=COLORS[name], lw=1.4)
        dot, = ax_f.plot([], [], "o", color="#c0392b", ms=8)
        txt = ax_f.text(0.98, 0.95, "", transform=ax_f.transAxes, ha="right", va="top", fontsize=8.5)
        ax_f.set_xlim(0, F_MAX_OVERVIEW or w_full[-1])
        ax_f.set_ylim(0, 1.2 * a_full[w_full >= MIN_PEAK_HZ].max())
        ax_f.set_ylabel("|fx|", fontsize=8)
        for ax in (ax_t, ax_f):
            ax.grid(color=GRID, lw=0.6)
            ax.spines[["top", "right"]].set_visible(False)
        arts.append((name, t, y, dt, used, line, dot, txt))
    axes[-1, 0].set_xlabel("Time (s) - coloured part = record used so far")
    axes[-1, 1].set_xlabel("f (Hz)")
    title = fig.suptitle("", fontweight="bold")
    fracs = np.linspace(0.15, 1.0, BUILDUP_FRAMES)

    def update(k):
        out = [title]
        for name, t, y, dt, used, line, dot, txt in arts:
            n = max(8, int(round(fracs[k] * len(t))))
            w, amp = full_fft(y[:n], dt)
            used.set_data(*thin(t[:n], y[:n]))
            line.set_data(*visible(w, amp, fmax=F_MAX_OVERVIEW))
            pk = peaks_of(w, amp, t[n - 1])
            if is_dominant(pk):
                dot.set_data([pk[0][0]], [pk[0][1]])
                txt.set_text(f"dominant {pk[0][0]:.4f} Hz  (T = {1 / pk[0][0]:.1f} s)\n"
                             f"record {t[n - 1]:.0f} s, df = {1 / (n * dt):.4f} Hz")
            else:
                dot.set_data([], [])
                txt.set_text(f"no dominant peak\nrecord {t[n - 1]:.0f} s")
            out += [used, line, dot, txt]
        title.set_text(f"{tr['name']}: FFT of a growing record ({100 * fracs[k]:.0f}% of the full length)")
        return out

    anim = animation.FuncAnimation(fig, update, frames=BUILDUP_FRAMES, interval=1000 / BUILDUP_FPS)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    gif = os.path.join(out_dir, f"{tr['name']}_full_fft_buildup.gif")
    anim.save(gif, writer=animation.PillowWriter(fps=BUILDUP_FPS), dpi=75)
    plt.close(fig)
    print(f"Saved: {gif}")
    if IN_NOTEBOOK:
        display(Image(filename=gif))


def plot_all(trials, out_dir):
    """All trials in one figure: one column per signal, one row per trial."""
    names = ["SEC"] + TARGET_COLS
    fig, axes = plt.subplots(len(trials), len(names), figsize=(4.6 * len(names), 1.55 * len(trials)),
                             sharex=True, squeeze=False)
    for i, tr in enumerate(trials):
        for j, name in enumerate(names):
            ax = axes[i, j]
            t, y, dt = tr["sigs"][name]
            w, amp = full_fft(y, dt)
            ax.plot(*visible(w, amp, fmax=F_MAX_OVERVIEW), color=COLORS[name], lw=1.1)
            pk = peaks_of(w, amp, t[-1])
            if is_dominant(pk):
                f, a, _ = pk[0]
                ax.plot(f, a, "o", color="#c0392b", ms=5)
                ax.text(0.98, 0.85, f"{f:.4f} Hz / {1 / f:.1f} s", transform=ax.transAxes,
                        ha="right", va="top", fontsize=7.5)
            else:
                ax.text(0.98, 0.85, "no dominant peak", transform=ax.transAxes, ha="right", va="top",
                        fontsize=7.5, color="#c0392b")
            ax.set_xlim(0, F_MAX_OVERVIEW or w[-1])
            ax.set_ylim(0, 1.25 * max(amp[w >= MIN_PEAK_HZ].max(), 1e-12))
            ax.set_yticks([])
            ax.grid(color=GRID, lw=0.5)
            ax.spines[["top", "right", "left"]].set_visible(False)
            if j == 0:
                ax.set_ylabel(tr["name"], fontsize=8.5, rotation=0, ha="right", va="center")
            if i == 0:
                ax.set_title("SEC" if name == "SEC" else f"MTS {name}", fontsize=10, fontweight="bold")
    for ax in axes[-1]:
        ax.set_xlabel("f (Hz)")
    fig.suptitle("Full-length Fourier transform |fx| of every trial (red = dominant peak)",
                 fontweight="bold")
    fig.tight_layout()
    png = os.path.join(out_dir, "All_trials_full_fft.png")
    fig.savefig(png, dpi=150)
    print(f"Saved: {png}")
    plt.show()
    plt.close(fig)


# ============================================================
# RUN
# ============================================================

out_dir = (to_local_path(OUTPUT_DIR) if OUTPUT_DIR
           else os.path.join(to_local_path(DATA_DIR), "tda_results", "full_fft"))
os.makedirs(out_dir, exist_ok=True)
print("Loading:")
trials = [load_trial(*p) for p in find_pairs()]

rows = []
for tr in trials:
    rows += plot_trial(tr, out_dir)
    if BUILDUP and (not BUILDUP_TRIALS or tr["name"] in BUILDUP_TRIALS):
        buildup_animation(tr, out_dir)
plot_all(trials, out_dir)

peaks = pd.DataFrame(rows)
first = peaks[peaks["Rank"] == 1].assign(Freq_Hz=lambda d: d["Freq_Hz"].where(d["Dominant"]))
dom = first.pivot(index="Trial", columns="Signal", values="Freq_Hz")
dom = dom[[c for c in ["SEC"] + TARGET_COLS if c in dom]]
print("\nDominant frequency (Hz) of the full record (NaN = no dominant peak / not periodic):")
print(dom.round(4).to_string())
print("\nDominant period (s):")
print((1 / dom).round(1).to_string())
xlsx = os.path.join(out_dir, "Full_FFT_peaks.xlsx")
with pd.ExcelWriter(xlsx, engine="openpyxl") as wr:
    dom.reset_index().to_excel(wr, sheet_name="Dominant_Hz", index=False)
    peaks.to_excel(wr, sheet_name="All_peaks", index=False)
print(f"Saved: {xlsx}")
