"""
SEC vs MTS performance matrix + live dominant-peak time series (trials 1-9). Paste into ONE
Jupyter cell and run (or: python sec_mts_performance_live.py). Stand-alone: loads and aligns the
files itself. The MTS measurement (Displacement_m, Force_N) is the TARGET, SEC (C1_filtered_pF)
the sensor.

    1. Pair SEC and MTS files by trial number (recognised by their columns), align them by
       cross-correlation (offset ~10 s), resample to 20 Hz.
    2. Performance matrix (trials x metrics, all 0..1, higher = better):
         alignment |r|, |r| and R2 of SEC vs displacement / force (calibrated within the trial and
         calibrated on the OTHER trials), SEC-MTS coherence in the loading band, SEC peak power.
    3. Live time series per trial: a window slides along SEC, displacement and force while the
       right panel shows the FFT of that window with the dominant peak of each signal marked.
       Saved as GIF and shown in the notebook.
    4. Dominant period over time for all trials (small multiples).

Output (in DATA_DIR/tda_results/performance):
    SEC_MTS_performance_matrix.png, SEC_MTS_dominant_period_over_time.png,
    <trial>_live_dominant_peak.gif, SEC_MTS_performance.xlsx

Requirements: pip install numpy pandas scipy matplotlib openpyxl pillow
"""

import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation
from matplotlib.colors import LinearSegmentedColormap
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
OUTPUT_DIR = None                       # None = DATA_DIR/tda_results/performance
TRIALS = None                           # None = every trial found (1-9); or e.g. ["trial_04", "trial_08"]

SEC_TIME_COL, SEC_COL = "time_s", "C1_filtered_pF"
MTS_TIME_COL = "RunningTime_s"
TARGET_COLS = ["Displacement_m", "Force_N"]
RESAMPLE_HZ = 20.0

OFFSET_GUESS_S, OFFSET_SEARCH_S = 10.0, 5.0     # MTS time = SEC time - offset
OFFSET_REF_COL = "Displacement_m"
MIN_OFFSET_CORR, GOOD_OFFSET_CORR = 0.2, 0.7

# Performance
CAL_DEGREE = 3                          # polynomial SEC -> target calibration (3 = S-shaped curve)
CAL_DECIMATE = 10                       # use every n-th sample for the calibration fits (speed)
COHERENCE_BAND_HZ = (0.01, 0.2)         # loading band (40 s cycle = 0.025 Hz and its harmonics)

# FFT / dominant peak
FFT_MIN_HZ, FFT_MAX_HZ = 0.01, 0.3
FFT_PAD = 8
FFT_MIN_POWER_PCT = 30.0                # below this the window is "not periodic"
FFT_MIN_CYCLES = 1.5                    # a period must fit at least this many times into the window

# Live time series (sliding window)
LIVE = True
LIVE_WINDOW_S = 80.0                    # window of the live FFT (>= 2 loading cycles of 40 s);
                                        # shortened automatically for short records
LIVE_STEP_S = 2.0                       # window moves this much per frame
LIVE_FPS = 8
LIVE_TRIALS = None                      # None = all trials; or e.g. ["trial_05"]

COLORS = {"SEC": "#52514e", "Displacement_m": "#6250d6", "Force_N": "#e87ba4"}
GRID, INK, MUTED = "#e4e3df", "#0b0b0b", "#52514e"


# ============================================================
# LOADING + ALIGNMENT
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
        if kind:
            (sec if kind == "sec" else mts).setdefault(name, (num, p))
    if not sec:
        raise FileNotFoundError(f"No SEC csv files (with a {SEC_COL!r} column) found in {os.path.abspath(base)}")
    pairs = []
    for name in sorted(sec):
        if TRIALS and name not in TRIALS:
            continue
        if name not in mts:
            print(f"  {name}: SEC file found but no MTS file - skipped")
            continue
        pairs.append((name, sec[name][0], sec[name][1], mts[name][1]))
        print(f"  {name}: SEC {os.path.basename(sec[name][1])}  +  MTS {os.path.basename(mts[name][1])}")
    print(f"Found {len(pairs)} SEC/MTS pairs in {os.path.abspath(base)}")
    return pairs


def estimate_offset(t_sec, x_sec, t_mts, y_mts):
    fs = 10.0
    a = np.interp(np.arange(0, t_sec[-1], 1 / fs), t_sec, x_sec)
    b = np.interp(np.arange(0, t_mts[-1], 1 / fs), t_mts, y_mts)
    best = (OFFSET_GUESS_S, 0.0)
    for lag in np.arange(OFFSET_GUESS_S - OFFSET_SEARCH_S, OFFSET_GUESS_S + OFFSET_SEARCH_S + 1e-9, 1 / fs):
        i0 = int(round(lag * fs))
        n = min(len(b), len(a) - i0)
        if i0 < 0 or n < 0.9 * len(b):
            continue
        aa, bb = a[i0:i0 + n], b[:n]
        if aa.std() > 0 and bb.std() > 0:
            r = abs(np.corrcoef(aa, bb)[0, 1])
            if r > best[1]:
                best = (float(lag), float(r))
    return best if best[1] >= MIN_OFFSET_CORR else (OFFSET_GUESS_S, best[1])


def load_pair(name, num, sec_path, mts_path):
    sec, mts = read_numeric_csv(sec_path), read_numeric_csv(mts_path)
    t_sec = sec[SEC_TIME_COL].to_numpy(float)
    t_sec = t_sec - t_sec[0]
    x = sec[SEC_COL].to_numpy(float)
    t_mts = mts[MTS_TIME_COL].to_numpy(float)
    t_mts = t_mts - t_mts[0]
    fs = (len(t_sec) - 1) / (t_sec[-1] - t_sec[0])
    offset, r = estimate_offset(t_sec, x, t_mts, mts[OFFSET_REF_COL].to_numpy(float))
    if RESAMPLE_HZ and RESAMPLE_HZ < fs:
        sos = signal.butter(4, 0.4 * RESAMPLE_HZ, btype="lowpass", fs=fs, output="sos")
        x = signal.sosfiltfilt(sos, x)
        t_new = np.arange(t_sec[0], t_sec[-1], 1 / RESAMPLE_HZ)
        x, t_sec, fs = np.interp(t_new, t_sec, x), t_new, RESAMPLE_HZ
    keep = (t_sec >= offset) & (t_sec <= offset + t_mts[-1])
    t_al = t_sec[keep] - offset
    targets = {c: np.interp(t_al, t_mts, mts[c].to_numpy(float)) for c in TARGET_COLS}
    on_edge = abs(abs(offset - OFFSET_GUESS_S) - OFFSET_SEARCH_S) < 0.11
    ok = bool(r >= GOOD_OFFSET_CORR and not on_edge)
    print(f"  {name}: offset {offset:.2f} s (|corr| {r:.2f}), {t_al[-1]:.0f} s aligned"
          + ("" if ok else "  <-- alignment UNRELIABLE"))
    return {"name": name, "num": num, "t": t_al, "x": x[keep], "fs": fs, "targets": targets,
            "offset": offset, "offset_corr": r, "offset_ok": ok}


# ============================================================
# FFT HELPERS
# ============================================================

def fft_spectrum(y, fs):
    """One-sided amplitude spectrum of a detrended, Hann-windowed series (zero-padded)."""
    y = signal.detrend(np.asarray(y, float))
    win = np.hanning(len(y))
    n = int(2 ** np.ceil(np.log2(len(y) * FFT_PAD)))
    return np.fft.rfftfreq(n, 1 / fs), 2 * np.abs(np.fft.rfft(y * win, n)) / win.sum()


def dominant_peak(f, amp, record_s):
    """(frequency, amplitude, % of band power within +-1 raw bin) of the strongest peak."""
    m = f >= max(FFT_MIN_HZ, FFT_MIN_CYCLES / record_s)     # power share over the whole spectrum
    fm, am = f[m], amp[m]
    look = fm <= FFT_MAX_HZ                                   # the peak itself is searched up to FFT_MAX_HZ
    if look.sum() < 3:
        return np.nan, np.nan, 0.0
    pk, _ = signal.find_peaks(am)                             # true local maxima only (not the band edge)
    pk = pk[look[pk]]
    if not pk.size:
        return np.nan, np.nan, 0.0
    i = int(pk[np.argmax(am[pk])])
    near = slice(max(0, i - FFT_PAD), i + FFT_PAD + 1)
    return float(fm[i]), float(am[i]), float(100 * np.sum(am[near] ** 2) / np.sum(am ** 2))


def signals_of(tr):
    return [("SEC", tr["x"])] + [(c, tr["targets"][c]) for c in TARGET_COLS]


# ============================================================
# 1) PERFORMANCE MATRIX
# ============================================================

def r2(y, p):
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def calib_xy(tr, c):
    k = slice(None, None, CAL_DECIMATE)
    x = tr["x"][k] - np.median(tr["x"])                 # zero per trial: baselines differ between trials
    return x, tr["targets"][c][k]


def performance(trials):
    pool = [tr for tr in trials if tr["offset_ok"]]      # calibration pool: reliably aligned trials
    rows, periods = [], []
    for tr in trials:
        row = {"Trial": tr["name"], "Alignment |r|": tr["offset_corr"]}
        for c in TARGET_COLS:
            short = c.split("_")[0]
            x, y = calib_xy(tr, c)
            row[f"SEC-{short} |r|"] = abs(np.corrcoef(x, y)[0, 1])
            row[f"SEC->{short} R2 (own trial)"] = r2(y, np.polyval(np.polyfit(x, y, CAL_DEGREE), x))
            others = [o for o in pool if o["name"] != tr["name"]]
            if others:
                xo = np.concatenate([calib_xy(o, c)[0] for o in others])
                yo = np.concatenate([calib_xy(o, c)[1] for o in others])
                row[f"SEC->{short} R2 (other trials)"] = r2(y, np.polyval(np.polyfit(xo, yo, CAL_DEGREE), x))
            nper = max(64, len(tr["x"]) // 4)
            fc, coh = signal.coherence(tr["x"], tr["targets"][c], fs=tr["fs"], nperseg=nper)
            _, pyy = signal.welch(tr["targets"][c], fs=tr["fs"], nperseg=nper)
            band = (fc >= COHERENCE_BAND_HZ[0]) & (fc <= COHERENCE_BAND_HZ[1])
            # weighted by the target's power: the loading cycle counts most, weak harmonics little
            row[f"Coherence SEC-{short}"] = (float(np.sum(coh[band] * pyy[band]) / np.sum(pyy[band]))
                                            if band.any() and pyy[band].sum() > 0 else np.nan)
        rec = tr["t"][-1] - tr["t"][0]
        prow = {"Trial": tr["name"], "Record_s": rec}
        for name, y in signals_of(tr):
            pf, _, pp = dominant_peak(*fft_spectrum(y, tr["fs"]), rec)
            prow[f"{name} period (s)"] = 1 / pf if pp >= FFT_MIN_POWER_PCT else np.nan
            prow[f"{name} peak power (%)"] = pp
        row["SEC peak power"] = prow["SEC peak power (%)"] / 100
        ps, pd_ = prow["SEC period (s)"], prow[f"{TARGET_COLS[0]} period (s)"]
        prow["SEC matches MTS"] = bool(np.isfinite(ps) and np.isfinite(pd_) and abs(ps - pd_) / pd_ < 0.1)
        rows.append(row)
        periods.append(prow)
    return pd.DataFrame(rows), pd.DataFrame(periods)


def plot_matrix(perf, periods, out_dir):
    cols = [c for c in perf.columns if c != "Trial"]
    a = perf[cols].to_numpy(float)
    cmap = LinearSegmentedColormap.from_list("blue", ["#f4f8fd", "#cde2fb", "#6da7ec", "#256abf", "#0d366b"])
    fig, ax = plt.subplots(figsize=(1.05 * len(cols) + 3.5, 0.62 * len(perf) + 2.6))
    im = ax.imshow(np.clip(a, 0, 1), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            v = a[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                        color="white" if v > 0.55 else INK)
            else:
                ax.text(j, i, "-", ha="center", va="center", fontsize=8.5, color=MUTED)
    ylab = []
    for _, p in periods.iterrows():
        ps, pd_ = p["SEC period (s)"], p[f"{TARGET_COLS[0]} period (s)"]
        tag = (f"T = {ps:.0f} / {pd_:.0f} s" if np.isfinite(ps) and np.isfinite(pd_) else "not periodic")
        ylab.append(f"{p['Trial']}   ({tag})")
    ax.set_yticks(range(len(perf)), ylab, fontsize=9)
    ax.set_xticks(range(len(cols)), cols, rotation=35, ha="right", fontsize=8.5)
    for i, ok in enumerate(perf["Alignment |r|"] >= GOOD_OFFSET_CORR):
        if not ok:
            ax.get_yticklabels()[i].set_color("#c0392b")
    ax.set_xticks(np.arange(-0.5, len(cols)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(perf)), minor=True)
    ax.grid(which="minor", color="white", lw=2)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("score (0 = none, 1 = perfect); values < 0 shown as 0", fontsize=8)
    ax.set_title("SEC sensor performance against the MTS target, per trial\n"
                 "(row label: dominant period SEC / MTS displacement; red = SEC-MTS alignment unreliable)",
                 fontsize=10, fontweight="bold", loc="left")
    fig.tight_layout()
    png = os.path.join(out_dir, "SEC_MTS_performance_matrix.png")
    fig.savefig(png, dpi=150)
    print(f"Saved: {png}")
    plt.show()
    plt.close(fig)


# ============================================================
# 2) LIVE DOMINANT PEAK (sliding-window FFT)
# ============================================================

def sliding_peaks(tr):
    """Window starts, window length and per-signal dominant peaks + spectra for every frame."""
    fs, n = tr["fs"], len(tr["x"])
    win_s = min(LIVE_WINDOW_S, 0.75 * (tr["t"][-1] - tr["t"][0]))
    w = int(round(win_s * fs))
    starts = np.arange(0, n - w + 1, max(1, int(round(LIVE_STEP_S * fs))))
    frames = []
    for s in starts:
        fr = {"start": s}
        for name, y in signals_of(tr):
            f, amp = fft_spectrum(y[s:s + w], fs)
            pf, pa, pp = dominant_peak(f, amp, win_s)
            fr[name] = (f, amp, pf, pa, pp)
        frames.append(fr)
    return frames, w, win_s


def peak_track(tr, frames, w):
    rows = []
    for fr in frames:
        row = {"Trial": tr["name"], "Window_centre_s": tr["t"][fr["start"] + w // 2]}
        for name, _ in signals_of(tr):
            _, _, pf, _, pp = fr[name]
            row[f"{name} period (s)"] = 1 / pf if np.isfinite(pf) else np.nan
            row[f"{name} peak power (%)"] = pp
        rows.append(row)
    return pd.DataFrame(rows)


def live_animation(tr, frames, w, win_s, out_dir):
    t = tr["t"]
    sigs = signals_of(tr)
    fig = plt.figure(figsize=(14, 7.2))
    gs = fig.add_gridspec(len(sigs), 2, width_ratios=[1.7, 1], hspace=0.35, wspace=0.18)
    ax_ts = [fig.add_subplot(gs[i, 0]) for i in range(len(sigs))]
    ax_f = fig.add_subplot(gs[:, 1])
    spans, cursors, lines, dots, labels = [], [], {}, {}, {}
    units = {"SEC": "pF", "Displacement_m": "m", "Force_N": "N"}
    for ax, (name, y) in zip(ax_ts, sigs):
        ax.plot(t, y, color=COLORS[name], lw=1.0)
        ax.set_ylabel(f"{'SEC' if name == 'SEC' else 'MTS ' + name.split('_')[0]}\n({units[name]})", fontsize=8)
        spans.append(ax.axvspan(t[0], t[w - 1], color="#eda100", alpha=0.18, lw=0))
        cursors.append(ax.axvline(t[w - 1], color="#eda100", lw=1.5))
        ax.grid(color=GRID, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(t[0], t[-1])
    ax_ts[-1].set_xlabel("Time (s), MTS clock")
    ax_ts[0].set_title("SEC sensor and MTS target - shaded = current FFT window", fontsize=9,
                       loc="left", fontweight="bold")
    for name, _ in sigs:
        lab = "SEC" if name == "SEC" else f"MTS {name.split('_')[0]} (target)"
        lines[name], = ax_f.plot([], [], color=COLORS[name], lw=1.8, label=lab)
        dots[name], = ax_f.plot([], [], "o", color=COLORS[name], ms=9, mec="white", mew=1.5)
    ax_f.set_xlim(0, FFT_MAX_HZ)
    ax_f.set_ylim(0, 1.25)
    ax_f.set_xlabel("Frequency (Hz)")
    ax_f.set_ylabel("Amplitude (normalised to each signal's peak)")
    ax_f.grid(color=GRID, lw=0.6)
    ax_f.spines[["top", "right"]].set_visible(False)
    ax_f.legend(loc="center right", frameon=False, fontsize=8)
    ax_f.set_title(f"Live FFT of the window ({win_s:.0f} s) - dot = dominant peak", fontsize=9,
                   loc="left", fontweight="bold")
    info = ax_f.text(0.02, 0.97, "", transform=ax_f.transAxes, va="top", fontsize=8.5, family="monospace",
                     color=INK)
    fig.suptitle(f"{tr['name']}: live dominant peak, SEC vs MTS target", fontweight="bold")

    def update(k):
        fr = frames[k]
        s = fr["start"]
        for sp, cu in zip(spans, cursors):
            if hasattr(sp, "set_width"):                      # matplotlib >= 3.9: axvspan is a Rectangle
                sp.set_x(t[s])
                sp.set_width(t[s + w - 1] - t[s])
            else:                                             # older matplotlib: a Polygon
                sp.set_xy([[t[s], 0], [t[s], 1], [t[s + w - 1], 1], [t[s + w - 1], 0]])
            cu.set_xdata([t[s + w - 1]])
        txt = [f"window {t[s]:5.1f}-{t[s + w - 1]:5.1f} s"]
        for name, _ in sigs:
            f, amp, pf, pa, pp = fr[name]
            band = (f >= FFT_MIN_HZ) & (f <= FFT_MAX_HZ)
            scale = amp[band].max() or 1.0
            lines[name].set_data(f[band], amp[band] / scale)
            periodic = np.isfinite(pf) and pp >= FFT_MIN_POWER_PCT
            dots[name].set_data([pf] if periodic else [], [pa / scale] if periodic else [])
            lab = "SEC" if name == "SEC" else name.split("_")[0]
            txt.append(f"{lab:<13}" + (f"T = {1 / pf:5.1f} s  ({pp:3.0f}% power)" if periodic
                                       else "not periodic"))
        info.set_text("\n".join(txt))
        return spans + cursors + list(lines.values()) + list(dots.values()) + [info]

    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000 / LIVE_FPS, blit=False)
    gif = os.path.join(out_dir, f"{tr['name']}_live_dominant_peak.gif")
    anim.save(gif, writer=animation.PillowWriter(fps=LIVE_FPS), dpi=80)
    plt.close(fig)
    print(f"Saved: {gif}  ({len(frames)} frames)")
    if IN_NOTEBOOK:
        display(Image(filename=gif))


def plot_period_over_time(tracks, out_dir):
    names = list(tracks)
    ncol = 3
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.0 * nrow), sharey=True, squeeze=False)
    for ax, name in zip(axes.flat, names):
        df = tracks[name]
        for sig in ["SEC"] + TARGET_COLS:
            per, pp = df[f"{sig} period (s)"], df[f"{sig} peak power (%)"]
            ok = pp >= FFT_MIN_POWER_PCT
            lab = "SEC" if sig == "SEC" else f"MTS {sig.split('_')[0]}"
            ax.plot(df["Window_centre_s"][ok], per[ok], "o-", color=COLORS[sig], ms=3.5, lw=1.4, label=lab)
            ax.plot(df["Window_centre_s"][~ok], per[~ok], "o", mfc="none", color=COLORS[sig], ms=3.5)
        ax.set_title(name, fontsize=9, fontweight="bold", loc="left")
        ax.set_ylim(0, 70)
        ax.grid(color=GRID, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Window centre (s)", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("Dominant period (s)", fontsize=8)
    for ax in list(axes.flat)[len(names):]:
        ax.set_visible(False)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("Dominant period over time (sliding-window FFT) - filled = periodic, hollow = "
                 f"weak peak (< {FFT_MIN_POWER_PCT:.0f}% power)", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    png = os.path.join(out_dir, "SEC_MTS_dominant_period_over_time.png")
    fig.savefig(png, dpi=150)
    print(f"Saved: {png}")
    plt.show()
    plt.close(fig)


# ============================================================
# RUN
# ============================================================

out_dir = (to_local_path(OUTPUT_DIR) if OUTPUT_DIR
           else os.path.join(to_local_path(DATA_DIR), "tda_results", "performance"))
os.makedirs(out_dir, exist_ok=True)
print("Loading and aligning:")
trials = [load_pair(*p) for p in find_pairs()]

perf, periods = performance(trials)
print("\nPerformance (MTS = target; R2 'other trials' = SEC calibrated on the other reliable trials):")
print(perf.round(3).to_string(index=False))
print("\nDominant periods (whole record):")
print(periods.round(2).to_string(index=False))
plot_matrix(perf, periods, out_dir)

tracks = {}
for tr in trials:
    frames, w, win_s = sliding_peaks(tr)
    tracks[tr["name"]] = peak_track(tr, frames, w)
    if LIVE and (not LIVE_TRIALS or tr["name"] in LIVE_TRIALS):
        live_animation(tr, frames, w, win_s, out_dir)
plot_period_over_time(tracks, out_dir)

xlsx = os.path.join(out_dir, "SEC_MTS_performance.xlsx")
with pd.ExcelWriter(xlsx, engine="openpyxl") as wr:
    perf.to_excel(wr, sheet_name="Performance", index=False)
    periods.to_excel(wr, sheet_name="Dominant_periods", index=False)
    pd.concat(tracks.values(), ignore_index=True).to_excel(wr, sheet_name="Peak_track", index=False)
print(f"Saved: {xlsx}")
