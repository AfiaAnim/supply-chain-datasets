"""
SEC + MTS: merge every trial into one dataset, with FastTDA and the Fourier transform. Paste into ONE Jupyter cell and run
(or: python sec_mts_merge_fasttda.py).

SEC  (trial_XX*.csv): time_s, C1_raw_pF, C1_clean_pF, C1_filtered_pF, dC1_pF
MTS  (trial_XX*.csv): RunningTime_s, Displacement_raw_m, Displacement_m, Force_raw_N, Force_N
Files may sit together in DATA_DIR (or in sec/ and mts/ subfolders); each one is recognised by its
columns (not its name) and paired by trial number, so trial_10.csv (MTS) + trial_10_1.csv (SEC) work.

Output (in DATA_DIR/tda_results/merged):
    SEC_all_trials.csv                   every SEC file stacked (original samples, time re-zeroed)
    MTS_all_trials.csv                   every MTS file stacked (original samples, time re-zeroed)
    SEC_MTS_aligned_all_trials.csv       SEC aligned to the MTS (cross-correlation offset), 20 Hz, with
                                         displacement / force, loading flag, cycle number and FastTDA
                                         (one value per sample, window centred on it) for every setting
    SEC_MTS_FastTDA_windows_all_trials.csv  one row per window (sliding by WINDOW_STEP_S): FastTDA,
                                         capacitance level / spread / slope, window means of the targets
    SEC_MTS_merged_summary.xlsx          Trials (files, lengths, offset, alignment, cycles), FastTDA
                                         summary per trial, and the window table
    SEC_MTS_merged_overview.png          all trials on one merged timeline: capacitance, displacement,
                                         force and FastTDA
    SEC_MTS_FFT_spectra_all_trials.csv   full-length FFT |X(f)| of SEC, displacement, force and FastTDA
                                         for every trial (long format: Trial, Signal, Freq_Hz, ...)
    SEC_MTS_merged_fft.png               spectra of all trials overlaid per signal, and the dominant
                                         frequency of every signal in every trial
    (the Trials sheet / table also lists each signal's dominant peak, period and power share)

Requirements:
    pip install numpy pandas scipy matplotlib openpyxl
"""

import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal

try:                                    # make plots appear inside Jupyter
    from IPython import get_ipython
    if get_ipython() is not None:
        get_ipython().run_line_magic("matplotlib", "inline")
except Exception:
    pass


# ============================================================
# SETTINGS
# ============================================================

DATA_DIR = "."                          # folder with the SEC and MTS csv files (or sec\ and mts\ subfolders)
SEC_SUBDIR, MTS_SUBDIR = "sec", "mts"
OUTPUT_DIR = None                       # None = DATA_DIR/tda_results/merged
TRIALS = None                           # None = every trial found (1-15); or e.g. ["trial_04", "trial_10"]

SEC_TIME_COL = "time_s"
SEC_COL = "C1_filtered_pF"              # signal used for FastTDA
MTS_TIME_COL = "RunningTime_s"
TARGET_COLS = ["Displacement_m", "Force_N"]
RESAMPLE_HZ = 20.0                      # aligned dataset rate (SEC is anti-aliased, then resampled)

OFFSET_GUESS_S = 10.0                   # SEC starts ~10 s before the MTS
OFFSET_SEARCH_S = 5.0
OFFSET_REF_COL = "Displacement_m"
MIN_OFFSET_CORR = 0.2
GOOD_OFFSET_CORR = 0.7                  # weaker alignment -> Alignment_OK = False (trials 01-03)

# FastTDA settings: (window s, Takens delay s). 8 s / 1 s = the validation setting; 40 s / 10 s = one
# loading period with a quarter-period delay (the dominant FFT period of trials 04-15).
FASTTDA_SETTINGS = [(8.0, 1.0), (40.0, 10.0)]
TAKENS_DIM = 2
MAX_POINTS = 300                        # long windows: embed every k-th point so a cloud has <= this many
WINDOW_STEP_S = 1.0                     # step of the per-window table
LOADING_PERIOD_S = 40.0                 # used to find the loading cycles
GAP_S = 10.0                            # gap between trials on the merged timeline (plot / Merged_time_s)

# Fourier transform: full aligned record of each trial, mean removed, |FFT| from 0 to Nyquist
FFT_MIN_HZ = 0.01                       # slower components (drift) are not reported as peaks
FFT_MIN_CYCLES = 3                      # a peak's period must fit this many times into the record
FFT_MIN_POWER_PCT = 20.0                # dominant peak must hold >= this % of the spectral power
FFT_N_PEAKS = 3                         # peaks listed per signal (dominant + next strongest)
FFT_F_MAX_PLOT = 0.2                    # x-axis limit of the spectra (loading = 0.025 Hz + harmonics)
FFT_FASTTDA = FASTTDA_SETTINGS[-1][0]   # FastTDA setting whose time series is also transformed
FFT_SPIKE_MADS = 6.0                    # FastTDA values this many robust SDs from the median (rare bad
                                        # ellipse fits) are interpolated over before its FFT

COLORS = {"cap": "#3d3d3a", "disp": "#6250d6", "force": "#e87ba4", "fasttda": "#eb6834", "bad": "#e34948"}


# ============================================================
# FILES
# ============================================================

def to_local_path(path):
    """Convert C:\\... to /mnt/c/... when running under Linux/WSL."""
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
    """'sec' or 'mts' from the csv header (file names are not reliable)."""
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
    for p in sorted(set(files)):
        name, num = trial_key(p)
        kind = file_kind(p) if name else None
        if not name or kind is None:
            print(f"  ignored {os.path.basename(p)}: " + ("no 'trial_NN' in the name" if not name
                                                          else "columns not recognised as SEC or MTS"))
            continue
        group = sec if kind == "sec" else mts
        if name in group:
            print(f"  {name}: two {kind.upper()} files - using {os.path.basename(group[name][1])}")
            continue
        group[name] = (num, p)
    if not sec:
        raise FileNotFoundError(f"No SEC csv files (with a {SEC_COL!r} column) found in {os.path.abspath(base)}")
    pairs = []
    for name in sorted(set(sec) | set(mts), key=lambda n: int(n.split("_")[1])):
        if TRIALS and name not in TRIALS:
            continue
        if name not in sec or name not in mts:
            print(f"  {name}: {'SEC' if name in sec else 'MTS'} file only - skipped")
            continue
        pairs.append((name, sec[name][0], sec[name][1], mts[name][1]))
        print(f"  {name}: SEC {os.path.basename(sec[name][1])}  +  MTS {os.path.basename(mts[name][1])}")
    print(f"Found {len(pairs)} SEC/MTS pairs in {os.path.abspath(base)}")
    return pairs


# ============================================================
# ALIGNMENT
# ============================================================

def estimate_offset(t_sec, x_sec, t_mts, y_mts):
    """Offset (s) with MTS time = SEC time - offset, from the cross-correlation peak."""
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
        if aa.std() == 0 or bb.std() == 0:
            continue
        r = abs(np.corrcoef(aa, bb)[0, 1])
        if r > best[1]:
            best = (float(lag), float(r))
    return best if best[1] >= MIN_OFFSET_CORR else (OFFSET_GUESS_S, best[1])


def load_trial(name, num, sec_path, mts_path):
    sec, mts = read_numeric_csv(sec_path), read_numeric_csv(mts_path)
    for c in (SEC_TIME_COL, SEC_COL):
        if c not in sec:
            raise KeyError(f"{sec_path}: column {c!r} not found. Columns: {list(sec.columns)}")
    for c in [MTS_TIME_COL] + TARGET_COLS:
        if c not in mts:
            raise KeyError(f"{mts_path}: column {c!r} not found. Columns: {list(mts.columns)}")
    sec[SEC_TIME_COL] = sec[SEC_TIME_COL] - sec[SEC_TIME_COL].iloc[0]       # re-zero both clocks
    mts[MTS_TIME_COL] = mts[MTS_TIME_COL] - mts[MTS_TIME_COL].iloc[0]
    t_sec, x = sec[SEC_TIME_COL].to_numpy(float), sec[SEC_COL].to_numpy(float)
    t_mts = mts[MTS_TIME_COL].to_numpy(float)
    fs_sec = (len(t_sec) - 1) / (t_sec[-1] - t_sec[0])
    offset, r = estimate_offset(t_sec, x, t_mts, mts[OFFSET_REF_COL].to_numpy(float))
    if RESAMPLE_HZ and RESAMPLE_HZ < fs_sec:                       # anti-alias, then resample
        sos = signal.butter(4, 0.4 * RESAMPLE_HZ, btype="lowpass", fs=fs_sec, output="sos")
        x = signal.sosfiltfilt(sos, x)
    fs = RESAMPLE_HZ or fs_sec
    t_al = np.arange(0, min(t_mts[-1], t_sec[-1] - offset), 1 / fs)  # aligned time = MTS time
    on_edge = abs(abs(offset - OFFSET_GUESS_S) - OFFSET_SEARCH_S) < 0.11
    reliable = bool(np.isfinite(r) and r >= GOOD_OFFSET_CORR and not on_edge)
    print(f"  {name}: SEC {t_sec[-1]:.1f} s, MTS {t_mts[-1]:.1f} s | offset {offset:.2f} s (|corr| {r:.2f})"
          + ("" if reliable else "  <-- alignment UNRELIABLE"))
    return {"name": name, "num": num, "sec": sec, "mts": mts, "sec_file": os.path.basename(sec_path),
            "mts_file": os.path.basename(mts_path), "fs": fs, "t": t_al, "offset": offset,
            "offset_corr": r, "offset_ok": reliable,
            "x": np.interp(t_al + offset, t_sec, x),
            "targets": {c: np.interp(t_al, t_mts, mts[c].to_numpy(float)) for c in TARGET_COLS}}


def cycles_of(d, fs):
    """(start, compression peak, end) index of every loading cycle deeper than 80% of the stroke."""
    dist = int(0.7 * LOADING_PERIOD_S * fs)
    if np.ptp(d) == 0:
        return []
    tops = list(signal.find_peaks(d, distance=dist, prominence=0.5 * np.ptp(d))[0])
    if not tops or tops[0] > dist // 2:
        tops = [0] + tops
    if tops[-1] < len(d) - dist // 2:
        tops = tops + [len(d) - 1]
    out = []
    for a, b in zip(tops[:-1], tops[1:]):
        pk = a + int(np.argmin(d[a:b + 1]))
        if -d[pk] >= 0.8 * np.max(-d):
            out.append((a, pk, b))
    return out


# ============================================================
# FASTTDA
# ============================================================

def point_clouds(x, window, delay, step):
    """Takens point clouds (dimension TAKENS_DIM) of the min-max scaled windows."""
    W = sliding_window_view(x, window)[::step].astype(float)
    lo, hi = W.min(axis=1, keepdims=True), W.max(axis=1, keepdims=True)
    W = 2 * (W - lo) / np.where(hi > lo, hi - lo, 1.0) - 1
    span = window - (TAKENS_DIM - 1) * delay - 1
    stride = max(1, int(np.ceil((span + 1) / MAX_POINTS)))
    first = span % stride
    last = first + (span // stride) * stride
    cols = [W[:, first + d * delay: last + d * delay + 1: stride] for d in range(TAKENS_DIM)]
    return np.arange(0, len(x) - window + 1, step), np.stack(cols, axis=2)


def fasttda_minor_axis(pc):
    """SVD-LSE ellipse fit (A + C = 1): minor axis 2b, NaN when the best conic is not an ellipse."""
    X, Y = pc[:, 0] - pc[:, 0].mean(), pc[:, 1] - pc[:, 1].mean()
    rng = max(np.abs(X).max(), np.abs(Y).max())
    if not np.isfinite(rng) or rng == 0:
        return np.nan
    X, Y = X * 1.5 / rng, Y * 1.5 / rng
    G = np.column_stack([X**2 - Y**2, X * Y, X, Y, np.ones_like(X)])
    U, S, Vt = np.linalg.svd(G, full_matrices=False)
    Sinv = np.where(S > np.finfo(float).eps * max(G.shape) * S[0], 1.0 / S, 0.0)
    A, B, D, E, F = (Vt.T * Sinv) @ (U.T @ -(Y**2))
    C = 1.0 - A
    if 4 * A * C - B**2 <= 0:
        return np.nan
    th = 0.5 * np.arctan2(B, A - C)
    c, s = np.cos(th), np.sin(th)
    Ap, Cp = A*c*c + B*c*s + C*s*s, A*s*s - B*c*s + C*c*c
    Dp, Ep = D*c + E*s, -D*s + E*c
    if Ap <= 0 or Cp <= 0:
        return np.nan
    Fn = F - Dp**2 / (4 * Ap) - Ep**2 / (4 * Cp)
    return np.nan if Fn >= 0 else 2.0 * min(np.sqrt(-Fn / Ap), np.sqrt(-Fn / Cp))


def tag(ws):
    return f"W{ws:g}s"


def fasttda_per_sample(tr, ws, ds):
    """FastTDA of the window centred on every sample (NaN where the window does not fit)."""
    w, d = int(round(ws * tr["fs"])), max(1, int(round(ds * tr["fs"])))
    out = np.full(len(tr["x"]), np.nan)
    if len(tr["x"]) < w:
        return out
    starts, clouds = point_clouds(tr["x"], w, d, step=1)
    out[starts + w // 2] = [fasttda_minor_axis(pc) for pc in clouds]
    return out


def window_table(tr, ws, ds, cycle, loading):
    """One row per window sliding by WINDOW_STEP_S: FastTDA, capacitance level / spread / slope and
    the window means of the targets."""
    fs = tr["fs"]
    w, d = int(round(ws * fs)), max(1, int(round(ds * fs)))
    step = max(1, int(round(WINDOW_STEP_S * fs)))
    if len(tr["x"]) < w:
        return pd.DataFrame()
    starts, clouds = point_clouds(tr["x"], w, d, step)
    ftda = np.array([fasttda_minor_axis(pc) for pc in clouds])
    X = sliding_window_view(tr["x"], w)[::step]
    tt = (np.arange(w) - (w - 1) / 2) / fs
    mid = starts + w // 2
    f = pd.DataFrame({"Trial": tr["name"], "Alignment_OK": tr["offset_ok"], "Window_s": ws, "Delay_s": d / fs,
                      "Start_s": tr["t"][starts], "End_s": tr["t"][starts + w - 1], "Centre_s": tr["t"][mid],
                      "Cycle": cycle[mid], "Loading": loading[mid],
                      "FastTDA_2b": ftda, "FastTDA_FitOK": np.isfinite(ftda),
                      "Cap_mean_pF": X.mean(axis=1), "Cap_std_pF": X.std(axis=1),
                      "Cap_slope_pF_per_s": (X - X.mean(axis=1, keepdims=True)) @ tt / (tt @ tt)})
    f["Cap_mean_zeroed_pF"] = f["Cap_mean_pF"] - np.median(tr["x"])
    for c in TARGET_COLS:
        f[f"{c}:mean"] = sliding_window_view(tr["targets"][c], w)[::step].mean(axis=1)
    return f


# ============================================================
# MERGE
# ============================================================

def merge(trials):
    sec_all = pd.concat([tr["sec"].assign(Trial=tr["name"], Trial_num=tr["num"], Source_file=tr["sec_file"])
                         for tr in trials], ignore_index=True)
    mts_all = pd.concat([tr["mts"].assign(Trial=tr["name"], Trial_num=tr["num"], Source_file=tr["mts_file"])
                         for tr in trials], ignore_index=True)
    front = ["Trial", "Trial_num", "Source_file"]
    sec_all = sec_all[front + [c for c in sec_all if c not in front]]
    mts_all = mts_all[front + [c for c in mts_all if c not in front]]

    aligned, windows, rows = [], [], []
    t0 = 0.0
    for tr in trials:
        d = tr["targets"]["Displacement_m"]
        cyc = cycles_of(d, tr["fs"])
        cycle = np.zeros(len(d), int)
        for k, (a, _, b) in enumerate(cyc):
            cycle[a:b + 1] = k + 1
        smooth = signal.savgol_filter(d, max(5, int(tr["fs"]) | 1), 2) if len(d) > tr["fs"] else d
        loading = np.gradient(smooth) < 0                         # pressing: displacement more negative
        a = pd.DataFrame({"Trial": tr["name"], "Trial_num": tr["num"], "Alignment_OK": tr["offset_ok"],
                          "Time_s": tr["t"], "Merged_time_s": t0 + tr["t"], "SEC_time_s": tr["t"] + tr["offset"],
                          SEC_COL: tr["x"], f"{SEC_COL}_zeroed": tr["x"] - np.median(tr["x"]),
                          **{c: tr["targets"][c] for c in TARGET_COLS},
                          "Displacement_mm": 1000 * d, "Loading": loading, "Cycle": cycle})
        info = {"Trial": tr["name"], "SEC_file": tr["sec_file"], "MTS_file": tr["mts_file"],
                "SEC_samples": len(tr["sec"]), "MTS_samples": len(tr["mts"]),
                "SEC_record_s": float(tr["sec"][SEC_TIME_COL].iloc[-1]),
                "MTS_record_s": float(tr["mts"][MTS_TIME_COL].iloc[-1]),
                "Offset_s": tr["offset"], "Alignment_|r|": tr["offset_corr"], "Alignment_OK": tr["offset_ok"],
                "Aligned_samples": len(a), "Loading_cycles": len(cyc),
                "Stroke_mm": float(1000 * np.ptp(d)), "Peak_force_N": float(np.max(np.abs(tr["targets"]["Force_N"])))}
        for ws, ds in FASTTDA_SETTINGS:
            v = fasttda_per_sample(tr, ws, ds)
            a[f"FastTDA_{tag(ws)}"] = v
            a[f"FastTDA_{tag(ws)}_FitOK"] = np.isfinite(v)
            covered = np.zeros(len(v), bool)
            w = int(round(ws * tr["fs"]))
            covered[w // 2: len(v) - (w - 1 - w // 2)] = True     # samples whose centred window fits
            info[f"FastTDA_{tag(ws)}_fit_ok_pct"] = (100 * float(np.mean(np.isfinite(v[covered])))
                                                     if covered.any() else np.nan)
            info[f"FastTDA_{tag(ws)}_mean_2b"] = float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
            windows.append(window_table(tr, ws, ds, cycle, loading))
        aligned.append(a)
        rows.append(info)
        print(f"  {tr['name']}: {len(a)} aligned samples, {len(cyc)} loading cycles, FastTDA fit ok "
              + ", ".join(f"{tag(ws)} {info[f'FastTDA_{tag(ws)}_fit_ok_pct']:.0f}%" for ws, _ in FASTTDA_SETTINGS))
        t0 += tr["t"][-1] + GAP_S
    return sec_all, mts_all, pd.concat(aligned, ignore_index=True), \
        pd.concat(windows, ignore_index=True), pd.DataFrame(rows)


# ============================================================
# FOURIER TRANSFORM (full aligned record of each trial)
# ============================================================

FFT_SIGNALS = [(f"{SEC_COL}", "SEC capacitance", COLORS["cap"]),
               ("Displacement_m", "MTS displacement", COLORS["disp"]),
               ("Force_N", "MTS force", COLORS["force"]),
               (f"FastTDA_{tag(FFT_FASTTDA)}", f"FastTDA {tag(FFT_FASTTDA)}", COLORS["fasttda"])]


def full_fft(y, fs):
    """|FFT| of the whole record (mean removed), bins 0 .. Nyquist; gaps (failed FastTDA fits) are
    interpolated and the edges where no window fits are dropped first."""
    y = np.asarray(y, float)
    ok = np.isfinite(y)
    if ok.sum() < 16:
        return np.array([]), np.array([])
    i0, i1 = np.flatnonzero(ok)[[0, -1]]
    y = y[i0:i1 + 1]
    k = np.arange(len(y))
    y = np.interp(k, k[np.isfinite(y)], y[np.isfinite(y)])
    y = y - y.mean()
    n = len(y)
    return np.arange(n // 2) / (n / fs), np.abs(np.fft.fft(y))[: n // 2]


def fft_peaks(f, amp, record_s):
    """Strongest local maxima whose period fits FFT_MIN_CYCLES times: [(Hz, |X|, % of power)]."""
    if not len(f):
        return []
    pk = signal.find_peaks(amp)[0]                               # whole spectrum, then filter
    pk = pk[f[pk] >= max(FFT_MIN_HZ, FFT_MIN_CYCLES / record_s)]
    pk = pk[np.argsort(amp[pk])[::-1][:FFT_N_PEAKS]]
    total = np.sum(amp[1:] ** 2)
    return [(float(f[i]), float(amp[i]), float(100 * amp[i] ** 2 / total)) for i in pk]


def fourier(aligned, summary):
    """Spectra (long table) and the dominant peak of every signal, added to the trial summary."""
    spectra, peaks = [], []
    for name, g in aligned.groupby("Trial", sort=False):
        fs = 1 / np.median(np.diff(g["Time_s"]))
        record = g["Time_s"].iloc[-1] - g["Time_s"].iloc[0]
        row = {"Trial": name}
        for col, label, _ in FFT_SIGNALS:
            y = g[col].to_numpy(float)
            if col.startswith("FastTDA"):
                med = np.nanmedian(y)
                mad = 1.4826 * np.nanmedian(np.abs(y - med))
                y = np.where(np.abs(y - med) > FFT_SPIKE_MADS * mad, np.nan, y)   # gaps get interpolated
            f, amp = full_fft(y, fs)
            pk = fft_peaks(f, amp, record)
            dom = bool(pk) and pk[0][2] >= FFT_MIN_POWER_PCT
            short = label.replace("MTS ", "").replace("SEC ", "").replace(" ", "_")
            row[f"{short}_peak_Hz"] = pk[0][0] if dom else np.nan
            row[f"{short}_period_s"] = 1 / pk[0][0] if dom else np.nan
            row[f"{short}_peak_power_pct"] = pk[0][2] if pk else np.nan
            row[f"{short}_other_peaks_Hz"] = ", ".join(f"{p[0]:.4f}" for p in pk[1:])
            if len(f):
                spectra.append(pd.DataFrame({"Trial": name, "Signal": label, "Freq_Hz": f, "Amplitude": amp,
                                             "Amplitude_norm": amp / amp[1:].max()}))
        peaks.append(row)
    return pd.concat(spectra, ignore_index=True), summary.merge(pd.DataFrame(peaks), on="Trial")


def plot_fft(spectra, summary, out_dir):
    names = list(summary["Trial"])
    cmap = plt.get_cmap("viridis")
    tcol = {n: cmap(i / max(1, len(names) - 1)) for i, n in enumerate(names)}
    bad = set(summary.loc[~summary["Alignment_OK"], "Trial"])
    fig = plt.figure(figsize=(18, 9.5))
    gs = fig.add_gridspec(2, len(FFT_SIGNALS), height_ratios=[1.2, 1], hspace=0.38)
    for k, (col, label, c) in enumerate(FFT_SIGNALS):
        ax = fig.add_subplot(gs[0, k])
        for n in names:
            sp = spectra[(spectra["Trial"] == n) & (spectra["Signal"] == label)]
            sp = sp[sp["Freq_Hz"] <= FFT_F_MAX_PLOT]
            ax.plot(sp["Freq_Hz"], sp["Amplitude_norm"], color=tcol[n], lw=1.1,
                    ls="--" if n in bad else "-", label=n.replace("trial_", "T"))
        ax.axvline(1 / LOADING_PERIOD_S, color="#c0392b", lw=0.8, ls=":")
        ax.text(1 / LOADING_PERIOD_S, 1.02, f"{1 / LOADING_PERIOD_S:g} Hz", color="#c0392b", fontsize=8,
                ha="center", transform=ax.get_xaxis_transform())
        ax.set_xlim(0, FFT_F_MAX_PLOT)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Frequency (Hz)")
        if k == 0:
            ax.set_ylabel("|FFT| / its maximum")
        ax.set_title(label, fontsize=10, fontweight="bold", color=c, pad=14)
        ax.grid(color="#e4e3df", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    h, l = fig.axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8, loc="center right", title="Trial\n(dashed =\nunreliable)",
               title_fontsize=8)
    ax = fig.add_subplot(gs[1, :])
    x = np.arange(len(names))
    for k, (col, label, c) in enumerate(FFT_SIGNALS):
        short = label.replace("MTS ", "").replace("SEC ", "").replace(" ", "_")
        per = summary[f"{short}_period_s"].to_numpy(float)
        pw = summary[f"{short}_peak_power_pct"].fillna(0).to_numpy(float)
        xs = x + (k - (len(FFT_SIGNALS) - 1) / 2) * 0.17
        ax.scatter(xs, per, s=20 + 3 * pw, color=c, label=f"{label} (size = power share)", zorder=3)
        none = ~np.isfinite(per)
        ax.scatter(xs[none], np.zeros(none.sum()), marker="x", color=c, s=30, zorder=3)
    ax.axhline(LOADING_PERIOD_S, color="#c0392b", lw=0.8, ls=":")
    ax.set_ylim(-2, 1.6 * LOADING_PERIOD_S)                     # from 0: bin-size differences stay small
    ax.set_xticks(x, [n.replace("trial_", "T") for n in names])
    ax.set_ylabel("Dominant period (s)\n(x at 0 = no dominant peak)")
    ax.set_title(f"Dominant peak of every signal in every trial (needs >= {FFT_MIN_POWER_PCT:g}% of the power "
                 f"and >= {FFT_MIN_CYCLES} cycles in the record)", fontsize=10, fontweight="bold")
    ax.legend(frameon=False, fontsize=8, ncol=len(FFT_SIGNALS), loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.grid(axis="y", color="#e4e3df", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Fourier transform of every trial (full aligned record, mean removed)", fontweight="bold")
    fig.subplots_adjust(left=0.06, right=0.91, top=0.9, bottom=0.12)
    png = os.path.join(out_dir, "SEC_MTS_merged_fft.png")
    fig.savefig(png, dpi=150)
    print(f"Saved: {png}")
    plt.show()
    plt.close(fig)


def plot_overview(aligned, summary, out_dir):
    ws_plot = FASTTDA_SETTINGS[0][0]
    fig, axes = plt.subplots(4, 1, figsize=(18, 10), sharex=True)
    panels = [(f"{SEC_COL}_zeroed", "Capacitance\n(zeroed, pF)", COLORS["cap"]),
              ("Displacement_mm", "Displacement (mm)", COLORS["disp"]),
              ("Force_N", "Force (N)", COLORS["force"]),
              (f"FastTDA_{tag(ws_plot)}", f"FastTDA 2b\n({tag(ws_plot)})", COLORS["fasttda"])]
    for ax, (col, label, c) in zip(axes, panels):
        for _, g in aligned.groupby("Trial", sort=False):
            if col.startswith("FastTDA"):
                ax.plot(g["Merged_time_s"], g[col], ".", color=c, ms=1.2)
            else:
                ax.plot(g["Merged_time_s"], g[col], color=c, lw=0.8)
        ax.set_ylabel(label)
        if col.startswith("FastTDA"):                            # a few bad fits would squash the scale
            top = np.nanpercentile(aligned[col], 99.5)
            ax.set_ylim(0, 1.15 * top)
            n_off = int((aligned[col] > 1.15 * top).sum())
            if n_off:
                ax.text(0.995, 0.95, f"{n_off} values above the axis", transform=ax.transAxes,
                        ha="right", va="top", fontsize=8, color="#8a8984")
        ax.grid(axis="y", color="#e4e3df", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    for _, g in aligned.groupby("Trial", sort=False):
        lo, hi = g["Merged_time_s"].iloc[0], g["Merged_time_s"].iloc[-1]
        ok = bool(g["Alignment_OK"].iloc[0])
        for ax in axes:
            if not ok:
                ax.axvspan(lo, hi, color=COLORS["bad"], alpha=0.07, lw=0)
        axes[0].text((lo + hi) / 2, 1.04, g["Trial"].iloc[0].replace("trial_", "T"), ha="center",
                     transform=axes[0].get_xaxis_transform(), fontsize=8,
                     color="#3d3d3a" if ok else COLORS["bad"], fontweight="bold")
    axes[-1].set_xlabel(f"Merged time (s): trials back to back, {GAP_S:g} s apart")
    n_bad = int((~summary["Alignment_OK"]).sum())
    fig.suptitle(f"All {len(summary)} trials merged (SEC aligned to MTS, {RESAMPLE_HZ:g} Hz)"
                 + (f"; red = alignment unreliable ({n_bad} trials)" if n_bad else ""),
                 fontweight="bold")
    fig.tight_layout()
    png = os.path.join(out_dir, "SEC_MTS_merged_overview.png")
    fig.savefig(png, dpi=150)
    print(f"Saved: {png}")
    plt.show()
    plt.close(fig)


# ============================================================
# RUN
# ============================================================

out_dir = to_local_path(OUTPUT_DIR) if OUTPUT_DIR else os.path.join(to_local_path(DATA_DIR), "tda_results", "merged")
os.makedirs(out_dir, exist_ok=True)
print("Pairing files:")
pairs = find_pairs()
print("\nLoading and aligning:")
trials = [load_trial(*p) for p in pairs]
print(f"\nMerging and FastTDA ({', '.join(f'{ws:g} s window / {ds:g} s delay' for ws, ds in FASTTDA_SETTINGS)}):")
sec_all, mts_all, aligned, windows, summary = merge(trials)
print("\nFourier transform (full aligned record of each trial):")
spectra, summary = fourier(aligned, summary)
fft_cols = ["Trial"] + [c for c in summary if c.endswith(("_period_s", "_peak_power_pct"))]
print(summary[fft_cols].round(2).to_string(index=False))

files = {"SEC_all_trials.csv": sec_all, "MTS_all_trials.csv": mts_all,
         "SEC_MTS_aligned_all_trials.csv": aligned, "SEC_MTS_FastTDA_windows_all_trials.csv": windows,
         "SEC_MTS_FFT_spectra_all_trials.csv": spectra}
for name, frame in files.items():
    path = os.path.join(out_dir, name)
    frame.to_csv(path, index=False)
    print(f"Saved: {path}  ({len(frame)} rows x {frame.shape[1]} columns)")
xlsx = os.path.join(out_dir, "SEC_MTS_merged_summary.xlsx")
with pd.ExcelWriter(xlsx, engine="openpyxl") as wr:
    summary.to_excel(wr, sheet_name="Trials", index=False)
    windows.to_excel(wr, sheet_name="FastTDA_windows", index=False)
print(f"Saved: {xlsx}")

print("\nTrials merged:")
print(summary.round(3).to_string(index=False))
plot_overview(aligned, summary, out_dir)
plot_fft(spectra, summary, out_dir)
