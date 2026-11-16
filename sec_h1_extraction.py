"""
Extract max H1 persistence (Vietoris-Rips) from the SEC capacitance and save it to Excel, with the
MTS displacement / force as the target. Same method as dropbear_h1_extraction.py, adapted to the
SEC / MTS tire data. Paste into ONE Jupyter cell and run (or: python sec_h1_extraction.py).

SEC  (trial_XX_1.csv): time_s, C1_raw_pF, C1_clean_pF, C1_filtered_pF, dC1_pF   (~261 Hz, uneven)
MTS  (trial_XX.csv)  : RunningTime_s, Displacement_raw_m, Displacement_m, Force_raw_N, Force_N
Files are recognised by their columns and paired by trial number.

Per trial:
    1. SEC onto a uniform grid at its own rate, low-pass at pass_value, resample to freq Hz
    2. align to the MTS (cross-correlation offset, ~9 s) and take the target on the same grid
    3. sliding windows (window from min_f / max_f, step s_W), each min-max scaled to [-1, 1]
    4. Takens embedding (dimension 2, delay tau, stride s) -> Vietoris-Rips H1 -> max persistence
    5. Time / target at the END of each window (real-time, as in the DROPBEAR code)

Output (in OUTPUT_DIR, default DATA_DIR/tda_results/h1):
    H1_Features_trial{NN}_minf{min_f}_maxf{max_f}.xlsx    one per trial (Features + Settings sheets)
    H1_Features_all_trials_minf..._maxf....xlsx           all trials together + per-trial summary
    SEC_H1_over_time.png                                  Max H1, capacitance and displacement per trial
    SEC_H1_by_trial.png                                   Max H1 per trial (condition indicator)

Requirements:
    pip install numpy pandas scipy matplotlib openpyxl ripser
    (or giotto-tda on Python <= 3.11 - used automatically when installed)
"""
import glob
import os
import re
import time

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


# === Parameters (from the SEC / MTS data, see the notes) ===
DATA_DIR = "."                # folder with the SEC and MTS csv files (or sec\ and mts\ subfolders)
OUTPUT_DIR = None             # None = DATA_DIR/tda_results/h1
trial_ids = None              # None = every trial found; or e.g. [10] or [10, 11, 12]
sec_col = "C1_raw_pF"         # input signal (DROPBEAR: acc); low-passed below
target_cols = ["Displacement_m", "Force_N"]   # targets (DROPBEAR: pinloc)

freq = 20                     # analysis rate (Hz) after low-pass + resampling
max_f = 0.075                 # highest significant frequency (3rd harmonic of the loading)
min_f = 0.025                 # lowest: the 40 s loading cycle
pass_value = 0.25             # low-pass cutoff (Hz): 3.3 x max_f, as in DROPBEAR (100 / 31.1)
s = 3                         # Takens stride: ~300 points per point cloud
s_W = 10                      # window step in samples (10 = 0.5 s)
window_size = None            # None = (1 / min_f + 2 * 0.25 / max_f) * freq  -> 933 samples (46.7 s)
tau = None                    # None = (0.25 / max_f) * freq / 2              ->  33 samples (1.67 s)
# Within-cycle tracking instead of one full loading cycle per window: min_f = max_f = 0.075
# (20 s window, same delay).

SKIP_START_S = 0.0            # drop windows ending before this time (files keep all others)
FIRST_CYCLE_S = 40.0          # windows starting before this include the first (conditioning) loading
                              # cycle: flagged in the files and left out of the per-trial summary
OFFSET_GUESS_S, OFFSET_SEARCH_S = 10.0, 5.0     # SEC starts ~10 s before the MTS
GOOD_OFFSET_CORR = 0.7        # weaker alignment -> trial flagged (trials 01-03)
N_JOBS = -1                   # giotto-tda only: CPU cores (-1 = all)


# === Files ===
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


def find_pairs():
    base = to_local_path(DATA_DIR)
    files = set(glob.glob(os.path.join(base, "*.csv")) + glob.glob(os.path.join(base, "sec", "*.csv"))
                + glob.glob(os.path.join(base, "mts", "*.csv")))
    sec, mts = {}, {}
    for p in sorted(files):
        m = re.search(r"trial[_\- ]?(\d+)", os.path.basename(p), re.I)
        if not m:
            continue
        with open(p, "r", encoding="utf-8-sig", errors="replace") as f:
            header = f.readline()
        num = int(m.group(1))
        if sec_col in header and "time_s" in header:
            sec.setdefault(num, p)
        elif "RunningTime_s" in header and all(c in header for c in target_cols):
            mts.setdefault(num, p)
    pairs = [(n, sec[n], mts[n]) for n in sorted(set(sec) & set(mts)) if not trial_ids or n in trial_ids]
    for n in sorted(set(sec) ^ set(mts)):
        print(f"  trial_{n:02d}: {'SEC' if n in sec else 'MTS'} file only - skipped")
    if not pairs:
        raise FileNotFoundError(f"No SEC/MTS pairs found in {os.path.abspath(base)}")
    print(f"Found {len(pairs)} SEC/MTS pairs in {os.path.abspath(base)}")
    return pairs


# === Signal Processing ===
def lowpass(x, pass_value, fs, N=6):
    sos = signal.butter(N=N, Wn=pass_value, btype="lowpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x)


def estimate_offset(t_sec, x_sec, t_mts, y_mts):
    """Offset (s) with MTS time = SEC time - offset, from the cross-correlation peak."""
    fs = 10.0
    a = np.interp(np.arange(0, t_sec[-1], 1 / fs), t_sec, x_sec)
    b = np.interp(np.arange(0, t_mts[-1], 1 / fs), t_mts, y_mts)
    best = (OFFSET_GUESS_S, 0.0)
    for lag in np.arange(OFFSET_GUESS_S - OFFSET_SEARCH_S, OFFSET_GUESS_S + OFFSET_SEARCH_S + 1e-9, 1 / fs):
        i0 = int(round(lag * fs))
        n = min(len(b), len(a) - i0)
        if i0 < 0 or n < 0.9 * len(b) or a[i0:i0 + n].std() == 0:
            continue
        r = abs(np.corrcoef(a[i0:i0 + n], b[:n])[0, 1])
        if r > best[1]:
            best = (float(lag), float(r))
    return best


def process_trial(num, sec_path, mts_path):
    """Low-passed SEC at `freq` Hz on the MTS clock, and the targets on the same grid."""
    sec, mts = read_numeric_csv(sec_path), read_numeric_csv(mts_path)
    t_raw = sec["time_s"].to_numpy(float)
    t_raw = t_raw - t_raw[0]
    raw_fs = (len(t_raw) - 1) / t_raw[-1]
    t_u = np.arange(0, t_raw[-1], 1 / raw_fs)                         # uneven -> uniform grid
    x = lowpass(np.interp(t_u, t_raw, sec[sec_col].to_numpy(float)), pass_value, raw_fs)
    t_m = mts["RunningTime_s"].to_numpy(float)
    t_m = t_m - t_m[0]
    offset, r = estimate_offset(t_u, x, t_m, mts["Displacement_m"].to_numpy(float))
    t = np.arange(0, min(t_m[-1], t_u[-1] - offset), 1 / freq)          # MTS clock at `freq` Hz
    on_edge = abs(abs(offset - OFFSET_GUESS_S) - OFFSET_SEARCH_S) < 0.11
    ok = bool(r >= GOOD_OFFSET_CORR and not on_edge)
    print(f"  trial_{num:02d}: SEC {raw_fs:.0f} Hz -> low-pass {pass_value:g} Hz -> {freq:g} Hz | "
          f"offset {offset:.2f} s (|corr| {r:.2f})" + ("" if ok else "  <-- alignment UNRELIABLE"))
    return {"num": num, "t": t, "x": np.interp(t + offset, t_u, x), "raw_fs": raw_fs, "offset": offset,
            "offset_corr": r, "ok": ok,
            "targets": {c: np.interp(t, t_m, mts[c].to_numpy(float)) for c in target_cols}}


# === Create Sliding Windows ===
def create_windows(array, window_size, step):
    return sliding_window_view(array, window_size)[::step]


# === Normalize each window to [-1, 1] ===
def min_max_scale(windows):
    lo, hi = windows.min(axis=1, keepdims=True), windows.max(axis=1, keepdims=True)
    return 2 * (windows - lo) / np.where(hi > lo, hi - lo, 1.0) - 1


# === Takens Embedding (same indexing as gtda.time_series.TakensEmbedding) ===
def takens_embedding(windows, delay, dimension, stride):
    n = windows.shape[1] - (dimension - 1) * delay
    idx = np.arange(0, n, stride)
    return np.stack([windows[:, idx + k * delay] for k in range(dimension)], axis=2)


# === Extract H1 Persistence (giotto-tda if installed, else ripser) ===
def extract_h1_persistence(embedded_windows):
    try:
        from gtda.homology import VietorisRipsPersistence
        dg = VietorisRipsPersistence(homology_dimensions=(1,), n_jobs=N_JOBS).fit_transform(embedded_windows)
        return np.max(dg[:, :, 1] - dg[:, :, 0], axis=1)
    except ImportError:
        pass
    try:
        from ripser import ripser
    except ImportError:
        raise ImportError("Install ripser (pip install ripser) or giotto-tda (Python <= 3.11).")
    out = np.zeros(len(embedded_windows))
    for i, pc in enumerate(embedded_windows):
        h1 = ripser(pc, maxdim=1)["dgms"][1]
        out[i] = float(np.max(h1[:, 1] - h1[:, 0])) if len(h1) else 0.0
    return out


# === Main Extraction Function ===
def extract_sec_h1_excel(tr, w, d, output_path):
    start_time = time.time()
    x_windows = create_windows(tr["x"], w, s_W)
    time_windows = create_windows(tr["t"], w, s_W)
    embedded = takens_embedding(min_max_scale(x_windows), delay=d, dimension=2, stride=s)
    max_h1 = extract_h1_persistence(embedded)

    df = pd.DataFrame({"Trial": f"trial_{tr['num']:02d}", "Time": time_windows[:, -1], "Max H1": max_h1})
    for c in target_cols:
        df[c] = create_windows(tr["targets"][c], w, s_W)[:, -1]       # target at the window end
        df[f"{c} window mean"] = create_windows(tr["targets"][c], w, s_W).mean(axis=1)
    df["Capacitance_pF"] = x_windows[:, -1]
    df["Includes_first_cycle"] = time_windows[:, 0] < FIRST_CYCLE_S
    df = df[df["Time"] >= SKIP_START_S].reset_index(drop=True)
    ms = (time.time() - start_time) / max(1, len(x_windows)) * 1000
    print(f"    {len(df)} windows ({embedded.shape[1]} points each), {ms:.1f} ms per window (TDA only)")

    settings = pd.DataFrame([{"trial": tr["num"], "sec_col": sec_col, "raw_fs_Hz": round(tr["raw_fs"], 2),
                              "freq_Hz": freq, "min_f_Hz": min_f, "max_f_Hz": max_f, "pass_value_Hz": pass_value,
                              "window_samples": w, "window_s": w / freq, "tau_samples": d, "tau_s": d / freq,
                              "takens_stride_s": s, "window_step_samples": s_W, "window_step_s": s_W / freq,
                              "offset_s": tr["offset"], "alignment_corr": tr["offset_corr"],
                              "alignment_ok": tr["ok"]}])
    out = os.path.join(output_path, f"H1_Features_trial{tr['num']:02d}_minf{min_f:g}_maxf{max_f:g}.xlsx")
    with pd.ExcelWriter(out, engine="openpyxl") as wr:
        df.to_excel(wr, sheet_name="Features", index=False)
        settings.to_excel(wr, sheet_name="Settings", index=False)
    print(f"    Saved: {out}")
    return df


def spearman(a, b):
    return float(pd.Series(a).corr(pd.Series(b), method="spearman"))


def plot_over_time(results, trials, output_path):
    fig, axes = plt.subplots(len(results), 1, figsize=(15, 2.3 * len(results)), sharex=True, squeeze=False)
    for ax, (df, tr) in zip(axes[:, 0], zip(results, trials)):
        ax.plot(tr["t"], 1000 * tr["targets"]["Displacement_m"], color="#b8b7b2", lw=1, ls="--",
                label="MTS displacement (mm)")
        ax.set_ylabel("mm", color="#8a8984")
        ax2 = ax.twinx()
        ax2.plot(df["Time"], df["Max H1"], color="#eb6834", lw=1.6, label="Max H1")
        ax2.set_ylabel("Max H1", color="#eb6834")
        rho = spearman(df["Max H1"], df["Displacement_m"])
        ax.set_title(f"trial_{tr['num']:02d}: Max H1 vs displacement at the window end, Spearman rho = {rho:.2f}"
                     + ("" if tr["ok"] else "   (alignment unreliable)"), fontsize=9, loc="left")
        ax.spines[["top"]].set_visible(False)
    axes[-1, 0].set_xlabel("Time (s, MTS clock; Max H1 plotted at the end of its window)")
    fig.suptitle(f"SEC Max H1 ({window_s:.1f} s windows, tau {tau_s:.2f} s, low-pass {pass_value:g} Hz)",
                 fontweight="bold")
    fig.tight_layout()
    png = os.path.join(output_path, "SEC_H1_over_time.png")
    fig.savefig(png, dpi=150)
    print(f"Saved: {png}")
    plt.show()
    plt.close(fig)


def plot_by_trial(summary, output_path):
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(summary) + 2), 4))
    x = np.arange(len(summary))
    ax.errorbar(x, summary["Max_H1_mean"], yerr=summary["Max_H1_sd"], fmt="none", ecolor="#3d3d3a", capsize=4)
    ax.scatter(x, summary["Max_H1_mean"], s=70, zorder=3,
               color=np.where(summary["Alignment_OK"], "#eb6834", "#b8b7b2"))
    ax.set_xticks(x, summary["Trial"].str.replace("trial_", "T"))
    ax.set_ylabel("Max H1 (mean +- sd over windows\nafter the first loading cycle)")
    ax.set_title("Max H1 per trial: does the loop shape change between trials?", fontweight="bold")
    ax.grid(axis="y", color="#e4e3df", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    png = os.path.join(output_path, "SEC_H1_by_trial.png")
    fig.savefig(png, dpi=150)
    print(f"Saved: {png}")
    plt.show()
    plt.close(fig)


# === Run ===
w = int(window_size or round((1 / min_f + 2 * 0.25 / max_f) * freq))
d = int(tau or max(1, round((0.25 / max_f) * freq / 2)))
window_s, tau_s = w / freq, d / freq
output_path = to_local_path(OUTPUT_DIR) if OUTPUT_DIR else os.path.join(to_local_path(DATA_DIR), "tda_results", "h1")
os.makedirs(output_path, exist_ok=True)
print(f"Window {w} samples ({window_s:.1f} s), tau {d} samples ({tau_s:.2f} s), "
      f"stride {s}, step {s_W} samples ({s_W / freq:g} s)")

trials, results = [], []
for num, sp, mp in find_pairs():
    tr = process_trial(num, sp, mp)
    if len(tr["x"]) < w:
        print(f"    trial_{num:02d}: record shorter than one window - skipped")
        continue
    trials.append(tr)
    results.append(extract_sec_h1_excel(tr, w, d, output_path))

allf = pd.concat(results, ignore_index=True)
later = [df[~df["Includes_first_cycle"]] if (~df["Includes_first_cycle"]).sum() > 2 else df for df in results]
summary = pd.DataFrame([{"Trial": f"trial_{tr['num']:02d}", "Alignment_OK": tr["ok"], "Windows": len(df),
                         "Max_H1_mean": df["Max H1"].mean(), "Max_H1_sd": df["Max H1"].std(),
                         "Spearman_vs_displacement_end": spearman(df["Max H1"], df["Displacement_m"]),
                         "Spearman_vs_displacement_window_mean": spearman(df["Max H1"], df["Displacement_m window mean"]),
                         "Spearman_vs_force_end": spearman(df["Max H1"], df["Force_N"])}
                        for df, tr in zip(later, trials)])
summary.insert(3, "Note", np.where([len(a) < len(b) for a, b in zip(later, results)],
                                   "windows after the first loading cycle", "all windows"))
out = os.path.join(output_path, f"H1_Features_all_trials_minf{min_f:g}_maxf{max_f:g}.xlsx")
with pd.ExcelWriter(out, engine="openpyxl") as wr:
    allf.to_excel(wr, sheet_name="Features", index=False)
    summary.to_excel(wr, sheet_name="Summary", index=False)
print(f"Saved: {out}")
print("\nPer-trial summary:")
print(summary.round(3).to_string(index=False))
plot_over_time(results, trials, output_path)
plot_by_trial(summary, output_path)
