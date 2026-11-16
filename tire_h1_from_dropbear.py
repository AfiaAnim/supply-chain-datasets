"""
Max H1 persistence of the SEC capacitance (signal) vs the MTS displacement / force (target), per
trial - the DROPBEAR H1 extraction fitted to the tire dataset. Paste into ONE Jupyter cell and run.

    DROPBEAR                         tire dataset
    acc (test file, column 3)    ->  SEC capacitance C1_raw_pF   (trial_NN_1.csv, ~261 Hz, uneven)
    pinloc (column 4 / 17.18)    ->  MTS Displacement_m x 1000 = mm, and Force_N (trial_NN.csv, ~100 Hz)
    one file, one clock          ->  two files: SEC aligned to the MTS by cross-correlation (~9 s)
    5000 Hz                      ->  low-pass at the file's own rate, resampled to `freq` = 20 Hz

Output (in output_path):
    H1_Features_trialNN_win{window}s.xlsx      Time, Displacement_mm, Force_N, Max H1 (one per trial)
    H1_Features_all_trials_win{window}s.xlsx   all trials + Summary (Max H1 level, correlation with target)
    H1_vs_target_win{window}s.png              Max H1 and the target over time, every trial

Requirements:
    pip install numpy pandas scipy scikit-learn matplotlib openpyxl ripser
    (giotto-tda is used instead when it is installed - it needs Python <= 3.11)
"""
import glob
import os
import re
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.preprocessing import MinMaxScaler

try:                                    # make plots appear inside Jupyter
    from IPython import get_ipython
    if get_ipython() is not None:
        get_ipython().run_line_magic("matplotlib", "inline")
except Exception:
    pass


# === Find the SEC and MTS files of each trial (by their columns: the file names are not reliable) ===
def find_trials(base_path, trial_ids=None):
    sec, mts = {}, {}
    for p in sorted(glob.glob(os.path.join(base_path, "**", "*.csv"), recursive=True)):
        m = re.search(r"trial[_\- ]?(\d+)", os.path.basename(p), re.I)
        if not m:
            continue
        with open(p, "r", encoding="utf-8-sig", errors="replace") as f:
            header = f.readline()
        if signal_col in header and "time_s" in header:
            sec.setdefault(int(m.group(1)), p)
        elif "RunningTime_s" in header and target_col in header:
            mts.setdefault(int(m.group(1)), p)
    found = sorted(set(sec) & set(mts))
    for n in sorted(set(sec) ^ set(mts)):
        print(f"trial_{n:02d}: {'SEC' if n in sec else 'MTS'} file only - skipped")
    return [(n, sec[n], mts[n]) for n in found if not trial_ids or n in trial_ids]


# === Signal Processing (signal = SEC capacitance, target = MTS displacement / force) ===
def process_signal(sec, mts, freq, pass_value):
    N = 6
    t_sec = sec["time_s"].to_numpy(float)
    t_sec = t_sec - t_sec[0]
    raw_fs = (len(t_sec) - 1) / t_sec[-1]                         # ~261 Hz, slightly uneven
    t_u = np.arange(0, t_sec[-1], 1 / raw_fs)                     # uniform grid before filtering
    cap = np.interp(t_u, t_sec, sec[signal_col].to_numpy(float))
    sos = signal.butter(N=N, Wn=pass_value, btype='lowpass', fs=raw_fs, output='sos')
    cap_filt = signal.sosfiltfilt(sos, cap)

    t_mts = mts["RunningTime_s"].to_numpy(float)
    t_mts = t_mts - t_mts[0]
    target = mts[target_col].to_numpy(float) * target_scale       # mm (DROPBEAR: pinloc / 17.18)
    force = mts["Force_N"].to_numpy(float)

    # SEC starts ~9 s before the MTS: offset = lag of the best cross-correlation (search 5-15 s)
    tt = np.arange(0, t_mts[-1], 0.1)
    lags = [lag for lag in np.arange(5.0, 15.0, 0.1) if tt[-1] + lag <= t_u[-1]]
    corr = [abs(np.corrcoef(np.interp(tt + lag, t_u, cap_filt), np.interp(tt, t_mts, target))[0, 1]) for lag in lags]
    offset, r = lags[int(np.argmax(corr))], max(corr)
    ok = r >= good_alignment
    print(f"  SEC {raw_fs:.0f} Hz -> {freq} Hz, offset {offset:.1f} s (|corr| {r:.2f})"
          + ("" if ok else "  <-- alignment UNRELIABLE"))

    t = np.arange(0, min(t_mts[-1], t_u[-1] - offset), 1 / freq)  # common grid on the MTS clock
    out = np.column_stack((np.interp(t + offset, t_u, cap_filt), np.interp(t, t_mts, target),
                           np.interp(t, t_mts, force), t))
    return out, ok


# === Create Sliding Windows ===
def create_windows(array, window_size, step):
    return np.array([array[i:i + window_size] for i in range(0, len(array) - window_size + 1, step)])


# === Normalize ===
def min_max_scale(array):
    scaler = MinMaxScaler((-1, 1))
    return scaler.fit_transform(array)


# === Takens Embedding (same indexing as gtda.time_series.TakensEmbedding) ===
def takens_embedding(array, delay, dimension, stride):
    n = array.shape[1] - (dimension - 1) * delay
    idx = np.arange(0, n, stride)
    return np.stack([array[:, idx + k * delay] for k in range(dimension)], axis=2)


# === Extract H1 Persistence (giotto-tda if installed, else ripser - same Vietoris-Rips H1) ===
def extract_h1_persistence(embedded_windows):
    try:
        from gtda.homology import VietorisRipsPersistence
        tda = VietorisRipsPersistence(homology_dimensions=(1,), n_jobs=1)
        diagrams = tda.fit_transform(embedded_windows)
        return np.max(diagrams[:, :, 1] - diagrams[:, :, 0], axis=1)
    except ImportError:
        from ripser import ripser                                   # pip install ripser
        out = np.zeros(len(embedded_windows))
        for i, pc in enumerate(embedded_windows):
            h1 = ripser(pc, maxdim=1)["dgms"][1]
            out[i] = float(np.max(h1[:, 1] - h1[:, 0])) if len(h1) else 0.0
        return out


# === Main Extraction Function ===
def extract_tire_h1_excel(trial_id, sec_file, mts_file, freq, max_f, min_f, s, s_W, pass_value, output_path):
    start_time = time.time()
    print(f"trial_{trial_id:02d}:")
    processed, ok = process_signal(pd.read_csv(sec_file), pd.read_csv(mts_file), freq, pass_value)

    cap = processed[:, 0]
    target = processed[:, 1]
    force = processed[:, 2]
    time_array = processed[:, 3]

    h1_window_size = int(round(((1 / min_f) + (0.25 / max_f) * 2) * freq))
    tau = int(round((0.25 / max_f) * freq / 2))

    cap_windows = create_windows(cap, h1_window_size, s_W)
    target_windows = create_windows(target, h1_window_size, s_W)
    force_windows = create_windows(force, h1_window_size, s_W)
    time_windows = create_windows(time_array, h1_window_size, s_W)

    scaled_windows = np.array([min_max_scale(w.reshape(-1, 1)).flatten() for w in cap_windows])
    embedded_windows = takens_embedding(scaled_windows, delay=tau, dimension=2, stride=s)

    max_h1 = extract_h1_persistence(embedded_windows)

    df = pd.DataFrame({
        'Trial': f"trial_{trial_id:02d}",
        'Time': time_windows[:, -1],                                # window END: real-time, as in DROPBEAR
        target_name: target_windows[:, -1],
        'Force_N': force_windows[:, -1],
        'Capacitance_pF': cap_windows[:, -1],
        'Max H1': max_h1,
        'After_first_cycle': time_windows[:, 0] >= first_cycle_s,   # window starts after the 1st cycle
        'Alignment_OK': ok,
    })

    duration = (time.time() - start_time) / len(df) * 1000
    print(f"  {len(df)} windows ({embedded_windows.shape[1]} points each), {duration:.2f} ms per window")
    output_file = os.path.join(output_path, f"H1_Features_trial{trial_id:02d}_win{h1_window_size / freq:g}s.xlsx")
    df.to_excel(output_file, index=False)
    print(f"  Results saved to: {output_file}")
    return df


def spearman(a, b):
    return float(pd.Series(a).corr(pd.Series(b), method="spearman"))


# === Parameters (fitted to the tire dataset) ===
signal_col = "C1_raw_pF"            # signal (DROPBEAR: acc)
target_col = "Displacement_m"       # target (DROPBEAR: pinloc)
target_scale = 1000                 # m -> mm (DROPBEAR: 1 / 17.18)
target_name = "Displacement_mm"
trial_ids = None                    # None = every trial with an SEC and an MTS file; or e.g. [10, 11, 12]

freq = 20           # Hz after resampling (DROPBEAR: 5000)
pass_value = 0.25   # low-pass cutoff in Hz (DROPBEAR: 100): above the loading harmonics, below the noise
# Window and delay come from min_f / max_f with the DROPBEAR formulas:
#   window = (1 / min_f + 2 * 0.25 / max_f) * freq,   tau = (0.25 / max_f) * freq / 2
min_f = 0.1         # -> window (10 + 10) s = 20 s = half a 40 s loading cycle: a window holding a
max_f = 0.05        #    turnaround traces a real loop -> tau 2.5 s. Max H1 ~1.4 in trials 10-13, ~0.2 in 14-15
# DROPBEAR rule with the loading frequencies (one whole cycle per window, Max H1 ~flat within a trial):
# min_f = 0.025     # 40 s loading cycle
# max_f = 0.075     # its 3rd harmonic   -> window 46.6 s, tau 1.65 s (use s = 3)
s = 2               # Takens stride (DROPBEAR: 1): 175 points per cloud, ~8x faster than s = 1 with
                    # the same Max H1 (within ~2%); use 3 for the 46.6 s window
s_W = 10            # window step: 10 samples = 0.5 s (DROPBEAR: 5 = 1 ms)
good_alignment = 0.7   # SEC-MTS |corr| below this: trial flagged (trials 01-03)
first_cycle_s = 40.0   # windows starting before this include the first (conditioning) loading cycle

base_path = r"C:/Users/daniel94/Code/tire_tda/data"          # folder with the SEC and MTS csv files
output_path = r"C:/Users/daniel94/Code/tire_tda/h1_results"


# === Run ===
os.makedirs(output_path, exist_ok=True)
win_s = round(((1 / min_f) + (0.25 / max_f) * 2) * freq) / freq
print(f"window {win_s:g} s, tau {round((0.25 / max_f) * freq / 2) / freq:g} s, stride {s}, step {s_W / freq:g} s\n")
results = [extract_tire_h1_excel(n, sp, mp, freq, max_f, min_f, s, s_W, pass_value, output_path)
           for n, sp, mp in find_trials(base_path, trial_ids)]
if not results:
    raise FileNotFoundError(f"No trial with both an SEC and an MTS csv file in {base_path}")

summary = []
for df in results:
    d = df[df["After_first_cycle"]] if df["After_first_cycle"].sum() > 2 else df
    summary.append({"Trial": d["Trial"].iloc[0], "Alignment_OK": bool(d["Alignment_OK"].iloc[0]),
                    "Max_H1_mean": d["Max H1"].mean(), "Max_H1_peak": d["Max H1"].max(),
                    f"Spearman_vs_{target_name}": spearman(d["Max H1"], d[target_name]),
                    "Spearman_vs_Force_N": spearman(d["Max H1"], d["Force_N"])})
summary = pd.DataFrame(summary)
with pd.ExcelWriter(os.path.join(output_path, f"H1_Features_all_trials_win{win_s:g}s.xlsx"), engine="openpyxl") as wr:
    pd.concat(results, ignore_index=True).to_excel(wr, sheet_name="Features", index=False)
    summary.to_excel(wr, sheet_name="Summary", index=False)
print("\nSummary (windows after the first loading cycle):")
print(summary.round(3).to_string(index=False))

# === Figure: Max H1 and the target over time, every trial ===
fig, axes = plt.subplots(len(results), 1, figsize=(14, 2.2 * len(results)), sharex=True, squeeze=False)
h1_top = 1.08 * max(df["Max H1"].max() for df in results)          # same Max H1 scale for every trial
for ax, df in zip(axes[:, 0], results):
    ax.plot(df["Time"], df[target_name], color="#8a8984", lw=1, ls="--")
    ax.set_ylabel(target_name.replace("_", " "), color="#8a8984", fontsize=8)
    ax2 = ax.twinx()
    ax2.plot(df["Time"], df["Max H1"], color="#6250d6", lw=1.5)
    ax2.set_ylabel("Max H1", color="#6250d6", fontsize=8)
    ax2.set_ylim(0, h1_top)
    row = summary[summary["Trial"] == df["Trial"].iloc[0]].iloc[0]
    ax.set_title(f"{row['Trial']}: Max H1 peak {row['Max_H1_peak']:.2f}, mean {row['Max_H1_mean']:.2f}"
                 + ("" if row["Alignment_OK"] else "  (alignment unreliable)"), fontsize=9, loc="left")
    ax.spines[["top"]].set_visible(False)
axes[-1, 0].set_xlabel("Time (s, end of window)")
fig.suptitle(f"Max H1 of the SEC signal (purple, same scale in every trial) and the MTS target (dashed)"
             f" - window {win_s:g} s",
             fontweight="bold")
fig.tight_layout()
png = os.path.join(output_path, f"H1_vs_target_win{win_s:g}s.png")
fig.savefig(png, dpi=150)
print(f"Saved: {png}")
plt.show()
