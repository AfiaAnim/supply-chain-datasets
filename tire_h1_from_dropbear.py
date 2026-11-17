"""
TDA features of the SEC capacitance (signal) vs the MTS displacement / force (target), per trial -
the DROPBEAR H1 extraction fitted to the tire dataset, with all three features studied:
    VR H1      Vietoris-Rips max H1 persistence   (giotto-tda, or ripser)
    Alpha H1   alpha complex max H1 persistence   (gudhi; 2*sqrt(alpha) = same distance scale as VR)
    FastTDA    minor axis 2b of the ellipse fitted to the point cloud (NaN where it is not an ellipse)
Paste into ONE Jupyter cell and run.

    DROPBEAR                         tire dataset
    acc (test file, column 3)    ->  SEC capacitance C1_raw_pF   (trial_NN_1.csv, ~261 Hz, uneven)
    pinloc (column 4 / 17.18)    ->  MTS Displacement_m x 1000 = mm, and Force_N (trial_NN.csv, ~100 Hz)
    one file, one clock          ->  two files: SEC aligned to the MTS by cross-correlation (~9 s)
    5000 Hz                      ->  low-pass at the file's own rate, resampled to `freq` = 20 Hz

Output (in output_path):
    H1_Features_trialNN_win{window}s.xlsx      Time, Displacement_mm, Force_N, VR H1, Alpha H1, FastTDA
    H1_Features_all_trials_win{window}s.xlsx   all trials + Summary (level and correlation with the target)
    TDA_features_vs_target_win{window}s.png    every feature and the target over time (trial x feature)
    TDA_features_by_trial_win{window}s.png     level of every feature per trial

Requirements:
    pip install numpy pandas scipy scikit-learn matplotlib openpyxl ripser gudhi
    (giotto-tda is used instead when it is installed and works; it fails with scikit-learn >= 1.6)
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


def to_local_path(path):
    """C:\\... -> /mnt/c/... when Jupyter runs under Linux / WSL; other paths unchanged."""
    path = str(path).strip().strip('"')
    if os.name != "nt" and len(path) > 2 and path[1] == ":" and path[2] in "\\/":
        return f"/mnt/{path[0].lower()}/{path[3:].replace(chr(92), '/')}"
    return path


# === Find the SEC and MTS files of each trial (by their columns: the file names are not reliable) ===
def find_trials(base_path, trial_ids=None):
    base = os.path.abspath(base_path)
    if not os.path.isdir(base):
        raise FileNotFoundError(f"Folder not found: {base}\nSet base_path to the folder with the csv files "
                                f"(\".\" = the folder of this notebook: {os.path.abspath('.')})")
    skip_dirs = {"lib", "lib64", "bin", "include", "share", "site-packages", "tda_results", ".ipynb_checkpoints"}
    files = [p for p in sorted(glob.glob(os.path.join(base, "**", "*.csv"), recursive=True))
             if not skip_dirs & set(os.path.relpath(p, base).split(os.sep)[:-1])]   # not in a venv / results folder
    files = [p for p in files if re.search(r"trial[_\- ]?\d+", os.path.basename(p), re.I)]
    print(f"Searching {base}: {len(files)} csv files with 'trial_NN' in the name")
    sec, mts = {}, {}
    for p in files:
        name = os.path.relpath(p, base)
        m = re.search(r"trial[_\- ]?(\d+)", os.path.basename(p), re.I)
        with open(p, "r", encoding="utf-8-sig", errors="replace") as f:
            header = f.readline()
        if signal_col in header and "time_s" in header:
            sec.setdefault(int(m.group(1)), p)
        elif "RunningTime_s" in header and target_col in header:
            mts.setdefault(int(m.group(1)), p)
        else:
            print(f"  skipped {name}: columns not recognised (first line: {header.strip()[:80]})")
    found = sorted(set(sec) & set(mts))
    print(f"  SEC files for trials {sorted(sec)}, MTS files for trials {sorted(mts)}")
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


# === Extract H1 Persistence (giotto-tda if it works, else ripser - same Vietoris-Rips H1) ===
H1_BACKEND = None                   # set on the first call: "gtda" or "ripser"


def extract_h1_persistence(embedded_windows):
    global H1_BACKEND
    if H1_BACKEND in (None, "gtda"):
        try:
            from gtda.homology import VietorisRipsPersistence
            tda = VietorisRipsPersistence(homology_dimensions=(1,), n_jobs=1)
            diagrams = tda.fit_transform(embedded_windows)
            H1_BACKEND = "gtda"
            return np.max(diagrams[:, :, 1] - diagrams[:, :, 0], axis=1)
        except Exception as e:      # not installed, or broken by a newer scikit-learn (force_all_finite)
            print(f"  giotto-tda not usable ({type(e).__name__}: {str(e)[:90]}) -> using ripser")
            H1_BACKEND = "ripser"
    try:
        from ripser import ripser
    except ImportError:
        raise ImportError("Install ripser:  %pip install ripser  (then restart the kernel)") from None
    out = np.zeros(len(embedded_windows))
    for i, pc in enumerate(embedded_windows):
        h1 = ripser(pc, maxdim=1)["dgms"][1]
        out[i] = float(np.max(h1[:, 1] - h1[:, 0])) if len(h1) else 0.0
    return out


# === Alpha H1 (gudhi alpha complex; 2*sqrt(alpha) puts it on the same distance scale as VR) ===
def alpha_max_h1(pc):
    import gudhi as gd
    try:
        st = gd.AlphaComplex(points=pc.tolist()).create_simplex_tree()
        st.persistence()
        iv = np.asarray(st.persistence_intervals_in_dimension(1), dtype=float)
    except Exception:
        return np.nan
    if iv.size == 0:
        return 0.0
    iv = iv[np.isfinite(iv).all(axis=1) & (iv[:, 1] > iv[:, 0])]
    if iv.shape[0] == 0:
        return 0.0
    scaled = 2.0 * np.sqrt(np.maximum(iv, 0.0))
    return float(np.max(scaled[:, 1] - scaled[:, 0]))


def extract_alpha_h1(embedded_windows):
    try:
        import gudhi  # noqa: F401
    except ImportError:
        print("  Alpha H1 skipped: install gudhi  (%pip install gudhi, then restart the kernel)")
        return np.full(len(embedded_windows), np.nan)
    return np.array([alpha_max_h1(pc) for pc in embedded_windows])


# === FastTDA (SVD least-squares ellipse fit, A + C = 1): minor axis 2b, NaN if not an ellipse ===
def fasttda_minor_axis(pc):
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
    c, sn = np.cos(th), np.sin(th)
    Ap, Cp = A*c*c + B*c*sn + C*sn*sn, A*sn*sn - B*c*sn + C*c*c
    Dp, Ep = D*c + E*sn, -D*sn + E*c
    if Ap <= 0 or Cp <= 0:
        return np.nan
    Fn = F - Dp**2 / (4 * Ap) - Ep**2 / (4 * Cp)
    return np.nan if Fn >= 0 else 2.0 * min(np.sqrt(-Fn / Ap), np.sqrt(-Fn / Cp))


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

    n_win = len(embedded_windows)
    vr_h1 = extract_h1_persistence(embedded_windows) if COMPUTE_VR else np.full(n_win, np.nan)
    alpha_h1 = extract_alpha_h1(embedded_windows) if COMPUTE_ALPHA else np.full(n_win, np.nan)
    fasttda = (np.array([fasttda_minor_axis(pc) for pc in embedded_windows]) if COMPUTE_FASTTDA
               else np.full(n_win, np.nan))

    df = pd.DataFrame({
        'Trial': f"trial_{trial_id:02d}",
        'Time': time_windows[:, -1],                                # window END: real-time, as in DROPBEAR
        target_name: target_windows[:, -1],
        'Force_N': force_windows[:, -1],
        'Capacitance_pF': cap_windows[:, -1],
        'VR H1': vr_h1,                                             # Vietoris-Rips max H1 persistence
        'Alpha H1': alpha_h1,                                       # alpha complex max H1 persistence
        'FastTDA': fasttda,                                         # ellipse minor axis 2b (NaN: no ellipse)
        'FastTDA_FitOK': np.isfinite(fasttda),
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
max_f = 0.05        #    turnaround traces a real loop -> tau 2.5 s. VR H1 peaks ~1.4 in trials 10-13, ~0.2 in 14-15
# DROPBEAR rule with the loading frequencies (one whole cycle per window, features ~flat within a trial):
# min_f = 0.025     # 40 s loading cycle
# max_f = 0.075     # its 3rd harmonic   -> window 46.6 s, tau 1.65 s (use s = 3)
s = 2               # Takens stride (DROPBEAR: 1): 175 points per cloud, ~8x faster than s = 1 with
                    # the same VR H1 (within ~2%); use 3 for the 46.6 s window
s_W = 10            # window step: 10 samples = 0.5 s (DROPBEAR: 5 = 1 ms)
COMPUTE_VR = True       # Vietoris-Rips H1  (giotto-tda, or ripser)
COMPUTE_ALPHA = True    # Alpha H1          (gudhi)
COMPUTE_FASTTDA = True  # FastTDA 2b        (no extra package)
good_alignment = 0.7   # SEC-MTS |corr| below this: trial flagged (trials 01-03)
first_cycle_s = 40.0   # windows starting before this include the first (conditioning) loading cycle
loading_period_s = 40.0   # one loading cycle (FFT dominant period): per-cycle peaks in the by-trial figure

base_path = "/home/barbara/tire_tda"     # folder with the SEC and MTS csv files (trial_NN*.csv)
output_path = None                       # None = /home/barbara/tire_tda/tda_results/h1_dropbear


# === Run ===
base_path = to_local_path(base_path)
trials = find_trials(base_path, trial_ids)                         # checks the folder before writing anything
output_path = to_local_path(output_path) if output_path else os.path.join(base_path, "tda_results", "h1_dropbear")
os.makedirs(output_path, exist_ok=True)
win_s = round(((1 / min_f) + (0.25 / max_f) * 2) * freq) / freq
tau_s = round((0.25 / max_f) * freq / 2) / freq
print(f"window {win_s:g} s, tau {round((0.25 / max_f) * freq / 2) / freq:g} s, stride {s}, step {s_W / freq:g} s\n")
results = [extract_tire_h1_excel(n, sp, mp, freq, max_f, min_f, s, s_W, pass_value, output_path)
           for n, sp, mp in trials]
if not results:
    raise FileNotFoundError(f"No trial with both an SEC and an MTS csv file in {os.path.abspath(base_path)} "
                            "(see the list above: which files were found and why they were skipped)")

FEATURES = [f for f, on in (("VR H1", COMPUTE_VR), ("Alpha H1", COMPUTE_ALPHA), ("FastTDA", COMPUTE_FASTTDA))
            if on and any(df[f].notna().any() for df in results)]
FEATURE_COLORS = {"VR H1": "#6250d6", "Alpha H1": "#1baf7a", "FastTDA": "#eb6834"}

summary = []
for df in results:
    d = df[df["After_first_cycle"]] if df["After_first_cycle"].sum() > 2 else df
    row = {"Trial": d["Trial"].iloc[0], "Alignment_OK": bool(d["Alignment_OK"].iloc[0])}
    for f in FEATURES:
        v = d[f]
        row[f"{f} mean"], row[f"{f} sd"], row[f"{f} peak"] = v.mean(), v.std(), v.max()
        cyc_peaks = v.groupby((d["Time"] // loading_period_s).astype(int)).max()   # peak in each cycle
        row[f"{f} cycle peak median"] = cyc_peaks.median()
        row[f"{f} cycle peaks"] = cyc_peaks.round(4).tolist()
        row[f"{f} rho_{target_name}"] = spearman(v, d[target_name])
        row[f"{f} rho_Force_N"] = spearman(v, d["Force_N"])
    if "FastTDA" in FEATURES:
        row["FastTDA fit ok %"] = 100 * d["FastTDA_FitOK"].mean()
    summary.append(row)
summary = pd.DataFrame(summary)
with pd.ExcelWriter(os.path.join(output_path, f"H1_Features_all_trials_win{win_s:g}s.xlsx"), engine="openpyxl") as wr:
    pd.concat(results, ignore_index=True).to_excel(wr, sheet_name="Features", index=False)
    summary.to_excel(wr, sheet_name="Summary", index=False)
print("\nSummary (windows after the first loading cycle; rho = Spearman correlation):")
for f in FEATURES:
    cols = ["Trial", f"{f} mean", f"{f} cycle peak median", f"{f} rho_{target_name}", f"{f} rho_Force_N"]
    print(f"\n  {f}")
    print(summary[cols].rename(columns=lambda c: c.replace(f + " ", "")).round(3).to_string(index=False))

# === Figure 1: every feature and the target over time (row = trial, column = feature) ===
n, m = len(results), len(FEATURES)
fig, axes = plt.subplots(n, m, figsize=(5.6 * m, 1.9 * n + 0.8), sharex=True, squeeze=False)
tops = {f: 1.08 * np.nanpercentile(pd.concat([df[f] for df in results]), 99.5) for f in FEATURES}
for i, df in enumerate(results):
    for j, f in enumerate(FEATURES):
        ax = axes[i, j]
        ax.plot(df["Time"], df[target_name], color="#b8b7b2", lw=0.9, ls="--")
        ax.set_yticks([])
        ax2 = ax.twinx()
        ax2.plot(df["Time"], df[f], color=FEATURE_COLORS[f], lw=1.3)
        ax2.set_ylim(0, tops[f])                                    # same scale for every trial
        ax2.tick_params(labelsize=7)
        if i == 0:
            ax.set_title(f"{f}  (dashed: {target_name.replace('_', ' ')})", fontsize=10, fontweight="bold",
                         color=FEATURE_COLORS[f])
        if j == 0:
            ok = bool(df["Alignment_OK"].iloc[0])
            ax.set_ylabel(df["Trial"].iloc[0].replace("trial_", "T") + ("" if ok else "\n(unreliable)"),
                          fontsize=9, rotation=0, ha="right", va="center", color="#3d3d3a" if ok else "#e34948")
        ax.spines[["top"]].set_visible(False)
for ax in axes[-1, :]:
    ax.set_xlabel("Time (s, end of window)")
fig.suptitle(f"TDA features of the SEC signal vs the MTS target - window {win_s:g} s, tau {tau_s:g} s "
             f"(each column: same scale for every trial)", fontweight="bold")
fig.tight_layout()
png = os.path.join(output_path, f"TDA_features_vs_target_win{win_s:g}s.png")
fig.savefig(png, dpi=150)
print(f"Saved: {png}")
plt.show()
plt.close(fig)

# === Figure 2: level of each feature per trial - its peak in every loading cycle (after the first) ===
# (VR / Alpha H1 are ~0 for most of a cycle with one sharp peak, so mean +- sd would mislead)
fig, axes = plt.subplots(1, m, figsize=(4.6 * m, 3.6), squeeze=False)
x = np.arange(len(summary))
for ax, f in zip(axes[0], FEATURES):
    col = np.where(summary["Alignment_OK"], FEATURE_COLORS[f], "#b8b7b2")
    for xi, peaks, c in zip(x, summary[f"{f} cycle peaks"], col):
        ax.scatter(np.full(len(peaks), xi), peaks, s=14, color=c, alpha=0.55, zorder=2)
    ax.scatter(x, summary[f"{f} cycle peak median"], s=90, marker="_", linewidths=2.5, color="#3d3d3a", zorder=3)
    ax.set_xticks(x, summary["Trial"].str.replace("trial_", "T"), fontsize=8)
    ax.set_ylim(bottom=0)
    ax.set_title(f, fontsize=10, fontweight="bold", color=FEATURE_COLORS[f])
    ax.grid(axis="y", color="#e4e3df", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
axes[0, 0].set_ylabel("peak in each loading cycle\n(dots; bar = median)")
fig.suptitle("Feature level per trial: peak in every loading cycle after the first (grey = alignment unreliable)",
             fontweight="bold")
fig.tight_layout()
png = os.path.join(output_path, f"TDA_features_by_trial_win{win_s:g}s.png")
fig.savefig(png, dpi=150)
print(f"Saved: {png}")
plt.show()
plt.close(fig)
