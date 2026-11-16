"""
Features from the SEC signal that behave like the MTS target (displacement / force).
Paste into ONE Jupyter cell and run (or: python sec_target_like_features.py).

Features (each mapped to the target's units with a calibration fitted on the OTHER trials only):
    SEC level           low-passed capacitance (tared on the unloaded start), straight-line calibration
    SEC S-curve         the same level through a cubic calibration (undoes the S-shaped response)
    SEC fundamental     the capacitance band-passed around the loading frequency (~0.025 Hz):
                        a clean sine that rises and falls with the loading
    SEC phase           where in the loading cycle the tire is, read from the SEC: the loading
                        frequency and phase are locked to the whole SEC record (sine + harmonics fit),
                        and the target is rebuilt from that phase as a sum of harmonics. Follows the
                        triangle displacement past the sensor's saturation and reproduces the shape of
                        the force cycle. Assumes a constant loading rate and the same cycle shape every
                        cycle (true for this test) and uses the whole record (offline, not real-time).

Scored from SCORE_FROM_S on (first loading cycle left out): r, R2, RMSE in mm / N per trial.

Output (in DATA_DIR/tda_results/target_features):
    SEC_target_features_all_trials.csv    every sample: time, targets, every feature
    SEC_target_features_scores.xlsx       per trial and pooled scores
    SEC_target_features.png               one trial over time + R2 of every feature

Requirements:
    pip install numpy pandas scipy matplotlib openpyxl
"""
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

try:                                    # make plots appear inside Jupyter
    from IPython import get_ipython
    if get_ipython() is not None:
        get_ipython().run_line_magic("matplotlib", "inline")
except Exception:
    pass

# === Parameters (same names as the DROPBEAR code) ===
DATA_DIR = "."                # folder with the SEC and MTS csv files (or sec\ and mts\ subfolders)
OUTPUT_DIR = None             # None = DATA_DIR/tda_results/target_features
trial_ids = None              # None = every trial found; or e.g. [10, 11, 12]
sec_col = "C1_raw_pF"         # input signal (DROPBEAR: acc)

freq = 20                     # analysis rate (Hz) after low-pass + resampling
min_f = 0.025                 # loading frequency (Hz): the 40 s cycle; the exact value is locked per trial
max_f = 0.075                 # highest significant SEC frequency (3rd harmonic of the loading)
pass_value = 0.25             # low-pass cutoff (Hz): 3.3 x max_f, as in DROPBEAR (100 / 31.1)
s_W = 1                       # output step in samples (1 = every sample at `freq`; 10 = every 0.5 s)

# Derived from the frequencies (like window_size / tau in the DROPBEAR code):
#   lock_harmonics = round(max_f / min_f)            -> 3  harmonics in the lock-in fit of the SEC
#   n_harmonics    = int(pass_value / min_f) // 2 + 1 -> 6  harmonics used to rebuild the target
#   band           = [0.5, 1.5] x min_f              -> 0.0125-0.0375 Hz for the SEC fundamental
lock_harmonics = None         # None = derived; or a number
n_harmonics = None            # None = derived; or a number
lock_search = 0.05            # loading frequency searched within +-5% of the SEC's dominant peak

# Other settings
TARGETS = {"Displacement_m": ("Displacement (mm)", 1000.0), "Force_N": ("Force (N)", 1.0)}
TARE_S = 2.0                  # capacitance baseline = median of the first TARE_S (unloaded)
SCORE_FROM_S = 40.0           # score from here on (first loading cycle = conditioning)
GOOD_OFFSET_CORR = 0.7        # trials with weaker SEC-MTS alignment are not used
EXAMPLE_TRIAL = None          # trial drawn in the figure; None = first one
COLORS = {"SEC level": "#b8b7b2", "SEC S-curve": "#2a78d6", "SEC fundamental": "#1baf7a", "SEC phase": "#eb6834"}


# === Loading and alignment ===
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
        elif "RunningTime_s" in header and all(c in header for c in TARGETS):
            mts.setdefault(int(m.group(1)), p)
    return [(n, sec[n], mts[n]) for n in sorted(set(sec) & set(mts)) if not trial_ids or n in trial_ids]


def lowpass(x, fc, fs, order=4):
    return signal.sosfiltfilt(signal.butter(order, fc, btype="lowpass", fs=fs, output="sos"), x)


def load_trial(num, sec_path, mts_path):
    sec, mts = read_numeric_csv(sec_path), read_numeric_csv(mts_path)
    t_s = sec["time_s"].to_numpy(float)
    t_s = t_s - t_s[0]
    fs = (len(t_s) - 1) / t_s[-1]
    t_u = np.arange(0, t_s[-1], 1 / fs)                                 # uneven -> uniform
    x = lowpass(np.interp(t_u, t_s, sec[sec_col].to_numpy(float)), pass_value, fs)
    t_m = mts["RunningTime_s"].to_numpy(float)
    t_m = t_m - t_m[0]
    d = mts["Displacement_m"].to_numpy(float)
    best = (10.0, 0.0)                                                  # SEC starts ~10 s early
    for lag in np.arange(5.0, 15.0, 0.1):
        tt = np.arange(0, t_m[-1], 0.1)
        if tt[-1] + lag > t_u[-1]:
            continue
        r = abs(np.corrcoef(np.interp(tt + lag, t_u, x), np.interp(tt, t_m, d))[0, 1])
        best = max(best, (float(lag), float(r)), key=lambda b: b[1])
    offset, r = best
    t = np.arange(0, min(t_m[-1], t_u[-1] - offset), 1 / freq)
    print(f"  trial_{num:02d}: offset {offset:.1f} s (|corr| {r:.2f})" + ("" if r >= GOOD_OFFSET_CORR else "  <-- NOT USED"))
    tr = {"name": f"trial_{num:02d}", "t": t, "ok": r >= GOOD_OFFSET_CORR,
          "level": np.interp(t + offset, t_u, x)}
    tr["level"] = tr["level"] - np.median(tr["level"][t < TARE_S])
    for c, (_, scale) in TARGETS.items():
        tr[c] = scale * np.interp(t, t_m, mts[c].to_numpy(float))
    return tr


# === SEC-only features ===
def dominant_freq(trials):
    """Loading frequency from the SEC spectra (median over trials)."""
    out = []
    for tr in trials:
        y = tr["level"] - tr["level"].mean()
        amp, f = np.abs(np.fft.rfft(y)), np.fft.rfftfreq(len(y), 1 / freq)
        ok = (f >= 0.5 * min_f) & (f <= 1.5 * min_f)                    # near the expected loading
        out.append(f[ok][np.argmax(amp[ok])])
    return float(np.median(out))


def add_sec_features(tr, f0):
    band = signal.butter(2, [0.5 * min_f, 1.5 * min_f], btype="bandpass", fs=freq, output="sos")
    tr["fundamental"] = signal.sosfiltfilt(band, tr["level"])
    # Phase locked to the SEC: the MTS loads at a constant rate, so fit the loading frequency and
    # phase to the whole SEC record (a sine + its harmonics, frequency searched near f0) and let the
    # phase advance at exactly that rate. (A band-pass + Hilbert phase drifts: a 160 s record holds
    # only 4 cycles and the filter needs about one period to settle at each end.)
    t, y = tr["t"], tr["level"] - tr["level"].mean()
    best = (-1.0, f0, 0.0)
    for f in f0 * np.linspace(1 - lock_search, 1 + lock_search, 401):
        X = np.column_stack([np.ones_like(t)] + [g(2 * np.pi * k * f * t) for k in range(1, lock_harmonics + 1)
                                                 for g in (np.cos, np.sin)])
        coef, res, *_ = np.linalg.lstsq(X, y, rcond=None)
        r2 = 1 - (res[0] if len(res) else np.sum((y - X @ coef) ** 2)) / np.sum(y ** 2)
        if r2 > best[0]:
            best = (r2, f, np.arctan2(-coef[2], coef[1]))                 # fundamental: cos(2 pi f t + phi)
    tr["lock_r2"], tr["lock_f"], phi = best
    tr["phase"] = 2 * np.pi * tr["lock_f"] * t + phi


def design(tr, feature):
    """Columns the calibration fits (intercept added later)."""
    if feature == "SEC level":
        return tr["level"][:, None]
    if feature == "SEC S-curve":
        return np.column_stack([tr["level"], tr["level"] ** 2, tr["level"] ** 3])
    if feature == "SEC fundamental":
        return tr["fundamental"][:, None]
    k = np.arange(1, n_harmonics + 1)
    ph = tr["phase"][:, None] * k
    return np.hstack([np.cos(ph), np.sin(ph)])                          # SEC phase: harmonic series


FEATURES = ["SEC level", "SEC S-curve", "SEC fundamental", "SEC phase"]


def loto(trials, feature, target):
    """Calibrate on the other trials, apply to each left-out trial: list of predictions."""
    out = []
    for i, tr in enumerate(trials):
        Xs, ys = [], []
        for j, o in enumerate(trials):
            if j != i:
                m = o["t"] >= SCORE_FROM_S
                Xs.append(design(o, feature)[m])
                ys.append(o[target][m])
        X = np.column_stack([np.ones(sum(len(y) for y in ys)), np.vstack(Xs)])
        beta = np.linalg.lstsq(X, np.concatenate(ys), rcond=None)[0]
        out.append(np.column_stack([np.ones(len(tr["t"])), design(tr, feature)]) @ beta)
    return out


# === Run ===
lock_harmonics = lock_harmonics or max(1, round(max_f / min_f))
n_harmonics = n_harmonics or int(pass_value / min_f) // 2 + 1
print(f"freq {freq:g} Hz, min_f {min_f:g} Hz, max_f {max_f:g} Hz, pass_value {pass_value:g} Hz -> "
      f"lock_harmonics {lock_harmonics}, n_harmonics {n_harmonics}, output every {s_W} sample(s)")
base = DATA_DIR
out_dir = OUTPUT_DIR or os.path.join(base, "tda_results", "target_features")
os.makedirs(out_dir, exist_ok=True)
print("Loading and aligning:")
trials = [tr for tr in (load_trial(*p) for p in find_pairs(base)) if tr["ok"]]
if len(trials) < 2:
    raise RuntimeError("Need at least 2 usable trials.")
f0 = dominant_freq(trials)
print(f"Loading frequency from the SEC: {f0:.4f} Hz (period {1 / f0:.1f} s; min_f = {min_f:g} Hz)")
for tr in trials:
    add_sec_features(tr, f0)
    print(f"  {tr['name']}: SEC locked at {tr['lock_f']:.5f} Hz (fit R2 {tr['lock_r2']:.3f})")

rows, samples = [], []
for target, (label, _) in TARGETS.items():
    for feature in FEATURES:
        for tr, pred in zip(trials, loto(trials, feature, target)):
            tr[f"{feature} -> {target}"] = pred
            m = tr["t"] >= SCORE_FROM_S
            y, p = tr[target][m], pred[m]
            rows.append({"Target": label, "Feature": feature, "Trial": tr["name"],
                         "r": np.corrcoef(p, y)[0, 1], "R2": 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2),
                         "RMSE": np.sqrt(np.mean((y - p) ** 2)), "RMSE_pct_range": 100 * np.sqrt(np.mean((y - p) ** 2)) / np.ptp(y)})
per_trial = pd.DataFrame(rows)
pooled = per_trial.groupby(["Target", "Feature"], sort=False).agg(
    r=("r", "mean"), R2=("R2", "mean"), R2_min=("R2", "min"), RMSE=("RMSE", "mean"),
    RMSE_pct_range=("RMSE_pct_range", "mean")).reset_index()
print("\nHow well each SEC feature follows the target (calibrated on the other trials; mean over trials):")
print(pooled.round(3).to_string(index=False))

for tr in trials:
    k = slice(None, None, max(1, int(s_W)))
    samples.append(pd.DataFrame({"Trial": tr["name"], "Time_s": tr["t"][k], "SEC_level_pF": tr["level"][k],
                                 "SEC_fundamental_pF": tr["fundamental"][k], "SEC_phase_rad": tr["phase"][k],
                                 **{label: tr[c][k] for c, (label, _) in TARGETS.items()},
                                 **{f"{f} -> {label}": tr[f"{f} -> {c}"][k] for c, (label, _) in TARGETS.items()
                                    for f in FEATURES}}))
pd.concat(samples, ignore_index=True).to_csv(os.path.join(out_dir, "SEC_target_features_all_trials.csv"), index=False)
with pd.ExcelWriter(os.path.join(out_dir, "SEC_target_features_scores.xlsx"), engine="openpyxl") as wr:
    pooled.to_excel(wr, sheet_name="Mean_over_trials", index=False)
    per_trial.to_excel(wr, sheet_name="Per_trial", index=False)
print(f"Saved: {out_dir}")

# === Figure: one trial over time, and R2 of every feature ===
ex = next((tr for tr in trials if tr["name"] == EXAMPLE_TRIAL), trials[0])
fig, axes = plt.subplots(2, 2, figsize=(17, 7.5), gridspec_kw={"width_ratios": [3.2, 1]})
for row, (target, (label, _)) in enumerate(TARGETS.items()):
    ax = axes[row, 0]
    m = ex["t"] >= SCORE_FROM_S
    ax.plot(ex["t"][m], ex[target][m], color="#3d3d3a", lw=2.6, label="MTS (target)")
    for f in FEATURES:
        ax.plot(ex["t"][m], ex[f"{f} -> {target}"][m], color=COLORS[f], lw=1.3, label=f)
    ax.set_ylabel(label)
    ax.grid(color="#e4e3df", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=5, loc="lower left")
    ax = axes[row, 1]
    p = pooled[pooled["Target"] == label]
    y = np.arange(len(p))[::-1]
    ax.barh(y, p["R2"], color=[COLORS[f] for f in p["Feature"]])
    for yi, r2, mn in zip(y, p["R2"], p["R2_min"]):
        ax.text(max(r2, 0) + 0.02, yi, f"{r2:.2f} (worst trial {mn:.2f})", va="center", fontsize=8)
    ax.set_yticks(y, p["Feature"], fontsize=8)
    ax.set_xlim(0, 1.45)
    ax.set_xlabel("R2 on left-out trials (mean)")
    ax.spines[["top", "right"]].set_visible(False)
axes[1, 0].set_xlabel("Time (s)")
fig.suptitle(f"SEC features that follow the target - {ex['name']} (every feature calibrated on the other trials)",
             fontweight="bold")
fig.tight_layout()
png = os.path.join(out_dir, "SEC_target_features.png")
fig.savefig(png, dpi=150)
print(f"Saved: {png}")
plt.show()
plt.close(fig)
