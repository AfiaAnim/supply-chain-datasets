"""
DROPBEAR Dataset 8: fine-tune window size and Takens time delay for max H1 persistence.

Same processing as the original extraction code (6th-order SOS low-pass at 5000 Hz,
per-window min-max scaling to [-1, 1], 2-D Takens embedding, Vietoris-Rips H1),
but runs every combination of WINDOW_SIZES x TAUS and saves them side by side.

Output workbook:
    one sheet per combination  (e.g. "W362_T20": Time, pinloc, Max H1)
    "Summary" sheet            (window/tau in samples and ms, H1 statistics, runtime)

Requirements:
    pip install numpy pandas scipy scikit-learn giotto-tda openpyxl
"""
import os
import time

import numpy as np
import pandas as pd
from gtda.homology import VietorisRipsPersistence
from gtda.time_series import TakensEmbedding
from scipy import signal
from sklearn.preprocessing import MinMaxScaler


# ============================================================
# USER CONFIGURATION
# ============================================================

DATA_PATH = "/mnt/c/Users/aphya/Downloads/test1.txt"    # DROPBEAR Dataset 8 trial file
OUTPUT_DIR = "/mnt/c/Users/aphya/Downloads"

freq = 5000          # sampling rate (Hz)
max_f = 31.1         # highest frequency of interest (Hz)
min_f = 17.7         # lowest frequency of interest (Hz)
s = 1                # Takens embedding stride
s_W = 5              # window step (samples)
pass_value = 100     # low-pass cutoff (Hz)
dimension = 2        # Takens embedding dimension

# Original formulas, kept as the baseline:
#   window = int(((1 / min_f) + (0.25 / max_f) * 2) * freq)  -> 362 samples (72.4 ms)
#   tau    = int((0.25 / max_f) * freq / 2)                  -> 20 samples (4.0 ms)
BASE_WINDOW = int(((1 / min_f) + (0.25 / max_f) * 2) * freq)
BASE_TAU = int((0.25 / max_f) * freq / 2)

# Values to try (in samples; at 5000 Hz, 5 samples = 1 ms). Every combination is run.
WINDOW_SIZES = [282, 322, BASE_WINDOW, 402, 442]
TAUS = [10, 15, BASE_TAU, 25, 30]


# ============================================================
# PROCESSING (same as the original code)
# ============================================================

def process_signal(data, pass_value=100, fs=5000):
    N = 6
    low_g_acc = data[:, 0] * 1000 / 102
    acc = data[:, 3]
    pinloc = data[:, 4] / 17.18
    t = data[:, 5].reshape(-1, 1)

    sos = signal.butter(N=N, Wn=pass_value, btype='lowpass', fs=fs, output='sos')
    low_g_acc_filt = signal.sosfiltfilt(sos, low_g_acc).reshape(-1, 1)
    acc_filt = signal.sosfiltfilt(sos, acc).reshape(-1, 1)
    pinloc_filt = signal.sosfiltfilt(sos, pinloc).reshape(-1, 1)

    return np.hstack((low_g_acc_filt, acc_filt, pinloc_filt, t))


def create_windows(array, window_size, step):
    return np.array([array[i:i + window_size] for i in range(0, len(array) - window_size + 1, step)])


def min_max_scale(array):
    scaler = MinMaxScaler((-1, 1))
    return scaler.fit_transform(array)


def takens_embedding(array, delay, dimension, stride):
    embedder = TakensEmbedding(time_delay=delay, dimension=dimension, stride=stride)
    return embedder.fit_transform(array)


def extract_h1_persistence(embedded_windows):
    tda = VietorisRipsPersistence(homology_dimensions=(1,), n_jobs=-1)
    diagrams = tda.fit_transform(embedded_windows)
    h1_persistence = np.max(diagrams[:, :, 1] - diagrams[:, :, 0], axis=1)
    return h1_persistence


def extract_h1(acc, pinloc, time_array, window_size, tau):
    """Max H1 per window for one (window_size, tau) combination."""
    acc_windows = create_windows(acc, window_size, s_W)
    pinloc_windows = create_windows(pinloc, window_size, s_W)
    time_windows = create_windows(time_array, window_size, s_W)

    scaled_windows = np.array([min_max_scale(w.reshape(-1, 1)).flatten() for w in acc_windows])
    embedded_windows = takens_embedding(scaled_windows, delay=tau, dimension=dimension, stride=s)

    max_h1 = extract_h1_persistence(embedded_windows)

    return pd.DataFrame({
        'Time': time_windows[:, -1],
        'pinloc': pinloc_windows[:, -1],
        'Max H1': max_h1
    })


# ============================================================
# SWEEP
# ============================================================

def run_sweep():
    if not os.path.isfile(DATA_PATH):
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

    print(f"Loading: {DATA_PATH}")
    data = np.loadtxt(DATA_PATH, skiprows=9)
    processed = process_signal(data, pass_value=pass_value, fs=freq)
    acc = processed[:, 1]
    pinloc = processed[:, 2]
    time_array = processed[:, 3]
    print(f"Signal: {len(acc)} samples ({len(acc) / freq:.1f} s)")

    combos = [(w, t) for w in WINDOW_SIZES for t in TAUS]
    print(f"Running {len(combos)} combinations "
          f"(windows {WINDOW_SIZES}, taus {TAUS}); baseline W{BASE_WINDOW}_T{BASE_TAU}\n")

    sheets = {}
    summary = []
    for i, (window_size, tau) in enumerate(combos, 1):
        name = f"W{window_size}_T{tau}"
        if tau * (dimension - 1) >= window_size:
            print(f"[{i}/{len(combos)}] {name}: skipped (tau too large for window)")
            continue
        if len(acc) < window_size:
            print(f"[{i}/{len(combos)}] {name}: skipped (signal shorter than window)")
            continue

        t0 = time.time()
        df = extract_h1(acc, pinloc, time_array, window_size, tau)
        elapsed = time.time() - t0
        sheets[name] = df

        h1 = df['Max H1']
        summary.append({
            'Profile': name,
            'Baseline': window_size == BASE_WINDOW and tau == BASE_TAU,
            'Window_samples': window_size,
            'Window_ms': window_size / freq * 1000,
            'Tau_samples': tau,
            'Tau_ms': tau / freq * 1000,
            'Points_per_window': (window_size - tau * (dimension - 1) - 1) // s + 1,
            'Windows': len(df),
            'H1_mean': h1.mean(),
            'H1_std': h1.std(),
            'H1_min': h1.min(),
            'H1_max': h1.max(),
            'Corr_H1_pinloc': np.corrcoef(h1, df['pinloc'])[0, 1],
            'Runtime_s': elapsed,
            'ms_per_window': elapsed / len(df) * 1000,
        })
        print(f"[{i}/{len(combos)}] {name}: {len(df)} windows, "
              f"H1 mean {h1.mean():.4f}, {elapsed:.1f} s")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stem = os.path.splitext(os.path.basename(DATA_PATH))[0]
    output_file = os.path.join(OUTPUT_DIR, f"H1_WindowTauSweep_{stem}.xlsx")
    summary_df = pd.DataFrame(summary)
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)

    print(f"\nResults saved to: {output_file}")
    return summary_df, sheets


# The __main__ guard is required on Windows when n_jobs != 1 (joblib spawns worker processes)
if __name__ == "__main__":
    summary_df, sheets = run_sweep()
    print(summary_df[['Profile', 'Window_ms', 'Tau_ms', 'H1_mean', 'H1_std', 'Corr_H1_pinloc']]
          .to_string(index=False))
