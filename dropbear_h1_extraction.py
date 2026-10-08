"""
Extract max H1 persistence features from DROPBEAR trial data and save them to Excel.

Requirements (Python 3.8 - 3.11 recommended; giotto-tda has no wheels for newer versions):
    pip install numpy pandas scipy scikit-learn giotto-tda openpyxl
"""
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from gtda.homology import VietorisRipsPersistence
from gtda.time_series import TakensEmbedding
from scipy import signal
from sklearn.preprocessing import MinMaxScaler


# === Load Data ===
def load_numeric_data(path, min_cols=2):
    """Read only the numeric rows of a text file, skipping any header/description text.

    Works with space, tab, comma or semicolon separated values.
    """
    rows = []
    preview = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if len(preview) < 15:
                preview.append(line.rstrip())
            parts = line.replace(",", " ").replace(";", " ").split()
            if len(parts) < min_cols:
                continue
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                continue  # text line (header, notes, column names) -> skip

    if not rows:
        raise ValueError(
            f"No rows with at least {min_cols} numeric columns found in {path}.\n"
            "First lines of the file:\n" + "\n".join(preview)
        )

    # Keep rows with the most common column count (drops stray partial lines)
    n_cols = max(set(len(r) for r in rows), key=[len(r) for r in rows].count)
    data = np.array([r for r in rows if len(r) == n_cols])
    print(f"Loaded {data.shape[0]} rows x {data.shape[1]} columns")
    print("Column summary (check the column numbers you chose below):")
    for i in range(data.shape[1]):
        c = data[:, i]
        print(f"  col {i}: min={c.min():.5g}  max={c.max():.5g}  mean={c.mean():.5g}  std={c.std():.5g}")
    return data


# === Signal Processing ===
def lowpass(x, pass_value, fs, N=6):
    sos = signal.butter(N=N, Wn=pass_value, btype='lowpass', fs=fs, output='sos')
    return signal.sosfiltfilt(sos, x)


def process_signal(data, time_col, acc_col, pinloc_col, pinloc_scale, pass_value, freq):
    """Low-pass filter the signals and resample them to `freq` Hz.

    Returns (acc, pinloc, time) at `freq` Hz. pinloc is None if pinloc_col is None.
    """
    t = data[:, time_col]
    raw_fs = 1.0 / np.median(np.diff(t))
    print(f"Detected sampling rate from time column: {raw_fs:.0f} Hz")

    acc = lowpass(data[:, acc_col], pass_value, raw_fs)
    pinloc = None
    if pinloc_col is not None:
        pinloc = lowpass(data[:, pinloc_col] * pinloc_scale, pass_value, raw_fs)

    # Downsample to `freq` (safe after the low-pass filter) so window size / tau stay in `freq` samples
    factor = int(round(raw_fs / freq))
    if factor < 1 or abs(raw_fs / factor - freq) / freq > 0.05:
        raise ValueError(f"Data is sampled at {raw_fs:.0f} Hz, which can't be evenly downsampled to "
                         f"freq={freq} Hz. Set freq to {raw_fs:.0f} or to {raw_fs:.0f} divided by a whole number.")
    if factor > 1:
        print(f"Downsampling {raw_fs:.0f} Hz -> {raw_fs / factor:.0f} Hz (every {factor}th sample)")
        acc, t = acc[::factor], t[::factor]
        if pinloc is not None:
            pinloc = pinloc[::factor]

    return acc, pinloc, t


# === Create Sliding Windows ===
def create_windows(array, window_size, step):
    # Strided view instead of a Python list -> much faster and lighter on memory
    windows = np.lib.stride_tricks.sliding_window_view(array, window_size)
    return windows[::step]


# === Normalize each window to [-1, 1] ===
def min_max_scale(array):
    scaler = MinMaxScaler((-1, 1))
    return scaler.fit_transform(array)


# === Takens Embedding ===
def takens_embedding(array, delay, dimension, stride):
    embedder = TakensEmbedding(time_delay=delay, dimension=dimension, stride=stride)
    return embedder.fit_transform(array)


# === Extract H1 Persistence ===
def extract_h1_persistence(embedded_windows, n_jobs=-1):
    tda = VietorisRipsPersistence(homology_dimensions=(1,), n_jobs=n_jobs)
    diagrams = tda.fit_transform(embedded_windows)
    # diagrams: (n_windows, n_points, 3) -> [birth, death, homology_dim]
    h1_persistence = np.max(diagrams[:, :, 1] - diagrams[:, :, 0], axis=1)
    return h1_persistence


# === Main Extraction Function ===
def extract_dropbear_h1_excel(data_path, freq, max_f, min_f, s, s_W,
                              pass_value, output_path, window_size=None, tau=None,
                              time_col=0, acc_col=1, pinloc_col=None, pinloc_scale=1.0, n_jobs=-1):
    start_time = time.time()

    path = Path(data_path)
    if not path.is_file():
        raise FileNotFoundError(f"Data file not found: {path}")

    print(f"Loading: {path}")
    used_cols = [c for c in (time_col, acc_col, pinloc_col) if c is not None]
    data = load_numeric_data(path, min_cols=max(used_cols) + 1)

    acc, pinloc, time_array = process_signal(data, time_col, acc_col, pinloc_col,
                                             pinloc_scale, pass_value, freq)

    # Window size and time delay: use the values given, otherwise compute from the frequencies
    if window_size is None:
        h1_window_size = int(((1 / min_f) + (0.25 / max_f) * 2) * freq)
    else:
        h1_window_size = int(window_size)
    if tau is None:
        tau = max(1, int((0.25 / max_f) * freq / 2))
    else:
        tau = int(tau)
    if tau < 1:
        raise ValueError(f"tau must be at least 1, got {tau}")
    if tau >= h1_window_size:
        raise ValueError(f"tau ({tau}) must be smaller than the window size ({h1_window_size})")
    print(f"Window size: {h1_window_size} samples ({h1_window_size / freq * 1000:.1f} ms), "
          f"tau: {tau} samples ({tau / freq * 1000:.1f} ms), window step: {s_W}")

    if len(acc) < h1_window_size:
        raise ValueError(f"Signal length ({len(acc)}) is shorter than window size ({h1_window_size})")

    acc_windows = create_windows(acc, h1_window_size, s_W)
    time_windows = create_windows(time_array, h1_window_size, s_W)
    print(f"Number of windows: {len(acc_windows)}")

    scaled_windows = np.array([min_max_scale(w.reshape(-1, 1)).flatten() for w in acc_windows])
    embedded_windows = takens_embedding(scaled_windows, delay=tau, dimension=2, stride=s)

    print("Computing Vietoris-Rips persistence (this can take a while)...")
    max_h1 = extract_h1_persistence(embedded_windows, n_jobs=n_jobs)

    df = pd.DataFrame({'Time': time_windows[:, -1]})
    if pinloc is not None:
        df['pinloc'] = create_windows(pinloc, h1_window_size, s_W)[:, -1]
    df['Max H1'] = max_h1

    duration = (time.time() - start_time) / len(df) * 1000
    print(f"Extraction complete: {len(df)} windows")
    print(f"Average time per window: {duration:.2f} ms")

    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, f"H1_Features_{path.stem}_win{h1_window_size}_tau{tau}.xlsx")
    df.to_excel(output_file, index=False)
    print(f"Results saved to: {output_file}")

    return df


# The __main__ guard is required on Windows when n_jobs != 1 (joblib spawns worker processes)
if __name__ == "__main__":
    # === Data file ===
    data_path = Path("/mnt/c/Users/aphya/Downloads/test1_data.txt")
    output_path = data_path.parent  # Excel file is saved next to the data file

    # === Which columns to use (0 = first column) ===
    # The script prints a min/max/mean/std summary of every column so you can check these.
    time_col = 0         # time in seconds
    acc_col = 1          # acceleration signal used for the H1 features
    pinloc_col = None    # pin location column, or None if the file has none
    pinloc_scale = 1.0   # multiply pin location by this (old DROPBEAR files used 1 / 17.18)

    # === Parameters ===
    freq = 5000          # analysis rate in Hz; data is downsampled to this after filtering
    max_f = 31.1
    min_f = 17.7
    s = 1
    s_W = 5
    pass_value = 100     # low-pass cutoff in Hz

    # === Window size and time delay (in samples at `freq`; 5000 samples = 1 s) ===
    # Set to None to compute automatically from min_f / max_f:
    #   window_size = ((1 / min_f) + 2 * (0.25 / max_f)) * freq  -> 362 samples
    #   tau         = (0.25 / max_f) * freq / 2                  -> 20 samples
    window_size = 362
    tau = 20

    # === Run ===
    extract_dropbear_h1_excel(data_path, freq, max_f, min_f, s, s_W,
                              pass_value, output_path, window_size=window_size, tau=tau,
                              time_col=time_col, acc_col=acc_col,
                              pinloc_col=pinloc_col, pinloc_scale=pinloc_scale)
