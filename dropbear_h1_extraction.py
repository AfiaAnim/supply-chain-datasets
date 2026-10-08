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
def load_numeric_data(path, min_cols=6):
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
    return data


# === Signal Processing ===
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
                              pass_value, output_path, window_size=None, tau=None, n_jobs=-1):
    start_time = time.time()

    path = Path(data_path)
    if not path.is_file():
        raise FileNotFoundError(f"Data file not found: {path}")

    print(f"Loading: {path}")
    data = load_numeric_data(path, min_cols=6)

    processed = process_signal(data, pass_value=pass_value, fs=freq)

    acc = processed[:, 1]
    pinloc = processed[:, 2]
    time_array = processed[:, 3]

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
    pinloc_windows = create_windows(pinloc, h1_window_size, s_W)
    time_windows = create_windows(time_array, h1_window_size, s_W)
    print(f"Number of windows: {len(acc_windows)}")

    scaled_windows = np.array([min_max_scale(w.reshape(-1, 1)).flatten() for w in acc_windows])
    embedded_windows = takens_embedding(scaled_windows, delay=tau, dimension=2, stride=s)

    print("Computing Vietoris-Rips persistence (this can take a while)...")
    max_h1 = extract_h1_persistence(embedded_windows, n_jobs=n_jobs)

    df = pd.DataFrame({
        'Time': time_windows[:, -1],
        'pinloc': pinloc_windows[:, -1],
        'Max H1': max_h1
    })

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
    # === Parameters ===
    freq = 5000
    max_f = 31.1
    min_f = 17.7
    s = 1
    s_W = 5
    pass_value = 100

    # === Window size and time delay (in samples; 5000 samples = 1 s) ===
    # Set to None to compute automatically from min_f / max_f:
    #   window_size = ((1 / min_f) + 2 * (0.25 / max_f)) * freq  -> 362 samples
    #   tau         = (0.25 / max_f) * freq / 2                  -> 20 samples
    window_size = 362
    tau = 20

    data_path = Path("/mnt/c/Users/aphya/Downloads/test1_data.txt")
    output_path = data_path.parent  # Excel file is saved next to the data file

    # === Run ===
    extract_dropbear_h1_excel(data_path, freq, max_f, min_f, s, s_W,
                              pass_value, output_path, window_size=window_size, tau=tau)
