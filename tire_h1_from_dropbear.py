import glob
import os
import re
import time

import numpy as np
import pandas as pd
from gtda.time_series import TakensEmbedding
from gtda.homology import VietorisRipsPersistence
from sklearn.preprocessing import MinMaxScaler
from scipy import signal


# === Find the SEC and MTS files of a trial (by their columns: the file names are not reliable) ===
def find_trial_files(base_path, trial_id):
    sec_file = mts_file = None
    for p in glob.glob(os.path.join(base_path, "**", "*.csv"), recursive=True):
        m = re.search(r"trial[_\- ]?(\d+)", os.path.basename(p), re.I)
        if not m or int(m.group(1)) != trial_id:
            continue
        with open(p, "r", encoding="utf-8-sig", errors="replace") as f:
            header = f.readline()
        if "C1_raw_pF" in header:
            sec_file = p
        elif "RunningTime_s" in header:
            mts_file = p
    if not sec_file or not mts_file:
        raise FileNotFoundError(f"trial {trial_id}: SEC file {sec_file}, MTS file {mts_file}")
    return sec_file, mts_file


# === Signal Processing (DROPBEAR: one file at 5000 Hz -> here: SEC + MTS, uneven rates, offset) ===
def process_signal(sec, mts, freq, pass_value=0.25):
    N = 6
    t_sec = sec["time_s"].to_numpy(float)
    t_sec = t_sec - t_sec[0]
    raw_fs = (len(t_sec) - 1) / t_sec[-1]                         # ~261 Hz, slightly uneven
    t_u = np.arange(0, t_sec[-1], 1 / raw_fs)                     # uniform grid before filtering
    cap = np.interp(t_u, t_sec, sec["C1_raw_pF"].to_numpy(float))
    sos = signal.butter(N=N, Wn=pass_value, btype='lowpass', fs=raw_fs, output='sos')
    cap_filt = signal.sosfiltfilt(sos, cap)

    t_mts = mts["RunningTime_s"].to_numpy(float)
    t_mts = t_mts - t_mts[0]
    disp = mts["Displacement_m"].to_numpy(float) * 1000           # mm (DROPBEAR: pinloc / 17.18)
    force = mts["Force_N"].to_numpy(float)

    # SEC starts ~9 s before the MTS: offset = lag of the best cross-correlation (search 5-15 s)
    tt = np.arange(0, t_mts[-1], 0.1)
    lags = [lag for lag in np.arange(5.0, 15.0, 0.1) if tt[-1] + lag <= t_u[-1]]
    corr = [abs(np.corrcoef(np.interp(tt + lag, t_u, cap_filt), np.interp(tt, t_mts, disp))[0, 1]) for lag in lags]
    offset = lags[int(np.argmax(corr))]
    print(f"  SEC {raw_fs:.0f} Hz -> {freq} Hz, offset {offset:.1f} s (|corr| {max(corr):.2f})")

    t = np.arange(0, min(t_mts[-1], t_u[-1] - offset), 1 / freq)  # common grid on the MTS clock
    cap_out = np.interp(t + offset, t_u, cap_filt)
    disp_out = np.interp(t, t_mts, disp)
    force_out = np.interp(t, t_mts, force)
    return np.column_stack((cap_out, disp_out, force_out, t))


# === Create Sliding Windows ===
def create_windows(array, window_size, step):
    return np.array([array[i:i + window_size] for i in range(0, len(array) - window_size + 1, step)])


# === Normalize ===
def min_max_scale(array):
    scaler = MinMaxScaler((-1, 1))
    return scaler.fit_transform(array)


# === Takens Embedding ===
def takens_embedding(array, delay, dimension, stride):
    embedder = TakensEmbedding(time_delay=delay, dimension=dimension, stride=stride)
    return embedder.fit_transform(array)


# === Extract H1 Persistence ===
def extract_h1_persistence(embedded_windows):
    tda = VietorisRipsPersistence(homology_dimensions=(1,), n_jobs=1)
    diagrams = tda.fit_transform(embedded_windows)
    h1_persistence = np.max(diagrams[:, :, 1] - diagrams[:, :, 0], axis=1)
    return h1_persistence


# === Main Extraction Function ===
def extract_tire_h1_excel(trial_id, freq, max_f, min_f, s, s_W, pass_value, base_path, output_path):
    start_time = time.time()
    print(f"trial_{trial_id:02d}:")
    sec_file, mts_file = find_trial_files(base_path, trial_id)
    processed = process_signal(pd.read_csv(sec_file), pd.read_csv(mts_file), freq, pass_value=pass_value)

    cap = processed[:, 0]
    disp = processed[:, 1]
    force = processed[:, 2]
    time_array = processed[:, 3]

    h1_window_size = int(((1 / min_f) + (0.25 / max_f) * 2) * freq)
    tau = int((0.25 / max_f) * freq / 2)
    print(f"  window {h1_window_size} samples ({h1_window_size / freq:.1f} s), tau {tau} samples ({tau / freq:.2f} s)")

    cap_windows = create_windows(cap, h1_window_size, s_W)
    disp_windows = create_windows(disp, h1_window_size, s_W)
    force_windows = create_windows(force, h1_window_size, s_W)
    time_windows = create_windows(time_array, h1_window_size, s_W)

    scaled_windows = np.array([min_max_scale(w.reshape(-1, 1)).flatten() for w in cap_windows])
    embedded_windows = takens_embedding(scaled_windows, delay=tau, dimension=2, stride=s)

    max_h1 = extract_h1_persistence(embedded_windows)

    df = pd.DataFrame({
        'Time': time_windows[:, -1],
        'Displacement_mm': disp_windows[:, -1],
        'Force_N': force_windows[:, -1],
        'Max H1': max_h1
    })

    duration = (time.time() - start_time) / len(df) * 1000
    print(f"  Extraction complete: {len(df)} windows, {duration:.2f} ms per window")

    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, f"H1_Features_trial{trial_id:02d}_freq{max_f}.xlsx")
    df.to_excel(output_file, index=False)
    print(f"  Results saved to: {output_file}")

    return df


# === Parameters ===
trial_ids = [10, 11, 12, 13, 14, 15]   # trials 01-03 have an unreliable SEC-MTS alignment
freq = 20           # Hz after resampling (DROPBEAR: 5000)
max_f = 0.075       # 3rd harmonic of the loading (DROPBEAR: 31.1)
min_f = 0.025       # 40 s loading cycle (DROPBEAR: 17.7)
s = 3               # Takens stride: 900 points per cloud -> 300 (VR speed)
s_W = 10            # window step: 10 samples = 0.5 s (DROPBEAR: 5)
pass_value = 0.25   # low-pass cutoff in Hz (DROPBEAR: 100)

base_path = r"C:/Users/daniel94/Code/tire_tda/data"          # folder with the SEC and MTS csv files
output_path = r"C:/Users/daniel94/Code/tire_tda/h1_results"

# === Run ===
for trial_id in trial_ids:
    extract_tire_h1_excel(trial_id, freq, max_f, min_f, s, s_W, pass_value, base_path, output_path)
