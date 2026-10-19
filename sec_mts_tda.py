"""
SEC + MTS TDA pipeline (FastTDA). Paste into ONE Jupyter cell and run (or: python sec_mts_tda.py).

SEC  (trial_XX*.csv): time_s, C1_raw_pF, C1_clean_pF, C1_filtered_pF, dC1_pF
MTS  (trial_XX*.csv): RunningTime_s, Displacement_raw_m, Displacement_m, Force_raw_N, Force_N
Files may sit together in DATA_DIR (or in sec/ and mts/ subfolders); each one is recognised by its
columns and paired by trial number. The MTS measurement is the TARGET.

    1. Pair SEC and MTS files, time-align them (cross-correlation, offset ~10 s), resample to 20 Hz.
    2. FastTDA profile per trial (window from the dominant FFT period, MODE = "fft"; or "auto" /
       "fixed"), with a scan of window lengths around that period: clean, smooth,
       correlate with the MTS displacement / force, plot profile, scatter and embedded point clouds
       with the fitted ellipse.
    3. Fourier analysis per trial: FFT spectrum and dominant peak of SEC, MTS and FastTDA, the
       signals wound around a circle at the dominant period, and their constituent sinusoids.
    4. Validation on the reliably aligned trials: short non-overlapping windows, permutation test,
       partial correlation, leave-one-trial-out models (capacitance / + FastTDA / + direction /
       + direction + FastTDA) and loading-vs-unloading classification.

Output (in DATA_DIR/tda_results):
    <trial>_SEC_MTS_TDA.xlsx, <trial>_SEC_MTS_profile.png, _scatter.png, _ellipses.png, _FFT.png,
    _FFT_winding.png (signals wound around a circle), _FFT_components.png/.xlsx (constituent sinusoids)
    SEC_MTS_window_vs_period.png, SEC_MTS_FastTDA_validation.png
    All_Trials_SEC_MTS_Summary.xlsx  (Trials incl. FFT peaks, Validation_R2, Validation_FastTDA,
                                      Validation_Branch, Validation_Windows)

Requirements:
    pip install numpy pandas scipy matplotlib openpyxl
    (+ giotto-tda gudhi only if FASTTDA_ONLY = False)
"""

import glob
import os
import re
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
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

DATA_DIR = "."                          # "." = the Jupyter notebook's folder (where uploaded files land).
                                        # SEC and MTS csv files can sit together in this folder (or in
                                        # sec\ and mts\ subfolders); each file is recognised by its columns
SEC_SUBDIR, MTS_SUBDIR = "sec", "mts"
OUTPUT_DIR = None                       # None = DATA_DIR/tda_results
TRIALS = None                           # None = every trial found; or e.g. ["trial_04", "trial_05"]
TUNE_FROM_TRIAL = 4                     # tune on trial_04 onward (updated setup); 1-3 are processed only

# SEC columns
SEC_TIME_COL = "time_s"
SEC_COL = "C1_filtered_pF"              # signal used for the TDA features
# MTS columns (the targets)
MTS_TIME_COL = "RunningTime_s"
TARGET_COLS = ["Displacement_m", "Force_N"]
LABEL_TARGET = "Displacement_m"         # 2nd column of the W<window> sheet (like CartLocation)

RESAMPLE_HZ = 20.0                      # SEC is resampled to this rate (anti-aliased) before TDA;
                                        # the loading is slow (40 s cycles), 261 Hz is not needed. None = keep

# Time alignment: MTS time = SEC time (from SEC start) - offset
OFFSET_GUESS_S = 10.0                   # "the two correspond from about 10 s of SEC"
OFFSET_SEARCH_S = 5.0                   # search the offset within guess +- this (0 = use the guess)
OFFSET_REF_COL = "Displacement_m"       # MTS column cross-correlated with SEC to find the offset
MIN_OFFSET_CORR = 0.2                   # below this |correlation| the guess is used instead
GOOD_OFFSET_CORR = 0.7                  # offsets with a weaker |correlation| (or on the edge of the
                                        # search range) are flagged as unreliable in the summary

MODE = "fft"                            # "fft"   = window / delay from the dominant FFT period of SEC
                                        # "auto"  = grid fine-tuning (~3 min)
                                        # "fixed" = use FIXED_* below
# Result of the auto-tuning on trials 04-07: W768_T125 = 38.4 s window (~1 loading cycle),
# 6.25 s delay, no detrend; 0 failed fits; FastTDA vs window-mean displacement rho +0.84..+0.89
FIXED_WINDOW, FIXED_DELAY, FIXED_DETREND = 768, 125, False      # samples at RESAMPLE_HZ = 20

# MODE = "fft": window = FFT_WINDOW_CYCLES x dominant period, delay = FFT_DELAY_FRACTION x period
# (a quarter period turns one cycle of a periodic signal into the roundest loop)
FFT_WINDOW_CYCLES = 1.0
FFT_DELAY_FRACTION = 0.25
FFT_WINDOW_SCAN = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]   # window lengths (x period) compared in a plot

# Fine-tuning grid (seconds) and refine multipliers
WINDOW_S = [2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0, 40.0]       # up to one 40 s loading cycle
DELAY_S = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0]
REFINE_WINDOW_FACTORS = [0.8, 0.9, 1.0, 1.1, 1.2]
REFINE_DELAY_FACTORS = [0.75, 0.875, 1.0, 1.125, 1.25]
DETREND_OPTIONS = [False, True]
MAX_DELAY_FRACTION = 0.25
RESOLUTION_WEIGHT = 1.0                 # weight of "shorter window" in the score
TARGET_WEIGHT = 2.0                     # weight of |correlation with the MTS target| in the score
TUNE_TARGET = "auto"                    # "auto" = best of the target stats below, or e.g. "Displacement_m:std"
TARGET_STATS = ["mean", "std"]          # per window: mean = level, std = cyclic amplitude
TUNE_STEP = 2                           # score every 2nd window while tuning (speed); final uses all
FIT_OK_RUN_S = 1.0                      # tuning starts where FastTDA_FitOK first stays True this long

# Embedding / windows
WINDOW_STEP = 1
TAKENS_DIM = 2
TAKENS_STRIDE = 1
MAX_POINTS = 300                        # long windows: embed every k-th point so a cloud has <= this many
TIME_LABEL = "midpoint"
FASTTDA_ONLY = True                     # True = FastTDA only (fast, no giotto-tda / gudhi needed);
                                        # False = also VR (giotto-tda) and Alpha (gudhi)
COMPUTE_VR = not FASTTDA_ONLY
VR_JOBS = -1                            # CPU cores for VR (-1 = all); falls back to 1 if the workers crash
COMPUTE_ALPHA = not FASTTDA_ONLY

# Cleaning / smoothing of VR, Alpha and FastTDA
TRIM_START_S = 0.0                      # extra trim after the aligned start (the SEC start-up is already cut)
EDGE_GUARD_WINDOWS = 5
SPIKE_MADS = 6.0
MAX_INTERP_GAP_S = 2.0
SMOOTH_S = 1.0
TDA_FEATURES = [f for f, on in (("VR", COMPUTE_VR), ("Alpha", COMPUTE_ALPHA), ("FastTDA", True)) if on]
CLEAN_FEATURES = TDA_FEATURES
FEATURE_COLORS = {"VR": "#2a78d6", "Alpha": "#1baf7a", "FastTDA": "#eb6834"}

# Embedded point clouds + fitted ellipses (one figure per trial)
PLOT_ELLIPSES = True
ELLIPSE_TIMES = None                    # None = ELLIPSE_N windows spread over the trial;
                                        # or window-midpoint times in s (MTS clock), e.g. [20, 60, 100]
ELLIPSE_N = 6

# Fourier transform: amplitude spectrum and dominant peak of each signal (per trial)
FFT_ANALYSIS = True
FFT_MIN_HZ = 0.01                       # ignore slower components (drift; period > 100 s)
FFT_MAX_HZ = 0.5                        # x-axis limit of the spectrum plots
FFT_PAD = 8                             # zero-padding factor (finer frequency grid for the peak)
FFT_N_PEAKS = 3                         # peaks listed per signal (dominant + next strongest)
FFT_MIN_CYCLES = 2.0                    # a period must fit at least this many times into the record
FFT_MIN_POWER_PCT = 30.0                # dominant peak must hold >= this % of the spectral power,
                                        # otherwise the signal is reported as "not periodic"
FFT_WINDING = True                      # "winding" plots: each signal wrapped around a circle
FFT_WINDING_OFF = 0.8                   # extra winding period = this x dominant period (misaligned)
FFT_COMPONENTS = True                   # each signal and its constituent sinusoids
FFT_N_COMPONENTS = 6                    # strongest Fourier components drawn per signal
FFT_COMPONENT_CYCLES = 2                # cycles of the dominant period shown
COMPONENT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

# FastTDA validation: short, non-overlapping windows (several per 40 s loading cycle), so every
# window is an (almost) independent data point. Leave-one-trial-out models compare
#   capacitance  |  + FastTDA  |  + direction (capacitance slope)  |  + direction + FastTDA
# for displacement and force, and test whether FastTDA tells loading from unloading.
VALIDATE = True
VAL_WINDOWS_S = [4.0, 8.0, 12.0]                # longer windows leave too few windows per trial
VAL_DELAY_FRACTION = 0.125              # Takens delay = this x window (best of 1/16, 1/8, 1/4 at 8 s)
VAL_TARGETS = ["Displacement_m:mean", "Force_N:mean"]
VAL_SKIP_START_S = 40.0                 # drop the first loading cycle (material conditioning)
VAL_PERMUTATIONS = 1000                 # circular-shift permutation test (respects autocorrelation)
VAL_RELIABLE_ONLY = True                # use only trials whose SEC-MTS alignment is reliable


# ============================================================
# HELPERS
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


def profile_name(window, delay, detrend):
    return f"W{window}_T{delay}" + ("_DT" if detrend else "")


def window_matrix(x, window, step, detrend):
    """All windows as rows, optionally with each window's straight-line trend removed,
    then min-max scaled to [-1, 1]."""
    W = sliding_window_view(x, window)[::step].astype(float)
    if detrend:
        idx = np.arange(window) - (window - 1) / 2
        W = W - W.mean(axis=1, keepdims=True)
        W = W - np.outer(W @ idx / (idx @ idx), idx)
    lo, hi = W.min(axis=1, keepdims=True), W.max(axis=1, keepdims=True)
    span = np.where(hi > lo, hi - lo, 1.0)
    return 2 * (W - lo) / span - 1


def embed_stride(window, delay):
    span = window - (TAKENS_DIM - 1) * delay - 1
    return max(TAKENS_STRIDE, int(np.ceil((span + 1) / MAX_POINTS)) if MAX_POINTS else 1)


def point_clouds(x, window, delay, detrend, step=WINDOW_STEP):
    """Takens point clouds (same indexing as gtda TakensEmbedding, aligned to the window end)."""
    W = window_matrix(x, window, step, detrend)
    stride = embed_stride(window, delay)
    span = window - (TAKENS_DIM - 1) * delay - 1
    first = span % stride
    last = first + (span // stride) * stride
    cols = [W[:, first + d * delay: last + d * delay + 1: stride] for d in range(TAKENS_DIM)]
    starts = np.arange(0, len(x) - window + 1, step)
    return starts, np.stack(cols, axis=2)


def window_stats(y, window, step):
    W = sliding_window_view(y, window)[::step]
    return W.mean(axis=1), W.std(axis=1)


def spearman(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return np.nan
    return float(pd.Series(a[m]).corr(pd.Series(b[m]), method="spearman"))


# ============================================================
# FEATURES: FastTDA, VR, Alpha
# ============================================================

def fasttda_fit(pc):
    """SVD-LSE ellipse fit (A + C = 1) on a 2-D point cloud.
    Returns (2b, fit_ok, ellipse) - ellipse = dict with the curve in point-cloud coordinates, or None."""
    X, Y = pc[:, 0], pc[:, 1]
    mx, my = X.mean(), Y.mean()
    Xc, Yc = X - mx, Y - my
    rng = max(np.abs(Xc).max(), np.abs(Yc).max())
    if not np.isfinite(rng) or rng == 0:
        return np.nan, False, None
    k = 1.5 / rng
    Xc, Yc = Xc * k, Yc * k
    G = np.column_stack([Xc**2 - Yc**2, Xc * Yc, Xc, Yc, np.ones_like(Xc)])
    U, S, Vt = np.linalg.svd(G, full_matrices=False)
    Sinv = np.where(S > np.finfo(float).eps * max(G.shape) * S[0], 1.0 / S, 0.0)
    A, B, D, E, F = (Vt.T * Sinv) @ (U.T @ -(Yc**2))
    C = 1.0 - A
    if 4 * A * C - B**2 <= 0:
        return np.nan, False, None
    th = 0.5 * np.arctan2(B, A - C)
    c, s = np.cos(th), np.sin(th)
    Ap, Cp = A*c*c + B*c*s + C*s*s, A*s*s - B*c*s + C*c*c
    Dp, Ep = D*c + E*s, -D*s + E*c
    if Ap <= 0 or Cp <= 0:
        return np.nan, False, None
    Fn = F - Dp**2 / (4 * Ap) - Ep**2 / (4 * Cp)
    if Fn >= 0:
        return np.nan, False, None
    a_u, a_v = np.sqrt(-Fn / Ap), np.sqrt(-Fn / Cp)            # semi-axes along the rotated u, v
    u0, v0 = -Dp / (2 * Ap), -Ep / (2 * Cp)                    # centre in the rotated frame
    to_pc = lambda u, v: (mx + (c * u - s * v) / k, my + (s * u + c * v) / k)
    t = np.linspace(0, 2 * np.pi, 200)
    ex, ey = to_pc(u0 + a_u * np.cos(t), v0 + a_v * np.sin(t))
    minor = (u0 + np.array([-a_u, a_u]), np.array([v0, v0])) if a_u < a_v else \
            (np.array([u0, u0]), v0 + np.array([-a_v, a_v]))
    ell = {"x": ex, "y": ey, "center": to_pc(u0, v0), "minor": to_pc(*minor)}
    return 2.0 * min(a_u, a_v), True, ell


def fasttda_minor_axis(pc):
    """FastTDA feature: minor axis 2b of the fitted ellipse (fit coordinates) -> (2b, fit_ok)."""
    return fasttda_fit(pc)[:2]


def vr_max_h1_batch(clouds):
    from gtda.homology import VietorisRipsPersistence
    try:
        diagrams = VietorisRipsPersistence(homology_dimensions=[1], n_jobs=VR_JOBS).fit_transform(clouds)
    except Exception as e:                              # e.g. BrokenProcessPool: workers can't import gtda
        print(f"    VR in parallel failed ({type(e).__name__}) - running on one core")
        diagrams = VietorisRipsPersistence(homology_dimensions=[1], n_jobs=1).fit_transform(clouds)
    pers = diagrams[:, :, 1] - diagrams[:, :, 0]
    pers[(diagrams[:, :, 2] != 1) | ~np.isfinite(pers)] = 0.0
    return pers.max(axis=1)


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


# ============================================================
# LOADING + TIME ALIGNMENT
# ============================================================

def file_kind(path):
    """'sec' or 'mts' from the csv header (file names are not reliable: trial_01_1 is MTS,
    trial_04_1 is SEC)."""
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
        if kind is None:
            continue
        group = sec if kind == "sec" else mts
        if name in group:
            print(f"  {name}: two {kind.upper()} files ({os.path.basename(group[name][1])}, "
                  f"{os.path.basename(p)}) - using the first")
            continue
        group[name] = (num, p)
    if not sec:
        raise FileNotFoundError(f"No SEC csv files (with a {SEC_COL!r} column) found in {os.path.abspath(base)}")
    pairs = []
    for name in sorted(sec):
        if TRIALS and name not in TRIALS:
            continue
        if name not in mts:
            print(f"  {name}: SEC file found but no MTS file - skipped")
            continue
        num, sp = sec[name]
        pairs.append((name, num, sp, mts[name][1]))
        print(f"  {name}: SEC {os.path.basename(sp)}  +  MTS {os.path.basename(mts[name][1])}")
    print(f"Found {len(pairs)} SEC/MTS pairs in {os.path.abspath(base)}")
    return pairs


def estimate_offset(t_sec, x_sec, t_mts, y_mts):
    """Offset (s) with MTS time = SEC time - offset, from the cross-correlation peak."""
    if not OFFSET_SEARCH_S:
        return OFFSET_GUESS_S, np.nan
    fs = 10.0                                            # the loading is slow: 10 Hz is enough
    a = np.interp(np.arange(0, t_sec[-1], 1 / fs), t_sec, x_sec)
    b = np.interp(np.arange(0, t_mts[-1], 1 / fs), t_mts, y_mts)
    best = (OFFSET_GUESS_S, 0.0)
    for lag in np.arange(OFFSET_GUESS_S - OFFSET_SEARCH_S, OFFSET_GUESS_S + OFFSET_SEARCH_S + 1e-9, 1 / fs):
        i0 = int(round(lag * fs))
        if i0 < 0:
            continue
        n = min(len(b), len(a) - i0)
        if n < 0.9 * len(b):                             # need (almost) the whole MTS record
            continue
        aa, bb = a[i0:i0 + n], b[:n]
        if aa.std() == 0 or bb.std() == 0:
            continue
        r = abs(np.corrcoef(aa, bb)[0, 1])
        if r > best[1]:
            best = (float(lag), float(r))
    if best[1] < MIN_OFFSET_CORR:
        print(f"    offset: best |corr| {best[1]:.2f} < {MIN_OFFSET_CORR} -> using the guess {OFFSET_GUESS_S} s")
        return OFFSET_GUESS_S, best[1]
    return best


def load_pair(name, num, sec_path, mts_path):
    sec = read_numeric_csv(sec_path)
    mts = read_numeric_csv(mts_path)
    for c in (SEC_TIME_COL, SEC_COL):
        if c not in sec:
            raise KeyError(f"{sec_path}: column {c!r} not found. Columns: {list(sec.columns)}")
    for c in [MTS_TIME_COL] + TARGET_COLS:
        if c not in mts:
            raise KeyError(f"{mts_path}: column {c!r} not found. Columns: {list(mts.columns)}")
    t_sec = sec[SEC_TIME_COL].to_numpy(float)
    t_sec = t_sec - t_sec[0]                             # SEC time starts at ~315 s -> re-zero
    x = sec[SEC_COL].to_numpy(float)
    t_mts = mts[MTS_TIME_COL].to_numpy(float)
    t_mts = t_mts - t_mts[0]
    fs = (len(t_sec) - 1) / (t_sec[-1] - t_sec[0])

    offset, r = estimate_offset(t_sec, x, t_mts, mts[OFFSET_REF_COL].to_numpy(float))
    if RESAMPLE_HZ and RESAMPLE_HZ < fs:                            # anti-alias, then resample
        sos = signal.butter(4, 0.4 * RESAMPLE_HZ, btype="lowpass", fs=fs, output="sos")
        x = signal.sosfiltfilt(sos, x)
        t_new = np.arange(t_sec[0], t_sec[-1], 1 / RESAMPLE_HZ)
        x, t_sec, fs = np.interp(t_new, t_sec, x), t_new, RESAMPLE_HZ
    keep = (t_sec >= offset) & (t_sec <= offset + t_mts[-1])        # overlap only
    t_al = t_sec[keep] - offset                                     # aligned = MTS time
    targets = {c: np.interp(t_al, t_mts, mts[c].to_numpy(float)) for c in TARGET_COLS}
    on_edge = OFFSET_SEARCH_S and abs(abs(offset - OFFSET_GUESS_S) - OFFSET_SEARCH_S) < 0.11
    reliable = bool(np.isfinite(r) and r >= GOOD_OFFSET_CORR and not on_edge)
    print(f"  {name}: SEC {t_sec[-1]:.1f} s ({fs:.0f} Hz used), MTS {t_mts[-1]:.1f} s | offset {offset:.2f} s"
          + (f" (|corr| {r:.2f})" if np.isfinite(r) else "")
          + f" | kept SEC {offset:.1f}-{offset + t_mts[-1]:.1f} s ({keep.sum()} samples)"
          + ("" if reliable else "  <-- alignment UNRELIABLE"))
    return {"name": name, "num": num, "t": t_al, "x": x[keep], "fs": fs, "targets": targets,
            "offset": offset, "offset_corr": r, "offset_ok": reliable, "mts_t": t_mts, "mts": mts}


# ============================================================
# 1) FINE-TUNING (FastTDA + target correlation)
# ============================================================

def first_fit_ok_index(fit_ok, run):
    fit_ok = np.asarray(fit_ok, bool)
    run = max(1, min(int(run), len(fit_ok)))
    full = np.flatnonzero(np.convolve(fit_ok.astype(int), np.ones(run, int), mode="valid") == run)
    return int(full[0]) if full.size else len(fit_ok)


def target_keys():
    return [f"{c}:{s}" for c in TARGET_COLS for s in TARGET_STATS]


def score_setting(tr, window, delay, detrend):
    step = TUNE_STEP
    starts, clouds = point_clouds(tr["x"], window, delay, detrend, step)
    res = [fasttda_minor_axis(pc) for pc in clouds]
    v = np.array([r[0] for r in res], float)
    ok = np.array([r[1] for r in res], bool)
    i0 = first_fit_ok_index(ok, round(FIT_OK_RUN_S * tr["fs"] / step))
    out = {"Invalid_pct": 100.0 * (1 - ok[i0:].mean()) if ok[i0:].size else 100.0}
    vv = v[i0:][ok[i0:]]
    if vv.size < 10:
        return {**out, "Spike_pct": np.nan, "Roughness": np.nan,
                **{f"rho_{k}": np.nan for k in target_keys()}}
    med = np.median(vv)
    mad = 1.4826 * np.median(np.abs(vv - med))
    iqr = np.subtract(*np.percentile(vv, [75, 25]))
    out["Spike_pct"] = 100.0 * np.mean(vv > med + SPIKE_MADS * mad) if mad > 0 else 0.0
    out["Roughness"] = float(np.median(np.abs(np.diff(vv))) / iqr) if iqr > 0 else np.nan
    for c in TARGET_COLS:
        mean, std = window_stats(tr["targets"][c], window, step)
        for s, arr in (("mean", mean), ("std", std)):
            if s in TARGET_STATS:
                out[f"rho_{c}:{s}"] = spearman(v[i0:][ok[i0:]], arr[i0:][ok[i0:]])
    return out


def add_scores(summary):
    s = summary.copy()
    ranks = [s["Invalid_pct"].rank(), s["Spike_pct"].rank(), s["Roughness"].rank()]
    weights = [1.0, 1.0, 1.0]
    if RESOLUTION_WEIGHT > 0:
        ranks.append(s["Window_s"].rank())
        weights.append(RESOLUTION_WEIGHT)
    if TARGET_WEIGHT > 0 and "Target_absrho" in s:
        ranks.append(s["Target_absrho"].rank(ascending=False))
        weights.append(TARGET_WEIGHT)
    r = pd.concat(ranks, axis=1).to_numpy(float)
    s["Score"] = (r * np.array(weights)).sum(axis=1) / sum(weights)
    s = s.sort_values(["Score", "Invalid_pct"]).reset_index(drop=True)
    s.insert(0, "Rank", np.arange(1, len(s) + 1))
    return s


def run_grid(settings, stage, trials, done):
    rows = []
    fs = np.mean([tr["fs"] for tr in trials])
    todo = [(w, d, dt) for w, d, dt in settings
            if (w, d, dt) not in done and d <= MAX_DELAY_FRACTION * w
            and all(w <= len(tr["x"]) for tr in trials)
            and (w - d * (TAKENS_DIM - 1) - 1) // embed_stride(w, d) + 1 >= 5]
    for i, (w, d, dt) in enumerate(todo, 1):
        per = pd.DataFrame([score_setting(tr, w, d, dt) for tr in trials])
        r = {"Profile": profile_name(w, d, dt), "Window_Size": w, "Takens_Delay": d, "Detrend": dt,
             "Window_s": w / fs, "Delay_s": d / fs, "Stage": stage,
             "Invalid_pct": per["Invalid_pct"].mean(), "Worst_Invalid_pct": per["Invalid_pct"].max(),
             "Spike_pct": per["Spike_pct"].mean(), "Roughness": per["Roughness"].mean()}
        for k in target_keys():
            r[f"absrho_{k}"] = per[f"rho_{k}"].abs().mean()        # mean |rho| over trials
        keys = target_keys() if TUNE_TARGET == "auto" else [TUNE_TARGET]
        best_k = max(keys, key=lambda k: -1 if not np.isfinite(r[f"absrho_{k}"]) else r[f"absrho_{k}"])
        r["Target"], r["Target_absrho"] = best_k, r[f"absrho_{best_k}"]
        done.add((w, d, dt))
        rows.append(r)
        print(f"  [{stage} {i}/{len(todo)}] {r['Profile']:12s} window {w / fs:4.2f} s, delay {d / fs:.3f} s | "
              f"invalid {r['Invalid_pct']:5.2f}%  rough {r['Roughness']:.3f} | "
              f"|rho| {r['Target_absrho']:.2f} with {best_k}")
    return rows


def fine_tune(trials):
    fs = np.mean([tr["fs"] for tr in trials])
    done = set()
    coarse = sorted({(int(round(w * fs)), max(1, int(round(d * fs))), bool(dt))
                     for w in WINDOW_S for d in DELAY_S for dt in DETREND_OPTIONS})
    print(f"\nFine-tuning on {', '.join(tr['name'] for tr in trials)} (every {TUNE_STEP}th window)")
    print(f"Stage 1: coarse grid, {len(coarse)} settings")
    rows = run_grid(coarse, "coarse", trials, done)
    best = add_scores(pd.DataFrame(rows)).iloc[0]
    bw, bd, bdt = int(best["Window_Size"]), int(best["Takens_Delay"]), bool(best["Detrend"])
    refine = sorted({(max(3, int(round(bw * fw))), max(1, int(round(bd * fd))), bdt)
                     for fw in REFINE_WINDOW_FACTORS for fd in REFINE_DELAY_FACTORS})
    print(f"\nStage 2: refine around {best['Profile']}, "
          f"{len([r for r in refine if r not in done])} new settings")
    rows += run_grid(refine, "refine", trials, done)
    return add_scores(pd.DataFrame(rows))


# ============================================================
# 2) PROFILE + 3) CLEANING
# ============================================================

def extract_profile(tr, window, delay, detrend):
    starts, clouds = point_clouds(tr["x"], window, delay, detrend)
    lab = starts + (window // 2 if TIME_LABEL == "midpoint" else window - 1)
    t0 = time.time()
    ftda = [fasttda_minor_axis(pc) for pc in clouds]
    vr = vr_max_h1_batch(clouds) if COMPUTE_VR else np.full(len(starts), np.nan)
    alpha = np.array([alpha_max_h1(pc) for pc in clouds]) if COMPUTE_ALPHA else np.full(len(starts), np.nan)
    print(f"    VR / Alpha / FastTDA for {len(starts)} windows: {time.time() - t0:.1f} s")
    f = np.array([v for v, _ in ftda], float)
    df = pd.DataFrame({
        "Time": tr["t"][lab],                            # aligned time = MTS time
        LABEL_TARGET: tr["targets"][LABEL_TARGET][lab],
        "VR": vr, "Alpha": alpha, "FastTDA": f,
        "FastTDA_ZeroInvalid": np.where(np.isfinite(f), f, 0.0),
        "FastTDA_FitOK": [ok for _, ok in ftda],
        "Window_Number": np.arange(len(starts)), "Window_Size": window,
    })
    tg = {"Time": df["Time"].to_numpy(), "SEC_Time_s": df["Time"].to_numpy() + tr["offset"]}
    for c in TARGET_COLS:
        mean, std = window_stats(tr["targets"][c], window, WINDOW_STEP)
        tg[f"{c}:mean"], tg[f"{c}:std"] = mean, std
    return df, pd.DataFrame(tg), tr["t"][starts + window - 1]


def clean_feature(df, col, t_end, fs):
    v = df[col].to_numpy(float)
    ok = np.isfinite(v)
    if col == "FastTDA":
        ok &= df["FastTDA_FitOK"].to_numpy(bool)
    reason = np.full(len(df), "ok", dtype=object)
    reason[~ok] = "invalid"
    near = np.convolve((~ok).astype(int), np.ones(2 * EDGE_GUARD_WINDOWS + 1, int), mode="same") > 0
    reason[near & ok] = "gap_edge"
    reason[t_end - t_end[0] < TRIM_START_S] = "startup"
    keep = reason == "ok"
    med = np.median(v[keep])
    mad = 1.4826 * np.median(np.abs(v[keep] - med))
    if mad > 0:
        reason[keep & (v > med + SPIKE_MADS * mad)] = "spike"
    good = reason == "ok"
    clean = np.where(good, v, np.nan)
    tt = df["Time"].to_numpy(float)
    filled = np.interp(tt, tt[good], v[good])
    max_gap = int(round(MAX_INTERP_GAP_S * fs / WINDOW_STEP))
    edges = np.flatnonzero(np.diff(np.r_[0, (~good).astype(int), 0]))
    for a, b in zip(edges[::2], edges[1::2]):
        if a > 0 and b < len(df) and b - a <= max_gap:
            clean[a:b] = filled[a:b]
    clean[reason == "startup"] = np.nan
    if SMOOTH_S:
        n = max(1, int(round(SMOOTH_S * fs / WINDOW_STEP)) | 1)
        smooth = pd.Series(clean).rolling(n, center=True, min_periods=n // 2 + 1).median().to_numpy()
    else:
        smooth = clean.copy()
    return clean, reason, smooth


def clean_profile(df, t_end, fs):
    out = {"Time": df["Time"].to_numpy(float)}
    for col in CLEAN_FEATURES:
        if not np.isfinite(df[col]).any():
            continue
        clean, mask, smooth = clean_feature(df, col, t_end, fs)
        out.update({col: df[col].to_numpy(float), f"{col}_clean": clean,
                    f"{col}_smooth": smooth, f"{col}_Mask": mask})
    return pd.DataFrame(out)


# ============================================================
# FIGURES
# ============================================================

def plot_tuning(summary, out_dir):
    s = summary[summary["Stage"] == "coarse"]
    if s.empty:
        return
    cmap = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#6da7ec", "#256abf", "#0d366b"])
    mats = [("Invalid_pct", "Invalid FastTDA fits (%) - lower is better", "min", "{:.1f}"),
            ("Target_absrho", "|Spearman| with MTS target - higher is better", "max", "{:.2f}"),
            ("Score", "Tuning score - lower is better", "min", "{:.1f}")]
    fig, axes = plt.subplots(1, 3, figsize=(17, 0.45 * s["Window_s"].nunique() + 2.8))
    for ax, (col, title, agg, fmt) in zip(axes, mats):
        m = s.pivot_table(index="Window_s", columns="Delay_s", values=col, aggfunc=agg)
        a = m.to_numpy(float)
        im = ax.imshow(a, cmap=cmap, aspect="auto")
        ax.set_xticks(range(a.shape[1]), [f"{c:.3f}" for c in m.columns], fontsize=8)
        ax.set_yticks(range(a.shape[0]), [f"{r:.2f}" for r in m.index], fontsize=8)
        ax.set_xlabel("Takens delay (s)")
        ax.set_ylabel("Window length (s)")
        ax.set_title(title, fontsize=10, fontweight="bold")
        lo, hi = np.nanmin(a), np.nanmax(a)
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                if np.isfinite(a[i, j]):
                    dark = (a[i, j] - lo) / (hi - lo + 1e-12) > 0.55
                    ax.text(j, i, fmt.format(a[i, j]), ha="center", va="center", fontsize=7,
                            color="white" if dark else "#0b0b0b")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("SEC/MTS fine-tuning (coarse grid; each cell = better of detrend on/off)", fontweight="bold")
    fig.tight_layout()
    png = os.path.join(out_dir, "SEC_MTS_tuning.png")
    fig.savefig(png, dpi=150)
    print(f"Saved: {png}")
    plt.show()
    plt.close(fig)


def plot_profile(tr, df, cdf, meta, best_key, rhos, out_dir):
    fig, axes = plt.subplots(2 + len(TDA_FEATURES), 1, figsize=(11, 4 + 2 * len(TDA_FEATURES)), sharex=True)
    t_al = tr["t"]
    ax = axes[0]
    ax.plot(t_al, tr["x"], color="#52514e", lw=0.8)
    ax.set_ylabel(f"SEC {SEC_COL}")
    ax = axes[1]
    c0 = TARGET_COLS[0]
    ax.plot(tr["mts_t"], tr["mts"][c0], color="#6250d6", lw=0.8, label=c0)
    ax.set_ylabel(c0)
    if len(TARGET_COLS) > 1:
        ax2 = ax.twinx()
        ax2.plot(tr["mts_t"], tr["mts"][TARGET_COLS[1]], color="#e87ba4", lw=0.8, alpha=0.8,
                 label=TARGET_COLS[1])
        ax2.set_ylabel(TARGET_COLS[1])
        ax2.spines[["top"]].set_visible(False)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper right", frameon=False, fontsize=8, ncol=2)
    for ax, col in zip(axes[2:], TDA_FEATURES):
        colr = FEATURE_COLORS[col]
        y = df[col].to_numpy(float)
        ax.plot(df["Time"], y, color=colr, lw=0.8, label=f"{col} raw")
        if f"{col}_smooth" in cdf:
            rho = rhos.get(col, np.nan)
            ax.plot(df["Time"], cdf[f"{col}_smooth"], color="#0d366b", lw=1.3,
                    label=f"{col}_smooth (rho {rho:+.2f} with {best_key})")
        if col == "FastTDA":
            fin = y[np.isfinite(y)]
            if fin.size:
                ax.set_ylim(np.nanmin(fin) * 0.9, np.percentile(fin, 99.5) * 1.1)
        ax.legend(loc="upper right", frameon=False, fontsize=8, ncol=2)
        ax.set_ylabel(col)
    for ax in axes:
        ax.grid(color="#e4e3df", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("Time (s), MTS clock (SEC time - offset)")
    fig.suptitle(f"{tr['name']} {meta['Profile']}: window {meta['Window_Size'] / tr['fs']:.2f} s, "
                 f"delay {meta['Takens_Delay'] / tr['fs']:.3f} s"
                 f"{', detrended' if meta['Detrend'] else ''} | SEC-MTS offset {tr['offset']:.2f} s",
                 fontweight="bold")
    fig.tight_layout()
    png = os.path.join(out_dir, f"{tr['name']}_SEC_MTS_profile.png")
    fig.savefig(png, dpi=150)
    print(f"Saved: {png}")
    plt.show()
    plt.close(fig)


def plot_scatter(tr, cdf, tg, best_key, out_dir):
    fig, axes = plt.subplots(1, len(TDA_FEATURES), figsize=(4.4 * len(TDA_FEATURES), 3.8), squeeze=False)
    axes = axes[0]
    y = tg[best_key].to_numpy(float)
    for ax, col in zip(axes, TDA_FEATURES):
        colr = FEATURE_COLORS[col]
        if f"{col}_smooth" not in cdf:
            ax.set_visible(False)
            continue
        x = cdf[f"{col}_smooth"].to_numpy(float)
        m = np.isfinite(x) & np.isfinite(y)
        ax.scatter(y[m][::5], x[m][::5], s=4, color=colr, alpha=0.4, lw=0)
        ax.set_xlabel(f"MTS {best_key} (per window)")
        ax.set_ylabel(f"{col}_smooth")
        ax.set_title(f"{col}: Spearman {spearman(x, y):+.2f}", fontsize=10, fontweight="bold")
        ax.grid(color="#e4e3df", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{tr['name']}: SEC TDA features vs MTS target", fontweight="bold")
    fig.tight_layout()
    png = os.path.join(out_dir, f"{tr['name']}_SEC_MTS_scatter.png")
    fig.savefig(png, dpi=150)
    print(f"Saved: {png}")
    plt.show()
    plt.close(fig)


def plot_ellipses(tr, df, tg, window, delay, detrend, best_key, out_dir):
    """Takens point cloud of selected windows with the FastTDA ellipse and its minor axis."""
    starts, clouds = point_clouds(tr["x"], window, delay, detrend)
    tt = df["Time"].to_numpy(float)
    if ELLIPSE_TIMES:
        idx = [int(np.argmin(np.abs(tt - t))) for t in ELLIPSE_TIMES]
    else:
        idx = np.linspace(0, len(tt) - 1, ELLIPSE_N + 2)[1:-1].round().astype(int).tolist()
    ncol = min(3, len(idx))
    nrow = int(np.ceil(len(idx) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.5 * nrow), squeeze=False,
                             gridspec_kw={"hspace": 0.35})
    for ax, i in zip(axes.flat, idx):
        pc = clouds[i]
        v, ok, ell = fasttda_fit(pc)
        sc = ax.scatter(pc[:, 0], pc[:, 1], c=np.arange(len(pc)), cmap="viridis", s=7, lw=0)
        ax.plot(pc[:, 0], pc[:, 1], color="#b8b7b2", lw=0.4, zorder=0)
        if ok:
            ax.plot(ell["x"], ell["y"], color="#eb6834", lw=2, label="fitted ellipse")
            ax.plot(*ell["minor"], color="#0d366b", lw=2, label=f"minor axis, FastTDA 2b = {v:.3f}")
            ax.plot(*ell["center"], "+", color="#0d366b", ms=10, mew=2)
        else:
            ax.text(0.5, 0.5, "ellipse fit failed", transform=ax.transAxes, ha="center",
                    color="#c0392b", fontweight="bold")
        ax.set_title(f"t = {tt[i]:.1f} s | {best_key} = {tg[best_key].iloc[i]:.3g}", fontsize=9,
                     fontweight="bold")
        ax.set_xlabel("x(t)")
        ax.set_ylabel(f"x(t + {delay / tr['fs']:.2f} s)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(color="#e4e3df", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        if ok:
            ax.legend(loc="upper left", frameon=False, fontsize=7)
    for ax in list(axes.flat)[len(idx):]:
        ax.set_visible(False)
    fig.colorbar(sc, ax=axes, fraction=0.02, pad=0.02, label="point order in window")
    fig.suptitle(f"{tr['name']}: Takens embedding of {SEC_COL} (window {window / tr['fs']:.1f} s) "
                 f"and FastTDA ellipse fit", fontweight="bold")
    png = os.path.join(out_dir, f"{tr['name']}_SEC_MTS_ellipses.png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"Saved: {png}")
    plt.show()
    plt.close(fig)


def fft_spectrum(y, fs):
    """One-sided amplitude spectrum of a detrended, Hann-windowed series (zero-padded)."""
    y = np.asarray(y, float)
    t = np.arange(len(y))
    good = np.isfinite(y)
    y = np.interp(t, t[good], y[good])                       # fill gaps (e.g. FastTDA failed fits)
    y = signal.detrend(y)                                     # remove offset + linear drift
    win = np.hanning(len(y))
    n = int(2 ** np.ceil(np.log2(len(y) * FFT_PAD)))
    amp = 2 * np.abs(np.fft.rfft(y * win, n)) / win.sum()     # amplitude in signal units
    return np.fft.rfftfreq(n, 1 / fs), amp


def fft_peaks(f, amp, record_s):
    """Strongest local maxima between max(FFT_MIN_HZ, FFT_MIN_CYCLES / record) and Nyquist, with the
    share of spectral power (in that band) that lies within +-1 raw frequency bin of each peak."""
    m = f >= max(FFT_MIN_HZ, FFT_MIN_CYCLES / record_s)
    fm, am = f[m], amp[m]
    pk, _ = signal.find_peaks(am, distance=max(1, int(2 * FFT_PAD)))
    pk = pk[np.argsort(am[pk])[::-1][:FFT_N_PEAKS]]
    total = np.sum(am ** 2)
    out = []
    for i in pk:
        near = slice(max(0, i - FFT_PAD), i + FFT_PAD + 1)       # +-1 raw bin
        out.append((float(fm[i]), float(am[i]), float(100 * np.sum(am[near] ** 2) / total)))
    return out


def fft_series(tr, df, cdf):
    """(name, time, values, fs, colour) for SEC, the MTS targets and the smoothed TDA features."""
    series = [(f"SEC {SEC_COL}", tr["t"], tr["x"], tr["fs"], "#52514e")]
    for c, colr in zip(TARGET_COLS, ["#6250d6", "#e87ba4"]):
        series.append((f"MTS {c}", tr["t"], tr["targets"][c], tr["fs"], colr))
    for col in TDA_FEATURES:
        if f"{col}_smooth" in cdf and np.isfinite(cdf[f"{col}_smooth"]).sum() > 20:
            y = cdf[f"{col}_smooth"].to_numpy(float)
            ok = np.flatnonzero(np.isfinite(y))
            sl = slice(ok[0], ok[-1] + 1)                     # drop the NaN start/end of the profile
            series.append((f"{col}_smooth", df["Time"].to_numpy(float)[sl], y[sl],
                           tr["fs"] / WINDOW_STEP, FEATURE_COLORS[col]))
    return series


def fft_analysis(tr, df, cdf, out_dir):
    """Time series and FFT spectrum of SEC, MTS targets and FastTDA, with the dominant peak marked."""
    series = fft_series(tr, df, cdf)
    fig, axes = plt.subplots(len(series), 2, figsize=(14, 2.3 * len(series)),
                             gridspec_kw={"width_ratios": [1.6, 1]}, squeeze=False)
    out = {}
    for (name, t, y, fs, colr), (ax_t, ax_f) in zip(series, axes):
        f, amp = fft_spectrum(y, fs)
        peaks = fft_peaks(f, amp, t[-1] - t[0])
        ax_t.plot(t, y, color=colr, lw=0.8)
        ax_t.set_ylabel(name, fontsize=8)
        ax_f.plot(f, amp, color=colr, lw=1.2)
        ax_f.set_xlim(0, FFT_MAX_HZ)
        ax_f.axvspan(0, max(FFT_MIN_HZ, FFT_MIN_CYCLES / (t[-1] - t[0])), color="#e4e3df", lw=0)
        periodic = bool(peaks) and peaks[0][2] >= FFT_MIN_POWER_PCT
        for k, (pf, pa, pp) in enumerate(peaks):
            main = k == 0 and periodic
            ax_f.plot(pf, pa, "o" if main else "^", color="#c0392b" if main else "#0d366b",
                      ms=7 if main else 5)
            ax_f.annotate(f"{pf:.4f} Hz, T = {1 / pf:.1f} s ({pp:.0f}% power)" if main else f"{1 / pf:.1f} s",
                          (pf, pa), xytext=(8, -3), textcoords="offset points",
                          fontsize=8 if main else 7, fontweight="bold" if main else "normal")
        ax_f.set_ylim(0, 1.25 * max(amp[f >= FFT_MIN_HZ].max(), 1e-12))
        ax_t.set_title(f"dominant period {1 / peaks[0][0]:.1f} s" if periodic else
                       "no dominant periodic peak (not periodic / record too short)",
                       fontsize=8, loc="right", color="#0b0b0b" if periodic else "#c0392b")
        key = name.split(" ")[-1]
        out[f"{key}_peak_Hz"] = peaks[0][0] if periodic else np.nan
        out[f"{key}_period_s"] = 1 / peaks[0][0] if periodic else np.nan
        out[f"{key}_peak_amp"] = peaks[0][1] if periodic else np.nan
        out[f"{key}_peak_power_pct"] = peaks[0][2] if peaks else np.nan
        out[f"{key}_other_periods_s"] = ", ".join(f"{1 / p[0]:.1f}" for p in peaks[1:]) if periodic else ""
        for ax in (ax_t, ax_f):
            ax.grid(color="#e4e3df", lw=0.6)
            ax.spines[["top", "right"]].set_visible(False)
    axes[-1, 0].set_xlabel("Time (s), MTS clock")
    axes[-1, 1].set_xlabel("Frequency (Hz)")
    fig.suptitle(f"{tr['name']}: Fourier analysis - time series (left) and FFT amplitude spectrum "
                 f"(right, Hann window, detrended); dominant peak in red, grey band = ignored",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    png = os.path.join(out_dir, f"{tr['name']}_SEC_MTS_FFT.png")
    fig.savefig(png, dpi=150)
    print(f"Saved: {png}")
    plt.show()
    plt.close(fig)
    sec_p, mts_p = out.get(f"{SEC_COL}_period_s"), out.get(f"{TARGET_COLS[0]}_period_s")
    print("    FFT dominant periods: " + ", ".join(
        f"{k[:-9]} {v:.1f} s" if np.isfinite(v) else f"{k[:-9]} not periodic"
        for k, v in out.items() if k.endswith("_period_s")))
    if sec_p and mts_p and np.isfinite(sec_p) and np.isfinite(mts_p):
        out["SEC_matches_MTS"] = bool(abs(sec_p - mts_p) / mts_p < 0.1)
    return out


def plot_fft_winding(tr, df, cdf, period, out_dir):
    """Each signal wrapped around a circle (angle = 2 pi t / winding period, radius = signal).
    At the dominant period the loops line up and the centre of mass (black dot) moves away from
    the centre - its distance is what the Fourier transform measures at that frequency."""
    series = fft_series(tr, df, cdf)
    periods = [FFT_WINDING_OFF * period, period]
    fig, axes = plt.subplots(len(periods), len(series), figsize=(3.4 * len(series), 4.3 * len(periods)),
                             subplot_kw={"projection": "polar"}, squeeze=False,
                             gridspec_kw={"hspace": 0.55, "wspace": 0.3})
    for row, pw in zip(axes, periods):
        for ax, (name, t, y, fs, colr) in zip(row, series):
            y = np.asarray(y, float)
            good = np.isfinite(y)
            t, y = t[good], y[good]
            r = 0.15 + 0.85 * (y - y.min()) / (np.ptp(y) or 1.0)        # positive radius 0.15..1
            th = 2 * np.pi * (t - t[0]) / pw
            ax.plot(th, r, color=colr, lw=1.1, label=f"{name.split(' ')[-1]}: period of {pw:.2f} s")
            g = np.mean(r * np.exp(1j * th))                             # centre of mass
            ax.plot(np.angle(g), np.abs(g), "o", color="#0b0b0b", ms=6, zorder=5)
            ax.set_ylim(0, 1.05)
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.grid(color="#e4e3df", lw=0.6)
            ax.spines["polar"].set_color("#e4e3df")
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), frameon=True, fontsize=8,
                      handlelength=1.5, edgecolor="#e4e3df")
            ax.text(0.5, -0.08, f"centre of mass |g| = {abs(g):.3f}", transform=ax.transAxes,
                    ha="center", fontsize=8, color="#52514e")
    fig.suptitle(f"{tr['name']}: signals wound around a circle - top: {FFT_WINDING_OFF:g} x period "
                 f"(misaligned), bottom: dominant period {period:.2f} s (loops align, dot moves out)",
                 fontweight="bold", fontsize=11, y=1.0)
    fig.subplots_adjust(top=0.9, bottom=0.06)
    png = os.path.join(out_dir, f"{tr['name']}_SEC_MTS_FFT_winding.png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"Saved: {png}")
    plt.show()
    plt.close(fig)


def plot_fft_components(tr, df, cdf, period, out_dir):
    """Each signal over FFT_COMPONENT_CYCLES cycles with its FFT_N_COMPONENTS strongest Fourier
    sinusoids and their sum (as in 'square wave and its constituent sinusoids')."""
    series = fft_series(tr, df, cdf)
    fig, axes = plt.subplots(len(series), 1, figsize=(12, 2.9 * len(series)), squeeze=False)
    rows = []
    for ax, (name, t, y, fs, colr) in zip(axes[:, 0], series):
        y = np.asarray(y, float)
        good = np.isfinite(y)
        y = np.interp(np.arange(len(y)), np.flatnonzero(good), y[good])
        n_per = int(round(period * fs))
        n = (len(y) // n_per) * n_per or len(y)                 # whole cycles only: no leakage
        t, y = t[:n] - t[0], y[:n] - y[:n].mean()
        c = np.fft.rfft(y) / n
        k = np.argsort(np.abs(c[1:]))[::-1][:FFT_N_COMPONENTS] + 1
        show = t <= FFT_COMPONENT_CYCLES * period
        recon = np.zeros(n)
        ax.plot(t[show], y[show], color=colr, lw=3.2, label=f"{name} (mean removed)", zorder=3)
        for kk in sorted(k):
            comp = 2 * np.abs(c[kk]) * np.cos(2 * np.pi * kk * np.arange(n) / n + np.angle(c[kk]))
            recon += comp
            fk = kk * fs / n
            h = fk * period                                     # harmonic number of the dominant period
            harmonic = abs(h - round(h)) < 0.05 and round(h) >= 1
            colored = harmonic and round(h) <= len(COMPONENT_COLORS)
            ax.plot(t[show], comp[show], lw=1.6,                # colour = harmonic, same in every panel
                    color=COMPONENT_COLORS[round(h) - 1] if colored else "#b8b7b2",
                    label=f"{'harmonic ' + str(round(h)) if harmonic else 'non-harmonic'}: "
                          f"T = {1 / fk:.1f} s, {fk:.4f} Hz, amp {2 * np.abs(c[kk]):.3g}")
            rows.append({"Trial": tr["name"], "Signal": name, "Freq_Hz": fk, "Period_s": 1 / fk,
                         "Harmonic_of_dominant": fk * period, "Amplitude": 2 * np.abs(c[kk])})
        expl = 100 * (1 - np.var(y - recon) / np.var(y))
        ax.plot(t[show], recon[show], color="#0b0b0b", lw=1.2, ls="--",
                label=f"sum of {len(k)} sinusoids ({expl:.0f}% of variance)", zorder=4)
        ax.axhline(0, color="#52514e", lw=0.6)
        for m in range(1, FFT_COMPONENT_CYCLES):
            ax.axvline(m * period, color="#b8b7b2", lw=0.8)
        ax.set_ylabel("Amplitude", fontsize=9)
        ax.grid(color="#e4e3df", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=7.5)
        ax.set_title(f"{name} and its constituent sinusoids", fontsize=10, fontweight="bold", loc="left")
    axes[-1, 0].set_xlabel(f"Time (s) - first {FFT_COMPONENT_CYCLES} cycles of the dominant period "
                           f"({period:.2f} s)")
    fig.suptitle(f"{tr['name']}: Fourier decomposition - {FFT_N_COMPONENTS} strongest components",
                 fontweight="bold")
    fig.tight_layout()
    png = os.path.join(out_dir, f"{tr['name']}_SEC_MTS_FFT_components.png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"Saved: {png}")
    plt.show()
    plt.close(fig)
    return pd.DataFrame(rows)


def dominant_period(trials):
    """Dominant FFT period of the SEC signal: median over the reliably aligned (periodic) trials."""
    use = [tr for tr in trials if tr["offset_ok"]] or trials
    rows = []
    for tr in use:
        f, amp = fft_spectrum(tr["x"], tr["fs"])
        peaks = fft_peaks(f, amp, tr["t"][-1] - tr["t"][0])
        ok = bool(peaks) and peaks[0][2] >= FFT_MIN_POWER_PCT
        rows.append({"Trial": tr["name"], "Period_s": 1 / peaks[0][0] if ok else np.nan,
                     "Peak_power_pct": peaks[0][2] if peaks else np.nan})
    tab = pd.DataFrame(rows)
    print("\nDominant SEC period per trial (FFT):")
    print(tab.round(2).to_string(index=False))
    if tab["Period_s"].notna().sum() == 0:
        raise RuntimeError("No trial has a dominant periodic FFT peak - use MODE = 'fixed' or 'auto'.")
    return float(tab["Period_s"].median()), tab


def window_scan(trials, period, out_dir):
    """FastTDA quality and target correlation for windows of FFT_WINDOW_SCAN x the dominant period."""
    use = [tr for tr in trials if tr["offset_ok"]] or trials
    fs = use[0]["fs"]
    rows = []
    for m in FFT_WINDOW_SCAN:
        w = int(round(m * period * fs))
        d = max(1, int(round(min(FFT_DELAY_FRACTION * period * fs, MAX_DELAY_FRACTION * w))))
        if any(w > len(tr["x"]) for tr in use):
            continue
        per = pd.DataFrame([score_setting(tr, w, d, False) for tr in use])
        row = {"Cycles": m, "Window_s": w / fs, "Delay_s": d / fs, "Window_Size": w, "Takens_Delay": d,
               "Invalid_pct": per["Invalid_pct"].mean(), "Roughness": per["Roughness"].mean()}
        for k in target_keys():
            row[f"|rho| {k}"] = per[f"rho_{k}"].abs().mean()
        for c in TARGET_COLS:                 # how much of the target's variation survives the averaging
            row[f"{c}:mean variation left (%)"] = 100 * np.mean(
                [window_stats(tr["targets"][c], w, TUNE_STEP)[0].std() / tr["targets"][c].std() for tr in use])
        rows.append(row)
    scan = pd.DataFrame(rows)
    print(f"\nWindow scan around the dominant period ({period:.1f} s), mean over "
          f"{', '.join(tr['name'] for tr in use)}:")
    print(scan.round(3).to_string(index=False))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    for k, colr in zip([f"{c}:mean" for c in TARGET_COLS], ["#6250d6", "#e87ba4"]):
        ax1.plot(scan["Window_s"], scan[f"|rho| {k}"], "o-", color=colr, label=f"|Spearman| FastTDA vs {k}")
    ax1.plot(scan["Window_s"], scan[f"{TARGET_COLS[0]}:mean variation left (%)"] / 100, ":",
             color="#52514e", lw=1.5, label=f"{TARGET_COLS[0]}:mean variation left (fraction)")
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("|Spearman|  /  fraction")
    ax2.plot(scan["Window_s"], scan["Invalid_pct"], "o-", color="#eb6834", label="failed FastTDA fits (%)")
    ax2.set_ylabel("Failed fits (%)")
    for ax in (ax1, ax2):
        for m in range(1, int(max(FFT_WINDOW_SCAN)) + 1):
            ax.axvline(m * period, color="#c0392b", lw=1, ls="--")
        ax.axvline(FFT_WINDOW_CYCLES * period, color="#c0392b", lw=2.5, alpha=0.4,
                   label=f"chosen: {FFT_WINDOW_CYCLES:g} x {period:.1f} s")
        ax.set_xlabel("Window length (s)   - dashed = multiples of the dominant period")
        ax.grid(color="#e4e3df", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"Window length vs dominant FFT period ({period:.1f} s = {1 / period:.4f} Hz)",
                 fontweight="bold")
    fig.tight_layout()
    png = os.path.join(out_dir, "SEC_MTS_window_vs_period.png")
    fig.savefig(png, dpi=150)
    print(f"Saved: {png}")
    plt.show()
    plt.close(fig)
    return scan


# ============================================================
# 4) FASTTDA VALIDATION: short non-overlapping windows, permutation test, partial correlation,
#    leave-one-trial-out models with and without FastTDA and signal direction
# ============================================================

VAL_MODELS = {
    "Capacitance": ["Cap_mean", "Cap_std"],
    "Capacitance + FastTDA": ["Cap_mean", "Cap_std", "FastTDA_ZeroInvalid", "FastTDA_FitOK"],
    "Capacitance + direction": ["Cap_mean", "Cap_std", "Cap_slope"],
    "Capacitance + direction + FastTDA": ["Cap_mean", "Cap_std", "Cap_slope",
                                          "FastTDA_ZeroInvalid", "FastTDA_FitOK"],
}
BRANCH_MODELS = {
    "FastTDA only": ["FastTDA_ZeroInvalid", "FastTDA_FitOK"],
    "Capacitance": ["Cap_mean", "Cap_std"],
    "Capacitance + direction": ["Cap_mean", "Cap_std", "Cap_slope"],
    "Capacitance + direction + FastTDA": VAL_MODELS["Capacitance + direction + FastTDA"],
}
MODEL_COLORS = ["#b8b7b2", "#eb6834", "#6da7ec", "#0d366b"]


def rank(a):
    return pd.Series(a).rank().to_numpy(float)


def partial_spearman(x, y, controls):
    """Spearman correlation of x and y after removing the (rank-linear) effect of the controls."""
    Z = np.column_stack([np.ones(len(x))] + [rank(c) for c in controls])
    rx, ry = rank(x), rank(y)
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1])


def circular_perm_p(groups, n_perm, seed=0):
    """p-value of the pooled |Spearman| when the target of each trial is circularly shifted by a
    random amount: keeps the autocorrelation of both series, breaks their alignment."""
    groups = [(x, y) for x, y in groups if len(x) >= 3]
    rng = np.random.default_rng(seed)
    obs = abs(spearman(np.concatenate([g[0] for g in groups]), np.concatenate([g[1] for g in groups])))
    hits = 0
    for _ in range(n_perm):
        ys = [np.roll(y, rng.integers(max(1, len(y) // 10), max(2, len(y) - len(y) // 10)))
              for _, y in groups]
        hits += abs(spearman(np.concatenate([g[0] for g in groups]), np.concatenate(ys))) >= obs
    return (hits + 1) / (n_perm + 1)


def loto_fit_predict(frames, target, cols):
    """Leave-one-trial-out linear model: fit on the other trials, predict the left-out one."""
    pred, true = [], []
    for i, test in enumerate(frames):
        train = pd.concat([f for j, f in enumerate(frames) if j != i])
        X = np.column_stack([np.ones(len(train))] + [train[c] for c in cols])
        beta = np.linalg.lstsq(X, train[target].to_numpy(float), rcond=None)[0]
        Xt = np.column_stack([np.ones(len(test))] + [test[c] for c in cols])
        pred.append(Xt @ beta)
        true.append(test[target].to_numpy(float))
    return np.concatenate(pred), np.concatenate(true)


def loto_r2(frames, target, cols):
    p, t = loto_fit_predict(frames, target, cols)
    return 1 - np.sum((t - p) ** 2) / np.sum((t - t.mean()) ** 2)


def loto_accuracy(frames, cols):
    p, t = loto_fit_predict(frames, "Loading", cols)
    return float(np.mean((p > 0.5) == (t > 0.5)))


def short_window_table(tr, window, delay):
    """Non-overlapping windows: FastTDA, capacitance level / spread / slope, targets, loading flag."""
    starts, clouds = point_clouds(tr["x"], window, delay, False, step=window)
    ftda = np.array([fasttda_minor_axis(pc)[0] for pc in clouds])
    idx = np.arange(window) - (window - 1) / 2
    slope = lambda W: (W - W.mean(axis=1, keepdims=True)) @ idx / (idx @ idx) * tr["fs"]   # per s
    X = sliding_window_view(tr["x"], window)[::window]
    D = sliding_window_view(tr["targets"]["Displacement_m"], window)[::window]
    cap_mean, cap_std = window_stats(tr["x"], window, window)
    out = {"Trial": tr["name"], "Time": tr["t"][starts + window // 2], "FastTDA": ftda,
           "FastTDA_ZeroInvalid": np.where(np.isfinite(ftda), ftda, 0.0),
           "FastTDA_FitOK": np.isfinite(ftda).astype(float),
           "Cap_mean": cap_mean - np.median(cap_mean),    # zero per trial: the baseline differs between
           "Cap_std": cap_std,                            # trials (trial_04 sits ~0.15 pF higher)
           "Cap_slope": slope(X),                         # direction: capacitance rising or falling
           "Loading": (slope(D) < 0).astype(float)}       # displacement becoming more negative
    for c in TARGET_COLS:
        out[f"{c}:mean"], out[f"{c}:std"] = window_stats(tr["targets"][c], window, window)
    f = pd.DataFrame(out)
    return f[f["Time"] >= VAL_SKIP_START_S].reset_index(drop=True)


def validate(trials, out_dir):
    use = [tr for tr in trials if tr["offset_ok"]] if VAL_RELIABLE_ONLY else trials
    if len(use) < 2:
        print("\nValidation skipped: needs at least 2 trials with a reliable alignment.")
        return None
    fs = use[0]["fs"]
    print(f"\n=== FASTTDA VALIDATION on {', '.join(tr['name'] for tr in use)} "
          f"(non-overlapping windows, first {VAL_SKIP_START_S:g} s dropped) ===")
    r2_rows, feat_rows, branch_rows, all_frames = [], [], [], []
    for ws in VAL_WINDOWS_S:
        w = int(round(ws * fs))
        d = max(1, int(round(VAL_DELAY_FRACTION * w)))
        frames = [short_window_table(tr, w, d) for tr in use]
        pooled = pd.concat(frames, ignore_index=True)
        all_frames.append(pooled.assign(Window_s=ws))
        valid = pooled["FastTDA"].notna()
        for tgt in VAL_TARGETS:
            r2 = {m: loto_r2(frames, tgt, cols) for m, cols in VAL_MODELS.items()}
            for m, v in r2.items():
                r2_rows.append({"Window_s": ws, "Target": tgt, "Model": m, "LOTO_R2": v})
            groups = [(f.loc[f["FastTDA"].notna(), "FastTDA"].to_numpy(float),
                       f.loc[f["FastTDA"].notna(), tgt].to_numpy(float)) for f in frames]
            feat_rows.append({
                "Window_s": ws, "Delay_s": d / fs, "Target": tgt, "N_windows": len(pooled),
                "FastTDA_valid_pct": 100 * valid.mean(),
                "Spearman_FastTDA": spearman(pooled.loc[valid, "FastTDA"].to_numpy(float),
                                             pooled.loc[valid, tgt].to_numpy(float)),
                "Spearman_Capacitance": spearman(pooled["Cap_mean"].to_numpy(float),
                                                 pooled[tgt].to_numpy(float)),
                "Perm_p_FastTDA": circular_perm_p(groups, VAL_PERMUTATIONS),
                "Partial_rho_FastTDA_given_Cap": partial_spearman(
                    pooled.loc[valid, "FastTDA"].to_numpy(float), pooled.loc[valid, tgt].to_numpy(float),
                    [pooled.loc[valid, c].to_numpy(float) for c in ("Cap_mean", "Cap_std", "Cap_slope")]),
                "R2_gain_FastTDA": r2["Capacitance + FastTDA"] - r2["Capacitance"],
                "R2_gain_FastTDA_with_direction": (r2["Capacitance + direction + FastTDA"]
                                                   - r2["Capacitance + direction"])})
        row = {"Window_s": ws, "N_windows": len(pooled), "Loading_share": pooled["Loading"].mean()}
        for m, cols in BRANCH_MODELS.items():
            row[f"Accuracy: {m}"] = loto_accuracy(frames, cols)
        branch_rows.append(row)
        print(f"  window {ws:4.1f} s (delay {d / fs:.2f} s): {len(pooled)} windows, "
              f"FastTDA valid {100 * valid.mean():.0f}%")
    r2_tab = pd.DataFrame(r2_rows).pivot_table(index=["Target", "Window_s"], columns="Model",
                                               values="LOTO_R2", sort=False)[list(VAL_MODELS)]
    feat = pd.DataFrame(feat_rows)
    branch = pd.DataFrame(branch_rows)
    print("\nLeave-one-trial-out R2 (fit on the other trials, tested on the left-out one):")
    print(r2_tab.round(3).to_string())
    print("\nFastTDA statistics:")
    print(feat.round(3).to_string(index=False))
    print("\nLoading vs unloading, leave-one-trial-out accuracy (Loading_share = accuracy of always guessing 'loading'):")
    print(branch.round(3).to_string(index=False))
    plot_validation(r2_tab, branch, out_dir)
    return {"Validation_R2": r2_tab.reset_index(), "Validation_FastTDA": feat,
            "Validation_Branch": branch, "Validation_Windows": pd.concat(all_frames, ignore_index=True)}


def plot_validation(r2_tab, branch, out_dir):
    fig, axes = plt.subplots(1, len(VAL_TARGETS) + 1, figsize=(5.2 * (len(VAL_TARGETS) + 1), 4.2))
    width = 0.8 / len(VAL_MODELS)
    for ax, tgt in zip(axes, VAL_TARGETS):
        t = r2_tab.loc[tgt]
        x = np.arange(len(t))
        for k, (m, c) in enumerate(zip(VAL_MODELS, MODEL_COLORS)):
            ax.bar(x + (k - (len(VAL_MODELS) - 1) / 2) * width, t[m], width, color=c, label=m)
        ax.set_xticks(x, [f"{w:g} s" for w in t.index])
        ax.set_ylim(min(0, t.min().min()), 1)
        ax.set_title(f"{tgt}: leave-one-trial-out R2", fontsize=10, fontweight="bold")
    ax = axes[-1]
    x = np.arange(len(branch))
    for k, (m, c) in enumerate(zip(BRANCH_MODELS, ["#eb6834", "#b8b7b2", "#6da7ec", "#0d366b"])):
        ax.bar(x + (k - 1.5) * 0.2, branch[f"Accuracy: {m}"], 0.2, color=c, label=m)
    ax.plot(x, branch["Loading_share"], "_", color="#c0392b", ms=40, mew=2, label="always guess 'loading'")
    ax.set_xticks(x, [f"{w:g} s" for w in branch["Window_s"]])
    ax.set_ylim(0, 1)
    ax.set_title("Loading vs unloading: accuracy", fontsize=10, fontweight="bold")
    for ax in axes:
        ax.set_xlabel("Window length")
        ax.grid(axis="y", color="#e4e3df", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.suptitle("FastTDA validation: non-overlapping windows, leave-one-trial-out", fontweight="bold")
    fig.tight_layout()
    png = os.path.join(out_dir, "SEC_MTS_FastTDA_validation.png")
    fig.savefig(png, dpi=150)
    print(f"Saved: {png}")
    plt.show()
    plt.close(fig)


# ============================================================
# RUN
# ============================================================

def save_workbook(path, sheets):
    def write(p):
        with pd.ExcelWriter(p, engine="openpyxl") as wr:
            for sheet, frame in sheets.items():
                if frame is not None:
                    frame.to_excel(wr, sheet_name=sheet[:31], index=False)
    try:
        write(path)
    except PermissionError:
        path = path.replace(".xlsx", f"_{time.strftime('%H%M%S')}.xlsx")
        print("Workbook is open in Excel; saving under a new name.")
        write(path)
    print(f"Saved: {path}")
    return path


def process_trial(tr, window, delay, detrend, best_key, summary, note, out_dir):
    print(f"\n=== {tr['name']}: {profile_name(window, delay, detrend)} "
          f"(window {window / tr['fs']:.2f} s, delay {delay / tr['fs']:.3f} s)")
    df, tg, t_end = extract_profile(tr, window, delay, detrend)
    n_bad = int((~df["FastTDA_FitOK"]).sum())
    cdf = clean_profile(df, t_end, tr["fs"])
    meta = {"Profile": f"W{window}", "Dataset": tr["name"], "Sampling_Hz": round(tr["fs"], 4),
            "Lowpass_Hz": None, "Filter_Order": None, "Window_Size": window, "Window_Step": WINDOW_STEP,
            "Takens_Delay": delay, "Takens_Dimension": TAKENS_DIM, "Takens_Stride": embed_stride(window, delay),
            "Time_Label": TIME_LABEL, "Rows": len(df), "FastTDA_Invalid_Count": n_bad, "Note": note,
            "Detrend": detrend, "SEC_MTS_Offset_s": tr["offset"], "Offset_Corr": tr["offset_corr"],
            "Target": best_key}
    stats = {"Trial": tr["name"], "Setup": "updated" if tr["num"] >= TUNE_FROM_TRIAL else "previous",
             "Profile": profile_name(window, delay, detrend), "Offset_s": round(tr["offset"], 3),
             "Offset_Corr": tr["offset_corr"], "Alignment_OK": tr["offset_ok"], "Windows": len(df),
             "Invalid_pct": round(100 * n_bad / len(df), 3)}
    rhos = {}
    print(f"    {len(df)} windows, {n_bad} invalid FastTDA fits ({100 * n_bad / len(df):.2f}%)")
    print("    Spearman correlation of each smoothed feature with each MTS target (per window):")
    table = []
    for col in CLEAN_FEATURES:
        if f"{col}_smooth" not in cdf:
            continue
        row = {"Feature": col}
        for k in target_keys():
            rho = spearman(cdf[f"{col}_smooth"].to_numpy(float), tg[k].to_numpy(float))
            row[k] = rho
            stats[f"{col}~{k}"] = rho
        rhos[col] = row[best_key]
        table.append(row)
    base = {"Feature": f"SEC {SEC_COL} (no TDA, window mean)"}
    cap_mean, _ = window_stats(tr["x"], window, WINDOW_STEP)
    for k in target_keys():
        base[k] = spearman(cap_mean, tg[k].to_numpy(float))
        stats[f"Capacitance~{k}"] = base[k]
    table.append(base)
    print(pd.DataFrame(table).round(3).to_string(index=False))
    save_workbook(os.path.join(out_dir, f"{tr['name']}_SEC_MTS_TDA.xlsx"), {
        f"W{window}": df, "Targets": tg, f"W{window}_Cleaned": cdf,
        "Metadata": pd.DataFrame([meta]), "Tuning_Summary": summary})
    plot_profile(tr, df, cdf, meta, best_key, rhos, out_dir)
    plot_scatter(tr, cdf, tg, best_key, out_dir)
    if PLOT_ELLIPSES:
        plot_ellipses(tr, df, tg, window, delay, detrend, best_key, out_dir)
    if FFT_ANALYSIS:
        fft_out = fft_analysis(tr, df, cdf, out_dir)
        stats.update(fft_out)
        period = fft_out.get(f"{SEC_COL}_period_s", np.nan)
        if not np.isfinite(period):
            print("    No dominant SEC period - winding / component plots skipped for this trial")
        else:
            if FFT_WINDING:
                plot_fft_winding(tr, df, cdf, period, out_dir)
            if FFT_COMPONENTS:
                comps = plot_fft_components(tr, df, cdf, period, out_dir)
                save_workbook(os.path.join(out_dir, f"{tr['name']}_SEC_MTS_FFT_components.xlsx"),
                              {"Components": comps})
    return stats


out_dir = to_local_path(OUTPUT_DIR) if OUTPUT_DIR else os.path.join(to_local_path(DATA_DIR), "tda_results")
os.makedirs(out_dir, exist_ok=True)
t_start = time.time()
print("Loading and aligning:")
trials = [load_pair(*p) for p in find_pairs()]
tune_trials = [tr for tr in trials if tr["num"] >= TUNE_FROM_TRIAL] or trials

summary = None
if MODE == "fft":
    period, period_tab = dominant_period(trials)
    fs0 = trials[0]["fs"]
    w = int(round(FFT_WINDOW_CYCLES * period * fs0))
    d = max(1, int(round(FFT_DELAY_FRACTION * period * fs0)))
    dt = False
    best_key = TUNE_TARGET if TUNE_TARGET != "auto" else f"{TARGET_COLS[0]}:mean"
    scan = window_scan(trials, period, out_dir)
    summary = scan
    note = (f"window = {FFT_WINDOW_CYCLES:g} x dominant FFT period {period:.2f} s, "
            f"delay = {FFT_DELAY_FRACTION:g} x period")
    print(f"\nDominant period {period:.2f} s -> window {w} samples ({w / fs0:.1f} s), "
          f"delay {d} samples ({d / fs0:.2f} s)")
elif MODE == "fixed":
    w, d, dt = int(FIXED_WINDOW), int(FIXED_DELAY), bool(FIXED_DETREND)
    best_key = TUNE_TARGET if TUNE_TARGET != "auto" else f"{TARGET_COLS[0]}:mean"
    note = "fixed setting"
else:
    summary = fine_tune(tune_trials)
    print("\nTop 10 settings:")
    cols = ["Rank", "Profile", "Stage", "Window_s", "Delay_s", "Detrend", "Invalid_pct",
            "Worst_Invalid_pct", "Roughness", "Target", "Target_absrho", "Score"]
    print(summary[cols].head(10).round(4).to_string(index=False))
    best = summary.iloc[0]
    w, d, dt, best_key = int(best["Window_Size"]), int(best["Takens_Delay"]), bool(best["Detrend"]), best["Target"]
    note = f"tuned on {len(tune_trials)} trials (trial_{TUNE_FROM_TRIAL:02d}+), Rank 1 of {len(summary)}"
    plot_tuning(summary, out_dir)

results = [process_trial(tr, w, d, dt, best_key, summary, note, out_dir) for tr in trials]
overview = pd.DataFrame(results)
print(f"\n=== All trials ({time.time() - t_start:.0f} s) - setting {profile_name(w, d, dt)}, "
      f"target {best_key} ===")
print(overview.round(3).to_string(index=False))
validation = validate(trials, out_dir) if VALIDATE else None
save_workbook(os.path.join(out_dir, "All_Trials_SEC_MTS_Summary.xlsx"),
              {"Trials": overview, "Tuning_Summary": summary, **(validation or {})})
