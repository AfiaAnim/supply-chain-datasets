"""
DROPBEAR_Code2_FeatureExtraction_Parity.py

Diagnostic Code 2 for DROPBEAR feature-extraction parity.

Why this exists
---------------
Code 1 restored the legacy metric logic but VR/FastTDA did not fully return to
old manuscript values. Therefore this script isolates extraction differences.

It intentionally DOES NOT choose a final profile yet. Instead, it exports the
legacy-relevant extraction profiles side-by-side so Code 3 can apply the same
raw-feature metric logic to every candidate and identify the actual parity
match empirically.

Profiles generated
------------------
Dataset 6 (Tests 1 and 2):
    W1500  : old VR Test-9 window convention
    W1612  : old FastTDA / formula-based single-loop convention

    Common DB6 settings:
        fs=25000 Hz, low-pass=100 Hz, order=5,
        window step=25 samples,
        Takens delay=100 samples,
        dimension=2,
        embedding stride=25,
        midpoint time/cart label.

Dataset 8 (Test 3):
    W362_S1_M31p1 : old VR profile (max_f=31.1 Hz, int-truncated window)
    W363_S1_M31p0 : old FastTDA profile (max_f=31.0 Hz)

    Common DB8 settings:
        fs=5000 Hz, low-pass=100 Hz, order=6 SOS,
        window step=5 samples,
        Takens delay=20 samples,
        dimension=2,
        embedding stride=1,
        endpoint time/cart label.

Features exported for every profile
-----------------------------------
    VR
        Maximum H1 persistence from Vietoris-Rips.

    Alpha
        Maximum H1 persistence from GUDHI AlphaComplex, converted from
        squared-radius filtration values using 2*sqrt(alpha).

    FastTDA
        SVD-LSE ellipse full minor-axis length (2b).

    FastTDA_ZeroInvalid
        Same FastTDA values, but invalid ellipse fits are filled with 0.0.
        The main FastTDA column uses NaN for invalid fits. This lets the next
        diagnostic determine whether the old DB6 'skip invalid window' behavior
        materially explains the mismatch.

Fine-tuning
-----------
Window size and Takens delay are swept over the grids DB6_WINDOW_SIZES x
DB6_DELAYS and DB8_WINDOW_SIZES x DB8_DELAYS. Every combination becomes one
sheet named W<window>_T<delay>; the legacy profiles above are included in the
default grids as baselines (marked Legacy=True in the Metadata sheet). The
Metadata sheet also lists per-profile feature statistics and the correlation
of each feature with the cart/pin location for quick comparison.

CSV trial (FastTDA analysis)
----------------------------
Any single trial saved as .csv (e.g. trial_01.csv) can be run through the same
window x delay fine-tuning grid. The loader detects the delimiter, the header,
the time / acceleration / position columns and the sampling rate, and prints a
column summary so the choice can be checked; each can also be set by hand in
the CSV configuration block. Window, delay and step are given in milliseconds,
so the grid means the same thing at any sampling rate. FastTDA is always
computed; VR and Alpha are optional (CSV_COMPUTE_VR / CSV_COMPUTE_ALPHA).

IMPORTANT
---------
This code only extracts features. It does not clip/normalize features for
metrics and it does not calculate J1/J2/J3. Run Code 3 (legacy metrics) on the
candidate sheets after this script finishes.

Requirements:
    pip install numpy pandas scipy scikit-learn giotto-tda gudhi openpyxl
"""

from __future__ import annotations

import os
import time
import numpy as np
import pandas as pd

from scipy import signal
from sklearn.preprocessing import MinMaxScaler
from gtda.time_series import SingleTakensEmbedding, TakensEmbedding
from gtda.homology import VietorisRipsPersistence


# ============================================================
# USER CONFIGURATION
# ============================================================

# Folder that holds the input .txt files (WSL path to C:\Users\aphya\Downloads\...).
# Output Excel files are written to the same folder.
WORK_DIR = "/mnt/c/Users/aphya/Downloads/TDA_Feature_Extraction_Code_and_Study_Cases"

# Input files that are missing from WORK_DIR are skipped with a message.
DB6_INPUTS = {
    "Test1_DROPBEAR_6.9": "test99.txt",
    "Test2_DROPBEAR_6.11": "test1111.txt",
}

DB8_INPUT = {
    "Test3_DROPBEAR_8.random": "test1.txt",
}

# Dataset 6 legacy/common settings
DB6_FS = 25000
DB6_CUTOFF = 100
DB6_FILTER_ORDER = 5
DB6_TIME_LIMIT = 6.0
DB6_WINDOW_STEP = 25
DB6_DIM = 2
DB6_EMBED_STRIDE = 25

# --- DB6 fine-tuning grid (samples at 25000 Hz; 25 samples = 1 ms) ---
# Every window size is run with every delay.
# Legacy: windows 1500 (old VR) and 1612 (old FastTDA), delay 100.
DB6_WINDOW_SIZES = [1250, 1500, 1612, 1750]
DB6_DELAYS = [75, 100, 125]
DB6_LEGACY_WINDOWS = (1500, 1612)
DB6_LEGACY_DELAY = 100

# Dataset 8 legacy/common settings
DB8_FS = 5000
DB8_CUTOFF = 100
DB8_FILTER_ORDER = 6
DB8_MIN_F = 17.7
DB8_WINDOW_STEP = 5
DB8_DIM = 2
DB8_EMBED_STRIDE = 1

# --- DB8 fine-tuning grid (samples at 5000 Hz; 5 samples = 1 ms) ---
# Every window size is run with every delay.
# Legacy (from the formulas in db8_window_and_tau):
#   max_f=31.1 -> window 362, delay 20 (old VR profile)
#   max_f=31.0 -> window 363, delay 20 (old FastTDA profile)
DB8_WINDOW_SIZES = [322, 362, 402]
DB8_DELAYS = [15, 20, 25]
DB8_LEGACY_MAX_F = (31.1, 31.0)

# ============================================================
# WHAT TO RUN
# ============================================================

RUN_DB6 = False
RUN_DB8 = False
RUN_CSV = True

# ============================================================
# CSV TRIAL CONFIGURATION (FastTDA analysis)
# ============================================================

# Windows paths (C:\...) are converted to WSL paths (/mnt/c/...) automatically.
CSV_PATH = r"C:\Users\aphya\Downloads\trial_01.csv"
CSV_OUTPUT_DIR = None        # None = save next to the CSV

# Columns: a header name or a 0-based index.
# trial_01.csv columns: time_s, capacitance_raw_pF, capacitance_zero_mean_pF,
#                       capacitance_filtered_pF, capacitance_filtered_zero_mean_pF
# (a leading unnamed row-index column, if present, is ignored).
CSV_TIME_COL = "time_s"
CSV_SIGNAL_COL = "capacitance_filtered_pF"   # signal analysed with FastTDA
CSV_POS_COL = None                           # position/label column, or None if there is none

CSV_FS = None                # sampling rate in Hz; None = estimate from the time column (~88.5 Hz)
CSV_CUTOFF = None            # extra low-pass cutoff (Hz); None = no extra filtering.
                             # Must be below half the sampling rate (~44 Hz). Use e.g. 5 with
                             # CSV_SIGNAL_COL = "capacitance_raw_pF" to filter the raw signal here.
CSV_FILTER_ORDER = 6         # zero-phase Butterworth (SOS), as in DB8
CSV_DIM = 2

# Fine-tuning grid in SECONDS (converted to samples at the sampling rate).
# Every window length is run with every delay. The script prints the dominant
# frequency of the signal and the DB8-rule suggestion to guide these values:
#   window ~ 1/min_f + 2*(0.25/max_f),  delay ~ (0.25/max_f)/2
CSV_WINDOW_S = [1.0, 1.5, 2.0]
CSV_DELAY_S = [0.03, 0.06, 0.09]
CSV_STEP_SAMPLES = 1         # window step in samples (1 sample ~ 11 ms here)
CSV_EMBED_STRIDE = 1         # samples between embedded points

CSV_COMPUTE_VR = False       # VR max H1 (slower)
CSV_COMPUTE_ALPHA = False    # Alpha max H1 (slowest; needs gudhi)
PROGRESS_EVERY_CSV = 2000

# Alpha can be much slower than VR/FastTDA for thousands of windows.
# Keep True for the requested 3-method parity extraction.
COMPUTE_ALPHA = True

# Console progress cadence
PROGRESS_EVERY_DB6 = 500
PROGRESS_EVERY_DB8 = 500


# ============================================================
# BASIC HELPERS
# ============================================================

def create_windows(array: np.ndarray, window_size: int, step: int) -> np.ndarray:
    array = np.asarray(array)
    return np.array([
        array[i:i + window_size]
        for i in range(0, len(array) - window_size + 1, step)
    ])


def minmax_minus1_plus1(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    return MinMaxScaler(feature_range=(-1, 1)).fit_transform(x).ravel()


def minmax_to_50_200(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    lo = float(np.nanmin(x))
    hi = float(np.nanmax(x))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return np.full_like(x, 50.0, dtype=float)
    return 50.0 + (x - lo) * 150.0 / (hi - lo)


def feature_summary(df: pd.DataFrame, label_col) -> dict:
    """Per-profile statistics used to compare fine-tuning combinations."""
    out = {}
    for col in ("VR", "Alpha", "FastTDA"):
        x = df[col].astype(float)
        out[f"{col}_mean"] = float(np.nanmean(x)) if x.notna().any() else np.nan
        out[f"{col}_std"] = float(np.nanstd(x)) if x.notna().any() else np.nan
        if label_col is None or label_col not in df:
            continue
        ok = x.notna() & df[label_col].notna()
        if ok.sum() > 2 and x[ok].std() > 0 and df.loc[ok, label_col].std() > 0:
            out[f"Corr_{col}_{label_col}"] = float(np.corrcoef(x[ok], df.loc[ok, label_col])[0, 1])
        else:
            out[f"Corr_{col}_{label_col}"] = np.nan
    return out


# ============================================================
# FAST TDA: SVD-LSE ELLIPSE FIT
# ============================================================

def fit_ellipse_AplusC_eq_1(X, Y, rcond=None):
    """
    Fit conic with scale constraint A + C = 1:

        A(x^2-y^2) + Bxy + Dx + Ey + F = -y^2,
        C = 1-A.

    Center/scale internally for numerical conditioning, matching the later
    stabilized SVD-LSE implementation.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    if X.size < 5 or Y.size < 5:
        return None

    Xc = X - np.mean(X)
    Yc = Y - np.mean(Y)

    max_range = max(np.max(np.abs(Xc)), np.max(np.abs(Yc)))
    if not np.isfinite(max_range) or max_range == 0:
        return None

    scale = 1.5 / max_range
    Xc = Xc * scale
    Yc = Yc * scale

    G = np.column_stack([
        Xc**2 - Yc**2,
        Xc * Yc,
        Xc,
        Yc,
        np.ones_like(Xc),
    ])
    rhs = -(Yc**2)

    try:
        U, S, Vt = np.linalg.svd(G, full_matrices=False)
    except np.linalg.LinAlgError:
        return None

    if rcond is None:
        rcond = np.finfo(float).eps * max(G.shape)

    smax = S[0] if S.size else 1.0
    Sinv = np.where(S > rcond * smax, 1.0 / S, 0.0)
    u = (Vt.T * Sinv) @ (U.T @ rhs)

    A, B, D, E, F = u
    C = 1.0 - A

    if 4.0 * A * C - B**2 <= 0.0:
        return None

    return A, B, C, D, E, F


def rotate_and_axes(A, B, C, D, E, F):
    theta = 0.5 * np.arctan2(B, A - C)
    c = np.cos(theta)
    s = np.sin(theta)

    Ap = A * c*c + B * c*s + C * s*s
    Cp = A * s*s - B * c*s + C * c*c
    Dp = D * c + E * s
    Ep = -D * s + E * c

    if Ap <= 0.0 or Cp <= 0.0:
        return None

    h = -Dp / (2.0 * Ap)
    k = -Ep / (2.0 * Cp)
    Fnew = F - (Dp**2) / (4.0 * Ap) - (Ep**2) / (4.0 * Cp)

    if Fnew >= 0.0:
        return None

    a = np.sqrt(-Fnew / Ap)
    b = np.sqrt(-Fnew / Cp)

    if not np.isfinite(a) or not np.isfinite(b) or a <= 0 or b <= 0:
        return None

    if a < b:
        a, b = b, a
        theta += np.pi / 2.0

    return theta, h, k, a, b


def compute_fasttda_minor_axis(point_cloud: np.ndarray) -> tuple[float, bool]:
    """Return (2b, valid_fit). Invalid fits return (NaN, False)."""
    pc = np.asarray(point_cloud, dtype=float)
    if pc.ndim != 2 or pc.shape[0] < 5 or pc.shape[1] < 2:
        return np.nan, False

    X, Y = pc[:, 0], pc[:, 1]
    params = fit_ellipse_AplusC_eq_1(X, Y)
    if params is None:
        return np.nan, False

    pose = rotate_and_axes(*params)
    if pose is None:
        return np.nan, False

    _, _, _, _, b = pose
    return float(2.0 * b), True


# ============================================================
# VR / ALPHA FEATURES
# ============================================================

def compute_vr_max_h1(point_cloud: np.ndarray, vr_model) -> float:
    pc = np.asarray(point_cloud, dtype=float)
    try:
        diagrams = vr_model.fit_transform(pc[None, :, :])
        D = np.asarray(diagrams[0], dtype=float)
    except Exception:
        return np.nan

    if D.size == 0:
        return 0.0

    if D.ndim == 2 and D.shape[1] >= 3:
        D = D[D[:, 2] == 1]

    if D.size == 0:
        return 0.0

    births = D[:, 0]
    deaths = D[:, 1]
    valid = np.isfinite(births) & np.isfinite(deaths) & (deaths > births)

    if not np.any(valid):
        return 0.0

    return float(np.max(deaths[valid] - births[valid]))


def compute_alpha_max_h1(point_cloud: np.ndarray) -> float:
    if not COMPUTE_ALPHA:
        return np.nan

    try:
        import gudhi as gd
    except Exception as exc:
        raise ImportError(
            "COMPUTE_ALPHA=True but GUDHI is not installed. "
            "Install gudhi or set COMPUTE_ALPHA=False for a VR/FastTDA-only diagnostic."
        ) from exc

    pc = np.asarray(point_cloud, dtype=float)

    try:
        ac = gd.AlphaComplex(points=pc.tolist())
        st = ac.create_simplex_tree()
        st.persistence()
        intervals = np.asarray(st.persistence_intervals_in_dimension(1), dtype=float)
    except Exception:
        return np.nan

    if intervals.size == 0:
        return 0.0

    valid = (
        np.isfinite(intervals[:, 0])
        & np.isfinite(intervals[:, 1])
        & (intervals[:, 1] > intervals[:, 0])
    )
    intervals = intervals[valid]

    if intervals.shape[0] == 0:
        return 0.0

    # GUDHI AlphaComplex filtration values are squared radii.
    intervals = np.maximum(intervals, 0.0)
    scaled = 2.0 * np.sqrt(intervals)
    persistence = scaled[:, 1] - scaled[:, 0]

    return float(np.max(persistence))


# ============================================================
# DATASET 6
# ============================================================

def load_dataset6(path: str):
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["Time", "Acceleration", "CartLocation"],
        engine="python",
    )
    df = df[df["Time"] <= DB6_TIME_LIMIT].reset_index(drop=True)

    t = df["Time"].to_numpy(dtype=float)
    acc_raw = df["Acceleration"].to_numpy(dtype=float)
    cart_raw = df["CartLocation"].to_numpy(dtype=float)

    max_acc = np.max(acc_raw)
    acc_scaled = acc_raw.copy() if max_acc == 0 else acc_raw * 20.0 / max_acc

    b, a = signal.butter(
        DB6_FILTER_ORDER,
        DB6_CUTOFF / (0.5 * DB6_FS),
        btype="low",
    )
    acc_filt = signal.filtfilt(b, a, acc_scaled)

    # Correct physical target range used by the paper / FTDA DB6 scripts.
    # A stale early VR script contains target_max=25; that is not used here.
    cart_mm = minmax_to_50_200(cart_raw)

    return t, acc_filt, cart_mm


def extract_db6_profile(t, acc, cart_mm, window_size: int, delay: int, progress_every=500):
    """
    Exact diagnostic profile on one DB6 window size.

    SingleTakensEmbedding is used window-by-window to match the old DB6 scripts.
    """
    vr_model = VietorisRipsPersistence(homology_dimensions=[1])
    embedder = SingleTakensEmbedding(
        parameters_type="fixed",
        time_delay=delay,
        dimension=DB6_DIM,
        stride=DB6_EMBED_STRIDE,
        n_jobs=-1,
    )

    rows = []
    n_windows = max(0, ((len(acc) - window_size) // DB6_WINDOW_STEP) + 1)
    t0 = time.time()

    k = 0
    for start in range(0, len(acc) - window_size + 1, DB6_WINDOW_STEP):
        stop = start + window_size
        acc_win = acc[start:stop]
        cart_win = cart_mm[start:stop]
        time_win = t[start:stop]

        scaled = minmax_minus1_plus1(acc_win)

        try:
            pc = embedder.fit_transform(scaled)
        except Exception:
            pc = np.empty((0, 2), dtype=float)

        midpoint = window_size // 2

        if pc.ndim != 2 or pc.shape[0] < 5:
            vr = np.nan
            alpha = np.nan
            ftda = np.nan
            fit_ok = False
        else:
            vr = compute_vr_max_h1(pc, vr_model)
            alpha = compute_alpha_max_h1(pc)
            ftda, fit_ok = compute_fasttda_minor_axis(pc)

        rows.append({
            "Time": float(time_win[midpoint]),
            "CartLocation": float(cart_win[midpoint]),
            "VR": vr,
            "Alpha": alpha,
            "FastTDA": ftda,
            "FastTDA_ZeroInvalid": float(ftda) if np.isfinite(ftda) else 0.0,
            "FastTDA_FitOK": bool(fit_ok),
            "Window_Number": k,
            "Window_Size": int(window_size),
            "Takens_Delay": int(delay),
        })

        k += 1
        if progress_every and k % progress_every == 0:
            print(
                f"    DB6 W{window_size}_T{delay}: {k}/{n_windows} windows "
                f"| elapsed {time.time()-t0:.1f} s"
            )

    return pd.DataFrame(rows)


def run_dataset6(tag: str, filename: str):
    input_path = os.path.join(WORK_DIR, filename)
    if not os.path.exists(input_path):
        print(f"\nSKIPPED {tag}: could not find DB6 input {input_path}")
        return None

    print("\n" + "=" * 80)
    print(f"Dataset 6 parity extraction: {tag} | {filename}")
    print("=" * 80)

    t, acc, cart_mm = load_dataset6(input_path)

    out_path = os.path.join(WORK_DIR, f"Code2_{tag}_DB6_FineTune.xlsx")

    metadata_rows = []
    profile_frames = {}

    combos = [(w, d) for w in DB6_WINDOW_SIZES for d in DB6_DELAYS]
    for i, (window_size, delay) in enumerate(combos, 1):
        profile_name = f"W{window_size}_T{delay}"
        if delay * (DB6_DIM - 1) >= window_size:
            print(f"  [{i}/{len(combos)}] {profile_name}: skipped (delay too large for window)")
            continue
        print(f"  [{i}/{len(combos)}] Running profile {profile_name} "
              f"(window {window_size / DB6_FS * 1000:.1f} ms, delay {delay / DB6_FS * 1000:.1f} ms)")
        t0 = time.time()
        df_profile = extract_db6_profile(
            t,
            acc,
            cart_mm,
            window_size=window_size,
            delay=delay,
            progress_every=PROGRESS_EVERY_DB6,
        )
        profile_frames[profile_name] = df_profile

        metadata_rows.append({
            "Profile": profile_name,
            "Dataset": tag,
            "Legacy": window_size in DB6_LEGACY_WINDOWS and delay == DB6_LEGACY_DELAY,
            "Sampling_Hz": DB6_FS,
            "Lowpass_Hz": DB6_CUTOFF,
            "Filter_Order": DB6_FILTER_ORDER,
            "Window_Size": window_size,
            "Window_ms": window_size / DB6_FS * 1000,
            "Window_Step": DB6_WINDOW_STEP,
            "Takens_Delay": delay,
            "Delay_ms": delay / DB6_FS * 1000,
            "Takens_Dimension": DB6_DIM,
            "Takens_Stride": DB6_EMBED_STRIDE,
            "Points_per_Window": (window_size - delay * (DB6_DIM - 1) - 1) // DB6_EMBED_STRIDE + 1,
            "Time_Label": "midpoint",
            "Rows": len(df_profile),
            "FastTDA_Invalid_Count": int((~df_profile["FastTDA_FitOK"]).sum()),
            **feature_summary(df_profile, "CartLocation"),
            "Runtime_s": time.time() - t0,
        })

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        pd.DataFrame(metadata_rows).to_excel(writer, sheet_name="Metadata", index=False)
        for profile_name, df_profile in profile_frames.items():
            df_profile.to_excel(writer, sheet_name=profile_name[:31], index=False)

    print(f"  Saved: {out_path}")
    return out_path


# ============================================================
# DATASET 8
# ============================================================

def load_dataset8(path: str):
    data = np.loadtxt(path, skiprows=9)

    acc = data[:, 3]
    pinloc = data[:, 4] / 17.18
    t = data[:, 5]

    sos = signal.butter(
        N=DB8_FILTER_ORDER,
        Wn=DB8_CUTOFF,
        btype="lowpass",
        fs=DB8_FS,
        output="sos",
    )

    acc_filt = signal.sosfiltfilt(sos, acc)
    pinloc_filt = signal.sosfiltfilt(sos, pinloc)

    return t, acc_filt, pinloc_filt


def db8_window_and_tau(max_f: float):
    window_size = int(((1.0 / DB8_MIN_F) + (0.25 / max_f) * 2.0) * DB8_FS)
    tau = int((0.25 / max_f) * DB8_FS / 2.0)
    return window_size, tau


def extract_db8_profile(t, acc, pinloc, window_size: int, tau: int, progress_every=500):
    """
    DB8 extraction using the legacy TakensEmbedding batch convention.
    """

    acc_windows = create_windows(acc, window_size, DB8_WINDOW_STEP)
    pin_windows = create_windows(pinloc, window_size, DB8_WINDOW_STEP)
    time_windows = create_windows(t, window_size, DB8_WINDOW_STEP)

    # Exact old preprocessing: independently min-max scale every window.
    scaled_windows = np.array([
        minmax_minus1_plus1(w)
        for w in acc_windows
    ])

    embedder = TakensEmbedding(
        time_delay=tau,
        dimension=DB8_DIM,
        stride=DB8_EMBED_STRIDE,
    )

    print(
        f"    Building DB8 embeddings: W={window_size}, tau={tau}, "
        f"stride={DB8_EMBED_STRIDE}, windows={len(scaled_windows)}"
    )
    embedded_windows = embedder.fit_transform(scaled_windows)

    vr_model = VietorisRipsPersistence(homology_dimensions=[1])

    rows = []
    t0 = time.time()

    for k, pc in enumerate(embedded_windows):
        pc = np.asarray(pc, dtype=float)

        vr = compute_vr_max_h1(pc, vr_model)
        alpha = compute_alpha_max_h1(pc)
        ftda, fit_ok = compute_fasttda_minor_axis(pc)

        rows.append({
            # Legacy DB8 files label at the END of each raw window.
            "Time": float(time_windows[k, -1]),
            "CartLocation": float(pin_windows[k, -1]),
            "VR": vr,
            "Alpha": alpha,
            "FastTDA": ftda,
            "FastTDA_ZeroInvalid": float(ftda) if np.isfinite(ftda) else 0.0,
            "FastTDA_FitOK": bool(fit_ok),
            "Window_Number": k,
            "Window_Size": int(window_size),
            "Takens_Delay": int(tau),
            "Takens_Stride": int(DB8_EMBED_STRIDE),
        })

        if progress_every and (k + 1) % progress_every == 0:
            print(
                f"    DB8 W{window_size}_T{tau}: {k+1}/{len(embedded_windows)} windows "
                f"| feature elapsed {time.time()-t0:.1f} s"
            )

    return pd.DataFrame(rows)


def run_dataset8(tag: str, filename: str):
    input_path = os.path.join(WORK_DIR, filename)
    if not os.path.exists(input_path):
        print(f"\nSKIPPED {tag}: could not find DB8 input {input_path}")
        return None

    print("\n" + "=" * 80)
    print(f"Dataset 8 parity extraction: {tag} | {filename}")
    print("=" * 80)

    t, acc, pinloc = load_dataset8(input_path)

    out_path = os.path.join(WORK_DIR, f"Code2_{tag}_DB8_FineTune.xlsx")

    metadata_rows = []
    profile_frames = {}

    legacy = {db8_window_and_tau(f) for f in DB8_LEGACY_MAX_F}
    combos = [(w, d) for w in DB8_WINDOW_SIZES for d in DB8_DELAYS]
    for i, (window_size, tau) in enumerate(combos, 1):
        name = f"W{window_size}_T{tau}"
        if tau * (DB8_DIM - 1) >= window_size:
            print(f"  [{i}/{len(combos)}] {name}: skipped (delay too large for window)")
            continue
        print(f"  [{i}/{len(combos)}] Running profile {name} "
              f"(window {window_size / DB8_FS * 1000:.1f} ms, delay {tau / DB8_FS * 1000:.1f} ms)")
        t0 = time.time()
        df_profile = extract_db8_profile(
            t,
            acc,
            pinloc,
            window_size=window_size,
            tau=tau,
            progress_every=PROGRESS_EVERY_DB8,
        )
        profile_frames[name] = df_profile

        metadata_rows.append({
            "Profile": name,
            "Dataset": tag,
            "Legacy": (window_size, tau) in legacy,
            "Sampling_Hz": DB8_FS,
            "Lowpass_Hz": DB8_CUTOFF,
            "Filter_Order": DB8_FILTER_ORDER,
            "Window_Size": window_size,
            "Window_ms": window_size / DB8_FS * 1000,
            "Window_Step": DB8_WINDOW_STEP,
            "Takens_Delay": tau,
            "Delay_ms": tau / DB8_FS * 1000,
            "Takens_Dimension": DB8_DIM,
            "Takens_Stride": DB8_EMBED_STRIDE,
            "Points_per_Window": (window_size - tau * (DB8_DIM - 1) - 1) // DB8_EMBED_STRIDE + 1,
            "Time_Label": "endpoint",
            "Rows": len(df_profile),
            "FastTDA_Invalid_Count": int((~df_profile["FastTDA_FitOK"]).sum()),
            **feature_summary(df_profile, "CartLocation"),
            "Runtime_s": time.time() - t0,
        })

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        pd.DataFrame(metadata_rows).to_excel(writer, sheet_name="Metadata", index=False)
        for profile_name, df_profile in profile_frames.items():
            df_profile.to_excel(writer, sheet_name=profile_name[:31], index=False)

    print(f"  Saved: {out_path}")
    return out_path


# ============================================================
# CSV TRIAL (FastTDA analysis)
# ============================================================

def to_local_path(path: str) -> str:
    """Convert C:\\... to /mnt/c/... when running under Linux/WSL."""
    path = str(path).strip().strip('"')
    if os.name != "nt" and len(path) > 2 and path[1] == ":" and path[2] in "\\/":
        drive = path[0].lower()
        rest = path[3:].replace("\\", "/")
        return f"/mnt/{drive}/{rest}"
    return path


def _resolve_column(df: pd.DataFrame, spec, what: str):
    if spec is None:
        return None
    if isinstance(spec, int):
        return df.columns[spec]
    if spec in df.columns:
        return spec
    raise KeyError(f"{what} column {spec!r} not found. Columns in the file: {list(df.columns)}")


def load_csv_trial(path: str):
    # Delimiter from the header line; a leading unnamed column (saved row index) is dropped
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        first = f.readline()
    counts = {d: first.count(d) for d in (",", ";", "\t")}
    sep = max(counts, key=counts.get) if max(counts.values()) > 0 else r"\s+"

    df = pd.read_csv(path, sep=sep, engine="python", encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, [c for c in df.columns if not c.startswith("Unnamed")]]
    df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all").dropna(axis=0, how="any")
    df = df.reset_index(drop=True)

    print(f"  Loaded {len(df)} rows x {df.shape[1]} numeric columns (delimiter {sep!r})")
    for col in df.columns:
        c = df[col]
        print(f"    {str(col)[:36]:36s} min={c.min():.6g}  max={c.max():.6g}  std={c.std():.4g}")

    time_col = _resolve_column(df, CSV_TIME_COL, "Time")
    sig_col = _resolve_column(df, CSV_SIGNAL_COL, "Signal")
    pos_col = _resolve_column(df, CSV_POS_COL, "Position")
    print(f"  Using: time={time_col!r}, signal={sig_col!r}, position={pos_col!r}")

    t = df[time_col].to_numpy(dtype=float)
    x = df[sig_col].to_numpy(dtype=float)
    pos = df[pos_col].to_numpy(dtype=float) if pos_col is not None else None

    if CSV_FS is not None:
        fs = float(CSV_FS)
    else:
        span = t[-1] - t[0]
        if not np.isfinite(span) or span <= 0:
            raise ValueError("Could not estimate the sampling rate from the time column; set CSV_FS.")
        fs = (len(t) - 1) / span
    if np.any(np.diff(t) <= 0):
        print(f"  Warning: time column is not strictly increasing; rebuilding it as sample/{fs:.2f}")
        t = t[0] + np.arange(len(t)) / fs
    print(f"  Sampling rate: {fs:.2f} Hz ({'set' if CSV_FS else 'estimated from time column'}), "
          f"duration {len(t) / fs:.1f} s")

    if CSV_CUTOFF is not None:
        if CSV_CUTOFF >= fs / 2:
            raise ValueError(f"CSV_CUTOFF ({CSV_CUTOFF} Hz) must be below half the sampling rate "
                             f"({fs / 2:.1f} Hz), or set CSV_CUTOFF = None.")
        sos = signal.butter(N=CSV_FILTER_ORDER, Wn=CSV_CUTOFF, btype="lowpass", fs=fs, output="sos")
        x = signal.sosfiltfilt(sos, x)
        if pos is not None:
            pos = signal.sosfiltfilt(sos, pos)
        print(f"  Low-pass filtered at {CSV_CUTOFF} Hz")

    report_spectrum(x, fs)
    return t, x, pos, fs, (pos_col or None)


def report_spectrum(x: np.ndarray, fs: float, min_hz: float = 0.2):
    """Print the dominant oscillation frequency and the DB8-rule window/delay it implies."""
    nperseg = int(min(len(x), max(256, 20 * fs)))
    f, pxx = signal.welch(x - np.mean(x), fs=fs, nperseg=nperseg)
    band = f >= min_hz
    if not band.any():
        return
    f, pxx = f[band], pxx[band]
    f_peak = f[np.argmax(pxx)]
    cum = np.cumsum(pxx) / np.sum(pxx)
    f_lo, f_hi = f[np.searchsorted(cum, 0.10)], f[min(np.searchsorted(cum, 0.90), len(f) - 1)]
    print(f"  Spectrum (>{min_hz} Hz): dominant {f_peak:.2f} Hz; 10-90% of power in {f_lo:.2f}-{f_hi:.2f} Hz")
    if f_lo > 0 and f_hi > 0:
        win = 1.0 / f_lo + 2 * (0.25 / f_hi)
        tau = (0.25 / f_hi) / 2
        print(f"  DB8-rule suggestion (min_f={f_lo:.2f}, max_f={f_hi:.2f}): "
              f"window ~ {win:.2f} s ({win * fs:.0f} samples), delay ~ {tau:.3f} s ({tau * fs:.1f} samples)")


def takens_points(x: np.ndarray, delay: int, dim: int, stride: int) -> np.ndarray:
    """Takens point cloud of one window; same indexing as gtda TakensEmbedding
    (points are aligned to the end of the window, so the last point uses the last sample)."""
    span = len(x) - (dim - 1) * delay - 1
    if span < 0:
        return np.empty((0, dim))
    n_points = span // stride + 1
    first = span % stride
    last = first + (n_points - 1) * stride
    return np.column_stack([x[first + d * delay: last + d * delay + 1: stride] for d in range(dim)])


def extract_csv_profile(t, acc, pos, window_size, delay, step, stride, progress_every):
    vr_model = VietorisRipsPersistence(homology_dimensions=[1]) if CSV_COMPUTE_VR else None
    rows = []
    n_windows = max(0, (len(acc) - window_size) // step + 1)
    t0 = time.time()
    for k, start in enumerate(range(0, len(acc) - window_size + 1, step)):
        stop = start + window_size
        pc = takens_points(minmax_minus1_plus1(acc[start:stop]), delay, CSV_DIM, stride)
        ftda, fit_ok = compute_fasttda_minor_axis(pc)
        row = {"Time": float(t[stop - 1])}               # endpoint label, as DB8
        if pos is not None:
            row["CartLocation"] = float(pos[stop - 1])
        row.update({
            "FastTDA": ftda,
            "FastTDA_ZeroInvalid": float(ftda) if np.isfinite(ftda) else 0.0,
            "FastTDA_FitOK": bool(fit_ok),
            "VR": compute_vr_max_h1(pc, vr_model) if CSV_COMPUTE_VR else np.nan,
            "Alpha": compute_alpha_max_h1(pc) if CSV_COMPUTE_ALPHA else np.nan,
            "Window_Number": k,
            "Window_Size": int(window_size),
            "Takens_Delay": int(delay),
        })
        rows.append(row)
        if progress_every and (k + 1) % progress_every == 0:
            print(f"    W{window_size}_T{delay}: {k+1}/{n_windows} windows | elapsed {time.time()-t0:.1f} s")
    return pd.DataFrame(rows)


def run_csv_trial(csv_path: str):
    input_path = to_local_path(csv_path)
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Could not find CSV trial:\n{input_path}")

    tag = os.path.splitext(os.path.basename(input_path))[0]
    print("\n" + "=" * 80)
    print(f"CSV FastTDA extraction: {tag}")
    print("=" * 80)

    t, acc, pos, fs, pos_col = load_csv_trial(input_path)

    step = max(1, int(CSV_STEP_SAMPLES))
    stride = max(1, int(CSV_EMBED_STRIDE))
    out_dir = to_local_path(CSV_OUTPUT_DIR) if CSV_OUTPUT_DIR else os.path.dirname(input_path)
    out_path = os.path.join(out_dir, f"Code2_{tag}_FastTDA_FineTune.xlsx")

    metadata_rows = []
    profile_frames = {}
    combos = [(w, d) for w in CSV_WINDOW_S for d in CSV_DELAY_S]
    for i, (w_s, d_s) in enumerate(combos, 1):
        window_size = int(round(w_s * fs))
        delay = max(1, int(round(d_s * fs)))
        name = f"W{window_size}_T{delay}"
        if name in profile_frames:
            continue
        n_points = (window_size - delay * (CSV_DIM - 1) - 1) // stride + 1
        if n_points < 5 or window_size > len(acc):
            print(f"  [{i}/{len(combos)}] {name}: skipped (fewer than 5 embedded points, "
                  f"or window longer than signal)")
            continue
        print(f"  [{i}/{len(combos)}] Running profile {name} "
              f"(window {window_size / fs:.3f} s, delay {delay / fs:.3f} s, "
              f"{n_points} points/window)")
        t0 = time.time()
        df_profile = extract_csv_profile(t, acc, pos, window_size, delay, step, stride, PROGRESS_EVERY_CSV)
        profile_frames[name] = df_profile
        metadata_rows.append({
            "Profile": name,
            "Dataset": tag,
            "Sampling_Hz": fs,
            "Lowpass_Hz": CSV_CUTOFF,
            "Filter_Order": CSV_FILTER_ORDER,
            "Window_Size": window_size,
            "Signal_Column": CSV_SIGNAL_COL,
            "Window_s": window_size / fs,
            "Window_Step": step,
            "Takens_Delay": delay,
            "Delay_s": delay / fs,
            "Takens_Dimension": CSV_DIM,
            "Takens_Stride": stride,
            "Points_per_Window": n_points,
            "Time_Label": "endpoint",
            "Rows": len(df_profile),
            "FastTDA_Invalid_Count": int((~df_profile["FastTDA_FitOK"]).sum()),
            **feature_summary(df_profile, "CartLocation" if pos is not None else None),
            "Runtime_s": time.time() - t0,
        })

    os.makedirs(out_dir, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        pd.DataFrame(metadata_rows).to_excel(writer, sheet_name="Metadata", index=False)
        for profile_name, df_profile in profile_frames.items():
            df_profile.to_excel(writer, sheet_name=profile_name[:31], index=False)

    print(f"  Saved: {out_path}")
    return out_path


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 80)
    print("DROPBEAR Code 2: Feature-extraction parity profiles")
    print("=" * 80)
    print(f"WORK_DIR: {WORK_DIR}")
    print("Outputs are written as *_FineTune.xlsx (parity files are not overwritten).")
    print(f"DB6 grid: windows {DB6_WINDOW_SIZES} x delays {DB6_DELAYS} "
          f"({len(DB6_WINDOW_SIZES) * len(DB6_DELAYS)} profiles per file)")
    print(f"DB8 grid: windows {DB8_WINDOW_SIZES} x delays {DB8_DELAYS} "
          f"({len(DB8_WINDOW_SIZES) * len(DB8_DELAYS)} profiles)")
    print(f"Run: DB6={RUN_DB6}, DB8={RUN_DB8}, CSV={RUN_CSV}")
    print(f"Alpha enabled: DB6/DB8={COMPUTE_ALPHA}, CSV={CSV_COMPUTE_ALPHA}")
    print("=" * 80)

    if (RUN_DB6 or RUN_DB8) and not os.path.isdir(WORK_DIR):
        raise FileNotFoundError(f"WORK_DIR does not exist:\n{WORK_DIR}")

    # Fail early (not on the first window) if Alpha is requested without GUDHI
    if (COMPUTE_ALPHA and (RUN_DB6 or RUN_DB8)) or (CSV_COMPUTE_ALPHA and RUN_CSV):
        try:
            import gudhi  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "COMPUTE_ALPHA=True but GUDHI is not installed. Run `pip install gudhi` "
                "or set COMPUTE_ALPHA=False for a VR/FastTDA-only diagnostic."
            ) from exc

    outputs = []

    if RUN_DB6:
        for tag, filename in DB6_INPUTS.items():
            outputs.append(run_dataset6(tag, filename))

    if RUN_DB8:
        for tag, filename in DB8_INPUT.items():
            outputs.append(run_dataset8(tag, filename))

    if RUN_CSV:
        outputs.append(run_csv_trial(CSV_PATH))

    outputs = [p for p in outputs if p is not None]

    print("\n" + "=" * 80)
    print("Code 2 extraction complete.")
    print("Next: apply legacy metrics (Code 3 diagnostic pass) to each profile.")
    print("Outputs:")
    for path in outputs:
        print("  -", path)
    if not outputs:
        print("  (none - no input files were found)")
    print("=" * 80)


if __name__ == "__main__":
    main()
