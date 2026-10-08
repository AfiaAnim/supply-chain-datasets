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
DB6_DELAY = 100
DB6_DIM = 2
DB6_EMBED_STRIDE = 25
DB6_WINDOW_PROFILES = (1500, 1612)

# Dataset 8 legacy/common settings
DB8_FS = 5000
DB8_CUTOFF = 100
DB8_FILTER_ORDER = 6
DB8_MIN_F = 17.7
DB8_WINDOW_STEP = 5
DB8_DIM = 2
DB8_EMBED_STRIDE = 1

# Two profiles are deliberate because the original VR/FastTDA scripts used
# slightly different max_f constants (31.1 vs 31.0), changing int(window_size).
DB8_PROFILES = [
    {
        "name": "W362_S1_M31p1",
        "max_f": 31.1,
        "legacy_role": "old VR profile",
    },
    {
        "name": "W363_S1_M31p0",
        "max_f": 31.0,
        "legacy_role": "old FastTDA profile",
    },
]

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


def extract_db6_profile(t, acc, cart_mm, window_size: int, progress_every=500):
    """
    Exact diagnostic profile on one DB6 window size.

    SingleTakensEmbedding is used window-by-window to match the old DB6 scripts.
    """
    vr_model = VietorisRipsPersistence(homology_dimensions=[1])
    embedder = SingleTakensEmbedding(
        parameters_type="fixed",
        time_delay=DB6_DELAY,
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
        })

        k += 1
        if progress_every and k % progress_every == 0:
            print(
                f"    DB6 W{window_size}: {k}/{n_windows} windows "
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

    out_path = os.path.join(WORK_DIR, f"Code2_{tag}_DB6_ParityProfiles.xlsx")

    metadata_rows = []
    profile_frames = {}

    for window_size in DB6_WINDOW_PROFILES:
        profile_name = f"W{window_size}"
        print(f"  Running profile {profile_name}")
        df_profile = extract_db6_profile(
            t,
            acc,
            cart_mm,
            window_size=window_size,
            progress_every=PROGRESS_EVERY_DB6,
        )
        profile_frames[profile_name] = df_profile

        metadata_rows.append({
            "Profile": profile_name,
            "Dataset": tag,
            "Sampling_Hz": DB6_FS,
            "Lowpass_Hz": DB6_CUTOFF,
            "Filter_Order": DB6_FILTER_ORDER,
            "Window_Size": window_size,
            "Window_Step": DB6_WINDOW_STEP,
            "Takens_Delay": DB6_DELAY,
            "Takens_Dimension": DB6_DIM,
            "Takens_Stride": DB6_EMBED_STRIDE,
            "Time_Label": "midpoint",
            "Rows": len(df_profile),
            "FastTDA_Invalid_Count": int((~df_profile["FastTDA_FitOK"]).sum()),
            "Note": "W1500 tests old VR convention; W1612 tests old FTDA/formula convention",
        })

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for profile_name, df_profile in profile_frames.items():
            df_profile.to_excel(writer, sheet_name=profile_name[:31], index=False)
        pd.DataFrame(metadata_rows).to_excel(writer, sheet_name="Metadata", index=False)

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


def extract_db8_profile(t, acc, pinloc, max_f: float, progress_every=500):
    """
    DB8 extraction using the legacy TakensEmbedding batch convention.
    """
    window_size, tau = db8_window_and_tau(max_f)

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
            "Max_F": float(max_f),
        })

        if progress_every and (k + 1) % progress_every == 0:
            print(
                f"    DB8 W{window_size} S1: {k+1}/{len(embedded_windows)} windows "
                f"| feature elapsed {time.time()-t0:.1f} s"
            )

    return pd.DataFrame(rows), window_size, tau


def run_dataset8(tag: str, filename: str):
    input_path = os.path.join(WORK_DIR, filename)
    if not os.path.exists(input_path):
        print(f"\nSKIPPED {tag}: could not find DB8 input {input_path}")
        return None

    print("\n" + "=" * 80)
    print(f"Dataset 8 parity extraction: {tag} | {filename}")
    print("=" * 80)

    t, acc, pinloc = load_dataset8(input_path)

    out_path = os.path.join(WORK_DIR, f"Code2_{tag}_DB8_ParityProfiles.xlsx")

    metadata_rows = []
    profile_frames = {}

    for profile in DB8_PROFILES:
        name = profile["name"]
        max_f = float(profile["max_f"])

        print(f"  Running profile {name} ({profile['legacy_role']})")
        df_profile, window_size, tau = extract_db8_profile(
            t,
            acc,
            pinloc,
            max_f=max_f,
            progress_every=PROGRESS_EVERY_DB8,
        )
        profile_frames[name] = df_profile

        metadata_rows.append({
            "Profile": name,
            "Dataset": tag,
            "Legacy_Role": profile["legacy_role"],
            "Sampling_Hz": DB8_FS,
            "Lowpass_Hz": DB8_CUTOFF,
            "Filter_Order": DB8_FILTER_ORDER,
            "Min_F": DB8_MIN_F,
            "Max_F": max_f,
            "Window_Size": window_size,
            "Window_Step": DB8_WINDOW_STEP,
            "Takens_Delay": tau,
            "Takens_Dimension": DB8_DIM,
            "Takens_Stride": DB8_EMBED_STRIDE,
            "Time_Label": "endpoint",
            "Rows": len(df_profile),
            "FastTDA_Invalid_Count": int((~df_profile["FastTDA_FitOK"]).sum()),
        })

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for profile_name, df_profile in profile_frames.items():
            df_profile.to_excel(writer, sheet_name=profile_name[:31], index=False)
        pd.DataFrame(metadata_rows).to_excel(writer, sheet_name="Metadata", index=False)

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
    print("No current feature Excel files will be overwritten.")
    print("DB6 profiles: W1500 and W1612")
    print("DB8 profiles: W362/S1/max_f31.1 and W363/S1/max_f31.0")
    print(f"Alpha enabled: {COMPUTE_ALPHA}")
    print("=" * 80)

    if not os.path.isdir(WORK_DIR):
        raise FileNotFoundError(f"WORK_DIR does not exist:\n{WORK_DIR}")

    # Fail early (not on the first window) if Alpha is requested without GUDHI
    if COMPUTE_ALPHA:
        try:
            import gudhi  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "COMPUTE_ALPHA=True but GUDHI is not installed. Run `pip install gudhi` "
                "or set COMPUTE_ALPHA=False for a VR/FastTDA-only diagnostic."
            ) from exc

    outputs = []

    for tag, filename in DB6_INPUTS.items():
        outputs.append(run_dataset6(tag, filename))

    for tag, filename in DB8_INPUT.items():
        outputs.append(run_dataset8(tag, filename))

    outputs = [p for p in outputs if p is not None]

    print("\n" + "=" * 80)
    print("Code 2 extraction complete.")
    print("Next: apply legacy metrics (Code 3 diagnostic pass) to each profile.")
    print("Outputs:")
    for path in outputs:
        print("  -", path)
    if not outputs:
        print("  (none - no input files were found in WORK_DIR)")
    print("=" * 80)


if __name__ == "__main__":
    main()
