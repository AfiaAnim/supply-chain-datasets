"""
FastTDA fine-tuning on a single capacitance trial (e.g. trial_01.csv).

For every (window length, Takens delay) combination in the grid:
    1. slide a window over the signal,
    2. min-max scale the window to [-1, 1],
    3. build a 2-D Takens embedding,
    4. FastTDA: fit an ellipse (SVD-LSE, A + C = 1) and report its minor axis 2b
       (optionally also VR / Alpha max H1 persistence).

Output: <trial>_FastTDA_FineTune.xlsx next to the CSV, with a "Summary" sheet and one
sheet per combination (W<window samples>_T<delay samples>).

Requirements:
    pip install numpy pandas scipy scikit-learn openpyxl
    (+ giotto-tda for VR, + gudhi for Alpha)
"""

import os
import time

import numpy as np
import pandas as pd
from scipy import signal
from sklearn.preprocessing import MinMaxScaler


# ============================================================
# USER CONFIGURATION
# ============================================================

# Windows paths (C:\...) are converted to WSL paths (/mnt/c/...) automatically.
CSV_PATH = r"C:\Users\aphya\Downloads\trial_01.csv"
OUTPUT_DIR = None            # None = save next to the CSV

# Columns (header names). A leading unnamed row-index column is ignored.
TIME_COL = "time_s"
SIGNAL_COL = "capacitance_filtered_pF"
POS_COL = None               # position/label column, or None if there is none

FS = None                    # sampling rate in Hz; None = estimate from the time column (~88 Hz)
CUTOFF = None                # extra low-pass (Hz); None = none. Must be < FS/2 (~44 Hz).
                             # e.g. SIGNAL_COL = "capacitance_raw_pF" with CUTOFF = 5
FILTER_ORDER = 6
DIM = 2

# Fine-tuning grid in SECONDS; every window is run with every delay.
WINDOW_S = [1.0, 1.5, 2.0]
DELAY_S = [0.03, 0.06, 0.09]
STEP_SAMPLES = 1             # window step in samples
EMBED_STRIDE = 1             # samples between embedded points

COMPUTE_VR = False           # VR max H1 (needs giotto-tda; slower)
COMPUTE_ALPHA = False        # Alpha max H1 (needs gudhi; slowest)
PROGRESS_EVERY = 2000


# ============================================================
# HELPERS
# ============================================================

def to_local_path(path: str) -> str:
    """Convert C:\\... to /mnt/c/... when running under Linux/WSL."""
    path = str(path).strip().strip('"')
    if os.name != "nt" and len(path) > 2 and path[1] == ":" and path[2] in "\\/":
        return f"/mnt/{path[0].lower()}/{path[3:].replace(chr(92), '/')}"
    return path


def minmax_minus1_plus1(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    return MinMaxScaler(feature_range=(-1, 1)).fit_transform(x).ravel()


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


# ============================================================
# FAST TDA: SVD-LSE ELLIPSE FIT
# ============================================================

def fit_ellipse_AplusC_eq_1(X, Y, rcond=None):
    """Fit conic A(x^2-y^2) + Bxy + Dx + Ey + F = -y^2 with C = 1 - A (centred and scaled)."""
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

    G = np.column_stack([Xc**2 - Yc**2, Xc * Yc, Xc, Yc, np.ones_like(Xc)])
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


def compute_fasttda_minor_axis(point_cloud: np.ndarray):
    """Return (2b, valid_fit). Invalid fits return (NaN, False)."""
    pc = np.asarray(point_cloud, dtype=float)
    if pc.ndim != 2 or pc.shape[0] < 5 or pc.shape[1] < 2:
        return np.nan, False
    params = fit_ellipse_AplusC_eq_1(pc[:, 0], pc[:, 1])
    if params is None:
        return np.nan, False
    pose = rotate_and_axes(*params)
    if pose is None:
        return np.nan, False
    return float(2.0 * pose[4]), True


# ============================================================
# OPTIONAL: VR / ALPHA MAX H1
# ============================================================

def compute_vr_max_h1(point_cloud: np.ndarray, vr_model) -> float:
    try:
        D = np.asarray(vr_model.fit_transform(point_cloud[None, :, :])[0], dtype=float)
    except Exception:
        return np.nan
    if D.size == 0:
        return 0.0
    D = D[D[:, 2] == 1]
    valid = np.isfinite(D[:, 0]) & np.isfinite(D[:, 1]) & (D[:, 1] > D[:, 0])
    return float(np.max(D[valid, 1] - D[valid, 0])) if valid.any() else 0.0


def compute_alpha_max_h1(point_cloud: np.ndarray) -> float:
    import gudhi as gd
    try:
        st = gd.AlphaComplex(points=point_cloud.tolist()).create_simplex_tree()
        st.persistence()
        iv = np.asarray(st.persistence_intervals_in_dimension(1), dtype=float)
    except Exception:
        return np.nan
    if iv.size == 0:
        return 0.0
    iv = iv[np.isfinite(iv[:, 0]) & np.isfinite(iv[:, 1]) & (iv[:, 1] > iv[:, 0])]
    if iv.shape[0] == 0:
        return 0.0
    scaled = 2.0 * np.sqrt(np.maximum(iv, 0.0))      # GUDHI alpha values are squared radii
    return float(np.max(scaled[:, 1] - scaled[:, 0]))


# ============================================================
# LOAD TRIAL
# ============================================================

def report_spectrum(x: np.ndarray, fs: float, min_hz: float = 0.2):
    """Print the dominant frequency band and the DB8-rule window/delay it implies."""
    nperseg = int(min(len(x), max(256, 20 * fs)))
    f, pxx = signal.welch(x - np.mean(x), fs=fs, nperseg=nperseg)
    band = f >= min_hz
    if not band.any():
        return
    f, pxx = f[band], pxx[band]
    cum = np.cumsum(pxx) / np.sum(pxx)
    f_lo = f[np.searchsorted(cum, 0.10)]
    f_hi = f[min(np.searchsorted(cum, 0.90), len(f) - 1)]
    print(f"  Spectrum (>{min_hz} Hz): dominant {f[np.argmax(pxx)]:.2f} Hz; "
          f"10-90% of power in {f_lo:.2f}-{f_hi:.2f} Hz")
    if f_lo > 0 and f_hi > 0:
        win = 1.0 / f_lo + 2 * (0.25 / f_hi)
        tau = (0.25 / f_hi) / 2
        print(f"  DB8-rule suggestion (min_f={f_lo:.2f}, max_f={f_hi:.2f}): window ~ {win:.2f} s "
              f"({win * fs:.0f} samples), delay ~ {tau:.3f} s ({tau * fs:.1f} samples)")


def load_trial(path: str):
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        first = f.readline()
    counts = {d: first.count(d) for d in (",", ";", "\t")}
    sep = max(counts, key=counts.get) if max(counts.values()) > 0 else r"\s+"

    df = pd.read_csv(path, sep=sep, engine="python", encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, [c for c in df.columns if not c.startswith("Unnamed")]]
    df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all").dropna(axis=0, how="any")
    df = df.reset_index(drop=True)

    print(f"  Loaded {len(df)} rows x {df.shape[1]} columns (delimiter {sep!r})")
    for col in df.columns:
        c = df[col]
        print(f"    {col[:36]:36s} min={c.min():.6g}  max={c.max():.6g}  std={c.std():.4g}")

    for name, col in (("TIME_COL", TIME_COL), ("SIGNAL_COL", SIGNAL_COL), ("POS_COL", POS_COL)):
        if col is not None and col not in df.columns:
            raise KeyError(f"{name} = {col!r} not found. Columns in the file: {list(df.columns)}")

    t = df[TIME_COL].to_numpy(dtype=float)
    x = df[SIGNAL_COL].to_numpy(dtype=float)
    pos = df[POS_COL].to_numpy(dtype=float) if POS_COL else None

    if FS is not None:
        fs = float(FS)
    else:
        span = t[-1] - t[0]
        if not np.isfinite(span) or span <= 0:
            raise ValueError("Could not estimate the sampling rate from the time column; set FS.")
        fs = (len(t) - 1) / span
    if np.any(np.diff(t) <= 0):
        print(f"  Warning: time column is not strictly increasing; rebuilding it as sample/{fs:.2f}")
        t = t[0] + np.arange(len(t)) / fs
    print(f"  Sampling rate: {fs:.2f} Hz, duration {len(t) / fs:.1f} s")

    if CUTOFF is not None:
        if CUTOFF >= fs / 2:
            raise ValueError(f"CUTOFF ({CUTOFF} Hz) must be below half the sampling rate "
                             f"({fs / 2:.1f} Hz), or set CUTOFF = None.")
        sos = signal.butter(N=FILTER_ORDER, Wn=CUTOFF, btype="lowpass", fs=fs, output="sos")
        x = signal.sosfiltfilt(sos, x)
        if pos is not None:
            pos = signal.sosfiltfilt(sos, pos)
        print(f"  Low-pass filtered at {CUTOFF} Hz")

    report_spectrum(x, fs)
    return t, x, pos, fs


# ============================================================
# EXTRACTION
# ============================================================

def extract_profile(t, x, pos, window_size, delay, step, stride, vr_model):
    rows = []
    n_windows = (len(x) - window_size) // step + 1
    t0 = time.time()
    for k, start in enumerate(range(0, len(x) - window_size + 1, step)):
        stop = start + window_size
        pc = takens_points(minmax_minus1_plus1(x[start:stop]), delay, DIM, stride)
        ftda, fit_ok = compute_fasttda_minor_axis(pc)

        row = {"Time": float(t[stop - 1])}                 # label at the window end
        if pos is not None:
            row["Position"] = float(pos[stop - 1])
        row.update({
            "FastTDA": ftda,
            "FastTDA_ZeroInvalid": float(ftda) if np.isfinite(ftda) else 0.0,
            "FastTDA_FitOK": bool(fit_ok),
        })
        if COMPUTE_VR:
            row["VR"] = compute_vr_max_h1(pc, vr_model)
        if COMPUTE_ALPHA:
            row["Alpha"] = compute_alpha_max_h1(pc)
        row.update({"Window_Number": k, "Window_Size": window_size, "Takens_Delay": delay})
        rows.append(row)

        if PROGRESS_EVERY and (k + 1) % PROGRESS_EVERY == 0:
            print(f"    W{window_size}_T{delay}: {k + 1}/{n_windows} windows | {time.time() - t0:.1f} s")
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict:
    out = {}
    for col in ("FastTDA", "VR", "Alpha"):
        if col not in df:
            continue
        x = df[col].astype(float)
        out[f"{col}_mean"] = float(np.nanmean(x)) if x.notna().any() else np.nan
        out[f"{col}_std"] = float(np.nanstd(x)) if x.notna().any() else np.nan
        if "Position" in df:
            ok = x.notna()
            out[f"Corr_{col}_Position"] = (float(np.corrcoef(x[ok], df.loc[ok, "Position"])[0, 1])
                                           if ok.sum() > 2 and x[ok].std() > 0 else np.nan)
    return out


def main():
    input_path = to_local_path(CSV_PATH)
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Could not find the trial CSV:\n{input_path}")
    tag = os.path.splitext(os.path.basename(input_path))[0]
    print("=" * 80)
    print(f"FastTDA fine-tuning: {input_path}")
    print("=" * 80)

    t, x, pos, fs = load_trial(input_path)

    vr_model = None
    if COMPUTE_VR:
        from gtda.homology import VietorisRipsPersistence
        vr_model = VietorisRipsPersistence(homology_dimensions=[1])
    if COMPUTE_ALPHA:
        import gudhi  # noqa: F401  (fail early if missing)

    step = max(1, int(STEP_SAMPLES))
    stride = max(1, int(EMBED_STRIDE))
    combos = [(w, d) for w in WINDOW_S for d in DELAY_S]

    sheets, summary = {}, []
    for i, (w_s, d_s) in enumerate(combos, 1):
        window_size = int(round(w_s * fs))
        delay = max(1, int(round(d_s * fs)))
        name = f"W{window_size}_T{delay}"
        if name in sheets:
            continue
        n_points = (window_size - delay * (DIM - 1) - 1) // stride + 1
        if n_points < 5 or window_size > len(x):
            print(f"  [{i}/{len(combos)}] {name}: skipped (fewer than 5 embedded points, "
                  f"or window longer than signal)")
            continue

        print(f"  [{i}/{len(combos)}] {name}: window {window_size / fs:.3f} s, "
              f"delay {delay / fs:.3f} s, {n_points} points/window")
        t0 = time.time()
        df = extract_profile(t, x, pos, window_size, delay, step, stride, vr_model)
        sheets[name] = df
        summary.append({
            "Profile": name,
            "Signal_Column": SIGNAL_COL,
            "Sampling_Hz": fs,
            "Lowpass_Hz": CUTOFF,
            "Window_Size": window_size,
            "Window_s": window_size / fs,
            "Takens_Delay": delay,
            "Delay_s": delay / fs,
            "Window_Step": step,
            "Takens_Stride": stride,
            "Points_per_Window": n_points,
            "Windows": len(df),
            "FastTDA_Invalid_Count": int((~df["FastTDA_FitOK"]).sum()),
            **summarize(df),
            "Runtime_s": time.time() - t0,
        })

    out_dir = to_local_path(OUTPUT_DIR) if OUTPUT_DIR else os.path.dirname(input_path)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{tag}_FastTDA_FineTune.xlsx")
    summary_df = pd.DataFrame(summary)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)

    print("\n" + summary_df[["Profile", "Window_s", "Delay_s", "Points_per_Window",
                             "FastTDA_mean", "FastTDA_std", "FastTDA_Invalid_Count"]]
          .round(4).to_string(index=False))
    print(f"\nSaved: {out_path}")
    return summary_df, sheets


if __name__ == "__main__":
    main()
