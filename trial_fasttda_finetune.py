"""
FastTDA fine-tuning on a single capacitance trial (e.g. trial_01.csv).

For every (window length, Takens delay) setting:
    1. slide a window over the signal,
    2. min-max scale the window to [-1, 1],
    3. build a 2-D Takens embedding,
    4. FastTDA: fit an ellipse (SVD-LSE, A + C = 1) and report its minor axis 2b
       (optionally also VR / Alpha max H1 persistence).

Auto-tuning (MODE = "auto"):
    Stage 1 - coarse grid over WINDOW_S x DELAY_S.
    Stage 2 - refine grid around the best coarse setting.
    Every setting is scored on how reliable its FastTDA output is (no label needed):
        Invalid_pct  - % of windows where the ellipse fit fails       (lower is better)
        Spike_pct    - % of values far above the typical level        (lower is better)
        Roughness    - median step-to-step jump / interquartile range (lower = smoother)
        Window_s     - window length (shorter = better time resolution)
    Longer windows always look smoother and fail less (they overlap more and blur fast
    changes), so window length is scored too; RESOLUTION_WEIGHT sets how much it counts.
    Score = weighted mean rank over these metrics (lower is better). Rank 1 is recommended.
    If the file has a label column (POS_COL), |correlation| with it is added to the score.

MODE = "grid" runs only WINDOW_S x DELAY_S (no refine stage).
MODE = "fixed" runs only FIXED_WINDOW_SAMPLES / FIXED_DELAY_SAMPLES (the chosen setting) and
saves it to <trial>_FastTDA_final.xlsx, leaving the tuning results untouched.

Cleaning (every profile sheet gets FastTDA_clean + Clean_Mask; FastTDA stays raw):
    startup    - windows ending before TRIM_START_S (start-up transient)    -> NaN
    ok         - kept as is
    fit_failed - ellipse fit failed
    gap_edge   - within EDGE_GUARD_WINDOWS of a failed fit (unstable, gives edge spikes)
    spike      - above median + CLEAN_SPIKE_MADS robust std
    Masked runs up to MAX_INTERP_GAP_S long are filled by linear interpolation in time.
    Use FastTDA_clean for analysis instead of FastTDA_ZeroInvalid (zeros are not real values).

Output: <trial>_FastTDA_FineTune.xlsx next to the CSV:
    "Summary"  - every setting tried, sorted by Score, with all metrics
    one sheet per top-ranked setting (W<window samples>_T<delay samples>)

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
POS_COL = None               # label column (e.g. position/damage), or None if there is none

FS = None                    # sampling rate in Hz; None = estimate from the time column (~88 Hz)
CUTOFF = None                # extra low-pass (Hz); None = none. Must be < FS/2 (~44 Hz).
                             # e.g. SIGNAL_COL = "capacitance_raw_pF" with CUTOFF = 5
FILTER_ORDER = 6
DIM = 2

MODE = "fixed"               # "fixed" = run the chosen setting only;
                             # "auto" = coarse grid + refine around the best; "grid" = grid only

# Chosen setting (from the auto-tuning run: Rank 1 = W159_T8, window 1.80 s, delay 0.091 s)
FIXED_WINDOW_SAMPLES = 159
FIXED_DELAY_SAMPLES = 8

# Coarse grid in SECONDS (every window is run with every delay)
WINDOW_S = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
DELAY_S = [0.03, 0.06, 0.09, 0.12, 0.16, 0.20, 0.25]

# Refine stage (MODE = "auto"): multipliers applied to the best coarse window / delay
REFINE_WINDOW_FACTORS = [0.8, 0.9, 1.0, 1.1, 1.2]
REFINE_DELAY_FACTORS = [0.75, 0.875, 1.0, 1.125, 1.25]

STEP_SAMPLES = 1             # window step in samples
EMBED_STRIDE = 1             # samples between embedded points
MAX_DELAY_FRACTION = 0.25    # skip settings whose delay is longer than this fraction of the window
RESOLUTION_WEIGHT = 1.0      # weight of "shorter window" in the score (0 = ignore time resolution)
SPIKE_MADS = 6.0             # a spike is a value above median + SPIKE_MADS * robust std
TOP_N_SHEETS = 9             # how many of the best settings get their own sheet

COMPUTE_VR = False           # VR max H1 (needs giotto-tda; slower)
COMPUTE_ALPHA = False        # Alpha max H1 (needs gudhi; slowest)
PROGRESS_EVERY = 0           # print progress every N windows (0 = off)

# Cleaning -> adds FastTDA_clean + Clean_Mask columns (the raw FastTDA column is kept as is)
TRIM_START_S = 2.7           # windows ending before this time are start-up transient -> dropped (NaN)
EDGE_GUARD_WINDOWS = 5       # also mask this many windows on each side of every failed fit
CLEAN_SPIKE_MADS = 6.0       # mask values above median + this many robust std (None = off)
MAX_INTERP_GAP_S = 1.0       # fill masked runs up to this long by linear interpolation; longer stay NaN


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
# EXTRACTION AND SCORING
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


def clean_fasttda(df: pd.DataFrame, fs: float, step: int) -> pd.DataFrame:
    """Add FastTDA_clean (masked + interpolated) and Clean_Mask (why each value was masked)."""
    df = df.copy()
    t = df["Time"].to_numpy(float)
    v = df["FastTDA"].to_numpy(float)
    ok = df["FastTDA_FitOK"].to_numpy(bool) & np.isfinite(v)
    reason = np.full(len(df), "ok", dtype=object)

    # 1) failed ellipse fits
    reason[~ok] = "fit_failed"

    # 2) windows next to a failed fit (the fit is unstable there -> edge spikes)
    if EDGE_GUARD_WINDOWS > 0 and (~ok).any():
        near = np.convolve((~ok).astype(int), np.ones(2 * EDGE_GUARD_WINDOWS + 1, int), mode="same") > 0
        reason[near & ok] = "gap_edge"

    # 3) remaining high spikes
    if CLEAN_SPIKE_MADS:
        keep = reason == "ok"
        if keep.sum() >= 10:
            med = np.median(v[keep])
            mad = 1.4826 * np.median(np.abs(v[keep] - med))
            if mad > 0:
                reason[keep & (v > med + CLEAN_SPIKE_MADS * mad)] = "spike"

    # 4) start-up transient
    if TRIM_START_S:
        reason[t < TRIM_START_S] = "startup"

    good = reason == "ok"
    clean = np.where(good, v, np.nan)
    # Fill short masked runs by linear interpolation in time; long gaps and the ends stay NaN
    max_gap = int(round(MAX_INTERP_GAP_S * fs / step)) if MAX_INTERP_GAP_S else 0
    if max_gap > 0 and good.sum() >= 2:
        filled = np.interp(t, t[good], v[good])
        edges = np.flatnonzero(np.diff(np.r_[0, (~good).astype(int), 0]))
        for a, b in zip(edges[::2], edges[1::2]):          # masked run = rows a .. b-1
            if a > 0 and b < len(df) and b - a <= max_gap:
                clean[a:b] = filled[a:b]
    clean[reason == "startup"] = np.nan

    i = df.columns.get_loc("FastTDA_FitOK") + 1          # place right after FastTDA_FitOK
    df.insert(i, "FastTDA_clean", clean)
    df.insert(i + 1, "Clean_Mask", reason)
    return df


def clean_report(df: pd.DataFrame) -> dict:
    r = df["Clean_Mask"]
    return {"Clean_Startup": int((r == "startup").sum()), "Clean_FitFailed": int((r == "fit_failed").sum()),
            "Clean_GapEdge": int((r == "gap_edge").sum()), "Clean_Spike": int((r == "spike").sum()),
            "Clean_Filled": int((r != "ok").sum() - df["FastTDA_clean"].isna().sum()),
            "Clean_NaN": int(df["FastTDA_clean"].isna().sum())}


def quality_metrics(df: pd.DataFrame) -> dict:
    """Reliability of a FastTDA series (no label needed)."""
    v = df["FastTDA"].to_numpy(float)
    ok = np.isfinite(v)
    out = {"Invalid_pct": 100.0 * (1 - ok.mean())}
    v = v[ok]
    if v.size < 10:
        return {**out, "Spike_pct": np.nan, "Roughness": np.nan,
                "FastTDA_median": np.nan, "FastTDA_mean": np.nan, "FastTDA_std": np.nan}
    med = np.median(v)
    mad = 1.4826 * np.median(np.abs(v - med))
    iqr = np.subtract(*np.percentile(v, [75, 25]))
    out["Spike_pct"] = 100.0 * np.mean(v > med + SPIKE_MADS * mad) if mad > 0 else 0.0
    out["Roughness"] = float(np.median(np.abs(np.diff(v))) / iqr) if iqr > 0 else np.nan
    out["FastTDA_median"] = float(med)
    out["FastTDA_mean"] = float(np.mean(v))
    out["FastTDA_std"] = float(np.std(v))
    for col in ("VR", "Alpha"):
        if col in df:
            out[f"{col}_mean"] = float(np.nanmean(df[col]))
    if "Position" in df:
        p = df.loc[ok, "Position"].to_numpy(float)
        out["Abs_Corr_Position"] = (float(abs(np.corrcoef(v, p)[0, 1]))
                                    if np.std(p) > 0 and np.std(v) > 0 else np.nan)
    return out


def add_scores(summary: pd.DataFrame) -> pd.DataFrame:
    """Score = weighted mean rank of reliability + time resolution (lower is better)."""
    s = summary.copy()
    ranks = [s["Invalid_pct"].rank(), s["Spike_pct"].rank(), s["Roughness"].rank()]
    weights = [1.0, 1.0, 1.0]
    if "Abs_Corr_Position" in s:
        ranks.append(s["Abs_Corr_Position"].rank(ascending=False))
        weights.append(1.0)
    if RESOLUTION_WEIGHT > 0:
        ranks.append(s["Window_s"].rank())
        weights.append(RESOLUTION_WEIGHT)
    r = pd.concat(ranks, axis=1).to_numpy(float)
    s["Score"] = (r * np.array(weights)).sum(axis=1) / sum(weights)
    s = s.sort_values(["Score", "Invalid_pct"]).reset_index(drop=True)
    s.insert(0, "Rank", np.arange(1, len(s) + 1))
    return s


def run_settings(settings, stage, t, x, pos, fs, step, stride, vr_model, done):
    """Run (window_samples, delay_samples) settings that have not been run yet."""
    rows = []
    todo = [(w, d) for w, d in settings if (w, d) not in done]
    for i, (w, d) in enumerate(todo, 1):
        name = f"W{w}_T{d}"
        n_points = (w - d * (DIM - 1) - 1) // stride + 1
        if n_points < 5 or w > len(x) or d > MAX_DELAY_FRACTION * w:
            done[(w, d)] = None
            continue
        t0 = time.time()
        df = extract_profile(t, x, pos, w, d, step, stride, vr_model)
        m = quality_metrics(df)
        df = clean_fasttda(df, fs, step)
        c = clean_report(df)
        done[(w, d)] = df
        rows.append({
            "Profile": name, "Stage": stage,
            "Window_Size": w, "Window_s": w / fs, "Takens_Delay": d, "Delay_s": d / fs,
            "Points_per_Window": n_points, "Windows": len(df),
            "FastTDA_Invalid_Count": int((~df["FastTDA_FitOK"]).sum()),
            **m, **c,
            "Signal_Column": SIGNAL_COL, "Sampling_Hz": fs, "Lowpass_Hz": CUTOFF,
            "Window_Step": step, "Takens_Stride": stride,
            "Runtime_s": time.time() - t0,
        })
        print(f"  [{stage} {i}/{len(todo)}] {name:10s} window {w / fs:5.2f} s, delay {d / fs:.3f} s | "
              f"invalid {m['Invalid_pct']:5.2f}%  spikes {m['Spike_pct']:5.2f}%  "
              f"roughness {m['Roughness']:.3f}")
        if stage == "final":
            print(f"  Cleaning: {c['Clean_Startup']} start-up, {c['Clean_FitFailed']} failed fits, "
                  f"{c['Clean_GapEdge']} gap edges, {c['Clean_Spike']} spikes masked -> "
                  f"{c['Clean_Filled']} filled by interpolation, {c['Clean_NaN']} left empty (NaN)")
    return rows


def main():
    input_path = to_local_path(CSV_PATH)
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Could not find the trial CSV:\n{input_path}")
    tag = os.path.splitext(os.path.basename(input_path))[0]
    print("=" * 80)
    print(f"FastTDA fine-tuning ({MODE}): {input_path}")
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
    done = {}

    if MODE == "fixed":
        w, d = int(FIXED_WINDOW_SAMPLES), int(FIXED_DELAY_SAMPLES)
        print(f"\nFixed setting: W{w}_T{d} (window {w / fs:.2f} s, delay {d / fs:.3f} s)")
        rows = run_settings([(w, d)], "final", t, x, pos, fs, step, stride, vr_model, done)
        if not rows:
            raise RuntimeError(f"W{w}_T{d} is not a valid setting for this signal "
                               f"(too few embedded points or delay > {MAX_DELAY_FRACTION} x window).")
    else:
        # ---- Stage 1: coarse grid
        coarse = sorted({(int(round(w * fs)), max(1, int(round(d * fs))))
                         for w in WINDOW_S for d in DELAY_S})
        print(f"\nStage 1: coarse grid, {len(coarse)} settings")
        rows = run_settings(coarse, "coarse", t, x, pos, fs, step, stride, vr_model, done)
        if not rows:
            raise RuntimeError("No valid settings in the coarse grid; check WINDOW_S / DELAY_S.")

    # ---- Stage 2: refine around the best coarse setting
    if MODE == "auto":
        best = add_scores(pd.DataFrame(rows)).iloc[0]
        bw, bd = int(best["Window_Size"]), int(best["Takens_Delay"])
        print(f"\nBest coarse setting: W{bw}_T{bd} (window {bw / fs:.2f} s, delay {bd / fs:.3f} s)")
        on_edge = []
        if best["Window_s"] >= max(WINDOW_S) * 0.999 or best["Window_s"] <= min(WINDOW_S) * 1.001:
            on_edge.append("window")
        if best["Delay_s"] >= max(DELAY_S) * 0.97 or best["Delay_s"] <= min(DELAY_S) * 1.03:
            on_edge.append("delay")
        if on_edge:
            print(f"  Note: best {' and '.join(on_edge)} is at the edge of the coarse grid; "
                  f"consider extending WINDOW_S / DELAY_S in that direction.")
        refine = sorted({(max(3, int(round(bw * fw))), max(1, int(round(bd * fd))))
                         for fw in REFINE_WINDOW_FACTORS for fd in REFINE_DELAY_FACTORS})
        print(f"Stage 2: refine grid around it, {len(refine)} settings")
        rows += run_settings(refine, "refine", t, x, pos, fs, step, stride, vr_model, done)

    summary = add_scores(pd.DataFrame(rows))

    out_dir = to_local_path(OUTPUT_DIR) if OUTPUT_DIR else os.path.dirname(input_path)
    os.makedirs(out_dir, exist_ok=True)
    suffix = "final" if MODE == "fixed" else "FineTune"
    out_path = os.path.join(out_dir, f"{tag}_FastTDA_{suffix}.xlsx")
    top = summary.head(TOP_N_SHEETS)

    def write_excel(path):
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            summary.to_excel(writer, sheet_name="Summary", index=False)
            for _, r in top.iterrows():
                done[(int(r["Window_Size"]), int(r["Takens_Delay"]))].to_excel(
                    writer, sheet_name=r["Profile"][:31], index=False)

    try:
        write_excel(out_path)
    except PermissionError:
        # Usually the file is open in Excel (Windows locks it) - save under a new name instead
        locked = out_path
        out_path = os.path.join(out_dir, f"{tag}_FastTDA_{suffix}_{time.strftime('%Y%m%d_%H%M%S')}.xlsx")
        print(f"\nCould not overwrite {locked} (is it open in Excel?). Saving to a new file instead.")
        write_excel(out_path)

    cols = ["Rank", "Profile", "Stage", "Window_s", "Delay_s", "Invalid_pct", "Spike_pct",
            "Roughness", "FastTDA_median", "Score"]
    if "Abs_Corr_Position" in summary:
        cols.insert(-1, "Abs_Corr_Position")
    print("\nTop settings (lower Score = more reliable FastTDA output):")
    print(summary[cols].head(10).round(4).to_string(index=False))
    b = summary.iloc[0]
    print(f"\nRecommended: {b['Profile']}  ->  window {b['Window_s']:.2f} s ({int(b['Window_Size'])} samples), "
          f"delay {b['Delay_s']:.3f} s ({int(b['Takens_Delay'])} samples)")
    print(f"Saved: {out_path}  (Summary of all {len(summary)} settings + sheets for the top {len(top)})")
    return summary


if __name__ == "__main__":
    main()
