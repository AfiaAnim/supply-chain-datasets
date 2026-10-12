"""
Parity-profile workbook for a capacitance trial (e.g. trial_01.csv), in the same layout as
Code2_<dataset>_ParityProfiles.xlsx:

    one sheet per profile (e.g. "W159") with the 9 columns
        Time, <label>, VR, Alpha, FastTDA, FastTDA_ZeroInvalid, FastTDA_FitOK,
        Window_Number, Window_Size
    "Metadata" sheet: Profile, Dataset, Sampling_Hz, Lowpass_Hz, Filter_Order, Window_Size,
        Window_Step, Takens_Delay, Takens_Dimension, Takens_Stride, Time_Label, Rows,
        FastTDA_Invalid_Count, Note
    "<profile>_Cleaned" sheet(s): for VR, Alpha and FastTDA the raw value, <f>_clean
        (start-up trimmed, invalid windows + neighbours and spikes masked, short gaps
        interpolated), <f>_smooth (rolling median of _clean) and <f>_Mask (why masked)

Per window: min-max scale to [-1, 1] -> 2-D Takens embedding -> VR max H1 (giotto-tda),
Alpha max H1 (GUDHI, 2*sqrt(alpha) convention), FastTDA ellipse minor axis 2b.

Paste the whole file into ONE Jupyter cell and run it (or: python trial_parity_profiles.py).

Requirements:
    pip install numpy pandas scipy matplotlib openpyxl giotto-tda gudhi
"""

import os
import time

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


# ============================================================
# SETTINGS
# ============================================================

CSV_PATH = r"C:\Users\aphya\Downloads\trial_01.csv"
TIME_COL = "time_s"
SIGNAL_COL = "capacitance_filtered_pF"  # signal used for the TDA features
LABEL_COL = "capacitance_filtered_pF"   # 2nd column of each sheet (CartLocation in DROPBEAR)
LABEL_NAME = "Capacitance_pF"           # its column name in the sheet (e.g. "CartLocation")

# (window samples, delay samples); add more tuples for more sheets, like W1500 / W1612
PROFILES = [(159, 8)]

WINDOW_STEP = 1                         # samples between windows
TAKENS_DIM = 2
TAKENS_STRIDE = 1
TIME_LABEL = "midpoint"                 # "midpoint" (as DB6 parity file) or "endpoint"
LOWPASS_HZ = None                       # extra low-pass in Hz (None = none; must be < fs/2)
FILTER_ORDER = 5
COMPUTE_VR = True                       # needs giotto-tda
COMPUTE_ALPHA = True                    # needs gudhi
NOTE = "trial_01 capacitance, W159_T8 = auto-tuned setting"

# Cleaning (extra "Cleaned" sheet; the 9-column sheets stay raw like the parity file)
ADD_CLEANED_SHEET = True
TRIM_START_S = 4.5                      # drop windows ENDING before this time (start-up)
EDGE_GUARD_WINDOWS = 5
SPIKE_MADS = 6.0
MAX_INTERP_GAP_S = 1.0
SMOOTH_S = 0.25
CLEAN_FEATURES = ["VR", "Alpha", "FastTDA"]


# ============================================================
# FEATURES
# ============================================================

def to_local_path(path):
    path = str(path).strip().strip('"')
    if os.name != "nt" and len(path) > 2 and path[1] == ":" and path[2] in "\\/":
        return f"/mnt/{path[0].lower()}/{path[3:].replace(chr(92), '/')}"
    return path


def minmax_minus1_plus1(x):
    lo, hi = x.min(), x.max()
    return np.zeros_like(x) if hi == lo else 2 * (x - lo) / (hi - lo) - 1


def takens_points(x, delay, dim, stride):
    """Same indexing as gtda TakensEmbedding (aligned to the end of the window)."""
    span = len(x) - (dim - 1) * delay - 1
    n_points = span // stride + 1
    first = span % stride
    last = first + (n_points - 1) * stride
    return np.column_stack([x[first + d * delay: last + d * delay + 1: stride] for d in range(dim)])


def fasttda_minor_axis(pc):
    """SVD-LSE ellipse fit (A + C = 1) on a 2-D point cloud -> (2b, fit_ok)."""
    X, Y = pc[:, 0], pc[:, 1]
    Xc, Yc = X - X.mean(), Y - Y.mean()
    rng = max(np.abs(Xc).max(), np.abs(Yc).max())
    if not np.isfinite(rng) or rng == 0:
        return np.nan, False
    Xc, Yc = Xc * 1.5 / rng, Yc * 1.5 / rng
    G = np.column_stack([Xc**2 - Yc**2, Xc * Yc, Xc, Yc, np.ones_like(Xc)])
    U, S, Vt = np.linalg.svd(G, full_matrices=False)
    Sinv = np.where(S > np.finfo(float).eps * max(G.shape) * S[0], 1.0 / S, 0.0)
    A, B, D, E, F = (Vt.T * Sinv) @ (U.T @ -(Yc**2))
    C = 1.0 - A
    if 4 * A * C - B**2 <= 0:
        return np.nan, False
    th = 0.5 * np.arctan2(B, A - C)
    c, s = np.cos(th), np.sin(th)
    Ap, Cp = A*c*c + B*c*s + C*s*s, A*s*s - B*c*s + C*c*c
    Dp, Ep = D*c + E*s, -D*s + E*c
    if Ap <= 0 or Cp <= 0:
        return np.nan, False
    Fn = F - Dp**2 / (4 * Ap) - Ep**2 / (4 * Cp)
    if Fn >= 0:
        return np.nan, False
    return 2.0 * min(np.sqrt(-Fn / Ap), np.sqrt(-Fn / Cp)), True


def vr_max_h1_batch(clouds):
    """VR max H1 persistence for many equal-size point clouds at once (giotto-tda)."""
    from gtda.homology import VietorisRipsPersistence
    diagrams = VietorisRipsPersistence(homology_dimensions=[1], n_jobs=-1).fit_transform(clouds)
    pers = diagrams[:, :, 1] - diagrams[:, :, 0]
    pers[(diagrams[:, :, 2] != 1) | ~np.isfinite(pers)] = 0.0
    return pers.max(axis=1)


def alpha_max_h1(pc):
    """Alpha max H1 persistence (GUDHI filtration is squared radius -> 2*sqrt(alpha))."""
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


def extract_profile(t, x, label, window, delay):
    starts = np.arange(0, len(x) - window + 1, WINDOW_STEP)
    lab_idx = starts + (window // 2 if TIME_LABEL == "midpoint" else window - 1)
    clouds = np.stack([takens_points(minmax_minus1_plus1(x[s:s + window]), delay, TAKENS_DIM, TAKENS_STRIDE)
                       for s in starts])
    t0 = time.time()
    ftda = [fasttda_minor_axis(pc) for pc in clouds]
    vr = vr_max_h1_batch(clouds) if COMPUTE_VR else np.full(len(starts), np.nan)
    print(f"    FastTDA + VR: {time.time() - t0:.1f} s")
    t0 = time.time()
    alpha = np.array([alpha_max_h1(pc) for pc in clouds]) if COMPUTE_ALPHA else np.full(len(starts), np.nan)
    if COMPUTE_ALPHA:
        print(f"    Alpha: {time.time() - t0:.1f} s")
    f = np.array([v for v, _ in ftda], dtype=float)
    return pd.DataFrame({
        "Time": t[lab_idx],
        LABEL_NAME: label[lab_idx],
        "VR": vr,
        "Alpha": alpha,
        "FastTDA": f,
        "FastTDA_ZeroInvalid": np.where(np.isfinite(f), f, 0.0),
        "FastTDA_FitOK": [ok for _, ok in ftda],
        "Window_Number": np.arange(len(starts)),
        "Window_Size": window,
    }), t[starts + window - 1]


def clean_feature(df, col, t_end, fs):
    """Clean one feature column: mask start-up, invalid windows (+ neighbours) and high spikes,
    interpolate short masked runs, then a rolling median. Returns (clean, mask, smooth)."""
    v = df[col].to_numpy(float)
    ok = np.isfinite(v)
    if col == "FastTDA":
        ok &= df["FastTDA_FitOK"].to_numpy(bool)
    reason = np.full(len(df), "ok", dtype=object)
    reason[~ok] = "invalid"
    near = np.convolve((~ok).astype(int), np.ones(2 * EDGE_GUARD_WINDOWS + 1, int), mode="same") > 0
    reason[near & ok] = "gap_edge"
    reason[t_end < TRIM_START_S] = "startup"
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
    n = max(1, int(round(SMOOTH_S * fs / WINDOW_STEP)) | 1)
    smooth = pd.Series(clean).rolling(n, center=True, min_periods=n // 2 + 1).median().to_numpy()
    return clean, reason, smooth


def clean_profile(df, t_end, fs):
    """Cleaned sheet: <feature>, <feature>_clean, <feature>_smooth, <feature>_Mask for VR, Alpha, FastTDA."""
    out = {"Time": df["Time"].to_numpy(float)}
    for col in CLEAN_FEATURES:
        if not np.isfinite(df[col]).any():
            continue                                    # feature not computed
        clean, mask, smooth = clean_feature(df, col, t_end, fs)
        out.update({col: df[col].to_numpy(float), f"{col}_clean": clean,
                    f"{col}_smooth": smooth, f"{col}_Mask": mask})
        counts = pd.Series(mask).value_counts().to_dict()
        print(f"    {col:8s} cleaned: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    return pd.DataFrame(out)


# ============================================================
# RUN
# ============================================================

csv_path = to_local_path(CSV_PATH)
dataset = os.path.splitext(os.path.basename(csv_path))[0]
print(f"Loading {csv_path}")
data = pd.read_csv(csv_path)
data = data.loc[:, [c for c in data.columns if not str(c).startswith("Unnamed")]]
data = data.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
t = data[TIME_COL].to_numpy(float)
x = data[SIGNAL_COL].to_numpy(float)
label = data[LABEL_COL].to_numpy(float)
fs = (len(t) - 1) / (t[-1] - t[0])
print(f"{len(x)} samples, {fs:.2f} Hz, {t[-1] - t[0]:.1f} s")
if LOWPASS_HZ:
    sos = signal.butter(FILTER_ORDER, LOWPASS_HZ, btype="lowpass", fs=fs, output="sos")
    x = signal.sosfiltfilt(sos, x)

sheets, cleaned, meta = {}, {}, []
names = [f"W{w}" for w, _ in PROFILES]
for (w, d), name in zip(PROFILES, names):
    if names.count(name) > 1:
        name = f"W{w}_T{d}"
    print(f"\nProfile {name}: window {w} samples ({w / fs:.2f} s), delay {d} ({d / fs:.3f} s)")
    df, t_end = extract_profile(t, x, label, w, d)
    n_bad = int((~df["FastTDA_FitOK"]).sum())
    print(f"    {len(df)} windows, {n_bad} invalid FastTDA fits")
    sheets[name] = df
    if ADD_CLEANED_SHEET:
        cleaned[name] = clean_profile(df, t_end, fs)
    meta.append({
        "Profile": name, "Dataset": dataset, "Sampling_Hz": round(fs, 4), "Lowpass_Hz": LOWPASS_HZ,
        "Filter_Order": FILTER_ORDER if LOWPASS_HZ else None, "Window_Size": w,
        "Window_Step": WINDOW_STEP, "Takens_Delay": d, "Takens_Dimension": TAKENS_DIM,
        "Takens_Stride": TAKENS_STRIDE, "Time_Label": TIME_LABEL, "Rows": len(df),
        "FastTDA_Invalid_Count": n_bad, "Note": NOTE,
    })

# ---- save
out_dir = os.path.dirname(csv_path)
xlsx = os.path.join(out_dir, f"{dataset}_ParityProfiles.xlsx")


def write(path):
    with pd.ExcelWriter(path, engine="openpyxl") as wr:
        for name, df in sheets.items():
            df.to_excel(wr, sheet_name=name[:31], index=False)
        pd.DataFrame(meta).to_excel(wr, sheet_name="Metadata", index=False)
        for name, df in cleaned.items():
            df.to_excel(wr, sheet_name=f"{name}_Cleaned"[:31], index=False)


try:
    write(xlsx)
except PermissionError:
    xlsx = os.path.join(out_dir, f"{dataset}_ParityProfiles_{time.strftime('%H%M%S')}.xlsx")
    print("Workbook is open in Excel; saving under a new name.")
    write(xlsx)
print(f"\nSaved: {xlsx}")

for name, df in sheets.items():
    print(f"\n{name} - first rows:")
    print(df.head().to_string(index=False))
    print(df.describe().T[["mean", "std", "min", "max"]].round(4).to_string())
print("\nMetadata:")
print(pd.DataFrame(meta).T.to_string(header=False))
for name, cdf in cleaned.items():
    rows = []
    for col in CLEAN_FEATURES:
        if f"{col}_smooth" in cdf:
            for kind in ("", "_clean", "_smooth"):
                v = cdf[f"{col}{kind}"]
                rows.append({"Column": f"{col}{kind}", "mean": v.mean(), "std": v.std(),
                             "min": v.min(), "max": v.max(), "empty": int(v.isna().sum())})
    print(f"\n{name}_Cleaned - raw vs clean vs smooth:")
    print(pd.DataFrame(rows).round(4).to_string(index=False))

# ---- plot: the feature columns of each profile over time
for name, df in sheets.items():
    cols = [LABEL_NAME, "VR", "Alpha", "FastTDA"]
    colors = ["#52514e", "#2a78d6", "#1baf7a", "#eb6834"]
    fig, axes = plt.subplots(len(cols), 1, figsize=(11, 8), sharex=True)
    for ax, col, colr in zip(axes, cols, colors):
        y = df[col].to_numpy(float)
        ax.plot(df["Time"], y, color=colr, lw=1.0)
        if name in cleaned and f"{col}_smooth" in cleaned[name]:
            ax.plot(df["Time"], cleaned[name][f"{col}_smooth"], color="#0d366b", lw=1.4,
                    label=f"{col}_smooth (Cleaned sheet)")
            ax.legend(loc="upper right", frameon=False, fontsize=8)
        if col == "FastTDA":
            fin = y[np.isfinite(y)]
            if fin.size:
                ax.set_ylim(np.nanmin(fin) * 0.9, np.percentile(fin, 99.5) * 1.1)
            bad = df[~df["FastTDA_FitOK"].astype(bool)]
            ax.plot(bad["Time"], np.full(len(bad), ax.get_ylim()[0]), "|", color="#e34948", ms=8)
        ax.set_ylabel(col)
        ax.grid(color="#e4e3df", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel(f"Time (s), window {TIME_LABEL}")
    m = next(r for r in meta if r["Profile"] == name)
    fig.suptitle(f"{dataset} {name}: window {m['Window_Size']}, delay {m['Takens_Delay']}, "
                 f"{m['Rows']} windows, {m['FastTDA_Invalid_Count']} invalid fits (red ticks)",
                 fontweight="bold")
    fig.tight_layout()
    png = os.path.join(out_dir, f"{dataset}_{name}_ParityProfile.png")
    fig.savefig(png, dpi=150)
    print(f"Saved: {png}")
    plt.show()
