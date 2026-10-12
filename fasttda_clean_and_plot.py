"""
One-cell FastTDA run for trial_01.csv: W159_T8 extraction -> cleaning -> Excel -> plot.

Paste the whole file into ONE Jupyter cell and run it (or: python fasttda_clean_and_plot.py).
It does not depend on any other script or on the earlier Excel files.

Output (next to the CSV):
    trial_01_FastTDA_clean.xlsx   - Time, FastTDA (raw), FastTDA_clean, Clean_Mask, FastTDA_smooth
    trial_01_FastTDA_clean.png    - signal + raw/clean FastTDA
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
SIGNAL_COL = "capacitance_filtered_pF"
RAW_COL = "capacitance_raw_pF"          # shown in the top plot (None to hide)

WINDOW = 159                            # samples (1.80 s at 88.4 Hz)
DELAY = 8                               # samples (0.091 s)

TRIM_START_S = 4.5                      # drop windows ending before this time (start-up)
EDGE_GUARD_WINDOWS = 5                  # also mask this many windows each side of a failed fit
SPIKE_MADS = 6.0                        # mask values above median + this many robust std
MAX_INTERP_GAP_S = 1.0                  # interpolate masked runs up to this long
SMOOTH_S = 0.25                         # rolling-median length for FastTDA_smooth (0 = off)


# ============================================================
# FASTTDA
# ============================================================

def to_local_path(path):
    path = str(path).strip().strip('"')
    if os.name != "nt" and len(path) > 2 and path[1] == ":" and path[2] in "\\/":
        return f"/mnt/{path[0].lower()}/{path[3:].replace(chr(92), '/')}"
    return path


def fasttda_minor_axis(x, delay):
    """Min-max scale to [-1, 1], 2-D Takens embedding, SVD-LSE ellipse fit (A + C = 1) -> 2b."""
    lo, hi = x.min(), x.max()
    if hi == lo:
        return np.nan
    x = 2 * (x - lo) / (hi - lo) - 1
    X, Y = x[:-delay], x[delay:]
    Xc, Yc = X - X.mean(), Y - Y.mean()
    scale = 1.5 / max(np.abs(Xc).max(), np.abs(Yc).max())
    Xc, Yc = Xc * scale, Yc * scale
    G = np.column_stack([Xc**2 - Yc**2, Xc * Yc, Xc, Yc, np.ones_like(Xc)])
    U, S, Vt = np.linalg.svd(G, full_matrices=False)
    Sinv = np.where(S > np.finfo(float).eps * max(G.shape) * S[0], 1.0 / S, 0.0)
    A, B, D, E, F = (Vt.T * Sinv) @ (U.T @ -(Yc**2))
    C = 1.0 - A
    if 4 * A * C - B**2 <= 0:
        return np.nan
    th = 0.5 * np.arctan2(B, A - C)
    c, s = np.cos(th), np.sin(th)
    Ap, Cp = A*c*c + B*c*s + C*s*s, A*s*s - B*c*s + C*c*c
    Dp, Ep = D*c + E*s, -D*s + E*c
    if Ap <= 0 or Cp <= 0:
        return np.nan
    Fn = F - Dp**2 / (4 * Ap) - Ep**2 / (4 * Cp)
    if Fn >= 0:
        return np.nan
    return 2.0 * min(np.sqrt(-Fn / Ap), np.sqrt(-Fn / Cp))


# ============================================================
# RUN
# ============================================================

csv_path = to_local_path(CSV_PATH)
print(f"Loading {csv_path}")
data = pd.read_csv(csv_path)
data = data.loc[:, [c for c in data.columns if not str(c).startswith("Unnamed")]]
data = data.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
t = data[TIME_COL].to_numpy(float)
x = data[SIGNAL_COL].to_numpy(float)
fs = (len(t) - 1) / (t[-1] - t[0])
print(f"{len(x)} samples, {fs:.2f} Hz, {t[-1] - t[0]:.1f} s")

t0 = time.time()
ends = np.arange(WINDOW, len(x) + 1)
vals = np.array([fasttda_minor_axis(x[e - WINDOW:e], DELAY) for e in ends])
df = pd.DataFrame({"Time": t[ends - 1], "FastTDA": vals, "FastTDA_FitOK": np.isfinite(vals)})
print(f"FastTDA W{WINDOW}_T{DELAY}: {len(df)} windows in {time.time() - t0:.1f} s, "
      f"{(~df.FastTDA_FitOK).sum()} failed fits ({100 * (~df.FastTDA_FitOK).mean():.2f}%)")

# ---- cleaning
ok = df["FastTDA_FitOK"].to_numpy()
v = df["FastTDA"].to_numpy()
reason = np.full(len(df), "ok", dtype=object)
reason[~ok] = "fit_failed"
near = np.convolve((~ok).astype(int), np.ones(2 * EDGE_GUARD_WINDOWS + 1, int), mode="same") > 0
reason[near & ok] = "gap_edge"
keep = reason == "ok"
med = np.median(v[keep])
mad = 1.4826 * np.median(np.abs(v[keep] - med))
reason[keep & (v > med + SPIKE_MADS * mad)] = "spike"
reason[df["Time"].to_numpy() < TRIM_START_S] = "startup"

good = reason == "ok"
clean = np.where(good, v, np.nan)
filled = np.interp(df["Time"], df["Time"][good], v[good])
max_gap = int(round(MAX_INTERP_GAP_S * fs))
edges = np.flatnonzero(np.diff(np.r_[0, (~good).astype(int), 0]))
for a, b in zip(edges[::2], edges[1::2]):
    if a > 0 and b < len(df) and b - a <= max_gap:
        clean[a:b] = filled[a:b]
clean[reason == "startup"] = np.nan
df["FastTDA_clean"] = clean
df["Clean_Mask"] = reason
# Rolling median removes the short 1-5 window jumps caused by per-window min-max rescaling
n_smooth = max(1, int(round(SMOOTH_S * fs)) | 1)
df["FastTDA_smooth"] = (df["FastTDA_clean"].rolling(n_smooth, center=True, min_periods=n_smooth // 2 + 1)
                        .median() if SMOOTH_S else df["FastTDA_clean"])

print("Clean_Mask counts:", pd.Series(reason).value_counts().to_dict())
print(f"FastTDA raw   max {np.nanmax(v):.3f}, mean {np.nanmean(v):.3f}")
print(f"FastTDA clean max {np.nanmax(clean):.3f}, mean {np.nanmean(clean):.3f}")
print(f"FastTDA smooth max {np.nanmax(df['FastTDA_smooth']):.3f}, mean {np.nanmean(df['FastTDA_smooth']):.3f}")

# ---- save
stem = os.path.splitext(csv_path)[0]
xlsx = f"{stem}_FastTDA_clean.xlsx"
try:
    df.to_excel(xlsx, index=False)
except PermissionError:
    xlsx = f"{stem}_FastTDA_clean_{time.strftime('%H%M%S')}.xlsx"
    df.to_excel(xlsx, index=False)
print(f"Saved: {xlsx}")

# ---- plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
if RAW_COL and RAW_COL in data:
    ax1.plot(t, data[RAW_COL], color="#a3a29c", lw=0.8, label=RAW_COL)
ax1.plot(t, x, color="#2a78d6", lw=1.5, label=SIGNAL_COL)
ax1.set_ylabel("Capacitance (pF)")
ax1.set_title("Capacitance signal")
ax1.legend(loc="upper right", ncol=2, frameon=False)

ax2.plot(df["Time"], v, color="#a3a29c", lw=0.8, label="FastTDA raw")
ax2.plot(df["Time"], clean, color="#9ec5f4", lw=1.0, label="FastTDA clean")
ax2.plot(df["Time"], df["FastTDA_smooth"], color="#2a78d6", lw=1.6,
         label=f"FastTDA smooth ({SMOOTH_S} s median)")
lo, hi = np.nanmin(clean), np.nanmax(clean)
ax2.set_ylim(lo - 0.15 * (hi - lo), hi + 0.1 * (hi - lo))
bad = df[~ok]
ax2.plot(bad["Time"], np.full(len(bad), lo - 0.1 * (hi - lo)), "|", color="#eb6834", ms=8,
         label=f"failed fit ({len(bad)})")
ax2.axvspan(df["Time"].iloc[0], TRIM_START_S, color="#e4e3df", lw=0, label="start-up (dropped)")
ax2.set_ylabel("FastTDA minor axis 2b")
ax2.set_xlabel("Time (s), window end")
ax2.set_title(f"FastTDA W{WINDOW}_T{DELAY}: raw (grey), clean (light blue), smooth (blue)")
ax2.legend(loc="upper right", ncol=2, frameon=False)
for ax in (ax1, ax2):
    ax.grid(color="#e4e3df", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
png = f"{stem}_FastTDA_clean.png"
fig.savefig(png, dpi=150)
print(f"Saved: {png}")
plt.show()
