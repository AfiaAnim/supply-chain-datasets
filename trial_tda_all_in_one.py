"""
All-in-one TDA pipeline for a capacitance trial (e.g. trial_01.csv). Paste into ONE Jupyter cell.

    1. Load the CSV (time + capacitance), optional low-pass filter.
    2. Fine-tune window length, Takens delay and per-window detrending with FastTDA (MODE = "auto"):
         coarse grid over WINDOW_S x DELAY_S, then a refine grid around the best setting.
         Each setting is scored only from where FastTDA_FitOK first turns True and stays True
         for FIT_OK_RUN_S seconds (skips the start-up), on:
            Invalid_pct (failed fits), Spike_pct, Roughness, Window_s (shorter = better)
         Score = weighted mean rank (lower = better); Rank 1 is used.
       MODE = "fixed" skips tuning and uses FIXED_WINDOW / FIXED_DELAY.
    3. Extract the chosen profile with VR, Alpha and FastTDA (parity-file layout, 9 columns).
    4. Clean + smooth VR, Alpha and FastTDA (start-up trim, invalid windows + neighbours and
       spikes masked, short gaps interpolated, rolling median).
    5. Save one workbook and show the figures.

Several trials: list them in CSV_PATHS. TUNE_MODE = "common" tunes one setting on all trials
together (metrics averaged; same window/delay/detrend for every trial, so they are comparable),
"per_trial" tunes each trial on its own. An All_Trials_TDA_Summary.xlsx compares the trials.

Output (next to each CSV):
    <trial>_TDA_Results.xlsx
        Tuning_Summary   every setting tried, ranked (only in MODE = "auto")
        W<window>        Time, <label>, VR, Alpha, FastTDA, FastTDA_ZeroInvalid,
                         FastTDA_FitOK, Window_Number, Window_Size
        Metadata         Profile, Dataset, Sampling_Hz, Lowpass_Hz, Filter_Order, Window_Size,
                         Window_Step, Takens_Delay, Takens_Dimension, Takens_Stride,
                         Time_Label, Rows, FastTDA_Invalid_Count, Note
        W<window>_Cleaned  Time, then for VR / Alpha / FastTDA: raw, _clean, _smooth, _Mask
    <trial>_tuning.png, <trial>_W<window>_profile.png

Requirements:
    pip install numpy pandas scipy matplotlib openpyxl giotto-tda gudhi
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
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

# One or more trials. All trials are processed; each gets its own workbook and figures.
CSV_PATHS = [
    r"C:\Users\aphya\Downloads\trial_01.csv",
    r"C:\Users\aphya\Downloads\trial_02.csv",
    r"C:\Users\aphya\Downloads\trial_03.csv",
]
TUNE_MODE = "common"                    # "common" = one setting tuned on all trials together
                                        #            (same window/delay -> trials are comparable)
                                        # "per_trial" = tune each trial separately
TIME_COL = "time_s"
SIGNAL_COL = "capacitance_filtered_pF"  # signal used for the TDA features
LABEL_COL = "capacitance_filtered_pF"   # 2nd column of the profile sheet
LABEL_NAME = "Capacitance_pF"           # its name in the sheet (e.g. "CartLocation")
LOWPASS_HZ = None                       # extra low-pass in Hz (None = none; must be < fs/2)
FILTER_ORDER = 5

MODE = "auto"                           # "auto" = fine-tune first; "fixed" = use FIXED_* below
FIXED_WINDOW = 159                      # samples (used when MODE = "fixed")
FIXED_DELAY = 8                         # samples
FIXED_DETREND = False                   # remove each window's linear trend (used when MODE = "fixed")

# Fine-tuning grid (seconds) and refine multipliers
WINDOW_S = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
DELAY_S = [0.03, 0.06, 0.09, 0.12, 0.16, 0.20, 0.25]
REFINE_WINDOW_FACTORS = [0.8, 0.9, 1.0, 1.1, 1.2]
REFINE_DELAY_FACTORS = [0.75, 0.875, 1.0, 1.125, 1.25]
DETREND_OPTIONS = [False, True]         # fine-tuning also tries removing each window's linear trend
                                        # (the slow capacitance drift can stop the ellipse fit)
MAX_DELAY_FRACTION = 0.25               # skip settings whose delay > this fraction of the window
RESOLUTION_WEIGHT = 1.0                 # weight of "shorter window" in the score (0 = ignore)
MIN_CYCLES_PER_WINDOW = 1.0             # a window must hold at least this many cycles of the
                                        # dominant oscillation (else the TDA loop is only a partial
                                        # arc); 0 = no limit. Dominant frequency is found per trial.
F_DOM_MIN_HZ = 0.3                      # dominant oscillation is searched above this frequency
TUNE_FROM_FIRST_FIT_OK = True           # score only from where FastTDA_FitOK first turns True
FIT_OK_RUN_S = 1.0                      # ... and stays True this long (skips the start-up)

# Embedding / windows
WINDOW_STEP = 1                         # samples between windows
TAKENS_DIM = 2
TAKENS_STRIDE = 1
TIME_LABEL = "midpoint"                 # "midpoint" (as the DB6 parity file) or "endpoint"
COMPUTE_VR = True                       # needs giotto-tda
COMPUTE_ALPHA = True                    # needs gudhi

# Cleaning / smoothing of VR, Alpha and FastTDA
TRIM_START_S = 4.5                      # drop windows ENDING before this time (start-up)
EDGE_GUARD_WINDOWS = 5                  # also mask this many windows each side of an invalid one
SPIKE_MADS = 6.0                        # spike = above median + this many robust std
MAX_INTERP_GAP_S = 1.0                  # interpolate masked runs up to this long
SMOOTH_S = 0.25                         # rolling-median length (0 = off)
CLEAN_FEATURES = ["VR", "Alpha", "FastTDA"]

NOTE = ""                               # free text for the Metadata sheet


# ============================================================
# HELPERS
# ============================================================

def to_local_path(path):
    """Convert C:\\... to /mnt/c/... when running under Linux/WSL."""
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


def point_clouds(x, window, delay, detrend=False):
    starts = np.arange(0, len(x) - window + 1, WINDOW_STEP)
    idx = np.arange(window)
    clouds = []
    for s in starts:
        seg = x[s:s + window]
        if detrend:                               # remove the window's straight-line drift
            seg = seg - np.polyval(np.polyfit(idx, seg, 1), idx)
        clouds.append(takens_points(minmax_minus1_plus1(seg), delay, TAKENS_DIM, TAKENS_STRIDE))
    return starts, np.stack(clouds)


def profile_name(window, delay, detrend):
    return f"W{window}_T{delay}" + ("_DT" if detrend else "")


# ============================================================
# FEATURES: FastTDA, VR, Alpha
# ============================================================

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


# ============================================================
# 1) FINE-TUNING (FastTDA only)
# ============================================================

def first_fit_ok_index(fit_ok, run):
    """Index of the first window that starts `run` consecutive FastTDA_FitOK = True windows."""
    fit_ok = np.asarray(fit_ok, bool)
    run = max(1, min(int(run), len(fit_ok)))
    full = np.flatnonzero(np.convolve(fit_ok.astype(int), np.ones(run, int), mode="valid") == run)
    return int(full[0]) if full.size else len(fit_ok)


def score_setting(x, t, fs, window, delay, detrend):
    starts, clouds = point_clouds(x, window, delay, detrend)
    res = [fasttda_minor_axis(pc) for pc in clouds]
    v = np.array([r[0] for r in res], dtype=float)
    ok = np.array([r[1] for r in res], dtype=bool)
    t_end = t[starts + window - 1]
    i0 = first_fit_ok_index(ok, round(FIT_OK_RUN_S * fs / WINDOW_STEP)) if TUNE_FROM_FIRST_FIT_OK else 0
    v, ok = v[i0:], ok[i0:]
    out = {"Profile": profile_name(window, delay, detrend), "Window_Size": window, "Window_s": window / fs,
           "Takens_Delay": delay, "Delay_s": delay / fs, "Detrend": bool(detrend), "Windows": len(starts),
           "Tune_From_s": float(t_end[i0]) if i0 < len(t_end) else np.nan,
           "Skipped_Start_s": float(t_end[min(i0, len(t_end) - 1)] - t_end[0]),
           "Invalid_pct": 100.0 * (1 - ok.mean()) if ok.size else 100.0}
    v = v[ok]
    if v.size < 10:
        return {**out, "Spike_pct": np.nan, "Roughness": np.nan, "FastTDA_median": np.nan}
    med = np.median(v)
    mad = 1.4826 * np.median(np.abs(v - med))
    iqr = np.subtract(*np.percentile(v, [75, 25]))
    out["Spike_pct"] = 100.0 * np.mean(v > med + SPIKE_MADS * mad) if mad > 0 else 0.0
    out["Roughness"] = float(np.median(np.abs(np.diff(v))) / iqr) if iqr > 0 else np.nan
    out["FastTDA_median"] = float(med)
    return out


def add_scores(summary):
    s = summary.copy()
    ranks = [s["Invalid_pct"].rank(), s["Spike_pct"].rank(), s["Roughness"].rank()]
    weights = [1.0, 1.0, 1.0]
    if RESOLUTION_WEIGHT > 0:
        ranks.append(s["Window_s"].rank())
        weights.append(RESOLUTION_WEIGHT)
    r = pd.concat(ranks, axis=1).to_numpy(float)
    s["Score"] = (r * np.array(weights)).sum(axis=1) / sum(weights)
    s = s.sort_values(["Score", "Invalid_pct"]).reset_index(drop=True)
    s.insert(0, "Rank", np.arange(1, len(s) + 1))
    return s


def run_grid(settings, stage, trials, done, min_cycles=None):
    """Score each setting on every trial; the tuning metrics are averaged over the trials."""
    rows, detail = [], []
    min_cycles = MIN_CYCLES_PER_WINDOW if min_cycles is None else min_cycles
    todo = [(w, d, dt) for w, d, dt in settings
            if (w, d, dt) not in done and d <= MAX_DELAY_FRACTION * w
            and all(w <= len(tr["x"]) for tr in trials)
            and all(w / tr["fs"] * tr["f_dom"] >= min_cycles for tr in trials)
            and (w - d * (TAKENS_DIM - 1) - 1) // TAKENS_STRIDE + 1 >= 5]
    fs = np.mean([tr["fs"] for tr in trials])
    for i, (w, d, dt) in enumerate(todo, 1):
        per = []
        for tr in trials:
            r = score_setting(tr["x"], tr["t"], tr["fs"], w, d, dt)
            r.update({"Trial": tr["name"], "Stage": stage})
            per.append(r)
        detail += per
        p = pd.DataFrame(per)
        r = {k: per[0][k] for k in ("Profile", "Window_Size", "Takens_Delay", "Detrend")}
        r.update({"Window_s": w / fs, "Delay_s": d / fs, "Stage": stage, "Trials": len(per),
                  "Min_Cycles_per_Window": min(w / tr["fs"] * tr["f_dom"] for tr in trials),
                  "Tune_From_s": p["Tune_From_s"].mean(), "Invalid_pct": p["Invalid_pct"].mean(),
                  "Worst_Invalid_pct": p["Invalid_pct"].max(), "Spike_pct": p["Spike_pct"].mean(),
                  "Roughness": p["Roughness"].mean(), "FastTDA_median": p["FastTDA_median"].mean()})
        done.add((w, d, dt))
        rows.append(r)
        print(f"  [{stage} {i}/{len(todo)}] {r['Profile']}: window {w / fs:5.2f} s, delay {d / fs:.3f} s | "
              f"invalid {r['Invalid_pct']:5.2f}% (worst {r['Worst_Invalid_pct']:5.2f}%)  "
              f"spikes {r['Spike_pct']:5.2f}%  roughness {r['Roughness']:.3f}")
    return rows, detail


def fine_tune(trials):
    """Coarse grid + refine around the best; returns (ranked summary, per-trial details)."""
    fs = np.mean([tr["fs"] for tr in trials])
    names = ", ".join(tr["name"] for tr in trials)
    done = set()
    coarse = sorted({(int(round(w * fs)), max(1, int(round(d * fs))), bool(dt))
                     for w in WINDOW_S for d in DELAY_S for dt in DETREND_OPTIONS})
    print(f"\nFine-tuning on {names}")
    if MIN_CYCLES_PER_WINDOW:
        min_w = max(MIN_CYCLES_PER_WINDOW / tr["f_dom"] for tr in trials)
        print(f"Windows shorter than {min_w:.2f} s ({MIN_CYCLES_PER_WINDOW:g} cycle of the slowest "
              f"dominant oscillation) are skipped")
    print(f"Stage 1: coarse grid, {len(coarse)} settings")
    min_cycles = MIN_CYCLES_PER_WINDOW
    rows, detail = run_grid(coarse, "coarse", trials, done, min_cycles)
    if not rows:
        min_cycles = 0
        print(f"  WARNING: no window in WINDOW_S (max {max(WINDOW_S)} s) holds {MIN_CYCLES_PER_WINDOW:g} "
              f"cycle(s); dominant frequencies found: "
              + ", ".join(f"{tr['name']} {tr['f_dom']:.2f} Hz" for tr in trials)
              + ". Tuning without the cycle limit - check these frequencies or add longer windows.")
        rows, detail = run_grid(coarse, "coarse", trials, done, min_cycles)
    best = add_scores(pd.DataFrame(rows)).iloc[0]
    bw, bd, bdt = int(best["Window_Size"]), int(best["Takens_Delay"]), bool(best["Detrend"])
    refine = sorted({(max(3, int(round(bw * fw))), max(1, int(round(bd * fd))), bdt)
                     for fw in REFINE_WINDOW_FACTORS for fd in REFINE_DELAY_FACTORS})
    n_new = len([r for r in refine if r not in done])
    print(f"\nStage 2: refine around {best['Profile']}, {n_new} new settings")
    r2, d2 = run_grid(refine, "refine", trials, done, min_cycles)
    return add_scores(pd.DataFrame(rows + r2)), pd.DataFrame(detail + d2)


# ============================================================
# 2) PROFILE (VR, Alpha, FastTDA) + 3) CLEANING
# ============================================================

def extract_profile(t, x, label, window, delay, detrend):
    starts, clouds = point_clouds(x, window, delay, detrend)
    lab_idx = starts + (window // 2 if TIME_LABEL == "midpoint" else window - 1)
    t0 = time.time()
    ftda = [fasttda_minor_axis(pc) for pc in clouds]
    vr = vr_max_h1_batch(clouds) if COMPUTE_VR else np.full(len(starts), np.nan)
    alpha = np.array([alpha_max_h1(pc) for pc in clouds]) if COMPUTE_ALPHA else np.full(len(starts), np.nan)
    print(f"    VR / Alpha / FastTDA for {len(starts)} windows: {time.time() - t0:.1f} s")
    f = np.array([v for v, _ in ftda], dtype=float)
    df = pd.DataFrame({
        "Time": t[lab_idx],
        LABEL_NAME: label[lab_idx],
        "VR": vr,
        "Alpha": alpha,
        "FastTDA": f,
        "FastTDA_ZeroInvalid": np.where(np.isfinite(f), f, 0.0),
        "FastTDA_FitOK": [ok for _, ok in ftda],
        "Window_Number": np.arange(len(starts)),
        "Window_Size": window,
    })
    return df, t[starts + window - 1]


def clean_feature(df, col, t_end, fs):
    """Mask start-up, invalid windows (+ neighbours) and high spikes, interpolate short gaps,
    rolling median. Returns (clean, mask, smooth)."""
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
            continue                                    # feature not computed
        clean, mask, smooth = clean_feature(df, col, t_end, fs)
        out.update({col: df[col].to_numpy(float), f"{col}_clean": clean,
                    f"{col}_smooth": smooth, f"{col}_Mask": mask})
        counts = pd.Series(mask).value_counts().to_dict()
        print(f"    {col:8s} cleaned: " + ", ".join(f"{k} {n}" for k, n in counts.items()))
    return pd.DataFrame(out)


# ============================================================
# FIGURES
# ============================================================

def plot_tuning(summary, dataset, out_dir):
    s = summary[summary["Stage"] == "coarse"]
    if s.empty:
        return
    # with several detrend options, each cell shows the better of them
    piv = s.pivot_table(index="Window_s", columns="Delay_s", values="Score", aggfunc="min")
    inv = s.pivot_table(index="Window_s", columns="Delay_s", values="Invalid_pct", aggfunc="min")
    cmap = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#6da7ec", "#256abf", "#0d366b"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.45 * len(piv) + 2.6))
    for ax, m, title, fmt in ((axes[0], inv, "Invalid FastTDA fits (%)", "{:.1f}"),
                              (axes[1], piv, "Tuning score (lower = better)", "{:.1f}")):
        a = m.to_numpy(float)
        im = ax.imshow(a, cmap=cmap, aspect="auto")
        ax.set_xticks(range(a.shape[1]), [f"{c:.3f}" for c in m.columns])
        ax.set_yticks(range(a.shape[0]), [f"{r:.2f}" for r in m.index])
        ax.set_xlabel("Takens delay (s)")
        ax.set_ylabel("Window length (s)")
        ax.set_title(title, fontweight="bold")
        lo, hi = np.nanmin(a), np.nanmax(a)
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                if np.isfinite(a[i, j]):
                    dark = (a[i, j] - lo) / (hi - lo + 1e-12) > 0.55
                    ax.text(j, i, fmt.format(a[i, j]), ha="center", va="center", fontsize=8,
                            color="white" if dark else "#0b0b0b")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"{dataset}: coarse fine-tuning grid", fontweight="bold")
    fig.tight_layout()
    png = os.path.join(out_dir, f"{dataset.replace(' ', '_')}_tuning.png")
    fig.savefig(png, dpi=150)
    print(f"Saved: {png}")
    plt.show()


def plot_profile(df, cdf, meta, dataset, out_dir):
    cols = [LABEL_NAME, "VR", "Alpha", "FastTDA"]
    colors = ["#52514e", "#2a78d6", "#1baf7a", "#eb6834"]
    fig, axes = plt.subplots(len(cols), 1, figsize=(11, 8), sharex=True)
    for ax, col, colr in zip(axes, cols, colors):
        y = df[col].to_numpy(float)
        ax.plot(df["Time"], y, color=colr, lw=1.0, label=f"{col} raw" if col != LABEL_NAME else col)
        if f"{col}_smooth" in cdf:
            ax.plot(df["Time"], cdf[f"{col}_smooth"], color="#0d366b", lw=1.4, label=f"{col}_smooth")
        if col == "FastTDA":
            fin = y[np.isfinite(y)]
            if fin.size:
                ax.set_ylim(np.nanmin(fin) * 0.9, np.percentile(fin, 99.5) * 1.1)
            bad = df[~df["FastTDA_FitOK"].astype(bool)]
            ax.plot(bad["Time"], np.full(len(bad), ax.get_ylim()[0]), "|", color="#e34948", ms=8,
                    label=f"invalid fit ({len(bad)})")
        ax.legend(loc="upper right", frameon=False, fontsize=8, ncol=3)
        ax.set_ylabel(col)
        ax.grid(color="#e4e3df", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel(f"Time (s), window {TIME_LABEL}")
    fig.suptitle(f"{dataset} {meta['Profile']}: window {meta['Window_Size']} "
                 f"({meta['Window_Size'] / meta['Sampling_Hz']:.2f} s), delay {meta['Takens_Delay']} "
                 f"({meta['Takens_Delay'] / meta['Sampling_Hz']:.3f} s)", fontweight="bold")
    fig.tight_layout()
    png = os.path.join(out_dir, f"{dataset}_{meta['Profile']}_profile.png")
    fig.savefig(png, dpi=150)
    print(f"Saved: {png}")
    plt.show()


# ============================================================
# RUN
# ============================================================

def load_trial(path):
    csv_path = to_local_path(path)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    data = pd.read_csv(csv_path, sep=None, engine="python")
    data.columns = [str(c).strip() for c in data.columns]
    data = data.loc[:, [c for c in data.columns if not c.startswith("Unnamed")]]
    data = data.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
    for c in (TIME_COL, SIGNAL_COL, LABEL_COL):
        if c not in data:
            raise KeyError(f"{csv_path}: column {c!r} not found. Columns: {list(data.columns)}")
    t = data[TIME_COL].to_numpy(float)
    x = data[SIGNAL_COL].to_numpy(float)
    fs = (len(t) - 1) / (t[-1] - t[0])
    if LOWPASS_HZ:
        if LOWPASS_HZ >= fs / 2:
            raise ValueError(f"LOWPASS_HZ must be below {fs / 2:.1f} Hz")
        x = signal.sosfiltfilt(signal.butter(FILTER_ORDER, LOWPASS_HZ, btype="lowpass", fs=fs, output="sos"), x)
    name = os.path.splitext(os.path.basename(csv_path))[0]
    # Dominant oscillation: remove the slow drift first (5 s moving average), else it wins the spectrum
    k = max(3, int(round(5 * fs)) | 1)
    x_hp = x - pd.Series(x).rolling(k, center=True, min_periods=1).mean().to_numpy()
    f, pxx = signal.welch(x_hp, fs=fs, nperseg=int(min(len(x), max(256, 20 * fs))), detrend="linear")
    band = f >= F_DOM_MIN_HZ
    f_dom = float(f[band][np.argmax(pxx[band])])
    print(f"Loaded {name}: {len(x)} samples, {fs:.2f} Hz, {t[-1] - t[0]:.1f} s, "
          f"dominant oscillation {f_dom:.2f} Hz (period {1 / f_dom:.2f} s)")
    return {"f_dom": f_dom, "name": name, "path": csv_path, "out_dir": os.path.dirname(csv_path),
            "t": t, "x": x, "label": data[LABEL_COL].to_numpy(float), "fs": fs}


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


def print_top(summary):
    cols = ["Rank", "Profile", "Stage", "Window_s", "Delay_s", "Detrend", "Min_Cycles_per_Window",
            "Tune_From_s", "Invalid_pct",
            "Worst_Invalid_pct", "Spike_pct", "Roughness", "FastTDA_median", "Score"]
    print(summary[[c for c in cols if c in summary]].head(10).round(4).to_string(index=False))


def process_trial(tr, window, delay, detrend, summary, detail, note):
    """Extract VR / Alpha / FastTDA with the chosen setting, clean, save and plot one trial."""
    fs = tr["fs"]
    print(f"\n=== {tr['name']}: {profile_name(window, delay, detrend)} "
          f"(window {window / fs:.2f} s, delay {delay / fs:.3f} s, detrend {'on' if detrend else 'off'})")
    name = f"W{window}"
    df, t_end = extract_profile(tr["t"], tr["x"], tr["label"], window, delay, detrend)
    n_bad = int((~df["FastTDA_FitOK"]).sum())
    print(f"    {len(df)} windows, {n_bad} invalid FastTDA fits")
    meta = {
        "Profile": name, "Dataset": tr["name"], "Sampling_Hz": round(fs, 4), "Lowpass_Hz": LOWPASS_HZ,
        "Filter_Order": FILTER_ORDER if LOWPASS_HZ else None, "Window_Size": window,
        "Window_Step": WINDOW_STEP, "Takens_Delay": delay, "Takens_Dimension": TAKENS_DIM,
        "Takens_Stride": TAKENS_STRIDE, "Time_Label": TIME_LABEL, "Rows": len(df),
        "FastTDA_Invalid_Count": n_bad, "Note": note, "Detrend": detrend,
    }
    cdf = clean_profile(df, t_end, fs)
    trial_detail = None
    if detail is not None and len(detail):
        trial_detail = detail[detail["Trial"] == tr["name"]]
    save_workbook(os.path.join(tr["out_dir"], f"{tr['name']}_TDA_Results.xlsx"), {
        "Tuning_Summary": summary, "Tuning_This_Trial": trial_detail,
        name: df, "Metadata": pd.DataFrame([meta]), f"{name}_Cleaned": cdf})

    stats = {"Trial": tr["name"], "Profile": profile_name(window, delay, detrend), "Windows": len(df),
             "FastTDA_Invalid_Count": n_bad, "Invalid_pct": round(100 * n_bad / len(df), 3)}
    rows = []
    for col in CLEAN_FEATURES:
        if f"{col}_smooth" in cdf:
            for kind in ("", "_clean", "_smooth"):
                v = cdf[f"{col}{kind}"]
                rows.append({"Column": f"{col}{kind}", "mean": v.mean(), "std": v.std(),
                             "min": v.min(), "max": v.max(), "empty": int(v.isna().sum())})
            stats[f"{col}_smooth_mean"] = cdf[f"{col}_smooth"].mean()
            stats[f"{col}_smooth_std"] = cdf[f"{col}_smooth"].std()
    print("    Raw vs clean vs smooth:")
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    if TUNE_MODE == "per_trial" and summary is not None:
        plot_tuning(summary, tr["name"], tr["out_dir"])
    plot_profile(df, cdf, meta, tr["name"], tr["out_dir"])
    return stats


trials = [load_trial(p) for p in CSV_PATHS]
results = []
common_summary = None
t_start = time.time()

if MODE == "fixed":
    for tr in trials:
        results.append(process_trial(tr, int(FIXED_WINDOW), int(FIXED_DELAY), bool(FIXED_DETREND),
                                     None, None, NOTE or "fixed setting"))
elif TUNE_MODE == "common":
    common_summary, detail = fine_tune(trials)
    print("\nTop 10 settings (metrics averaged over all trials):")
    print_top(common_summary)
    best = common_summary.iloc[0]
    w, d, dt = int(best["Window_Size"]), int(best["Takens_Delay"]), bool(best["Detrend"])
    note = NOTE or f"common setting tuned on {len(trials)} trials (Rank 1 of {len(common_summary)})"
    plot_tuning(common_summary, "All trials", trials[0]["out_dir"])
    for tr in trials:
        results.append(process_trial(tr, w, d, dt, common_summary, detail, note))
else:
    for tr in trials:
        summary, detail = fine_tune([tr])
        print(f"\nTop 10 settings for {tr['name']}:")
        print_top(summary)
        best = summary.iloc[0]
        results.append(process_trial(tr, int(best["Window_Size"]), int(best["Takens_Delay"]),
                                     bool(best["Detrend"]), summary, detail,
                                     NOTE or f"tuned on this trial (Rank 1 of {len(summary)})"))

overview = pd.DataFrame(results)
print(f"\n=== All trials ({time.time() - t_start:.0f} s) ===")
print(overview.round(4).to_string(index=False))
if len(trials) > 1:
    save_workbook(os.path.join(trials[0]["out_dir"], "All_Trials_TDA_Summary.xlsx"),
                  {"Trials": overview, "Tuning_Summary": common_summary})
