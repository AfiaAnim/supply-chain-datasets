"""
SEC + MTS TDA pipeline. Paste into ONE Jupyter cell (or: python sec_mts_tda.py).

SEC  (data/processed/sec/trial_XX*.csv): time_s, C1_raw_pF, C1_clean_pF, C1_filtered_pF, dC1_pF
MTS  (data/processed/mts/trial_XX*.csv): RunningTime_s, Displacement_raw_m, Displacement_m,
                                          Force_raw_N, Force_N
The MTS measurement is the TARGET: the TDA features of the SEC signal are judged by how well
they follow the MTS displacement / force.

    1. Pair SEC and MTS files by trial name (trial_04 <-> trial_04).
    2. Time-align: SEC time is re-zeroed, then the SEC-MTS offset (about 10 s: MTS time =
       SEC time - offset) is refined by cross-correlation. Only the overlap is kept
       (about SEC 10-170 s), so the first ~10 s and last ~30 s of SEC are dropped.
    3. For every window, the target is the MTS displacement / force over the SAME window:
       its mean (level) and std (cyclic amplitude).
    4. Fine-tune window, delay and per-window detrending with FastTDA on the trials from
       TUNE_FROM_TRIAL onward (the updated setup). Score = failed fits, spikes, roughness,
       window length AND |Spearman correlation| with the target (weight TARGET_WEIGHT).
    5. Extract VR / Alpha / FastTDA with the chosen setting for every trial, clean + smooth,
       report the correlation of each feature with each target, save and plot.

Output (in OUTPUT_DIR):
    <trial>_SEC_MTS_TDA.xlsx   W<window> (9 parity columns, target = MTS at the window midpoint),
                               Targets (window mean / std of displacement and force),
                               <W>_Cleaned, Metadata, Tuning_Summary
    All_Trials_SEC_MTS_Summary.xlsx   offsets, failed fits, feature-target correlations per trial
    <trial>_SEC_MTS_profile.png, <trial>_SEC_MTS_scatter.png, SEC_MTS_tuning.png

Requirements:
    pip install numpy pandas scipy matplotlib openpyxl giotto-tda gudhi
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

DATA_DIR = r"C:\Users\aphya\Downloads\data\processed"   # folder that holds sec\ and mts\
SEC_SUBDIR, MTS_SUBDIR = "sec", "mts"
OUTPUT_DIR = None                       # None = DATA_DIR\tda_results
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

MODE = "fixed"                          # "auto" = fine-tune (~3 min); "fixed" = use FIXED_* below
# Result of the auto-tuning on trials 04-07: W768_T125 = 38.4 s window (~1 loading cycle),
# 6.25 s delay, no detrend; 0 failed fits; FastTDA vs window-mean displacement rho +0.84..+0.89
FIXED_WINDOW, FIXED_DELAY, FIXED_DETREND = 768, 125, False      # samples at RESAMPLE_HZ = 20

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
TUNE_STEP = 2                           # score every 4th window while tuning (speed); final uses all
FIT_OK_RUN_S = 1.0                      # tuning starts where FastTDA_FitOK first stays True this long

# Embedding / windows
WINDOW_STEP = 1
TAKENS_DIM = 2
TAKENS_STRIDE = 1
MAX_POINTS = 300                        # long windows: embed every k-th point so a cloud has <= this many
TIME_LABEL = "midpoint"
COMPUTE_VR = True                       # needs giotto-tda
COMPUTE_ALPHA = True                    # needs gudhi

# Cleaning / smoothing of VR, Alpha and FastTDA
TRIM_START_S = 0.0                      # extra trim after the aligned start (the SEC start-up is already cut)
EDGE_GUARD_WINDOWS = 5
SPIKE_MADS = 6.0
MAX_INTERP_GAP_S = 2.0
SMOOTH_S = 1.0
CLEAN_FEATURES = ["VR", "Alpha", "FastTDA"]


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
    from gtda.homology import VietorisRipsPersistence
    diagrams = VietorisRipsPersistence(homology_dimensions=[1], n_jobs=-1).fit_transform(clouds)
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

def find_pairs():
    base = to_local_path(DATA_DIR)
    sec_files = sorted(glob.glob(os.path.join(base, SEC_SUBDIR, "*.csv")))
    mts_files = {trial_key(p)[0]: p for p in glob.glob(os.path.join(base, MTS_SUBDIR, "*.csv"))}
    if not sec_files:
        raise FileNotFoundError(f"No SEC csv files in {os.path.join(base, SEC_SUBDIR)}")
    pairs = []
    for p in sec_files:
        name, num = trial_key(p)
        if name is None or (TRIALS and name not in TRIALS):
            continue
        if name not in mts_files:
            print(f"  {name}: no MTS file with the same trial number - skipped")
            continue
        pairs.append((name, num, p, mts_files[name]))
    print(f"Found {len(pairs)} SEC/MTS pairs: " + ", ".join(n for n, *_ in pairs))
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


def plot_profile(tr, df, cdf, meta, best_key, rhos, out_dir):
    fig, axes = plt.subplots(5, 1, figsize=(11, 10), sharex=True)
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
    for ax, col, colr in zip(axes[2:], ["VR", "Alpha", "FastTDA"], ["#2a78d6", "#1baf7a", "#eb6834"]):
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


def plot_scatter(tr, cdf, tg, best_key, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    y = tg[best_key].to_numpy(float)
    for ax, col, colr in zip(axes, ["VR", "Alpha", "FastTDA"], ["#2a78d6", "#1baf7a", "#eb6834"]):
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
    return stats


out_dir = to_local_path(OUTPUT_DIR) if OUTPUT_DIR else os.path.join(to_local_path(DATA_DIR), "tda_results")
os.makedirs(out_dir, exist_ok=True)
t_start = time.time()
print("Loading and aligning:")
trials = [load_pair(*p) for p in find_pairs()]
tune_trials = [tr for tr in trials if tr["num"] >= TUNE_FROM_TRIAL] or trials

summary = None
if MODE == "fixed":
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
save_workbook(os.path.join(out_dir, "All_Trials_SEC_MTS_Summary.xlsx"),
              {"Trials": overview, "Tuning_Summary": summary})
