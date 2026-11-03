"""
SEC + MTS: checks for a journal paper. Paste into ONE Jupyter cell and run (or: python sec_mts_journal_checks.py).

Answers the issues a reviewer will raise, using the data already recorded (MTS = target):

    J1  Is the FastTDA gain real?        paired test over the left-out trials (Wilcoxon), bootstrap
                                         confidence interval over trials, permutation null (FastTDA
                                         circularly shifted within each trial), Holm correction
    J1  Was the window cherry-picked?    the gain at every window length in GAIN_WINDOWS_S
    J1  Does a simpler feature do as well?  FastTDA vs within-window curvature of the capacitance
    J2  Sensor characterisation          sensitivity dC/dx vs displacement, usable (linear) range,
                                         saturation, hysteresis of capacitance vs force, noise and
                                         resolution in mm
    J3  Repeatability                    per loading cycle: capacitance swing, unloaded baseline
                                         (drift), peak force, hysteresis energy, snap-through position;
                                         cycle-to-cycle and trial-to-trial variation, first-cycle effect
    J4  Too few trials?                  learning curve: error vs number of training trials
    J0  Which trials are used and why    inclusion table (alignment, record length, loading cycles)

Output (in DATA_DIR/tda_results/journal):
    J1_fasttda_gain_by_window.png, J2_sensitivity_hysteresis.png, J3_repeatability.png,
    J4_learning_curve.png, SEC_MTS_journal_checks.xlsx

Requirements:
    pip install numpy pandas scipy matplotlib openpyxl scikit-learn
"""

import glob
import itertools
import os
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal, stats

try:                                    # make plots appear inside Jupyter
    from IPython import get_ipython
    if get_ipython() is not None:
        get_ipython().run_line_magic("matplotlib", "inline")
except Exception:
    pass


# ============================================================
# SETTINGS
# ============================================================

DATA_DIR = "."                          # folder with the SEC and MTS csv files (or sec\ and mts\ subfolders)
SEC_SUBDIR, MTS_SUBDIR = "sec", "mts"
OUTPUT_DIR = None                       # None = DATA_DIR/tda_results/journal
TRIALS = None                           # None = every trial found; or e.g. ["trial_04", "trial_08"]

SEC_TIME_COL = "time_s"
SEC_COL = "C1_filtered_pF"
MTS_TIME_COL = "RunningTime_s"
TARGET_COLS = ["Displacement_m", "Force_N"]
UNIT_SCALE = {"Displacement_m": (1000.0, "mm"), "Force_N": (1.0, "N")}
RESAMPLE_HZ = 20.0

OFFSET_GUESS_S = 10.0                   # SEC starts ~10 s before the MTS
OFFSET_SEARCH_S = 5.0
OFFSET_REF_COL = "Displacement_m"
MIN_OFFSET_CORR = 0.2
GOOD_OFFSET_CORR = 0.7                  # weaker alignment = trial excluded from the analysis

LOADING_PERIOD_S = 40.0                 # dominant FFT period of the loading (all trials 04-09)
SKIP_START_S = 40.0                     # first loading cycle dropped from the models (conditioning)

# J1: FastTDA gain
GAIN_WINDOWS_S = [4.0, 6.0, 8.0, 10.0, 12.0]
WINDOW_STEP_S = 1.0                     # windows slide by this much. Non-overlapping windows whose length
                                        # divides the 40 s period (4, 8, 10 s) sit at the same few phases
                                        # of every cycle, which flatters the errors; sliding windows cover
                                        # every phase. Trials stay separate, so leave-one-trial-out is fair.
DELAY_FRACTION = 0.125                  # Takens delay = this x window (as in the validation)
TAKENS_DIM = 2
MAX_POINTS = 300
N_PERMUTATIONS = 200                    # permutation null of the gain (linear models)
N_BOOTSTRAP = 2000                      # bootstrap over trials for the confidence interval
USE_RANDOM_FOREST = True                # needs scikit-learn

# J2 / J3: sensor and cycles
BIN_MM = 2.0                            # displacement bin for the calibration curves
LINEAR_FRACTION = 0.5                   # usable range: sensitivity >= this x its maximum
NOISE_SMOOTH_S = 2.0                    # noise = residual from a Savitzky-Golay fit over this span
SNAP_RANGE_MM = (-35.0, -10.0)          # look for the snap-through force dip in this range

# J4: learning curve
LEARN_WINDOW_S = 8.0

COLORS = {"FastTDA": "#eb6834", "curvature": "#2a78d6", "rf": "#0d366b", "base": "#8a8984"}
TRIAL_CMAP = plt.get_cmap("viridis")


# ============================================================
# LOADING AND ALIGNMENT (same as sec_mts_tda.py)
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


def file_kind(path):
    """'sec' or 'mts' from the csv header (file names are not reliable)."""
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
        if not name or kind is None:
            print(f"  ignored {os.path.basename(p)}: " + ("no 'trial_NN' in the name" if not name
                                                          else "columns not recognised as SEC or MTS"))
            continue
        group = sec if kind == "sec" else mts
        group.setdefault(name, (num, p))
    if not sec:
        raise FileNotFoundError(f"No SEC csv files (with a {SEC_COL!r} column) found in {os.path.abspath(base)}")
    pairs = []
    for name in sorted(sec):
        if TRIALS and name not in TRIALS:
            continue
        if name not in mts:
            print(f"  {name}: SEC file found but no MTS file - skipped")
            continue
        pairs.append((name, sec[name][0], sec[name][1], mts[name][1]))
    print(f"Found {len(pairs)} SEC/MTS pairs in {os.path.abspath(base)}")
    return pairs


def estimate_offset(t_sec, x_sec, t_mts, y_mts):
    """Offset (s) with MTS time = SEC time - offset, from the cross-correlation peak."""
    fs = 10.0
    a = np.interp(np.arange(0, t_sec[-1], 1 / fs), t_sec, x_sec)
    b = np.interp(np.arange(0, t_mts[-1], 1 / fs), t_mts, y_mts)
    best = (OFFSET_GUESS_S, 0.0)
    for lag in np.arange(OFFSET_GUESS_S - OFFSET_SEARCH_S, OFFSET_GUESS_S + OFFSET_SEARCH_S + 1e-9, 1 / fs):
        i0 = int(round(lag * fs))
        n = min(len(b), len(a) - i0)
        if i0 < 0 or n < 0.9 * len(b):
            continue
        aa, bb = a[i0:i0 + n], b[:n]
        if aa.std() == 0 or bb.std() == 0:
            continue
        r = abs(np.corrcoef(aa, bb)[0, 1])
        if r > best[1]:
            best = (float(lag), float(r))
    return best if best[1] >= MIN_OFFSET_CORR else (OFFSET_GUESS_S, best[1])


def load_pair(name, num, sec_path, mts_path):
    sec, mts = read_numeric_csv(sec_path), read_numeric_csv(mts_path)
    t_sec = sec[SEC_TIME_COL].to_numpy(float)
    t_sec = t_sec - t_sec[0]
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
    keep = (t_sec >= offset) & (t_sec <= offset + t_mts[-1])
    t_al = t_sec[keep] - offset
    targets = {c: np.interp(t_al, t_mts, mts[c].to_numpy(float)) for c in TARGET_COLS}
    on_edge = abs(abs(offset - OFFSET_GUESS_S) - OFFSET_SEARCH_S) < 0.11
    reliable = bool(np.isfinite(r) and r >= GOOD_OFFSET_CORR and not on_edge)
    print(f"  {name}: MTS {t_mts[-1]:.1f} s | offset {offset:.2f} s (|corr| {r:.2f})"
          + ("" if reliable else "  <-- alignment UNRELIABLE"))
    return {"name": name, "num": num, "t": t_al, "x": x[keep], "fs": fs, "targets": targets,
            "offset": offset, "offset_corr": r, "offset_ok": reliable, "record_s": float(t_mts[-1])}


# ============================================================
# FASTTDA AND WINDOW FEATURES (same as sec_mts_tda.py)
# ============================================================

def point_clouds(x, window, delay, step):
    """Takens point clouds of min-max scaled windows (aligned to the window end)."""
    W = sliding_window_view(x, window)[::step].astype(float)
    lo, hi = W.min(axis=1, keepdims=True), W.max(axis=1, keepdims=True)
    W = 2 * (W - lo) / np.where(hi > lo, hi - lo, 1.0) - 1
    span = window - (TAKENS_DIM - 1) * delay - 1
    stride = max(1, int(np.ceil((span + 1) / MAX_POINTS)))
    first = span % stride
    last = first + (span // stride) * stride
    cols = [W[:, first + d * delay: last + d * delay + 1: stride] for d in range(TAKENS_DIM)]
    return np.arange(0, len(x) - window + 1, step), np.stack(cols, axis=2)


def fasttda_minor_axis(pc):
    """SVD-LSE ellipse fit (A + C = 1): minor axis 2b, NaN when the fit is not an ellipse."""
    X, Y = pc[:, 0] - pc[:, 0].mean(), pc[:, 1] - pc[:, 1].mean()
    rng = max(np.abs(X).max(), np.abs(Y).max())
    if not np.isfinite(rng) or rng == 0:
        return np.nan
    X, Y = X * 1.5 / rng, Y * 1.5 / rng
    G = np.column_stack([X**2 - Y**2, X * Y, X, Y, np.ones_like(X)])
    U, S, Vt = np.linalg.svd(G, full_matrices=False)
    Sinv = np.where(S > np.finfo(float).eps * max(G.shape) * S[0], 1.0 / S, 0.0)
    A, B, D, E, F = (Vt.T * Sinv) @ (U.T @ -(Y**2))
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
    return np.nan if Fn >= 0 else 2.0 * min(np.sqrt(-Fn / Ap), np.sqrt(-Fn / Cp))


def window_table(tr, ws):
    """Windows of ws seconds sliding by WINDOW_STEP_S: capacitance level / spread / slope /
    curvature, FastTDA, baseline features and the window means of the targets."""
    w = int(round(ws * tr["fs"]))
    step = max(1, int(round(WINDOW_STEP_S * tr["fs"])))
    d = max(1, int(round(DELAY_FRACTION * w)))
    starts, clouds = point_clouds(tr["x"], w, d, step=step)
    ftda = np.array([fasttda_minor_axis(pc) for pc in clouds])
    X = sliding_window_view(tr["x"], w)[::step]
    tt = (np.arange(w) - (w - 1) / 2) / tr["fs"]                   # seconds from the window centre
    V = np.column_stack([np.ones(w), tt, tt ** 2])
    coef = np.linalg.lstsq(V, X.T, rcond=None)[0]                   # per window: level, slope, curvature
    f = pd.DataFrame({"Trial": tr["name"], "Time": tr["t"][starts + w // 2],
                      "Cap_mean": X.mean(axis=1) - np.median(X.mean(axis=1)),   # zero per trial
                      "Cap_std": X.std(axis=1), "Cap_slope": coef[1], "Cap_curv": coef[2],
                      "FastTDA_ZeroInvalid": np.where(np.isfinite(ftda), ftda, 0.0),
                      "FastTDA_FitOK": np.isfinite(ftda).astype(float)})
    for c in TARGET_COLS:
        f[f"{c}:mean"] = sliding_window_view(tr["targets"][c], w)[::step].mean(axis=1)
    f["Cap_mean2"], f["Cap_mean3"] = f["Cap_mean"] ** 2, f["Cap_mean"] ** 3
    f["Cap_dir"] = np.sign(f["Cap_slope"])
    f["Cap_mean_dir"], f["Cap_mean2_dir"] = f["Cap_mean"] * f["Cap_dir"], f["Cap_mean2"] * f["Cap_dir"]
    f["Cap_prev"] = f["Cap_mean"].shift(max(1, w // step)).fillna(f["Cap_mean"])   # the window just before
    return f[f["Time"] >= SKIP_START_S].reset_index(drop=True)


CUBIC = ["Cap_mean", "Cap_mean2", "Cap_mean3", "Cap_std"]
TWO_BRANCH = CUBIC + ["Cap_dir", "Cap_mean_dir", "Cap_mean2_dir"]
RF_COLS = ["Cap_mean", "Cap_std", "Cap_slope", "Cap_prev"]
ADDED = {"FastTDA": ["FastTDA_ZeroInvalid", "FastTDA_FitOK"], "curvature": ["Cap_curv"]}
TARGETS = [f"{c}:mean" for c in TARGET_COLS]


def have_sklearn():
    try:
        import sklearn  # noqa: F401
        return True
    except ImportError:
        return False


def base_models():
    models = {"Two-branch cubic": ("linear", TWO_BRANCH),
              "Two-branch cubic + curvature": ("linear", TWO_BRANCH + ADDED["curvature"])}
    if USE_RANDOM_FOREST and have_sklearn():
        models["Random forest"] = ("rf", RF_COLS)
    elif USE_RANDOM_FOREST:
        print("  scikit-learn not found: random-forest comparisons skipped")
    return models


def fit_predict(kind, train, test, cols, target):
    X, y, Z = train[cols].to_numpy(float), train[target].to_numpy(float), test[cols].to_numpy(float)
    if kind == "rf":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=0,
                                     n_jobs=1).fit(X, y).predict(Z)
    beta = np.linalg.lstsq(np.column_stack([np.ones(len(X)), X]), y, rcond=None)[0]
    return np.column_stack([np.ones(len(Z)), Z]) @ beta


def loto_sse(frames, target, cols, kind):
    """Leave-one-trial-out: (sum of squared errors, n) per left-out trial, in physical units."""
    scale = UNIT_SCALE[target.split(":")[0]][0]
    out = []
    for i, test in enumerate(frames):
        train = pd.concat([f for j, f in enumerate(frames) if j != i])
        e = scale * (fit_predict(kind, train, test, cols, target) - test[target].to_numpy(float))
        out.append((float(np.sum(e ** 2)), len(e)))
    return np.array(out)


def rmse(sse):
    return float(np.sqrt(sse[:, 0].sum() / sse[:, 1].sum()))


def shifted(frames, rng):
    """FastTDA circularly shifted within each trial: same values and autocorrelation, wrong times."""
    out = []
    for f in frames:
        g = f.copy()
        k = rng.integers(max(1, len(f) // 5), max(2, len(f) - len(f) // 5))
        for c in ADDED["FastTDA"]:
            g[c] = np.roll(f[c].to_numpy(), k)
        out.append(g)
    return out


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


# ============================================================
# J0: TRIAL INCLUSION
# ============================================================

def inclusion_table(trials):
    rows = []
    for tr in trials:
        d = tr["targets"]["Displacement_m"]
        peaks, _ = signal.find_peaks(-d, height=0.8 * np.max(-d), distance=int(0.7 * LOADING_PERIOD_S * tr["fs"]))
        rows.append({"Trial": tr["name"], "Record_s": tr["record_s"], "Offset_s": tr["offset"],
                     "Alignment_|r|": tr["offset_corr"], "Loading_cycles": len(peaks),
                     "Max_compression_mm": 1000 * float(np.max(-d)),
                     "Included": tr["offset_ok"] and len(peaks) >= 2,
                     "Reason": ("" if tr["offset_ok"] and len(peaks) >= 2 else
                                "alignment |r| < %.1f" % GOOD_OFFSET_CORR if not tr["offset_ok"]
                                else "fewer than 2 loading cycles")})
    return pd.DataFrame(rows)


# ============================================================
# J1: IS THE FASTTDA GAIN REAL, AT EVERY WINDOW, AND VS A SIMPLER FEATURE?
# ============================================================

def gain_tests(use):
    rng = np.random.default_rng(0)
    models = base_models()
    rows = []
    for ws in GAIN_WINDOWS_S:
        frames = [window_table(tr, ws) for tr in use]
        print(f"  window {ws:4.1f} s (step {WINDOW_STEP_S:g} s): {sum(len(f) for f in frames)} windows, FastTDA fit ok "
              f"{100 * np.mean(np.concatenate([f['FastTDA_FitOK'] for f in frames])):.0f}%")
        for tgt in TARGETS:
            unit = UNIT_SCALE[tgt.split(":")[0]][1]
            for bname, (kind, bcols) in models.items():
                base = loto_sse(frames, tgt, bcols, kind)
                for add, acols in ADDED.items():
                    if add == "curvature" and (kind == "rf" or "curvature" in bname):
                        continue                                      # FastTDA only on these baselines
                    withf = loto_sse(frames, tgt, bcols + acols, kind)
                    rb, rw = np.sqrt(base[:, 0] / base[:, 1]), np.sqrt(withf[:, 0] / withf[:, 1])
                    red = 100 * (1 - rmse(withf) / rmse(base))
                    boot = []
                    for _ in range(N_BOOTSTRAP):                        # resample trials
                        k = rng.integers(0, len(frames), len(frames))
                        boot.append(100 * (1 - rmse(withf[k]) / rmse(base[k])))
                    p_wil = (stats.wilcoxon(rw, rb).pvalue if np.any(rw != rb) else 1.0)
                    p_perm = np.nan
                    if add == "FastTDA" and kind == "linear" and N_PERMUTATIONS:
                        null = [100 * (1 - rmse(loto_sse(shifted(frames, rng), tgt, bcols + acols, kind))
                                       / rmse(base)) for _ in range(N_PERMUTATIONS)]
                        p_perm = (np.sum(np.array(null) >= red) + 1) / (N_PERMUTATIONS + 1)
                    rows.append({"Target": tgt, "Unit": unit, "Window_s": ws, "Baseline": bname,
                                 "Added": add, "RMSE_baseline": rmse(base), "RMSE_with": rmse(withf),
                                 "RMSE_reduction_pct": red,
                                 "CI95_low_pct": np.percentile(boot, 2.5),
                                 "CI95_high_pct": np.percentile(boot, 97.5),
                                 "Trials_improved": f"{int(np.sum(rw < rb))}/{len(rb)}",
                                 "Wilcoxon_p": p_wil, "Permutation_p": p_perm})
    g = pd.DataFrame(rows)
    for (tgt, b, a), idx in g.groupby(["Target", "Baseline", "Added"]).groups.items():
        g.loc[idx, "Wilcoxon_p_Holm"] = holm(g.loc[idx, "Wilcoxon_p"])
        if g.loc[idx, "Permutation_p"].notna().all():
            g.loc[idx, "Permutation_p_Holm"] = holm(g.loc[idx, "Permutation_p"])
    return g


def plot_gain(g, out_dir):
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(6.2 * len(TARGETS), 4.4), sharey=True)
    series = [("Two-branch cubic", "FastTDA", COLORS["FastTDA"], "two-branch + FastTDA"),
              ("Two-branch cubic", "curvature", COLORS["curvature"], "two-branch + curvature (no TDA)"),
              ("Two-branch cubic + curvature", "FastTDA", "#1baf7a", "two-branch + curvature, then + FastTDA"),
              ("Random forest", "FastTDA", COLORS["rf"], "random forest + FastTDA")]
    for ax, tgt in zip(np.atleast_1d(axes), TARGETS):
        for k, (b, a, c, label) in enumerate(series):
            s = g[(g["Target"] == tgt) & (g["Baseline"] == b) & (g["Added"] == a)]
            if s.empty:
                continue
            x = s["Window_s"] + (k - 1.5) * 0.22
            ax.errorbar(x, s["RMSE_reduction_pct"],
                        yerr=[s["RMSE_reduction_pct"] - s["CI95_low_pct"], s["CI95_high_pct"] - s["RMSE_reduction_pct"]],
                        color=c, lw=1.6, marker="o", ms=6, capsize=3, label=label)
            sig = s["Permutation_p_Holm"] if "Permutation_p_Holm" in s and s["Permutation_p_Holm"].notna().all() \
                else s["Wilcoxon_p_Holm"]
            ax.scatter(x[sig < 0.05], s["RMSE_reduction_pct"][sig < 0.05], s=110, facecolors="none",
                       edgecolors=c, lw=1.4)
        ax.axhline(0, color="#3d3d3a", lw=0.8)
        ax.set_xticks(GAIN_WINDOWS_S)
        ax.set_xlabel("Window length (s)")
        ax.set_title(f"{tgt}", fontsize=10, fontweight="bold")
        ax.grid(axis="y", color="#e4e3df", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("RMSE reduction vs baseline (%)\n(95% CI over trials; ring = Holm p < 0.05)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=9, loc="lower center", ncol=len(l))
    fig.suptitle("J1  Does FastTDA lower the error of the strongest baselines, at every window?",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, out_dir, "J1_fasttda_gain_by_window.png")


# ============================================================
# J2 / J3: SENSOR CHARACTERISATION AND REPEATABILITY (per loading cycle)
# ============================================================

def cycles_of(tr):
    """Index ranges of the loading cycles: from one return to 0 mm to the next, each with one
    compression peak deeper than 80% of the stroke."""
    d = tr["targets"]["Displacement_m"]
    dist = int(0.7 * LOADING_PERIOD_S * tr["fs"])
    tops, _ = signal.find_peaks(d, distance=dist, prominence=0.5 * np.ptp(d))
    tops = list(tops)
    if not tops or tops[0] > dist // 2:
        tops = [0] + tops
    if tops[-1] < len(d) - dist // 2:
        tops = tops + [len(d) - 1]
    out = []
    for a, b in zip(tops[:-1], tops[1:]):
        pk = a + int(np.argmin(d[a:b + 1]))
        if -d[pk] >= 0.8 * np.max(-d):
            out.append((a, pk, b))
    return out


def branch_curve(x_mm, y, grid):
    """Mean of y in displacement bins (NaN where a bin is empty)."""
    idx = np.digitize(x_mm, grid) - 1
    return np.array([np.mean(y[idx == k]) if np.any(idx == k) else np.nan for k in range(len(grid) - 1)])


def hysteresis_index(x_mm, y, load, grid):
    """Largest gap between the loading and the unloading curve, as % of the range of y."""
    yl, yu = branch_curve(x_mm[load], y[load], grid), branch_curve(x_mm[~load], y[~load], grid)
    gap = np.nanmax(np.abs(yl - yu))
    return 100 * gap / np.ptp(y), yl, yu


def cycle_table(trials):
    rows, curves = [], {}
    for tr in trials:
        x_mm = 1000 * tr["targets"]["Displacement_m"]
        F, C = tr["targets"]["Force_N"], tr["x"]
        grid = np.arange(np.floor(x_mm.min() / BIN_MM) * BIN_MM, BIN_MM + 1e-9, BIN_MM)
        for k, (a, pk, b) in enumerate(cycles_of(tr)):
            seg = slice(a, b + 1)
            xs, Fs, Cs = x_mm[seg], F[seg], C[seg]
            load = np.arange(a, b + 1) <= pk
            h_c, cl, cu = hysteresis_index(xs, Cs, load, grid)
            h_f, fl, fu = hysteresis_index(xs, Fs, load, grid)
            near0, deep = xs > xs.max() - 2.0, xs < xs.min() + 2.0
            snap = (xs >= SNAP_RANGE_MM[0]) & (xs <= SNAP_RANGE_MM[1]) & load
            fl_snap = np.where(snap, Fs, np.inf)
            i_snap = int(np.argmin(fl_snap)) if np.isfinite(fl_snap).any() else None
            rows.append({"Trial": tr["name"], "Cycle": k + 1,
                         "Cap_unloaded_pF": float(np.mean(Cs[near0])),
                         "Cap_compressed_pF": float(np.mean(Cs[deep])),
                         "Cap_swing_pF": float(np.mean(Cs[deep]) - np.mean(Cs[near0])),
                         "Peak_force_N": float(np.max(np.abs(Fs))),                    # magnitude
                         "Hysteresis_energy_J": float(abs(np.trapz(Fs, xs / 1000))),   # loop area, N*m
                         "Cap_hysteresis_pct": h_c, "Force_hysteresis_pct": h_f,
                         "Snap_through_mm": float(xs[i_snap]) if i_snap is not None else np.nan,
                         "Snap_through_force_N": float(Fs[i_snap]) if i_snap is not None else np.nan})
            curves[(tr["name"], k + 1)] = {"grid": grid, "cap": (cl, cu), "force": (fl, fu)}
    return pd.DataFrame(rows), curves


def sensor_table(trials, cyc):
    """Sensitivity dC/dx from the mean calibration curve (cycles 2+), usable range, noise and
    resolution."""
    rows, sens_curves = [], {}
    for tr in trials:
        x_mm = 1000 * tr["targets"]["Displacement_m"]
        C = tr["x"] - np.median(tr["x"])
        later = tr["t"] >= SKIP_START_S
        grid = np.arange(np.floor(x_mm.min() / BIN_MM) * BIN_MM, BIN_MM + 1e-9, BIN_MM)
        centres = (grid[:-1] + grid[1:]) / 2
        curve = branch_curve(x_mm[later], C[later], grid)
        ok = np.isfinite(curve)
        sens = np.gradient(curve[ok], centres[ok])                    # pF per mm
        sens = np.convolve(sens, np.ones(3) / 3, mode="same")         # light smoothing
        mag = np.abs(sens)
        lin = mag >= LINEAR_FRACTION * mag.max()
        # contiguous usable range around the most sensitive point
        k0 = int(np.argmax(mag))
        lo, hi = k0, k0
        while lo > 0 and lin[lo - 1]:
            lo -= 1
        while hi < len(lin) - 1 and lin[hi + 1]:
            hi += 1
        win = max(5, int(NOISE_SMOOTH_S * tr["fs"]) | 1)
        resid = tr["x"] - signal.savgol_filter(tr["x"], win, 2)
        noise = 1.4826 * float(np.median(np.abs(resid - np.median(resid))))
        s_lin = float(np.mean(mag[lo:hi + 1]))
        c = cyc[cyc["Trial"] == tr["name"]]
        rows.append({"Trial": tr["name"], "Peak_sensitivity_pF_per_mm": float(mag.max()),
                     "Mean_sensitivity_in_range_pF_per_mm": s_lin,
                     "Usable_range_from_mm": float(centres[ok][lo]), "Usable_range_to_mm": float(centres[ok][hi]),
                     "Usable_span_mm": float(centres[ok][hi] - centres[ok][lo] + BIN_MM),
                     "Sensitivity_deep_vs_peak_pct": 100 * float(np.mean(mag[centres[ok] < -40])) / mag.max(),
                     "Noise_pF": noise, "Resolution_mm": noise / s_lin, "Detection_limit_3sigma_mm": 3 * noise / s_lin,
                     "Cap_hysteresis_pct": float(c["Cap_hysteresis_pct"].iloc[1:].mean()),
                     "Force_hysteresis_pct": float(c["Force_hysteresis_pct"].iloc[1:].mean())})
        sens_curves[tr["name"]] = (centres[ok], sens, (centres[ok][lo], centres[ok][hi]))
    return pd.DataFrame(rows), sens_curves


def repeatability(cyc):
    """Cycle-to-cycle (within trial, cycles 2+) and trial-to-trial variation; first-cycle effect;
    baseline drift per cycle."""
    cols = ["Cap_swing_pF", "Peak_force_N", "Hysteresis_energy_J", "Snap_through_mm"]
    rows = []
    later = cyc[cyc["Cycle"] > 1]
    for c in cols:
        within = later.groupby("Trial")[c].agg(lambda v: 100 * np.std(v, ddof=1) / abs(np.mean(v)))
        means = later.groupby("Trial")[c].mean()
        first = cyc[cyc["Cycle"] == 1].set_index("Trial")[c]
        rows.append({"Quantity": c, "Mean (cycles 2+)": float(later[c].mean()),
                     "Cycle-to-cycle CV %": float(within.mean()),
                     "Trial-to-trial CV %": float(100 * means.std(ddof=1) / abs(means.mean())),
                     "First cycle vs later %": float(np.mean(100 * (first - means) / means.abs()))})
    drift = [np.polyfit(g["Cycle"], g["Cap_unloaded_pF"], 1)[0] for _, g in cyc.groupby("Trial") if len(g) > 1]
    rows.append({"Quantity": "Cap_unloaded drift (pF per cycle)", "Mean (cycles 2+)": float(np.mean(drift)),
                 "Cycle-to-cycle CV %": np.nan, "Trial-to-trial CV %": np.nan,
                 "First cycle vs later %": np.nan})
    return pd.DataFrame(rows)


def plot_sensor(sens_curves, curves, sensor, out_dir):
    names = list(sens_curves)
    colors = {n: TRIAL_CMAP(i / max(1, len(names) - 1)) for i, n in enumerate(names)}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    ax = axes[0]
    for n in names:
        x, s, (lo, hi) = sens_curves[n]
        ax.plot(x, np.abs(s) * 1000, color=colors[n], lw=1.5, label=n)
    lo, hi = sensor["Usable_range_from_mm"].median(), sensor["Usable_range_to_mm"].median()
    ax.axvspan(lo, hi, color="#eb6834", alpha=0.12, lw=0)
    ax.text((lo + hi) / 2, -0.12, "usable range", ha="center", va="top", fontsize=8, color="#a8471f",
            transform=ax.get_xaxis_transform())                              # below the axis, clear of the lines
    ax.set_xlabel("Displacement (mm)", labelpad=14)
    ax.set_ylabel("|dC/dx| (fF per mm)")
    ax.set_title("Sensitivity vs displacement (saturation)", fontsize=10, fontweight="bold")
    ax.legend(frameon=False, fontsize=7)
    for ax, key, unit, title in [(axes[1], "cap", "pF", "Capacitance: loading vs unloading"),
                                 (axes[2], "force", "N", "Force: loading vs unloading")]:
        warnings.filterwarnings("ignore", message="Mean of empty slice")   # bins a branch never visits
        for n in names:
            ks = [k for k in curves if k[0] == n and k[1] > 1]
            if not ks:
                continue
            g = curves[ks[0]]["grid"]
            centres = (g[:-1] + g[1:]) / 2
            yl = np.nanmean([curves[k][key][0] for k in ks], axis=0)
            yu = np.nanmean([curves[k][key][1] for k in ks], axis=0)
            ax.plot(centres, yl, color=colors[n], lw=1.5)
            ax.plot(centres, yu, color=colors[n], lw=1.5, ls="--")
        ax.set_xlabel("Displacement (mm)")
        ax.set_ylabel(unit)
        col = "Cap_hysteresis_pct" if key == "cap" else "Force_hysteresis_pct"
        ax.set_title(f"{title}\n(solid = loading, dashed = unloading; hysteresis "
                     f"{sensor[col].mean():.0f}% of range)", fontsize=10, fontweight="bold")
    for ax in axes:
        ax.grid(color="#e4e3df", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("J2  Sensor characterisation (cycles 2+, trials used)", fontweight="bold")
    fig.tight_layout()
    save(fig, out_dir, "J2_sensitivity_hysteresis.png")


def plot_repeatability(cyc, out_dir):
    names = list(dict.fromkeys(cyc["Trial"]))
    colors = {n: TRIAL_CMAP(i / max(1, len(names) - 1)) for i, n in enumerate(names)}
    panels = [("Cap_swing_pF", "Capacitance swing (pF)"), ("Peak_force_N", "Peak force (N)"),
              ("Hysteresis_energy_J", "Hysteresis energy (J)"), ("Snap_through_mm", "Snap-through position (mm)")]
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.9))
    for ax, (c, label) in zip(axes, panels):
        for n in names:
            g = cyc[cyc["Trial"] == n]
            ax.plot(g["Cycle"], g[c], "-o", color=colors[n], ms=4, lw=1.2, label=n)
        ax.axvspan(0.6, 1.4, color="#e4e3df", alpha=0.6, lw=0)
        ax.set_xlabel("Loading cycle (shaded = first)")
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xticks(sorted(cyc["Cycle"].unique()))
        ax.grid(color="#e4e3df", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("J3  Repeatability: every loading cycle of every trial", fontweight="bold")
    fig.tight_layout()
    save(fig, out_dir, "J3_repeatability.png")


# ============================================================
# J4: LEARNING CURVE (error vs number of training trials)
# ============================================================

def learning_curve(use):
    frames = [window_table(tr, LEARN_WINDOW_S) for tr in use]
    models = {"Two-branch cubic": TWO_BRANCH, "Two-branch cubic + FastTDA": TWO_BRANCH + ADDED["FastTDA"]}
    rows = []
    for tgt in TARGETS:
        scale = UNIT_SCALE[tgt.split(":")[0]][0]
        for k in range(1, len(frames)):
            for m, cols in models.items():
                vals = []
                for i in range(len(frames)):
                    others = [j for j in range(len(frames)) if j != i]
                    for combo in itertools.combinations(others, k):     # every training subset of size k
                        train = pd.concat([frames[j] for j in combo])
                        e = scale * (fit_predict("linear", train, frames[i], cols, tgt)
                                     - frames[i][tgt].to_numpy(float))
                        vals.append(float(np.sqrt(np.mean(e ** 2))))
                rows.append({"Target": tgt, "Unit": UNIT_SCALE[tgt.split(":")[0]][1], "Model": m,
                             "Training_trials": k, "RMSE_median": float(np.median(vals)),
                             "RMSE_q25": float(np.percentile(vals, 25)), "RMSE_q75": float(np.percentile(vals, 75)),
                             "N_splits": len(vals)})
    return pd.DataFrame(rows)


def plot_learning(lc, out_dir):
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(6.2 * len(TARGETS), 4.0))
    for ax, tgt in zip(np.atleast_1d(axes), TARGETS):
        for m, c in [("Two-branch cubic", COLORS["base"]), ("Two-branch cubic + FastTDA", COLORS["FastTDA"])]:
            s = lc[(lc["Target"] == tgt) & (lc["Model"] == m)]
            ax.fill_between(s["Training_trials"], s["RMSE_q25"], s["RMSE_q75"], color=c, alpha=0.18, lw=0)
            ax.plot(s["Training_trials"], s["RMSE_median"], "-o", color=c, lw=1.8, ms=5, label=m)
        ax.set_xlabel("Number of training trials")
        ax.set_ylabel(f"RMSE on a left-out trial ({lc.loc[lc['Target'] == tgt, 'Unit'].iloc[0]})")
        ax.set_title(f"{tgt} ({LEARN_WINDOW_S:g} s windows; band = 25-75%)", fontsize=10, fontweight="bold")
        ax.set_xticks(sorted(lc["Training_trials"].unique()))
        ax.set_ylim(bottom=0)
        ax.grid(color="#e4e3df", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8)
    fig.suptitle("J4  Would more trials of the same test help?", fontweight="bold")
    fig.tight_layout()
    save(fig, out_dir, "J4_learning_curve.png")


# ============================================================
# RUN
# ============================================================

def save(fig, out_dir, name):
    png = os.path.join(out_dir, name)
    fig.savefig(png, dpi=150)
    print(f"Saved: {png}")
    plt.show()
    plt.close(fig)


def answers(inc, g, sensor, rep, lc):
    """Plain-language answers to the reviewer issues, from the numbers above."""
    print("\n" + "=" * 78 + "\nANSWERS TO THE REVIEWER ISSUES\n" + "=" * 78)
    used, dropped = inc[inc["Included"]], inc[~inc["Included"]]
    print(f"\nJ0  Trials: {len(used)} used ({', '.join(used['Trial'])}); "
          + (f"{len(dropped)} excluded ({', '.join(dropped['Trial'] + ': ' + dropped['Reason'])})."
             if len(dropped) else "none excluded."))
    for tgt in TARGETS:
        s = g[(g["Target"] == tgt) & (g["Baseline"] == "Two-branch cubic")]
        f, c = s[s["Added"] == "FastTDA"], s[s["Added"] == "curvature"]
        unit = s["Unit"].iloc[0]
        pcol = "Permutation_p_Holm" if f["Permutation_p_Holm"].notna().all() else "Wilcoxon_p_Holm"
        best = f.loc[f["RMSE_reduction_pct"].idxmax()]
        sig = f[f[pcol] < 0.05]
        print(f"\nJ1  {tgt}: FastTDA lowers the two-branch error at {int((f['RMSE_reduction_pct'] > 0).sum())}"
              f" of {len(f)} windows; best {best['RMSE_reduction_pct']:.0f}% at {best['Window_s']:g} s "
              f"({best['RMSE_baseline']:.2f} -> {best['RMSE_with']:.2f} {unit}, 95% CI "
              f"{best['CI95_low_pct']:.0f} to {best['CI95_high_pct']:.0f}%, {best['Trials_improved']} trials). "
              f"Significant after Holm correction at: "
              f"{', '.join(f'{w:g} s' for w in sig['Window_s']) or 'no window'}.")
        print(f"    Simpler feature (curvature, no TDA): mean change {c['RMSE_reduction_pct'].mean():.0f}% "
              f"vs FastTDA {f['RMSE_reduction_pct'].mean():.0f}% (averaged over windows).")
        fc = g[(g["Target"] == tgt) & (g["Baseline"] == "Two-branch cubic + curvature")]
        print(f"    FastTDA on top of curvature: mean change {fc['RMSE_reduction_pct'].mean():.0f}% over windows "
              f"(does TDA add anything a simple shape feature does not?).")
        rf = g[(g["Target"] == tgt) & (g["Baseline"] == "Random forest")]
        if len(rf):
            print(f"    Random forest + FastTDA: mean change {rf['RMSE_reduction_pct'].mean():.0f}% over windows.")
    print(f"\nJ2  Sensitivity {1000 * sensor['Peak_sensitivity_pF_per_mm'].mean():.1f} fF/mm at its peak; usable "
          f"range about {sensor['Usable_range_from_mm'].median():.0f} to {sensor['Usable_range_to_mm'].median():.0f} mm; "
          f"beyond -40 mm sensitivity falls to {sensor['Sensitivity_deep_vs_peak_pct'].mean():.0f}% of the peak "
          f"(saturation). Noise {1000 * sensor['Noise_pF'].mean():.2f} fF -> resolution "
          f"{sensor['Resolution_mm'].mean():.2f} mm (3-sigma {sensor['Detection_limit_3sigma_mm'].mean():.2f} mm).")
    print(f"    Hysteresis: capacitance {sensor['Cap_hysteresis_pct'].mean():.0f}% of its range vs force "
          f"{sensor['Force_hysteresis_pct'].mean():.0f}% - the sensor reads displacement, not force.")
    r = rep.set_index("Quantity")
    print(f"\nJ3  Capacitance swing: cycle-to-cycle CV {r.loc['Cap_swing_pF', 'Cycle-to-cycle CV %']:.1f}%, "
          f"trial-to-trial CV {r.loc['Cap_swing_pF', 'Trial-to-trial CV %']:.1f}%; first cycle "
          f"{r.loc['Cap_swing_pF', 'First cycle vs later %']:+.0f}% vs later cycles. Peak force first cycle "
          f"{r.loc['Peak_force_N', 'First cycle vs later %']:+.0f}% (material conditioning). Snap-through at "
          f"{r.loc['Snap_through_mm', 'Mean (cycles 2+)']:.1f} mm (CV {r.loc['Snap_through_mm', 'Cycle-to-cycle CV %']:.1f}%). "
          f"Baseline drift {1000 * r.loc['Cap_unloaded drift (pF per cycle)', 'Mean (cycles 2+)']:+.2f} fF per cycle.")
    for tgt in TARGETS:
        s = lc[(lc["Target"] == tgt) & (lc["Model"] == "Two-branch cubic + FastTDA")]
        a, b = s["RMSE_median"].iloc[0], s["RMSE_median"].iloc[-1]
        tail = s["RMSE_median"].iloc[-2] - b if len(s) > 1 else np.nan
        print(f"\nJ4  {tgt}: RMSE {a:.2f} -> {b:.2f} {s['Unit'].iloc[0]} from 1 to {s['Training_trials'].max()} "
              f"training trials; the last added trial changes it by {tail:.2f}. "
              + ("The curve has flattened: more trials of the SAME test add little - new amplitudes / rates are needed."
                 if abs(tail) < 0.05 * b else "Still falling: more trials would lower the error further."))


out_dir = to_local_path(OUTPUT_DIR) if OUTPUT_DIR else os.path.join(to_local_path(DATA_DIR), "tda_results", "journal")
os.makedirs(out_dir, exist_ok=True)
print("Loading and aligning:")
trials = [load_pair(*p) for p in find_pairs()]
inc = inclusion_table(trials)
print("\nJ0  Trial inclusion:")
print(inc.round(2).to_string(index=False))
use = [tr for tr in trials if inc.set_index("Trial").loc[tr["name"], "Included"]]
if len(use) < 3:
    raise RuntimeError("Need at least 3 included trials for the leave-one-trial-out checks.")

print(f"\nJ1  FastTDA gain on {len(use)} trials ({N_PERMUTATIONS} permutations, {N_BOOTSTRAP} bootstrap samples):")
gain = gain_tests(use)
print(gain.round(3).to_string(index=False))
plot_gain(gain, out_dir)

cyc, curves = cycle_table(use)
sensor, sens_curves = sensor_table(use, cyc)
rep = repeatability(cyc)
print("\nJ2  Sensor characterisation:")
print(sensor.round(4).to_string(index=False))
print("\nJ3  Per loading cycle:")
print(cyc.round(3).to_string(index=False))
print("\nJ3  Repeatability:")
print(rep.round(2).to_string(index=False))
plot_sensor(sens_curves, curves, sensor, out_dir)
plot_repeatability(cyc, out_dir)

print(f"\nJ4  Learning curve ({LEARN_WINDOW_S:g} s windows):")
lc = learning_curve(use)
print(lc.round(3).to_string(index=False))
plot_learning(lc, out_dir)

answers(inc, gain, sensor, rep, lc)
xlsx = os.path.join(out_dir, "SEC_MTS_journal_checks.xlsx")
with pd.ExcelWriter(xlsx, engine="openpyxl") as wr:
    for sheet, frame in {"J0_Inclusion": inc, "J1_FastTDA_gain": gain, "J2_Sensor": sensor,
                         "J3_Cycles": cyc, "J3_Repeatability": rep, "J4_Learning_curve": lc}.items():
        frame.to_excel(wr, sheet_name=sheet, index=False)
print(f"\nSaved: {xlsx}")
