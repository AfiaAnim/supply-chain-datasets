"""
SEC + MTS: which model turns the SEC capacitance into MTS displacement and force? Paste into ONE
Jupyter cell and run (or: python sec_mts_models.py).

Real-time (causal) prediction: at each time t (every STEP_S) a model sees only the capacitance up to
t, as a deployed sensor would, and predicts the MTS displacement and force at t. Capacitance is tared
on the first TARE_S of each trial (tire unloaded).

Models (capacitance only - the MTS is never an input):
    Linear calibration              C
    Cubic calibration               C, C^2, C^3                         (the S-shaped sensor curve)
    Two-branch cubic + curvature    cubic + pressing/releasing branch + curvature + spread
    NARX-style (lagged C)           cubic + capacitance 2, 4, 8 s earlier + short slope (memory)
    Gradient boosting               trees on all capacitance features
    Gradient boosting + FastTDA     the same + FastTDA of the last WINDOW_S
    Gaussian process                smooth regression with a 95% uncertainty band
    Bouc-Wen (SEC)                  force only: hysteresis model driven by the displacement that the
                                    SEC estimates (cubic calibration) - physics-based
    Bouc-Wen (true displacement)    force only, REFERENCE: same model driven by the MTS displacement
                                    (shows how much error is the model and how much the sensor)
    Hybrid: boosted disp -> Bouc-Wen  force only: the tire's Bouc-Wen model (fitted on MTS data of the
                                    training trials) driven by the displacement the SEC estimates with
                                    gradient boosting - sensor model + physics model

Validation:
    A  leave-one-trial-out over every usable trial (alignment reliable, >= 2 loading cycles)
    B  train on TRAIN_TRIALS, test on TEST_TRIALS (default trials 04-09 -> 10-15: an independent set)
Errors in mm and N (RMSE, MAE, max, % of range, R2), per left-out / test trial, GP band coverage.

Output (in DATA_DIR/tda_results/models):
    M1_model_errors.png        RMSE of every model, both validations, displacement and force
    M2_predictions.png         one test trial over time: MTS vs the main models (GP band)
    M3_force_loops.png         force vs displacement: MTS loop vs predicted loops
    SEC_MTS_models.xlsx        Metrics_LOTO, Metrics_Split, Per_trial, Predictions, Bouc_Wen_params

Requirements:
    pip install numpy pandas scipy matplotlib openpyxl scikit-learn
"""

import glob
import os
import re
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy import optimize, signal

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
OUTPUT_DIR = None                       # None = DATA_DIR/tda_results/models
TRIALS = None                           # None = every trial found

SEC_TIME_COL = "time_s"
SEC_COL = "C1_filtered_pF"
MTS_TIME_COL = "RunningTime_s"
TARGET_COLS = ["Displacement_m", "Force_N"]
RESAMPLE_HZ = 20.0

OFFSET_GUESS_S = 10.0
OFFSET_SEARCH_S = 5.0
OFFSET_REF_COL = "Displacement_m"
MIN_OFFSET_CORR = 0.2
GOOD_OFFSET_CORR = 0.7                  # weaker alignment -> trial not used (trials 01-03)
LOADING_PERIOD_S = 40.0

TRAIN_TRIALS = [f"trial_{i:02d}" for i in range(4, 10)]    # validation B: train ...
TEST_TRIALS = [f"trial_{i:02d}" for i in range(10, 16)]    # ... and test on these

TARE_S = 2.0                            # capacitance baseline = median of the first TARE_S (unloaded)
STEP_S = 0.5                            # a prediction every STEP_S
SKIP_START_S = 40.0                     # first loading cycle not scored (material conditioning)
WINDOW_S = 8.0                          # trailing window for spread / slope / curvature / FastTDA
SHORT_SLOPE_S = 2.0                     # short trailing slope: quick direction signal
LAGS_S = [2.0, 4.0, 8.0]                # NARX-style: capacitance this long ago
LEVEL_S = 1.0                           # "current" capacitance = mean of the last LEVEL_S
FASTTDA_DELAY_S = 1.0
GP_MAX_TRAIN = 800                      # Gaussian process: training points (random subset, speed)
BW_HZ = 10.0                            # Bouc-Wen simulated at this rate
EXAMPLE_TRIAL = None                    # trial drawn in M2 / M3; None = first test trial
POST_MEDIAN_N = 3                       # causal median of the last N predictions (removes one-step jumps
                                        # between branches where the capacitance is flat); 1 = off

UNITS = {"Displacement_mm": "mm", "Force_N": "N"}
TARGETS = ["Displacement_mm", "Force_N"]
MODEL_COLORS = {"Linear calibration": "#b8b7b2", "Cubic calibration": "#8a8984",
                "Two-branch cubic + curvature": "#2a78d6", "NARX-style (lagged C)": "#1baf7a",
                "Gradient boosting": "#0d366b", "Gradient boosting + FastTDA": "#eb6834",
                "Gaussian process": "#6250d6", "Bouc-Wen (SEC)": "#e87ba4",
                "Bouc-Wen (true displacement)": "#e34948", "Hybrid: boosted disp -> Bouc-Wen": "#eda100"}


# ============================================================
# FILES AND ALIGNMENT (same as the other SEC/MTS scripts)
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
    for p in sorted(set(files)):
        name, num = trial_key(p)
        kind = file_kind(p) if name else None
        if name and kind:
            (sec if kind == "sec" else mts).setdefault(name, (num, p))
    if not sec:
        raise FileNotFoundError(f"No SEC csv files (with a {SEC_COL!r} column) found in {os.path.abspath(base)}")
    pairs = []
    for name in sorted(set(sec) | set(mts), key=lambda n: int(n.split("_")[1])):
        if TRIALS and name not in TRIALS:
            continue
        if name not in sec or name not in mts:
            print(f"  {name}: {'SEC' if name in sec else 'MTS'} file only - skipped")
            continue
        pairs.append((name, sec[name][0], sec[name][1], mts[name][1]))
    print(f"Found {len(pairs)} SEC/MTS pairs in {os.path.abspath(base)}")
    return pairs


def estimate_offset(t_sec, x_sec, t_mts, y_mts):
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


def load_trial(name, num, sec_path, mts_path):
    sec, mts = read_numeric_csv(sec_path), read_numeric_csv(mts_path)
    t_sec = sec[SEC_TIME_COL].to_numpy(float)
    t_sec = t_sec - t_sec[0]
    x = sec[SEC_COL].to_numpy(float)
    t_mts = mts[MTS_TIME_COL].to_numpy(float)
    t_mts = t_mts - t_mts[0]
    fs_sec = (len(t_sec) - 1) / (t_sec[-1] - t_sec[0])
    offset, r = estimate_offset(t_sec, x, t_mts, mts[OFFSET_REF_COL].to_numpy(float))
    if RESAMPLE_HZ < fs_sec:
        x = signal.sosfiltfilt(signal.butter(4, 0.4 * RESAMPLE_HZ, btype="lowpass", fs=fs_sec, output="sos"), x)
    t = np.arange(0, min(t_mts[-1], t_sec[-1] - offset), 1 / RESAMPLE_HZ)
    on_edge = abs(abs(offset - OFFSET_GUESS_S) - OFFSET_SEARCH_S) < 0.11
    d = np.interp(t, t_mts, mts["Displacement_m"].to_numpy(float))
    n_cyc = len(signal.find_peaks(-d, height=0.8 * np.max(-d), distance=int(0.7 * LOADING_PERIOD_S * RESAMPLE_HZ))[0])
    ok = bool(r >= GOOD_OFFSET_CORR and not on_edge and n_cyc >= 2)
    print(f"  {name}: offset {offset:.2f} s (|corr| {r:.2f}), {n_cyc} cycles" + ("" if ok else "  <-- NOT USED"))
    return {"name": name, "fs": RESAMPLE_HZ, "t": t, "x": np.interp(t + offset, t_sec, x), "ok": ok,
            "Displacement_mm": 1000 * d,
            "Force_N": np.interp(t, t_mts, mts["Force_N"].to_numpy(float))}


# ============================================================
# CAUSAL FEATURES (capacitance up to the prediction time only)
# ============================================================

def fasttda_minor_axis(pc):
    """SVD-LSE ellipse fit (A + C = 1): minor axis 2b, NaN when the best conic is not an ellipse."""
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


def features(tr, start_s=SKIP_START_S):
    """One row per prediction time t (every STEP_S, t >= start_s), built from x[: t]."""
    fs, x = tr["fs"], tr["x"] - np.median(tr["x"][tr["t"] < TARE_S])         # tare: unloaded start
    w, wl, ws = int(WINDOW_S * fs), int(LEVEL_S * fs), int(SHORT_SLOPE_S * fs)
    d = max(1, int(FASTTDA_DELAY_S * fs))
    level = np.convolve(x, np.ones(wl) / wl)[: len(x)]                       # trailing mean, causal
    ends = np.arange(int(start_s * fs), len(x), int(STEP_S * fs))
    ends = ends[ends >= max(w, int(max(LAGS_S) * fs) + wl) - 1]
    W = np.stack([x[e - w + 1: e + 1] for e in ends])
    tt = (np.arange(w) - (w - 1) / 2) / fs
    V = np.column_stack([np.ones(w), tt, tt ** 2])
    coef = np.linalg.lstsq(V, W.T, rcond=None)[0]
    Ws = W[:, -ws:]
    ts = (np.arange(ws) - (ws - 1) / 2) / fs
    f = pd.DataFrame({"Trial": tr["name"], "Time_s": tr["t"][ends], "Cap": level[ends],
                      "Cap_std": W.std(axis=1), "Cap_slope": coef[1], "Cap_curv": coef[2],
                      "Cap_slope2": (Ws - Ws.mean(axis=1, keepdims=True)) @ ts / (ts @ ts)})
    for lag in LAGS_S:
        f[f"Cap_lag{lag:g}"] = level[ends - int(lag * fs)]
    f["Cap2"], f["Cap3"] = f["Cap"] ** 2, f["Cap"] ** 3
    f["Cap_dir"] = np.sign(f["Cap_slope2"])
    f["Cap_x_dir"], f["Cap2_x_dir"] = f["Cap"] * f["Cap_dir"], f["Cap2"] * f["Cap_dir"]
    lo, hi = W.min(axis=1, keepdims=True), W.max(axis=1, keepdims=True)
    Z = 2 * (W - lo) / np.where(hi > lo, hi - lo, 1.0) - 1                     # FastTDA: scaled window,
    stride = max(1, int(np.ceil((w - d) / 300)))                            # delay-embedded
    ftda = np.array([fasttda_minor_axis(np.column_stack([z[:-d:stride], z[d::stride]])) for z in Z])
    f["FastTDA"], f["FastTDA_ok"] = np.where(np.isfinite(ftda), ftda, 0.0), np.isfinite(ftda).astype(float)
    for c in TARGETS:
        f[c] = tr[c][ends]
    return f


CUBIC = ["Cap", "Cap2", "Cap3"]
CAP_ALL = ["Cap", "Cap_std", "Cap_slope", "Cap_slope2", "Cap_curv"] + [f"Cap_lag{l:g}" for l in LAGS_S]
MODELS = {   # name: (kind, features, targets)
    "Linear calibration": ("linear", ["Cap"], TARGETS),
    "Cubic calibration": ("linear", CUBIC, TARGETS),
    "Two-branch cubic + curvature": ("linear", CUBIC + ["Cap_dir", "Cap_x_dir", "Cap2_x_dir", "Cap_curv",
                                                        "Cap_std"], TARGETS),
    "NARX-style (lagged C)": ("linear", CUBIC + [f"Cap_lag{l:g}" for l in LAGS_S] + ["Cap_slope2"], TARGETS),
    "Gradient boosting": ("gb", CAP_ALL, TARGETS),
    "Gradient boosting + FastTDA": ("gb", CAP_ALL + ["FastTDA", "FastTDA_ok"], TARGETS),
    "Gaussian process": ("gp", ["Cap", "Cap_slope2", "Cap_curv", "Cap_lag4"], TARGETS),
    "Bouc-Wen (SEC)": ("bw", CUBIC, ["Force_N"]),
    "Bouc-Wen (true displacement)": ("bw_true", [], ["Force_N"]),
    "Hybrid: boosted disp -> Bouc-Wen": ("hybrid", CAP_ALL, ["Force_N"]),
}


# ============================================================
# MODELS
# ============================================================

def have_sklearn():
    try:
        import sklearn  # noqa: F401
        return True
    except ImportError:
        return False


def causal_median(pred):
    """Median of the current and the previous POST_MEDIAN_N - 1 predictions (no future data)."""
    if POST_MEDIAN_N <= 1:
        return pred
    return pd.Series(pred).rolling(POST_MEDIAN_N, min_periods=1).median().to_numpy()


def fit_tabular(kind, train, cols, target, seed=0):
    """Returns predict(frame) -> (mean, sd or None)."""
    X, y = train[cols].to_numpy(float), train[target].to_numpy(float)
    if kind == "linear":
        beta = np.linalg.lstsq(np.column_stack([np.ones(len(X)), X]), y, rcond=None)[0]
        return lambda f: (np.column_stack([np.ones(len(f)), f[cols].to_numpy(float)]) @ beta, None)
    if kind == "gb":
        from sklearn.ensemble import HistGradientBoostingRegressor
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                          min_samples_leaf=20, l2_regularization=1.0, random_state=seed).fit(X, y)
        return lambda f: (m.predict(f[cols].to_numpy(float)), None)
    if kind == "gp":
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
        rng = np.random.default_rng(seed)
        k = rng.choice(len(X), min(GP_MAX_TRAIN, len(X)), replace=False)
        mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-12
        kernel = ConstantKernel(1.0) * RBF(np.ones(X.shape[1])) + WhiteKernel(0.1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = GaussianProcessRegressor(kernel, normalize_y=True, random_state=seed).fit((X[k] - mu) / sd, y[k])
        return lambda f: m.predict((f[cols].to_numpy(float) - mu) / sd, return_std=True)
    raise ValueError(kind)


def bw_simulate(x, beta, gamma):
    """Bouc-Wen hysteretic variable (n = 1, A = 1): dz = dx - beta |dx| z - gamma dx |z|."""
    z, out = 0.0, [0.0]
    dxs = np.diff(x).tolist()
    for dx in dxs:
        z += dx - beta * abs(dx) * z - gamma * dx * abs(z)
        out.append(z)
    return np.array(out)


def bw_design(x, z):
    return np.column_stack([np.ones_like(x), x, x ** 2, x ** 3, z])         # F = poly(x) + alpha z


def bw_input(tr, calib):
    """Displacement driving the model at BW_HZ: from the SEC (cubic calibration), the MTS (calib None)
    or a given displacement estimate (calib = ("estimate", times, values))."""
    k = max(1, int(round(tr["fs"] / BW_HZ)))
    if isinstance(calib, tuple):
        _, te, xe = calib
        x = np.interp(tr["t"], te, xe)                                       # before the first estimate: held
    elif calib is None:
        x = tr["Displacement_mm"]
    else:
        xs = tr["x"] - np.median(tr["x"][tr["t"] < TARE_S])
        xs = np.convolve(xs, np.ones(int(LEVEL_S * tr["fs"])) / int(LEVEL_S * tr["fs"]))[: len(xs)]
        x = np.polyval(calib, xs)
    x = signal.savgol_filter(x, int(tr["fs"]) | 1, 2)                        # smooth: |dx| sums noise
    return tr["t"][::k], x[::k], tr["Force_N"][::k]


def fit_bouc_wen(train_trials, calib):
    """beta, gamma by Nelder-Mead; polynomial and alpha by least squares inside (nested)."""
    data = [bw_input(tr, calib) for tr in train_trials]
    keep = [t >= SKIP_START_S for t, _, _ in data]

    def solve(p):
        beta, gamma = p
        Xs, ys = [], []
        for (t, x, F), m in zip(data, keep):
            Xs.append(bw_design(x, bw_simulate(x, beta, gamma))[m])
            ys.append(F[m])
        X, y = np.vstack(Xs), np.concatenate(ys)
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        return coef, float(np.sum((X @ coef - y) ** 2))

    best = None
    for b0, g0 in [(0.05, 0.02), (0.2, 0.1), (0.5, -0.2)]:
        res = optimize.minimize(lambda p: solve(p)[1] if p[0] >= 0 and abs(p[1]) <= p[0] + 1 else 1e30,
                                [b0, g0], method="Nelder-Mead", options={"maxiter": 120, "xatol": 1e-4})
        if best is None or res.fun < best.fun:
            best = res
    coef, _ = solve(best.x)
    return {"beta": best.x[0], "gamma": best.x[1], "coef": coef}


def predict_bouc_wen(tr, params, calib, times):
    t, x, _ = bw_input(tr, calib)
    F = bw_design(x, bw_simulate(x, params["beta"], params["gamma"])) @ params["coef"]
    return np.interp(times, t, F)


def cubic_calibration(train_frames):
    tr = pd.concat(train_frames)
    return np.polyfit(tr["Cap"], tr["Displacement_mm"], 3)


def run_fold(train_trials, test_trial, frames, rf_ok, fold_seed=0, full_frames=None):
    """Fit every model on train_trials, predict test_trial: {model: {target: (pred, sd)}}."""
    train = pd.concat([frames[t["name"]] for t in train_trials], ignore_index=True)
    test = frames[test_trial["name"]]
    out, bw_true = {}, None
    for m, (kind, cols, targets) in MODELS.items():
        if kind in ("gb", "gp", "hybrid") and not rf_ok:
            continue
        out[m] = {}
        for tgt in targets:
            if kind == "bw":
                calib = cubic_calibration([frames[t["name"]] for t in train_trials])
                p = fit_bouc_wen(train_trials, calib)
                out[m][tgt] = (predict_bouc_wen(test_trial, p, calib, test["Time_s"].to_numpy()), None, p)
            elif kind == "bw_true":
                p = bw_true = fit_bouc_wen(train_trials, None)
                out[m][tgt] = (predict_bouc_wen(test_trial, p, None, test["Time_s"].to_numpy()), None, p)
            elif kind == "hybrid":
                # the displacement the SEC estimates from the start of the record (hysteresis has memory)
                p = bw_true or fit_bouc_wen(train_trials, None)
                early = full_frames[test_trial["name"]]
                xd, _ = fit_tabular("gb", train, cols, "Displacement_mm", fold_seed)(early)
                xd = causal_median(xd)
                est = ("estimate", early["Time_s"].to_numpy(), xd)
                out[m][tgt] = (predict_bouc_wen(test_trial, p, est, test["Time_s"].to_numpy()), None, p)
            else:
                pred, sd = fit_tabular(kind, train, cols, tgt, fold_seed)(test)
                out[m][tgt] = (causal_median(pred), sd, None)
    return out


# ============================================================
# EVALUATION
# ============================================================

def metrics(rows_pred, scheme):
    """rows_pred: list of (trial, model, target, true, pred, sd)."""
    per_trial, pooled = [], []
    df = pd.DataFrame(rows_pred, columns=["Trial", "Model", "Target", "True", "Pred", "SD"])
    for (m, tgt), g in df.groupby(["Model", "Target"], sort=False):
        t, p = np.concatenate(g["True"].tolist()), np.concatenate(g["Pred"].tolist())
        e = p - t
        rm = [float(np.sqrt(np.mean((pp - tt) ** 2))) for tt, pp in zip(g["True"], g["Pred"])]
        row = {"Validation": scheme, "Target": tgt, "Unit": UNITS[tgt], "Model": m,
               "RMSE": float(np.sqrt(np.mean(e ** 2))), "MAE": float(np.mean(np.abs(e))),
               "Max_error": float(np.max(np.abs(e))), "R2": float(1 - np.sum(e ** 2) / np.sum((t - t.mean()) ** 2)),
               "RMSE_pct_range": float(100 * np.sqrt(np.mean(e ** 2)) / np.ptp(t)),
               "RMSE_trial_mean": float(np.mean(rm)), "RMSE_trial_sd": float(np.std(rm)),
               "Worst_trial": g["Trial"].iloc[int(np.argmax(rm))], "Worst_trial_RMSE": float(np.max(rm))}
        if g["SD"].iloc[0] is not None:
            sd = np.concatenate(g["SD"].tolist())
            row["GP_95pct_coverage"] = float(100 * np.mean(np.abs(e) <= 1.96 * sd))
        pooled.append(row)
        per_trial += [{"Validation": scheme, "Target": tgt, "Model": m, "Trial": tr_, "RMSE": r, "Unit": UNITS[tgt]}
                      for tr_, r in zip(g["Trial"], rm)]
    return pd.DataFrame(pooled), pd.DataFrame(per_trial)


def evaluate(scheme, folds, frames, rf_ok, full_frames):
    """folds: list of (train_trials, test_trial)."""
    rows, preds, bw_rows = [], [], []
    for k, (train, test) in enumerate(folds):
        t0 = time.time()
        out = run_fold(train, test, frames, rf_ok, k, full_frames)
        f = frames[test["name"]]
        for m, d in out.items():
            for tgt, (p, sd, params) in d.items():
                rows.append((test["name"], m, tgt, f[tgt].to_numpy(float), p, sd))
                preds.append(pd.DataFrame({"Validation": scheme, "Trial": test["name"], "Model": m, "Target": tgt,
                                           "Time_s": f["Time_s"], "True": f[tgt], "Pred": p,
                                           "SD": sd if sd is not None else np.nan}))
                if params is not None:
                    bw_rows.append({"Validation": scheme, "Test_trial": test["name"], "Model": m,
                                    "beta": params["beta"], "gamma": params["gamma"],
                                    **{f"c{i}": c for i, c in enumerate(params["coef"])}})
        print(f"  {scheme}: test {test['name']} (train {len(train)} trials) - {time.time() - t0:.0f} s")
    pooled, per_trial = metrics(rows, scheme)
    return pooled, per_trial, pd.concat(preds, ignore_index=True), pd.DataFrame(bw_rows)


# ============================================================
# FIGURES
# ============================================================

def save(fig, out_dir, name):
    png = os.path.join(out_dir, name)
    fig.savefig(png, dpi=150)
    print(f"Saved: {png}")
    plt.show()
    plt.close(fig)


def plot_errors(metrics_all, out_dir):
    schemes = list(dict.fromkeys(metrics_all["Validation"]))
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(7.5 * len(TARGETS), 5.2))
    for ax, tgt in zip(axes, TARGETS):
        models = [m for m in MODELS if tgt in MODELS[m][2] and m in set(metrics_all["Model"])]
        y = np.arange(len(models))[::-1]
        h = 0.8 / len(schemes)
        for k, sc in enumerate(schemes):
            e = metrics_all[(metrics_all["Validation"] == sc) & (metrics_all["Target"] == tgt)].set_index("Model")
            e = e.reindex(models)
            yy = y + (len(schemes) - 1) / 2 * h - k * h
            ax.barh(yy, e["RMSE"], h * 0.9, color=[MODEL_COLORS[m] for m in models],
                    alpha=1.0 if k == 0 else 0.55, hatch=None if k == 0 else "//", edgecolor="white")
            ax.errorbar(e["RMSE"], yy, xerr=e["RMSE_trial_sd"], fmt="none", ecolor="#3d3d3a", lw=0.8, capsize=2)
            for yi, r, sd, r2 in zip(yy, e["RMSE"], e["RMSE_trial_sd"], e["R2"]):
                if np.isfinite(r):
                    ax.text(r + sd + 0.01 * np.nanmax(metrics_all.loc[metrics_all["Target"] == tgt, "RMSE"]),
                            yi, f"{r:.1f} (R2 {r2:.2f})", va="center", fontsize=7)
        ax.set_yticks(y, models, fontsize=8)
        ax.set_xlabel(f"RMSE ({UNITS[tgt]})")
        ax.set_xlim(0, 1.35 * np.nanmax(metrics_all.loc[metrics_all["Target"] == tgt, "RMSE"]
                                        + metrics_all.loc[metrics_all["Target"] == tgt, "RMSE_trial_sd"]))
        ax.set_title(tgt, fontsize=10, fontweight="bold")
        ax.grid(axis="x", color="#e4e3df", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    labels = [f"{sc} (solid)" if k == 0 else f"{sc} (hatched)" for k, sc in enumerate(schemes)]
    fig.suptitle("M1  Real-time prediction error of every model:  " + "  |  ".join(labels)
                 + "\n(whisker = sd over test trials; red = uses the MTS displacement, reference only)",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    save(fig, out_dir, "M1_model_errors.png")


def plot_predictions(preds, trial, scheme, out_dir):
    show = {"Displacement_mm": ["Cubic calibration", "Two-branch cubic + curvature", "Gradient boosting",
                                "Gaussian process"],
            "Force_N": ["Two-branch cubic + curvature", "Gradient boosting", "Hybrid: boosted disp -> Bouc-Wen",
                        "Gaussian process"]}
    fig, axes = plt.subplots(len(TARGETS), 1, figsize=(16, 8), sharex=True)
    for ax, tgt in zip(axes, TARGETS):
        p = preds[(preds["Validation"] == scheme) & (preds["Trial"] == trial) & (preds["Target"] == tgt)]
        first = p[p["Model"] == p["Model"].iloc[0]]
        ax.plot(first["Time_s"], first["True"], color="#3d3d3a", lw=2.2, label="MTS (target)")
        for m in show[tgt]:
            q = p[p["Model"] == m]
            if q.empty:
                continue
            ax.plot(q["Time_s"], q["Pred"], color=MODEL_COLORS[m], lw=1.3, label=m)
            if q["SD"].notna().any():
                ax.fill_between(q["Time_s"], q["Pred"] - 1.96 * q["SD"], q["Pred"] + 1.96 * q["SD"],
                                color=MODEL_COLORS[m], alpha=0.15, lw=0, label="GP 95% band")
        ax.set_ylabel(f"{tgt.split('_')[0]} ({UNITS[tgt]})")
        ax.grid(color="#e4e3df", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower left")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"M2  {trial} ({scheme}): real-time prediction from the SEC only", fontweight="bold")
    fig.tight_layout()
    save(fig, out_dir, "M2_predictions.png")


def plot_loops(preds, trial, scheme, out_dir):
    models = ["Two-branch cubic + curvature", "Gradient boosting", "Bouc-Wen (SEC)",
              "Hybrid: boosted disp -> Bouc-Wen", "Bouc-Wen (true displacement)"]
    p = preds[(preds["Validation"] == scheme) & (preds["Trial"] == trial)]
    disp = p[(p["Target"] == "Displacement_mm") & (p["Model"] == "Cubic calibration")].set_index("Time_s")["True"]
    models = [m for m in models if not p[(p["Model"] == m) & (p["Target"] == "Force_N")].empty]
    fig, axes = plt.subplots(1, len(models), figsize=(4.4 * len(models), 4.2), sharey=True)
    for ax, m in zip(np.atleast_1d(axes), models):
        q = p[(p["Model"] == m) & (p["Target"] == "Force_N")].set_index("Time_s")
        x = disp.reindex(q.index)
        ax.plot(x, q["True"], color="#b8b7b2", lw=2.2, label="MTS")
        ax.plot(x, q["Pred"], color=MODEL_COLORS[m], lw=1.2, label="predicted")
        r = np.sqrt(np.mean((q["Pred"] - q["True"]) ** 2))
        ax.set_title(f"{m}\nRMSE {r:.1f} N", fontsize=9, fontweight="bold", color=MODEL_COLORS[m])
        ax.set_xlabel("MTS displacement (mm)")
        ax.grid(color="#e4e3df", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    np.atleast_1d(axes)[0].set_ylabel("Force (N)")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"M3  {trial} ({scheme}): does the model reproduce the force hysteresis loop?", fontweight="bold")
    fig.tight_layout()
    save(fig, out_dir, "M3_force_loops.png")


# ============================================================
# RUN
# ============================================================

out_dir = to_local_path(OUTPUT_DIR) if OUTPUT_DIR else os.path.join(to_local_path(DATA_DIR), "tda_results", "models")
os.makedirs(out_dir, exist_ok=True)
print("Loading and aligning:")
trials = [load_trial(*p) for p in find_pairs()]
use = [tr for tr in trials if tr["ok"]]
if len(use) < 3:
    raise RuntimeError("Need at least 3 usable trials.")
frames = {tr["name"]: features(tr) for tr in use}
full_frames = {tr["name"]: features(tr, start_s=0.0) for tr in use}       # hybrid: estimate from the start
rf_ok = have_sklearn()
if not rf_ok:
    print("scikit-learn not found: gradient boosting and Gaussian process skipped")
print(f"\n{len(use)} usable trials, {sum(len(f) for f in frames.values())} prediction times "
      f"(every {STEP_S:g} s from {SKIP_START_S:g} s)")

results = []
print("\nA  Leave-one-trial-out:")
results.append(evaluate("Leave-one-trial-out", [([t for t in use if t is not tr], tr) for tr in use], frames, rf_ok,
                        full_frames))
train = [t for t in use if t["name"] in TRAIN_TRIALS]
test = [t for t in use if t["name"] in TEST_TRIALS]
if train and test:
    scheme_b = f"Train {train[0]['name'][-2:]}-{train[-1]['name'][-2:]} -> test {test[0]['name'][-2:]}-{test[-1]['name'][-2:]}"
    print(f"\nB  {scheme_b}:")
    results.append(evaluate(scheme_b, [(train, tr) for tr in test], frames, rf_ok, full_frames))
else:
    scheme_b = None
    print(f"\nB  skipped: needs usable trials in TRAIN_TRIALS ({len(train)} found) and TEST_TRIALS ({len(test)} found)")

metrics_all = pd.concat([r[0] for r in results], ignore_index=True)
per_trial = pd.concat([r[1] for r in results], ignore_index=True)
preds = pd.concat([r[2] for r in results], ignore_index=True)
bw = pd.concat([r[3] for r in results], ignore_index=True)

for sc in metrics_all["Validation"].unique():
    for tgt in TARGETS:
        e = metrics_all[(metrics_all["Validation"] == sc) & (metrics_all["Target"] == tgt)]
        print(f"\n{sc}: {tgt} [{UNITS[tgt]}]")
        cols = ["Model", "RMSE", "MAE", "Max_error", "R2", "RMSE_pct_range", "RMSE_trial_sd", "Worst_trial",
                "Worst_trial_RMSE"] + (["GP_95pct_coverage"] if "GP_95pct_coverage" in e else [])
        print(e[cols].sort_values("RMSE").round(3).to_string(index=False))

plot_errors(metrics_all, out_dir)
scheme_plot = scheme_b or "Leave-one-trial-out"
example = EXAMPLE_TRIAL or (test[0]["name"] if scheme_b else use[0]["name"])
plot_predictions(preds, example, scheme_plot, out_dir)
plot_loops(preds, example, scheme_plot, out_dir)

print("\nBEST MODEL per target (lowest RMSE; Bouc-Wen with the true displacement is a reference only):")
for sc in metrics_all["Validation"].unique():
    for tgt in TARGETS:
        e = metrics_all[(metrics_all["Validation"] == sc) & (metrics_all["Target"] == tgt)
                        & (metrics_all["Model"] != "Bouc-Wen (true displacement)")].sort_values("RMSE")
        b, simple = e.iloc[0], e[e["Model"] == "Two-branch cubic + curvature"].iloc[0]
        print(f"  {sc:30s} {tgt:16s} {b['Model']:30s} RMSE {b['RMSE']:.2f} {b['Unit']} "
              f"({b['RMSE_pct_range']:.1f}% of range, R2 {b['R2']:.3f}); interpretable two-branch model "
              f"{simple['RMSE']:.2f} {b['Unit']}")

xlsx = os.path.join(out_dir, "SEC_MTS_models.xlsx")
with pd.ExcelWriter(xlsx, engine="openpyxl") as wr:
    for sc, g in metrics_all.groupby("Validation", sort=False):
        g.to_excel(wr, sheet_name="Metrics_LOTO" if sc == "Leave-one-trial-out" else "Metrics_Split", index=False)
    per_trial.to_excel(wr, sheet_name="Per_trial", index=False)
    preds.to_excel(wr, sheet_name="Predictions", index=False)
    bw.to_excel(wr, sheet_name="Bouc_Wen_params", index=False)
print(f"Saved: {xlsx}")
