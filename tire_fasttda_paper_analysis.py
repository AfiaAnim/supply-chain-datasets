"""
The Fast-TDA paper's analysis applied to the tire project
(Razmarashooli, Salazar Martinez, Laflamme, Hu, Schrader, Comert, Begashaw, Dodson:
 "Fast-TDA Implementation for High Rate Dynamic Systems in Noisy Environment and Introduction to Chaotic System").
Paste into ONE Jupyter cell and run.

    paper (DROPBEAR)                              tire project
    accelerometer on the beam (signal)        ->  SEC capacitance C1_raw_pF              (trial_NN_1.csv)
    cart / roller location in mm (target)     ->  MTS displacement in mm, and force in N (trial_NN.csv)
    three sections of the excitation          ->  sections = trials (each trial is one test)

What it reproduces
    Table 1    least-squares mapping target = a0 + a1*F for every feature F, with R^2
               (paper: x^2 = a0 + a1*F because f = E/L^2 for the beam; set MAPPING = "paper" to use that form)
    Figure 2   SEC time series + the target mapped from STFT, max H0, max H1 and the Fast-TDA minor axis
    Table 2    J1 (MAE), J2,i (% of time the error is above i), J3,i (MAE while the error is below i), per section
    Timing     ms per window for each feature (the paper's motivation: Fast-TDA is fast)
    Fig 3 / 4  window-size sensitivity of TDA (max H1) and Fast-TDA (minor axis)
    Fig 5 / 6  noise sensitivity on one window: nested sliding windows (w1 = 1/f_min + tau, w2 = 1/f_max + tau),
               fitted ellipses at local minima of the fit error, ellipse intersections, largest inscribed
               circles (Voronoi), and the H1 persistence diagram of TDA vs Fast-TDA
    Table 3/4  birth and death of the first 10 points, TDA vs Fast-TDA, clean vs noisy

Output (output_path):
    Paper_analysis_tables.xlsx        Table1, Table2, Timing, Window_sensitivity, Table3_4_noise, Windows
    Fig2_<target>_trialNN.png         per trial, and Fig2_<target>_all_trials.png
    Fig3_TDA_window_sizes.png, Fig4_FastTDA_window_sizes.png
    Fig5_6_noise_<condition>.png      clean (low-passed SEC), raw SEC (real sensor noise), + synthetic noise

Requirements:  pip install numpy pandas scipy scikit-learn matplotlib openpyxl ripser
"""
import glob
import os
import re
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, Rectangle
from scipy import signal
from scipy.signal import argrelmin
from scipy.spatial import Voronoi, cKDTree

try:                                    # make plots appear inside Jupyter
    from IPython import get_ipython
    if get_ipython() is not None:
        get_ipython().run_line_magic("matplotlib", "inline")
except Exception:
    pass

try:
    from ripser import ripser
except ImportError:
    raise ImportError("Install ripser:  %pip install ripser  (then restart the kernel)") from None


# =====================================================================================================
# === Parameters ===
# =====================================================================================================
base_path = "/home/barbara/tire_tda"     # folder with the SEC and MTS csv files (trial_NN*.csv)
output_path = None                       # None = <base_path>/tda_results/paper_analysis
trial_ids = None                         # None = every trial with an SEC and an MTS file; or e.g. [10, 11, 12]

signal_col = "C1_raw_pF"                 # signal (paper: beam acceleration)
targets = {                              # target name -> (MTS column, scale, unit, J thresholds i)
    "Displacement": ("Displacement_m", 1000.0, "mm", (2, 5, 10)),    # paper: 5, 10, 20 mm on a ~250 mm range
    "Force": ("Force_N", 1.0, "N", (10, 25, 50)),
}

freq = 20             # Hz after resampling
pass_value = 0.25     # low-pass cutoff in Hz (the "clean" signal)
good_alignment = 0.7  # SEC-MTS |corr| below this: trial flagged

# Feature windows (sliding, as in DROPBEAR): window = 1/min_f + 2*0.25/max_f, tau = 0.25/max_f / 2
min_f = 0.1           # -> window 20 s, tau 2.5 s (half a 40 s loading cycle: a turnaround traces a loop)
max_f = 0.05
s = 2                 # Takens stride
s_W = 10              # window step in samples (0.5 s)
STFT_FEATURE = "centroid"   # "centroid" = power-weighted mean frequency; "peak" = dominant frequency (paper)
                            # (the tire's loading frequency is fixed at 0.025 Hz, so "peak" barely moves)

MAPPING = "linear"          # "linear": target = a0 + a1*F ;  "paper": target^2 = a0 + a1*F (beam physics)
MAP_VALIDATION = "in_sample"   # "in_sample" = fit on all trials, as in the paper; "loto" = leave one trial out

# Window-size sensitivity (paper Fig 3 / 4): same tau, several windows
SENS_WINDOWS_S = [10, 15, 20, 30]
SENS_TRIAL = None           # trial drawn in Fig 3 / 4 (None = the first trial)

# Noise sensitivity (paper Fig 5 / 6, Tables 3 / 4): nested windows on one stretch of the signal
NESTED_MIN_F = 0.025        # loading frequency  -> w1 = 1/f_min + tau = 42.5 s (one full loop)
NESTED_MAX_F = 0.075        # 3rd harmonic       -> w2 = 1/f_max + tau = 15.8 s
NESTED_TAU_S = 2.5          # tau < 0.25/f_max = 3.3 s so that the cloud holds several ellipses (paper rule)
NOISE_START_S = None        # start of window 1 (None = the window with the largest max H1 in SENS_TRIAL)
NOISE_LEVEL = 0.05          # synthetic Gaussian noise, sd as a fraction of the window's range
FIT_ERR_ORDER = 10          # local-minimum neighbourhood (samples) of the fit error along window 2
MAX_AXIS_RATIO = 1.5        # an ellipse is accepted when its major semi-axis <= this x half the cloud's diagonal
OFFDIAG_EPS = 0.01          # a TDA point counts as "off-diagonal" when death - birth > this

INCLUDE_SEC_LEVEL = True    # extra baseline (not in the paper): the low-passed SEC value at the window end
FEATURES = ["STFT", "Max H0", "Max H1", "Fast-TDA"] + (["SEC level"] if INCLUDE_SEC_LEVEL else [])
COLORS = {"STFT": "#2bb6c4", "Max H0": "#8e44ad", "Max H1": "#e0457b", "Fast-TDA": "#1f6fb5", "SEC level": "#f0a020"}
STYLES = {"STFT": "-", "Max H0": "-.", "Max H1": ":", "Fast-TDA": (0, (1, 1)), "SEC level": "--"}
LABELS = {"STFT": "STFT", "Max H0": "Maximum persistence of H$_0$", "Max H1": "Maximum persistence of H$_1$",
          "Fast-TDA": "Minor axis in fitted ellipse", "SEC level": "SEC level (baseline)"}


# =====================================================================================================
# === Data loading (same as tire_h1_from_dropbear.py) ===
# =====================================================================================================
def to_local_path(path):
    """C:\\... -> /mnt/c/... when Jupyter runs under Linux / WSL; other paths unchanged."""
    path = str(path).strip().strip('"')
    if os.name != "nt" and len(path) > 2 and path[1] == ":" and path[2] in "\\/":
        return f"/mnt/{path[0].lower()}/{path[3:].replace(chr(92), '/')}"
    return path


def find_trials(base_path, trial_ids=None):
    base = os.path.abspath(base_path)
    if not os.path.isdir(base):
        raise FileNotFoundError(f"Folder not found: {base}\nSet base_path to the folder with the csv files")
    skip_dirs = {"lib", "lib64", "bin", "include", "share", "site-packages", "tda_results", ".ipynb_checkpoints"}
    files = [p for p in sorted(glob.glob(os.path.join(base, "**", "*.csv"), recursive=True))
             if not skip_dirs & set(os.path.relpath(p, base).split(os.sep)[:-1])]
    files = [p for p in files if re.search(r"trial[_\- ]?\d+", os.path.basename(p), re.I)]
    sec, mts = {}, {}
    for p in files:
        m = re.search(r"trial[_\- ]?(\d+)", os.path.basename(p), re.I)
        with open(p, "r", encoding="utf-8-sig", errors="replace") as f:
            header = f.readline()
        if signal_col in header and "time_s" in header:
            sec.setdefault(int(m.group(1)), p)
        elif "RunningTime_s" in header and "Displacement_m" in header:
            mts.setdefault(int(m.group(1)), p)
    found = sorted(set(sec) & set(mts))
    print(f"Searching {base}: SEC files for trials {sorted(sec)}, MTS files for trials {sorted(mts)}")
    return [(n, sec[n], mts[n]) for n in found if not trial_ids or n in trial_ids]


def load_trial(sec_file, mts_file):
    """SEC (filtered and raw) and MTS targets on one 20 Hz grid on the MTS clock."""
    sec, mts = pd.read_csv(sec_file), pd.read_csv(mts_file)
    t_sec = sec["time_s"].to_numpy(float)
    t_sec = t_sec - t_sec[0]
    raw_fs = (len(t_sec) - 1) / t_sec[-1]
    t_u = np.arange(0, t_sec[-1], 1 / raw_fs)
    cap_raw = np.interp(t_u, t_sec, sec[signal_col].to_numpy(float))
    sos = signal.butter(N=6, Wn=pass_value, btype="lowpass", fs=raw_fs, output="sos")
    cap_filt = signal.sosfiltfilt(sos, cap_raw)

    t_mts = mts["RunningTime_s"].to_numpy(float)
    t_mts = t_mts - t_mts[0]
    disp = mts["Displacement_m"].to_numpy(float)
    tt = np.arange(0, t_mts[-1], 0.1)                       # SEC starts ~9 s early: cross-correlation lag
    lags = [lag for lag in np.arange(5.0, 15.0, 0.1) if tt[-1] + lag <= t_u[-1]]
    corr = [abs(np.corrcoef(np.interp(tt + lag, t_u, cap_filt), np.interp(tt, t_mts, disp))[0, 1]) for lag in lags]
    offset, r = lags[int(np.argmax(corr))], max(corr)

    t = np.arange(0, min(t_mts[-1], t_u[-1] - offset), 1 / freq)
    out = {"time": t, "sec": np.interp(t + offset, t_u, cap_filt),
           "sec_raw": np.interp(t + offset, t_u, cap_raw)}     # raw = real sensor noise kept
    for name, (col, scale, unit, _) in targets.items():
        out[name] = np.interp(t, t_mts, mts[col].to_numpy(float) * scale)
    print(f"  SEC {raw_fs:.0f} Hz -> {freq} Hz, offset {offset:.1f} s (|corr| {r:.2f})"
          + ("" if r >= good_alignment else "  <-- alignment UNRELIABLE"))
    return out, r >= good_alignment


# =====================================================================================================
# === Features ===
# =====================================================================================================
def windows_of(x, w, step):
    return np.array([x[i:i + w] for i in range(0, len(x) - w + 1, step)])


def scale_pm1(w):
    lo, hi = w.min(), w.max()
    return np.zeros_like(w) if hi == lo else 2 * (w - lo) / (hi - lo) - 1


def embed(w, tau, stride=1):
    idx = np.arange(0, len(w) - tau, stride)
    return np.column_stack([w[idx], w[idx + tau]])


def stft_feature(w):
    """Frequency of the window (paper: STFT). Hann-tapered spectrum, band (0, pass_value]."""
    x = (w - w.mean()) * np.hanning(len(w))
    n = 8 * len(w)                                           # zero-padding: finer frequency grid
    P = np.abs(np.fft.rfft(x, n)) ** 2
    f = np.fft.rfftfreq(n, 1 / freq)
    band = (f > 0) & (f <= pass_value)
    if STFT_FEATURE == "peak":
        return f[band][np.argmax(P[band])]
    return float(np.sum(f[band] * P[band]) / np.sum(P[band]))


def vr_max_h0_h1(pc):
    """Max persistence of H0 (finite bars) and H1 (Vietoris-Rips, ripser)."""
    d = ripser(pc, maxdim=1)["dgms"]
    h0 = d[0][np.isfinite(d[0][:, 1])]
    h0 = float(np.max(h0[:, 1] - h0[:, 0])) if len(h0) else 0.0
    h1 = float(np.max(d[1][:, 1] - d[1][:, 0])) if len(d[1]) else 0.0
    return h0, h1


def fit_ellipse(pc):
    """Least-squares conic fit (SVD, A + C = 1). Returns (cx, cy, a, b, theta, rel_err) or None.
    a >= b are the semi-axes; rel_err = RMS geometric residual / b."""
    mx, my = pc[:, 0].mean(), pc[:, 1].mean()
    rng = max(np.abs(pc[:, 0] - mx).max(), np.abs(pc[:, 1] - my).max())
    if not np.isfinite(rng) or rng == 0 or len(pc) < 6:
        return None
    k = 1.5 / rng
    X, Y = (pc[:, 0] - mx) * k, (pc[:, 1] - my) * k
    G = np.column_stack([X**2 - Y**2, X * Y, X, Y, np.ones_like(X)])
    U, S, Vt = np.linalg.svd(G, full_matrices=False)
    Sinv = np.where(S > np.finfo(float).eps * max(G.shape) * S[0], 1.0 / S, 0.0)
    A, B, D, E, F = (Vt.T * Sinv) @ (U.T @ -(Y**2))
    C = 1.0 - A
    if 4 * A * C - B**2 <= 0:
        return None                                          # not an ellipse
    x0, y0 = np.linalg.solve([[2 * A, B], [B, 2 * C]], [-D, -E])
    F0 = A * x0**2 + B * x0 * y0 + C * y0**2 + D * x0 + E * y0 + F
    lam, vec = np.linalg.eigh([[A, B / 2], [B / 2, C]])
    ax2 = -F0 / lam
    if np.any(ax2 <= 0):
        return None
    semi = np.sqrt(ax2)                                      # semi-axis along each eigenvector
    i_major = int(np.argmax(semi))
    theta = float(np.arctan2(vec[1, i_major], vec[0, i_major]))
    Q = A * X**2 + B * X * Y + C * Y**2 + D * X + E * Y + F  # geometric residual |Q| / |grad Q|
    grad = np.hypot(2 * A * X + B * Y + D, B * X + 2 * C * Y + E)
    err = np.sqrt(np.mean((Q / np.maximum(grad, 1e-12)) ** 2)) / semi.min()
    # a least-squares conic can be an ellipse that runs far outside the points (open arcs):
    # keep it only when its centre lies inside the cloud's box and it is not much larger than the cloud
    lo, hi = pc.min(axis=0), pc.max(axis=0)
    cx, cy = x0 / k + mx, y0 / k + my
    if not (lo[0] <= cx <= hi[0] and lo[1] <= cy <= hi[1]) or semi.max() / k > MAX_AXIS_RATIO * np.hypot(*(hi - lo)) / 2:
        return None
    return (cx, cy, semi.max() / k, semi.min() / k, theta, float(err))


def fasttda_minor(pc):
    e = fit_ellipse(pc)
    return np.nan if e is None else 2 * e[3]                 # minor axis 2b (paper: Fast-TDA feature)


def features_for_trial(d, win, tau, timings):
    """One row per window (feature at the window END: real-time, as in the paper)."""
    W = windows_of(d["sec"], win, s_W)
    T = windows_of(d["time"], win, s_W)[:, -1]
    rows = {"Time": T}
    for name in targets:
        rows[name] = windows_of(d[name], win, s_W)[:, -1]
    feats = {f: np.zeros(len(W)) for f in FEATURES}
    for i, w in enumerate(W):
        t0 = time.perf_counter()
        feats["STFT"][i] = stft_feature(w)
        t1 = time.perf_counter()
        pc = embed(scale_pm1(w), tau, s)
        h0, h1 = vr_max_h0_h1(pc)
        t2 = time.perf_counter()
        feats["Max H0"][i], feats["Max H1"][i] = h0, h1
        feats["Fast-TDA"][i] = fasttda_minor(pc)
        t3 = time.perf_counter()
        if INCLUDE_SEC_LEVEL:
            feats["SEC level"][i] = w[-1]
        timings["STFT"].append(t1 - t0)
        timings["TDA (H0 + H1)"].append(t2 - t1)
        timings["Fast-TDA"].append(t3 - t2)
    rows.update(feats)
    return pd.DataFrame(rows)


# =====================================================================================================
# === Mapping feature -> target (paper Eq. 7) and the J metrics (paper Table 2) ===
# =====================================================================================================
def lse(F, y):
    """Least squares y = a0 + a1*F (paper: x^2 = a0 + a1*F). Returns a0, a1."""
    ok = np.isfinite(F) & np.isfinite(y)
    yy = y[ok] ** 2 if MAPPING == "paper" else y[ok]
    a1, a0 = np.polyfit(F[ok], yy, 1)
    return a0, a1


def apply_map(F, a0, a1, sign):
    z = a0 + a1 * F
    return sign * np.sqrt(np.maximum(z, 0)) if MAPPING == "paper" else z


def r2(y, yhat):
    ok = np.isfinite(y) & np.isfinite(yhat)
    return 1 - np.sum((y[ok] - yhat[ok]) ** 2) / np.sum((y[ok] - y[ok].mean()) ** 2)


def mapped(df, feat, target, validation=None):
    """Mapped target per window, in-sample (paper) or leave-one-trial-out."""
    sign = np.sign(np.nanmean(df[target])) or 1.0           # displacement is negative: keep its sign
    out = np.full(len(df), np.nan)
    if (validation or MAP_VALIDATION) == "loto":
        for tr in df.Trial.unique():
            tr_rows, te = df.Trial != tr, df.Trial == tr
            a0, a1 = lse(df.loc[tr_rows, feat].to_numpy(), df.loc[tr_rows, target].to_numpy())
            out[te.to_numpy()] = apply_map(df.loc[te, feat].to_numpy(), a0, a1, sign)
    else:
        a0, a1 = lse(df[feat].to_numpy(), df[target].to_numpy())
        out = apply_map(df[feat].to_numpy(), a0, a1, sign)
    return out


def j_metrics(y, yhat, thresholds):
    """J1 = MAE; J2,i = % of samples with |error| > i; J3,i = MAE of the samples with |error| <= i."""
    ok = np.isfinite(y) & np.isfinite(yhat)
    e = np.abs(y[ok] - yhat[ok])
    out = {"J1": e.mean()}
    for i in thresholds:
        out[f"J2,{i} (%)"] = 100 * np.mean(e > i)
    for i in thresholds:
        out[f"J3,{i}"] = e[e <= i].mean() if np.any(e <= i) else np.nan
    return out


# =====================================================================================================
# === Nested-window Fast-TDA: ellipses, intersections, inscribed circles (paper Fig 5 / 6) ===
# =====================================================================================================
def ellipse_polygon(e, n=180):
    cx, cy, a, b, th = e[:5]
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x, y = a * np.cos(t), b * np.sin(t)
    return np.column_stack([cx + x * np.cos(th) - y * np.sin(th), cy + x * np.sin(th) + y * np.cos(th)])


def clip_convex(subject, clip):
    """Sutherland-Hodgman: intersection of two convex polygons (both counter-clockwise)."""
    out = subject
    for i in range(len(clip)):
        if len(out) == 0:
            break
        p, q = clip[i], clip[(i + 1) % len(clip)]
        inside = lambda r: (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]) >= 0
        inp, new = out, []
        for j in range(len(inp)):
            cur, prev = inp[j], inp[j - 1]
            if inside(cur):
                if not inside(prev):
                    new.append(_cross(prev, cur, p, q))
                new.append(cur)
            elif inside(prev):
                new.append(_cross(prev, cur, p, q))
        out = np.array(new)
    return out


def _cross(a, b, p, q):
    d1, d2 = b - a, q - p
    den = d1[0] * d2[1] - d1[1] * d2[0]
    t = ((p[0] - a[0]) * d2[1] - (p[1] - a[1]) * d2[0]) / den
    return a + t * d1


def inscribed_circle(poly):
    """Largest circle inside a convex polygon: Voronoi vertices of the boundary are the candidate centres."""
    if len(poly) < 4:
        return None
    seg = np.vstack([poly, poly[:1]])
    pts = np.vstack([np.linspace(seg[i], seg[i + 1], 6, endpoint=False) for i in range(len(poly))])
    edges = np.diff(seg, axis=0)
    normals = np.column_stack([-edges[:, 1], edges[:, 0]]) / np.linalg.norm(edges, axis=1, keepdims=True)

    def dist(c):                                          # signed distance to every edge (inside: all >= 0)
        return np.min(np.sum((c[None, :] - poly) * normals, axis=1))
    try:
        cand = Voronoi(pts).vertices
    except Exception:
        cand = poly.mean(axis=0, keepdims=True)
    d = np.array([dist(c) for c in cand])
    if not np.any(d > 0):
        return None
    k = int(np.argmax(d))
    return cand[k], float(d[k])


def nested_fasttda(sig, w1, w2, tau):
    """Window 1 (w1 samples) holds the cloud; window 2 (w2 samples) slides inside it; an ellipse is
    captured where the fit error has a local minimum. Returns the cloud, ellipses, circles and the
    Fast-TDA diagram (birth = mean nearest-neighbour distance, death = diameters)."""
    pc = embed(scale_pm1(sig[:w1]), tau)
    n2 = w2 - tau
    fits = [fit_ellipse(pc[i:i + n2]) for i in range(0, len(pc) - n2 + 1)]
    err = np.array([f[5] if f is not None else np.inf for f in fits])
    finite = np.where(np.isfinite(err), err, np.nanmax(err[np.isfinite(err)]) * 10 if np.isfinite(err).any() else 1)
    mins = argrelmin(finite, order=FIT_ERR_ORDER, mode="clip")[0]
    ellipses = [fits[i] for i in mins if fits[i] is not None]
    outer = fit_ellipse(pc)
    if outer is not None:
        ellipses.insert(0, outer)
    kept = []                                              # drop near-duplicates
    for e in ellipses:
        if all(np.hypot(e[0] - k[0], e[1] - k[1]) > 0.05 or abs(e[3] - k[3]) > 0.02 for k in kept):
            kept.append(e)
    polys = [ellipse_polygon(e) for e in kept]

    circles = [((e[0], e[1]), e[3]) for e in kept]           # inscribed circle of an ellipse = minor axis
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            inter = clip_convex(polys[i], polys[j])
            c = inscribed_circle(inter) if len(inter) >= 4 else None
            if c is not None:
                circles.append(c)
    birth = float(np.mean(cKDTree(pc).query(pc, k=2)[0][:, 1]))
    deaths = sorted({round(2 * r, 4) for _, r in circles}, reverse=True)
    return pc, kept, circles, birth, deaths


def tda_diagram(pc):
    h1 = ripser(pc, maxdim=1)["dgms"][1]
    return h1[np.argsort(-(h1[:, 1] - h1[:, 0]))] if len(h1) else np.zeros((0, 2))


# =====================================================================================================
# === Run ===
# =====================================================================================================
base_path = to_local_path(base_path)
trials = find_trials(base_path, trial_ids)
if not trials:
    raise RuntimeError("No trial has both an SEC and an MTS file")
output_path = to_local_path(output_path) if output_path else os.path.join(base_path, "tda_results", "paper_analysis")
os.makedirs(output_path, exist_ok=True)

win = int(round(((1 / min_f) + (0.25 / max_f) * 2) * freq))
tau = int(round((0.25 / max_f) * freq / 2))
print(f"Feature window {win / freq:g} s, tau {tau / freq:g} s, step {s_W / freq:g} s, mapping '{MAPPING}', "
      f"validation '{MAP_VALIDATION}'")

data, frames = {}, []
timings = {"STFT": [], "TDA (H0 + H1)": [], "Fast-TDA": []}
for n, sp, mp in trials:
    print(f"trial_{n:02d}:")
    d, ok = load_trial(sp, mp)
    data[n] = d
    df = features_for_trial(d, win, tau, timings)
    df.insert(0, "Trial", f"trial_{n:02d}")
    df["Alignment_OK"] = ok
    frames.append(df)
    print(f"  {len(df)} windows")
W = pd.concat(frames, ignore_index=True)
sens_trial = SENS_TRIAL if SENS_TRIAL in data else trials[0][0]

# ---- Table 1: LSE mapping ----
table1 = []
for target in targets:
    sign = np.sign(np.nanmean(W[target])) or 1.0
    for f in FEATURES:
        a0, a1 = lse(W[f].to_numpy(), W[target].to_numpy())
        r2_in = r2(W[target].to_numpy(), apply_map(W[f].to_numpy(), a0, a1, sign))
        r2_loto = r2(W[target].to_numpy(), mapped(W, f, target, "loto"))
        W[f"{target} from {f}"] = mapped(W, f, target)
        table1.append({"Target": target, "Method": f, "a0": a0, "a1": a1, "R2 (in-sample, paper)": r2_in,
                       "R2 (leave one trial out)": r2_loto, "Fast-TDA fit failures %":
                       100 * W[f].isna().mean() if f == "Fast-TDA" else 0.0})
table1 = pd.DataFrame(table1)
print("\nTable 1 - LSE mapping " + ("target = a0 + a1*F" if MAPPING == "linear" else "target^2 = a0 + a1*F"))
print(table1.round(3).to_string(index=False))

# ---- Table 2: J metrics per section (= trial) ----
table2 = []
for target, (_, _, unit, th) in targets.items():
    for tr, g in W.groupby("Trial"):
        for f in FEATURES:
            m = j_metrics(g[target].to_numpy(), g[f"{target} from {f}"].to_numpy(), th)
            table2.append({"Target": target, "Unit": unit, "Section": tr, "Method": f, **m})
    for f in FEATURES:                                   # all sections together
        m = j_metrics(W[target].to_numpy(), W[f"{target} from {f}"].to_numpy(), th)
        table2.append({"Target": target, "Unit": unit, "Section": "All", "Method": f, **m})
table2 = pd.DataFrame(table2)
print("\nTable 2 - performance per section (J1, J3 in the target's unit; J2 in %)")
for target in targets:
    print(f"\n{target}:")
    th = targets[target][3]
    cols = ["Section", "Method", "J1"] + [f"J2,{i} (%)" for i in th] + [f"J3,{i}" for i in th]
    print(table2.loc[table2.Target == target, cols].round(2).to_string(index=False))

timing = pd.DataFrame([{"Method": k, "ms per window (mean)": 1000 * np.mean(v), "ms per window (max)":
                        1000 * np.max(v)} for k, v in timings.items()])
print("\nComputation time\n" + timing.round(3).to_string(index=False))

# ---- Figure 2: mapped target from every feature over the SEC signal ----
def fig2(ax, g, d, target, unit, show_legend):
    t0, t1 = g.Time.min() - win / freq, g.Time.max()
    keep = (d["time"] >= 0) & (d["time"] <= t1)
    ax.plot(d["time"][keep], d["sec"][keep], color="0.82", lw=0.8, label="Time series (SEC)")
    ax.set_ylabel("Capacitance (pF)", color="0.5")
    ax2 = ax.twinx()
    for f in FEATURES:
        ax2.plot(g.Time, g[f"{target} from {f}"], color=COLORS[f], ls=STYLES[f], lw=1.3, label=LABELS[f])
    ax2.plot(g.Time, g[target], color="k", lw=1.6, label=f"MTS {target.lower()}")
    lo, hi = ax.get_ylim()
    ax.add_patch(Rectangle((0, lo), win / freq, hi - lo, fill=False, ec="green", ls="--", lw=1.5, label="Window"))
    ax2.set_ylabel(f"Mapped {target.lower()} ({unit})", fontweight="bold")
    ax.set_xlabel("Time (s)")
    if show_legend:
        h = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
        l = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
        ax.figure.legend(h, l, loc="upper center", ncol=4, fontsize=9, frameon=True)


for target, (_, _, unit, _) in targets.items():
    for n in data:
        g = W[W.Trial == f"trial_{n:02d}"]
        fig, ax = plt.subplots(figsize=(12, 4.6))
        fig2(ax, g, data[n], target, unit, True)
        ax.set_title(f"trial_{n:02d}: {target.lower()} localisation from SEC features", pad=4)
        fig.tight_layout(rect=(0, 0, 1, 0.86))
        fig.savefig(os.path.join(output_path, f"Fig2_{target}_trial{n:02d}.png"), dpi=150)
        plt.close(fig)
    fig, axes = plt.subplots(len(data), 1, figsize=(12, 2.6 * len(data) + 1), squeeze=False)
    for k, n in enumerate(data):
        fig2(axes[k, 0], W[W.Trial == f"trial_{n:02d}"], data[n], target, unit, k == 0)
        axes[k, 0].set_title(f"trial_{n:02d}", fontsize=10, loc="left")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.9 / (2.6 * len(data) + 1)))
    fig.savefig(os.path.join(output_path, f"Fig2_{target}_all_trials.png"), dpi=120)
    plt.show() if target == list(targets)[0] else plt.close(fig)

# ---- Figures 3 / 4 and the window-size table ----
sens_rows, sens_series = [], {}
for ws in SENS_WINDOWS_S:
    w = int(round(ws * freq))
    if w <= 2 * tau + 6:
        continue
    parts = []
    for n, d in data.items():
        Wn = windows_of(d["sec"], w, s_W)
        h1 = np.array([vr_max_h0_h1(embed(scale_pm1(x), tau, s))[1] for x in Wn])
        fm = np.array([fasttda_minor(embed(scale_pm1(x), tau, s)) for x in Wn])
        df = pd.DataFrame({"Trial": n, "Time": windows_of(d["time"], w, s_W)[:, -1], "Max H1": h1, "Fast-TDA": fm})
        for target in targets:
            df[target] = windows_of(d[target], w, s_W)[:, -1]
        parts.append(df)
    P = pd.concat(parts, ignore_index=True)
    sens_series[ws] = P
    for f in ["Max H1", "Fast-TDA"]:
        row = {"Window (s)": ws, "Method": f, "Fit failures %": 100 * P[f].isna().mean()}
        for target in targets:
            a0, a1 = lse(P[f].to_numpy(), P[target].to_numpy())
            row[f"R2 {target}"] = r2(P[target].to_numpy(),
                                     apply_map(P[f].to_numpy(), a0, a1, np.sign(np.nanmean(P[target])) or 1.0))
        sens_rows.append(row)
sens = pd.DataFrame(sens_rows)
ref = 20 if 20 in sens_series else SENS_WINDOWS_S[len(SENS_WINDOWS_S) // 2]
for f in ["Max H1", "Fast-TDA"]:                       # agreement with the reference window, same end times
    for ws, P in sens_series.items():
        m = P.merge(sens_series[ref], on=["Trial", "Time"], suffixes=("", "_ref"))
        sens.loc[(sens["Window (s)"] == ws) & (sens.Method == f), f"Corr with {ref} s window"] = \
            m[f].corr(m[f + "_ref"])
print("\nWindow-size sensitivity (paper Fig 3 / 4)\n" + sens.round(3).to_string(index=False))

wcolors = ["#f0a020", "#2040d0", "#2a9d4b", "#e03030", "#8e44ad", "#555555"]
wstyles = ["-", ":", "-.", "--", "-", ":"]
for fig_no, f, ylab in [(3, "Max H1", "H$_1$ TDA feature"), (4, "Fast-TDA", "Minor axis Fast-TDA feature")]:
    d = data[sens_trial]
    fig, ax = plt.subplots(figsize=(12, 4.4))
    ax.plot(d["time"], d["sec"], color="0.55", lw=0.8, label="Time series")
    ax.set_ylabel("Capacitance (pF)", color="0.5")
    ax2 = ax.twinx()
    lo, hi = ax.get_ylim()
    for k, (ws, P) in enumerate(sens_series.items()):
        g = P[P.Trial == sens_trial]
        ax2.plot(g.Time, g[f], color=wcolors[k], ls=wstyles[k], lw=1.4,
                 label=f"{'H$_1$ feature' if f == 'Max H1' else 'Minor axis length'} - {ws:g} s")
        ax.add_patch(Rectangle((0, lo), ws, hi - lo, fill=False, ec=wcolors[k], ls="--", lw=1.6,
                               label=f"Window size {ws:g} s"))
    ax2.set_ylabel(ylab, fontweight="bold")
    ax.set_xlabel("Time (s)")
    h = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    l = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    fig.legend(h, l, loc="upper center", ncol=4, fontsize=9)
    ax.set_title(f"Figure {fig_no}: {'TDA' if f == 'Max H1' else 'Fast-TDA'} feature evolution for various "
                 f"window sizes (trial_{sens_trial:02d})", pad=4)
    fig.tight_layout(rect=(0, 0, 1, 0.83))
    fig.savefig(os.path.join(output_path, f"Fig{fig_no}_{'TDA' if f == 'Max H1' else 'FastTDA'}_window_sizes.png"),
                dpi=150)
    plt.show()

# ---- Figures 5 / 6 and Tables 3 / 4: noise sensitivity ----
w1 = int(round((1 / NESTED_MIN_F + NESTED_TAU_S) * freq))
w2 = int(round((1 / NESTED_MAX_F + NESTED_TAU_S) * freq))
ntau = int(round(NESTED_TAU_S * freq))
d = data[sens_trial]
if NOISE_START_S is None:                              # window 1 ending at the largest max H1 of the trial
    g = W[W.Trial == f"trial_{sens_trial:02d}"]
    t_end = g.Time.iloc[int(np.nanargmax(g["Max H1"]))]
    start = int(np.clip(round(t_end * freq) - w1 + 1, 0, len(d["sec"]) - w1))
else:
    start = int(round(NOISE_START_S * freq))
clean = d["sec"][start:start + w1]
rng = np.random.default_rng(0)
conditions = {
    "clean": ("Low-passed SEC (noise free)", clean),
    "raw": ("Raw SEC (real sensor noise)", d["sec_raw"][start:start + w1]),
    "synthetic": (f"Low-passed SEC + {NOISE_LEVEL:.0%} Gaussian noise",
                  clean + rng.normal(0, NOISE_LEVEL * np.ptp(clean), len(clean))),
}
print(f"\nNoise sensitivity: trial_{sens_trial:02d}, window 1 = {w1 / freq:g} s from t = {start / freq:g} s, "
      f"window 2 = {w2 / freq:g} s, tau = {ntau / freq:g} s")

rows34, summary = [], {}
for key, (label, sig) in conditions.items():
    pc, ellipses, circles, birth, deaths = nested_fasttda(sig, w1, w2, ntau)
    dg = tda_diagram(pc)
    pers = dg[:, 1] - dg[:, 0] if len(dg) else np.zeros(0)
    summary[key] = {"TDA max persistence": float(pers.max()) if len(pers) else 0.0,
                    "TDA off-diagonal points": int(np.sum(pers > OFFDIAG_EPS)),
                    "Fast-TDA max persistence": (deaths[0] - birth) if deaths else 0.0,
                    "Fast-TDA points": len(deaths), "Fast-TDA ellipses": len(ellipses)}
    for name, b, dd in [("TDA (H1)", dg[:10, 0], dg[:10, 1]),
                        ("Fast-TDA (Diameter)", np.full(min(10, len(deaths)), birth), np.array(deaths[:10]))]:
        for feat, vals in [("Birth", b), ("Death", dd)]:
            rows34.append({"Condition": label, "Method": name, "Feature": feat,
                           **{f"Point {i + 1}": round(float(v), 3) for i, v in enumerate(vals)}})

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw={"width_ratios": [1.25, 1, 1]})
    tt = np.arange(w1) / freq
    axes[0].plot(tt, sig, color="0.6", lw=0.9, label="Time series")
    lo, hi = sig.min(), sig.max()
    axes[0].add_patch(Rectangle((0, lo), w2 / freq, hi - lo, fill=False, ec="red", ls="--", lw=1.6, label="Window 2"))
    axes[0].add_patch(Rectangle((0, lo), w1 / freq, hi - lo, fill=False, ec="green", ls="--", lw=1.6, label="Window 1"))
    axes[0].set(xlabel="Time (s)", ylabel="Capacitance (pF)", title="(a)")
    axes[0].legend(fontsize=8, loc="upper right")
    axes[1].scatter(pc[:, 0], pc[:, 1], s=4, color="0.45", label="Point cloud")
    for k, e in enumerate(ellipses):
        p = np.vstack([ellipse_polygon(e), ellipse_polygon(e)[:1]])
        axes[1].plot(p[:, 0], p[:, 1], color="red", lw=1.1, label="Fitted ellipses" if k == 0 else None)
    for k, (c, r) in enumerate(circles):
        axes[1].add_patch(Circle(c, r, fill=False, ec="k", lw=1.0, label="Inscribed circles" if k == 0 else None))
    axes[1].set(xlabel=r"$x(t)$", ylabel=r"$x(t-\tau)$", title="(b)", aspect="equal")
    axes[1].legend(fontsize=8, loc="lower right")
    top = max([0.05] + list(dg[:, 1] if len(dg) else []) + deaths) * 1.1
    axes[2].scatter(dg[:, 0], dg[:, 1], color="#3050e0", s=22, label="TDA")
    axes[2].scatter(np.full(len(deaths), birth), deaths, color="#e03030", s=22, label="Fast TDA")
    axes[2].plot([0, top], [0, top], "k:", lw=1)
    axes[2].set(xlim=(0, top), ylim=(0, top), xlabel="Birth", ylabel="Death", title="(c)", aspect="equal")
    axes[2].legend(fontsize=8, loc="lower right")
    axes[2].grid(alpha=0.3)
    fig.suptitle(f"TDA and Fast-TDA feature comparison: {label} (trial_{sens_trial:02d})")
    fig.tight_layout()
    fig.savefig(os.path.join(output_path, f"Fig5_6_noise_{key}.png"), dpi=150)
    plt.show()

table34 = pd.DataFrame(rows34)
noise = pd.DataFrame(summary).T
for col in ["TDA max persistence", "Fast-TDA max persistence"]:
    noise[f"{col} change vs clean %"] = 100 * (noise[col] / noise.loc["clean", col] - 1)
print("\nTables 3 / 4 - birth and death of the first 10 points\n" + table34.to_string(index=False))
print("\nNoise effect (paper: TDA max persistence +60%, Fast-TDA +1%)\n" + noise.round(3).to_string())

with pd.ExcelWriter(os.path.join(output_path, "Paper_analysis_tables.xlsx")) as xw:
    table1.to_excel(xw, sheet_name="Table1_LSE", index=False)
    table2.to_excel(xw, sheet_name="Table2_J_metrics", index=False)
    timing.to_excel(xw, sheet_name="Timing", index=False)
    sens.to_excel(xw, sheet_name="Window_sensitivity", index=False)
    table34.to_excel(xw, sheet_name="Table3_4_noise", index=False)
    noise.to_excel(xw, sheet_name="Noise_summary")
    W.to_excel(xw, sheet_name="Windows", index=False)
print(f"\nAll tables and figures saved to: {output_path}")
