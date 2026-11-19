"""
Tidal boundary condition (signal) vs ditch water depth (target): Fourier, sliding-window TDA and the
feature comparison (VR H1, Alpha H1, FastTDA ellipse, bivariate-Gaussian ellipse, genus-1 cubic oval),
and whether each feature improves the prediction of the ditch depth. Paste into ONE Jupyter cell.

Data (one Excel workbook per tide event, found automatically in base_path):
    sheet "BC Data" / "Boundary Condition Data"  DateTime, WSE (NAVD88) m   tide at 6202 Old Point Road
    sheet "Validation Data"                      DateTime, depth from channel bed (m)   ditch, 1235 Hillside Dr
    the notes column names the tide type (Regular Tide / King Tide)

Steps:
    1. both series on one time grid (freq samples per hour); with harmonize_hourly the finer record is
       first reduced to hourly values so both events carry the same information (Nov is ~hourly)
    2. tide -> ditch lag (cross-correlation); FFT of both -> min_f (dominant tidal frequency) and max_f
       (highest significant harmonic) -> window and delay by the DROPBEAR formulas
    3. trailing windows (real time): point clouds of the tide (Takens), of the ditch depth (Takens) and
       of the tide-ditch loop (tide vs depth), and per cloud: VR H1, Alpha H1, FastTDA 2b, BV-Gaussian
       minor axis and axis ratio, cubic (genus-1) oval size
    4. comparison: fit failures, speed, correlation with the ditch depth, Regular vs King tide separation
    5. prediction of the ditch depth from the tide only (leave-one-day-out over both events): tide level,
       threshold (hinge) model, + memory, + each tide shape feature, gradient boosting

Output (in output_path): tide_tda_windows.xlsx (all windows + tables), F1_overview.png,
    F2_window_<event>.png, F3_feature_comparison.png, F4_prediction.png

Requirements:
    pip install numpy pandas scipy matplotlib openpyxl ripser gudhi scikit-learn
"""
import glob
import os
import re
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import PchipInterpolator

warnings.filterwarnings("ignore", category=RuntimeWarning)
try:                                    # make plots appear inside Jupyter
    from IPython import get_ipython
    if get_ipython() is not None:
        get_ipython().run_line_magic("matplotlib", "inline")
except Exception:
    pass

# === Parameters ===
base_path = "/home/barbara/tire_tda"   # folder with the tide workbooks (*.xlsx); searched with subfolders
files = None                  # None = every .xlsx with a "Validation" sheet; or a list of file paths
output_path = None            # None = base_path/tda_results/tide

freq = 4                      # analysis grid: samples per hour (4 = 15 min)
harmonize_hourly = True       # reduce every record to hourly values first (Nov is ~hourly, Dec 15 min):
                              # otherwise shape features would separate the events by sampling alone
pass_value = 0.5              # low-pass cutoff (cycles per hour) - keeps tides up to 2 h periods
min_f = None                  # None = dominant tidal frequency from the FFT (semidiurnal, ~1/12.4 h)
max_f = None                  # None = highest significant harmonic in the spectra (DROPBEAR rule ->
                              # tau 0.5 h); max_f = min_f gives tau ~1.5 h, where the combined tide TDA
                              # features improved the gradient-boosting prediction by ~11% (Nov/Dec 2025)
#   window = (1 / min_f + 2 * 0.25 / max_f) * freq,   tau = (0.25 / max_f) * freq / 2   (DROPBEAR)
s = 1                         # Takens stride
s_W = 1                       # window step (samples)
lag_search_h = (-2.0, 6.0)    # tide -> ditch lag searched in this range (hours)
COMPUTE = {"VR H1": True, "Alpha H1": True, "FastTDA": True, "BVG minor": True, "BVG ratio": True,
           "Cubic oval": True}
CLOUDS = ["tide", "ditch", "loop"]   # tide = Takens of the tide, ditch = Takens of the depth,
                                     # loop = tide vs depth in the window (input-output hysteresis)
COLORS = {"VR H1": "#6250d6", "Alpha H1": "#1baf7a", "FastTDA": "#eb6834", "BVG minor": "#2a78d6",
          "BVG ratio": "#0d366b", "Cubic oval": "#e87ba4"}


# === Loading ===
def to_local_path(path):
    path = str(path).strip().strip('"')
    if os.name != "nt" and len(path) > 2 and path[1] == ":" and path[2] in "\\/":
        return f"/mnt/{path[0].lower()}/{path[3:].replace(chr(92), '/')}"
    return path


def find_workbooks(base):
    skip = {"lib", "lib64", "bin", "site-packages", "tda_results", ".ipynb_checkpoints"}
    out = []
    for p in sorted(glob.glob(os.path.join(base, "**", "*.xlsx"), recursive=True)):
        if skip & set(os.path.relpath(p, base).split(os.sep)[:-1]) or os.path.basename(p).startswith("~$"):
            continue
        try:
            names = pd.ExcelFile(p).sheet_names
        except Exception:
            continue
        if any("valid" in n.lower() for n in names):
            out.append(p)
    return out


def read_series(path, sheet):
    d = pd.read_excel(path, sheet_name=sheet, header=None)
    t = pd.to_datetime(d[0].iloc[1:], errors="coerce")
    v = pd.to_numeric(d[1].iloc[1:], errors="coerce")
    ok = t.notna() & v.notna()
    notes = [str(x) for x in d[d.columns[-1]].dropna().tolist()]
    return pd.Series(v[ok].values, index=pd.DatetimeIndex(t[ok].values)).sort_index(), str(d.iloc[0, 1]), notes


def load_event(path):
    names = pd.ExcelFile(path).sheet_names
    bc_sheet = next(n for n in names if "valid" not in n.lower())
    val_sheet = next(n for n in names if "valid" in n.lower())
    bc, bc_name, notes = read_series(path, bc_sheet)
    val, val_name, _ = read_series(path, val_sheet)
    tide_type = next((re.sub(r"^Tide Type:\s*", "", n) for n in notes if "tide" in n.lower()), "Tide")
    t0, t1 = max(bc.index[0], val.index[0]), min(bc.index[-1], val.index[-1])
    name = f"{tide_type} ({t0:%d %b}-{t1:%d %b %Y})"
    step = lambda x: float(np.median(np.diff(x.index.values).astype("timedelta64[s]").astype(float))) / 60
    print(f"  {os.path.basename(path)}: {name}; tide '{bc_name}' every {step(bc):.0f} min, "
          f"ditch '{val_name}' every {step(val):.0f} min; overlap {(t1 - t0).total_seconds() / 3600:.1f} h")
    return {"name": name, "type": tide_type, "bc": bc, "val": val, "t0": t0, "t1": t1}


def on_grid(series, t0, t1):
    """Series on the analysis grid (freq per hour): optionally hourly first, then PCHIP, then low-pass."""
    hrs = lambda idx: (idx - t0).total_seconds().values / 3600
    x, y = hrs(series.index), series.values.astype(float)
    if harmonize_hourly:
        xh = np.arange(np.ceil(x[0]), np.floor(min(x[-1], (t1 - t0).total_seconds() / 3600)) + 1e-9, 1.0)
        x, y = xh, np.interp(xh, x, y)
    grid = np.arange(0, (t1 - t0).total_seconds() / 3600, 1 / freq)
    grid = grid[(grid >= x[0]) & (grid <= x[-1])]
    yg = PchipInterpolator(x, y)(grid)
    if pass_value and pass_value < freq / 2:
        yg = signal.sosfiltfilt(signal.butter(4, pass_value, btype="lowpass", fs=freq, output="sos"), yg)
    return grid, yg


def prepare(ev):
    gt, tide = on_grid(ev["bc"], ev["t0"], ev["t1"])
    gv, depth = on_grid(ev["val"], ev["t0"], ev["t1"])
    lo, hi = max(gt[0], gv[0]), min(gt[-1], gv[-1])
    h = np.arange(lo, hi + 1e-9, 1 / freq)
    ev["h"], ev["tide"], ev["depth"] = h, np.interp(h, gt, tide), np.interp(h, gv, depth)
    lags = np.arange(int(lag_search_h[0] * freq), int(lag_search_h[1] * freq) + 1)
    n = len(h)
    cc = [np.corrcoef(ev["tide"][max(0, -L):n - max(0, L)], ev["depth"][max(0, L):n - max(0, -L)])[0, 1] for L in lags]
    ev["lag_h"], ev["lag_r"] = lags[int(np.argmax(cc))] / freq, float(np.max(cc))


# === Fourier ===
def spectrum(y):
    y = y - y.mean()
    n = 8 * len(y)
    amp = np.abs(np.fft.rfft(y * np.hanning(len(y)), n=n))
    return np.fft.rfftfreq(n, 1 / freq), amp


def spectral_peaks(f, amp, record_h, rel=0.25):
    pk = signal.find_peaks(amp)[0]
    pk = pk[(f[pk] >= 2 / record_h) & (f[pk] <= pass_value)]
    pk = pk[amp[pk] >= rel * amp[pk].max()]
    return sorted(pk, key=lambda i: -amp[i])


# === Features ===
def fasttda_fit(pc):
    """SVD-LSE ellipse fit (A + C = 1) -> (minor axis 2b, ellipse curve) or (NaN, None)."""
    mx, my = pc[:, 0].mean(), pc[:, 1].mean()
    X, Y = pc[:, 0] - mx, pc[:, 1] - my
    rng = max(np.abs(X).max(), np.abs(Y).max())
    if not np.isfinite(rng) or rng == 0:
        return np.nan, None
    k = 1.5 / rng
    X, Y = X * k, Y * k
    G = np.column_stack([X**2 - Y**2, X * Y, X, Y, np.ones_like(X)])
    U, S, Vt = np.linalg.svd(G, full_matrices=False)
    Sinv = np.where(S > np.finfo(float).eps * max(G.shape) * S[0], 1.0 / S, 0.0)
    A, B, D, E, F = (Vt.T * Sinv) @ (U.T @ -(Y**2))
    C = 1.0 - A
    if 4 * A * C - B**2 <= 0:
        return np.nan, None
    th = 0.5 * np.arctan2(B, A - C)
    c, sn = np.cos(th), np.sin(th)
    Ap, Cp = A*c*c + B*c*sn + C*sn*sn, A*sn*sn - B*c*sn + C*c*c
    Dp, Ep = D*c + E*sn, -D*sn + E*c
    if Ap <= 0 or Cp <= 0:
        return np.nan, None
    Fn = F - Dp**2 / (4 * Ap) - Ep**2 / (4 * Cp)
    if Fn >= 0:
        return np.nan, None
    a_u, a_v = np.sqrt(-Fn / Ap), np.sqrt(-Fn / Cp)
    u0, v0 = -Dp / (2 * Ap), -Ep / (2 * Cp)
    ang = np.linspace(0, 2 * np.pi, 200)
    u, v = u0 + a_u * np.cos(ang), v0 + a_v * np.sin(ang)
    return 2.0 * min(a_u, a_v), (mx + (c * u - sn * v) / k, my + (sn * u + c * v) / k)


def bvg(pc):
    """Bivariate-Gaussian (covariance) ellipse: 2-sigma minor axis (4 sqrt(lambda_min)) and axis ratio."""
    lam, vec = np.linalg.eigh(np.cov(pc.T))
    lam = np.maximum(lam, 0)
    return 4 * np.sqrt(lam[0]), (np.sqrt(lam[0] / lam[1]) if lam[1] > 0 else np.nan), lam, vec


GX, GY = np.meshgrid(np.linspace(-1.4, 1.4, 140), np.linspace(-1.4, 1.4, 140))


def cubic_oval(pc):
    """Implicit cubic (genus-1, elliptic-type curve) by algebraic least squares; equivalent diameter of
    its largest closed oval inside the box (0 = no oval). Exploratory."""
    x, y = pc[:, 0], pc[:, 1]
    M = np.column_stack([x**3, x*x*y, x*y*y, y**3, x*x, x*y, y*y, x, y, np.ones_like(x)])
    c = np.linalg.svd(M, full_matrices=False)[2][-1]
    Fv = (c[0]*GX**3 + c[1]*GX*GX*GY + c[2]*GX*GY*GY + c[3]*GY**3 + c[4]*GX*GX + c[5]*GX*GY + c[6]*GY*GY
          + c[7]*GX + c[8]*GY + c[9])
    cs = plt.contour(GX, GY, Fv, levels=[0])
    plt.close()
    best = 0.0
    for seg in cs.allsegs[0]:
        if len(seg) > 8 and np.linalg.norm(seg[0] - seg[-1]) < 1e-6 and np.abs(seg).max() < 1.38:
            best = max(best, 0.5 * abs(np.dot(seg[:, 0], np.roll(seg[:, 1], 1)) - np.dot(seg[:, 1], np.roll(seg[:, 0], 1))))
    return 2 * np.sqrt(best / np.pi)


H1_BACKEND = None


def vr_h1(pc):
    global H1_BACKEND
    if H1_BACKEND in (None, "gtda"):
        try:
            from gtda.homology import VietorisRipsPersistence
            dg = VietorisRipsPersistence(homology_dimensions=(1,), n_jobs=1).fit_transform(pc[None])
            H1_BACKEND = "gtda"
            return float(np.max(dg[0, :, 1] - dg[0, :, 0]))
        except Exception:
            H1_BACKEND = "ripser"
    from ripser import ripser
    h1 = ripser(pc, maxdim=1)["dgms"][1]
    return float(np.max(h1[:, 1] - h1[:, 0])) if len(h1) else 0.0


def alpha_h1(pc):
    try:
        import gudhi as gd
    except ImportError:
        return np.nan
    try:
        st = gd.AlphaComplex(points=pc.tolist()).create_simplex_tree()
        st.persistence()
        iv = np.asarray(st.persistence_intervals_in_dimension(1), dtype=float)
    except Exception:
        return np.nan
    iv = iv[np.isfinite(iv).all(axis=1) & (iv[:, 1] > iv[:, 0])] if iv.size else iv
    if iv.size == 0:
        return 0.0
    sc = 2.0 * np.sqrt(np.maximum(iv, 0.0))
    return float(np.max(sc[:, 1] - sc[:, 0]))


def scale(w):
    lo, hi = w.min(), w.max()
    return 2 * (w - lo) / (hi - lo) - 1 if hi > lo else np.zeros_like(w)


def clouds_of(ev, e, w, tau):
    """Point clouds of the trailing window ending at index e."""
    tw, dw = ev["tide"][e - w + 1:e + 1], ev["depth"][e - w + 1:e + 1]
    st, sd = scale(tw), scale(dw)
    emb = lambda z: np.column_stack([z[:-tau:s], z[tau::s]])
    return {"tide": emb(st), "ditch": emb(sd), "loop": np.column_stack([st[::s], sd[::s]])}


FEATURE_FUNS = {"VR H1": vr_h1, "Alpha H1": alpha_h1, "FastTDA": lambda pc: fasttda_fit(pc)[0],
                "BVG minor": lambda pc: bvg(pc)[0], "BVG ratio": lambda pc: bvg(pc)[1], "Cubic oval": cubic_oval}


def window_features(ev, w, tau):
    rows, timing = [], {f: 0.0 for f in FEATURE_FUNS}
    for e in range(w - 1, len(ev["h"]), s_W):
        cl = clouds_of(ev, e, w, tau)
        row = {"Event": ev["name"], "Tide_type": ev["type"], "Hour": ev["h"][e], "Time": ev["t0"] + pd.Timedelta(hours=ev["h"][e]),
               "Tide_WSE_m": ev["tide"][e], "Ditch_depth_m": ev["depth"][e]}
        for f, fun in FEATURE_FUNS.items():
            if not COMPUTE[f]:
                continue
            for c in CLOUDS:
                t0 = time.perf_counter()
                row[f"{c}: {f}"] = fun(cl[c])
                timing[f] += time.perf_counter() - t0
        rows.append(row)
    return pd.DataFrame(rows), {f: 1000 * v / (len(rows) * len(CLOUDS)) for f, v in timing.items()}


# === Prediction of the ditch depth from the tide only ===
def tide_memory(ev):
    """Per grid time: tide now and 1-3 h ago, 1 h slope (rising / falling), hours since the tide was last
    above the flooding level (drain-down memory) - all from the tide only."""
    t, n = ev["tide"], len(ev["tide"])
    lag = lambda hrs: np.concatenate([np.full(int(hrs * freq), t[0]), t[:n - int(hrs * freq)]])
    return pd.DataFrame({"tide": t, "tide_1h": lag(1), "tide_2h": lag(2), "tide_3h": lag(3),
                         "slope_1h": t - lag(1), "tide_max_6h": pd.Series(t).rolling(int(6 * freq), 1).max().values})


def fit_hinge(X, y, cols):
    """Least squares with hinge terms max(tide_k - h, 0); flooding level h chosen on the training data."""
    best = None
    for h in np.arange(-1.0, 1.51, 0.05):
        A = np.column_stack([np.ones(len(X))] + [X[c] for c in cols] +
                            [np.maximum(X[c] - h, 0) for c in ("tide", "tide_1h", "tide_max_6h") if c in cols])
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        sse = np.sum((A @ beta - y) ** 2)
        if best is None or sse < best[0]:
            best = (sse, h, beta)
    _, h, beta = best

    def predict(Z):
        A = np.column_stack([np.ones(len(Z))] + [Z[c] for c in cols] +
                            [np.maximum(Z[c] - h, 0) for c in ("tide", "tide_1h", "tide_max_6h") if c in cols])
        return A @ beta
    return predict, h


def blocked_cv(table, models):
    """Leave-one-day-out over both events (each day of each event is one block)."""
    blocks = table["Event"] + " day " + (table["Hour"] // 24).astype(int).astype(str)
    preds = {m: np.full(len(table), np.nan) for m in models}
    levels = []
    for b in blocks.unique():
        te = (blocks == b).to_numpy()
        tr = ~te
        for m, (kind, cols) in models.items():
            Xtr, Xte = table.loc[tr, cols].fillna(0), table.loc[te, cols].fillna(0)
            ytr = table.loc[tr, "Ditch_depth_m"].to_numpy()
            if kind == "linear":
                A = np.column_stack([np.ones(tr.sum()), Xtr.to_numpy()])
                beta, *_ = np.linalg.lstsq(A, ytr, rcond=None)
                preds[m][te] = np.column_stack([np.ones(te.sum()), Xte.to_numpy()]) @ beta
            elif kind == "hinge":
                hcols = ["tide", "tide_1h", "tide_2h", "tide_3h", "slope_1h", "tide_max_6h"] if "memory" in m else ["tide"]
                extra = [c for c in cols if c not in hcols]
                fit, h = fit_hinge(Xtr, ytr, hcols + extra)
                preds[m][te] = fit(Xte)
                if m.startswith("Hinge (threshold)"):
                    levels.append(h)
            else:
                from sklearn.ensemble import HistGradientBoostingRegressor
                gb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                                   min_samples_leaf=20, random_state=0).fit(Xtr, ytr)
                preds[m][te] = gb.predict(Xte)
    return preds, levels


# === Run ===
base_path = to_local_path(base_path)
output_path = to_local_path(output_path) if output_path else os.path.join(base_path, "tda_results", "tide")
paths = [to_local_path(p) for p in files] if files else find_workbooks(base_path)
if not paths:
    raise FileNotFoundError(f"No .xlsx with a 'Validation' sheet in {os.path.abspath(base_path)}")
os.makedirs(output_path, exist_ok=True)
print("Events:")
events = [load_event(p) for p in paths]
for ev in events:
    prepare(ev)
    print(f"  {ev['name']}: {len(ev['h'])} samples on the {60 / freq:g}-min grid"
          f"{' (hourly information)' if harmonize_hourly else ''}; tide -> ditch lag {ev['lag_h']:+.2f} h "
          f"(r = {ev['lag_r']:.2f})")
    v = ev["bc"][ev["t0"]:ev["t1"]]          # surge: > 0.4 m above every value 2-12 h before / after it
    th = (v.index - ev["t0"]).total_seconds().values / 3600
    far = lambda i: (np.abs(th - th[i]) >= 2) & (np.abs(th - th[i]) <= 12)
    spike = v[[bool(far(i).any()) and v.iloc[i] > v.values[far(i)].max() + 0.4 for i in range(len(v))]]
    if len(spike):
        print(f"    note: tide peak(s) well above the neighbouring high tides (possible surge) - check the "
              f"ditch response: "
              + ", ".join(f"{t:%d %b %H:%M} {v:.2f} m" for t, v in spike.items()))

print("\nFourier (period in hours, strongest first):")
peaks_f = []
for ev in events:
    rec = ev["h"][-1] - ev["h"][0]
    for nm, y in (("tide", ev["tide"]), ("ditch", ev["depth"])):
        f, amp = spectrum(y)
        pk = spectral_peaks(f, amp, rec)
        ev[f"spec_{nm}"] = (f, amp)
        peaks_f += [f[i] for i in pk]
        print(f"  {ev['name']:32s} {nm:5s}: " + ", ".join(f"{1 / f[i]:.1f} h" for i in pk[:5]))
        if nm == "tide":
            ev["dominant_f"] = f[pk[0]]
min_f = min_f or float(np.median([ev["dominant_f"] for ev in events]))
max_f = max_f or float(max(peaks_f))
w = int(round((1 / min_f + 2 * 0.25 / max_f) * freq))
tau = max(1, int(round((0.25 / max_f) * freq / 2)))
print(f"\nmin_f = {min_f:.4f} cycles/h ({1 / min_f:.1f} h), max_f = {max_f:.4f} cycles/h ({1 / max_f:.1f} h) -> "
      f"window {w} samples ({w / freq:.1f} h), tau {tau} samples ({tau / freq:.2f} h), "
      f"{len(range(0, w - tau, s))} points per Takens cloud")

print("\nWindow features (tide / ditch / loop clouds):")
tables, timing = [], {}
for ev in events:
    tb, tm = window_features(ev, w, tau)
    mem = tide_memory(ev).iloc[w - 1::s_W].reset_index(drop=True)
    tables.append(pd.concat([tb, mem], axis=1))
    timing = {f: timing.get(f, 0) + v / len(events) for f, v in tm.items()}
    print(f"  {ev['name']}: {len(tb)} windows")
table = pd.concat(tables, ignore_index=True)
feats = [f for f in FEATURE_FUNS if COMPUTE[f] and table[f"tide: {f}"].notna().any()]

# --- comparison table ---
types = list(dict.fromkeys(table["Tide_type"]))
comp = []
for c in CLOUDS:
    for f in feats:
        col = f"{c}: {f}"
        row = {"Cloud": c, "Feature": f, "Fit failures %": 100 * table[col].isna().mean(),
               "ms per window": timing[f]}
        for ty in types:
            m = table["Tide_type"] == ty
            row[f"rho with depth ({ty})"] = table.loc[m, col].corr(table.loc[m, "Ditch_depth_m"], method="spearman")
        if len(types) == 2:                                          # condition separation, per tidal cycle
            cyc = table.assign(c=(table["Hour"] // (1 / min_f)).astype(int)).groupby(["Event", "c"]).agg(
                v=(col, "max"), ty=("Tide_type", "first"))
            a, b = cyc.loc[cyc.ty == types[0], "v"].dropna(), cyc.loc[cyc.ty == types[1], "v"].dropna()
            pooled = np.sqrt((a.var() + b.var()) / 2)
            row[f"cycle peak {types[0]}"], row[f"cycle peak {types[1]}"] = a.median(), b.median()
            row["separation (Cohen d)"] = (a.median() - b.median()) / pooled if pooled > 0 else np.nan
        comp.append(row)
comp = pd.DataFrame(comp)
print("\nFeature comparison:")
print(comp.round(3).to_string(index=False))

# --- prediction ---
fill = {f: f"tide: {f}" for f in feats}
table["tide: FastTDA+BVG"] = table["tide: FastTDA"].fillna(table["tide: BVG minor"]) if {"FastTDA", "BVG minor"} <= set(feats) else np.nan
memory = ["tide", "tide_1h", "tide_2h", "tide_3h", "slope_1h", "tide_max_6h"]
models = {"Tide level (linear)": ("linear", ["tide"]),
          "Hinge (threshold) model": ("hinge", ["tide"]),
          "Hinge + memory": ("hinge", memory)}
for f in feats:
    models[f"Hinge + memory + {f}"] = ("hinge", memory + [fill[f]])
if table["tide: FastTDA+BVG"].notna().any():
    models["Hinge + memory + FastTDA/BVG hybrid"] = ("hinge", memory + ["tide: FastTDA+BVG"])
try:
    import sklearn  # noqa: F401
    models["Gradient boosting (memory)"] = ("gb", memory)
    for f in feats:                                              # which shape feature carries the gain?
        models[f"Gradient boosting (memory + {f})"] = ("gb", memory + [fill[f]])
    models["Gradient boosting (memory + all tide TDA)"] = ("gb", memory + [fill[f] for f in feats])
except ImportError:
    print("scikit-learn not found: gradient boosting skipped")
preds, levels = blocked_cv(table, models)
y = table["Ditch_depth_m"].to_numpy()
pred_rows = []
for m, p in preds.items():
    row = {"Model": m, "RMSE m": np.sqrt(np.nanmean((p - y) ** 2)),
           "R2": 1 - np.nansum((p - y) ** 2) / np.sum((y - y.mean()) ** 2)}
    for ty in types:
        k = (table["Tide_type"] == ty).to_numpy()
        row[f"RMSE m ({ty})"] = np.sqrt(np.nanmean((p[k] - y[k]) ** 2))
    pred_rows.append(row)
pred = pd.DataFrame(pred_rows)
base_rmse = pred.loc[pred.Model == "Hinge + memory", "RMSE m"].iloc[0]
pred["vs hinge + memory %"] = 100 * (1 - pred["RMSE m"] / base_rmse)
if "Gradient boosting (memory)" in preds:                          # gain of each feature within the model type
    gb_rmse = pred.loc[pred.Model == "Gradient boosting (memory)", "RMSE m"].iloc[0]
    pred["vs same model without shape %"] = np.where(pred.Model.str.startswith("Gradient"),
                                                     100 * (1 - pred["RMSE m"] / gb_rmse), pred["vs hinge + memory %"])
print(f"\nDitch depth from the tide only (leave-one-day-out; flooding level learned: "
      f"{np.median(levels):.2f} m NAVD88 median):")
print(pred.round(4).to_string(index=False))

with pd.ExcelWriter(os.path.join(output_path, "tide_tda_windows.xlsx"), engine="openpyxl") as wr:
    pd.DataFrame([{"event": ev["name"], "lag_h": ev["lag_h"], "lag_r": ev["lag_r"], "samples": len(ev["h"])} for ev in events]
                 + [{"event": "settings", "lag_h": np.nan, "lag_r": np.nan, "samples": np.nan, "min_f": min_f,
                     "max_f": max_f, "window_h": w / freq, "tau_h": tau / freq, "freq": freq,
                     "harmonize_hourly": harmonize_hourly}]).to_excel(wr, sheet_name="Events", index=False)
    comp.to_excel(wr, sheet_name="Feature_comparison", index=False)
    pred.to_excel(wr, sheet_name="Prediction", index=False)
    table.assign(**{f"pred: {m}": p for m, p in preds.items()}).to_excel(wr, sheet_name="Windows", index=False)
print(f"Saved: {os.path.join(output_path, 'tide_tda_windows.xlsx')}")


def save(fig, name):
    p = os.path.join(output_path, name)
    fig.savefig(p, dpi=150)
    print(f"Saved: {p}")
    plt.show()
    plt.close(fig)


# === F1: overview - signals, spectra, ditch depth vs tide ===
fig, axes = plt.subplots(len(events), 3, figsize=(18, 4.2 * len(events)), squeeze=False,
                         gridspec_kw={"width_ratios": [2.2, 1, 1]})
for r, ev in enumerate(events):
    ax = axes[r, 0]
    ax.plot(ev["h"], ev["tide"], color="#2a78d6", lw=1.2, label="tide WSE (m NAVD88)")
    ax.plot(ev["h"], ev["depth"], color="#eb6834", lw=1.2, label="ditch depth (m)")
    ax.set_title(f"{ev['name']}: lag {ev['lag_h']:+.2f} h", fontsize=10, fontweight="bold")
    ax.set_xlabel("Hours from start")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[r, 1]
    for nm, c in (("tide", "#2a78d6"), ("ditch", "#eb6834")):
        f, amp = ev[f"spec_{nm}"]
        ax.plot(f, amp / amp.max(), color=c, lw=1.2, label=nm)
    for fk in (min_f / 2, min_f, 2 * min_f):                  # diurnal, semidiurnal, quarter-diurnal
        ax.axvline(fk, color="#b8b7b2", lw=0.6, ls=":")
    ax.set_xlim(0, min(pass_value, 0.35))
    ax.set_xlabel("cycles per hour (dotted: diurnal, semidiurnal, quarter-diurnal)")
    ax.set_title("Fourier spectrum", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax = axes[r, 2]
    rising = np.gradient(ev["tide"]) > 0
    ax.scatter(ev["tide"][rising], ev["depth"][rising], s=5, color="#2a78d6", label="tide rising")
    ax.scatter(ev["tide"][~rising], ev["depth"][~rising], s=5, color="#eb6834", label="tide falling")
    ax.axvline(np.median(levels), color="#3d3d3a", lw=0.8, ls="--")
    ax.set_xlabel("tide WSE (m NAVD88)")
    ax.set_ylabel("ditch depth (m)")
    ax.set_title("Ditch vs tide (dashed: learned flooding level)", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
for ax in axes.flat:
    ax.grid(color="#e4e3df", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
save(fig, "F1_overview.png")

# === F2: sliding-window figure per event (tide cloud, FastTDA and Gaussian ellipses, features) ===
for ev in events:
    tb = table[table["Event"] == ev["name"]].reset_index(drop=True)
    i = int(np.nanargmax(tb["tide: VR H1"])) if "VR H1" in feats else len(tb) // 2
    e = int(round(tb["Hour"].iloc[i] * freq - ev["h"][0] * freq))
    cl = clouds_of(ev, e, w, tau)
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(8, 12), gridspec_kw={"height_ratios": [1, 1.3, 1]})
    a1.plot(ev["h"], ev["tide"], color="#2a78d6", lw=1.1, label="tide WSE")
    a1.plot(ev["h"], ev["depth"], color="#eb6834", lw=1.1, label="ditch depth")
    a1.axvspan(ev["h"][e - w + 1], ev["h"][e], color="#f2c744", alpha=0.45, lw=0)
    a1.set_title(f"{ev['name']} - window {ev['h'][e - w + 1]:.1f}-{ev['h'][e]:.1f} h (yellow)", fontsize=10)
    a1.set_xlabel("Hours from start")
    a1.legend(frameon=False, fontsize=8)
    pc = cl["tide"]
    a2.plot(pc[:, 0], pc[:, 1], ".", color="#2a78d6", ms=4)
    two_b, ell = fasttda_fit(pc)
    if ell is not None:
        a2.plot(*ell, color="#e34948", lw=2, label=f"FastTDA ellipse (2b = {two_b:.2f})")
    bmin, bratio, lam, vec = bvg(pc)
    ang = np.linspace(0, 2 * np.pi, 200)
    circ = np.stack([2 * np.sqrt(lam[1]) * np.cos(ang), 2 * np.sqrt(lam[0]) * np.sin(ang)])
    gx = pc.mean(axis=0)[:, None] + vec[:, [1, 0]] @ circ
    a2.plot(gx[0], gx[1], color="#0d366b", lw=1.8, ls="--", label=f"Gaussian 2-sigma ellipse (minor {bmin:.2f})")
    a2.set_xlim(-1.4, 1.4)
    a2.set_ylim(-1.4, 1.4)
    a2.set_aspect("equal")
    a2.set_xlabel("tide x(t) (window scaled)")
    a2.set_ylabel(f"x(t + {tau / freq:g} h)")
    a2.set_title(f"Takens embedding of the tide: VR H1 = {tb['tide: VR H1'].iloc[i]:.2f}" if "VR H1" in feats else "",
                 fontsize=10)
    a2.legend(frameon=False, fontsize=8, loc="lower right")
    a2.grid(color="#e4e3df", lw=0.6)
    for f in [f for f in feats if f != "Cubic oval"]:              # cubic oval: 0 / 1 switching, tables only
        v = tb[f"tide: {f}"]
        a3.plot(tb["Hour"], v / np.nanmax(np.abs(v)), color=COLORS[f], lw=1.1, label=f)
    a3b = a3.twinx()
    a3b.plot(ev["h"], ev["depth"], color="#b8b7b2", lw=0.9, ls="--")
    a3b.set_ylabel("ditch depth (m), dashed", color="#8a8984")
    a3.axvline(tb["Hour"].iloc[i], color="#3d3d3a", lw=0.8)
    a3.set_ylabel("feature / its maximum")
    a3.set_xlabel("Hours from start (end of window)")
    a3.set_title("Tide shape features over time", fontsize=10)
    a3.legend(frameon=False, fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    fig.suptitle(f"Sliding-window TDA of the tide (window {w / freq:.1f} h, tau {tau / freq:.2f} h)", fontweight="bold")
    fig.tight_layout()
    save(fig, f"F2_window_{re.sub(r'[^A-Za-z0-9]+', '_', ev['type'])}.png")

# === F3: feature comparison ===
metrics = [c for c in comp.columns if c.startswith("rho with depth")] + (["separation (Cohen d)"] if len(types) == 2 else [])
fig, axes = plt.subplots(1, len(metrics) + 1, figsize=(4.6 * (len(metrics) + 1), 4.2), squeeze=False)
labels = [f"{r.Cloud}: {r.Feature}" for r in comp.itertuples()]
yy = np.arange(len(comp))[::-1]
for ax, mcol in zip(axes[0], ["Fit failures %"] + metrics):
    ax.barh(yy, comp[mcol], color=[COLORS[f] for f in comp["Feature"]])
    ax.set_title(mcol, fontsize=9, fontweight="bold")
    ax.axvline(0, color="#3d3d3a", lw=0.6)
    ax.set_yticks(yy, labels if ax is axes[0, 0] else [""] * len(labels), fontsize=7)
    ax.grid(axis="x", color="#e4e3df", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("Feature comparison (tide / ditch / loop clouds)", fontweight="bold")
fig.tight_layout()
save(fig, "F3_feature_comparison.png")

# === F4: prediction ===
fig, ax = plt.subplots(figsize=(9, 0.42 * len(pred) + 1.5))
yy = np.arange(len(pred))[::-1]
ax.barh(yy, pred["RMSE m"], color=["#8a8984" if ("TDA" not in m and not any(f in m for f in feats)) else "#eb6834"
                                   for m in pred["Model"]])
for yi, r, g in zip(yy, pred["RMSE m"], pred["vs hinge + memory %"]):
    ax.text(r * 1.01, yi, f"{r:.3f} m ({g:+.0f}%)", va="center", fontsize=8)
ax.set_yticks(yy, pred["Model"], fontsize=8)
ax.set_xlabel("RMSE of the ditch depth (m), leave-one-day-out; (%) = change vs hinge + memory")
ax.set_xlim(0, 1.3 * pred["RMSE m"].max())
ax.grid(axis="x", color="#e4e3df", lw=0.6)
ax.spines[["top", "right"]].set_visible(False)
ax.set_title("Does a tide shape feature improve the ditch-depth prediction? (orange = uses a shape feature)",
             fontsize=10, fontweight="bold")
fig.tight_layout()
save(fig, "F4_prediction.png")
