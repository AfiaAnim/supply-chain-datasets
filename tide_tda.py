"""
Tide TDA: topological data analysis of tidal water-level records (Hanahan, SC).

Paste into ONE Jupyter cell and run (or: python tide_tda.py [file1.xlsx file2.xlsx ...]).

Input workbooks (every sheet is read; column A = date/time, then one or more numeric columns,
the right-most column holds "Key: Value" metadata such as "Tide Type: King Tide"):
    10_Nov_to_25Nov_2025.xlsx  -> Regular tide: BC Data (WSE NAVD88, river boundary) +
                                  Validation Data (water depth in the ditch)
    Dec_1_to_Dec5_2025.xlsx    -> King tide:    Boundary Condition Data (WSE) +
                                  Validation Data (depth + WSE in the ditch)

What it does for every series:
    1. Resample the irregular record (15 / 30 / 60 / 120 min) onto a uniform grid, no filling
       across long gaps.
    2. Dominant period from the periodogram (expected: M2 semidiurnal tide, about 12.42 h).
    3. Takens delay embedding x(t), x(t+tau), x(t+2tau) with tau = period / 4, so one tidal
       cycle traces one loop in the embedding.
    4. Vietoris-Rips persistent homology (ripser) of the embedded point cloud:
         H0 = clusters, H1 = loops.  One long H1 bar = a clean periodic tide; its length is
         how strong / regular the cycle is. Extra mid-length H1 bars = cycle-to-cycle
         variation (diurnal inequality, spring-neap, surge or king-tide distortion).
    5. Sub/superlevel-set persistence of the raw signal (0-dim, exact): every high tide and
         low tide becomes a (birth, death) pair whose persistence is its prominence. Significant
         highs / lows give the tidal range per cycle.
    6. Sliding-window persistence: the max H1 persistence in a moving window (one tidal day)
         shows WHEN the periodic pattern is strong or broken.
Across series:
    7. Bottleneck and Wasserstein distances between the H1 diagrams (z-scored shape) and
       between the high-tide prominence diagrams (absolute metres) -> which records have
       the same pattern.
    8. Lag between the river boundary and the ditch (cross-correlation) for each event.

Output (OUTPUT_DIR): <series>_TDA.png per series, TDA_overview.png, TDA_sliding_window.png,
                     Tide_TDA_Summary.xlsx

Requirements:
    pip install numpy pandas scipy matplotlib openpyxl ripser persim
"""

import glob
import os
import re
import sys

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

from persim import bottleneck, wasserstein
from ripser import ripser


# ============================================================
# SETTINGS
# ============================================================

DATA_DIR = "."                          # folder with the .xlsx files (Jupyter: the notebook folder)
FILES = None                            # None = every *.xlsx in DATA_DIR; or a list of paths
OUTPUT_DIR = None                       # None = DATA_DIR/tide_tda_results

RESAMPLE_MIN = 30                       # uniform grid step (minutes); 30 = coarsest common step
MAX_GAP_H = 3.0                         # do not interpolate across gaps longer than this (hours)
PERIOD_RANGE_H = (6.0, 30.0)            # search band for the dominant tidal period
EMB_DIM = 3                             # Takens embedding dimension
TAU_H = None                            # None = dominant period / 4; or a fixed lag in hours
MAX_POINTS = 600                        # farthest-point subsample size for Rips (speed)
WINDOW_H = 24.84                        # sliding window = one tidal day (2 x M2)
STEP_H = 2.0                            # sliding window step
PROMINENCE_FRAC = 0.25                  # a high / low tide counts if its prominence >= this x
                                        # the series' full range
SHOW_PLOTS = True                       # plt.show() at the end (Jupyter); files are always saved

COLORS = {"H0": "#2a78d6", "H1": "#eb6834", "high": "#eb6834", "low": "#2a78d6",
          "line": "#52514e", "muted": "#a3a29c", "grid": "#e6e5e0"}
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e34948", "#008300"]

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": COLORS["muted"], "axes.grid": True, "grid.color": COLORS["grid"],
    "grid.linewidth": 0.6, "axes.titleweight": "bold", "axes.titlesize": 10,
    "lines.linewidth": 1.4,
})


# ============================================================
# LOADING
# ============================================================

def _metadata(df):
    """Collect 'Key: Value' strings anywhere in the text columns of a sheet."""
    meta = {}
    for col in df.columns:
        for v in df[col].dropna():
            if isinstance(v, str) and ":" in v:
                k, val = v.split(":", 1)
                meta[k.strip().lower()] = val.strip()
            elif isinstance(v, str) and "tide" in v.lower():
                meta.setdefault("tide type", v.strip())
        if isinstance(col, str) and "tide" in col.lower():
            meta.setdefault("tide type", col.strip())
    return meta


def _short(text):
    text = re.sub(r"\(.*?\)", "", str(text))
    return re.sub(r"\s+", " ", text).strip()


def load_series(paths):
    """Return a list of dicts: name, tide, site, variable, unit, time (DatetimeIndex), values."""
    out = []
    for path in paths:
        for sheet, df in pd.read_excel(path, sheet_name=None).items():
            if df.shape[1] < 2:
                continue
            t = pd.to_datetime(df.iloc[:, 0], errors="coerce")
            meta = _metadata(df)
            tide = meta.get("tide type", "Unknown tide")
            tide = "King tide" if "king" in tide.lower() else (
                "Regular tide" if "regular" in tide.lower() else tide)
            usage = (meta.get("usage", "") + " " + sheet).lower()
            site = "Boundary (river)" if ("boundary" in usage or "bc" in sheet.lower()) else (
                "Ditch (validation)" if "validation" in usage else sheet)
            kept = []
            for col in df.columns[1:]:
                y = pd.to_numeric(df[col], errors="coerce")
                ok = t.notna() & y.notna()
                if ok.sum() < 20 or str(col).startswith("Unnamed"):
                    continue
                s = pd.Series(y[ok].values, index=t[ok].values).sort_index()
                s = s[~s.index.duplicated()]
                # skip a column that is just an offset copy of one already kept (e.g. WSE = depth + c)
                dup = next((k for k in kept if len(k) == len(s)
                            and np.allclose(np.diff(k.values), np.diff(s.values), atol=1e-3)), None)
                if dup is not None:
                    print(f"  [{os.path.basename(path)} / {sheet}] '{col}' is an offset copy of "
                          f"'{dup.name}' -> skipped (same topology)")
                    continue
                s.name = str(col)
                kept.append(s)
                var = "WSE" if "wse" in str(col).lower() else (
                    "Depth" if "depth" in str(col).lower() else _short(col))
                out.append(dict(name=f"{tide} | {site} | {var}", tide=tide, site=site,
                                variable=var, column=_short(col), file=os.path.basename(path),
                                sheet=sheet, raw=s, meta=meta))
    return out


def resample(s, step_min=RESAMPLE_MIN, max_gap_h=MAX_GAP_H):
    """Uniform grid by time interpolation; keep the longest stretch without long gaps."""
    grid = pd.date_range(s.index[0].ceil(f"{step_min}min"), s.index[-1], freq=f"{step_min}min")
    u = s.reindex(s.index.union(grid)).interpolate("time").reindex(grid)
    # mask grid points that fall inside a gap longer than max_gap_h
    gaps = np.diff(s.index.values).astype("timedelta64[m]").astype(float) / 60
    for g_start, g_len in zip(s.index[:-1][gaps > max_gap_h], gaps[gaps > max_gap_h]):
        u[(u.index > g_start) & (u.index < g_start + pd.Timedelta(hours=g_len))] = np.nan
    good = u.notna().values
    runs = np.split(np.arange(len(u)), np.where(np.diff(good.astype(int)) != 0)[0] + 1)
    best = max((r for r in runs if good[r[0]]), key=len)
    return u.iloc[best]


# ============================================================
# TDA BUILDING BLOCKS
# ============================================================

def dominant_period(y, dt_h, band=PERIOD_RANGE_H):
    f, p = signal.periodogram(signal.detrend(y), fs=1.0 / dt_h, window="hann")
    m = (f > 1 / band[1]) & (f < 1 / band[0])
    i = np.argmax(p[m])
    return 1.0 / f[m][i], f, p


def takens(y, dim, tau):
    n = len(y) - (dim - 1) * tau
    return np.column_stack([y[i * tau: i * tau + n] for i in range(dim)])


def farthest_point_sample(X, k, seed=0):
    if len(X) <= k:
        return np.arange(len(X))
    idx = [np.random.default_rng(seed).integers(len(X))]
    d = np.linalg.norm(X - X[idx[0]], axis=1)
    for _ in range(k - 1):
        idx.append(int(np.argmax(d)))
        d = np.minimum(d, np.linalg.norm(X - X[idx[-1]], axis=1))
    return np.sort(np.array(idx))


def rips_diagrams(X, maxdim=1):
    dgms = ripser(X, maxdim=maxdim)["dgms"]
    return [d[np.isfinite(d[:, 1])] if i > 0 else d for i, d in enumerate(dgms)]


def levelset_persistence(y):
    """Exact 0-dim sublevel-set persistence of a 1-D signal (union-find on the path graph).
    Returns array of (birth, death, index_of_minimum). Each local minimum is born at its value
    and dies when it merges with a deeper basin; the global minimum dies at max(y)."""
    order = np.argsort(y, kind="stable")
    parent = -np.ones(len(y), dtype=int)
    root_min = {}
    pairs = []

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in order:
        parent[i] = i
        root_min[i] = i
        for j in (i - 1, i + 1):
            if not (0 <= j < len(y)) or parent[j] < 0:
                continue
            ri, rj = find(i), find(j)
            if ri == rj:
                continue
            # elder rule: the basin with the higher minimum (younger) dies at y[i]
            young, old = (ri, rj) if y[root_min[ri]] > y[root_min[rj]] else (rj, ri)
            if root_min[young] != i:                    # skip zero-length pair of i itself
                pairs.append((y[root_min[young]], y[i], root_min[young]))
            parent[young] = old
    gmin = int(np.argmin(y))
    pairs.append((y[gmin], y.max(), gmin))
    p = np.array(pairs)
    return p[p[:, 1] - p[:, 0] > 0]


def tide_extremes(y, frac=PROMINENCE_FRAC):
    """High and low tides as persistence pairs with prominence >= frac * range."""
    rng = np.ptp(y)
    lows = levelset_persistence(y)
    highs = levelset_persistence(-y)
    highs = np.column_stack([-highs[:, 0], -highs[:, 1], highs[:, 2]])   # birth = peak height
    lows = lows[(lows[:, 1] - lows[:, 0]) >= frac * rng]
    highs = highs[(highs[:, 0] - highs[:, 1]) >= frac * rng]
    return highs[np.argsort(highs[:, 2])], lows[np.argsort(lows[:, 2])]


def h1_stats(h1):
    if len(h1) == 0:
        return dict(h1_count=0, h1_max=0.0, h1_second=0.0, h1_total=0.0, h1_ratio=np.nan)
    pers = np.sort(h1[:, 1] - h1[:, 0])[::-1]
    second = pers[1] if len(pers) > 1 else 0.0
    return dict(h1_count=int(len(pers)), h1_max=float(pers[0]), h1_second=float(second),
                h1_total=float(pers.sum()), h1_ratio=float(pers[0] / second) if second else np.inf)


def sliding_h1(z, dt_h, tau, window_h=WINDOW_H, step_h=STEP_H, dim=EMB_DIM):
    w, st = int(round(window_h / dt_h)), max(1, int(round(step_h / dt_h)))
    centers, h1max = [], []
    for s in range(0, len(z) - w + 1, st):
        seg = z[s:s + w]
        X = takens(seg, dim, tau)
        if len(X) < 8:
            continue
        h1 = rips_diagrams(X)[1]
        centers.append(s + w // 2)
        h1max.append(float(np.max(h1[:, 1] - h1[:, 0])) if len(h1) else 0.0)
    return np.array(centers, dtype=int), np.array(h1max)


# ============================================================
# PLOTTING
# ============================================================

def plot_diagram(ax, dgms, title):
    allv = np.concatenate([d[np.isfinite(d).all(1)] for d in dgms if len(d)])
    hi = float(allv.max()) * 1.08 if len(allv) else 1.0
    ax.plot([0, hi], [0, hi], color=COLORS["muted"], lw=1)
    for k, d in enumerate(dgms):
        d = d.copy()
        d[~np.isfinite(d[:, 1]), 1] = hi * 0.98
        ax.scatter(d[:, 0], d[:, 1], s=16 if k == 0 else 26, color=COLORS[f"H{k}"],
                   edgecolor="white", linewidth=0.6, label=f"H{k} ({len(d)})", zorder=3)
    ax.set_xlim(-0.02 * hi, hi)
    ax.set_ylim(-0.02 * hi, hi)
    ax.set_xlabel("birth")
    ax.set_ylabel("death")
    ax.set_title(title)
    ax.legend(loc="lower right", frameon=False)


def plot_series(r, outdir):
    t, y, z = r["time"], r["y"], r["z"]
    fig = plt.figure(figsize=(15, 8.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.15], hspace=0.38, wspace=0.28)
    fig.suptitle(f"{r['name']}   ({t[0]:%d %b %H:%M} to {t[-1]:%d %b %Y %H:%M}, "
                 f"{RESAMPLE_MIN}-min grid)", fontsize=12, fontweight="bold", x=0.01, ha="left")

    ax = fig.add_subplot(gs[0, :2])
    ax.plot(t, y, color=COLORS["line"], lw=1.1)
    hi, lo = r["highs"], r["lows"]
    ax.scatter(t[hi[:, 2].astype(int)], hi[:, 0], s=30, color=COLORS["high"], zorder=3,
               edgecolor="white", linewidth=0.6, label=f"high tides ({len(hi)})")
    ax.scatter(t[lo[:, 2].astype(int)], lo[:, 0], s=30, color=COLORS["low"], zorder=3,
               edgecolor="white", linewidth=0.6, label=f"low tides ({len(lo)})")
    ax.set_ylabel(f"{r['column']} (m)")
    ax.set_title("Signal with topologically significant highs / lows "
                 f"(prominence >= {PROMINENCE_FRAC:.0%} of range)")
    ax.legend(loc="upper left", frameon=False, ncol=2)

    ax = fig.add_subplot(gs[0, 2])
    ax.semilogy(1 / r["freq"][1:], r["power"][1:], color=COLORS["line"], lw=1.1)
    ax.axvline(r["period_h"], color=COLORS["H1"], lw=1.4, ls="--")
    ax.text(r["period_h"], ax.get_ylim()[1], f" {r['period_h']:.2f} h", color="#0b0b0b",
            va="top", ha="left")
    ax.set_xlim(2, 40)
    ax.set_xlabel("period (h)")
    ax.set_ylabel("power")
    ax.set_title("Periodogram (dominant tidal period)")

    ax = fig.add_subplot(gs[1, 0], projection="3d")
    X = r["X"]
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=np.arange(len(X)), cmap="Blues", s=5)
    ax.plot(X[:, 0], X[:, 1], X[:, 2], color=COLORS["muted"], lw=0.4, alpha=0.6)
    ax.set_title(f"Takens embedding (dim {EMB_DIM}, tau = {r['tau_h']:.2f} h)\n"
                 "colour = time, one loop = one tide")
    ax.set_xlabel("z(t)")
    ax.set_ylabel("z(t+tau)")
    ax.set_zlabel("z(t+2tau)")

    plot_diagram(fig.add_subplot(gs[1, 1]), r["dgms"],
                 f"Rips persistence diagram\nmax H1 = {r['h1_max']:.2f}, "
                 f"2nd H1 = {r['h1_second']:.2f}")

    ax = fig.add_subplot(gs[1, 2])
    h1 = r["dgms"][1]
    if len(h1):
        h1 = h1[np.argsort(h1[:, 1] - h1[:, 0])[::-1]][:25]
        for k, (b, d) in enumerate(h1):
            ax.hlines(k, b, d, color=COLORS["H1"], lw=3.5, capstyle="round")
        ax.invert_yaxis()
    ax.set_yticks([])
    ax.set_xlabel("filtration radius")
    ax.set_title("H1 barcode (top 25 loops)\none long bar = one clean cycle")

    path = os.path.join(outdir, re.sub(r"[^A-Za-z0-9]+", "_", r["name"]).strip("_") + "_TDA.png")
    fig.savefig(path, bbox_inches="tight")
    return fig, path


def plot_overview(results, dist_shape, dist_amp, outdir):
    names = [r["name"] for r in results]
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.8),
                             gridspec_kw=dict(width_ratios=[1.1, 1.1, 1, 1], wspace=0.45))

    ax = axes[0]
    hi = max(float(r["dgms"][1].max()) if len(r["dgms"][1]) else 0 for r in results) * 1.08
    ax.plot([0, hi], [0, hi], color=COLORS["muted"], lw=1)
    for r, c in zip(results, SERIES_COLORS):
        d = r["dgms"][1]
        ax.scatter(d[:, 0], d[:, 1], s=26, color=c, edgecolor="white", lw=0.6, label=r["name"])
    ax.set_xlabel("birth")
    ax.set_ylabel("death")
    ax.set_title("H1 diagrams, all series (z-scored shape)")
    ax.legend(fontsize=7, frameon=False, loc="lower right")

    ax = axes[1]
    for r, c in zip(results, SERIES_COLORS):
        p = r["highs"][:, 0] - r["highs"][:, 1]
        ax.scatter(r["highs"][:, 0], p, s=26, color=c, edgecolor="white", lw=0.6, label=r["name"])
    ax.set_xlabel("high-tide level (m)")
    ax.set_ylabel("prominence = tidal range (m)")
    ax.set_title("Superlevel persistence: high tides")

    short = [n.replace(" | ", "\n") for n in names]
    for ax, D, ttl in [(axes[2], dist_shape, "Bottleneck distance, H1 (shape)"),
                       (axes[3], dist_amp, "Wasserstein distance, high tides (m)")]:
        im = ax.imshow(D, cmap="Blues")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(short, fontsize=6.5, rotation=90)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(short, fontsize=6.5)
        ax.grid(False)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{D[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if D[i, j] > 0.6 * D.max() else "#0b0b0b")
        ax.set_title(ttl)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    path = os.path.join(outdir, "TDA_overview.png")
    fig.savefig(path, bbox_inches="tight")
    return fig, path


def plot_sliding(results, outdir):
    fig, axes = plt.subplots(len(results), 1, figsize=(14, 2.3 * len(results)), squeeze=False)
    for ax, r in zip(axes[:, 0], results):
        ax2 = ax.twinx()                                    # signal is context only (grey, no axis)
        ax2.plot(r["time"], r["y"], color=COLORS["muted"], lw=0.8)
        ax2.set_yticks([])
        ax2.grid(False)
        ax2.spines[:].set_visible(False)
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)
        ax.plot(r["time"][r["sw_idx"]], r["sw_h1"], color=COLORS["H1"], lw=1.8)
        ax.set_ylabel("max H1")
        ax.set_title(f"{r['name']}: sliding-window loop strength "
                     f"({WINDOW_H:g} h window; grey = signal, for timing only)", loc="left")
    fig.tight_layout()
    path = os.path.join(outdir, "TDA_sliding_window.png")
    fig.savefig(path, bbox_inches="tight")
    return fig, path


# ============================================================
# MAIN
# ============================================================

def analyse(entry):
    s = resample(entry["raw"])
    dt_h = RESAMPLE_MIN / 60
    y = s.values.astype(float)
    z = (y - y.mean()) / y.std()
    period_h, f, p = dominant_period(y, dt_h)
    tau = max(1, int(round((TAU_H if TAU_H else period_h / 4) / dt_h)))
    X = takens(z, EMB_DIM, tau)
    dgms = rips_diagrams(X[farthest_point_sample(X, MAX_POINTS)])
    highs, lows = tide_extremes(y)
    sw_idx, sw_h1 = sliding_h1(z, dt_h, tau)
    r = dict(entry, time=s.index, y=y, z=z, period_h=period_h, freq=f, power=p,
             tau_h=tau * dt_h, X=X, dgms=dgms, highs=highs, lows=lows,
             sw_idx=sw_idx, sw_h1=sw_h1, **h1_stats(dgms[1]))
    ranges = highs[:, 0] - highs[:, 1]
    gaps_h = np.diff(highs[:, 2]) * dt_h if len(highs) > 1 else np.array([np.nan])
    r.update(n_highs=len(highs), n_lows=len(lows),
             mean_high_m=float(highs[:, 0].mean()), max_high_m=float(highs[:, 0].max()),
             mean_low_m=float(lows[:, 0].mean()), min_low_m=float(lows[:, 0].min()),
             mean_range_m=float(ranges.mean()), max_range_m=float(ranges.max()),
             high_to_high_h=float(np.nanmedian(gaps_h)),
             sw_h1_mean=float(sw_h1.mean()) if len(sw_h1) else np.nan,
             sw_h1_min=float(sw_h1.min()) if len(sw_h1) else np.nan,
             weakest_window=(s.index[sw_idx[np.argmin(sw_h1)]] if len(sw_h1) else pd.NaT))
    return r


def lag_hours(a, b):
    """Lag (h) of b behind a, both on the same grid, from the peak of the cross-correlation."""
    common = a["time"].intersection(b["time"])
    if len(common) < 24:
        return np.nan, np.nan
    x = pd.Series(a["y"], a["time"]).loc[common].values
    y = pd.Series(b["y"], b["time"]).loc[common].values
    x, y = (x - x.mean()) / x.std(), (y - y.mean()) / y.std()
    max_lag = int(6 * 60 / RESAMPLE_MIN)
    lags = np.arange(-max_lag, max_lag + 1)
    cc = [np.corrcoef(x[max(0, -k):len(x) - max(0, k)], y[max(0, k):len(y) - max(0, -k)])[0, 1]
          for k in lags]
    k = int(np.argmax(cc))
    return lags[k] * RESAMPLE_MIN / 60, cc[k]


def main(files=None):
    files = files or FILES or sorted(glob.glob(os.path.join(DATA_DIR, "*.xlsx")))
    files = [f for f in files if not os.path.basename(f).startswith(("~$", "Tide_TDA"))]
    if not files:
        raise SystemExit(f"No .xlsx files found in {os.path.abspath(DATA_DIR)}")
    outdir = OUTPUT_DIR or os.path.join(DATA_DIR, "tide_tda_results")
    os.makedirs(outdir, exist_ok=True)

    print("Loading:", *[os.path.basename(f) for f in files], sep="\n  ")
    entries = load_series(files)
    tide_rank = {"Regular tide": 0, "King tide": 1}
    entries.sort(key=lambda e: (tide_rank.get(e["tide"], 2), e["site"] != "Boundary (river)"))

    results = []
    for e in entries:
        r = analyse(e)
        results.append(r)
        print(f"\n{r['name']}  [{r['file']} / {r['sheet']}]  n = {len(r['y'])}")
        print(f"  dominant period {r['period_h']:.2f} h, tau {r['tau_h']:.2f} h")
        print(f"  H1: {r['h1_count']} loops, max persistence {r['h1_max']:.3f}, "
              f"2nd {r['h1_second']:.3f} (ratio {r['h1_ratio']:.1f})")
        print(f"  {r['n_highs']} highs (mean {r['mean_high_m']:.2f} m, max {r['max_high_m']:.2f} m), "
              f"{r['n_lows']} lows, mean range {r['mean_range_m']:.2f} m, "
              f"high-to-high {r['high_to_high_h']:.1f} h")
        plot_series(r, outdir)

    n = len(results)
    dist_shape, dist_amp = np.zeros((n, n)), np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            a, b = results[i], results[j]
            dist_shape[i, j] = dist_shape[j, i] = bottleneck(a["dgms"][1], b["dgms"][1])
            ha, hb = a["highs"][:, [1, 0]], b["highs"][:, [1, 0]]   # (merge level, peak): death > birth
            dist_amp[i, j] = dist_amp[j, i] = wasserstein(ha, hb)
    plot_overview(results, dist_shape, dist_amp, outdir)
    plot_sliding(results, outdir)

    lags = []
    for tide in dict.fromkeys(r["tide"] for r in results):
        bcs = [r for r in results if r["tide"] == tide and r["site"].startswith("Boundary")]
        dit = [r for r in results if r["tide"] == tide and r["site"].startswith("Ditch")]
        for a in bcs:
            for b in dit:
                lag, cc = lag_hours(a, b)
                lags.append(dict(tide=tide, boundary=a["name"], ditch=b["name"],
                                 ditch_lag_h=lag, max_corr=cc))
                print(f"\n{tide}: ditch lags the river boundary by {lag:+.1f} h (r = {cc:.2f})")

    cols = ["name", "tide", "site", "variable", "file", "sheet", "period_h", "tau_h",
            "h1_count", "h1_max", "h1_second", "h1_ratio", "h1_total", "n_highs", "n_lows",
            "mean_high_m", "max_high_m", "mean_low_m", "min_low_m", "mean_range_m",
            "max_range_m", "high_to_high_h", "sw_h1_mean", "sw_h1_min", "weakest_window"]
    summary = pd.DataFrame([{c: r[c] for c in cols} for r in results])
    names = [r["name"] for r in results]
    xlsx = os.path.join(outdir, "Tide_TDA_Summary.xlsx")
    with pd.ExcelWriter(xlsx) as xw:
        summary.to_excel(xw, sheet_name="Summary", index=False)
        pd.DataFrame(dist_shape, names, names).to_excel(xw, sheet_name="Bottleneck_H1_shape")
        pd.DataFrame(dist_amp, names, names).to_excel(xw, sheet_name="Wasserstein_highs_m")
        pd.DataFrame(lags).to_excel(xw, sheet_name="Boundary_to_ditch_lag", index=False)
        for k, r in enumerate(results):
            ext = pd.concat([
                pd.DataFrame(dict(type="high", time=r["time"][r["highs"][:, 2].astype(int)],
                                  level_m=r["highs"][:, 0],
                                  prominence_m=r["highs"][:, 0] - r["highs"][:, 1])),
                pd.DataFrame(dict(type="low", time=r["time"][r["lows"][:, 2].astype(int)],
                                  level_m=r["lows"][:, 0],
                                  prominence_m=r["lows"][:, 1] - r["lows"][:, 0]))
            ]).sort_values("time")
            ext.to_excel(xw, sheet_name=f"Extremes_{k + 1}", index=False)
            pd.DataFrame(r["dgms"][1], columns=["birth", "death"]).assign(
                persistence=lambda d: d.death - d.birth).sort_values(
                "persistence", ascending=False).to_excel(xw, sheet_name=f"H1_{k + 1}", index=False)
    print(f"\nSaved figures and {os.path.basename(xlsx)} in {os.path.abspath(outdir)}")
    print("Extremes_k / H1_k sheets follow the row order of the Summary sheet.")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close("all")
    return results, summary


if __name__ == "__main__":
    if any(a.endswith(".xlsx") for a in sys.argv[1:]):
        SHOW_PLOTS = False
        main([a for a in sys.argv[1:] if a.endswith(".xlsx")])
    else:
        main()
