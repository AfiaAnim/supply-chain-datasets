"""
TDA on Rockingham FY24 daily trip counts: how regular is the weekly rhythm, and when
does it break (holidays, service disruptions)?

Method (same pipeline as the DROPBEAR / SEC scripts in this repo):
    1. Daily trip counts per calendar day, Jul 2023 - Jun 2024 (no-service days = 0),
       for all trips and for the main trip purposes (not dialysis: see PURPOSES below).
    2. Sliding windows of WINDOW_DAYS days, step 1 day, each min-max scaled to [-1, 1].
    3. Takens embedding of each window with dimension TAKENS_DIM and delay TAKENS_DELAY.
       With dim 7, delay 1, each point is one week of consecutive days, so a steady
       weekly pattern traces a closed loop.
    4. Vietoris-Rips persistent homology (ripser): max H1 persistence = how strong that
       loop is. High = regular weekly cycle; low = the cycle broke inside the window.
    5. Noise reference: the same feature on N_SHUFFLES day-shuffled copies of each window
       (weekly order destroyed). The 95th percentile is the "no rhythm" level.
    6. Unusual-window flag: robust z-score of max H1 (vs that series' own median) beyond
       +/- Z_FLAG. LOW = weekly loop broke down (e.g. Christmas week). HIGH = loop shape
       changed sharply, typically one weekday dropped by a holiday (e.g. Labor Day).

Outputs (in OUT_DIR):
    tda_daily_trips.csv          daily counts used (all trips + purposes, holiday flag)
    tda_window_features.csv      one row per window per series: H1 max, 2nd H1, noise
                                 reference, robust z, flag (usable as ML features)
    tda_unusual_periods.csv      flagged periods (consecutive flagged windows merged)
    tda_monthly.csv              monthly mean H1 per series, joined to the merged monthly
                                 table if found
    tda_overview.png, tda_point_clouds.png, tda_by_purpose.png

In Jupyter: just run the cell. Requirements:
    pip install numpy pandas matplotlib ripser openpyxl
    (ripser works on Python 3.12; giotto-tda does not)
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from pandas.tseries.holiday import USFederalHolidayCalendar
from ripser import ripser

# ============================================================
# USER CONFIGURATION
# ============================================================
TRIPS_PATH = "merged/rockingham_fy24_trips_clean.csv"   # or the original FY24 .xlsx
MONTHLY_PATH = "merged/rockingham_fy24_monthly_merged.csv"
OUT_DIR = "tda_output"

WINDOW_DAYS = 21          # sliding window length (3 weeks)
TAKENS_DIM = 7            # 7 consecutive days per point = one week
TAKENS_DELAY = 1
N_SHUFFLES = 30           # day-shuffled copies per window for the noise reference
Z_FLAG = 2.5              # |robust z| above this = unusual weekly rhythm
SEED = 0

# Analysed separately, plus "ALL". Dialysis is left out on purpose: its week (Mon/Wed/Fri
# high, Tue/Thu/Sat lower, Sunday off) is very regular but not loop-shaped, so max H1 sits
# BELOW the shuffled-noise level for every embedding tried (dim/delay 7/1, 3/2, 4/2, 2/2,
# 3/3, with and without point normalisation). H1 would wrongly call it irregular.
PURPOSES = ["Medical", "Employment"]
COLORS = {"ALL": "#2a6fdb", "Dialysis": "#e07b00", "Medical": "#2e9e5b", "Employment": "#8e5cc4"}


# ============================================================
# DATA
# ============================================================
def find(path, pattern):
    if path and os.path.exists(path):
        return path
    for root in (os.getcwd(), os.path.expanduser("~")):
        hits = [h for h in glob.glob(os.path.join(root, "**", pattern), recursive=True)
                if "site-packages" not in h]
        if hits:
            hits.sort(key=os.path.getmtime, reverse=True)
            print(f"Using: {hits[0]}")
            return hits[0]
    return None


def load_trips():
    p = find(TRIPS_PATH, "rockingham_fy24_trips_clean*.csv")
    if p:
        t = pd.read_csv(p, parse_dates=["TRIP_DATE"])
    else:
        p = find(None, "*FY24*Geocode*Rockingham*.xlsx")
        if not p:
            raise FileNotFoundError("No trips file found. Run merge_rockingham_data.py first "
                                    "or set TRIPS_PATH at the top of this script.")
        t = pd.read_excel(p).rename(columns={"DateOfTrip": "TRIP_DATE", "Trip Purpose": "TRIP_PURPOSE"})
        t["TRIP_DATE"] = pd.to_datetime(t["TRIP_DATE"])
    return t


def daily_counts(t):
    days = pd.date_range(t["TRIP_DATE"].min().to_period("M").to_timestamp(),
                         t["TRIP_DATE"].max().to_period("M").to_timestamp("M"), freq="D")
    d = pd.DataFrame(index=days)
    d["ALL"] = t.groupby("TRIP_DATE").size()
    for p in PURPOSES:
        d[p] = t[t["TRIP_PURPOSE"] == p].groupby("TRIP_DATE").size()
    d = d.fillna(0).astype(int)
    hol = USFederalHolidayCalendar().holidays(days.min(), days.max())
    d["HOLIDAY"] = d.index.isin(hol)
    d.index.name = "DATE"
    return d


# ============================================================
# TDA
# ============================================================
def scale_windows(W):
    lo, hi = W.min(axis=1, keepdims=True), W.max(axis=1, keepdims=True)
    return 2 * (W - lo) / np.where(hi > lo, hi - lo, 1.0) - 1


def takens(w):
    n = len(w) - (TAKENS_DIM - 1) * TAKENS_DELAY
    return np.stack([w[i * TAKENS_DELAY: i * TAKENS_DELAY + n] for i in range(TAKENS_DIM)], axis=1)


def h1_top2(cloud):
    dgm = ripser(cloud, maxdim=1)["dgms"][1]
    if len(dgm) == 0:
        return 0.0, 0.0, dgm
    pers = np.sort(dgm[:, 1] - dgm[:, 0])[::-1]
    return float(pers[0]), float(pers[1]) if len(pers) > 1 else 0.0, dgm


def window_features(series, name, rng):
    x = series.to_numpy(float)
    W = scale_windows(sliding_window_view(x, WINDOW_DAYS))
    centers = series.index[WINDOW_DAYS // 2: WINDOW_DAYS // 2 + len(W)]
    rows = []
    for i, w in enumerate(W):
        h1, h1b, _ = h1_top2(takens(w))
        null = [h1_top2(takens(rng.permutation(w)))[0] for _ in range(N_SHUFFLES)]
        rows.append({"SERIES": name, "WINDOW_CENTER": centers[i],
                     "WINDOW_START": series.index[i], "WINDOW_END": series.index[i + WINDOW_DAYS - 1],
                     "H1_MAX": h1, "H1_SECOND": h1b,
                     "NULL_H1_P95": float(np.percentile(null, 95)),
                     "MEAN_TRIPS": float(x[i:i + WINDOW_DAYS].mean())})
    f = pd.DataFrame(rows)
    med = f["H1_MAX"].median()
    mad = 1.4826 * (f["H1_MAX"] - med).abs().median()
    f["ROBUST_Z"] = (f["H1_MAX"] - med) / (mad if mad > 0 else 1.0)
    f["ABOVE_NOISE"] = f["H1_MAX"] > f["NULL_H1_P95"]
    f["FLAG"] = np.where(f["ROBUST_Z"] < -Z_FLAG, "LOW", np.where(f["ROBUST_Z"] > Z_FLAG, "HIGH", ""))
    return f


def unusual_periods(f, daily):
    out = []
    for name, g in f.groupby("SERIES", sort=False):
        g = g.reset_index(drop=True)
        flagged = g["FLAG"] != ""
        run = (g["FLAG"] != g["FLAG"].shift()).cumsum()
        for _, r in g[flagged].groupby(run[flagged]):
            start, end = r["WINDOW_CENTER"].min(), r["WINDOW_CENTER"].max()
            span = daily.loc[r["WINDOW_START"].min():r["WINDOW_END"].max()]
            hol = [f"{d:%Y-%m-%d}" for d in span.index[span["HOLIDAY"]]]
            low_wd = span[(span.index.dayofweek < 5) & (span["ALL"] < 0.5 * daily["ALL"][daily.index.dayofweek < 5].median())]
            z = r["ROBUST_Z"]
            out.append({"SERIES": name, "FLAG": r["FLAG"].iloc[0],
                        "CENTER_FROM": start, "CENTER_TO": end, "N_WINDOWS": len(r),
                        "PEAK_ROBUST_Z": z.loc[z.abs().idxmax()],
                        "HOLIDAYS_IN_SPAN": ", ".join(hol),
                        "LOW_WEEKDAYS_IN_SPAN": ", ".join(f"{d:%Y-%m-%d}" for d in low_wd.index)})
    return pd.DataFrame(out)


# ============================================================
# FIGURES
# ============================================================
def shade_holidays(ax, daily):
    for d in daily.index[daily["HOLIDAY"]]:
        ax.axvline(d, color="#c0392b", alpha=0.35, lw=1)


def fig_overview(daily, f, out_png):
    g = f[f["SERIES"] == "ALL"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True, gridspec_kw={"height_ratios": [1, 1.2]})
    a1.bar(daily.index, daily["ALL"], width=1.0, color="#9db8e8")
    shade_holidays(a1, daily)
    a1.set_title("Daily trips (red lines = federal holidays)", loc="left", fontsize=10)
    a1.set_ylabel("trips")

    a2.plot(g["WINDOW_CENTER"], g["H1_MAX"], color=COLORS["ALL"], lw=1.8, label="max H1 persistence")
    a2.plot(g["WINDOW_CENTER"], g["NULL_H1_P95"], color="#8a8a8a", lw=1, ls="--",
            label="noise reference (95th pct, days shuffled)")
    for _, r in g[g["FLAG"] != ""].iterrows():
        a2.axvspan(r["WINDOW_CENTER"] - pd.Timedelta(hours=12), r["WINDOW_CENTER"] + pd.Timedelta(hours=12),
                   color="#c0392b" if r["FLAG"] == "LOW" else "#2e9e5b", alpha=0.2, lw=0)
    shade_holidays(a2, daily)
    a2.set_title(f"Weekly-rhythm strength: max H1 of {WINDOW_DAYS}-day windows "
                 f"(Takens dim {TAKENS_DIM}, delay {TAKENS_DELAY}); bands: red = LOW, green = HIGH",
                 loc="left", fontsize=10)
    a2.set_ylabel("persistence")
    a2.legend(fontsize=8, loc="lower left")
    for a in (a1, a2):
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_png}")
    plt.show()


def pca2(X):
    Xc = X - X.mean(axis=0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T


def fig_point_clouds(daily, f, out_png):
    g = f[f["SERIES"] == "ALL"].reset_index(drop=True)
    hol = daily.index[daily["HOLIDAY"]]
    clean = g[[not ((hol >= r.WINDOW_START) & (hol <= r.WINDOW_END)).any() for r in g.itertuples()]]
    picks = [("Typical window", g.loc[(clean["ROBUST_Z"]).abs().idxmin()]),
             ("Lowest (rhythm broken)", g.loc[g["ROBUST_Z"].idxmin()]),
             ("Highest (one weekday dropped)", g.loc[g["ROBUST_Z"].idxmax()])]
    fig, axes = plt.subplots(3, 3, figsize=(15, 12.5))
    for row, (label, r) in enumerate(picks):
        seg = daily.loc[r["WINDOW_START"]:r["WINDOW_END"], "ALL"]
        w = scale_windows(seg.to_numpy(float)[None, :])[0]
        cloud = takens(w)
        h1, _, dgm = h1_top2(cloud)

        ax = axes[row, 0]
        ax.bar(seg.index, seg.values, color=COLORS["ALL"], width=0.8)
        for d in seg.index[daily.loc[seg.index, "HOLIDAY"]]:
            ax.bar(d, seg[d], color="#c0392b", width=0.8)
        ax.set_title(f"{label}: {r['WINDOW_START']:%b %d} - {r['WINDOW_END']:%b %d, %Y}",
                     loc="left", fontsize=10)
        ax.tick_params(axis="x", rotation=45, labelsize=7)

        ax = axes[row, 1]
        p = pca2(cloud)
        ax.plot(p[:, 0], p[:, 1], color="#bbbbbb", lw=0.8, zorder=1)
        sc = ax.scatter(p[:, 0], p[:, 1], c=np.arange(len(p)), cmap="viridis", s=40, zorder=2)
        ax.set_title("Takens point cloud (PCA view, colour = time)", loc="left", fontsize=10)
        ax.set_aspect("equal", adjustable="datalim")

        ax = axes[row, 2]
        if len(dgm):
            ax.scatter(dgm[:, 0], dgm[:, 1], color=COLORS["ALL"], s=30)
            m = max(dgm.max(), 0.1) * 1.1
        else:
            m = 1
        ax.plot([0, m], [0, m], color="#8a8a8a", lw=1)
        ax.set_xlim(0, m); ax.set_ylim(0, m)
        ax.set_xlabel("birth"); ax.set_ylabel("death")
        ax.set_title(f"H1 persistence diagram (max H1 = {h1:.2f})", loc="left", fontsize=10)
        for a in axes[row]:
            a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_png}")
    plt.show()


def fig_by_purpose(daily, f, out_png):
    names = list(dict.fromkeys(f["SERIES"]))
    fig, axes = plt.subplots(len(names), 1, figsize=(15, 2.4 * len(names)), sharex=True)
    for ax, name in zip(axes, names):
        g = f[f["SERIES"] == name]
        ax.plot(g["WINDOW_CENTER"], g["H1_MAX"], color=COLORS.get(name, "#333"), lw=1.6)
        ax.plot(g["WINDOW_CENTER"], g["NULL_H1_P95"], color="#8a8a8a", lw=1, ls="--")
        for flag, col in (("LOW", "#c0392b"), ("HIGH", "#2e9e5b")):
            m = g["FLAG"] == flag
            ax.scatter(g.loc[m, "WINDOW_CENTER"], g.loc[m, "H1_MAX"], color=col, s=12, zorder=3)
        shade_holidays(ax, daily)
        share = 100 * g["ABOVE_NOISE"].mean()
        ax.set_title(f"{name}: median max H1 {g['H1_MAX'].median():.2f}; "
                     f"{share:.0f}% of windows above noise reference", loc="left", fontsize=10)
        ax.grid(alpha=0.3)
    fig.suptitle("Weekly-rhythm strength by trip purpose (dashed = noise reference, red = LOW, green = HIGH)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_png}")
    plt.show()


# ============================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    out = lambda n: os.path.join(OUT_DIR, n)
    rng = np.random.default_rng(SEED)

    daily = daily_counts(load_trips())
    daily.to_csv(out("tda_daily_trips.csv"))
    print(f"Daily series: {len(daily)} days, {daily.index.min():%Y-%m-%d} to {daily.index.max():%Y-%m-%d}")

    f = pd.concat([window_features(daily[c], c, rng) for c in ["ALL"] + PURPOSES], ignore_index=True)
    f.to_csv(out("tda_window_features.csv"), index=False)

    summary = f.groupby("SERIES", sort=False).agg(
        median_H1=("H1_MAX", "median"), median_noise_ref=("NULL_H1_P95", "median"),
        pct_above_noise=("ABOVE_NOISE", lambda s: 100 * s.mean()),
        low_windows=("FLAG", lambda s: (s == "LOW").sum()),
        high_windows=("FLAG", lambda s: (s == "HIGH").sum()))
    print("\nWeekly-rhythm strength by series:")
    print(summary.round(3).to_string())

    dis = unusual_periods(f, daily)
    dis.to_csv(out("tda_unusual_periods.csv"), index=False)
    print("\nUnusual periods (all trips; window centres):")
    cols = ["FLAG", "CENTER_FROM", "CENTER_TO", "PEAK_ROBUST_Z", "HOLIDAYS_IN_SPAN", "LOW_WEEKDAYS_IN_SPAN"]
    a = dis[dis["SERIES"] == "ALL"][cols]
    print(a.to_string(index=False) if len(a) else "  none")

    monthly = (f.assign(MONTH_START=f["WINDOW_CENTER"].dt.to_period("M").dt.to_timestamp())
                 .pivot_table(index="MONTH_START", columns="SERIES", values="H1_MAX", aggfunc="mean")
                 .add_prefix("TDA_H1_MEAN_").reset_index())
    mp = find(MONTHLY_PATH, "rockingham_fy24_monthly_merged*.csv")
    if mp:
        monthly = pd.read_csv(mp, parse_dates=["MONTH_START"]).merge(monthly, on="MONTH_START", how="left")
    monthly.to_csv(out("tda_monthly.csv"), index=False)

    fig_overview(daily, f, out("tda_overview.png"))
    fig_point_clouds(daily, f, out("tda_point_clouds.png"))
    fig_by_purpose(daily, f, out("tda_by_purpose.png"))
    print(f"\nOutputs in: {os.path.abspath(OUT_DIR)}")
    return daily, f, dis


if __name__ == "__main__":
    daily, features, unusual = main()
