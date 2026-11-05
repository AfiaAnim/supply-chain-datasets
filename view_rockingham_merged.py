"""
View the merged Rockingham FY24 dataset made by merge_rockingham_data.py
(merged/rockingham_fy24_monthly_merged.csv: one row per month, Jul 2023 - Jun 2024).

Shows:
    - the table, summary statistics, and missing values
    - Figure 1: foot traffic vs FY24 trips by month (4 panels)
    - Figure 2: trips by purpose per month (stacked bars) + share of each purpose
    - Figure 3: correlation between foot-traffic and trip measures
Figures are shown and saved as PNGs next to the CSV.

In Jupyter: just run the cell. Command line:
    python view_rockingham_merged.py [path/to/rockingham_fy24_monthly_merged.csv]

Requirements
    pip install pandas matplotlib
"""

import glob
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

# ============================================================
# USER CONFIGURATION (optional: put the full path here)
# ============================================================
MERGED_PATH = "merged/rockingham_fy24_monthly_merged.csv"

BLUE, ORANGE, GREY = "#2a6fdb", "#e07b00", "#8a8a8a"


def find_merged(path):
    if path and os.path.exists(path):
        return path
    for root in (os.getcwd(), os.path.expanduser("~")):
        hits = glob.glob(os.path.join(root, "**", "rockingham_fy24_monthly_merged*.csv"),
                         recursive=True)
        if hits:
            hits.sort(key=os.path.getmtime, reverse=True)
            print(f"Using: {hits[0]}")
            return hits[0]
    raise FileNotFoundError(
        "Could not find rockingham_fy24_monthly_merged.csv. Run merge_rockingham_data.py "
        "first, or set MERGED_PATH at the top of this script.")


def show(df):
    """Pretty table in Jupyter, plain text elsewhere."""
    try:
        from IPython.display import display
        get_ipython  # noqa: F821  (only defined inside IPython/Jupyter)
        display(df)
    except (ImportError, NameError):
        print(df.to_string())


def overview(df):
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", "{:,.2f}".format)

    print(f"Rows: {len(df)}   Columns: {df.shape[1]}   "
          f"Months: {df['MONTH_START'].min():%Y-%m} to {df['MONTH_START'].max():%Y-%m}")

    main_cols = ["MONTH_START", "FT_N_BLOCK_GROUPS", "RAW_DEVICE_COUNTS", "RAW_STOP_COUNTS",
                 "DEVICES_PER_BLOCK_GROUP", "DISTANCE_FROM_HOME_WTD", "MEDIAN_DWELL_WTD",
                 "TOTAL_TRIPS", "SERVICE_DAYS", "TRIPS_PER_SERVICE_DAY",
                 "TRIPS_WITH_MOBILITY_AID", "TRIPS_BAD_COORDS"]
    main_cols = [c for c in main_cols if c in df.columns]
    print("\nMain columns by month:")
    show(df[main_cols].assign(MONTH_START=df["MONTH_START"].dt.strftime("%Y-%m")))

    print("\nSummary statistics:")
    show(df[main_cols[1:]].describe().T)

    purpose = [c for c in df.columns if c.startswith("TRIPS_")
               and c not in ("TRIPS_PER_SERVICE_DAY", "TRIPS_WITH_MOBILITY_AID", "TRIPS_BAD_COORDS")]
    totals = df[purpose].sum().sort_values(ascending=False)
    print("\nTrips by purpose (whole FY24):")
    show(pd.DataFrame({"trips": totals, "share_%": 100 * totals / totals.sum()}))

    missing = df.isna().sum()
    print("\nMissing values:", "none" if not missing.any() else "")
    if missing.any():
        print(missing[missing > 0].to_string())
    return purpose


def fig_over_time(df, out_png):
    x = df["MONTH_START"]
    labels = x.dt.strftime("%b\n%Y")
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)

    ax = axes[0, 0]
    ax.bar(x, df["RAW_DEVICE_COUNTS"], width=20, color=BLUE, alpha=0.85)
    ax.set_title("Foot traffic: devices seen (sum of block groups)", loc="left", fontsize=10)
    ax2 = ax.twinx()
    ax2.plot(x, df["FT_N_BLOCK_GROUPS"], color=ORANGE, marker="o")
    ax2.set_ylabel("block groups reporting", color=ORANGE, fontsize=8)
    ax2.tick_params(axis="y", colors=ORANGE, labelsize=8)

    ax = axes[0, 1]
    ax.plot(x, df["DEVICES_PER_BLOCK_GROUP"], color=BLUE, marker="o")
    ax.set_title("Devices per reporting block group (coverage-adjusted)", loc="left", fontsize=10)

    ax = axes[1, 0]
    ax.bar(x, df["TOTAL_TRIPS"], width=20, color=GREY, alpha=0.85, label="total trips")
    ax.set_title("FY24 trips per month", loc="left", fontsize=10)
    ax2 = ax.twinx()
    ax2.plot(x, df["TRIPS_PER_SERVICE_DAY"], color=ORANGE, marker="o")
    ax2.set_ylabel("trips per service day", color=ORANGE, fontsize=8)
    ax2.tick_params(axis="y", colors=ORANGE, labelsize=8)

    ax = axes[1, 1]
    ax.plot(x, df["DISTANCE_FROM_HOME_WTD"] / 1000, color=BLUE, marker="o", label="distance from home (km)")
    ax.set_ylabel("km", color=BLUE, fontsize=8)
    ax2 = ax.twinx()
    ax2.plot(x, df["MEDIAN_DWELL_WTD"], color=ORANGE, marker="s", label="median dwell (min)")
    ax2.set_ylabel("minutes", color=ORANGE, fontsize=8)
    ax.set_title("Visitor distance from home (blue) and dwell time (orange), stop-weighted",
                 loc="left", fontsize=10)

    for ax in axes.flat:
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=8)
        ax.set_xticks(x, labels)
    fig.suptitle("Rockingham County FY24: foot traffic and trips by month", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_png}")
    plt.show()


def fig_purpose(df, purpose, out_png):
    order = df[purpose].sum().sort_values(ascending=False).index
    top = list(order[:7])
    data = df.set_index(df["MONTH_START"].dt.strftime("%b %Y"))[top].copy()
    data["All smaller purposes"] = df[[c for c in order if c not in top]].sum(axis=1).values
    data.columns = [c.replace("TRIPS_", "").replace("_", " ") for c in data.columns]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw={"width_ratios": [2.2, 1]})
    data.plot(kind="bar", stacked=True, ax=ax1, colormap="tab10", width=0.8)
    ax1.set_title("Trips by purpose per month", loc="left", fontsize=10)
    ax1.set_xlabel("")
    ax1.tick_params(axis="x", rotation=45, labelsize=8)
    ax1.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False)
    ax1.grid(axis="y", alpha=0.3)

    share = data.sum().sort_values()
    ax2.barh(share.index, 100 * share / share.sum(), color=BLUE)
    for i, v in enumerate(100 * share / share.sum()):
        ax2.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=8)
    ax2.set_title("Share of FY24 trips", loc="left", fontsize=10)
    ax2.tick_params(labelsize=8)
    ax2.set_xlim(0, 100 * share.max() / share.sum() + 8)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_png}")
    plt.show()


def fig_corr(df, out_png):
    cols = ["RAW_DEVICE_COUNTS", "DEVICES_PER_BLOCK_GROUP", "FT_N_BLOCK_GROUPS",
            "DISTANCE_FROM_HOME_WTD", "MEDIAN_DWELL_WTD", "TOTAL_TRIPS",
            "TRIPS_PER_SERVICE_DAY", "TRIPS_DIALYSIS", "TRIPS_MEDICAL", "TRIPS_EMPLOYMENT"]
    cols = [c for c in cols if c in df.columns]
    corr = df[cols].corr()
    n = len(cols)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    names = [c.replace("_", " ").title() for c in cols]
    ax.set_xticks(range(n), names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n), names, fontsize=8)
    for r in range(n):
        for c in range(n):
            v = corr.iloc[r, c]
            ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f"Correlation across months (n = {len(df)}: treat as rough)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_png}")
    plt.show()


def main(path=None):
    if path is None and len(sys.argv) > 1 and sys.argv[1].endswith(".csv"):
        path = sys.argv[1]
    path = find_merged(path or MERGED_PATH)
    df = pd.read_csv(path, parse_dates=["MONTH_START"]).sort_values("MONTH_START")

    purpose = overview(df)
    base = os.path.splitext(path)[0]
    fig_over_time(df, base + "_over_time.png")
    fig_purpose(df, purpose, base + "_by_purpose.png")
    fig_corr(df, base + "_correlation.png")
    return df


if __name__ == "__main__":
    merged = main()
