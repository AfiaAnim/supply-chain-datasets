"""
View the Daily Merged Transportation dataset.

Prints an overview in the terminal (shape, columns, the data itself, summary
statistics, missing values, date gaps, weekday vs weekend averages) and draws one
figure with:
    1. Each metric over time (one small panel per metric, weekends marked)
    2. Correlation heatmap between the metrics

The figure is shown and also saved as a PNG next to the CSV.

Usage:
    python view_transportation_data.py path/to/Daily_Merged_Transportation_Data.csv
    python view_transportation_data.py            # uses CSV_PATH below

Requirements:
    pip install pandas matplotlib
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# USER CONFIGURATION
# ============================================================
CSV_PATH = "Daily_Merged_Transportation_Data.csv"

METRICS = [
    "RAW_DEVICE_COUNTS",
    "RAW_STOP_COUNTS",
    "DISTANCE_FROM_HOME",
    "MEDIAN_DWELL",
    "TOTAL_TRIPS",
]

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def load(path):
    df = pd.read_csv(path, parse_dates=["DATE"])
    df = df.sort_values("DATE").reset_index(drop=True)
    if "DAY_OF_WEEK" in df.columns:
        df["DAY_NAME"] = df["DAY_OF_WEEK"].map(dict(enumerate(DAY_NAMES)))
    return df


def print_overview(df):
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", "{:,.2f}".format)

    print("=" * 70)
    print(f"Rows: {len(df)}   Columns: {df.shape[1]}")
    print(f"Date range: {df['DATE'].min().date()} to {df['DATE'].max().date()}")
    print("=" * 70)

    print("\nColumn types:")
    print(df.dtypes.to_string())

    print("\nData:")
    print(df.to_string(index=False))

    metrics = [m for m in METRICS if m in df.columns]
    print("\nSummary statistics:")
    print(df[metrics].describe().T.to_string())

    missing = df.isna().sum()
    print("\nMissing values:")
    print(missing[missing > 0].to_string() if missing.any() else "  none")

    # Gaps: report the spacing between dates and any months with no row.
    steps = df["DATE"].diff().dt.days.dropna()
    if len(steps):
        print(f"\nDays between rows: min {steps.min():.0f}, "
              f"median {steps.median():.0f}, max {steps.max():.0f}")
    months = df["DATE"].dt.to_period("M")
    all_months = pd.period_range(months.min(), months.max(), freq="M")
    missing_months = sorted(set(all_months) - set(months))
    print("Months with no data: "
          + (", ".join(str(m) for m in missing_months) if missing_months else "none"))

    if "WEEKEND" in df.columns:
        print("\nMean by weekday (0) vs weekend (1):")
        print(df.groupby("WEEKEND")[metrics].mean().to_string())


def plot(df, out_path):
    metrics = [m for m in METRICS if m in df.columns]
    n = len(metrics)

    fig = plt.figure(figsize=(14, 2.4 * n))
    grid = fig.add_gridspec(n, 2, width_ratios=[2.2, 1], hspace=0.5, wspace=0.25)

    weekend = df["WEEKEND"].astype(bool) if "WEEKEND" in df.columns else None

    for i, col in enumerate(metrics):
        ax = fig.add_subplot(grid[i, 0])
        ax.plot(df["DATE"], df[col], color="#2a6fdb", marker="o", linewidth=1.5)
        if weekend is not None and weekend.any():
            ax.scatter(df.loc[weekend, "DATE"], df.loc[weekend, col],
                       color="#e07b00", zorder=3, s=45,
                       label="Weekend" if i == 0 else None)
        ax.set_title(col, loc="left", fontsize=10)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=8)
        if i < n - 1:
            ax.tick_params(labelbottom=False)
        if i == 0 and weekend is not None and weekend.any():
            ax.legend(fontsize=8, loc="upper right")

    ax = fig.add_subplot(grid[: max(1, (n + 1) // 2 + 1), 1])
    corr = df[metrics].corr()
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    labels = [m.replace("_", "\n") for m in metrics]
    ax.set_xticks(range(n), labels, fontsize=7)
    ax.set_yticks(range(n), labels, fontsize=7)
    for r in range(n):
        for c in range(n):
            v = corr.iloc[r, c]
            ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.6 else "black")
    ax.set_title("Correlation", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.5)

    fig.suptitle("Daily Merged Transportation Data", fontsize=13)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to: {out_path}")
    plt.show()


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else CSV_PATH
    if not os.path.exists(path):
        sys.exit(f"File not found: {path}")

    df = load(path)
    print_overview(df)

    out_path = os.path.splitext(path)[0] + "_overview.png"
    plot(df, out_path)


if __name__ == "__main__":
    main()
