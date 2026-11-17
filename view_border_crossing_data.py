"""
View the already-downloaded Border_Crossing_Entry_Data.csv (US BTS border crossing data).

Paste the whole file into ONE Jupyter cell and run it, or from a terminal:
    python view_border_crossing_data.py
    python view_border_crossing_data.py "C:/path/to/Border_Crossing_Entry_Data.csv"

Nothing is downloaded: the script only reads the CSV you already have.
If CSV_PATH below is wrong, it also looks in the current folder, the script's
folder and your Downloads folder.

Shows:
    - shape, columns, data types, missing values, first rows
    - date range, unique ports / states / borders / measures
    - top ports, totals by border and by measure
    - 3 plots: monthly totals per border, top 10 ports, totals by measure
"""

import glob
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

try:                                    # make plots appear inside Jupyter
    from IPython import get_ipython
    if get_ipython() is not None:
        get_ipython().run_line_magic("matplotlib", "inline")
except Exception:
    pass


# ============================================================
# SETTINGS
# ============================================================

CSV_PATH = "Border_Crossing_Entry_Data.csv"     # <- put your full path here if needed
N_PREVIEW_ROWS = 10
TOP_N = 10
SHOW_PLOTS = True


# ============================================================
# FIND THE FILE
# ============================================================

def find_csv(path):
    if os.path.isfile(path):
        return path
    name = os.path.basename(path)
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:                   # running inside Jupyter
        script_dir = os.getcwd()
    folders = [os.getcwd(), script_dir, os.path.join(os.path.expanduser("~"), "Downloads")]
    for folder in folders:
        candidate = os.path.join(folder, name)
        if os.path.isfile(candidate):
            return candidate
        # browsers rename repeat downloads, e.g. "Border_Crossing_Entry_Data (1).csv"
        matches = sorted(glob.glob(os.path.join(folder, "Border_Crossing_Entry_Data*.csv")))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"Could not find {name}. Set CSV_PATH at the top of this script to its full path.\n"
        f"Looked in: {folders}"
    )


# Command-line argument overrides CSV_PATH (ignored inside Jupyter, which passes its own args)
if len(sys.argv) > 1 and sys.argv[1].lower().endswith(".csv"):
    CSV_PATH = sys.argv[1]

csv_file = find_csv(CSV_PATH)


# ============================================================
# LOAD
# ============================================================

print(f"Loading: {csv_file}")
df = pd.read_csv(csv_file, low_memory=False)
df.columns = [c.strip() for c in df.columns]

# The BTS file stores Date as text like "Jan 2024" (older copies: "01/01/2024 12:00:00 AM")
if "Date" in df.columns:
    parsed = pd.to_datetime(df["Date"], format="%b %Y", errors="coerce")
    if parsed.isna().mean() > 0.5:
        parsed = pd.to_datetime(df["Date"], errors="coerce")
    df["Date"] = parsed

if "Value" in df.columns:
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)


# ============================================================
# OVERVIEW
# ============================================================

def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)

section("SHAPE")
print(f"{df.shape[0]:,} rows x {df.shape[1]} columns")

section("COLUMNS AND TYPES")
print(df.dtypes.to_string())

section("MISSING VALUES")
missing = df.isna().sum()
print(missing[missing > 0].to_string() if missing.any() else "None")

section(f"FIRST {N_PREVIEW_ROWS} ROWS")
print(df.head(N_PREVIEW_ROWS).to_string())

if "Date" in df.columns and df["Date"].notna().any():
    section("DATE RANGE")
    print(f"{df['Date'].min():%b %Y}  ->  {df['Date'].max():%b %Y}")

section("UNIQUE COUNTS")
for col in ["Port Name", "State", "Border", "Measure"]:
    if col in df.columns:
        print(f"{col:<10}: {df[col].nunique():,}")

for col in ["Border", "Measure"]:
    if col in df.columns:
        print(f"\n{col} values: {sorted(df[col].dropna().unique())}")

has_value = "Value" in df.columns

if has_value and "Border" in df.columns:
    section("TOTAL CROSSINGS BY BORDER")
    print(df.groupby("Border")["Value"].sum().sort_values(ascending=False).to_string())

if has_value and "Measure" in df.columns:
    section("TOTAL CROSSINGS BY MEASURE")
    by_measure = df.groupby("Measure")["Value"].sum().sort_values(ascending=False)
    print(by_measure.to_string())

if has_value and "Port Name" in df.columns:
    section(f"TOP {TOP_N} PORTS (all measures, all years)")
    port_cols = [c for c in ["Port Name", "State", "Border"] if c in df.columns]
    by_port = df.groupby(port_cols)["Value"].sum().sort_values(ascending=False).head(TOP_N)
    print(by_port.to_string())


# ============================================================
# PLOTS
# ============================================================

if SHOW_PLOTS and has_value:
    fig, axes = plt.subplots(3, 1, figsize=(12, 14))

    ax = axes[0]
    if "Date" in df.columns and "Border" in df.columns:
        monthly = df.groupby(["Date", "Border"])["Value"].sum().unstack("Border")
        (monthly / 1e6).plot(ax=ax)
        ax.set_title("Monthly total crossings by border")
        ax.set_ylabel("Crossings (millions)")
        ax.set_xlabel("")
        ax.grid(alpha=0.3)

    ax = axes[1]
    if "Port Name" in df.columns:
        top_ports = df.groupby(["Port Name", "State"])["Value"].sum().nlargest(TOP_N)
        labels = [f"{p}, {s}" for p, s in top_ports.index]
        ax.barh(labels[::-1], top_ports.values[::-1] / 1e6)
        ax.set_title(f"Top {TOP_N} ports (all years, all measures)")
        ax.set_xlabel("Crossings (millions)")
        ax.grid(axis="x", alpha=0.3)

    ax = axes[2]
    if "Measure" in df.columns:
        ax.barh(by_measure.index[::-1], by_measure.values[::-1] / 1e6)
        ax.set_title("Total by measure (all years)")
        ax.set_xlabel("Count (millions)")
        ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.show()

# `df` stays in memory, so in Jupyter you can keep exploring, e.g.:
#   df[df["Port Name"] == "El Paso"].head()
#   df[df["Measure"] == "Trucks"].groupby("Date")["Value"].sum().plot()
