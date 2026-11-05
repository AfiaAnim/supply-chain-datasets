"""
Merge the 3 Rockingham datasets and view the result (one Jupyter cell).

    1. rockingham_county_data.csv           foot traffic, calendar 2024, per census block group
    2. Rockingham_Foot_Traffic_2023.csv      foot traffic, calendar 2023, same columns
    3. FY24_Geocode_FINAL_Rockingham_for_NCAT.xlsx
                                             one row per trip, FY24 (Jul 2023 - Jun 2024)
                                             (OrigX/DestX are LATITUDE, OrigY/DestY LONGITUDE)

MERGE
    A. Stack 2023 + 2024 foot traffic -> 24 months, one row per block group per month.
    B. Monthly merge: foot traffic summed per month + trips per month, all 24 months.
       IN_FY24 marks the 12 months that have trips; trip columns are blank in the others.
    C. Block-group merge (needs geopandas + internet once): each trip's pickup and drop-off
       is placed in a 2010 census block group (the foot-traffic data's boundaries), giving
       a block group x month table with trips next to that block group's foot traffic.
       Census boundaries are downloaded once from www2.census.gov into data/census/.

VIEW
    Tables printed in the notebook, and figures:
    Fig 1  monthly foot traffic (2023 + 2024), block groups reporting, FY24 trips by purpose
    Fig 2  coverage: which block group-months have foot traffic and trips (step C)
    Fig 3  block group-months with both: foot traffic vs trips starting there (step C)

READ BEFORE INTERPRETING
    Each foot-traffic month holds only a few block groups (1 to 18), so county monthly
    sums mostly reflect how many block groups were exported; use DEVICES_PER_BLOCK_GROUP.

In Jupyter: paste into one cell and run. Outputs go to OUT_DIR. Requirements:
    pip install pandas numpy matplotlib openpyxl      (+ geopandas for step C)
"""

import fnmatch
import json
import os
import re
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================================================
# SETTINGS: put full paths here if the files are elsewhere
# ============================================================
FOOT_2024_PATH = "rockingham_county_data.csv"
FOOT_2023_PATH = "Rockingham_Foot_Traffic_2023.csv"
TRIPS_PATH = "FY24_Geocode_FINAL_Rockingham_for_NCAT.xlsx"
OUT_DIR = "merged"
CENSUS_DIR = "data/census"
DO_BLOCK_GROUPS = True        # step C (needs geopandas)

BG2010_URL = "https://www2.census.gov/geo/tiger/TIGER2010/BG/2010/tl_2010_37157_bg10.zip"
PATTERNS = {"foot_2024": "*rockingham_county_data*.csv",
            "foot_2023": "*Rockingham_Foot_Traffic_2023*.csv",
            "trips": "*FY24*Geocode*Rockingham*.xlsx"}
NC_LAT, NC_LON = (33.8, 36.6), (-84.4, -75.4)     # flags impossible coordinates

# colours: categorical slots + one-hue blue ramp
C = {"2023": "#2a78d6", "2024": "#eb6834", "ink": "#0b0b0b", "ink2": "#52514e",
     "grid": "#e4e3df", "surface": "#fcfcfb", "none": "#efeeeb"}
PURPOSE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#8a8986"]
plt.rcParams.update({"figure.facecolor": C["surface"], "axes.facecolor": C["surface"],
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
                     "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "left"})


# ============================================================
# FILES
# ============================================================
def find_file(path, pattern, label):
    """Use `path` if it exists, else search this folder and then home by name pattern."""
    if path and os.path.exists(path):
        return path
    for root in (os.getcwd(), os.path.expanduser("~")):
        hits = []
        for d, dirs, files in os.walk(root):
            dirs[:] = [x for x in dirs if not x.startswith(".") and x not in ("site-packages", OUT_DIR)
                       and not os.path.exists(os.path.join(d, x, "pyvenv.cfg"))]
            hits += [os.path.join(d, f) for f in files if fnmatch.fnmatch(f.lower(), pattern.lower())]
        if hits:
            hits.sort(key=os.path.getmtime, reverse=True)
            print(f"Using {label}: {hits[0]}")
            return hits[0]
    raise FileNotFoundError(f"Could not find the {label} file ({pattern}). Copy it into "
                            f"{os.getcwd()} or set its path in SETTINGS.")


def show(df, n=None):
    df = df if n is None else df.head(n)
    try:
        from IPython.display import display
        get_ipython  # noqa: F821  (only defined in Jupyter)
        display(df)
    except (ImportError, NameError):
        print(df.to_string())


# ============================================================
# A. FOOT TRAFFIC 2023 + 2024
# ============================================================
def device_counts(cell):
    """DEVICE_TYPE cells look like '"{\\"android\\":198,\\"ios\\":70}"'."""
    if not isinstance(cell, str):
        return {}
    s = cell.strip().strip('"').replace('\\"', '"').replace('""', '"')
    try:
        return json.loads(s)
    except ValueError:
        return {}


def load_foot_traffic(path_2023, path_2024):
    parts = []
    for year, p in (("2023", path_2023), ("2024", path_2024)):
        d = pd.read_csv(p, dtype={"AREA": str})
        d["SOURCE_YEAR"] = year
        parts.append(d)
    ft = pd.concat(parts, ignore_index=True)
    ft["MONTH_START"] = pd.to_datetime(ft["DATE_RANGE_START"]).dt.to_period("M").dt.to_timestamp()
    dev = ft["DEVICE_TYPE"].map(device_counts)
    ft["ANDROID_DEVICES"] = dev.map(lambda x: x.get("android"))
    ft["IOS_DEVICES"] = dev.map(lambda x: x.get("ios"))
    ft = ft.drop_duplicates(["AREA", "MONTH_START"])
    first = ["AREA", "MONTH_START", "SOURCE_YEAR", "RAW_DEVICE_COUNTS", "RAW_STOP_COUNTS",
             "DISTANCE_FROM_HOME", "MEDIAN_DWELL", "ANDROID_DEVICES", "IOS_DEVICES"]
    return ft[first + [c for c in ft.columns if c not in first]].sort_values(["MONTH_START", "AREA"])


def monthly_foot_traffic(ft):
    g = ft.groupby("MONTH_START")
    m = pd.DataFrame({"FT_BLOCK_GROUPS": g["AREA"].nunique(),
                      "FT_DEVICES": g["RAW_DEVICE_COUNTS"].sum(),
                      "FT_STOPS": g["RAW_STOP_COUNTS"].sum(),
                      "FT_MEDIAN_DWELL_MEAN": g["MEDIAN_DWELL"].mean(),
                      "FT_DISTANCE_FROM_HOME_MEAN": g["DISTANCE_FROM_HOME"].mean()})
    m["FT_DEVICES_PER_BLOCK_GROUP"] = m["FT_DEVICES"] / m["FT_BLOCK_GROUPS"]
    return m


# ============================================================
# TRIPS
# ============================================================
def load_trips(path):
    t = pd.read_excel(path).rename(columns={
        "OrigX": "ORIG_LAT", "OrigY": "ORIG_LON", "DestX": "DEST_LAT", "DestY": "DEST_LON",
        "DateOfTrip": "TRIP_DATE", "Mobility Aid": "MOBILITY_AID", "Trip Purpose": "TRIP_PURPOSE"})
    t["TRIP_DATE"] = pd.to_datetime(t["TRIP_DATE"])
    t["MONTH_START"] = t["TRIP_DATE"].dt.to_period("M").dt.to_timestamp()
    for e in ("ORIG", "DEST"):
        t[f"{e}_COORD_OK"] = t[f"{e}_LAT"].between(*NC_LAT) & t[f"{e}_LON"].between(*NC_LON)
    t["HEALTHCARE"] = t["TRIP_PURPOSE"].isin(["Dialysis", "Medical"])
    t["RECORDED_AID"] = t["MOBILITY_AID"].notna()
    return t


def monthly_trips(t):
    g = t.groupby("MONTH_START")
    m = pd.DataFrame({"TRIPS": g.size(), "TRIP_SERVICE_DAYS": g["TRIP_DATE"].nunique(),
                      "TRIPS_HEALTHCARE": g["HEALTHCARE"].sum(),
                      "TRIPS_WITH_RECORDED_AID": g["RECORDED_AID"].sum()})
    m["TRIPS_PER_SERVICE_DAY"] = m["TRIPS"] / m["TRIP_SERVICE_DAYS"]
    slug = lambda s: "TRIPS_" + re.sub(r"[^A-Z0-9]+", "_", str(s).upper()).strip("_")
    by_purpose = (t.assign(P=t["TRIP_PURPOSE"].fillna("Unknown").map(slug))
                   .pivot_table(index="MONTH_START", columns="P", aggfunc="size", fill_value=0))
    return m.join(by_purpose)


# ============================================================
# C. BLOCK GROUPS
# ============================================================
def block_group_merge(t, ft):
    import geopandas as gpd
    os.makedirs(CENSUS_DIR, exist_ok=True)
    shp = os.path.join(CENSUS_DIR, os.path.basename(BG2010_URL))
    if not os.path.exists(shp):
        print(f"Downloading census block groups (2010) from {BG2010_URL} ...")
        urllib.request.urlretrieve(BG2010_URL, shp + ".part")
        os.replace(shp + ".part", shp)
    bg = gpd.read_file(shp).rename(columns={"GEOID10": "BG2010"})[["BG2010", "geometry"]].to_crs("EPSG:4326")

    for e in ("ORIG", "DEST"):
        ok = t[f"{e}_COORD_OK"]
        pts = gpd.GeoDataFrame(index=t.index[ok], crs="EPSG:4326",
                               geometry=gpd.points_from_xy(t.loc[ok, f"{e}_LON"], t.loc[ok, f"{e}_LAT"]))
        j = gpd.sjoin(pts, bg, how="left", predicate="within")
        t[f"{e}_BG2010"] = j.loc[~j.index.duplicated(), "BG2010"].reindex(t.index)

    months = sorted(set(ft["MONTH_START"]) | set(t["MONTH_START"]))
    grid = pd.MultiIndex.from_product([sorted(bg["BG2010"]), months], names=["BG2010", "MONTH_START"])
    frm = t.groupby(["ORIG_BG2010", "MONTH_START"]).agg(
        TRIPS_FROM=("TRIP_DATE", "size"), HEALTHCARE_TRIPS_FROM=("HEALTHCARE", "sum"),
        TRIPS_FROM_WITH_AID=("RECORDED_AID", "sum")).rename_axis(["BG2010", "MONTH_START"])
    to = t.groupby(["DEST_BG2010", "MONTH_START"]).size().rename("TRIPS_TO").rename_axis(["BG2010", "MONTH_START"])
    bgm = frm.join(to, how="outer").reindex(grid).reset_index()
    bgm["IN_FY24"] = bgm["MONTH_START"].isin(t["MONTH_START"].unique())
    cols = ["TRIPS_FROM", "HEALTHCARE_TRIPS_FROM", "TRIPS_FROM_WITH_AID", "TRIPS_TO"]
    bgm.loc[bgm["IN_FY24"], cols] = bgm.loc[bgm["IN_FY24"], cols].fillna(0)   # 0 inside FY24, blank outside
    bgm[cols] = bgm[cols].astype("Int64")
    bgm = bgm.merge(ft.rename(columns={"AREA": "BG2010"})[
        ["BG2010", "MONTH_START", "RAW_DEVICE_COUNTS", "RAW_STOP_COUNTS", "DISTANCE_FROM_HOME", "MEDIAN_DWELL"]],
        on=["BG2010", "MONTH_START"], how="left")
    bgm["FT_AVAILABLE"] = bgm["RAW_DEVICE_COUNTS"].notna()
    return t, bgm


# ============================================================
# VIEW
# ============================================================
def fig_monthly(m, out_png):
    x = np.arange(len(m))
    labels = [d.strftime("%b\n%Y") for d in m.index]
    year = m.index.year.astype(str)
    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True,
                             gridspec_kw={"height_ratios": [1.1, 0.7, 1.1, 1.3]})

    ax = axes[0]
    ax.bar(x, m["FT_DEVICES"], 0.75, color=[C[y] for y in year], edgecolor=C["surface"], linewidth=1.5)
    ax.set_title("Foot traffic: devices seen (sum of the block groups in that month's file)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C[y], label=f"{y} file") for y in ("2023", "2024")],
              frameon=False, loc="upper right", ncol=2)

    ax = axes[1]
    ax.bar(x, m["FT_BLOCK_GROUPS"], 0.75, color=[C[y] for y in year], edgecolor=C["surface"], linewidth=1.5)
    for i, v in enumerate(m["FT_BLOCK_GROUPS"]):
        ax.text(i, v, f"{int(v)}", ha="center", va="bottom", fontsize=7.5, color=C["ink2"])
    ax.set_title("Block groups in the foot-traffic file that month (of 69)")

    ax = axes[2]
    ax.plot(x, m["FT_DEVICES_PER_BLOCK_GROUP"], color=C["ink2"], lw=2, marker="o", ms=5)
    ax.set_title("Foot traffic per reporting block group (adjusts for how many were exported)")

    ax = axes[3]
    purposes = [c for c in m.columns if c.startswith("TRIPS_") and c not in
                ("TRIPS_HEALTHCARE", "TRIPS_WITH_RECORDED_AID", "TRIPS_PER_SERVICE_DAY")]
    order = m[purposes].sum().sort_values(ascending=False).index
    top, rest = list(order[:7]), list(order[7:])
    bottom = np.zeros(len(m))
    for col, colr in zip(top, PURPOSE_COLORS):
        v = m[col].fillna(0).to_numpy()
        ax.bar(x, v, 0.75, bottom=bottom, color=colr, edgecolor=C["surface"], linewidth=1,
               label=col.replace("TRIPS_", "").replace("_", " ").title())
        bottom += v
    if rest:
        v = m[rest].fillna(0).sum(axis=1).to_numpy()
        ax.bar(x, v, 0.75, bottom=bottom, color=PURPOSE_COLORS[7], edgecolor=C["surface"], linewidth=1,
               label="All smaller purposes")
    nofy = ~m["IN_FY24"].to_numpy()
    for i in np.where(nofy)[0]:
        ax.text(i, 0, "no trip\ndata", ha="center", va="bottom", fontsize=7, color=C["ink2"])
    ax.set_title("FY24 trips by purpose (trip file covers Jul 2023 - Jun 2024 only)")
    ax.legend(frameon=False, fontsize=7.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.32))
    ax.set_xticks(x, labels)

    for ax in axes:
        ax.grid(axis="y", color=C["grid"], lw=0.8)
        ax.set_axisbelow(True)
        ax.axvspan(-0.5, 5.5, color=C["none"], zorder=0)       # months outside FY24
        ax.axvspan(17.5, 23.5, color=C["none"], zorder=0)
    fig.suptitle("Rockingham County: foot traffic 2023-2024 merged with FY24 trips (grey = outside FY24)",
                 fontsize=12, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_png}")
    plt.show()


def fig_coverage(bgm, out_png):
    """Each cell = block group x month: foot traffic only / trips only / both / neither."""
    has_trips = bgm["TRIPS_FROM"].fillna(0) > 0
    state = np.select([bgm["FT_AVAILABLE"] & has_trips, bgm["FT_AVAILABLE"], has_trips],
                      [3, 2, 1], default=0)
    grid = bgm.assign(S=state).pivot(index="BG2010", columns="MONTH_START", values="S")
    grid = grid.loc[grid.sum(axis=1).sort_values(ascending=False).index]
    cmap = plt.matplotlib.colors.ListedColormap([C["none"], "#9ec5f4", "#eb6834", "#104281"])
    fig, ax = plt.subplots(figsize=(13, 10))
    ax.imshow(grid.to_numpy(), cmap=cmap, vmin=-0.5, vmax=3.5, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(grid.shape[1]), [d.strftime("%b %y") for d in grid.columns], rotation=90, fontsize=7)
    ax.set_yticks(range(grid.shape[0]), grid.index, fontsize=5.5)
    ax.set_xticks(np.arange(-0.5, grid.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0]), minor=True)
    ax.grid(which="minor", color=C["surface"], lw=0.8)
    ax.tick_params(which="minor", length=0)
    for lab, col in [("neither", C["none"]), ("trips only", "#9ec5f4"),
                     ("foot traffic only", "#eb6834"), ("both", "#104281")]:
        ax.scatter([], [], marker="s", s=80, color=col, label=lab)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.04))
    n_both = int((state == 3).sum())
    ax.set_title(f"Block group x month coverage: {int(bgm['FT_AVAILABLE'].sum())} with foot traffic, "
                 f"{n_both} with both foot traffic and trips", pad=28)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_png}")
    plt.show()


def fig_both(bgm, out_png):
    b = bgm[bgm["FT_AVAILABLE"] & bgm["IN_FY24"]].copy()
    b["TRIPS_FROM"] = b["TRIPS_FROM"].astype(float)
    rho = b["RAW_DEVICE_COUNTS"].corr(b["TRIPS_FROM"], method="spearman")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(b["RAW_DEVICE_COUNTS"], b["TRIPS_FROM"] + 1, s=36, color=C["2023"],
               edgecolor=C["surface"], linewidth=1.5, alpha=0.85)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("foot traffic: devices seen in the block group that month")
    ax.set_ylabel("FY24 trips starting there that month (+1, log scale)")
    ax.grid(color=C["grid"], lw=0.8)
    ax.set_title(f"FY24 block group-months with foot traffic (n = {len(b)}): "
                 f"Spearman rank correlation {rho:.2f}")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_png}")
    plt.show()


# ============================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    out = lambda n: os.path.join(OUT_DIR, n)

    # A
    ft = load_foot_traffic(find_file(FOOT_2023_PATH, PATTERNS["foot_2023"], "2023 foot traffic"),
                           find_file(FOOT_2024_PATH, PATTERNS["foot_2024"], "2024 foot traffic"))
    ft.to_csv(out("rockingham_foot_traffic_2023_2024.csv"), index=False)
    t = load_trips(find_file(TRIPS_PATH, PATTERNS["trips"], "FY24 trips"))

    # B
    m = monthly_foot_traffic(ft).join(monthly_trips(t), how="outer").sort_index()
    m.insert(0, "IN_FY24", m.index.isin(t["MONTH_START"].unique()))
    m.index.name = "MONTH_START"
    m.to_csv(out("rockingham_2023_2024_monthly_merged.csv"))

    print(f"\nFoot traffic: {len(ft)} rows, {ft['AREA'].nunique()} block groups, "
          f"{ft['MONTH_START'].min():%b %Y} - {ft['MONTH_START'].max():%b %Y} "
          f"(2023 file {int((ft['SOURCE_YEAR'] == '2023').sum())} rows, 2024 file "
          f"{int((ft['SOURCE_YEAR'] == '2024').sum())} rows)")
    print(f"Trips: {len(t):,}, {t['TRIP_DATE'].min():%d %b %Y} - {t['TRIP_DATE'].max():%d %b %Y}; "
          f"{int((~(t['ORIG_COORD_OK'] & t['DEST_COORD_OK'])).sum())} with missing/impossible coordinates")
    print("\nMonthly merge (24 months; trip columns blank outside FY24):")
    show(m[["IN_FY24", "FT_BLOCK_GROUPS", "FT_DEVICES", "FT_DEVICES_PER_BLOCK_GROUP", "TRIPS",
            "TRIPS_PER_SERVICE_DAY", "TRIPS_HEALTHCARE", "TRIPS_WITH_RECORDED_AID"]].round(1))
    fig_monthly(m, out("view_monthly.png"))

    # C
    bgm = None
    if DO_BLOCK_GROUPS:
        try:
            import geopandas  # noqa: F401
        except ImportError:
            print("\nBlock-group merge skipped: needs geopandas (%pip install geopandas, restart, re-run).")
        else:
            t, bgm = block_group_merge(t, ft)
            bgm.to_csv(out("rockingham_2023_2024_bg_month_merged.csv"), index=False)
            t.to_csv(out("rockingham_fy24_trips_with_block_groups.csv"), index=False)
            inside = t["ORIG_BG2010"].notna().sum()
            print(f"\nBlock-group merge: {inside:,} trips start inside a Rockingham block group; "
                  f"{len(bgm):,} block group-months ({bgm['BG2010'].nunique()} block groups x "
                  f"{bgm['MONTH_START'].nunique()} months)")
            print("Block group-months with foot traffic (first rows):")
            show(bgm[bgm["FT_AVAILABLE"]][["BG2010", "MONTH_START", "IN_FY24", "RAW_DEVICE_COUNTS",
                                          "RAW_STOP_COUNTS", "TRIPS_FROM", "TRIPS_TO",
                                          "HEALTHCARE_TRIPS_FROM"]], n=15)
            fig_coverage(bgm, out("view_coverage.png"))
            fig_both(bgm, out("view_foot_traffic_vs_trips.png"))

    print(f"\nOutputs in: {os.path.abspath(OUT_DIR)}")
    return ft, t, m, bgm


if __name__ == "__main__":
    foot_traffic, trips, monthly, bg_month = main()
