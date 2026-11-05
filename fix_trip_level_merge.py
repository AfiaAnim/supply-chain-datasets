"""
Correct the trip-level merge of the Rockingham datasets (one Jupyter cell).

THE PROBLEM (Trip_Level_Merged_Data.csv)
    Trips were joined to foot traffic on YEAR + MONTH only. Each foot-traffic month holds
    several block groups (1 to 18), so every trip was copied once per block group:
    297,856 rows for 39,037 trips, February trips 18 times, January trips once, and the
    foot-traffic values on a row have nothing to do with where that trip starts.

THE FIX: one row per trip (39,037 rows), same 13 columns, plus link columns
    Option A (default, needs geopandas): the trip's pickup is placed in its 2010 census
        block group (the boundaries the foot-traffic data uses); the foot traffic of THAT
        block group in THAT month is attached. Blank when that block group is not in the
        foot-traffic file that month (FT_AVAILABLE = False).
    Option B (no geopandas needed): one county-level summary per month instead
        (mean per reporting block group, plus how many block groups reported).

    Added columns:
        TRIP_ID             row number in the original FY24 trip file (unique key)
        ORIG_BG2010         pickup block group (option A)
        FT_SOURCE           "pickup block group" / "county monthly mean" / "none"
        FT_AVAILABLE        True if the foot-traffic columns are filled
        FT_BLOCK_GROUPS     block groups in that month's foot-traffic file

    It also checks the result: row count = trips, no duplicated TRIP_ID, and (if found)
    that the old file is exactly trips x block groups per month.

Outputs (OUT_DIR):
    Trip_Level_Merged_Data_fixed.csv            option A (or B if geopandas is missing)
    Trip_Level_Merged_Data_county_monthly.csv   option B, always written

Requirements:  pip install pandas openpyxl   (+ geopandas for option A)
"""

import fnmatch
import os
import urllib.request

import numpy as np
import pandas as pd

# ============================================================
# SETTINGS (full paths here if the files are elsewhere)
# ============================================================
FOOT_2024_PATH = "rockingham_county_data.csv"
FOOT_2023_PATH = "Rockingham_Foot_Traffic_2023.csv"
TRIPS_PATH = "FY24_Geocode_FINAL_Rockingham_for_NCAT.xlsx"
OLD_MERGED_PATH = "Trip_Level_Merged_Data.csv"            # optional: only used for the check
OUT_DIR = "merged"
CENSUS_DIR = "data/census"
BG2010_URL = "https://www2.census.gov/geo/tiger/TIGER2010/BG/2010/tl_2010_37157_bg10.zip"

PATTERNS = {"foot_2024": "*rockingham_county_data*.csv", "foot_2023": "*Rockingham_Foot_Traffic_2023*.csv",
            "trips": "*FY24*Geocode*Rockingham*.xlsx", "old": "*Trip_Level_Merged_Data*.csv"}
FT_COLS = ["RAW_DEVICE_COUNTS", "RAW_STOP_COUNTS", "DISTANCE_FROM_HOME", "MEDIAN_DWELL"]
TRIP_COLS = ["DateOfTrip", "Mobility Aid", "Trip Purpose", "OrigX", "OrigY", "DestX", "DestY"]
NC_LAT, NC_LON = (33.8, 36.6), (-84.4, -75.4)


def find_file(path, pattern, label, required=True):
    if path and os.path.exists(path):
        return path
    for root in (os.getcwd(), os.path.expanduser("~")):
        hits = []
        for d, dirs, files in os.walk(root):
            dirs[:] = [x for x in dirs if not x.startswith(".") and x not in ("site-packages", OUT_DIR)
                       and not os.path.exists(os.path.join(d, x, "pyvenv.cfg"))]
            hits += [os.path.join(d, f) for f in files if fnmatch.fnmatch(f.lower(), pattern.lower())
                     and "_fixed" not in f and "_county_monthly" not in f]
        if hits:
            hits.sort(key=os.path.getmtime, reverse=True)
            print(f"Using {label}: {hits[0]}")
            return hits[0]
    if required:
        raise FileNotFoundError(f"Could not find the {label} file ({pattern}); set its path in SETTINGS.")
    return None


def load_inputs():
    ft = pd.concat([pd.read_csv(find_file(FOOT_2023_PATH, PATTERNS["foot_2023"], "2023 foot traffic"),
                                dtype={"AREA": str}),
                    pd.read_csv(find_file(FOOT_2024_PATH, PATTERNS["foot_2024"], "2024 foot traffic"),
                                dtype={"AREA": str})], ignore_index=True)
    ft["YM"] = pd.to_datetime(ft["DATE_RANGE_START"]).dt.to_period("M")
    ft = ft.drop_duplicates(["AREA", "YM"])

    t = pd.read_excel(find_file(TRIPS_PATH, PATTERNS["trips"], "FY24 trips"))
    t.insert(0, "TRIP_ID", np.arange(1, len(t) + 1))
    t["DateOfTrip"] = pd.to_datetime(t["DateOfTrip"])
    t["YEAR"], t["MONTH"] = t["DateOfTrip"].dt.year, t["DateOfTrip"].dt.month
    t["YM"] = t["DateOfTrip"].dt.to_period("M")
    return ft, t


def option_a(t, ft):
    """Foot traffic of the trip's own pickup block group in the trip's month."""
    import geopandas as gpd
    os.makedirs(CENSUS_DIR, exist_ok=True)
    shp = os.path.join(CENSUS_DIR, os.path.basename(BG2010_URL))
    if not os.path.exists(shp):
        print(f"Downloading census block groups (2010) from {BG2010_URL} ...")
        urllib.request.urlretrieve(BG2010_URL, shp + ".part")
        os.replace(shp + ".part", shp)
    bg = gpd.read_file(shp).rename(columns={"GEOID10": "ORIG_BG2010"})[["ORIG_BG2010", "geometry"]]
    bg = bg.to_crs("EPSG:4326")

    ok = t["OrigX"].between(*NC_LAT) & t["OrigY"].between(*NC_LON)     # OrigX = latitude
    pts = gpd.GeoDataFrame(index=t.index[ok], crs="EPSG:4326",
                           geometry=gpd.points_from_xy(t.loc[ok, "OrigY"], t.loc[ok, "OrigX"]))
    j = gpd.sjoin(pts, bg, how="left", predicate="within")
    out = t.copy()
    out["ORIG_BG2010"] = j.loc[~j.index.duplicated(), "ORIG_BG2010"].reindex(t.index)

    f = ft.rename(columns={"AREA": "ORIG_BG2010"})[["ORIG_BG2010", "YM"] + FT_COLS]
    out = out.merge(f, on=["ORIG_BG2010", "YM"], how="left", validate="many_to_one")
    out["FT_AVAILABLE"] = out["RAW_DEVICE_COUNTS"].notna()
    out["FT_SOURCE"] = np.where(out["FT_AVAILABLE"], "pickup block group", "none")
    return out


def option_b(t, ft):
    """One county-level foot-traffic summary per month (mean per reporting block group)."""
    m = ft.groupby("YM")[FT_COLS].mean()
    out = t.merge(m, left_on="YM", right_index=True, how="left", validate="many_to_one")
    out["ORIG_BG2010"] = np.nan
    out["FT_AVAILABLE"] = out["RAW_DEVICE_COUNTS"].notna()
    out["FT_SOURCE"] = np.where(out["FT_AVAILABLE"], "county monthly mean", "none")
    return out


def finish(out, ft):
    out["FT_BLOCK_GROUPS"] = out["YM"].map(ft.groupby("YM")["AREA"].nunique()).fillna(0).astype(int)
    out["DateOfTrip"] = out["DateOfTrip"].dt.strftime("%Y-%m-%d")
    cols = ["TRIP_ID"] + TRIP_COLS + ["YEAR", "MONTH"] + FT_COLS + \
           ["ORIG_BG2010", "FT_SOURCE", "FT_AVAILABLE", "FT_BLOCK_GROUPS"]
    return out[cols]


def check(fixed, t, ft):
    assert len(fixed) == len(t), "row count differs from the number of trips"
    assert fixed["TRIP_ID"].is_unique, "a trip appears more than once"
    print(f"\nCHECK  fixed file: {len(fixed):,} rows = {len(t):,} trips, each TRIP_ID once.")
    old_p = find_file(OLD_MERGED_PATH, PATTERNS["old"], "old merged file", required=False)
    if old_p:
        old = pd.read_csv(old_p, usecols=["DateOfTrip"])
        old_m = pd.to_datetime(old["DateOfTrip"]).dt.to_period("M").value_counts().sort_index()
        n_bg = ft.groupby("YM")["AREA"].nunique()
        expected = (t.groupby("YM").size() * n_bg).dropna().astype(int)
        same = bool((old_m.reindex(expected.index) == expected).all())
        print(f"CHECK  old file: {len(old):,} rows; equals trips x block groups in every month: {same}")
        fixed_m = pd.to_datetime(fixed["DateOfTrip"]).dt.to_period("M").value_counts()
        show = pd.DataFrame({"trips": t.groupby("YM").size(), "block_groups": n_bg,
                             "old_rows": old_m, "fixed_rows": fixed_m})
        print(show.dropna().astype(int).to_string())


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ft, t = load_inputs()

    b = finish(option_b(t, ft), ft)
    b.to_csv(os.path.join(OUT_DIR, "Trip_Level_Merged_Data_county_monthly.csv"), index=False)

    try:
        import geopandas  # noqa: F401
        fixed = finish(option_a(t, ft), ft)
        kind = "A (pickup block group)"
    except ImportError:
        print("geopandas not installed: writing option B as the fixed file "
              "(%pip install geopandas, restart, re-run for option A).")
        fixed, kind = b, "B (county monthly mean)"
    path = os.path.join(OUT_DIR, "Trip_Level_Merged_Data_fixed.csv")
    fixed.to_csv(path, index=False)

    check(fixed, t, ft)
    n_ft = int(fixed["FT_AVAILABLE"].sum())
    print(f"\nFixed file, option {kind}: {path}")
    print(f"  trips with foot traffic attached: {n_ft:,} ({100 * n_ft / len(fixed):.1f}%)")
    if kind.startswith("A"):
        print(f"  trips whose pickup is outside a Rockingham block group or has bad coordinates: "
              f"{int(fixed['ORIG_BG2010'].isna().sum()):,}")
    try:
        from IPython.display import display
        get_ipython  # noqa: F821
        display(fixed.head(10))
    except (ImportError, NameError):
        print(fixed.head(10).to_string())
    return fixed, b


if __name__ == "__main__":
    fixed_trips, county_monthly_trips = main()
