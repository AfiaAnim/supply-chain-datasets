"""
Merge the Rockingham County foot-traffic files with the FY24 trip records.

Inputs
    1. rockingham_county_data.csv          monthly foot traffic per census block group, 2024
    2. Rockingham_Foot_Traffic_2023.csv     same columns, 2023
    3. FY24_Geocode_FINAL_Rockingham_for_NCAT.xlsx
                                            one row per trip (Jul 2023 - Jun 2024): date,
                                            mobility aid, purpose, origin/destination coords.
                                            NOTE: OrigX/DestX hold LATITUDE and OrigY/DestY
                                            hold LONGITUDE (the names are swapped).

Steps
    A. Stack the two foot-traffic files into one table (one row per block group per month).
    B. Aggregate both sources to calendar month and join them on MONTH. Trips are counted
       for the WHOLE month (not just the 1st, which is what Daily_Merged_Transportation_Data
       did).
    C. Census linkage (needs geopandas; files are downloaded once from www2.census.gov into
       CENSUS_DIR and reused):
       - each trip's pickup and drop-off is placed in a 2010 census block group. The
         foot-traffic data uses 2010 block groups: all 65 of its IDs exist in 2010, while
         4 (tract 415, split in 2020) do not exist in 2020/2023 boundaries.
       - each trip's pickup is placed in a 2020 census tract and given that tract's share
         of households with no vehicle (ACS 2019-2023 5-year, table B25044).
       - the pickup block group's foot traffic for the trip's month is attached.
       Then: block group x month table (all 69 block groups of 2010 x all 24 foot-traffic
       months, trips next to foot traffic), block group summary, and tract summary with
       vehicle availability.

    TWO YEARS OF FOOT TRAFFIC, ONE FISCAL YEAR OF TRIPS
       The foot-traffic files cover calendar 2023 and 2024 (24 months); the trips cover
       FY24 only (Jul 2023 - Jun 2024). The 2023_2024 outputs keep all 24 months, with
       IN_FY24 = True for the 12 months that have trip data; trip columns are blank (not 0)
       in the other 12 months, because no trips were recorded in this file for them.

Outputs (written to --out-dir)
    rockingham_foot_traffic_2023_2024.csv   step A
    rockingham_2023_2024_monthly_merged.csv step B, all 24 foot-traffic months + trips (FY24)
    rockingham_fy24_monthly_merged.csv      step B, the 12 FY24 months only
    rockingham_fy24_trips_clean.csv         FY24 trips with clearly named coords + flags
    rockingham_fy24_trips_enriched.csv      step C: trips + block group, tract, foot traffic, ACS
    rockingham_2023_2024_bg_month_merged.csv step C: block group x 24 months, trips + foot traffic
    rockingham_fy24_bg_month_merged.csv     step C: the same, FY24 months only
    rockingham_bg_summary.csv               step C: one row per 2010 block group, FY24
    rockingham_tract_summary.csv            step C: one row per 2020 tract, trips + vehicles

Usage
    python merge_rockingham_data.py \
        --foot-2024 rockingham_county_data.csv \
        --foot-2023 Rockingham_Foot_Traffic_2023.csv \
        --trips FY24_Geocode_FINAL_Rockingham_for_NCAT.xlsx \
        --out-dir merged/ [--no-census]

    In Jupyter: just run the cell. If an input file is not at the path set below, the
    script searches the notebook folder and then your home folder for it by name, so
    names like "FY24_..._NCAT (1).xlsx" are found too. Or pass paths explicitly:
        main(["--trips", "/full/path/to/FY24_Geocode_FINAL_Rockingham_for_NCAT.xlsx"])

NOTES on the foot-traffic data
    Every row is a Rockingham County block group, but each month holds only a few of
    them (1 to 18 of 69; most appear in 1-4 of 24 months). Block-group-month foot traffic
    is therefore missing for most trips (FT_AVAILABLE flags it), and county monthly sums
    mostly reflect how many block groups were exported that month.

Requirements
    pip install pandas openpyxl geopandas     (geopandas only for step C)
"""

import argparse
import fnmatch
import json
import os
import re

import numpy as np
import pandas as pd


# ============================================================
# USER CONFIGURATION: put full paths here if you like (optional)
# ============================================================
FOOT_2024_PATH = "rockingham_county_data.csv"
FOOT_2023_PATH = "Rockingham_Foot_Traffic_2023.csv"
TRIPS_PATH = "FY24_Geocode_FINAL_Rockingham_for_NCAT.xlsx"
OUT_DIR = "merged"
CENSUS_DIR = "data/census"        # downloaded census files are cached here

CENSUS = {
    "bg2010": "https://www2.census.gov/geo/tiger/TIGER2010/BG/2010/tl_2010_37157_bg10.zip",
    "tract2023": "https://www2.census.gov/geo/tiger/TIGER2023/TRACT/tl_2023_37_tract.zip",
    "acs_b25044": "https://www2.census.gov/programs-surveys/acs/summary_file/2023/"
                  "table-based-SF/data/5YRData/acsdt5y2023-b25044.dat",
}
STATE_COUNTY = "37157"            # North Carolina, Rockingham County

# Name patterns used to search for a file that is not at the path above.
SEARCH_PATTERNS = {
    "foot_2024": "*rockingham_county_data*.csv",
    "foot_2023": "*Rockingham_Foot_Traffic_2023*.csv",
    "trips": "*FY24*Geocode*Rockingham*.xlsx",
}
SKIP_DIRS = {"site-packages", "node_modules", "__pycache__", "merged"}

# Rough North Carolina bounding box, used to flag bad coordinates
# (the FY24 file has a few at lon -99, lon 0, lat 42).
NC_LAT = (33.8, 36.6)
NC_LON = (-84.4, -75.4)


# ------------------------------------------------------------
# Locating input files
# ------------------------------------------------------------
def _search(root, pattern):
    hits = []
    for dirpath, dirnames, filenames in os.walk(root):
        # skip hidden folders (.cache, .local, ...), Python environments and outputs
        dirnames[:] = [d for d in dirnames
                       if not d.startswith(".") and d not in SKIP_DIRS
                       and not os.path.exists(os.path.join(dirpath, d, "pyvenv.cfg"))]
        hits += [os.path.join(dirpath, f) for f in filenames
                 if fnmatch.fnmatch(f.lower(), pattern.lower()) and not f.startswith("~$")]
    return hits


def find_file(path, pattern, label):
    """Return `path` if it exists, else search the current folder, then home, by pattern."""
    if path and os.path.exists(path):
        return path
    for root in (os.getcwd(), os.path.expanduser("~")):
        hits = _search(root, pattern)
        if hits:
            hits.sort(key=os.path.getmtime, reverse=True)
            if len(hits) > 1:
                print(f"NOTE: {len(hits)} files match {label}; using the newest. Others:")
                for h in hits[1:]:
                    print(f"      {h}")
            print(f"Using {label}: {hits[0]}")
            return hits[0]
    raise FileNotFoundError(
        f"Could not find the {label} file.\n"
        f"  Looked for: {path}\n"
        f"  Searched {os.getcwd()} and {os.path.expanduser('~')} for '{pattern}'.\n"
        f"  Copy the file into {os.getcwd()} or set its full path at the top of the script.")


# ------------------------------------------------------------
# A. Foot traffic
# ------------------------------------------------------------
def parse_json_cell(value):
    """Foot-traffic JSON cells are double-quoted, e.g. '"{\\"android\\":198}"'."""
    if not isinstance(value, str) or not value.strip():
        return None
    s = value.strip()
    if s.startswith('"') and s.endswith('"'):
        s = s[1:-1]
    s = s.replace('\\"', '"').replace('""', '"')
    try:
        return json.loads(s)
    except ValueError:
        return None


def load_foot_traffic(paths):
    frames = []
    for p in paths:
        df = pd.read_csv(p, dtype={"AREA": str})
        df["SOURCE_FILE"] = os.path.basename(p)
        frames.append(df)
    ft = pd.concat(frames, ignore_index=True)

    ft["MONTH_START"] = pd.to_datetime(ft["DATE_RANGE_START"]).dt.to_period("M").dt.to_timestamp()
    devices = ft["DEVICE_TYPE"].map(parse_json_cell)
    ft["ANDROID_DEVICES"] = devices.map(lambda d: (d or {}).get("android"))
    ft["IOS_DEVICES"] = devices.map(lambda d: (d or {}).get("ios"))

    dupes = ft.duplicated(["AREA", "MONTH_START"]).sum()
    if dupes:
        print(f"WARNING: {dupes} duplicate block-group/month rows in foot traffic; keeping first")
        ft = ft.drop_duplicates(["AREA", "MONTH_START"])

    front = ["AREA", "MONTH_START", "Y", "M", "RAW_DEVICE_COUNTS", "RAW_STOP_COUNTS",
             "DISTANCE_FROM_HOME", "MEDIAN_DWELL", "ANDROID_DEVICES", "IOS_DEVICES"]
    ft = ft[front + [c for c in ft.columns if c not in front]]
    return ft.sort_values(["MONTH_START", "AREA"]).reset_index(drop=True)


def monthly_foot_traffic(ft):
    ft = ft.assign(_dist_w=ft["DISTANCE_FROM_HOME"] * ft["RAW_STOP_COUNTS"],
                   _dwell_w=ft["MEDIAN_DWELL"] * ft["RAW_STOP_COUNTS"])
    g = ft.groupby("MONTH_START")
    out = pd.DataFrame({
        "FT_N_BLOCK_GROUPS": g["AREA"].nunique(),
        "RAW_DEVICE_COUNTS": g["RAW_DEVICE_COUNTS"].sum(),
        "RAW_STOP_COUNTS": g["RAW_STOP_COUNTS"].sum(),
        # Plain mean across block groups (matches the earlier merged file) ...
        "DISTANCE_FROM_HOME": g["DISTANCE_FROM_HOME"].mean(),
        "MEDIAN_DWELL": g["MEDIAN_DWELL"].mean(),
        # ... and a stop-weighted mean, less swayed by tiny block groups.
        "DISTANCE_FROM_HOME_WTD": g["_dist_w"].sum() / g["RAW_STOP_COUNTS"].sum(),
        "MEDIAN_DWELL_WTD": g["_dwell_w"].sum() / g["RAW_STOP_COUNTS"].sum(),
    })
    out["DEVICES_PER_BLOCK_GROUP"] = out["RAW_DEVICE_COUNTS"] / out["FT_N_BLOCK_GROUPS"]
    return out.reset_index()


# ------------------------------------------------------------
# Trips
# ------------------------------------------------------------
def slug(text):
    return re.sub(r"[^A-Z0-9]+", "_", str(text).upper()).strip("_")


def load_trips(path):
    t = pd.read_excel(path)
    t = t.rename(columns={"OrigX": "ORIG_LAT", "OrigY": "ORIG_LON",
                          "DestX": "DEST_LAT", "DestY": "DEST_LON",
                          "DateOfTrip": "TRIP_DATE", "Mobility Aid": "MOBILITY_AID",
                          "Trip Purpose": "TRIP_PURPOSE"})
    t["TRIP_DATE"] = pd.to_datetime(t["TRIP_DATE"])
    t["MONTH_START"] = t["TRIP_DATE"].dt.to_period("M").dt.to_timestamp()

    def ok(lat, lon):
        return (t[lat].between(*NC_LAT) & t[lon].between(*NC_LON))
    t["ORIG_COORD_OK"] = ok("ORIG_LAT", "ORIG_LON")
    t["DEST_COORD_OK"] = ok("DEST_LAT", "DEST_LON")
    return t


def monthly_trips(t):
    g = t.groupby("MONTH_START")
    out = pd.DataFrame({
        "TOTAL_TRIPS": g.size(),
        "SERVICE_DAYS": g["TRIP_DATE"].nunique(),
        "TRIPS_WITH_MOBILITY_AID": g["MOBILITY_AID"].count(),
        "TRIPS_BAD_COORDS": g.apply(lambda x: (~(x.ORIG_COORD_OK & x.DEST_COORD_OK)).sum(),
                                    include_groups=False),
    })
    out["TRIPS_PER_SERVICE_DAY"] = out["TOTAL_TRIPS"] / out["SERVICE_DAYS"]

    purpose = (t.assign(P="TRIPS_" + t["TRIP_PURPOSE"].fillna("Unknown").map(slug))
                 .pivot_table(index="MONTH_START", columns="P", aggfunc="size", fill_value=0))
    return out.join(purpose).reset_index()


# ------------------------------------------------------------
# C. Census linkage
# ------------------------------------------------------------
def fetch(key):
    """Download a census file once into CENSUS_DIR and return its local path."""
    import urllib.request
    os.makedirs(CENSUS_DIR, exist_ok=True)
    path = os.path.join(CENSUS_DIR, os.path.basename(CENSUS[key]))
    if not os.path.exists(path):
        print(f"    downloading {CENSUS[key]} ...")
        tmp = path + ".part"
        urllib.request.urlretrieve(CENSUS[key], tmp)
        os.replace(tmp, path)
    return path


def acs_no_vehicle_tracts():
    """ACS 2019-2023 5-yr B25044 for Rockingham tracts: households, no-vehicle households."""
    cache = os.path.join(CENSUS_DIR, f"acs2023_b25044_tracts_{STATE_COUNTY}.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache, dtype={"TRACT": str})
    dat = fetch("acs_b25044")
    prefix = f"1400000US{STATE_COUNTY}"
    rows = []
    with open(dat) as fh:
        header = fh.readline().strip().split("|")
        for line in fh:
            if line.startswith(prefix):
                rows.append(line.strip().split("|"))
    a = pd.DataFrame(rows, columns=header)
    num = lambda c: pd.to_numeric(a[c], errors="coerce")
    out = pd.DataFrame({
        "TRACT": a["GEO_ID"].str.replace("1400000US", "", regex=False),
        "HOUSEHOLDS": num("B25044_E001"), "HOUSEHOLDS_MOE": num("B25044_M001"),
        # no vehicle = owner-occupied (E003) + renter-occupied (E010)
        "NO_VEHICLE_HH": num("B25044_E003") + num("B25044_E010"),
        "NO_VEHICLE_HH_MOE": np.sqrt(num("B25044_M003") ** 2 + num("B25044_M010") ** 2),
    })
    out["NO_VEHICLE_SHARE"] = out["NO_VEHICLE_HH"] / out["HOUSEHOLDS"]
    out.to_csv(cache, index=False)
    os.remove(dat)                                   # the national file is ~60 MB; keep the extract
    return out


def load_polygons():
    import geopandas as gpd
    bg = gpd.read_file(fetch("bg2010"))
    bg = bg.rename(columns={"GEOID10": "BG2010"})[["BG2010", "geometry"]].to_crs("EPSG:4326")
    tr = gpd.read_file(fetch("tract2023"))
    tr = tr[tr["COUNTYFP"] == STATE_COUNTY[2:]].rename(columns={"GEOID": "TRACT2020"})
    tr = tr[["TRACT2020", "geometry"]].to_crs("EPSG:4326")
    return bg, tr


def locate(t, polys, end, idcol):
    import geopandas as gpd
    ok = t[f"{end}_COORD_OK"]
    pts = gpd.GeoDataFrame(index=t.index[ok], geometry=gpd.points_from_xy(
        t.loc[ok, f"{end}_LON"], t.loc[ok, f"{end}_LAT"]), crs="EPSG:4326")
    j = gpd.sjoin(pts, polys, how="left", predicate="within")
    j = j[~j.index.duplicated()]
    return j[idcol].reindex(t.index)


def census_linkage(t, ft):
    bg, tr = load_polygons()
    acs = acs_no_vehicle_tracts()

    e = t.copy()
    e["ORIG_BG2010"] = locate(t, bg, "ORIG", "BG2010")
    e["DEST_BG2010"] = locate(t, bg, "DEST", "BG2010")
    e["ORIG_TRACT2020"] = locate(t, tr, "ORIG", "TRACT2020")
    e["DEST_TRACT2020"] = locate(t, tr, "DEST", "TRACT2020")
    e["ORIG_IN_COUNTY"] = e["ORIG_BG2010"].notna()
    e["HEALTHCARE"] = e["TRIP_PURPOSE"].isin(["Dialysis", "Medical"])
    e["RECORDED_AID"] = e["MOBILITY_AID"].notna()

    # foot traffic of the pickup block group in the trip's month
    ftc = ft[["AREA", "MONTH_START", "RAW_DEVICE_COUNTS", "RAW_STOP_COUNTS",
              "DISTANCE_FROM_HOME", "MEDIAN_DWELL"]].rename(columns={
        "AREA": "ORIG_BG2010", "RAW_DEVICE_COUNTS": "FT_ORIG_DEVICES",
        "RAW_STOP_COUNTS": "FT_ORIG_STOPS", "DISTANCE_FROM_HOME": "FT_ORIG_DISTANCE_FROM_HOME",
        "MEDIAN_DWELL": "FT_ORIG_MEDIAN_DWELL"})
    e = e.merge(ftc, on=["ORIG_BG2010", "MONTH_START"], how="left")
    e["FT_AVAILABLE"] = e["FT_ORIG_DEVICES"].notna()

    # vehicle availability of the pickup tract
    e = e.merge(acs.rename(columns={"TRACT": "ORIG_TRACT2020", "NO_VEHICLE_SHARE": "ORIG_TRACT_NO_VEHICLE_SHARE",
                                    "HOUSEHOLDS": "ORIG_TRACT_HOUSEHOLDS"})
                [["ORIG_TRACT2020", "ORIG_TRACT_HOUSEHOLDS", "ORIG_TRACT_NO_VEHICLE_SHARE"]],
                on="ORIG_TRACT2020", how="left")

    # block group x month: complete grid over ALL foot-traffic months (2023 + 2024).
    # Trip columns are counts (0 = no trips) inside the FY24 window and blank outside it,
    # because there are no trip records for those months.
    fy_months = sorted(e["MONTH_START"].unique())
    all_months = sorted(set(ft["MONTH_START"]) | set(fy_months))
    grid = pd.MultiIndex.from_product([sorted(bg["BG2010"]), all_months], names=["BG2010", "MONTH_START"])
    out_ = e.groupby(["ORIG_BG2010", "MONTH_START"]).agg(
        TRIPS_FROM=("TRIP_DATE", "size"), HEALTHCARE_TRIPS_FROM=("HEALTHCARE", "sum"),
        TRIPS_FROM_WITH_AID=("RECORDED_AID", "sum"))
    out_.index.names = ["BG2010", "MONTH_START"]
    in_ = e.groupby(["DEST_BG2010", "MONTH_START"]).size().rename("TRIPS_TO")
    in_.index.names = ["BG2010", "MONTH_START"]
    bgm = out_.join(in_, how="outer").reindex(grid).reset_index()
    bgm["IN_FY24"] = bgm["MONTH_START"].isin(fy_months)
    trip_cols = ["TRIPS_FROM", "HEALTHCARE_TRIPS_FROM", "TRIPS_FROM_WITH_AID", "TRIPS_TO"]
    bgm.loc[bgm["IN_FY24"], trip_cols] = bgm.loc[bgm["IN_FY24"], trip_cols].fillna(0)
    bgm[trip_cols] = bgm[trip_cols].astype("Int64")
    ftm = ft.rename(columns={"AREA": "BG2010"})[["BG2010", "MONTH_START", "RAW_DEVICE_COUNTS",
                                                "RAW_STOP_COUNTS", "DISTANCE_FROM_HOME", "MEDIAN_DWELL"]]
    bgm = bgm.merge(ftm, on=["BG2010", "MONTH_START"], how="left")
    bgm["FT_AVAILABLE"] = bgm["RAW_DEVICE_COUNTS"].notna()

    # block group summary over FY24 (+ foot traffic over all 24 months)
    fy = bgm[bgm["IN_FY24"]].groupby("BG2010").agg(
        TRIPS_FROM=("TRIPS_FROM", "sum"), TRIPS_TO=("TRIPS_TO", "sum"),
        HEALTHCARE_TRIPS_FROM=("HEALTHCARE_TRIPS_FROM", "sum"),
        TRIPS_FROM_WITH_AID=("TRIPS_FROM_WITH_AID", "sum"),
        FT_MONTHS_IN_FY24=("FT_AVAILABLE", "sum"))
    ftall = ft.groupby("AREA").agg(FT_MONTHS_2023_2024=("MONTH_START", "nunique"),
                                   FT_MEAN_DEVICES=("RAW_DEVICE_COUNTS", "mean"),
                                   FT_MEAN_STOPS=("RAW_STOP_COUNTS", "mean"),
                                   FT_MEAN_MEDIAN_DWELL=("MEDIAN_DWELL", "mean"))
    bgs = fy.join(ftall).fillna({"FT_MONTHS_2023_2024": 0}).reset_index()
    bgs["FT_MONTHS_2023_2024"] = bgs["FT_MONTHS_2023_2024"].astype(int)

    # tract summary with vehicle availability
    trs = e.groupby("ORIG_TRACT2020").agg(
        TRIPS_FROM=("TRIP_DATE", "size"), HEALTHCARE_TRIPS_FROM=("HEALTHCARE", "sum"),
        TRIPS_FROM_WITH_AID=("RECORDED_AID", "sum"))
    trs = acs.rename(columns={"TRACT": "TRACT2020"}).set_index("TRACT2020").join(trs).fillna(
        {"TRIPS_FROM": 0, "HEALTHCARE_TRIPS_FROM": 0, "TRIPS_FROM_WITH_AID": 0}).reset_index()
    trs["TRIPS_FROM_PER_100_HH"] = 100 * trs["TRIPS_FROM"] / trs["HOUSEHOLDS"]
    return e, bgm, bgs, trs


# ------------------------------------------------------------
def add_calendar(df):
    df.insert(1, "YEAR", df["MONTH_START"].dt.year)
    df.insert(2, "MONTH", df["MONTH_START"].dt.month)
    fy = df["MONTH_START"].dt.year + (df["MONTH_START"].dt.month >= 7)
    df.insert(3, "FISCAL_YEAR", fy)
    return df


def main(argv=None):
    """argv: list of options, e.g. ["--out-dir", "merged"]. None reads the command line.
    Unknown options (like the "-f kernel.json" Jupyter adds) are ignored."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--foot-2024", default=FOOT_2024_PATH)
    ap.add_argument("--foot-2023", default=FOOT_2023_PATH)
    ap.add_argument("--trips", default=TRIPS_PATH)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--no-census", action="store_true", help="skip step C (census linkage)")
    args, _ = ap.parse_known_args(argv)

    args.foot_2024 = find_file(args.foot_2024, SEARCH_PATTERNS["foot_2024"], "2024 foot traffic")
    args.foot_2023 = find_file(args.foot_2023, SEARCH_PATTERNS["foot_2023"], "2023 foot traffic")
    args.trips = find_file(args.trips, SEARCH_PATTERNS["trips"], "FY24 trips")
    os.makedirs(args.out_dir, exist_ok=True)
    out = lambda name: os.path.join(args.out_dir, name)

    # A
    ft = load_foot_traffic([args.foot_2023, args.foot_2024])
    ft.to_csv(out("rockingham_foot_traffic_2023_2024.csv"), index=False)
    print(f"[A] foot traffic stacked: {len(ft)} rows, {ft['AREA'].nunique()} block groups, "
          f"{ft['MONTH_START'].min():%Y-%m} to {ft['MONTH_START'].max():%Y-%m}")

    # Trips
    t = load_trips(args.trips)
    t.to_csv(out("rockingham_fy24_trips_clean.csv"), index=False)
    bad = (~(t.ORIG_COORD_OK & t.DEST_COORD_OK)).sum()
    print(f"    FY24 trips: {len(t)} rows, {t['TRIP_DATE'].min():%Y-%m-%d} to "
          f"{t['TRIP_DATE'].max():%Y-%m-%d}, {bad} with missing/out-of-NC coordinates")

    # B
    mt = monthly_trips(t)
    m = monthly_foot_traffic(ft).merge(mt, on="MONTH_START", how="inner")
    m = add_calendar(m)
    m.to_csv(out("rockingham_fy24_monthly_merged.csv"), index=False)
    print(f"[B] monthly merge (FY24 window): {len(m)} months "
          f"({m['MONTH_START'].min():%Y-%m} to {m['MONTH_START'].max():%Y-%m})")
    # All 24 foot-traffic months (2023 + 2024); trip columns blank outside FY24.
    m24 = monthly_foot_traffic(ft).merge(mt, on="MONTH_START", how="outer").sort_values("MONTH_START")
    m24.insert(1, "IN_FY24", m24["MONTH_START"].isin(mt["MONTH_START"]))
    m24 = add_calendar(m24.reset_index(drop=True))
    m24.to_csv(out("rockingham_2023_2024_monthly_merged.csv"), index=False)
    print(f"    all foot-traffic months: {len(m24)} months "
          f"({m24['MONTH_START'].min():%Y-%m} to {m24['MONTH_START'].max():%Y-%m}), "
          f"{int(m24['IN_FY24'].sum())} with trip data")

    # C
    if args.no_census:
        print("[C] skipped (--no-census)")
    else:
        try:
            import geopandas  # noqa: F401
        except ImportError:
            print("[C] skipped: needs geopandas (in Jupyter: %pip install geopandas, restart, re-run)")
        else:
            e, bgm, bgs, trs = census_linkage(t, ft)
            e.to_csv(out("rockingham_fy24_trips_enriched.csv"), index=False)
            bgm = add_calendar(bgm)
            bgm.to_csv(out("rockingham_2023_2024_bg_month_merged.csv"), index=False)
            bgm[bgm["IN_FY24"]].to_csv(out("rockingham_fy24_bg_month_merged.csv"), index=False)
            bgs.to_csv(out("rockingham_bg_summary.csv"), index=False)
            trs.to_csv(out("rockingham_tract_summary.csv"), index=False)
            ok = e["ORIG_COORD_OK"]
            print(f"[C] trips with pickup in a Rockingham block group: {e['ORIG_IN_COUNTY'].sum():,} "
                  f"of {ok.sum():,} with valid coordinates")
            print(f"    trips whose pickup block group has foot traffic that month: "
                  f"{e['FT_AVAILABLE'].sum():,} ({100 * e['FT_AVAILABLE'].mean():.1f}%)")
            print(f"    block group x month: {len(bgm):,} rows; {bgm['FT_AVAILABLE'].sum()} with foot traffic, "
                  f"{(bgm['FT_AVAILABLE'] & (bgm['TRIPS_FROM'] > 0)).sum()} with both")
            print(f"    tracts: {len(trs)}; no-vehicle share {trs['NO_VEHICLE_SHARE'].min():.1%} "
                  f"to {trs['NO_VEHICLE_SHARE'].max():.1%}")

    print(f"\nOutputs in: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
