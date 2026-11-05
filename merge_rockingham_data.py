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
    C. Optional, needs a census block group shapefile (--bg-shapefile): place each trip's
       origin and destination in a block group and join to foot traffic on block group +
       month. Download for North Carolina:
       https://www2.census.gov/geo/tiger/TIGER2023/BG/tl_2023_37_bg.zip
       (requires: pip install geopandas)

Outputs (written to --out-dir)
    rockingham_foot_traffic_2023_2024.csv   step A
    rockingham_fy24_monthly_merged.csv      step B, months in the FY24 window
    rockingham_fy24_trips_clean.csv         FY24 trips with clearly named coords + flags
    rockingham_fy24_bg_month_merged.csv     step C (only with --bg-shapefile)

Usage
    python merge_rockingham_data.py \
        --foot-2024 rockingham_county_data.csv \
        --foot-2023 Rockingham_Foot_Traffic_2023.csv \
        --trips FY24_Geocode_FINAL_Rockingham_for_NCAT.xlsx \
        --out-dir merged/ [--bg-shapefile tl_2023_37_bg.zip]

Requirements
    pip install pandas openpyxl
"""

import argparse
import json
import os
import re

import numpy as np
import pandas as pd

# Rough North Carolina bounding box, used to flag bad coordinates
# (the FY24 file has a few at lon -99, lon 0, lat 42).
NC_LAT = (33.8, 36.6)
NC_LON = (-84.4, -75.4)


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
# C. Optional block-group join
# ------------------------------------------------------------
def bg_month_merge(t, ft, shapefile):
    import geopandas as gpd

    bg = gpd.read_file(shapefile)[["GEOID", "geometry"]].to_crs("EPSG:4326")
    out = []
    for end in ("ORIG", "DEST"):
        pts = t[t[f"{end}_COORD_OK"]]
        g = gpd.GeoDataFrame(pts[["MONTH_START"]],
                             geometry=gpd.points_from_xy(pts[f"{end}_LON"], pts[f"{end}_LAT"]),
                             crs="EPSG:4326")
        j = gpd.sjoin(g, bg, how="inner", predicate="within")
        out.append(j.groupby(["GEOID", "MONTH_START"]).size().rename(f"TRIPS_{end}"))
    trips_bg = pd.concat(out, axis=1).fillna(0).astype(int).reset_index()
    trips_bg = trips_bg.rename(columns={"GEOID": "AREA"})

    keep = ["AREA", "MONTH_START", "RAW_DEVICE_COUNTS", "RAW_STOP_COUNTS",
            "DISTANCE_FROM_HOME", "MEDIAN_DWELL", "ANDROID_DEVICES", "IOS_DEVICES"]
    m = ft[keep].merge(trips_bg, on=["AREA", "MONTH_START"], how="outer", indicator="MATCH")
    m["MATCH"] = m["MATCH"].map({"both": "both", "left_only": "foot_traffic_only",
                                 "right_only": "trips_only"})
    window = (m["MONTH_START"] >= t["MONTH_START"].min()) & (m["MONTH_START"] <= t["MONTH_START"].max())
    m = m[window]
    m[["TRIPS_ORIG", "TRIPS_DEST"]] = m[["TRIPS_ORIG", "TRIPS_DEST"]].fillna(0).astype(int)
    return m.sort_values(["MONTH_START", "AREA"]).reset_index(drop=True)


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
    ap.add_argument("--foot-2024", default="rockingham_county_data.csv")
    ap.add_argument("--foot-2023", default="Rockingham_Foot_Traffic_2023.csv")
    ap.add_argument("--trips", default="FY24_Geocode_FINAL_Rockingham_for_NCAT.xlsx")
    ap.add_argument("--out-dir", default="merged")
    ap.add_argument("--bg-shapefile", help="census block group shapefile (.zip or .shp)")
    args, _ = ap.parse_known_args(argv)
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
    m = monthly_foot_traffic(ft).merge(monthly_trips(t), on="MONTH_START", how="inner")
    m = add_calendar(m)
    m.to_csv(out("rockingham_fy24_monthly_merged.csv"), index=False)
    print(f"[B] monthly merge: {len(m)} months "
          f"({m['MONTH_START'].min():%Y-%m} to {m['MONTH_START'].max():%Y-%m})")

    # C
    if args.bg_shapefile:
        bgm = add_calendar(bg_month_merge(t, ft, args.bg_shapefile))
        bgm.to_csv(out("rockingham_fy24_bg_month_merged.csv"), index=False)
        print(f"[C] block-group/month merge: {len(bgm)} rows; "
              + ", ".join(f"{k}={v}" for k, v in bgm["MATCH"].value_counts().items()))
    else:
        print("[C] skipped (pass --bg-shapefile to join on block group + month)")

    print(f"\nOutputs in: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
