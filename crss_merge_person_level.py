"""
crss_merge_person_level.py

Merge every table of an NHTSA CRSS year (CSV release) into ONE person-level
analysis file: one row per person in person.csv.

How each table is attached
--------------------------
Level is read from the keys a table carries:
    crash   : CASENUM                    -> joined on CASENUM
    vehicle : CASENUM, VEH_NO            -> joined on CASENUM + LINK_VEH
    person  : CASENUM, VEH_NO, PER_NO    -> joined on CASENUM + VEH_NO + PER_NO

  * accident.csv            crash level, 1 row per crash. Columns already copied
                            into person.csv are dropped; the rest keep their names
                            (WEATHER, LGT_COND, ...).
  * vehicle.csv             prefixed "veh_".
  * any other 1-row-per-key table (vpicdecode, safetyeq, pbtype, ...) is joined
    directly, columns prefixed with the file name ("vpicdecode_...").
  * any many-rows-per-key table (weather, crashrf, cevent, vevent, vsoe, damage,
    distract, drimpair, driverrf, factor, maneuver, violatn, vision, vehiclesf,
    personrf, nm*, vpictrailerdecode, ...) is collapsed to one row per key:
        <file>_n              number of rows
        <file>_<COL>NAME      the labels joined with " | " (event tables keep
                              their EVENTNUM/VEVENTNUM order, " > " separated)
        <file>__<label>       0/1 flag per category, for tables with a single
                              coded field (e.g. driverrf__drowsy_..., damage__12_clock)
  * parkwork.csv / pvehiclesf.csv describe parked or working vehicles that hold no
    person in person.csv, so they are summarised per crash.

Non-motorists (pedestrians, cyclists) have VEH_NO = 0. Their vehicle-level
columns come from the striking vehicle (person.STR_VEH); VEH_LINK_SOURCE says
which vehicle ("own", "striking" or "none") the veh_* columns describe.

Survey-design columns (PSU, PSU_VAR, PSUSTRAT, REGION, URBANICITY, STRATUM, PJ,
WEIGHT) are kept once, from person.csv. Use WEIGHT for national estimates.

Outcome helpers added from INJ_SEV (KABCO):
    INJ_GROUP      K_fatal, A_serious, B_minor, C_possible, O_none, other/unknown
    SERIOUS_FATAL  1 = K or A, 0 = B or C, blank otherwise (uninjured, unknown)

Usage
-----
    python crss_merge_person_level.py --data-dir "C:/CRSS2024" --out crss2024_person.csv
    python crss_merge_person_level.py --data-dir ./CRSS2024 --out crss2024_person.csv.gz --labels names

    --labels both|codes|names   keep coded and/or text-label columns (default both)
    --no-flags                  skip the 0/1 category flags
    --skip a b ...              file stems to leave out (e.g. --skip vsoe vevent)

Aggregated columns are blank (not 0) where the table has no row for that key,
e.g. driver-factor tables for a passenger or for a non-motorist with no striking
vehicle. Post-crash fields (EMS times, hospital, towing) are merged like any
other column; drop them before modelling to avoid outcome leakage.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CRASH_KEY = ["CASENUM"]
VEH_KEY = ["CASENUM", "VEH_NO"]
PER_KEY = ["CASENUM", "VEH_NO", "PER_NO"]

DESIGN_COLS = ["PSU", "PSU_VAR", "PSUSTRAT", "REGION", "REGIONNAME", "URBANICITY",
               "URBANICITYNAME", "STRATUM", "STRATUMNAME", "PJ", "WEIGHT"]
SEQ_COLS = ["EVENTNUM", "VEVENTNUM"]
# parked/working vehicles: VEH_NO is not an in-transport vehicle, summarise per crash
CRASH_LEVEL_OVERRIDE = {"parkwork", "pvehiclesf"}

INJ_GROUPS = {4: "K_fatal", 3: "A_serious", 2: "B_minor", 1: "C_possible", 0: "O_none"}


def read_table(path):
    for enc in ("utf-8", "cp1252", "latin1"):
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False)
            break
        except UnicodeDecodeError:
            continue
    df.columns = [c.strip().upper() for c in df.columns]
    return df


def slug(text, maxlen=40):
    s = re.sub(r"[^0-9a-z]+", "_", str(text).lower()).strip("_")
    return s[:maxlen] or "blank"


def level_of(stem, df):
    if stem in CRASH_LEVEL_OVERRIDE or "VEH_NO" not in df.columns:
        return "crash"
    if "PER_NO" in df.columns:
        return "person"
    return "vehicle"


def join_labels(values, ordered):
    vals = [str(v) for v in values if pd.notna(v)]
    if ordered:
        return " > ".join(vals)
    return " | ".join(sorted(set(vals)))


def collapse(df, key, stem, make_flags):
    """Many rows per key -> one row per key (count, joined labels, 0/1 flags)."""
    seq = [c for c in SEQ_COLS if c in df.columns]
    if seq:
        df = df.sort_values(key + seq)
    value_cols = [c for c in df.columns if c not in key + seq]
    name_cols = [c for c in value_cols if c.endswith("NAME")]
    text_cols = name_cols or value_cols

    g = df.groupby(key, sort=False)
    out = g.size().rename(f"{stem}_n").to_frame()
    for c in text_cols:
        out[f"{stem}_{c}"] = g[c].agg(lambda s: join_labels(s, ordered=bool(seq)))

    if make_flags and len(name_cols) == 1:
        lab = df[name_cols[0]].fillna("missing").astype(str)
        flags = pd.crosstab([df[k] for k in key], lab).clip(upper=1)
        names, seen = [], set()
        for c in flags.columns:
            n = f"{stem}__{slug(c)}"
            while n in seen:
                n += "_"
            seen.add(n)
            names.append(n)
        flags.columns = names
        out = out.join(flags)
    return out.reset_index()


def prefixed(df, key, prefix):
    return df.rename(columns={c: f"{prefix}_{c}" for c in df.columns if c not in key})


def merge_checked(base, other, left_on, right_on, name):
    n = len(base)
    other = other.rename(columns=dict(zip(right_on, left_on)))
    merged = base.merge(other, how="left", on=left_on,
                        validate="many_to_one", suffixes=("", f"__{name}"))
    assert len(merged) == n, f"{name}: row count changed {n} -> {len(merged)}"
    return merged


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=".", help="folder holding the CRSS CSV files")
    ap.add_argument("--out", default="crss_person_merged.csv", help=".csv, .csv.gz or .parquet")
    ap.add_argument("--labels", choices=["both", "codes", "names"], default="both")
    ap.add_argument("--no-flags", action="store_true")
    ap.add_argument("--skip", nargs="*", default=[], help="file stems to leave out")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    files = {p.stem.lower(): p for p in sorted(data_dir.glob("*.csv"))}
    if "person" not in files:
        sys.exit(f"person.csv not found in {data_dir.resolve()}")
    skip = {s.lower() for s in args.skip}
    make_flags = not args.no_flags

    # ---- spine: one row per person -------------------------------------------
    person = read_table(files["person"])
    if person.duplicated(PER_KEY).any():
        sys.exit("person.csv is not unique on CASENUM, VEH_NO, PER_NO")
    print(f"person      {len(person):>8,} rows  {person.shape[1]:>4} cols  (spine)")

    # vehicle that describes this person: own vehicle, else the striking vehicle
    link = person["VEH_NO"].where(person["VEH_NO"] > 0)
    source = pd.Series(np.where(link.notna(), "own", "none"), index=person.index)
    if "STR_VEH" in person.columns:
        strv = person["STR_VEH"].where((person["VEH_NO"] == 0) & person["STR_VEH"].between(1, 900))
        source[strv.notna()] = "striking"
        link = link.fillna(strv)
    person["LINK_VEH"] = link.astype("Int64")
    person["VEH_LINK_SOURCE"] = source

    inj = pd.to_numeric(person.get("INJ_SEV"), errors="coerce")
    if inj is not None:
        person["INJ_GROUP"] = inj.map(INJ_GROUPS).fillna("other_unknown")
        person["SERIOUS_FATAL"] = np.select([inj.isin([3, 4]), inj.isin([1, 2])], [1, 0], np.nan)
        person["SERIOUS_FATAL"] = person["SERIOUS_FATAL"].astype("Int64")

    merged = person
    report = []

    order = ["accident", "vehicle"] + [s for s in files if s not in ("person", "accident", "vehicle")]
    for stem in order:
        if stem not in files or stem in skip:
            continue
        df = read_table(files[stem])
        df = df.drop(columns=[c for c in DESIGN_COLS if c in df.columns])
        lvl = level_of(stem, df)
        key = {"crash": CRASH_KEY, "vehicle": VEH_KEY, "person": PER_KEY}[lvl]
        if lvl == "crash":
            df = df.drop(columns=[c for c in ("VEH_NO", "PER_NO") if c in df.columns and stem in CRASH_LEVEL_OVERRIDE])
        missing = [k for k in key if k not in df.columns]
        if missing:
            print(f"  ! {stem}: missing key {missing}, skipped")
            continue

        unique = not df.duplicated(key).any()
        if stem == "accident":
            dup = [c for c in df.columns if c in merged.columns and c not in key]
            table = df.drop(columns=dup)
        elif unique:
            table = prefixed(df, key, "veh" if stem == "vehicle" else stem)
        else:
            table = collapse(df, key, stem, make_flags)

        left_on = {"crash": CRASH_KEY, "vehicle": ["CASENUM", "LINK_VEH"], "person": PER_KEY}[lvl]
        if lvl == "vehicle":
            table = table.astype({"VEH_NO": "Int64"})
        merged = merge_checked(merged, table, left_on, key, stem)
        how = "1:1" if unique else "collapsed"
        report.append((stem, lvl, how, len(df), table.shape[1] - len(key)))
        print(f"{stem:<11} {len(df):>8,} rows  {lvl:<7} {how:<9} +{table.shape[1] - len(key)} cols")

    # ---- label handling ------------------------------------------------------
    if args.labels != "both":
        cols = set(merged.columns)
        pairs = [(c[:-4], c) for c in merged.columns if c.endswith("NAME") and c[:-4] in cols]
        drop = [code for code, _ in pairs] if args.labels == "names" else [name for _, name in pairs]
        merged = merged.drop(columns=drop)

    out = Path(args.out)
    if out.suffix == ".parquet":
        merged.to_parquet(out, index=False)
    else:
        merged.to_csv(out, index=False)

    print(f"\nwrote {out}  {len(merged):,} rows x {merged.shape[1]:,} cols")
    print("VEH_LINK_SOURCE:", merged["VEH_LINK_SOURCE"].value_counts().to_dict())
    if "INJ_GROUP" in merged.columns:
        print("INJ_GROUP:", merged["INJ_GROUP"].value_counts().to_dict())
    unused = sorted(set(files) - {"person"} - {r[0] for r in report} - skip)
    if unused:
        print("not merged:", unused)


if __name__ == "__main__":
    main()
