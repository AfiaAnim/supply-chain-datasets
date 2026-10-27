# ===== View a CSV file: shape, columns/dtypes, missing values, preview rows and numeric summary =====
# Command line:  python view_csv.py data.csv [--rows 20] [--tail] [--cols A,B,C] [--sep ";"] [--all]
# Notebook:      from view_csv import view_csv;  df = view_csv("data.csv", rows=20)
import argparse
import sys

import pandas as pd


def load_csv(path, sep=None, encoding=None):
    """Read a CSV, sniffing the delimiter when none is given and falling back to latin-1 on bad bytes."""
    kw = {"sep": sep} if sep else {"sep": None, "engine": "python"}   # sep=None -> pandas sniffs , ; \t |
    try:
        return pd.read_csv(path, encoding=encoding or "utf-8-sig", **kw)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1", **kw)


def view_csv(path, rows=10, tail=False, cols=None, sep=None, encoding=None, show_all=False):
    df = load_csv(path, sep=sep, encoding=encoding)
    if cols:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"Columns not found: {missing}. Available: {list(df.columns)}")
        df = df[cols]

    with pd.option_context("display.max_columns", None, "display.width", 200,
                           "display.max_colwidth", 40,
                           "display.max_rows", None if show_all else rows):
        print(f"File: {path}")
        print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns\n")

        info = pd.DataFrame({"dtype": df.dtypes.astype(str),
                             "non-null": df.notna().sum(),
                             "missing": df.isna().sum(),
                             "unique": df.nunique()})
        print("Columns:")
        print(info.to_string(), "\n")

        if show_all:
            print("All rows:")
            print(df.to_string())
        else:
            print(f"{'Last' if tail else 'First'} {min(rows, len(df))} rows:")
            print((df.tail(rows) if tail else df.head(rows)).to_string())

        num = df.select_dtypes("number")
        if not num.empty:
            print("\nNumeric summary:")
            print(num.describe().T.round(4).to_string())
    return df


def main(argv=None):
    p = argparse.ArgumentParser(description="View a CSV file in the terminal.")
    p.add_argument("path", help="path to the CSV file")
    p.add_argument("-n", "--rows", type=int, default=10, help="number of rows to preview (default 10)")
    p.add_argument("--tail", action="store_true", help="preview the last rows instead of the first")
    p.add_argument("--cols", help="comma-separated list of columns to show")
    p.add_argument("--sep", help="delimiter (default: auto-detect)")
    p.add_argument("--encoding", help="file encoding (default: utf-8, falls back to latin-1)")
    p.add_argument("--all", dest="show_all", action="store_true", help="print every row")
    a = p.parse_args(argv)
    try:
        view_csv(a.path, rows=a.rows, tail=a.tail,
                 cols=[c.strip() for c in a.cols.split(",")] if a.cols else None,
                 sep=a.sep, encoding=a.encoding, show_all=a.show_all)
    except (FileNotFoundError, KeyError, pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        sys.exit(f"Error: {e}")


if __name__ == "__main__":
    main()
