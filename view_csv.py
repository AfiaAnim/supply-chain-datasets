# ===== View a CSV file: shape, columns/dtypes, missing values, preview rows and numeric summary =====
# Command line:  python view_csv.py data.csv [--rows 20] [--tail] [--cols A,B,C] [--sep ";"] [--nrows N] [--all]
# Notebook:      from view_csv import view_csv;  df = view_csv("data.csv", rows=20)
import argparse
import csv
import sys

import pandas as pd


def sniff_sep(path, encoding):
    """Guess the delimiter from the first 64 KB so the fast C parser can read the rest."""
    with open(path, encoding=encoding, errors="replace", newline="") as f:
        sample = f.read(65536)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def load_csv(path, sep=None, encoding=None, nrows=None):
    """Read a CSV (optionally only the first nrows), sniffing the delimiter and falling back to latin-1."""
    enc = encoding or "utf-8-sig"
    sep = sep or sniff_sep(path, enc)
    try:
        return pd.read_csv(path, sep=sep, encoding=enc, nrows=nrows, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, sep=sep, encoding="latin-1", nrows=nrows, low_memory=False)


def view_csv(path, rows=10, tail=False, cols=None, sep=None, encoding=None, show_all=False, nrows=None):
    df = load_csv(path, sep=sep, encoding=encoding, nrows=nrows)
    if cols:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"Columns not found: {missing}. Available: {list(df.columns)}")
        df = df[cols]

    with pd.option_context("display.max_columns", None, "display.width", 200,
                           "display.max_colwidth", 40,
                           "display.max_rows", None if show_all else rows):
        print(f"File: {path}" + (f"  (first {nrows:,} rows only)" if nrows else ""))
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
    p.add_argument("--nrows", type=int, help="read only the first N rows (useful for very large files)")
    p.add_argument("--all", dest="show_all", action="store_true", help="print every row")
    a = p.parse_args(argv)
    try:
        view_csv(a.path, rows=a.rows, tail=a.tail,
                 cols=[c.strip() for c in a.cols.split(",")] if a.cols else None,
                 sep=a.sep, encoding=a.encoding, show_all=a.show_all, nrows=a.nrows)
    except (FileNotFoundError, KeyError, pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        sys.exit(f"Error: {e}")


if __name__ == "__main__":
    main()
