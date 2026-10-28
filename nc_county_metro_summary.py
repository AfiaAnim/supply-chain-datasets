# ===== NC crash counts and fatal-outcome proportions by county and metropolitan status =====
# Run AFTER cells 1 and 1b of crash_fatal_models_v2.py (uses `crash` with COUNTY_FIPS / RUCC_2023 / METRO,
# `rucc`, `parse_report_date`, FEDERAL_RECORDABLE_ONLY and DROP_TRANSACTION_CODES).
# Unit = one crash report (a crash with two trucks counts once); fatal = at least one fatality.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

MIN_CRASHES = 100      # counties with fewer crashes are flagged: their proportions are unstable
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
COLORS = {"Metro": "#2a78d6", "Nonmetro": "#eb6834", "Missing": "#898781"}


def wilson(k, n, z=1.96):
    """95% Wilson interval for a proportion (behaves well for small n and rare events)."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = np.where(n > 0, k / np.maximum(n, 1), np.nan)
    den = 1 + z ** 2 / n
    mid = (p + z ** 2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return mid - half, mid + half


def summarise(df, by):
    g = df.groupby(by, dropna=False).agg(crashes=("FATAL", "size"), fatal_crashes=("FATAL", "sum"),
                                         fatalities=("FATALITIES", "sum"))
    g["fatal_%"] = 100 * g["fatal_crashes"] / g["crashes"]
    lo, hi = wilson(g["fatal_crashes"], g["crashes"])
    g["95% CI"] = [f"{100 * a:.2f}-{100 * b:.2f}" for a, b in zip(lo, hi)]
    g["ci_lo"], g["ci_hi"] = 100 * lo, 100 * hi
    g["share_of_crashes_%"] = 100 * g["crashes"] / g["crashes"].sum()
    return g


# --- NC crash reports: latest version of each record, same filters as the model, one row per report ---
nc = crash[crash["STATE"].astype("string").str.strip().str.upper() == "NC"].copy()
nc["_changed"] = pd.to_datetime(nc["CHANGE_DATE"], errors="coerce")
nc = nc.sort_values("_changed").drop_duplicates("CRASH_ID", keep="last")
if DROP_TRANSACTION_CODES:
    nc = nc[~nc["TRANSACTION_CODE"].isin(DROP_TRANSACTION_CODES)]
if FEDERAL_RECORDABLE_ONLY:
    nc = nc[nc["FEDERAL_RECORDABLE"].astype(str).str.upper() == "Y"]
nc["GROUP_ID"] = nc["REPORT_STATE"].astype(str) + "_" + nc["REPORT_NUMBER"].astype(str)
nc["FATALITIES"] = nc.groupby("GROUP_ID")["FATALITIES"].transform("max")      # crash-level outcome
nc = nc.sort_values(["GROUP_ID", "REPORT_SEQ_NO"]).drop_duplicates("GROUP_ID", keep="first")
nc["FATAL"] = (nc["FATALITIES"] > 0).astype(int)
years = parse_report_date(nc["REPORT_DATE"]).dt.year
n_years = years.nunique()
print(f"NC crash reports: {len(nc):,} | fatal: {nc['FATAL'].sum():,} ({nc['FATAL'].mean():.2%}) | "
      f"years {int(years.min())}-{int(years.max())} ({n_years} years)")
print(f"Without a matched county: {nc['RUCC_2023'].isna().sum():,} ({nc['RUCC_2023'].isna().mean():.2%})")

# --- 1. Metro vs nonmetro ---
metro = summarise(nc, "METRO")
display(metro.drop(columns=["ci_lo", "ci_hi"]).round(2))
known = nc[nc["METRO"].isin(["Metro", "Nonmetro"])]
table = pd.crosstab(known["METRO"], known["FATAL"])
chi2, p_val, _, _ = chi2_contingency(table)
r_m, r_n = known.groupby("METRO")["FATAL"].mean()[["Metro", "Nonmetro"]]
n_m, n_n = table.sum(axis=1)[["Metro", "Nonmetro"]]
se = np.sqrt((1 - r_n) / (r_n * n_n) + (1 - r_m) / (r_m * n_m))
rr = r_n / r_m
print(f"Nonmetro vs metro risk ratio: {rr:.2f} (95% CI {rr * np.exp(-1.96 * se):.2f}-{rr * np.exp(1.96 * se):.2f}) "
      f"| chi-square = {chi2:.1f}, p = {p_val:.3g}")

# --- 2. RUCC 2023 code (1-3 metro, 4-9 nonmetro) ---
rucc_tab = summarise(nc, "RUCC_2023")
display(rucc_tab.drop(columns=["ci_lo", "ci_hi"]).round(2))

# --- 3. County ---
nc_counties = rucc[rucc["State"] == "NC"].set_index("FIPS")[["County_Name", "RUCC_2023", "Population_2020"]]
county = summarise(nc, "COUNTY_FIPS").join(nc_counties, how="outer")
county[["crashes", "fatal_crashes", "fatalities"]] = county[["crashes", "fatal_crashes", "fatalities"]].fillna(0)
county["County_Name"] = county["County_Name"].fillna(
    pd.Series(county.index, index=county.index).map(lambda f: "(county missing)" if pd.isna(f) else f"(unmatched {f})"))
county["is_county"] = county.index.isin(nc_counties.index)
county["METRO"] = county["RUCC_2023"].map(lambda r: "Missing" if pd.isna(r) else "Metro" if r <= 3 else "Nonmetro")
county["crashes_per_100k_per_year"] = 1e5 * county["crashes"] / pd.to_numeric(county["Population_2020"]) / n_years
county["small_n"] = county["crashes"] < MIN_CRASHES
cols = ["County_Name", "METRO", "RUCC_2023", "crashes", "fatal_crashes", "fatal_%", "95% CI", "fatalities",
        "crashes_per_100k_per_year", "small_n"]
county = county.sort_values("crashes", ascending=False)
print(f"\nNC counties: {int(county['is_county'].sum())} | with fewer than {MIN_CRASHES} crashes: "
      f"{int((county['small_n'] & county['is_county']).sum())} | crashes without a matched county: "
      f"{int(county.loc[~county['is_county'], 'crashes'].sum()):,}")
display(county[cols].head(15).round(2))
stable = county[~county["small_n"] & county["is_county"]]
print(f"Highest fatal proportions (counties with {MIN_CRASHES}+ crashes):")
display(stable.sort_values("fatal_%", ascending=False)[cols].head(10).round(2))
county[cols].round(4).to_csv("NC_crashes_by_county.csv")
metro.drop(columns=["ci_lo", "ci_hi"]).round(4).to_csv("NC_crashes_by_metro.csv")
print("Saved: NC_crashes_by_county.csv, NC_crashes_by_metro.csv")


# --- Charts: fatal proportion with 95% CI, one shared scale, metro status by colour ---
def style(ax):
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors=INK2, length=0)


nc_rate = 100 * nc["FATAL"].mean()
r = rucc_tab[rucc_tab.index.notna()].sort_index()
fig, ax = plt.subplots(figsize=(8, 4.2))
for code, row in r.iterrows():
    c = COLORS["Metro" if code <= 3 else "Nonmetro"]
    ax.plot([row["ci_lo"], row["ci_hi"]], [code, code], color=c, lw=2, solid_capstyle="round")
    ax.plot(row["fatal_%"], code, "o", ms=8, color=c, mec="#fcfcfb", mew=2)
    ax.text(row["ci_hi"] + 0.08, code, f"{row['fatal_%']:.1f}%  (n={int(row['crashes']):,})",
            va="center", fontsize=8, color=INK2)
ax.axvline(nc_rate, color=MUTED, ls="--", lw=1)
ax.text(nc_rate, 9.55, f" NC overall {nc_rate:.1f}%", color=MUTED, fontsize=8, va="bottom")
ax.set_xlim(right=r["ci_hi"].max() + 0.25 * (r["ci_hi"].max() - r["ci_lo"].min()))
ax.set_yticks(r.index, [f"RUCC {int(k)}" for k in r.index])
ax.set_ylim(9.6, 0.2)
ax.set_xlabel("Fatal crashes (% of crash reports), 95% CI", color=INK2)
ax.set_title("NC: fatal-outcome proportion by rural-urban code", loc="left", color=INK, fontweight="bold")
ax.plot([], [], "o", color=COLORS["Metro"], label="Metro (RUCC 1-3)")
ax.plot([], [], "o", color=COLORS["Nonmetro"], label="Nonmetro (RUCC 4-9)")
ax.legend(frameon=False, loc="lower right", fontsize=8)
style(ax)
plt.tight_layout()
plt.show()

cty = county[county["is_county"] & (county["crashes"] > 0)].sort_values("fatal_%")
fig, ax = plt.subplots(figsize=(8, 0.16 * len(cty) + 1.2))
y = np.arange(len(cty))
for status, c in COLORS.items():
    m = (cty["METRO"] == status).values
    if not m.any():
        continue
    ax.hlines(y[m], cty["ci_lo"][m], cty["ci_hi"][m], color=c, lw=1.2, alpha=0.45)
    sm = m & cty["small_n"].values
    ax.plot(cty["fatal_%"][m & ~sm], y[m & ~sm], "o", ms=5, color=c, label=status)
    ax.plot(cty["fatal_%"][sm], y[sm], "o", ms=5, mfc="#fcfcfb", mec=c, mew=1.2)   # hollow = small n
ax.axvline(nc_rate, color=MUTED, ls="--", lw=1)
ax.set_yticks(y, cty["County_Name"].str.replace(" County", ""), fontsize=6.5)
ax.set_ylim(-1, len(cty))
ax.set_xlabel("Fatal crashes (% of crash reports), 95% CI", color=INK2)
ax.set_title(f"NC counties: fatal-outcome proportion (hollow = fewer than {MIN_CRASHES} crashes)",
             loc="left", color=INK, fontweight="bold", fontsize=10)
ax.legend(frameon=False, loc="lower right", fontsize=8)
style(ax)
plt.tight_layout()
plt.show()
