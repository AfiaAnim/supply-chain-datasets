# ===== NC: adjusted association between metropolitan status and fatal crash outcome =====
# Logistic regression of fatal (any death in the crash report) on metro status, adjusting for vehicle,
# roadway and environmental characteristics and year fixed effects; standard errors clustered by county.
# Run AFTER cells 1 and 1b of crash_fatal_models_v2.py (needs `crash` with COUNTY_FIPS / RUCC_2023 / METRO).
# Needs: %pip install statsmodels
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.stats import chi2_contingency

YEARS = (2020, 2025)          # study period, both ends included (full calendar years only)
MIN_LEVEL_N = 200             # category levels with fewer crashes (or no fatal crashes) are pooled into "Other"

VEHICLE = ["TRUCK_BUS_IND", "CARGO_BODY_TYPE_ID", "GVW_RATING_ID", "VEHICLE_CONFIGURATION_ID",
           "VEHICLE_HAZMAT_PLACARD"]
ROADWAY = ["TRAFFICWAY_ID", "ACCESS_CONTROL_ID", "ROAD_SURFACE_CONDITION_ID"]
ENVIRONMENT = ["WEATHER_CONDITION_ID", "LIGHT_CONDITION_ID"]
COVARIATES = VEHICLE + ROADWAY + ENVIRONMENT

assert "METRO" in crash.columns, "Run cell 1b (RUCC merge) first so crash has METRO"

# --- 1. NC crash reports, one row per report (same cleaning as the model and summaries) ---
keep = ["CRASH_ID", "REPORT_STATE", "REPORT_NUMBER", "REPORT_SEQ_NO", "REPORT_DATE", "CHANGE_DATE",
        "FEDERAL_RECORDABLE", "FATALITIES", "COUNTY_FIPS", "RUCC_2023", "METRO"] + COVARIATES
ncr = crash.loc[crash["STATE"].astype("string").str.strip().str.upper() == "NC", keep].copy()
ncr["_changed"] = pd.to_datetime(ncr["CHANGE_DATE"], errors="coerce")
ncr = ncr.sort_values("_changed").drop_duplicates("CRASH_ID", keep="last")
ncr = ncr[ncr["FEDERAL_RECORDABLE"].astype(str).str.upper() == "Y"]
ncr["report_id"] = ncr["REPORT_STATE"].astype(str) + "_" + ncr["REPORT_NUMBER"].astype(str)
ncr["fatal"] = (ncr.groupby("report_id")["FATALITIES"].transform("max") > 0).astype(int)
ncr = ncr.sort_values(["report_id", "REPORT_SEQ_NO"]).drop_duplicates("report_id", keep="first")
ncr["year"] = pd.to_datetime(pd.to_numeric(ncr["REPORT_DATE"], errors="coerce").astype("Int64").astype(str),
                             format="%Y%m%d", errors="coerce").dt.year
ncr = ncr[ncr["year"].between(*YEARS)]

# Analysis sample: metro status and county known (needed for the exposure and the clusters)
n_all = len(ncr)
dat = ncr[ncr["METRO"].isin(["Metro", "Nonmetro"]) & ncr["COUNTY_FIPS"].notna()].copy()
print(f"NC crash reports {YEARS[0]}-{YEARS[1]}: {n_all:,} | analysed: {len(dat):,} "
      f"(excluded {n_all - len(dat):,} without a matched county) | counties (clusters): {dat['COUNTY_FIPS'].nunique()}")


# --- 2. Covariates as categories: codes are labels, "Missing" is its own level, rare levels pooled ---
def as_level(v):
    if pd.isna(v):
        return "Missing"
    return str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)


for col in COVARIATES:
    x = dat[col].map(as_level)
    stats = dat.groupby(x)["fatal"].agg(["size", "sum"])
    rare = stats.index[(stats["size"] < MIN_LEVEL_N) | (stats["sum"] == 0)]
    dat[col] = x.where(~x.isin(rare), "Other")
dat["year"] = dat["year"].astype(int)


def ref_level(col):   # most common real level (never "Missing" or "Other") is the reference
    counts = dat[col].value_counts()
    real = counts.drop(["Missing", "Other"], errors="ignore")
    return (real if len(real) else counts).index[0]


def term(col):
    return f"C({col}, Treatment(reference='{ref_level(col)}'))"


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


metro_term = "C(METRO, Treatment(reference='Metro'))"
metro_coef = f"{metro_term}[T.Nonmetro]"
models = {
    "1. Unadjusted": [],
    "2. + year": ["C(year)"],
    "3. + vehicle": ["C(year)"] + [term(c) for c in VEHICLE],
    "4. + roadway": ["C(year)"] + [term(c) for c in VEHICLE + ROADWAY],
    "5. + environment (full)": ["C(year)"] + [term(c) for c in COVARIATES],
}

groups = pd.factorize(dat["COUNTY_FIPS"])[0]
fits, rows = {}, []
for name, terms in models.items():
    formula = "fatal ~ " + " + ".join([metro_term] + terms)
    fit = smf.glm(formula, data=dat, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": groups})
    fits[name] = fit
    b, se = fit.params[metro_coef], fit.bse[metro_coef]
    rows.append({"Model": name, "n": int(fit.nobs), "Parameters": len(fit.params),
                 "OR nonmetro vs metro": np.exp(b),
                 "95% CI": f"{np.exp(b - 1.96 * se):.2f}-{np.exp(b + 1.96 * se):.2f}",
                 "p": fmt_p(fit.pvalues[metro_coef])})
or_table = pd.DataFrame(rows).set_index("Model")

# --- 3. Unadjusted proportions and proportion (risk) ratio, same sample ---
props = dat.groupby("METRO")["fatal"].agg(crashes="size", fatal_crashes="sum")
props["fatal_pct"] = 100 * props["fatal_crashes"] / props["crashes"]
r_m, r_n = props.loc["Metro", "fatal_crashes"] / props.loc["Metro", "crashes"], \
    props.loc["Nonmetro", "fatal_crashes"] / props.loc["Nonmetro", "crashes"]
rr = r_n / r_m
se_rr = np.sqrt((1 - r_n) / props.loc["Nonmetro", "fatal_crashes"] + (1 - r_m) / props.loc["Metro", "fatal_crashes"])
chi2, p_chi, _, _ = chi2_contingency(pd.crosstab(dat["METRO"], dat["fatal"]))

print("\nA. Unadjusted fatal proportions (same analysis sample)")
print(props.round(2).to_string())
print(f"Unadjusted proportion ratio (nonmetro / metro): {rr:.2f} "
      f"(95% CI {rr * np.exp(-1.96 * se_rr):.2f}-{rr * np.exp(1.96 * se_rr):.2f}; "
      f"chi-square {chi2:.1f}, p = {p_chi:.3g}). Assumes independent crashes, not clustered.")

print("\nB. Odds ratios for nonmetro vs metro, logistic regression, county-clustered 95% CIs")
display(or_table.round(4))

full = fits["5. + environment (full)"]
b, se = full.params[metro_coef], full.bse[metro_coef]
print(f"Adjusted OR (model 5): {np.exp(b):.2f} (95% CI {np.exp(b - 1.96 * se):.2f}-{np.exp(b + 1.96 * se):.2f}), "
      f"p {'< 0.001' if full.pvalues[metro_coef] < 0.001 else f'= {full.pvalues[metro_coef]:.3f}'}")
print("The OR compares odds, not proportions. With a fatal share of a few percent the two are close, "
      "but report them as different measures.")

# --- 4. Full model: every adjusted OR, for the appendix ---
ci = full.conf_int()
coefs = pd.DataFrame({"OR": np.exp(full.params), "CI low": np.exp(ci[0]), "CI high": np.exp(ci[1]),
                      "p": full.pvalues}).drop(index="Intercept")
coefs.index = (coefs.index.str.replace(r"C\((\w+)(?:, Treatment\(reference='([^']*)'\))?\)\[T\.(.+)\]",
                                       lambda m: f"{m.group(1)}: {m.group(3)} vs {m.group(2) or 'first year'}",
                                       regex=True))
display(coefs.round(3))
coefs.round(4).to_csv("NC_adjusted_logit_full_model.csv")
or_table.round(4).to_csv("NC_metro_odds_ratios.csv")
print("Saved: NC_metro_odds_ratios.csv, NC_adjusted_logit_full_model.csv")
print("Reference levels (most common real level): " + ", ".join(f"{c}={ref_level(c)}" for c in COVARIATES))
