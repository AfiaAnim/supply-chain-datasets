# ===== NC: adjusted association between metropolitan status and fatal crash outcome =====
# 1 category diagnostics -> 2 one vehicle classification (+ weight rating) -> 3 documented category grouping ->
# 4 progressive logistic fits with checks (county-clustered SEs) -> 5 Firth bias-reduced fit with its own
# uncertainty (profile penalized likelihood; county cluster bootstrap).
# Run AFTER cells 1 and 1b of crash_fatal_models_v2.py (needs `crash` with COUNTY_FIPS / RUCC_2023 / METRO).
# Needs: %pip install statsmodels
# Run cell by cell in Jupyter (# %% marks a cell).

# %% A. Settings and analysis sample
import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from scipy.optimize import brentq
from scipy.special import expit
from scipy.stats import chi2, chi2_contingency

YEARS = (2020, 2025)          # study period, both ends included (full calendar years only)
MIN_LEVEL_N = 200             # a level needs at least this many crashes ...
MIN_LEVEL_EVENTS = 5          # ... and this many fatal and nonfatal crashes, or it is grouped (step 3)
VEHICLE = ["VEHICLE_CONFIGURATION_ID", "GVW_RATING_ID"]   # one classification + weight rating (step 2)
OPTIONAL_VEHICLE = []         # e.g. ["VEHICLE_HAZMAT_PLACARD"]; add only if supported by step 4
ROADWAY = ["TRAFFICWAY_ID", "ACCESS_CONTROL_ID", "ROAD_SURFACE_CONDITION_ID"]
ENVIRONMENT = ["WEATHER_CONDITION_ID", "LIGHT_CONDITION_ID"]
COVARIATES = VEHICLE + OPTIONAL_VEHICLE + ROADWAY + ENVIRONMENT

# Step 3: meaningful groupings, filled in from the MCMIS crash file data dictionary (codes as text).
# Codes not listed keep their own level. Decide these from definitions, before looking at model results.
CODE_GROUPS = {
    # "VEHICLE_CONFIGURATION_ID": {"5": "Tractor/semitrailer", "6": "Tractor/double", "7": "Tractor/triple"},
}
UNKNOWN_CODES = {             # codes that mean "unknown" (kept apart from blank/missing)
    # "LIGHT_CONDITION_ID": ["9"],
}
MAX_ABS_COEF = 5              # |log-odds| above this (OR > ~150) suggests separation
MAX_SE = 3                    # standard error above this suggests separation or near-aliasing
N_BOOT = 200                  # county cluster-bootstrap resamples for the Firth estimate
SEED = 42

assert "METRO" in crash.columns, "Run cell 1b (RUCC merge) first so crash has METRO"

# NC crash reports, one row per report (same cleaning as the model and summaries)
all_vehicle = ["TRUCK_BUS_IND", "CARGO_BODY_TYPE_ID", "GVW_RATING_ID", "VEHICLE_CONFIGURATION_ID",
               "VEHICLE_HAZMAT_PLACARD"]
keep = ["CRASH_ID", "REPORT_STATE", "REPORT_NUMBER", "REPORT_SEQ_NO", "REPORT_DATE", "CHANGE_DATE",
        "FEDERAL_RECORDABLE", "FATALITIES", "COUNTY_FIPS", "RUCC_2023", "METRO"] \
       + list(dict.fromkeys(all_vehicle + COVARIATES))
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

n_all = len(ncr)
dat = ncr[ncr["METRO"].isin(["Metro", "Nonmetro"]) & ncr["COUNTY_FIPS"].notna()].copy()
dat["year"] = dat["year"].astype(int)
print(f"NC crash reports {YEARS[0]}-{YEARS[1]}: {n_all:,} | analysed: {len(dat):,} "
      f"(excluded {n_all - len(dat):,} without a matched county) | fatal: {dat['fatal'].sum():,} | "
      f"counties (clusters): {dat['COUNTY_FIPS'].nunique()}")


def as_level(v):
    if pd.isna(v):
        return "Missing"
    return str(int(v)) if isinstance(v, float) and v.is_integer() else str(v).strip()


for col in all_vehicle + COVARIATES:
    dat[col] = dat[col].map(as_level)


# %% B. Step 1: outcome counts in every category
def level_table(col):
    t = dat.groupby(col)["fatal"].agg(crashes="size", fatal="sum")
    t["nonfatal"] = t["crashes"] - t["fatal"]
    t["fatal_%"] = 100 * t["fatal"] / t["crashes"]
    t["flag"] = np.select([t["fatal"] == 0, t["nonfatal"] == 0, t["crashes"] < MIN_LEVEL_N,
                           t["fatal"] < MIN_LEVEL_EVENTS],
                          ["NO FATAL", "NO NONFATAL", "few crashes", "few fatal"], "")
    return t.sort_values("crashes", ascending=False)


for col in ["year", "METRO"] + list(dict.fromkeys(all_vehicle + COVARIATES)):
    t = level_table(col)
    print(f"\n{col}: {len(t)} levels, {int((t['flag'] != '').sum())} flagged")
    print(t.round(2).to_string())


def cramers_v(a, b):
    tab = pd.crosstab(a, b)
    c2 = chi2_contingency(tab, correction=False)[0]
    return np.sqrt(c2 / (tab.values.sum() * (min(tab.shape) - 1)))


print("\nOverlap between vehicle fields (Cramer's V: 0 = unrelated, 1 = same information):")
display(pd.DataFrame([[cramers_v(dat[a], dat[b]) for b in all_vehicle] for a in all_vehicle],
                     index=all_vehicle, columns=all_vehicle).round(2))
miss = dat[all_vehicle].eq("Missing")
print(f"Crashes with ALL vehicle fields missing: {miss.all(axis=1).mean():.1%} | with ANY missing: "
      f"{miss.any(axis=1).mean():.1%}")

# %% C. Steps 2-3: one vehicle classification + weight rating; documented grouping of sparse levels
recode_log = []
# Records with no vehicle detail at all: one shared level ("No vehicle detail") on the first vehicle variable,
# reference level on the others, so they carry one parameter instead of several near-identical "Missing" ones.
veh = VEHICLE + OPTIONAL_VEHICLE
no_detail = dat[veh].eq("Missing").all(axis=1)
if len(veh) > 1 and no_detail.any():
    dat.loc[no_detail, veh[0]] = "No vehicle detail"
    for col in veh[1:]:
        ref = dat.loc[~no_detail, col].value_counts().drop(["Missing", "Unknown"], errors="ignore").index[0]
        dat.loc[no_detail, col] = ref
    recode_log.append((", ".join(veh), f"{no_detail.sum():,} crashes with no vehicle detail -> "
                                       f"'No vehicle detail' on {veh[0]} (one shared parameter)"))


def sparse(t, lvl):
    return t.loc[lvl, "crashes"] < MIN_LEVEL_N or min(t.loc[lvl, "fatal"], t.loc[lvl, "nonfatal"]) < MIN_LEVEL_EVENTS


for col in COVARIATES:
    x = dat[col].replace(CODE_GROUPS.get(col, {}))                     # 1) definitions from the data dictionary
    x = x.where(~x.isin(UNKNOWN_CODES.get(col, [])), "Unknown")          # 2) coded unknown, kept apart from blank
    dat[col] = x
    t = level_table(col)
    known = [lvl for lvl in t.index if lvl not in ("Missing", "Unknown", "No vehicle detail")]
    small = [lvl for lvl in known if sparse(t, lvl)]
    if small:                                                            # 3) sparse known levels -> "Other"
        dat.loc[dat[col].isin(small), col] = "Other"
        recode_log.append((col, f"{', '.join(small)} -> Other"))
    t = level_table(col)
    unk = [lvl for lvl in ("Missing", "Unknown") if lvl in t.index]
    if any(sparse(t, lvl) for lvl in unk) and len(unk) == 2:            # 4) only if needed: Missing + Unknown
        dat.loc[dat[col].isin(unk), col] = "Unknown/missing"
        recode_log.append((col, "Missing + Unknown -> Unknown/missing"))
    t = level_table(col)
    for lvl in ("Missing", "Unknown", "Unknown/missing", "No vehicle detail"):
        if lvl in t.index and sparse(t, lvl):                           # 5) last resort: too few to stand alone
            dat.loc[dat[col] == lvl, col] = "Other"
            recode_log.append((col, f"{lvl} too sparse to keep separate -> Other"))
    t = level_table(col)
    if "Other" in t.index and sparse(t, "Other"):                       # 6) "Other" itself too small -> reference
        ref = t.drop(["Other", "Missing", "Unknown", "Unknown/missing", "No vehicle detail"], errors="ignore").index[0]
        dat.loc[dat[col] == "Other", col] = ref
        recode_log.append((col, f"Other still too sparse ({int(t.loc['Other', 'crashes'])} crashes) -> merged into {ref}"))
    t = level_table(col)
    for lvl in t.index[t["flag"].isin(["NO FATAL", "NO NONFATAL"])]:
        recode_log.append((col, f"STILL SPARSE after grouping: {lvl} ({t.loc[lvl, 'flag']}) - Firth handles this"))

print("Grouping applied (rules depend only on counts and definitions, never on model results):")
for col, msg in recode_log:
    print(f"  {col}: {msg}")
for col in COVARIATES:
    t = level_table(col)
    print(f"\n{col} after grouping:\n{t.round(2).to_string()}")


# %% D. Step 4: progressive refits, each checked
def ref_level(col):   # most common real level (never Missing/Unknown/Other) is the reference
    counts = dat[col].value_counts()
    real = counts.drop(["Missing", "Unknown", "Unknown/missing", "Other", "No vehicle detail"], errors="ignore")
    return (real if len(real) else counts).index[0]


def term(col):
    return f"C({col}, Treatment(reference='{ref_level(col)}'))"


def label(name):   # "C(GVW_RATING_ID, Treatment(reference='3'))[T.Missing]" -> "GVW_RATING_ID: Missing vs 3"
    return pd.Series([name]).str.replace(
        r"C\((\w+)(?:, Treatment\(reference='([^']*)'\))?\)\[T\.(.+)\]",
        lambda m: f"{m.group(1)}: {m.group(3)} vs {m.group(2) or 'first year'}", regex=True)[0]


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def drop_aliased(X):
    """Drop columns that are exact linear combinations of earlier ones; earlier columns (metro) always win.
    Unpivoted QR: |R[i, i]| is what column i adds beyond the columns before it."""
    A = X.to_numpy(float)
    r = np.abs(np.diag(np.linalg.qr(A, mode="r")))
    keep = r > 1e-7 * np.maximum(np.linalg.norm(A, axis=0), 1e-12)
    return X.loc[:, keep], list(X.columns[~keep])


metro_term = "C(METRO, Treatment(reference='Metro'))"
metro_coef = f"{metro_term}[T.Nonmetro]"
steps = {
    "1. Metro + year": [],
    "2. + vehicle": VEHICLE + OPTIONAL_VEHICLE,
    "3. + roadway": VEHICLE + OPTIONAL_VEHICLE + ROADWAY,
    "4. + environment (full)": COVARIATES,
}
groups = pd.factorize(dat["COUNTY_FIPS"])[0]
y = dat["fatal"].to_numpy(float)
fits, designs, rows = {}, {}, []
for name, cols in steps.items():
    _, X = patsy.dmatrices("fatal ~ " + " + ".join([metro_term, "C(year)"] + [term(c) for c in cols]),
                           dat, return_type="dataframe")
    X, dropped = drop_aliased(X)
    fit = sm.GLM(y, X, family=sm.families.Binomial()).fit(
        maxiter=200, cov_type="cluster", cov_kwds={"groups": groups})
    fits[name], designs[name] = fit, X
    b, se = fit.params.drop("Intercept"), fit.bse.drop("Intercept")
    bad = b.index[(b.abs() > MAX_ABS_COEF) | (se > MAX_SE) | ~np.isfinite(b) | ~np.isfinite(se)]
    ok = fit.converged and np.isfinite(fit.params).all() and np.isfinite(fit.bse).all() and len(bad) == 0
    bm, sm_ = fit.params[metro_coef], fit.bse[metro_coef]
    rows.append({"Step": name, "n": int(fit.nobs), "Parameters": len(fit.params), "Aliased dropped": len(dropped),
                 "Converged": fit.converged, "Max |coef|": b.abs().max(), "Max SE": se.max(),
                 "Flagged terms": len(bad), "Usable": ok,
                 "OR nonmetro vs metro": np.exp(bm),
                 "95% CI (county-clustered)": f"{np.exp(bm - 1.96 * sm_):.2f}-{np.exp(bm + 1.96 * sm_):.2f}",
                 "p": fmt_p(fit.pvalues[metro_coef])})
    for c in dropped:
        print(f"{name}: aliased, dropped - {label(c)}")
    for c in bad:
        print(f"{name}: FLAGGED (separation/instability) - {label(c)}: coef {b[c]:.2f}, SE {se[c]:.2f}")
steps_table = pd.DataFrame(rows).set_index("Step")
display(steps_table.round(3))

# Does each variable add information? Joint cluster-robust Wald test per variable (full model)
full = fits["4. + environment (full)"]
V = full.cov_params()
wald = []
for col in ["year"] + COVARIATES:
    idx = [c for c in full.params.index if c.startswith(f"C({col}")]
    if idx:
        bb = full.params[idx].to_numpy()
        stat = float(bb @ np.linalg.pinv(V.loc[idx, idx].to_numpy()) @ bb)
        wald.append({"Variable": col, "df": len(idx), "Wald chi2": stat, "p": fmt_p(chi2.sf(stat, len(idx)))})
print("Joint cluster-robust Wald tests in the full model (is each variable supported?):")
display(pd.DataFrame(wald).set_index("Variable").round(2))


# %% E. Step 5: Firth bias-reduced logistic regression with matching uncertainty
def firth(X, y, fixed=None, beta0=None, maxiter=200, tol=1e-8):
    """Firth (Jeffreys-prior penalized) logistic regression. `fixed` = (index, value) holds one coefficient
    constant (for the profile likelihood). Returns coefficients, penalized log-likelihood, converged flag."""
    k = X.shape[1]
    beta = np.zeros(k) if beta0 is None else beta0.astype(float).copy()
    free = np.ones(k, bool)
    if fixed is not None:
        free[fixed[0]] = False
        beta[fixed[0]] = fixed[1]

    def pll(b):
        p = np.clip(expit(X @ b), 1e-15, 1 - 1e-15)
        w = p * (1 - p)
        sign, logdet = np.linalg.slogdet(X.T @ (X * w[:, None]))
        if sign <= 0:
            return -np.inf, p, w
        return np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)) + 0.5 * logdet, p, w

    cur, p, w = pll(beta)
    for _ in range(maxiter):
        info = X.T @ (X * w[:, None])
        info_inv = np.linalg.pinv(info)
        h = w * np.sum((X @ info_inv) * X, axis=1)                      # leverages
        score = X.T @ (y - p + h * (0.5 - p))                           # Firth-modified score
        step = np.zeros(k)
        step[free] = np.linalg.lstsq(info[np.ix_(free, free)], score[free], rcond=None)[0]
        step = np.clip(step, -5, 5)
        for _ in range(30):                                             # step-halving on penalized likelihood
            new, p_new, w_new = pll(beta + step)
            if not (np.isfinite(new) and np.isfinite(cur)) or new >= cur - 1e-8 * max(1.0, abs(cur)):
                break                                                   # (log-det unreliable -> plain Newton)
            step /= 2
        beta, cur, p, w = beta + step, new, p_new, w_new
        if np.max(np.abs(step)) < tol or np.max(np.abs(score[free])) < tol:
            return beta, cur, True
    return beta, cur, False


Xf = designs["4. + environment (full)"]
cols_f = list(Xf.columns)
j = cols_f.index(metro_coef)
Xn = Xf.to_numpy(float)
beta_f, pll_f, conv_f = firth(Xn, y)
print(f"Firth full model converged: {conv_f} | max |coef| {np.max(np.abs(beta_f[1:])):.2f}")

# Profile penalized-likelihood 95% CI for the metro coefficient (model-based, crashes independent)
target = pll_f - chi2.ppf(0.95, 1) / 2
se_guess = max(full.bse[metro_coef], 0.05)


def profile_gap(bj):
    return firth(Xn, y, fixed=(j, bj), beta0=beta_f)[1] - target


lo_b, hi_b = beta_f[j] - se_guess, beta_f[j] + se_guess
while profile_gap(lo_b) > 0:
    lo_b -= se_guess
while profile_gap(hi_b) > 0:
    hi_b += se_guess
ci_prof = (brentq(profile_gap, lo_b, beta_f[j]), brentq(profile_gap, beta_f[j], hi_b))

# County cluster bootstrap for the Firth estimate (accounts for crashes within a county being related)
rng = np.random.default_rng(SEED)
county_rows = [np.where(groups == g)[0] for g in range(groups.max() + 1)]
boot, failed = [], 0
for _ in range(N_BOOT):
    idx = np.concatenate([county_rows[g] for g in rng.integers(0, len(county_rows), len(county_rows))])
    Xb, _ = drop_aliased(Xf.iloc[idx])                                 # levels absent/aliased in this resample
    if metro_coef not in Xb.columns:
        failed += 1
        continue
    bb, _, ok = firth(Xb.to_numpy(float), y[idx], beta0=pd.Series(beta_f, index=cols_f)[Xb.columns].to_numpy())
    if ok:
        boot.append(bb[list(Xb.columns).index(metro_coef)])
    else:
        failed += 1
ci_boot = np.percentile(boot, [2.5, 97.5])
print(f"Cluster bootstrap: {len(boot)} usable resamples, {failed} failed")

# %% F. Report: unadjusted proportion ratio vs adjusted odds ratios
props = dat.groupby("METRO")["fatal"].agg(crashes="size", fatal_crashes="sum")
props["fatal_%"] = 100 * props["fatal_crashes"] / props["crashes"]
r_m, r_n = (props.loc[g, "fatal_crashes"] / props.loc[g, "crashes"] for g in ("Metro", "Nonmetro"))
rr = r_n / r_m
se_rr = np.sqrt((1 - r_n) / props.loc["Nonmetro", "fatal_crashes"] + (1 - r_m) / props.loc["Metro", "fatal_crashes"])
print("Fatal proportions (analysis sample):")
print(props.round(2).to_string())

u = fits["1. Metro + year"]
bm_u, se_u = u.params[metro_coef], u.bse[metro_coef]
bm_f, se_fm = full.params[metro_coef], full.bse[metro_coef]
report = pd.DataFrame([
    {"Measure": "Unadjusted proportion ratio (nonmetro / metro)", "Estimate": rr,
     "95% CI": f"{rr * np.exp(-1.96 * se_rr):.2f}-{rr * np.exp(1.96 * se_rr):.2f}",
     "CI method": "Wald on log ratio, crashes independent"},
    {"Measure": "OR, metro + year (ML logistic)", "Estimate": np.exp(bm_u),
     "95% CI": f"{np.exp(bm_u - 1.96 * se_u):.2f}-{np.exp(bm_u + 1.96 * se_u):.2f}",
     "CI method": "Wald, county-clustered"},
    {"Measure": "Adjusted OR, full model (ML logistic)", "Estimate": np.exp(bm_f),
     "95% CI": f"{np.exp(bm_f - 1.96 * se_fm):.2f}-{np.exp(bm_f + 1.96 * se_fm):.2f}",
     "CI method": "Wald, county-clustered" + ("" if steps_table.loc["4. + environment (full)", "Usable"]
                                             else " - NOT USABLE, see flags")},
    {"Measure": "Adjusted OR, full model (Firth)", "Estimate": np.exp(beta_f[j]),
     "95% CI": f"{np.exp(ci_prof[0]):.2f}-{np.exp(ci_prof[1]):.2f}",
     "CI method": "profile penalized likelihood, crashes independent"},
    {"Measure": "Adjusted OR, full model (Firth)", "Estimate": np.exp(beta_f[j]),
     "95% CI": f"{np.exp(ci_boot[0]):.2f}-{np.exp(ci_boot[1]):.2f}",
     "CI method": f"county cluster bootstrap, percentile ({len(boot)} resamples)"},
]).set_index("Measure")
display(report.round(3))
report.round(4).to_csv("NC_metro_fatal_ratios.csv")
steps_table.round(4).to_csv("NC_metro_progressive_models.csv")

ci = full.conf_int()
coefs = pd.DataFrame({"OR (ML)": np.exp(full.params), "CI low": np.exp(ci[0]), "CI high": np.exp(ci[1]),
                      "p": full.pvalues, "OR (Firth)": np.exp(pd.Series(beta_f, index=cols_f))}).drop(index="Intercept")
coefs.index = [label(c) for c in coefs.index]
display(coefs.round(3))
coefs.round(4).to_csv("NC_adjusted_logit_full_model.csv")
print("Saved: NC_metro_fatal_ratios.csv, NC_metro_progressive_models.csv, NC_adjusted_logit_full_model.csv")
print("Reference levels: " + ", ".join(f"{c}={ref_level(c)}" for c in COVARIATES))
