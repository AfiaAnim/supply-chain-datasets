# ===== NC: predicting fatal vs nonfatal recorded crashes, 2020-2025, with an untouched later-period test set =====
# 1 filter to NC 2020-2025 first -> 2 reserve the test period (latest year that is complete enough and has enough
# fatal crashes) -> 3 tune on training/validation records only -> 4 evaluate the one selected model on the test set.
# Run AFTER cells 1 and 1b of crash_fatal_models_v2.py (needs `crash` with COUNTY_FIPS / RUCC_2023 / METRO).
# Run cell by cell in Jupyter (# %% marks a cell).

# %% A. Settings and NC 2020-2025 dataset (filtered before any split)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

YEARS = (2020, 2025)          # study period, both ends included
MIN_TEST_FATAL = 100          # the test period needs at least this many fatal crashes ...
MIN_TEST_COMPLETENESS = 0.85  # ... and at least this share of the average yearly record count of earlier years
SEED = 42
N_TRIALS = 20                 # XGBoost random-search trials (validation PR-AUC)
TARGET_RECALL = 0.50          # operating point: highest threshold reaching this recall on validation records
N_BOOT = 500                  # county cluster-bootstrap resamples for test-set confidence intervals
CAL_BINS = 8                  # calibration-curve bins (quantiles of predicted probability)

FEATURES = {
    "Vehicle": ["VEHICLE_CONFIGURATION_ID", "GVW_RATING_ID", "CARGO_BODY_TYPE_ID", "TRUCK_BUS_IND",
                "VEHICLE_HAZMAT_PLACARD"],
    "Roadway": ["TRAFFICWAY_ID", "ACCESS_CONTROL_ID", "ROAD_SURFACE_CONDITION_ID"],
    "Environment": ["WEATHER_CONDITION_ID", "LIGHT_CONDITION_ID"],
    "Time": ["HOUR", "WEEKDAY", "MONTH"],
    "Geography": ["METRO", "RUCC_2023"],
    "Crash context": ["VEHICLES_IN_ACCIDENT"],
}
NUMERIC = {"VEHICLES_IN_ACCIDENT"}
features = [c for cols in FEATURES.values() for c in cols]
# Never features: FATALITIES/INJURIES and other outcomes, report/carrier IDs, and YEAR (a later test year cannot
# be learned from training years, so year terms would only add noise).

assert "METRO" in crash.columns, "Run cell 1b (RUCC merge) first so crash has METRO"
raw = ["CRASH_ID", "REPORT_STATE", "REPORT_NUMBER", "REPORT_SEQ_NO", "REPORT_DATE", "REPORT_TIME", "CHANGE_DATE",
       "FEDERAL_RECORDABLE", "FATALITIES", "COUNTY_FIPS", "RUCC_2023", "METRO", "VEHICLES_IN_ACCIDENT"] \
      + FEATURES["Vehicle"] + FEATURES["Roadway"] + FEATURES["Environment"]
nc = crash.loc[crash["STATE"].astype("string").str.strip().str.upper() == "NC", list(dict.fromkeys(raw))].copy()
nc["_changed"] = pd.to_datetime(nc["CHANGE_DATE"], errors="coerce")
nc = nc.sort_values("_changed").drop_duplicates("CRASH_ID", keep="last")
nc = nc[nc["FEDERAL_RECORDABLE"].astype(str).str.upper() == "Y"]
nc["report_id"] = nc["REPORT_STATE"].astype(str) + "_" + nc["REPORT_NUMBER"].astype(str)
nc["fatal"] = (nc.groupby("report_id")["FATALITIES"].transform("max") > 0).astype(int)
nc = nc.sort_values(["report_id", "REPORT_SEQ_NO"]).drop_duplicates("report_id", keep="first")
date = pd.to_datetime(pd.to_numeric(nc["REPORT_DATE"], errors="coerce").astype("Int64").astype(str),
                      format="%Y%m%d", errors="coerce")
nc["year"] = date.dt.year
nc = nc[nc["year"].between(*YEARS)].copy()                       # 1) NC, 2020-2025, before anything else
date = date[nc.index]
t = pd.to_numeric(nc["REPORT_TIME"], errors="coerce")
nc["HOUR"] = (t // 100).where((t // 100).between(0, 23) & (t % 100).between(0, 59))
nc["WEEKDAY"] = date.dt.day_name().str[:3]
nc["MONTH"] = date.dt.month
nc["VEHICLES_IN_ACCIDENT"] = pd.to_numeric(nc["VEHICLES_IN_ACCIDENT"], errors="coerce").clip(upper=10)


def as_level(v):
    if pd.isna(v):
        return "Missing"
    return str(int(v)) if isinstance(v, float) and v.is_integer() else str(v).strip()


for col in features:
    if col not in NUMERIC:
        nc[col] = nc[col].map(as_level)

by_year = nc.groupby("year")["fatal"].agg(crashes="size", fatal="sum")
by_year["fatal_%"] = 100 * by_year["fatal"] / by_year["crashes"]
print(f"NC crash reports {YEARS[0]}-{YEARS[1]}: {len(nc):,} | fatal {nc['fatal'].sum():,} ({nc['fatal'].mean():.2%})")
print(by_year.round(2).to_string())

# %% B. Step 2: reserve the test set (untouched until cell E)
test_year = None
for yr in sorted(by_year.index, reverse=True):                    # latest year that passes both checks
    earlier = by_year.loc[by_year.index < yr, "crashes"]
    if len(earlier) == 0:
        break
    complete = by_year.loc[yr, "crashes"] / earlier.mean()
    enough = by_year.loc[yr, "fatal"] >= MIN_TEST_FATAL
    print(f"{yr}: {by_year.loc[yr, 'crashes']:,} crashes ({complete:.0%} of earlier-year average), "
          f"{by_year.loc[yr, 'fatal']} fatal -> {'OK' if complete >= MIN_TEST_COMPLETENESS and enough else 'not usable'}")
    if complete >= MIN_TEST_COMPLETENESS and enough:
        test_year = yr
        break

if test_year is not None:
    test = nc[nc["year"] == test_year]
    dev = nc[nc["year"] < test_year]          # later years than the test year (if any) are left out entirely
    split_note = f"temporal: train/validate on {YEARS[0]}-{test_year - 1}, test on {test_year}"
else:
    dev, test = train_test_split(nc, test_size=0.20, stratify=nc["fatal"], random_state=SEED)
    split_note = "random stratified 20% test (no later year was complete enough with enough fatal crashes)"
print(f"\nSplit: {split_note}")

# Development records -> training (60%) / validation-tune (20%) / validation-calibrate (20%), stratified
train, rest = train_test_split(dev, test_size=0.40, stratify=dev["fatal"], random_state=SEED)
val, cal = train_test_split(rest, test_size=0.50, stratify=rest["fatal"], random_state=SEED)
for name, s in [("train", train), ("validation (tuning)", val), ("validation (calibration, threshold)", cal),
                ("TEST (untouched)", test)]:
    print(f"{name:38} {len(s):>7,} crashes | {s['fatal'].sum():>4} fatal | prevalence {s['fatal'].mean():.2%}")
y_tr, y_va, y_ca, y_te = (s["fatal"].to_numpy() for s in (train, val, cal, test))


def make_prep(scale):
    cat = [c for c in features if c not in NUMERIC]
    num = [c for c in features if c in NUMERIC]
    num_steps = [("impute", SimpleImputer(strategy="median"))] + ([("scale", StandardScaler())] if scale else [])
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=30), cat),
        ("num", Pipeline(num_steps), num)], verbose_feature_names_out=False)


# %% C. Step 3: tune on training + validation records only (test set not used)
results = []

# Logistic regression (unweighted, L2): choose the penalty on validation PR-AUC
lr_best = None
for C in [0.01, 0.03, 0.1, 0.3, 1.0]:
    m = Pipeline([("prep", make_prep(scale=True)),
                  ("clf", LogisticRegression(C=C, max_iter=5000))]).fit(train[features], y_tr)
    ap = average_precision_score(y_va, m.predict_proba(val[features])[:, 1])
    results.append({"Model": "Logistic regression", "Setting": f"C={C}", "Validation PR-AUC": ap})
    if lr_best is None or ap > lr_best[0]:
        lr_best = (ap, m, f"C={C}")

# XGBoost: random search, early stopping on the validation set
prep_x = make_prep(scale=False).fit(train[features])
Xtr, Xva = prep_x.transform(train[features]), prep_x.transform(val[features])
pw = (y_tr == 0).sum() / max(y_tr.sum(), 1)
space = {"max_depth": [2, 3, 4, 6], "min_child_weight": [5, 20, 50, 100], "subsample": [0.6, 0.8, 1.0],
         "colsample_bytree": [0.5, 0.8, 1.0], "reg_lambda": [1, 10, 50], "gamma": [0, 1, 5],
         "scale_pos_weight": [1, round(np.sqrt(pw), 2)]}
rng = np.random.default_rng(SEED)
xgb_best = None
for i in range(N_TRIALS):
    params = {k: v[rng.integers(len(v))] for k, v in space.items()}
    m = XGBClassifier(**params, learning_rate=0.05, n_estimators=2000, early_stopping_rounds=100,
                      eval_metric="aucpr", tree_method="hist", n_jobs=-1, random_state=SEED)
    m.fit(Xtr, y_tr, eval_set=[(Xva, y_va)], verbose=False)
    ap = average_precision_score(y_va, m.predict_proba(Xva)[:, 1])
    results.append({"Model": "XGBoost", "Setting": str(params), "Validation PR-AUC": ap})
    if xgb_best is None or ap > xgb_best[0]:
        xgb_best = (ap, Pipeline([("prep", prep_x), ("clf", m)]), str(params))

tuning = pd.DataFrame(results).sort_values("Validation PR-AUC", ascending=False)
print(f"Validation prevalence (PR-AUC of a random ranking): {y_va.mean():.4f}")
display(tuning.head(10).round(4))

# Select ONE final model on validation PR-AUC
candidates = {"Logistic regression": lr_best, "XGBoost": xgb_best}
final_name = max(candidates, key=lambda k: candidates[k][0])
final = candidates[final_name][1]
print(f"Selected on validation PR-AUC: {final_name} ({candidates[final_name][2]}), "
      f"validation PR-AUC {candidates[final_name][0]:.4f}")

# Sigmoid calibration and the threshold, both from the calibration part of the validation records
def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


raw_ca = final.predict_proba(cal[features])[:, 1]
platt = LogisticRegression(C=1e6).fit(logit(raw_ca), y_ca)
calibrate = lambda q: platt.predict_proba(logit(q))[:, 1]
p_ca = calibrate(raw_ca)
prec_c, rec_c, thr_c = precision_recall_curve(y_ca, p_ca)
threshold = thr_c[np.where(rec_c[:-1] >= TARGET_RECALL)[0][-1]]   # highest threshold that still reaches it
f1 = 2 * prec_c[:-1] * rec_c[:-1] / np.clip(prec_c[:-1] + rec_c[:-1], 1e-12, None)
threshold_f1 = thr_c[np.argmax(f1)]
print(f"Thresholds chosen on validation records: {threshold:.4f} (recall >= {TARGET_RECALL:.0%}), "
      f"{threshold_f1:.4f} (max F1)")

# %% D. (nothing below this line was used for any choice above)
# %% E. Step 4: evaluate the final model once on the test set
p_te = calibrate(final.predict_proba(test[features])[:, 1])
prev = y_te.mean()
groups_te = pd.factorize(test["COUNTY_FIPS"].fillna("missing"))[0]
ng = groups_te.max() + 1
boot_w = [np.bincount(rng.integers(0, ng, ng), minlength=ng)[groups_te] for _ in range(N_BOOT)]


def ci(fn, *args, digits=3):
    vals = [fn(*args, w) for w in boot_w]
    return f"{np.percentile(vals, 2.5):.{digits}f}-{np.percentile(vals, 97.5):.{digits}f}"


def prec_rec(y, p, thr, w=None):
    w = np.ones_like(y, float) if w is None else w
    flag = p >= thr
    tp = np.sum(w * (flag & (y == 1)))
    return tp / max(np.sum(w * flag), 1e-12), tp / max(np.sum(w * (y == 1)), 1e-12)


brier_ref = brier_score_loss(y_te, np.full(len(y_te), y_tr.mean()))
rows = [
    ("Test crashes / fatal", f"{len(y_te):,} / {int(y_te.sum())}", "", "Size of the test period"),
    ("Test fatal prevalence", f"{prev:.4f}", "", "PR-AUC of a random ranking (the rare-outcome baseline)"),
    ("PR-AUC", f"{average_precision_score(y_te, p_te):.4f}",
     ci(lambda y, p, w: average_precision_score(y, p, sample_weight=w), y_te, p_te),
     f"{average_precision_score(y_te, p_te) / prev:.1f}x the baseline"),
    ("ROC-AUC", f"{roc_auc_score(y_te, p_te):.4f}",
     ci(lambda y, p, w: roc_auc_score(y, p, sample_weight=w), y_te, p_te), "Ranking ability (0.5 = random)"),
    ("Brier score", f"{brier_score_loss(y_te, p_te):.5f}",
     ci(lambda y, p, w: brier_score_loss(y, p, sample_weight=w), y_te, p_te, digits=5),
     f"Constant-probability reference {brier_ref:.5f}; skill {1 - brier_score_loss(y_te, p_te) / brier_ref:.3f}"),
    ("Mean predicted vs observed", f"{p_te.mean():.4f} vs {prev:.4f}", "", "Calibration-in-the-large"),
]
for thr, name in [(threshold, f"recall >= {TARGET_RECALL:.0%} on validation"), (threshold_f1, "max F1 on validation")]:
    pr, rc = prec_rec(y_te, p_te, thr)
    rows += [(f"Fatal precision @ {thr:.4f}", f"{pr:.4f}", ci(lambda y, p, w: prec_rec(y, p, thr, w)[0], y_te, p_te),
              f"Threshold: {name}; {pr / prev:.1f}x prevalence"),
             (f"Fatal recall @ {thr:.4f}", f"{rc:.4f}", ci(lambda y, p, w: prec_rec(y, p, thr, w)[1], y_te, p_te),
              f"Crashes flagged: {(p_te >= thr).mean():.1%}")]
report = pd.DataFrame(rows, columns=["Metric", "Value", "95% CI (county bootstrap)", "Note"]).set_index("Metric")
print(f"Final model: {final_name} | {split_note}")
display(report)
report.to_csv("NC_fatal_prediction_test_metrics.csv")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
prec_t, rec_t, _ = precision_recall_curve(y_te, p_te)
a1.plot(rec_t, prec_t, color="#2a78d6", lw=2, label=f"{final_name} (PR-AUC {average_precision_score(y_te, p_te):.3f})")
a1.axhline(prev, color="#898781", ls="--", lw=1, label=f"Random ranking ({prev:.3f})")
for thr, mk, lab in [(threshold, "o", f"Threshold: recall >= {TARGET_RECALL:.0%} (validation)"),
                     (threshold_f1, "s", "Threshold: max F1 (validation)")]:
    pr, rc = prec_rec(y_te, p_te, thr)
    a1.plot(rc, pr, mk, color="#eb6834", ms=8, mec="#fcfcfb", mew=1.5, label=lab)
a1.set_ylim(0, 1.3 * prec_t[rec_t >= 0.05].max())          # precision at tiny recall rests on 1-2 crashes
a1.set(xlabel="Fatal recall", ylabel="Fatal precision", title="Precision-recall on the test set")
frac, mean_p = calibration_curve(y_te, p_te, n_bins=CAL_BINS, strategy="quantile")
top = max(mean_p.max(), frac.max()) * 1.1
a2.plot([0, top], [0, top], color="#898781", ls="--", lw=1, label="Perfect calibration")
a2.plot(mean_p, frac, "o-", color="#2a78d6", lw=2, ms=6, label=final_name)
a2.set(xlim=(0, top), ylim=(0, top), xlabel="Mean predicted probability", ylabel="Observed fatal share",
       title=f"Calibration on the test set ({CAL_BINS} equal-size bins)")
for a in (a1, a2):
    a.legend(frameon=False, fontsize=9)
    a.grid(color="#e1e0d9", lw=0.6)
    a.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.show()
