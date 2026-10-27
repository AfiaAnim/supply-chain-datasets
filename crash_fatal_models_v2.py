# ===== Fatal vs nonfatal recorded crashes (FMCSA MCMIS crash file), version 2 =====
# Data checks -> cleaner analysis dataset with more predictors -> grouped (or later-year) split ->
# unweighted vs weighted baselines -> XGBoost tuned on validation PR-AUC -> sigmoid calibration on a
# separate calibration set -> test metrics with bootstrap CIs -> threshold for a target recall -> SHAP.
# Run cell by cell in Jupyter (# %% marks a cell). Needs: %pip install scikit-learn xgboost shap matplotlib
# Expects the crash data already loaded as `crash`; cell 1 loads it otherwise.

# %% 1. Settings and data checks (look at this output before trusting any model)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss, precision_recall_curve,
                             roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
import shap

PATH = "Crash_File_20261026.csv"
SEED = 42
UNIT = "report"                  # "report" = one row per crash report; "record" = one row per vehicle record
FEDERAL_RECORDABLE_ONLY = True   # keep only FEDERAL_RECORDABLE == "Y"
DROP_TRANSACTION_CODES = []      # fill in after cell 1, e.g. ["D"] if a code marks deleted records
UNKNOWN_CODES = {}               # e.g. {"LIGHT_CONDITION_ID": ["9"]} -> labelled "Unknown" (kept apart from "Missing")
MAX_ROWS = 1_000_000             # random sample after cleaning (None = everything)
TEST_MODE = "later_year"         # "later_year" = test on the latest full year; "random" = grouped 20% sample
INCLUDE_EVENTS = False           # CRASH_EVENT_SEQ_ID_DESC: happens during the crash, retrospective use only
TARGET_RECALLS = [0.3, 0.5, 0.7] # operating points for the threshold table
N_TRIALS = 16                    # XGBoost random-search trials
N_BOOT = 200                     # bootstrap resamples for confidence intervals
MIN_CAT_COUNT = 200              # rarer category levels are merged into "infrequent"

if "crash" not in globals():
    crash = pd.read_csv(PATH, low_memory=False)


def parse_report_date(s):
    return pd.to_datetime(pd.to_numeric(s, errors="coerce").astype("Int64").astype(str), format="%Y%m%d",
                          errors="coerce")


c = crash
fatal_row = c["FATALITIES"] > 0
report_id = c["REPORT_STATE"].astype(str) + "_" + c["REPORT_NUMBER"].astype(str)
print(f"Rows: {len(c):,} | unique CRASH_ID: {c['CRASH_ID'].nunique():,} | unique reports: {report_id.nunique():,}")
print(f"CRASH_IDs with more than one row (versions): {(c['CRASH_ID'].value_counts() > 1).sum():,}")
print("Rows per report:", report_id.value_counts().value_counts().sort_index().head(8).to_dict())
print(f"Reports whose rows disagree on FATALITIES: {(c.groupby(report_id)['FATALITIES'].nunique() > 1).sum():,}")
print(f"FATALITIES < 0: {(c['FATALITIES'] < 0).sum():,} | fatal rows: {fatal_row.mean():.2%}")
print("FATALITIES values:", c["FATALITIES"].value_counts().sort_index().head(8).to_dict())

for col in ["TRANSACTION_CODE", "CI_STATUS_CODE", "FEDERAL_RECORDABLE", "STATE_RECORDABLE", "TRUCK_BUS_IND"]:
    t = fatal_row.groupby(c[col].fillna("Missing")).agg(rows="size", fatal_rate="mean")
    print(f"\n{col}:\n{t.sort_values('rows', ascending=False).head(10).round(4).to_string()}")

date = parse_report_date(c["REPORT_DATE"])
t = pd.to_numeric(c["REPORT_TIME"], errors="coerce")
print(f"\nREPORT_DATE examples: {c['REPORT_DATE'].head(3).tolist()} | unparseable: {date.isna().mean():.2%} | "
      f"range {date.min():%Y-%m-%d} to {date.max():%Y-%m-%d}")
print(f"REPORT_TIME examples: {c['REPORT_TIME'].head(3).tolist()} | invalid HHMM: "
      f"{(~((t // 100).between(0, 23) & (t % 100).between(0, 59))).mean():.2%}")
print("\nFatal rate by year:")
print(fatal_row.groupby(date.dt.year).agg(rows="size", fatal_rate="mean").round(4).to_string())

# %% 2. Analysis dataset: clean records, one outcome per unit, more predictors
GROUPS = {
    "Vehicle": ["TRUCK_BUS_IND", "CARGO_BODY_TYPE_ID", "GVW_RATING_ID", "VEHICLE_CONFIGURATION_ID",
                "VEHICLE_HAZMAT_PLACARD"],
    "Roadway": ["TRAFFICWAY_ID", "ACCESS_CONTROL_ID", "ROAD_SURFACE_CONDITION_ID"],
    "Environment": ["WEATHER_CONDITION_ID", "LIGHT_CONDITION_ID"],
    "Time": ["HOUR", "WEEKDAY", "MONTH"],
    "Geography": ["CRASH_STATE", "COUNTY"],
    "Crash context": ["VEHICLES_IN_ACCIDENT", "N_RECORDS"],
}
if INCLUDE_EVENTS:
    GROUPS["Crash events"] = ["CRASH_EVENT_SEQ_ID_DESC"]
NUMERIC = {"VEHICLES_IN_ACCIDENT", "N_RECORDS"}
# Interactions, made explicit for logistic regression (trees find interactions on their own)
INTERACTIONS = [("LIGHT_CONDITION_ID", "TRAFFICWAY_ID"), ("WEATHER_CONDITION_ID", "ROAD_SURFACE_CONDITION_ID"),
                ("VEHICLE_CONFIGURATION_ID", "TRAFFICWAY_ID")]
# Never features: crash outcomes/consequences (INJURIES, TOW_AWAY, HAZMAT_RELEASED, *_RECORDABLE) would leak
# the answer; IDs, VIN, plates, carrier names/addresses and upload fields carry no real signal.

raw_cols = ["CRASH_ID", "REPORT_STATE", "REPORT_NUMBER", "REPORT_SEQ_NO", "REPORT_DATE", "REPORT_TIME",
            "CHANGE_DATE", "TRANSACTION_CODE", "FEDERAL_RECORDABLE", "FATALITIES", "STATE", "COUNTY_CODE",
            "VEHICLES_IN_ACCIDENT"] + GROUPS["Vehicle"] + GROUPS["Roadway"] + GROUPS["Environment"] \
           + (["CRASH_EVENT_SEQ_ID_DESC"] if INCLUDE_EVENTS else [])
d = crash[raw_cols].copy()
n0 = len(d)

d["_changed"] = pd.to_datetime(d["CHANGE_DATE"], errors="coerce")
d = d.sort_values("_changed").drop_duplicates("CRASH_ID", keep="last")          # latest version of each record
print(f"Latest version per CRASH_ID: {n0:,} -> {len(d):,}")
if DROP_TRANSACTION_CODES:
    d = d[~d["TRANSACTION_CODE"].isin(DROP_TRANSACTION_CODES)]
    print(f"After dropping TRANSACTION_CODE {DROP_TRANSACTION_CODES}: {len(d):,}")
if FEDERAL_RECORDABLE_ONLY:
    d = d[d["FEDERAL_RECORDABLE"].astype(str).str.upper() == "Y"]
    print(f"Federal recordable only: {len(d):,}")
d = d[d["FATALITIES"] >= 0]

d["GROUP_ID"] = d["REPORT_STATE"].astype(str) + "_" + d["REPORT_NUMBER"].astype(str)
d["FATAL"] = (d.groupby("GROUP_ID")["FATALITIES"].transform("max") > 0).astype(int)   # crash-level outcome
d["N_RECORDS"] = d.groupby("GROUP_ID")["CRASH_ID"].transform("size")              # vehicle records in report
if UNIT == "report":   # one row per crash; vehicle fields come from the first vehicle record
    d = d.sort_values(["GROUP_ID", "REPORT_SEQ_NO"]).drop_duplicates("GROUP_ID", keep="first")
    print(f"One row per crash report: {len(d):,}")

date = parse_report_date(d["REPORT_DATE"])
t = pd.to_numeric(d["REPORT_TIME"], errors="coerce")
d["YEAR"] = date.dt.year
d["MONTH"] = date.dt.month
d["WEEKDAY"] = date.dt.day_name().str[:3]
d["HOUR"] = (t // 100).where((t // 100).between(0, 23) & (t % 100).between(0, 59))
d["CRASH_STATE"] = d["STATE"].fillna(d["REPORT_STATE"])
d["COUNTY"] = d["CRASH_STATE"].astype(str) + "_" + pd.to_numeric(d["COUNTY_CODE"], errors="coerce") \
    .astype("Int64").astype(str).replace("<NA>", "Missing")
d["VEHICLES_IN_ACCIDENT"] = d["VEHICLES_IN_ACCIDENT"].clip(upper=10)

features = [c for g in GROUPS.values() for c in g]
categorical = [c for c in features if c not in NUMERIC]


def as_category(v):
    if pd.isna(v):
        return "Missing"
    return str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)


for col in categorical:   # codes are categories, never quantities; missing gets its own level
    d[col] = d[col].map(as_category)
    if col in UNKNOWN_CODES:
        d.loc[d[col].isin(UNKNOWN_CODES[col]), col] = "Unknown"
inter_cols = []
for a, b in INTERACTIONS:
    d[f"{a}_x_{b}"] = d[a] + "|" + d[b]
    inter_cols.append(f"{a}_x_{b}")

if MAX_ROWS and len(d) > MAX_ROWS:
    d = d.sample(MAX_ROWS, random_state=SEED)
print(f"\nAnalysis rows: {len(d):,} | fatal: {d['FATAL'].sum():,} ({d['FATAL'].mean():.2%})")
print("Features per group:", {g: len(cols) for g, cols in GROUPS.items()})
for col in ["LIGHT_CONDITION_ID", "WEATHER_CONDITION_ID", "TRAFFICWAY_ID"]:   # spot-check codes (incl. unknowns)
    print(f"{col}:", d[col].value_counts().head(12).to_dict())

# %% 3. Split: train (fit) / validation (tuning, early stopping) / calibration (calibration, threshold) / test
def group_split(df, test_size, seed):
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
                  .split(df, groups=df["GROUP_ID"]))
    return df.iloc[tr], df.iloc[te]


if TEST_MODE == "later_year":
    years = d["YEAR"].value_counts().sort_index()
    test_year = years.index[-1] if years.iloc[-1] >= 0.5 * years.iloc[:-1].median() else years.index[-2]
    test, rest = d[d["YEAR"] == test_year], d[d["YEAR"] < test_year]   # partial last year is skipped
    print(f"Test = crashes in {int(test_year)}; model built on earlier years")
else:
    rest, test = group_split(d, 0.20, SEED)
train_val, cal = group_split(rest, 0.15, SEED)
train, val = group_split(train_val, 0.15 / 0.85, SEED)

y_tr, y_va, y_ca, y_te = (s["FATAL"].values for s in (train, val, cal, test))
for name, s in [("train", train), ("validation", val), ("calibration", cal), ("test", test)]:
    print(f"{name:12} {len(s):>9,} rows | fatal rate {s['FATAL'].mean():.2%}")
base_rate = y_tr.mean()
print(f"Constant-probability Brier baseline on test: {brier_score_loss(y_te, np.full(len(y_te), base_rate)):.4f}")


def make_prep(cols, scale):
    cat = [c for c in cols if c not in NUMERIC]
    num = [c for c in cols if c in NUMERIC]
    num_steps = [("impute", SimpleImputer(strategy="median"))] + ([("scale", StandardScaler())] if scale else [])
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=MIN_CAT_COUNT), cat),
        ("num", Pipeline(num_steps), num)], verbose_feature_names_out=False)


# %% 4. Baselines: does class weighting help ranking, and what does it do to the probabilities?
def lr_model(cols, weight):
    return Pipeline([("prep", make_prep(cols, scale=True)),
                     ("clf", LogisticRegression(max_iter=3000, class_weight=weight))])


models = {
    "LR unweighted": (lr_model(features, None), features),
    "LR balanced weights": (lr_model(features, "balanced"), features),
    "LR unweighted + interactions": (lr_model(features + inter_cols, None), features + inter_cols),
    "Random forest unweighted": (Pipeline([("prep", make_prep(features, scale=False)),
                                           ("clf", RandomForestClassifier(
                                               n_estimators=300, min_samples_leaf=50,
                                               max_samples=min(200_000, len(train)), n_jobs=-1,
                                               random_state=SEED))]), features),
}
for name, (m, cols) in models.items():
    m.fit(train[cols], y_tr)
    p = m.predict_proba(val[cols])[:, 1]
    print(f"{name:30} val PR-AUC {average_precision_score(y_va, p):.4f} | ROC-AUC {roc_auc_score(y_va, p):.4f} "
          f"| Brier {brier_score_loss(y_va, p):.4f} | mean p {p.mean():.3f} vs rate {y_va.mean():.3f}")

# %% 5. XGBoost: random search on validation PR-AUC (preprocessing fitted on training data only)
xgb_prep = make_prep(features, scale=False).fit(train[features])
Xtr, Xva = xgb_prep.transform(train[features]), xgb_prep.transform(val[features])
pw = (y_tr == 0).sum() / max(y_tr.sum(), 1)
space = {"max_depth": [3, 4, 6, 8], "min_child_weight": [1, 5, 20, 50, 100], "subsample": [0.6, 0.8, 1.0],
         "colsample_bytree": [0.5, 0.8, 1.0], "reg_lambda": [1, 5, 20], "reg_alpha": [0, 1, 5],
         "gamma": [0, 1], "scale_pos_weight": [1, round(np.sqrt(pw), 2), round(pw, 2)]}
rng = np.random.default_rng(SEED)
trials = [{"max_depth": 6, "min_child_weight": 1, "subsample": 1.0, "colsample_bytree": 1.0, "reg_lambda": 1,
           "reg_alpha": 0, "gamma": 0, "scale_pos_weight": 1}]   # XGBoost defaults, unweighted
trials += [{k: v[rng.integers(len(v))] for k, v in space.items()} for _ in range(N_TRIALS - 1)]


def make_xgb(params, lr, n):
    return XGBClassifier(**params, learning_rate=lr, n_estimators=n, early_stopping_rounds=max(50, n // 20),
                         eval_metric="aucpr", tree_method="hist", n_jobs=-1, random_state=SEED)


search = []
for i, params in enumerate(trials, 1):
    m = make_xgb(params, 0.1, 1000).fit(Xtr, y_tr, eval_set=[(Xva, y_va)], verbose=False)
    p = m.predict_proba(Xva)[:, 1]
    search.append({**params, "trees": m.best_iteration + 1, "val PR-AUC": average_precision_score(y_va, p),
                   "val Brier (raw)": brier_score_loss(y_va, p)})
    print(f"trial {i:2}/{len(trials)}: PR-AUC {search[-1]['val PR-AUC']:.4f}")
search = pd.DataFrame(search).sort_values("val PR-AUC", ascending=False)
display(search.round(4).head(10))

best = search.iloc[0][list(space)].to_dict()
best = {k: (int(v) if k in ("max_depth",) else float(v)) for k, v in best.items()}
print("Best settings:", best, "-> refitting with learning rate 0.03")
xgb = make_xgb(best, 0.03, 5000).fit(Xtr, y_tr, eval_set=[(Xva, y_va)], verbose=False)
print(f"Final XGBoost trees: {xgb.best_iteration + 1}")
models["XGBoost tuned"] = (Pipeline([("prep", xgb_prep), ("clf", xgb)]), features)

# %% 6. Sigmoid calibration on the calibration set, then test metrics with bootstrap 95% CIs
def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def fit_sigmoid(p, y):   # Platt scaling on the model's log-odds
    lr = LogisticRegression(C=1e6).fit(logit(p), y)
    return lambda q: lr.predict_proba(logit(q))[:, 1]


groups_te = pd.factorize(test["GROUP_ID"])[0]
boot_w = [np.bincount(rng.integers(0, groups_te.max() + 1, groups_te.max() + 1),
                      minlength=groups_te.max() + 1)[groups_te] for _ in range(N_BOOT)]   # resample crashes


def ci(fn, y, p):
    v = [fn(y, p, sample_weight=w) for w in boot_w]
    return f"[{np.percentile(v, 2.5):.4f}, {np.percentile(v, 97.5):.4f}]"


brier_ref = brier_score_loss(y_te, np.full(len(y_te), base_rate))
rows, p_cal_set, p_test = [], {}, {}
for name, (m, cols) in models.items():
    raw_ca, raw_te = m.predict_proba(cal[cols])[:, 1], m.predict_proba(test[cols])[:, 1]
    calib = fit_sigmoid(raw_ca, y_ca)
    p_cal_set[name], p_test[name] = calib(raw_ca), calib(raw_te)
    pc = p_test[name]
    rows.append({"Model": name,
                 "ROC-AUC": roc_auc_score(y_te, pc),
                 "PR-AUC": average_precision_score(y_te, pc),
                 "PR-AUC 95% CI": ci(average_precision_score, y_te, pc),
                 "Brier raw": brier_score_loss(y_te, raw_te),
                 "Brier calibrated": brier_score_loss(y_te, pc),
                 "Brier 95% CI": ci(brier_score_loss, y_te, pc),
                 "Brier skill": 1 - brier_score_loss(y_te, pc) / brier_ref,
                 "Log loss": log_loss(y_te, pc),
                 "Mean p": pc.mean()})
results = pd.DataFrame(rows).set_index("Model")
print(f"Test: {len(y_te):,} rows | fatal rate {y_te.mean():.2%} | constant-probability Brier {brier_ref:.4f}")
print("Brier skill > 0 means better than always predicting the base rate. Calibration does not change "
      "ROC-AUC or PR-AUC.")
display(results.round(4))

fig, ax = plt.subplots(figsize=(6, 6))
top = max(np.quantile(np.concatenate(list(p_test.values())), 0.995), 2 * y_te.mean())
for name, pc in p_test.items():
    frac, mean_p = calibration_curve(y_te, pc, n_bins=10, strategy="quantile")
    ax.plot(mean_p, frac, "o-", ms=4, label=name)
ax.plot([0, top], [0, top], "k--", lw=1, label="Perfect calibration")
ax.set(xlim=(0, top), ylim=(0, top), xlabel="Mean predicted probability", ylabel="Observed fatal rate",
       title="Reliability (test, after calibration)")
ax.legend(frameon=False, fontsize=8)
ax.grid(color="#e4e3df", lw=0.6)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.show()

# %% 7. Thresholds chosen on the calibration set for a target recall, checked on the test set
chosen = max(p_cal_set, key=lambda n: average_precision_score(y_ca, p_cal_set[n]))   # picked without test data
print(f"Model used for thresholds (best calibration-set PR-AUC): {chosen}")
prec_c, rec_c, thr_c = precision_recall_curve(y_ca, p_cal_set[chosen])
pt = p_test[chosen]
ops = []
for target in TARGET_RECALLS:
    ok = np.where(rec_c[:-1] >= target)[0]
    t = thr_c[ok[-1]]                       # highest threshold that still reaches the target recall
    flag = pt >= t
    tp = (flag & (y_te == 1)).sum()
    ops.append({"Target recall": target, "Threshold": t,
                "Test recall": tp / y_te.sum(), "Test precision": tp / max(flag.sum(), 1),
                "Flagged per 1,000 crashes": 1000 * flag.mean(),
                "Precision / base rate": tp / max(flag.sum(), 1) / y_te.mean()})
ops = pd.DataFrame(ops)
display(ops.round(4))

prec_t, rec_t, _ = precision_recall_curve(y_te, pt)
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(rec_t, prec_t, label=chosen)
ax.scatter(ops["Test recall"], ops["Test precision"], color="#eb6834", zorder=3, label="Chosen thresholds")
ax.axhline(y_te.mean(), color="k", ls="--", lw=1, label=f"Random ({y_te.mean():.3f})")
ax.set(xlabel="Recall (fatal)", ylabel="Precision (fatal)", title="Precision-recall on test")
ax.legend(frameon=False)
ax.grid(color="#e4e3df", lw=0.6)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.show()

# %% 8. SHAP on the tuned XGBoost (calibration is monotone, so it does not change these rankings)
sample = test.sample(min(5000, len(test)), random_state=SEED)
X_shap = xgb_prep.transform(sample[features])
X_shap = X_shap.toarray() if hasattr(X_shap, "toarray") else X_shap
names = xgb_prep.get_feature_names_out()
sv = shap.TreeExplainer(xgb).shap_values(X_shap)

shap.summary_plot(sv, X_shap, feature_names=names, max_display=20, show=False)
plt.title("SHAP values: tuned XGBoost (top 20 category levels)")
plt.tight_layout()
plt.show()


def source_var(col):
    return next(f for f in sorted(features, key=len, reverse=True) if col == f or col.startswith(f + "_"))


group_of = {c: g for g, cols in GROUPS.items() for c in cols}
imp = (pd.DataFrame({"var": [source_var(n) for n in names], "mean_abs_shap": np.abs(sv).mean(axis=0)})
       .groupby("var")["mean_abs_shap"].sum().sort_values(ascending=False).to_frame())
imp["group"] = imp.index.map(group_of)
imp["share"] = imp["mean_abs_shap"] / imp["mean_abs_shap"].sum()
display(imp.round(4))
display(imp.groupby("group")[["mean_abs_shap", "share"]].sum().sort_values("share", ascending=False).round(4))
