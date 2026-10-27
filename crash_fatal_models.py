# ===== Fatal vs nonfatal recorded crashes (FMCSA MCMIS crash file) =====
# How accurately can vehicle, roadway and environmental characteristics distinguish fatal crashes?
# Logistic regression baseline -> Random Forest -> XGBoost, then SHAP on XGBoost.
# Run cell by cell in Jupyter (# %% marks a cell). Needs: pip install scikit-learn xgboost shap matplotlib
# Expects the crash data already loaded as `crash` (e.g. crash = pd.read_csv(...)); cell 1 loads it otherwise.

# %% 1. Settings and data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
import shap

PATH = "Crash_File_20261026.csv"
MAX_ROWS = 1_000_000          # model on a random sample of crashes (None = all ~5M rows; RF gets slow)
INCLUDE_CONTEXT = False       # True adds crash context (vehicles involved, hour, month) as a 4th group
SEED = 42

# Variable groups. Codes stored as numbers (e.g. WEATHER_CONDITION_ID) are categories, not quantities.
GROUPS = {
    "Vehicle": ["TRUCK_BUS_IND", "CARGO_BODY_TYPE_ID", "GVW_RATING_ID", "VEHICLE_CONFIGURATION_ID",
                "VEHICLE_HAZMAT_PLACARD"],
    "Roadway": ["TRAFFICWAY_ID", "ACCESS_CONTROL_ID", "ROAD_SURFACE_CONDITION_ID"],
    "Environment": ["WEATHER_CONDITION_ID", "LIGHT_CONDITION_ID"],
}
if INCLUDE_CONTEXT:
    GROUPS["Context"] = ["VEHICLES_IN_ACCIDENT", "HOUR", "MONTH"]
NUMERIC = {"VEHICLES_IN_ACCIDENT", "HOUR"}
# Never used as features: outcomes or consequences of the crash (INJURIES, TOW_AWAY, HAZMAT_RELEASED,
# *_RECORDABLE) would leak the answer; IDs, VIN, plates, carrier names and upload fields carry no signal.

needed = ["CRASH_ID", "REPORT_STATE", "REPORT_NUMBER", "REPORT_DATE", "REPORT_TIME", "CHANGE_DATE",
          "FATALITIES", "VEHICLES_IN_ACCIDENT"] + [c for g in GROUPS.values() for c in g if c not in ("HOUR", "MONTH")]
needed = list(dict.fromkeys(needed))
if "crash" not in globals():
    crash = pd.read_csv(PATH, usecols=needed, low_memory=False)
data = crash[needed].copy()

# One row per crash record: keep the latest version of each CRASH_ID
data = data.sort_values("CHANGE_DATE").drop_duplicates("CRASH_ID", keep="last")
data["HOUR"] = (pd.to_numeric(data["REPORT_TIME"], errors="coerce") // 100).where(lambda h: h.between(0, 23))
data["MONTH"] = pd.to_datetime(data["REPORT_DATE"].astype(str), format="%Y%m%d", errors="coerce").dt.month
data["FATAL"] = (data["FATALITIES"] > 0).astype(int)
# Records from the same crash report share one outcome; keep them on the same side of the split
data["GROUP_ID"] = data["REPORT_STATE"].astype(str) + "_" + data["REPORT_NUMBER"].astype(str)
if MAX_ROWS and len(data) > MAX_ROWS:
    data = data.sample(MAX_ROWS, random_state=SEED)

features = [c for g in GROUPS.values() for c in g]
categorical = [c for c in features if c not in NUMERIC]
for c in categorical:   # codes -> text categories; missing becomes its own category
    data[c] = data[c].map(lambda v: "Missing" if pd.isna(v) else str(int(v)) if isinstance(v, float) and v.is_integer() else str(v))

print(f"{len(data):,} crash records | fatal: {data['FATAL'].sum():,} ({data['FATAL'].mean():.2%})")
print("Features:", {g: len(c) for g, c in GROUPS.items()})

# %% 2. Train / validation / test split (60/20/20, grouped by crash report)
def group_split(df, test_size, seed):
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
                  .split(df, groups=df["GROUP_ID"]))
    return df.iloc[tr], df.iloc[te]

train_val, test = group_split(data, 0.20, SEED)
train, val = group_split(train_val, 0.25, SEED)
X_tr, y_tr = train[features], train["FATAL"].values
X_va, y_va = val[features], val["FATAL"].values
X_te, y_te = test[features], test["FATAL"].values
print(f"train {len(train):,} | val {len(val):,} | test {len(test):,} | "
      f"fatal rate {y_tr.mean():.2%} / {y_va.mean():.2%} / {y_te.mean():.2%}")


def make_prep(cols, scale):
    cat = [c for c in cols if c not in NUMERIC]
    num = [c for c in cols if c in NUMERIC]
    num_steps = [("impute", SimpleImputer(strategy="median"))] + ([("scale", StandardScaler())] if scale else [])
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.001), cat),
        ("num", Pipeline(num_steps), num)], verbose_feature_names_out=False)


# %% 3. Fit the three models (fatal crashes are rare, so each one up-weights the fatal class)
pos_weight = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
models = {
    "Logistic regression": Pipeline([("prep", make_prep(features, scale=True)),
                                     ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
    "Random forest": Pipeline([("prep", make_prep(features, scale=False)),
                               ("clf", RandomForestClassifier(n_estimators=300, min_samples_leaf=20,
                                                              max_samples=min(200_000, len(train)),
                                                              class_weight="balanced_subsample",
                                                              n_jobs=-1, random_state=SEED))]),
}
for name, m in models.items():
    print(f"Fitting {name} ...")
    m.fit(X_tr, y_tr)

print("Fitting XGBoost (early stopping on validation) ...")
xgb_prep = make_prep(features, scale=False).fit(X_tr)
xgb = XGBClassifier(n_estimators=2000, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                    min_child_weight=5, scale_pos_weight=pos_weight, eval_metric="aucpr",
                    early_stopping_rounds=100, tree_method="hist", n_jobs=-1, random_state=SEED)
xgb.fit(xgb_prep.transform(X_tr), y_tr, eval_set=[(xgb_prep.transform(X_va), y_va)], verbose=False)
models["XGBoost"] = Pipeline([("prep", xgb_prep), ("clf", xgb)])
print(f"XGBoost best iteration: {xgb.best_iteration}")

# %% 4. How accurately? Test-set metrics
# Accuracy is misleading when ~97% of crashes are nonfatal, so report ranking metrics (ROC-AUC, PR-AUC)
# and precision/recall at a threshold chosen on the validation set (max F1), never on the test set.
rows, probs = [], {}
for name, m in models.items():
    p_va, p_te = m.predict_proba(X_va)[:, 1], m.predict_proba(X_te)[:, 1]
    prec, rec, thr = precision_recall_curve(y_va, p_va)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    t = thr[np.argmax(f1[:-1])]
    pred = (p_te >= t).astype(int)
    probs[name] = p_te
    rows.append({"Model": name,
                 "ROC-AUC": roc_auc_score(y_te, p_te),
                 "PR-AUC": average_precision_score(y_te, p_te),
                 "PR-AUC / base rate": average_precision_score(y_te, p_te) / y_te.mean(),
                 "Recall (fatal)": recall_score(y_te, pred),
                 "Precision (fatal)": precision_score(y_te, pred, zero_division=0),
                 "F1": f1_score(y_te, pred),
                 "Brier": brier_score_loss(y_te, p_te)})
results = pd.DataFrame(rows).set_index("Model")
print(f"Test set: {len(y_te):,} records, fatal base rate {y_te.mean():.2%} (= PR-AUC of random guessing)")
display(results.round(4))

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
for name, p in probs.items():
    fpr, tpr, _ = roc_curve(y_te, p)
    a1.plot(fpr, tpr, label=f"{name} (AUC {results.loc[name, 'ROC-AUC']:.3f})")
    prec, rec, _ = precision_recall_curve(y_te, p)
    a2.plot(rec, prec, label=f"{name} (AP {results.loc[name, 'PR-AUC']:.3f})")
a1.plot([0, 1], [0, 1], "k--", lw=1)
a2.axhline(y_te.mean(), color="k", ls="--", lw=1, label=f"Random ({y_te.mean():.3f})")
a1.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve (test)")
a2.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve (test)")
for a in (a1, a2):
    a.legend(frameon=False, fontsize=9)
    a.grid(color="#e4e3df", lw=0.6)
    a.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.show()

# %% 5. How much does each group contribute on its own? (XGBoost per group, same split)
ablation = []
for g, cols in list(GROUPS.items()) + [("All groups", features)]:
    prep = make_prep(cols, scale=False).fit(X_tr[cols])
    m = XGBClassifier(n_estimators=2000, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                      min_child_weight=5, scale_pos_weight=pos_weight, eval_metric="aucpr",
                      early_stopping_rounds=100, tree_method="hist", n_jobs=-1, random_state=SEED)
    m.fit(prep.transform(X_tr[cols]), y_tr, eval_set=[(prep.transform(X_va[cols]), y_va)], verbose=False)
    p = m.predict_proba(prep.transform(X_te[cols]))[:, 1]
    ablation.append({"Variables": g, "n vars": len(cols),
                     "ROC-AUC": roc_auc_score(y_te, p), "PR-AUC": average_precision_score(y_te, p)})
display(pd.DataFrame(ablation).set_index("Variables").round(4))

# %% 6. SHAP: which characteristics drive the XGBoost predictions?
n_shap = min(5000, len(X_te))
X_shap = xgb_prep.transform(X_te.sample(n_shap, random_state=SEED))
X_shap = X_shap.toarray() if hasattr(X_shap, "toarray") else X_shap
names = xgb_prep.get_feature_names_out()
explainer = shap.TreeExplainer(xgb)
sv = explainer.shap_values(X_shap)

shap.summary_plot(sv, X_shap, feature_names=names, max_display=20, show=False)   # one-hot level detail
plt.title("SHAP values: XGBoost, fatal vs nonfatal (top 20 levels)")
plt.tight_layout()
plt.show()

# Add up |SHAP| over each variable's one-hot columns, then over each group
def source_var(col):
    return next(f for f in sorted(features, key=len, reverse=True) if col == f or col.startswith(f + "_"))

group_of = {c: g for g, cols in GROUPS.items() for c in cols}
imp = (pd.DataFrame({"var": [source_var(n) for n in names], "mean_abs_shap": np.abs(sv).mean(axis=0)})
       .groupby("var")["mean_abs_shap"].sum().sort_values(ascending=False).to_frame())
imp["group"] = imp.index.map(group_of)
imp["share"] = imp["mean_abs_shap"] / imp["mean_abs_shap"].sum()
display(imp.round(4))
display(imp.groupby("group")[["mean_abs_shap", "share"]].sum().sort_values("share", ascending=False).round(4))

colors = {"Vehicle": "#2a78b5", "Roadway": "#eb6834", "Environment": "#3a9e5f", "Context": "#8a63b8"}
fig, ax = plt.subplots(figsize=(8, 0.4 * len(imp) + 1))
ax.barh(imp.index[::-1], imp["mean_abs_shap"][::-1], color=[colors[g] for g in imp["group"][::-1]])
for g, c in colors.items():
    if g in GROUPS:
        ax.barh([], [], color=c, label=g)
ax.set(xlabel="Mean |SHAP| (log-odds), summed over categories", title="Variable importance for fatal crashes")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.show()
