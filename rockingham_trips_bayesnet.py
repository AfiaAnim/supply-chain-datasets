"""
Bayesian network for the Rockingham FY24 trips: which trip characteristics depend on
which, and how well does that structure predict new trips?

One row per trip (39,037 trips, minus those with missing/out-of-NC coordinates), with
8 discrete variables:
    calendar : DOW (Mon-Sat), SEASON, HOLIDAY_WEEK (week contains a federal holiday)
    trip     : PURPOSE (top purposes + Other), MOBILITY_AID (No aid / Wheelchair / Walker /
               Other aid), DISTANCE (straight-line km band)
    place    : ORIG_ZONE, DEST_ZONE (k-means clusters of pickup / drop-off coordinates)

Steps
    1. Time split: learn on Jul 2023 - Apr 2024, test on May - Jun 2024 (unseen months).
    2. Structure: hill-climbing search with the BIC score, with a variable ORDER:
       calendar -> rider need (PURPOSE, MOBILITY_AID) -> geography (zones, DISTANCE).
       Arrows may only point forward in this order (or within a tier, except between
       calendar variables). Without the order (USE_ORDER = False) the search made the
       zones parents of PURPOSE, and the BIC penalty then kept DOW out of PURPOSE: the
       network said Saturday trips are 44% dialysis when they are 100%. With the order
       (test months): purpose accuracy 63% vs 56%, Saturday dialysis 99.9%, and the joint
       fit of all 8 variables also improves (9.23 vs 9.33 nats per trip).
    3. Parameters: Bayesian estimation (BDeu prior), so unseen combinations are not 0.
    4. Stability: re-learn the structure on N_BOOT bootstrap samples and report how often
       each arrow appears (direction-free). Arrows below ~70% are not reliable.
    5. Test: predict PURPOSE and MOBILITY_AID for each test trip from all other variables,
       compared with always guessing the most common class; and the average
       log-likelihood per trip compared with a model where every variable is independent.
    6. Queries (exact inference): P(mobility aid | purpose), P(distance | purpose),
       P(purpose | day of week), P(purpose | drop-off zone).

NOTES
    - MOBILITY_AID: a blank in the FY24 file is read as "No aid". The file does not say
      whether blank means "no aid" or "not recorded". Codes WC = wheelchair and
      WK = walker are assumed; all other codes (BL, CA, NL, OX, RL, IS) are "Other aid".
    - Zones are data-driven clusters, labelled by their centre; a nearby town is named
      only if the centre is within 5 km of it.
    - A Bayesian network shows dependence, not proof of cause. Arrow direction inside the
      trip/place variables is often not identifiable from data alone.

Outputs (OUT_DIR): bn_dataset.csv, bn_edges.csv, bn_cpds.txt, bn_evaluation.csv,
    bn_query_*.csv, bn_structure.png, bn_queries.png, bn_evaluation.png

In Jupyter: just run the cell. Requirements:
    pip install "pgmpy>=1.1" pandas numpy matplotlib scikit-learn networkx
"""

import glob
import os
import warnings

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    from pgmpy.causal_discovery import ExpertKnowledge
    from pgmpy.estimators import HillClimbSearch
    from pgmpy.inference import VariableElimination
    from pgmpy.models import DiscreteBayesianNetwork
    from pgmpy.parameter_estimator import DiscreteBayesianEstimator
except ImportError as e:
    raise ImportError(f"This script needs pgmpy 1.1 or newer ({e}).\n"
                      'Run:  pip install --upgrade "pgmpy>=1.1"') from e

# ============================================================
# USER CONFIGURATION
# ============================================================
TRIPS_PATH = "merged/rockingham_fy24_trips_clean.csv"   # from merge_rockingham_data.py
OUT_DIR = "bn_output"

TEST_FROM = "2024-05-01"     # trips on/after this date are the test set
N_ZONES = 8                  # k-means clusters for pickup/drop-off zones
MAX_INDEGREE = 3             # max parents per variable (keeps tables estimable)
EQUIV_SAMPLE_SIZE = 10       # BDeu prior strength
N_BOOT = 30                  # bootstrap structure re-learns (0 = skip)
USE_ORDER = True             # calendar -> rider need -> geography (see docstring)
SEED = 0

PURPOSE_KEEP = ["Dialysis", "Medical", "UMAR", "LEAF Center", "Community Support", "Employment"]
DISTANCE_BINS = [0, 3, 10, 25, np.inf]
DISTANCE_LABELS = ["<3 km", "3-10 km", "10-25 km", ">25 km"]
CALENDAR = ["DOW", "SEASON", "HOLIDAY_WEEK"]
NEED = ["PURPOSE", "MOBILITY_AID"]
GEOGRAPHY = ["ORIG_ZONE", "DEST_ZONE", "DISTANCE"]
DOW_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
VARIABLES = CALENDAR + ["PURPOSE", "MOBILITY_AID", "DISTANCE", "ORIG_ZONE", "DEST_ZONE"]

# Town centres used only to give zones a readable name (within 5 km).
TOWNS = {"Reidsville": (36.354, -79.664), "Eden": (36.488, -79.767),
         "Madison": (36.385, -79.960), "Mayodan": (36.412, -79.970),
         "Stoneville": (36.465, -79.908), "Wentworth": (36.400, -79.774),
         "Greensboro": (36.073, -79.792), "Winston-Salem": (36.100, -80.244),
         "Danville VA": (36.586, -79.395)}
GROUP_COLORS = {"calendar": "#8a8a8a", "trip": "#2a6fdb", "place": "#e07b00"}


# ============================================================
# DATA
# ============================================================
def find(path, pattern):
    if path and os.path.exists(path):
        return path
    for root in (os.getcwd(), os.path.expanduser("~")):
        hits = [h for h in glob.glob(os.path.join(root, "**", pattern), recursive=True)
                if "site-packages" not in h]
        if hits:
            hits.sort(key=os.path.getmtime, reverse=True)
            print(f"Using: {hits[0]}")
            return hits[0]
    raise FileNotFoundError(f"Could not find {pattern}. Run merge_rockingham_data.py first "
                            "or set TRIPS_PATH at the top of this script.")


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def season(month):
    return {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
            6: "Summer", 7: "Summer", 8: "Summer"}.get(month, "Fall")


def mobility(code):
    if pd.isna(code):
        return "No aid"
    return {"WC": "Wheelchair", "WK": "Walker"}.get(str(code).strip().upper(), "Other aid")


def purpose(p):
    if pd.isna(p):
        return "Other"
    p = "Community Support" if str(p).startswith("Community Support") else str(p)
    return p if p in PURPOSE_KEEP else "Other"


def zone_names(centres, counts):
    names = {}
    order = np.argsort(-counts)
    for rank, k in enumerate(order, 1):
        lat, lon = centres[k]
        near = min(TOWNS, key=lambda t: haversine_km(lat, lon, *TOWNS[t]))
        d = haversine_km(lat, lon, *TOWNS[near])
        label = f"Z{rank} {near}" if d <= 5 else f"Z{rank} ({lat:.2f},{lon:.2f})"
        names[k] = label
    return names


def build_dataset(t):
    t = t[t["ORIG_COORD_OK"] & t["DEST_COORD_OK"]].copy()
    t["TRIP_DATE"] = pd.to_datetime(t["TRIP_DATE"])
    hol = USFederalHolidayCalendar().holidays(t["TRIP_DATE"].min(), t["TRIP_DATE"].max())
    hol_weeks = set(pd.DatetimeIndex(hol).to_period("W"))

    d = pd.DataFrame(index=t.index)
    d["DATE"] = t["TRIP_DATE"]
    d["DOW"] = t["TRIP_DATE"].dt.day_name().str[:3]
    d["SEASON"] = t["TRIP_DATE"].dt.month.map(season)
    d["HOLIDAY_WEEK"] = np.where(t["TRIP_DATE"].dt.to_period("W").isin(hol_weeks), "Yes", "No")
    d["PURPOSE"] = t["TRIP_PURPOSE"].map(purpose)
    d["MOBILITY_AID"] = t["MOBILITY_AID"].map(mobility)
    km = haversine_km(t["ORIG_LAT"], t["ORIG_LON"], t["DEST_LAT"], t["DEST_LON"])
    d["DISTANCE_KM"] = km
    d["DISTANCE"] = pd.cut(km, DISTANCE_BINS, labels=DISTANCE_LABELS, right=False).astype(str)

    # Zones: k-means on all endpoints of the TRAINING trips (lon scaled to ~equal km).
    train = d["DATE"] < TEST_FROM
    sx = np.cos(np.radians(36.4))
    ends = np.vstack([t.loc[train, ["ORIG_LAT", "ORIG_LON"]].to_numpy(),
                      t.loc[train, ["DEST_LAT", "DEST_LON"]].to_numpy()])
    km_model = KMeans(N_ZONES, n_init=10, random_state=SEED).fit(ends * [1, sx])
    centres = km_model.cluster_centers_ / [1, sx]
    names = zone_names(centres, np.bincount(km_model.labels_, minlength=N_ZONES))
    for end in ("ORIG", "DEST"):
        lab = km_model.predict(t[[f"{end}_LAT", f"{end}_LON"]].to_numpy() * [1, sx])
        d[f"{end}_ZONE"] = pd.Series(lab, index=t.index).map(names)
    zones = pd.DataFrame([(names[k], *centres[k]) for k in range(N_ZONES)],
                         columns=["ZONE", "LAT", "LON"]).sort_values("ZONE")
    return d.reset_index(drop=True), zones


# ============================================================
# MODEL
# ============================================================
def expert_knowledge():
    # nothing points INTO a calendar variable (also not calendar -> calendar)
    forbidden = [(v, c) for c in CALENDAR for v in VARIABLES if v != c]
    if USE_ORDER:
        forbidden += [(v, n) for n in NEED for v in GEOGRAPHY]
    return ExpertKnowledge(forbidden_edges=forbidden)


def ordered_states(data):
    states = {v: sorted(data[v].unique()) for v in VARIABLES}
    states["DOW"] = [d for d in DOW_ORDER if d in set(data["DOW"])]
    states["DISTANCE"] = [d for d in DISTANCE_LABELS if d in set(data["DISTANCE"])]
    return states


def learn_structure(data):
    hc = HillClimbSearch(data[VARIABLES])
    return hc.estimate(scoring_method="bic-d", max_indegree=MAX_INDEGREE,
                       expert_knowledge=expert_knowledge(), show_progress=False)


def fit(dag, data, states):
    model = DiscreteBayesianNetwork(dag.edges())
    model.add_nodes_from(VARIABLES)
    est = DiscreteBayesianEstimator(state_names=states, prior_type="BDeu",
                                    equivalent_sample_size=EQUIV_SAMPLE_SIZE)
    return model.fit(data[VARIABLES], estimator=est)


def bootstrap_edges(data, n, rng):
    counts = {}
    for b in range(n):
        sample = data.sample(len(data), replace=True, random_state=int(rng.integers(1e9)))
        for u, v in learn_structure(sample).edges():
            key = tuple(sorted((u, v)))
            counts[key] = counts.get(key, 0) + 1
        print(f"\r  bootstrap {b + 1}/{n}", end="")
    print()
    return {k: c / n for k, c in counts.items()}


def log_lik(model, data):
    """Log-likelihood of each row under the fitted network (vectorised CPD lookup)."""
    ll = np.zeros(len(data))
    for cpd in model.get_cpds():
        idx = []
        for var in cpd.variables:
            pos = {s: i for i, s in enumerate(cpd.state_names[var])}
            idx.append(data[var].map(pos).to_numpy())
        ll += np.log(cpd.values[tuple(idx)])
    return ll


def independent_log_lik(train, test, states):
    ll = np.zeros(len(test))
    for v in VARIABLES:
        counts = train[v].value_counts().reindex(states[v], fill_value=0) + 0.5
        p = counts / counts.sum()
        ll += np.log(test[v].map(p).to_numpy())
    return ll


def predict(model, data, target):
    """Posterior of `target` given all other variables, for each row (cached by evidence)."""
    infer = VariableElimination(model)
    others = [v for v in VARIABLES if v != target]
    states = model.get_cpds(target).state_names[target]
    cache = {}
    probs = np.zeros((len(data), len(states)))
    for i, row in enumerate(data[others].itertuples(index=False, name=None)):
        if row not in cache:
            q = infer.query([target], evidence=dict(zip(others, row)), show_progress=False)
            cache[row] = q.values
        probs[i] = cache[row]
    return states, probs


def evaluate(model, train, test, states):
    rows = []
    for target in ("PURPOSE", "MOBILITY_AID"):
        labels, probs = predict(model, test, target)
        y = test[target].map({s: i for i, s in enumerate(labels)}).to_numpy()
        acc = (probs.argmax(axis=1) == y).mean()
        logloss = -np.mean(np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1)))
        freq = train[target].value_counts(normalize=True).reindex(labels, fill_value=0) + 1e-6
        base_acc = (test[target] == train[target].mode()[0]).mean()
        base_ll = -np.mean(np.log(test[target].map(freq).to_numpy()))
        rows.append({"TARGET": target, "BN_ACCURACY": acc, "BASELINE_ACCURACY": base_acc,
                     "BN_LOG_LOSS": logloss, "BASELINE_LOG_LOSS": base_ll,
                     "BASELINE": f"always '{train[target].mode()[0]}' / class frequencies"})
    bn = log_lik(model, test).mean()
    ind = independent_log_lik(train, test, states).mean()
    rows.append({"TARGET": "ALL VARIABLES (avg log-likelihood per trip)",
                 "BN_LOG_LOSS": -bn, "BASELINE_LOG_LOSS": -ind,
                 "BASELINE": "all variables independent"})
    return pd.DataFrame(rows)


def query_table(model, target, given):
    infer = VariableElimination(model)
    g_states = model.get_cpds(given).state_names[given]
    t_states = model.get_cpds(target).state_names[target]
    tab = pd.DataFrame(index=g_states, columns=t_states, dtype=float)
    for g in g_states:
        tab.loc[g] = infer.query([target], evidence={given: g}, show_progress=False).values
    tab.index.name, tab.columns.name = given, target
    return tab


# ============================================================
# FIGURES
# ============================================================
def group(v):
    return "calendar" if v in CALENDAR else "place" if "ZONE" in v else "trip"


# Fixed node positions: calendar on top, rider need in the middle, geography below.
NODE_POS = {"DOW": (0, 3), "SEASON": (2, 3), "HOLIDAY_WEEK": (4, 3),
            "PURPOSE": (1, 1.9), "MOBILITY_AID": (3, 1.9),
            "ORIG_ZONE": (0, 0.6), "DISTANCE": (2, 0), "DEST_ZONE": (4, 0.6)}


def fig_structure(dag, freq, out_png):
    G = nx.DiGraph(dag.edges())
    G.add_nodes_from(VARIABLES)
    pos = {n: NODE_POS.get(n, (2, -1)) for n in G.nodes()}
    fig, ax = plt.subplots(figsize=(13, 8.5))
    widths = [1 + 4 * freq.get(tuple(sorted(e)), 0) for e in G.edges()]
    nx.draw_networkx_edges(G, pos, ax=ax, width=widths, edge_color="#555", arrowsize=18,
                           node_size=4200, connectionstyle="arc3,rad=0.08")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=4200,
                           node_color=[GROUP_COLORS[group(n)] for n in G.nodes()], alpha=0.9)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_color="white", font_weight="bold",
                            labels={n: n.replace("_", "\n") for n in G.nodes()})
    if freq:
        nx.draw_networkx_edge_labels(G, pos, ax=ax, font_size=7,
                                     edge_labels={e: f"{100 * freq.get(tuple(sorted(e)), 0):.0f}%"
                                                  for e in G.edges()})
    for g, c in GROUP_COLORS.items():
        ax.scatter([], [], color=c, s=120, label=g)
    ax.legend(loc="lower left", fontsize=9, frameon=False)
    ax.set_title("Learned Bayesian network (BIC hill climbing). Edge label = % of "
                 f"{N_BOOT} bootstrap re-learns containing the link", fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_png}")
    plt.show()


def heat(ax, tab, title):
    im = ax.imshow(tab.to_numpy(float), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(tab.shape[1]), tab.columns, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(tab.shape[0]), tab.index, fontsize=8)
    for r in range(tab.shape[0]):
        for c in range(tab.shape[1]):
            v = tab.iat[r, c]
            ax.text(c, r, f"{100 * v:.0f}", ha="center", va="center", fontsize=7,
                    color="white" if v > 0.55 else "black")
    ax.set_title(title, loc="left", fontsize=10)
    return im


def fig_queries(queries, out_png):
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    for ax, (name, tab) in zip(axes.flat, queries.items()):
        im = heat(ax, tab, name + "  (% of row)")
    fig.colorbar(im, ax=axes, shrink=0.6, label="probability")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_png}")
    plt.show()


def fig_evaluation(ev, out_png):
    e = ev[ev["TARGET"].isin(["PURPOSE", "MOBILITY_AID"])]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2))
    x = np.arange(len(e))
    for ax, (bn, base, title) in zip((a1, a2), [("BN_ACCURACY", "BASELINE_ACCURACY", "Accuracy (higher is better)"),
                                                ("BN_LOG_LOSS", "BASELINE_LOG_LOSS", "Log loss (lower is better)")]):
        ax.bar(x - 0.18, e[base], 0.36, color="#bbbbbb", label="baseline")
        ax.bar(x + 0.18, e[bn], 0.36, color="#2a6fdb", label="Bayesian network")
        for i, (b, n) in enumerate(zip(e[base], e[bn])):
            ax.text(i - 0.18, b, f"{b:.2f}", ha="center", va="bottom", fontsize=8)
            ax.text(i + 0.18, n, f"{n:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x, e["TARGET"])
        ax.set_title(title, loc="left", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    a1.legend(fontsize=8)
    fig.suptitle(f"Test on unseen months (trips from {TEST_FROM})", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_png}")
    plt.show()


# ============================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    out = lambda n: os.path.join(OUT_DIR, n)
    rng = np.random.default_rng(SEED)

    trips = pd.read_csv(find(TRIPS_PATH, "rockingham_fy24_trips_clean*.csv"))
    data, zones = build_dataset(trips)
    data.to_csv(out("bn_dataset.csv"), index=False)
    train, test = data[data["DATE"] < TEST_FROM], data[data["DATE"] >= TEST_FROM]
    states = ordered_states(data)
    print(f"Trips used: {len(data)} (train {len(train)}, test {len(test)}, from {TEST_FROM})")
    print("\nZones (k-means centres):")
    print(zones.to_string(index=False))

    dag = learn_structure(train)
    model = fit(dag, train, states)
    print(f"\nLearned {len(dag.edges())} arrows:")

    freq = bootstrap_edges(train, N_BOOT, rng) if N_BOOT else {}
    edges = pd.DataFrame([(u, v, freq.get(tuple(sorted((u, v))), np.nan)) for u, v in dag.edges()],
                         columns=["FROM", "TO", "BOOTSTRAP_FREQ"]).sort_values("BOOTSTRAP_FREQ", ascending=False)
    print(edges.to_string(index=False))
    edges.to_csv(out("bn_edges.csv"), index=False)
    with open(out("bn_cpds.txt"), "w") as fh:
        for cpd in model.get_cpds():
            fh.write(str(cpd) + "\n\n")

    print("\nEvaluating on test months ...")
    ev = evaluate(model, train, test, states)
    ev.to_csv(out("bn_evaluation.csv"), index=False)
    print(ev.round(3).to_string(index=False))

    queries = {"P(mobility aid | purpose)": query_table(model, "MOBILITY_AID", "PURPOSE"),
               "P(distance | purpose)": query_table(model, "DISTANCE", "PURPOSE"),
               "P(purpose | day of week)": query_table(model, "PURPOSE", "DOW"),
               "P(purpose | drop-off zone)": query_table(model, "PURPOSE", "DEST_ZONE")}
    for name, tab in queries.items():
        slug = name.split("(")[1].split(")")[0].replace(" | ", "_given_").replace(" ", "_")
        tab.to_csv(out(f"bn_query_{slug}.csv"))
    print("\nP(mobility aid | purpose):")
    print((100 * queries["P(mobility aid | purpose)"]).round(1).to_string())

    fig_structure(dag, freq, out("bn_structure.png"))
    fig_queries(queries, out("bn_queries.png"))
    fig_evaluation(ev, out("bn_evaluation.png"))
    print(f"\nOutputs in: {os.path.abspath(OUT_DIR)}")
    return model, data, ev


if __name__ == "__main__":
    model, bn_data, evaluation = main()
