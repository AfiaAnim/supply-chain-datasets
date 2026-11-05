"""
Healthcare transportation in Rockingham County, FY2024: five report figures.

    Fig 1  Where healthcare rides originate: clinic-bound pickups by census tract with
           clinic destinations (left); households without a vehicle by tract with the same
           pickups as tract-level circles, overlap tracts outlined (right). Falls back to
           pickup areas + hexagons if geopandas or the census files are unavailable.
    Fig 2  Healthcare transportation across FY2024: monthly trip legs (stacked bars) and
           each purpose's trips per service day (lines), as two separate panels.
    Fig 3  Bayesian network structure for a recorded mobility aid, with the assumed link
           and the links learned from the data shown differently.
    Fig 4  Estimated probability of a recorded mobility aid, dialysis vs other medical:
           posterior medians, 95% credible intervals, observed counts.
    Fig 5  Where to review accessible service: weekday x pickup-area heatmap of
           clinic-bound trips, with recorded-aid counts in well-populated cells.

DEFINITIONS (read before quoting numbers)
    Healthcare trip      TRIP_PURPOSE "Dialysis" or "Medical" ("Other medical" below).
    Clinic (inferred)    a drop-off point (coordinates rounded to 3 dp, ~100 m) that
                         receives healthcare trips from at least MIN_DISTINCT_ORIGINS
                         different pickup points that are not themselves clinics,
                         iterated until stable. This separates clinics from homes and
                         from residences with many riders. Points within SITE_MERGE_KM
                         are merged into one site. Clinics are NOT verified against a
                         facility list: see clinic_sites_inferred.csv. Copy it to
                         clinic_sites_verified.csv, fill NAME and set VERIFIED = yes for
                         checked sites, and re-run: the map then marks those as verified.
    Clinic-bound leg     a healthcare trip that ends at a clinic and does not start at
                         one (the outbound leg; the return leg is not counted twice).
    Recorded aid         MOBILITY_AID field is not blank. A blank may mean "no aid" or
                         "not recorded"; the file does not say, hence "recorded".
    Service day          a day with at least one trip of that purpose.
    Pickup area          k-means cluster of clinic-bound pickup locations (N_AREAS),
                         named after a town within 5 km of its centre when there is one.

UNCERTAINTY (Fig 4)
    Bayesian network posterior: Dirichlet posterior of every probability table (BDeu
    prior + counts), N_DRAWS draws pushed through the network. This treats trip legs
    as independent, but the same riders travel many times, so it is too narrow.
    Repeat-rider check: a Bayesian bootstrap over pickup locations (a stand-in for
    riders: a home may hold several riders, and a rider may move). Its interval is the
    more honest one to quote.

VEHICLE AVAILABILITY (Fig 1 right)
    ACS 2019-2023 5-year table B25044: households with no vehicle (owner + renter) / all
    households, by 2020 census tract, with margins of error (fig1_tracts.csv). Overlap
    tracts = above the county median on BOTH clinic-bound pickups and no-vehicle share.

Outputs (OUT_DIR): fig1_origins_map.png ... fig5_review_heatmap.png, plus the tables
behind each figure (CSV) and clinic_sites_inferred.csv.

In Jupyter: just run the cell. Requirements:
    pip install "pgmpy>=1.1" pandas numpy scipy matplotlib scikit-learn networkx
"""

import glob
import json
import os
import warnings

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Polygon as MplPolygon
from matplotlib.path import Path as MplPath
from scipy.spatial import Voronoi
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    from pgmpy.causal_discovery import ExpertKnowledge
    from pgmpy.estimators import HillClimbSearch
except ImportError as e:
    raise ImportError(f"This script needs pgmpy 1.1 or newer ({e}).\n"
                      'In Jupyter run:  %pip install "pgmpy>=1.1"   then restart the kernel.') from e

# ============================================================
# USER CONFIGURATION
# ============================================================
TRIPS_PATH = "merged/rockingham_fy24_trips_clean.csv"      # from merge_rockingham_data.py
COUNTY_GEOJSON = "data/rockingham_area_counties.geojson"   # county outlines (in the repo)
COUNTY_URL = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
COUNTY_FIPS = {"37157": "Rockingham", "37081": "Guilford", "37067": "Forsyth", "37169": "Stokes",
               "37033": "Caswell", "51143": "Pittsylvania VA", "51089": "Henry VA", "51590": "Danville VA"}
OUT_DIR = "report_figures"

MIN_DISTINCT_ORIGINS = 10    # clinic rule (see DEFINITIONS)
SITE_MERGE_KM = 0.25         # merge clinic points closer than this into one site
N_AREAS = 8                  # pickup areas
HEX_KM = 1.5                 # hexagon size on the fine map
MIN_CELL_SHOW = 10           # cells with fewer trips are suppressed ("<10")
MIN_CELL_AID = 30            # recorded-aid counts shown only in cells with >= this many trips
N_DRAWS = 2000               # posterior draws
EQUIV_SAMPLE_SIZE = 10       # BDeu prior strength
N_BOOT = 30                  # bootstrap re-learns for edge stability
SEED = 0

# Census (Fig 1 by tract + vehicle availability). Downloaded once into CENSUS_DIR.
CENSUS_DIR = "data/census"
TRACTS_URL = "https://www2.census.gov/geo/tiger/TIGER2023/TRACT/tl_2023_37_tract.zip"
ACS_URL = ("https://www2.census.gov/programs-surveys/acs/summary_file/2023/table-based-SF/"
           "data/5YRData/acsdt5y2023-b25044.dat")       # ACS 2019-2023 5-yr, tenure by vehicles
STATE_COUNTY = "37157"
ORANGES = ["#fde3d6", "#f9c3a8", "#f4a07b", "#ec7c50", "#d95926", "#b4441a", "#8a3313"]

PURPOSES = ["Dialysis", "Other medical"]
DOW_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
DISTANCE_BINS = [0, 3, 10, 25, np.inf]
DISTANCE_LABELS = ["<3 km", "3-10 km", "10-25 km", ">25 km"]
TOWNS = {"Reidsville": (36.354, -79.664), "Eden": (36.488, -79.767),
         "Madison": (36.385, -79.960), "Mayodan": (36.412, -79.970),
         "Stoneville": (36.465, -79.908), "Wentworth": (36.400, -79.774),
         "Greensboro": (36.073, -79.792), "Winston-Salem": (36.100, -80.244),
         "Danville VA": (36.586, -79.395)}

# Palette: validated categorical slots 1-2, one-hue blue sequential ramp, neutral inks.
C = {"Dialysis": "#2a78d6", "Other medical": "#eb6834",
     "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#8a8986",
     "grid": "#e4e3df", "outline": "#b9b8b3", "highlight": "#e34948"}
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]

plt.rcParams.update({"figure.facecolor": C["surface"], "axes.facecolor": C["surface"],
                     "savefig.facecolor": C["surface"], "axes.edgecolor": C["outline"],
                     "axes.labelcolor": C["ink2"], "xtick.color": C["ink2"], "ytick.color": C["ink2"],
                     "text.color": C["ink"], "axes.titlesize": 11, "axes.titleweight": "bold",
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
SX = np.cos(np.radians(36.4))    # km per degree longitude relative to latitude


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
    return None


def km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def town_label(lat, lon):
    near = min(TOWNS, key=lambda t: km(lat, lon, *TOWNS[t]))
    return near if km(lat, lon, *TOWNS[near]) <= 5 else f"{lat:.2f}, {lon:.2f}"


def load_healthcare():
    p = find(TRIPS_PATH, "rockingham_fy24_trips_clean*.csv")
    if not p:
        raise FileNotFoundError("Run merge_rockingham_data.py first or set TRIPS_PATH.")
    t = pd.read_csv(p, parse_dates=["TRIP_DATE"])
    h = t[t["TRIP_PURPOSE"].isin(["Dialysis", "Medical"])].copy()
    h["PURPOSE"] = h["TRIP_PURPOSE"].replace({"Medical": "Other medical"})
    h["RECORDED_AID"] = np.where(h["MOBILITY_AID"].notna(), "Yes", "No")
    h["DOW"] = h["TRIP_DATE"].dt.day_name().str[:3]
    h["MONTH"] = h["TRIP_DATE"].dt.to_period("M").dt.to_timestamp()
    ok = h["ORIG_COORD_OK"] & h["DEST_COORD_OK"]
    for e in ("ORIG", "DEST"):
        h[f"{e}_PT"] = list(zip(h[f"{e}_LAT"].round(3), h[f"{e}_LON"].round(3)))
    return h, ok


def infer_clinics(h):
    """Iterate: clinic = drop-off point with >= MIN_DISTINCT_ORIGINS non-clinic origins."""
    clinics = set()
    for _ in range(20):
        d = h[~h["ORIG_PT"].isin(clinics)].groupby("DEST_PT")["ORIG_PT"].nunique()
        new = set(d[d >= MIN_DISTINCT_ORIGINS].index)
        if new == clinics:
            break
        clinics = new
    return clinics


def clinic_sites(h, clinics):
    """Merge clinic points within SITE_MERGE_KM into sites; summarise each site."""
    pts = sorted(clinics, key=lambda p: -(h["DEST_PT"] == p).sum())
    site_of, centres = {}, []
    for p in pts:
        for i, c in enumerate(centres):
            if km(*p, *c) <= SITE_MERGE_KM:
                site_of[p] = i
                break
        else:
            site_of[p] = len(centres)
            centres.append(p)
    rows = []
    cb = h[h["CLINIC_BOUND"]]
    for i, c in enumerate(centres):
        members = [p for p, s in site_of.items() if s == i]
        g = cb[cb["DEST_PT"].isin(members)]
        rows.append({"SITE_ID": f"S{i + 1:02d}", "LAT": c[0], "LON": c[1], "NEAR": town_label(*c),
                     "CLINIC_BOUND_TRIPS": len(g),
                     "DIALYSIS_SHARE": (g["PURPOSE"] == "Dialysis").mean() if len(g) else np.nan,
                     "DISTINCT_PICKUP_POINTS": g["ORIG_PT"].nunique(),
                     "N_POINTS_MERGED": len(members), "NAME": "", "VERIFIED": "no"})
    sites = pd.DataFrame(rows).sort_values("CLINIC_BOUND_TRIPS", ascending=False)
    sites["TYPE"] = np.where(sites["DIALYSIS_SHARE"] >= 0.5, "Dialysis", "Other medical")
    return sites


def read_verification(sites):
    p = os.path.join(OUT_DIR, "clinic_sites_verified.csv")
    if os.path.exists(p):
        v = pd.read_csv(p, dtype=str).fillna("")
        v = v.set_index("SITE_ID")[["NAME", "VERIFIED"]]
        sites = sites.set_index("SITE_ID")
        sites.update(v)
        sites = sites.reset_index()
        print(f"Read verification for {(sites['VERIFIED'].str.lower() == 'yes').sum()} sites from {p}")
    sites["IS_VERIFIED"] = sites["VERIFIED"].str.lower().eq("yes")
    return sites


def pickup_areas(cb):
    X = cb[["ORIG_LAT", "ORIG_LON"]].to_numpy() * [1, SX]
    model = KMeans(N_AREAS, n_init=10, random_state=SEED).fit(X)
    centres = model.cluster_centers_ / [1, SX]
    counts = np.bincount(model.labels_, minlength=N_AREAS)
    order = np.argsort(-counts)
    names = {}
    for rank, k in enumerate(order, 1):
        names[k] = f"A{rank} {town_label(*centres[k])}"
    areas = pd.DataFrame({"AREA": [names[k] for k in range(N_AREAS)],
                          "LAT": centres[:, 0], "LON": centres[:, 1]}).sort_values("AREA")
    return pd.Series(model.labels_, index=cb.index).map(names), areas


def load_counties():
    p = find(COUNTY_GEOJSON, "rockingham_area_counties.geojson")
    if p:
        g = json.load(open(p))
    else:
        try:   # US Census cartographic county boundaries (via plotly/datasets); saved for next time
            import urllib.request
            print(f"Downloading county outlines from {COUNTY_URL} ...")
            with urllib.request.urlopen(COUNTY_URL, timeout=60) as r:
                allc = json.load(r)
            feats = [f for f in allc["features"] if f["id"] in COUNTY_FIPS]
            for f in feats:
                f["properties"] = {"FIPS": f["id"], "NAME": COUNTY_FIPS[f["id"]]}
            g = {"type": "FeatureCollection", "features": feats}
            os.makedirs(os.path.dirname(COUNTY_GEOJSON) or ".", exist_ok=True)
            json.dump(g, open(COUNTY_GEOJSON, "w"))
        except Exception as e:
            print(f"County outlines not available ({e}): maps are drawn without them.")
            return {}
    out = {}
    for f in g["features"]:
        geom = f["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        out[f["properties"]["NAME"]] = [np.array(poly[0]) for poly in polys]
    return out


# ============================================================
# FIG 1: where healthcare rides originate
# ============================================================
def draw_context(ax, counties, home_name="Rockingham", towns=True):
    for name, polys in counties.items():
        for ring in polys:
            ax.plot(ring[:, 0], ring[:, 1], color=C["outline"] if name != home_name else C["ink2"],
                    lw=0.6 if name != home_name else 1.2, zorder=1)
    for town, (lat, lon) in (TOWNS.items() if towns else []):
        ax.text(lon, lat + 0.012, town, fontsize=7, color=C["muted"], ha="center", va="bottom",
                zorder=5, clip_on=True)


def draw_clinics(ax, sites):
    s = sites[sites["CLINIC_BOUND_TRIPS"] > 0]
    size = 20 + 180 * np.sqrt(s["CLINIC_BOUND_TRIPS"] / s["CLINIC_BOUND_TRIPS"].max())
    for typ in PURPOSES:
        for ver, marker in ((True, "^"), (False, "o")):
            m = (s["TYPE"] == typ) & (s["IS_VERIFIED"] == ver)
            if m.any():
                ax.scatter(s.loc[m, "LON"], s.loc[m, "LAT"], s=size[m], marker=marker,
                           facecolor=C[typ], edgecolor=C["surface"], linewidth=1.5, zorder=6,
                           label=f"{typ} destination ({'verified' if ver else 'inferred'})")


def set_map_extent(ax, lats, lons, pad=0.04):
    ax.set_xlim(np.nanmin(lons) - pad, np.nanmax(lons) + pad)
    ax.set_ylim(np.nanmin(lats) - pad, np.nanmax(lats) + pad)
    ax.set_aspect(1 / SX)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(""); ax.set_ylabel("")
    for s in ax.spines.values():
        s.set_visible(False)


def voronoi_cells(centres_lonlat):
    """Finite Voronoi polygons (lon/lat) for each centre, bounded by far dummy points."""
    P = centres_lonlat * [SX, 1]
    far = P.mean(axis=0) + np.array([[-50, -50], [-50, 50], [50, -50], [50, 50]])
    vor = Voronoi(np.vstack([P, far]))
    cells = []
    for i in range(len(P)):
        region = vor.regions[vor.point_region[i]]
        poly = vor.vertices[region] / [SX, 1]
        cells.append(poly)
    return cells


def fig1(cb, areas, sites, counties, out_png):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.4))
    home = counties.get("Rockingham", [None])[0]
    in_cb = cb[cb["ORIG_COORD_OK"]]

    # Left: pickup areas (Voronoi cells clipped to the county)
    ax = axes[0]
    counts = in_cb.groupby("AREA").size().reindex(areas["AREA"]).fillna(0).to_numpy()
    cells = voronoi_cells(areas[["LON", "LAT"]].to_numpy())
    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("blues", BLUES)
    norm = plt.matplotlib.colors.Normalize(0, counts.max())
    clip = None
    if home is not None:
        clip = MplPolygon(home, closed=True, transform=ax.transData, facecolor="none")
        ax.add_patch(clip)
    for (_, a), cell, n in zip(areas.iterrows(), cells, counts):
        patch = MplPolygon(cell, closed=True, facecolor=cmap(norm(n)), edgecolor=C["surface"],
                           linewidth=2, zorder=2)
        ax.add_patch(patch)
        if clip is not None:
            patch.set_clip_path(clip)
        inside = home is None or MplPath(home).contains_point((a["LON"], a["LAT"]))
        label = f"{a['AREA']}\n{int(n):,} pickups" if inside else None
        if label:
            ax.text(a["LON"], a["LAT"], label, ha="center", va="center", fontsize=7.5,
                    color="white" if norm(n) > 0.55 else C["ink"], zorder=7, fontweight="bold",
                    clip_on=True)
    draw_context(ax, counties, towns=False)
    outside = 0 if home is None else (~MplPath(home).contains_points(in_cb[["ORIG_LON", "ORIG_LAT"]].to_numpy())).sum()
    ax.set_title("Clinic-bound pickups by pickup area (labels: area, count)", loc="left")
    ax.text(0, -0.02, f"{len(in_cb):,} clinic-bound trip legs FY2024; {outside:,} picked up outside "
            "the county (not shaded).", transform=ax.transAxes, fontsize=8, color=C["ink2"], va="top")
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.6,
                 label="clinic-bound pickups (trip legs)")

    # Right: hexagon aggregation with small-cell suppression
    ax = axes[1]
    span_lon = in_cb["ORIG_LON"].max() - in_cb["ORIG_LON"].min()
    span_lat = in_cb["ORIG_LAT"].max() - in_cb["ORIG_LAT"].min()
    nx_ = max(5, int(round(span_lon * 111 * SX / HEX_KM)))
    ny_ = max(3, int(round(span_lat * 111 / (HEX_KM * np.sqrt(3)))))
    hb = ax.hexbin(in_cb["ORIG_LON"], in_cb["ORIG_LAT"], gridsize=(nx_, ny_),
                   mincnt=MIN_CELL_SHOW, cmap=cmap, linewidths=0.6, edgecolors=C["surface"], zorder=2)
    draw_context(ax, counties)
    draw_clinics(ax, sites)
    ax.set_title(f"Clinic-bound pickups per ~{HEX_KM:g} km hexagon", loc="left")
    ax.text(0, -0.02, f"Hexagons with fewer than {MIN_CELL_SHOW} trip legs are not shown. "
            "No individual pickup points are plotted.", transform=ax.transAxes, fontsize=8,
            color=C["ink2"], va="top")
    fig.colorbar(hb, ax=ax, shrink=0.6, label="clinic-bound pickups (trip legs)")

    ref = in_cb if home is None else pd.DataFrame({"ORIG_LAT": home[:, 1], "ORIG_LON": home[:, 0]})
    for a in axes:
        set_map_extent(a, ref["ORIG_LAT"], ref["ORIG_LON"])
    handles, labels = axes[1].get_legend_handles_labels()
    if not handles:
        handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8,
               markerscale=0.7, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Where healthcare rides originate (FY2024)", fontsize=13, fontweight="bold", x=0.01, ha="left")
    fig.text(0.01, 0.915, "Destinations are inferred from trip patterns (marker size = clinic-bound "
             "arrivals) and are not verified against a facility list unless marked verified. "
             "Vehicle-availability panel not included: tract data not available.",
             fontsize=8, color=C["ink2"])
    fig.tight_layout(rect=(0, 0.04, 1, 0.9))
    save(fig, out_png)



# ============================================================
# FIG 1 (tracts): pickups by tract + households without vehicles
# ============================================================
def census_file(url):
    import urllib.request
    os.makedirs(CENSUS_DIR, exist_ok=True)
    path = os.path.join(CENSUS_DIR, os.path.basename(url))
    if not os.path.exists(path):
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, path + ".part")
        os.replace(path + ".part", path)
    return path


def acs_no_vehicle():
    """Households and households with no vehicle per tract (ACS B25044), with MOEs."""
    cache = os.path.join(CENSUS_DIR, f"acs2023_b25044_tracts_{STATE_COUNTY}.csv")
    if not os.path.exists(cache):
        dat, prefix, rows = census_file(ACS_URL), f"1400000US{STATE_COUNTY}", []
        with open(dat) as fh:
            header = fh.readline().strip().split("|")
            rows = [ln.strip().split("|") for ln in fh if ln.startswith(prefix)]
        a = pd.DataFrame(rows, columns=header)
        num = lambda c: pd.to_numeric(a[c], errors="coerce")
        pd.DataFrame({"TRACT": a["GEO_ID"].str.replace("1400000US", "", regex=False),
                      "HOUSEHOLDS": num("B25044_E001"), "HOUSEHOLDS_MOE": num("B25044_M001"),
                      "NO_VEHICLE_HH": num("B25044_E003") + num("B25044_E010"),
                      "NO_VEHICLE_HH_MOE": np.sqrt(num("B25044_M003") ** 2 + num("B25044_M010") ** 2)}
                     ).to_csv(cache, index=False)
        os.remove(dat)                                   # ~60 MB national file; keep the extract
    acs = pd.read_csv(cache, dtype={"TRACT": str})
    acs["NO_VEHICLE_SHARE"] = acs["NO_VEHICLE_HH"] / acs["HOUSEHOLDS"]
    # ACS approximation for the MOE of a proportion
    p = acs["NO_VEHICLE_SHARE"]
    inner = acs["NO_VEHICLE_HH_MOE"] ** 2 - (p ** 2) * acs["HOUSEHOLDS_MOE"] ** 2
    inner = inner.where(inner > 0, acs["NO_VEHICLE_HH_MOE"] ** 2 + (p ** 2) * acs["HOUSEHOLDS_MOE"] ** 2)
    acs["NO_VEHICLE_SHARE_MOE"] = np.sqrt(inner) / acs["HOUSEHOLDS"]
    return acs


def fig1_tracts(cb, sites, counties, out_png, out_csv):
    import geopandas as gpd
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    tr = gpd.read_file(census_file(TRACTS_URL))
    tr = tr[tr["COUNTYFP"] == STATE_COUNTY[2:]][["GEOID", "geometry"]].rename(columns={"GEOID": "TRACT"})
    tr = tr.to_crs("EPSG:4326")

    pts = cb[cb["ORIG_COORD_OK"]]
    g = gpd.GeoDataFrame(index=pts.index, crs="EPSG:4326",
                         geometry=gpd.points_from_xy(pts["ORIG_LON"], pts["ORIG_LAT"]))
    j = gpd.sjoin(g, tr, how="left", predicate="within")
    j = j[~j.index.duplicated()]
    tract_of = j["TRACT"].reindex(pts.index)
    per = pts.assign(TRACT=tract_of).groupby("TRACT").agg(
        CLINIC_BOUND_PICKUPS=("PURPOSE", "size"),
        DIALYSIS=("PURPOSE", lambda s: (s == "Dialysis").sum()),
        WITH_RECORDED_AID=("RECORDED_AID", lambda s: (s == "Yes").sum()))
    outside = int(tract_of.isna().sum())

    t = tr.merge(acs_no_vehicle(), on="TRACT", how="left").merge(per, on="TRACT", how="left")
    t[["CLINIC_BOUND_PICKUPS", "DIALYSIS", "WITH_RECORDED_AID"]] = \
        t[["CLINIC_BOUND_PICKUPS", "DIALYSIS", "WITH_RECORDED_AID"]].fillna(0).astype(int)
    t["PICKUPS_PER_100_HH"] = 100 * t["CLINIC_BOUND_PICKUPS"] / t["HOUSEHOLDS"]
    med_p, med_v = t["CLINIC_BOUND_PICKUPS"].median(), t["NO_VEHICLE_SHARE"].median()
    t["OVERLAP"] = (t["CLINIC_BOUND_PICKUPS"] > med_p) & (t["NO_VEHICLE_SHARE"] > med_v)
    t.drop(columns="geometry").sort_values("CLINIC_BOUND_PICKUPS", ascending=False).to_csv(out_csv, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.8))
    blues = LinearSegmentedColormap.from_list("blues", BLUES)
    oranges = LinearSegmentedColormap.from_list("oranges", ORANGES)

    # Left: pickups per tract (counts under MIN_CELL_SHOW are drawn as "<10", same pale fill)
    ax = axes[0]
    shown = t["CLINIC_BOUND_PICKUPS"].where(t["CLINIC_BOUND_PICKUPS"] >= MIN_CELL_SHOW, 0)
    norm_p = Normalize(0, shown.max())
    t.assign(V=shown).plot(ax=ax, column="V", cmap=blues, norm=norm_p, edgecolor=C["surface"], linewidth=1.6, zorder=2)
    for _, r in t.nlargest(6, "CLINIC_BOUND_PICKUPS").iterrows():
        pt = r.geometry.representative_point()
        ax.text(pt.x, pt.y, f"{r['CLINIC_BOUND_PICKUPS']:,}", ha="center", va="center", fontsize=7.5,
                fontweight="bold", color=C["ink"], zorder=8,
                bbox=dict(boxstyle="round,pad=0.2", fc=C["surface"], ec="none", alpha=0.85))
    draw_context(ax, counties, towns=False)
    draw_clinics(ax, sites)
    ax.set_title("Clinic-bound pickups by census tract (6 largest labelled)", loc="left")
    ax.text(0, -0.13, f"{int(t['CLINIC_BOUND_PICKUPS'].sum()):,} clinic-bound trip legs in {len(t)} tracts; "
            f"{outside} picked up outside the county. No individual pickups are shown.",
            transform=ax.transAxes, fontsize=8, color=C["ink2"], va="top")
    fig.colorbar(plt.cm.ScalarMappable(norm=norm_p, cmap=blues), ax=ax, shrink=0.6,
                 label="clinic-bound pickups (trip legs), FY2024")

    # Right: no-vehicle share + pickups as circles at tract centres; overlap tracts outlined
    ax = axes[1]
    norm_v = Normalize(0, t["NO_VEHICLE_SHARE"].max())
    t.plot(ax=ax, column="NO_VEHICLE_SHARE", cmap=oranges, norm=norm_v, edgecolor=C["surface"],
           linewidth=1.6, zorder=2)
    t[t["OVERLAP"]].boundary.plot(ax=ax, color=C["ink"], linewidth=2.2, zorder=4)
    rp = t.geometry.representative_point()
    size = lambda n: 12 + 520 * n / t["CLINIC_BOUND_PICKUPS"].max()
    big = t["CLINIC_BOUND_PICKUPS"] >= MIN_CELL_SHOW
    ax.scatter(rp.x[big], rp.y[big], s=size(t.loc[big, "CLINIC_BOUND_PICKUPS"]), facecolor=C["Dialysis"],
               alpha=0.55, edgecolor=C["surface"], linewidth=1.2, zorder=5)
    draw_context(ax, counties)
    for n in (100, 500, 1500):
        ax.scatter([], [], s=size(n), facecolor=C["Dialysis"], alpha=0.55, edgecolor=C["surface"],
                   label=f"{n:,} pickups")
    ax.plot([], [], color=C["ink"], lw=2.2, label="above county median on both")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=4, frameon=False, fontsize=7.5,
              columnspacing=1.6)
    ax.set_title("Households without a vehicle, with clinic-bound pickups", loc="left")
    ax.text(0, -0.13, f"ACS 2019-2023 5-year (B25044), 2020 tracts; county median {med_v:.1%}. "
            "Margins of error in fig1_tracts.csv.", transform=ax.transAxes, fontsize=8,
            color=C["ink2"], va="top")
    cb_ = fig.colorbar(plt.cm.ScalarMappable(norm=norm_v, cmap=oranges), ax=ax, shrink=0.6,
                       label="share of households with no vehicle")
    cb_.formatter = plt.matplotlib.ticker.PercentFormatter(1)
    cb_.update_ticks()

    bounds = t.total_bounds
    for a in axes:
        set_map_extent(a, [bounds[1], bounds[3]], [bounds[0], bounds[2]], pad=0.02)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=2, frameon=False, fontsize=7.5,
                   markerscale=0.7)
    fig.suptitle("Where healthcare rides originate (FY2024)", fontsize=13, fontweight="bold", x=0.01, ha="left")
    fig.text(0.01, 0.915, "Clinic destinations are inferred from trip patterns (marker size = clinic-bound "
             "arrivals) and are not verified against a facility list unless marked verified.",
             fontsize=8, color=C["ink2"])
    fig.tight_layout(rect=(0, 0.04, 1, 0.9))
    save(fig, out_png)
    ov = t[t["OVERLAP"]].sort_values("CLINIC_BOUND_PICKUPS", ascending=False)
    print(f"Overlap tracts (above median on pickups and no-vehicle share): {len(ov)}")
    print(ov[["TRACT", "CLINIC_BOUND_PICKUPS", "NO_VEHICLE_SHARE", "NO_VEHICLE_SHARE_MOE"]]
          .round(3).to_string(index=False))
    return t

# ============================================================
# FIG 2: healthcare transportation across FY2024
# ============================================================
def fig2(h, out_png, out_csv):
    m = h.groupby(["MONTH", "PURPOSE"]).size().unstack(fill_value=0).reindex(columns=PURPOSES)
    days = h.groupby(["MONTH", "PURPOSE"])["TRIP_DATE"].nunique().unstack(fill_value=0).reindex(columns=PURPOSES)
    per_day = m / days
    table = pd.concat({"trip_legs": m, "service_days": days, "legs_per_service_day": per_day.round(1)}, axis=1)
    table.to_csv(out_csv)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [1.15, 1]})
    x = np.arange(len(m))
    bottom = np.zeros(len(m))
    for p in PURPOSES:
        a1.bar(x, m[p], 0.7, bottom=bottom, color=C[p], edgecolor=C["surface"], linewidth=2, label=p)
        bottom += m[p].to_numpy()
    for i, tot in enumerate(bottom):
        a1.text(i, tot + 20, f"{int(tot):,}", ha="center", va="bottom", fontsize=8, color=C["ink2"])
    a1.set_ylabel("trip legs per month")
    a1.set_title("Healthcare trip legs per month", loc="left")
    a1.legend(frameon=False, loc="lower right", bbox_to_anchor=(1, 1.0), ncol=2)
    a1.set_ylim(0, bottom.max() * 1.1)
    a1.grid(axis="y", color=C["grid"], lw=0.8)
    a1.set_axisbelow(True)

    for p in PURPOSES:
        a2.plot(x, per_day[p], color=C[p], lw=2, marker="o", ms=6, label=p)
        a2.text(x[-1] + 0.15, per_day[p].iloc[-1], p, color=C["ink"], va="center", fontsize=8)
    a2.set_ylabel("trip legs per service day")
    a2.set_title("Trips per service day (each purpose's own service days)", loc="left")
    a2.grid(axis="y", color=C["grid"], lw=0.8)
    a2.set_ylim(0, per_day.max().max() * 1.15)
    a2.set_xticks(x, [d.strftime("%b\n%Y") for d in m.index])
    a2.set_xlim(-0.6, len(x) - 0.1)
    fig.suptitle("Healthcare transportation across FY2024", fontsize=13, fontweight="bold", x=0.01, ha="left")
    fig.text(0.01, 0.005, "Trip legs count both directions. Service day = a day with at least one "
             "trip of that purpose (dialysis also runs Saturdays). Table: fig2_monthly_table.csv",
             fontsize=8, color=C["ink2"])
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    save(fig, out_png)
    return table


# ============================================================
# BAYESIAN NETWORK (Figs 3-4)
# ============================================================
BN_VARS = ["DOW", "SEASON", "PURPOSE", "PICKUP_AREA", "DISTANCE", "RECORDED_AID"]
CALENDAR = ["DOW", "SEASON"]
TARGET = "RECORDED_AID"
ASSUMED = [("PURPOSE", "RECORDED_AID")]


def season(month):
    return {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
            6: "Summer", 7: "Summer", 8: "Summer"}.get(month, "Fall")


def bn_data(cb):
    d = pd.DataFrame({"DOW": cb["DOW"], "SEASON": cb["TRIP_DATE"].dt.month.map(season),
                      "PURPOSE": cb["PURPOSE"], "PICKUP_AREA": cb["AREA"],
                      "DISTANCE": pd.cut(km(cb["ORIG_LAT"], cb["ORIG_LON"], cb["DEST_LAT"], cb["DEST_LON"]),
                                         DISTANCE_BINS, labels=DISTANCE_LABELS, right=False).astype(str),
                      "RECORDED_AID": cb["RECORDED_AID"]})
    states = {v: sorted(d[v].unique()) for v in BN_VARS}
    states["DOW"] = [x for x in DOW_ORDER if x in set(d["DOW"])]
    states["DISTANCE"] = [x for x in DISTANCE_LABELS if x in set(d["DISTANCE"])]
    states["RECORDED_AID"] = ["No", "Yes"]
    states["PURPOSE"] = PURPOSES
    return d, states


def learn(d):
    forbidden = [(v, c) for c in CALENDAR for v in BN_VARS if v != c]       # calendar = roots
    forbidden += [(TARGET, v) for v in BN_VARS if v != TARGET]               # outcome = sink
    ek = ExpertKnowledge(forbidden_edges=forbidden, required_edges=ASSUMED)
    return HillClimbSearch(d[BN_VARS]).estimate(scoring_method="bic-d", max_indegree=3,
                                                expert_knowledge=ek, show_progress=False)


def edge_stability(d, rng):
    counts = {}
    for b in range(N_BOOT):
        s = d.sample(len(d), replace=True, random_state=int(rng.integers(1e9)))
        for e in learn(s).edges():                       # direction-free: a link either way
            k = tuple(sorted(e))
            counts[k] = counts.get(k, 0) + 1
        print(f"\r  bootstrap {b + 1}/{N_BOOT}", end="")
    print()
    return {e: c / N_BOOT for e, c in counts.items()}


def posterior_draws(d, states, parents, rng):
    """Dirichlet posterior draws of every CPT (BDeu prior); returns P(aid=Yes | purpose) draws."""
    codes = {v: d[v].map({s: i for i, s in enumerate(states[v])}).to_numpy() for v in BN_VARS}
    letters = {v: chr(97 + i) for i, v in enumerate(BN_VARS)}
    tables = {}
    for v in BN_VARS:
        pa = parents[v]
        shape = [len(states[v])] + [len(states[p]) for p in pa]
        counts = np.zeros(shape)
        np.add.at(counts, tuple(codes[x] for x in [v] + pa), 1)
        alpha = EQUIV_SAMPLE_SIZE / np.prod(shape)
        tables[v] = (counts + alpha, pa)
    out = np.zeros((N_DRAWS, len(PURPOSES)))
    spec = ",".join(letters[v] + "".join(letters[p] for p in pa) for v, (_, pa) in tables.items())
    result = letters["PURPOSE"] + letters[TARGET]
    for k in range(N_DRAWS):
        arrays = []
        for v, (a, pa) in tables.items():
            g = rng.gamma(a)
            arrays.append(g / g.sum(axis=0, keepdims=True))
        joint = np.einsum(f"{spec}->{result}", *arrays)          # P(purpose, aid)
        out[k] = joint[:, 1] / joint.sum(axis=1)
    return out


def rider_bootstrap(cb, rng):
    """Bayesian bootstrap over pickup locations (stand-in for riders)."""
    out = np.zeros((N_DRAWS, len(PURPOSES)))
    for j, p in enumerate(PURPOSES):
        g = cb[cb["PURPOSE"] == p].groupby("ORIG_PT")["RECORDED_AID"].agg(
            n="size", y=lambda s: (s == "Yes").sum())
        w = rng.dirichlet(np.ones(len(g)), size=N_DRAWS)
        out[:, j] = (w @ g["y"].to_numpy()) / (w @ g["n"].to_numpy())
    return out


def summarise(draws):
    return np.median(draws, axis=0), np.percentile(draws, 2.5, axis=0), np.percentile(draws, 97.5, axis=0)


# ============================================================
# FIG 3: network structure
# ============================================================
NODE_POS = {"DOW": (0, 2.1), "SEASON": (2.6, 2.1), "PURPOSE": (0, 1), "PICKUP_AREA": (2.6, 1),
            "DISTANCE": (2.6, -0.1), "RECORDED_AID": (1.1, -0.9)}
NODE_LABEL = {"DOW": "Day of week", "SEASON": "Season", "PURPOSE": "Trip purpose\n(dialysis /\nother medical)",
              "PICKUP_AREA": "Pickup area", "DISTANCE": "Trip distance",
              "RECORDED_AID": "Recorded\nmobility aid"}


def fig3(dag, freq, out_png):
    fig, ax = plt.subplots(figsize=(11, 8))
    r = 0.34
    for v, (x, y) in NODE_POS.items():
        is_t = v == TARGET
        ax.add_patch(plt.Circle((x, y), r * (1.15 if is_t else 1), facecolor=C["highlight"] if is_t else "#eef3fb",
                                edgecolor=C["highlight"] if is_t else C["Dialysis"], lw=2.5 if is_t else 1.2, zorder=3))
        ax.text(x, y, NODE_LABEL[v], ha="center", va="center", fontsize=8.5, zorder=4,
                color="white" if is_t else C["ink"], fontweight="bold")
    for u, v in dag.edges():
        assumed = (u, v) in ASSUMED
        (x1, y1), (x2, y2) = NODE_POS[u], NODE_POS[v]
        arr = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=18,
                              lw=2.2, color=C["Other medical"] if assumed else C["ink2"],
                              linestyle=(0, (5, 3)) if assumed else "-", shrinkA=34, shrinkB=40,
                              connectionstyle="arc3,rad=0.12", zorder=2)
        ax.add_patch(arr)
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        lab = "assumed" if assumed else f"learned · {100 * freq.get(tuple(sorted((u, v))), 0):.0f}%"
        ax.text(mx + 0.05, my + 0.04, lab, fontsize=8, color=C["ink2"],
                bbox=dict(boxstyle="round,pad=0.2", fc=C["surface"], ec="none"), zorder=5)
    ax.plot([], [], color=C["Other medical"], lw=2.2, ls=(0, (5, 3)), label="assumed link (set by the analyst)")
    ax.plot([], [], color=C["ink2"], lw=2.2, label=f"learned link (% of {N_BOOT} bootstrap re-learns\n"
                                                   "with this link, either direction)")
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    ax.set_xlim(-0.7, 3.6); ax.set_ylim(-1.45, 2.55); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("Bayesian network for a recorded mobility aid (clinic-bound trip legs)",
                 loc="left", fontsize=12)
    ax.text(-0.7, -1.62, "Constraints, not links: day of week and season may only point outward; "
            "the recorded aid may only receive arrows. Learned links: hill-climbing search with "
            "the BIC score.", fontsize=7.5, color=C["ink2"], wrap=True)
    save(fig, out_png)


# ============================================================
# FIG 4: estimated probability of a recorded aid
# ============================================================
def fig4(cb, bn_draws, rb_draws, out_png, out_csv):
    obs = cb.groupby("PURPOSE").agg(n=("RECORDED_AID", "size"), y=("RECORDED_AID", lambda s: (s == "Yes").sum()),
                                    locations=("ORIG_PT", "nunique")).reindex(PURPOSES)
    bm, bl, bh = summarise(bn_draws)
    rm, rl, rh = summarise(rb_draws)
    diff_bn, diff_rb = bn_draws[:, 0] - bn_draws[:, 1], rb_draws[:, 0] - rb_draws[:, 1]
    rows = []
    for j, p in enumerate(PURPOSES):
        rows.append({"PURPOSE": p, "TRIP_LEGS": obs.loc[p, "n"], "WITH_RECORDED_AID": obs.loc[p, "y"],
                     "PICKUP_LOCATIONS": obs.loc[p, "locations"], "OBSERVED": obs.loc[p, "y"] / obs.loc[p, "n"],
                     "BN_MEDIAN": bm[j], "BN_CI_LOW": bl[j], "BN_CI_HIGH": bh[j],
                     "RIDER_ADJ_MEDIAN": rm[j], "RIDER_ADJ_LOW": rl[j], "RIDER_ADJ_HIGH": rh[j]})
    for name, dd in (("DIFFERENCE (dialysis - other medical), BN", diff_bn),
                     ("DIFFERENCE (dialysis - other medical), rider-adjusted", diff_rb)):
        rows.append({"PURPOSE": name, "BN_MEDIAN": np.median(dd), "BN_CI_LOW": np.percentile(dd, 2.5),
                     "BN_CI_HIGH": np.percentile(dd, 97.5)})
    summary = pd.DataFrame(rows)
    summary.to_csv(out_csv, index=False)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 5.6), gridspec_kw={"height_ratios": [2, 1]})
    for j, p in enumerate(PURPOSES):
        y = 1 - j
        a1.plot([rl[j], rh[j]], [y - 0.12] * 2, color=C[p], lw=2, alpha=0.55, solid_capstyle="round")
        a1.plot([bl[j], bh[j]], [y + 0.12] * 2, color=C[p], lw=5, solid_capstyle="round")
        a1.scatter([bm[j]], [y + 0.12], s=22, color=C["surface"], edgecolor=C[p], linewidth=1.5, zorder=3)
        a1.scatter([rm[j]], [y - 0.12], s=55, color=C[p], alpha=0.55, edgecolor=C["surface"], linewidth=2, zorder=3)
        o = obs.loc[p]
        a1.text(1.01, y, f"{int(o['n']):,} trip legs · {int(o['y']):,} with recorded aid\n"
                f"{int(o['locations']):,} pickup locations · observed {100 * o['y'] / o['n']:.1f}%",
                transform=a1.get_yaxis_transform(), va="center", fontsize=8, color=C["ink2"])
    a1.set_yticks([1, 0], PURPOSES)
    a1.set_ylim(-0.5, 1.5)
    a1.set_xlim(0, 0.8)
    a1.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1))
    a1.grid(axis="x", color=C["grid"], lw=0.8)
    a1.set_axisbelow(True)
    a1.set_title("Probability that a clinic-bound trip leg has a recorded mobility aid", loc="left")
    a1.plot([], [], color=C["ink2"], lw=5, label="Bayesian network: median, 95% credible interval (trip legs independent)")
    a1.plot([], [], color=C["ink2"], lw=2, alpha=0.55, label="repeat-rider check: Bayesian bootstrap over pickup locations")
    a1.legend(loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=1, frameon=False, fontsize=8)

    lo_b, hi_b = np.percentile(diff_bn, [2.5, 97.5])
    lo_r, hi_r = np.percentile(diff_rb, [2.5, 97.5])
    a2.plot([lo_r, hi_r], [-0.12] * 2, color=C["ink2"], lw=2, alpha=0.55)
    a2.plot([lo_b, hi_b], [0.12] * 2, color=C["ink"], lw=5)
    a2.scatter([np.median(diff_bn)], [0.12], s=22, color=C["surface"], edgecolor=C["ink"], linewidth=1.5, zorder=3)
    a2.scatter([np.median(diff_rb)], [-0.12], s=55, color=C["ink2"], alpha=0.55, edgecolor=C["surface"], linewidth=2, zorder=3)
    a2.axvline(0, color=C["muted"], lw=1)
    a2.set_yticks([0], ["Dialysis minus\nother medical"])
    a2.set_ylim(-0.5, 0.5)
    a2.set_xlim(-0.1, 0.5)
    a2.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1))
    a2.grid(axis="x", color=C["grid"], lw=0.8)
    a2.set_title(f"Difference: {100 * np.median(diff_bn):.0f} percentage points "
                 f"(95% CrI {100 * lo_b:.0f} to {100 * hi_b:.0f}; repeat-rider check "
                 f"{100 * lo_r:.0f} to {100 * hi_r:.0f})", loc="left", fontsize=10)
    fig.tight_layout(h_pad=3.5)
    save(fig, out_png)
    return summary


# ============================================================
# FIG 5: where to review accessible service
# ============================================================
def fig5(cb, areas, out_png, out_csv):
    order = list(areas["AREA"])
    n = cb.groupby(["DOW", "AREA"]).size().unstack(fill_value=0).reindex(index=DOW_ORDER, columns=order, fill_value=0)
    a = cb[cb["RECORDED_AID"] == "Yes"].groupby(["DOW", "AREA"]).size().unstack(fill_value=0) \
        .reindex(index=DOW_ORDER, columns=order, fill_value=0)
    cells = pd.concat({"trips": n, "with_recorded_aid": a.where(n >= MIN_CELL_AID)}, axis=1)
    cells.to_csv(out_csv)

    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("blues", BLUES)
    shown = n.where(n >= MIN_CELL_SHOW)
    fig, ax = plt.subplots(figsize=(13, 5.8))
    im = ax.imshow(shown.to_numpy(float), cmap=cmap, aspect="auto", vmin=0)
    norm = im.norm
    for i in range(n.shape[0]):
        for j in range(n.shape[1]):
            v, aid = n.iat[i, j], a.iat[i, j]
            if v < MIN_CELL_SHOW:
                txt, col = (f"<{MIN_CELL_SHOW}" if v > 0 else "0"), C["muted"]
            elif v < MIN_CELL_AID:
                txt, col = f"{v:,}", C["ink"]
            else:
                txt = f"{v:,}\n{aid:,} with aid ({100 * aid / v:.0f}%)"
                col = "white" if norm(v) > 0.55 else C["ink"]
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.5, color=col)
    ax.set_xticks(range(len(order)), [f"{c}\n({n[c].sum():,})" for c in order], fontsize=8)
    ax.set_yticks(range(len(DOW_ORDER)), [f"{d} ({n.loc[d].sum():,})" for d in DOW_ORDER])
    ax.set_xticks(np.arange(-0.5, len(order)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(DOW_ORDER)), minor=True)
    ax.grid(which="minor", color=C["surface"], lw=2)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="clinic-bound trip legs, FY2024")
    ax.set_title("Clinic-bound trips by weekday and pickup area (totals in brackets)", loc="left")
    fig.text(0.01, 0.01, f"Cells with fewer than {MIN_CELL_SHOW} trips are suppressed; recorded-aid "
             f"counts are shown only where a cell has at least {MIN_CELL_AID} trips. "
             "Pickup areas are k-means clusters of pickup locations. Table: fig5_cells.csv", fontsize=8, color=C["ink2"])
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, out_png)
    return cells


def save(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.show()


# ============================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    out = lambda n: os.path.join(OUT_DIR, n)
    rng = np.random.default_rng(SEED)

    h, ok = load_healthcare()
    clinics = infer_clinics(h[ok])
    h["CLINIC_BOUND"] = ok & h["DEST_PT"].isin(clinics) & ~h["ORIG_PT"].isin(clinics)
    cb = h[h["CLINIC_BOUND"]].copy()
    sites = read_verification(clinic_sites(h, clinics))
    sites.drop(columns="IS_VERIFIED").to_csv(out("clinic_sites_inferred.csv"), index=False)
    cb["AREA"], areas = pickup_areas(cb)
    print(f"Healthcare trip legs: {len(h):,}; clinic points: {len(clinics)} -> {len(sites)} sites; "
          f"clinic-bound legs: {len(cb):,}")

    counties = load_counties()
    try:
        fig1_tracts(cb, sites, counties, out("fig1_origins_map.png"), out("fig1_tracts.csv"))
    except Exception as e:                       # no geopandas / no census access
        print(f"Tract map not available ({type(e).__name__}: {e}); drawing pickup-area map instead.")
        fig1(cb, areas, sites, counties, out("fig1_origins_map.png"))
    t2 = fig2(h, out("fig2_monthly.png"), out("fig2_monthly_table.csv"))
    print(t2.to_string())

    d, states = bn_data(cb)
    dag = learn(d)
    freq = edge_stability(d, rng)
    pd.DataFrame([(u, v, "assumed" if (u, v) in ASSUMED else "learned", freq.get(tuple(sorted((u, v))), np.nan))
                  for u, v in dag.edges()], columns=["FROM", "TO", "KIND", "BOOTSTRAP_FREQ"]) \
        .to_csv(out("fig3_edges.csv"), index=False)
    print("Network links:", sorted(dag.edges()))
    fig3(dag, freq, out("fig3_network.png"))

    parents = {v: sorted(dag.get_parents(v)) for v in BN_VARS}
    bn_draws = posterior_draws(d, states, parents, rng)
    rb_draws = rider_bootstrap(cb, rng)
    s4 = fig4(cb, bn_draws, rb_draws, out("fig4_aid_probability.png"), out("fig4_posterior_summary.csv"))
    print(s4.round(3).to_string(index=False))

    fig5(cb, areas, out("fig5_review_heatmap.png"), out("fig5_cells.csv"))
    print(f"\nOutputs in: {os.path.abspath(OUT_DIR)}")
    return h, cb, sites, dag


if __name__ == "__main__":
    healthcare, clinic_bound, clinic_site_table, network = main()
