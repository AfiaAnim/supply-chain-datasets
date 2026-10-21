"""
Flood rerouting: getting residents of flood-prone streets to Hanahan Middle School.

Problem
    Sellers Road and the south end of Dickson Avenue (near Remount Road,
    North Charleston / Hanahan, SC) flood and become impassable. People living
    around them still need to reach Hanahan Middle School (e.g. as a shelter or
    pickup point). Which routes should they take, how much longer are they,
    who is cut off, and which roads matter most?

Method
    1. Download the drivable OpenStreetMap network covering both flood areas and
       the school (OSMnx / Overpass, with retries across mirror servers).
    2. Mark flooded edges as blocked (FLOOD_ZONES: a whole street, part of a
       street near a cross street, or a circle around a coordinate).
    3. Origins = every intersection within --affected-radius-m of a flooded
       road (the neighbourhoods that normally use those roads).
    4. Routing: one reverse Dijkstra from the school gives every origin's
       fastest route, before and after flooding. Origins with no dry route are
       reported as stranded.
    5. Optimization
       a. Congestion-aware assignment: everyone leaving at once loads the same
          detour streets. Edge times are updated with the BPR function
          t = t0 * (1 + 0.15 (v/c)^4) and routes recomputed by the method of
          successive averages, so traffic spreads over parallel streets.
       b. Critical detour roads: streets carrying the most rerouted vehicles
          (where to put traffic control / keep clear).
       c. Mitigation priority: reopen each flood zone on its own (pumps,
          barriers, raising the road) and rank zones by the delay and
          stranding they remove.

Requirements
    pip install osmnx networkx pandas matplotlib

Usage
    python flood_rerouting_hanahan_ms.py
    Jupyter:  %run flood_rerouting_hanahan_ms.py   or paste into a cell; then
              main(["--vehicles-per-origin", 15])
    Options:
              --school-latlon LAT LON         (default: Hanahan Middle School below)
              --radius-m 4500                 (enlarge the download if a street is missing)
              --osm-file map.osm              (use a downloaded OpenStreetMap file, no server needed)
              --flood-circle LAT LON RADIUS_M   (repeatable; adds a flood zone)
"""

import argparse
import datetime as dt
import json
import math
import os
import sys
import time

import networkx as nx
import osmnx as ox
import pandas as pd

SCHOOL_NAME = "Hanahan Middle School"
SCHOOL_LATLON = (32.90752, -79.994)

# Flood zones (circled on the study map). Each zone blocks the edges matching:
#   street            - street name (abbreviations like "Rd" are handled)
#   near, radius_m    - optional: only the part of `street` within radius_m of
#                       its junction with `near`
#   lat, lon, radius_m- alternatively: every road within radius_m of a point
FLOOD_ZONES = [
    {"name": "Sellers Road", "street": "Sellers Road"},
    {"name": "Dickson Avenue (south end)", "street": "Dickson Avenue",
     "near": "Remount Road", "radius_m": 450},
]

# Free-flow speeds (mph) for highway classes without a maxspeed tag.
DEFAULT_SPEEDS_MPH = {
    "motorway": 60, "trunk": 50, "primary": 45, "secondary": 40,
    "tertiary": 35, "unclassified": 30, "residential": 25,
    "living_street": 15, "service": 15,
}
# Planning capacities (vehicles per hour per lane) for the BPR congestion model.
CAPACITY_VPH_PER_LANE = {
    "motorway": 1900, "trunk": 1600, "primary": 1200, "secondary": 900,
    "tertiary": 700, "unclassified": 500, "residential": 500,
    "living_street": 300, "service": 300,
}
BPR_ALPHA, BPR_BETA = 0.15, 4

M_PER_MI = 1609.344
# Roads within this distance of the school are downloaded; it covers Sellers Rd,
# the south end of Dickson Ave and the detour streets between them and the school.
DOWNLOAD_RADIUS_M = 3500
T0 = time.time()


def step(msg):
    print(f"[{time.time() - T0:6.0f} s] {msg}", flush=True)

OVERPASS_URLS = [
    "https://overpass-api.de/api",
    "https://overpass.kumi.systems/api",
    "https://overpass.private.coffee/api",
    "https://maps.mail.ru/osm/tools/overpass/api",
]


# --- helpers ---------------------------------------------------------------

def norm(name):
    """Lower-case a street name and expand common suffix abbreviations."""
    s = str(name).lower().replace(".", "").strip()
    for short, full in ((" rd", " road"), (" dr", " drive"), (" ave", " avenue"),
                        (" av", " avenue"), (" st", " street")):
        if s.endswith(short):
            s = s[: -len(short)] + full
    return s


def edge_names(data):
    names = data.get("name", [])
    if not isinstance(names, list):
        names = [names]
    return {norm(n) for n in names if n}


def on_street(data, street):
    """Full-name match, so 'Sellers Road' does not also catch 'Sellers Court'."""
    full = norm(street)
    return any(full in n for n in edge_names(data))


def first(value):
    return value[0] if isinstance(value, list) else value


def dist_m(G, n, lat, lon):
    return ox.distance.great_circle(lat, lon, G.nodes[n]["y"], G.nodes[n]["x"])


def edge_mid(G, u, v):
    return ((G.nodes[u]["y"] + G.nodes[v]["y"]) / 2, (G.nodes[u]["x"] + G.nodes[v]["x"]) / 2)


def overpass_retry(fn, *args, rounds=2, wait_s=10, **kwargs):
    """Call an OSMnx download, rotating through Overpass servers with backoff."""
    last = None
    for r in range(rounds):
        for url in OVERPASS_URLS:
            ox.settings.overpass_url = url
            step(f"requesting road data from {url} ...")
            try:
                return fn(*args, **kwargs)
            except Exception as e:  # busy server, timeout, bad gateway, ...
                last = e
                print(f"  Overpass {url} failed ({type(e).__name__}: {str(e).splitlines()[0][:80]})")
        if r < rounds - 1:
            print(f"  all servers busy; retrying in {wait_s} s")
            time.sleep(wait_s)
            wait_s *= 2
    raise RuntimeError("Every Overpass server failed; try again in a few minutes.") from last


def download_network(center, radius_m):
    step(f"downloading drive network within {radius_m:.0f} m of ({center[0]:.5f}, {center[1]:.5f}). "
         "This is the slow step (usually 1-5 min the first time, instant once cached).")
    return overpass_retry(ox.graph_from_point, center, dist=radius_m, network_type="drive")


DRIVABLE = {
    "motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link",
    "secondary", "secondary_link", "tertiary", "tertiary_link", "unclassified",
    "residential", "living_street", "service", "road",
}


def load_osm_file(path):
    """Drivable network from a downloaded .osm file (exports also hold buildings, footpaths, ...)."""
    step(f"reading {path}")
    G = ox.graph_from_xml(path, simplify=False, retain_all=True)
    G.remove_edges_from([(u, v, k) for u, v, k, d in G.edges(keys=True, data=True)
                         if first(d.get("highway")) not in DRIVABLE
                         or first(d.get("access")) in {"no", "private"}
                         or first(d.get("service")) in {"parking_aisle", "driveway"}])
    G.remove_nodes_from([n for n in list(G.nodes) if G.degree(n) == 0])
    if G.number_of_edges() == 0:
        raise SystemExit(f"No drivable roads found in {path}.")
    G = ox.simplify_graph(G)
    return G.subgraph(max(nx.weakly_connected_components(G), key=len)).copy()


def add_travel_times_and_capacity(G):
    G = ox.add_edge_speeds(G, hwy_speeds={k: v * 1.609344 for k, v in DEFAULT_SPEEDS_MPH.items()},
                           fallback=25 * 1.609344)
    G = ox.add_edge_travel_times(G)
    for _, _, d in G.edges(data=True):
        try:
            lanes = max(1, int(float(first(d.get("lanes", 1)))))
        except (TypeError, ValueError):
            lanes = 1
        if not d.get("oneway", False) and lanes > 1:
            lanes = max(1, lanes // 2)  # OSM `lanes` counts both directions on two-way roads
        d["capacity_vph"] = CAPACITY_VPH_PER_LANE.get(first(d.get("highway")), 500) * lanes
        d["t0"] = d["travel_time"]
    return G


# --- flooding ----------------------------------------------------------------

def junction_point(G, street, cross):
    """(lat, lon) where `street` meets `cross`, or their closest approach."""
    na = {n for u, v, d in G.edges(data=True) if on_street(d, street) for n in (u, v)}
    nb = {n for u, v, d in G.edges(data=True) if on_street(d, cross) for n in (u, v)}
    if not na or not nb:
        return None, None
    gap, a = min((dist_m(G, a, G.nodes[b]["y"], G.nodes[b]["x"]), a) for a in na for b in nb)
    return (G.nodes[a]["y"], G.nodes[a]["x"]), gap


def mark_flooded(G, zones):
    """Set d['zone'] on every flooded edge; return per-zone edge lists."""
    for _, _, d in G.edges(data=True):
        d.pop("zone", None)
    hits = {z["name"]: [] for z in zones}
    for z in zones:
        center = None
        if "lat" in z:
            center = (z["lat"], z["lon"])
        elif "near" in z:
            center, gap = junction_point(G, z["street"], z["near"])
            if center is None:
                print(f"WARNING: {z['name']}: '{z['street']}' or '{z['near']}' not in the network; "
                      "zone skipped (use --flood-circle)")
                continue
            if gap > 50:
                print(f"  {z['name']}: {z['street']} and {z['near']} are {gap:.0f} m apart; "
                      "using their closest point")
            z["center"] = center
        for u, v, k, d in G.edges(keys=True, data=True):
            if "street" in z and not on_street(d, z["street"]):
                continue
            if center is not None:
                pts = [(G.nodes[u]["y"], G.nodes[u]["x"]), (G.nodes[v]["y"], G.nodes[v]["x"]), edge_mid(G, u, v)]
                if min(ox.distance.great_circle(*center, *p) for p in pts) > z["radius_m"]:
                    continue
            d["zone"] = z["name"]
            hits[z["name"]].append((u, v, k))
        segs = {frozenset((u, v)) for u, v, _ in hits[z["name"]]}
        print(f"  flood zone '{z['name']}': {len(segs)} road segment(s) blocked")
        if not segs:
            print(f"WARNING: no roads matched '{z['name']}'; it may lie outside the download "
                  "(try --radius-m 4500) or need a --flood-circle")
    return hits


# --- routing -----------------------------------------------------------------

def edge_cost(d, weight, closed_zones):
    if d.get("zone") in closed_zones:
        return None
    return d[weight]


def routes_to(G, dest, weight, closed_zones):
    """Fastest route from every node to dest (one reverse Dijkstra)."""
    R = G.reverse(copy=False)

    def w(u, v, dd):
        costs = [c for c in (edge_cost(d, weight, closed_zones) for d in dd.values()) if c is not None]
        return min(costs) if costs else None

    times, paths = nx.single_source_dijkstra(R, dest, weight=w)
    return times, {n: p[::-1] for n, p in paths.items()}


def best_key(G, u, v, weight, closed_zones):
    opts = [(edge_cost(d, weight, closed_zones), k) for k, d in G[u][v].items()]
    return min((c, k) for c, k in opts if c is not None)[1]


def path_edges(G, path, weight, closed_zones):
    return [(u, v, best_key(G, u, v, weight, closed_zones)) for u, v in zip(path[:-1], path[1:])]


def path_stats(G, edges):
    return (sum(G.edges[e]["length"] for e in edges),
            sum(G.edges[e]["t0"] for e in edges),
            sum(G.edges[e]["travel_time"] for e in edges))


def street_sequence(G, edges):
    seq = []
    for e in edges:
        name = first(G.edges[e].get("name", "(unnamed)")) or "(unnamed)"
        if not seq or seq[-1][0] != name:
            seq.append([name, 0.0])
        seq[-1][1] += G.edges[e]["length"]
    return " -> ".join(f"{n} ({m / M_PER_MI:.2f} mi)" for n, m in seq)


def assign_with_congestion(G, origins, dest, closed_zones, vehicles, iterations):
    """Method of successive averages with BPR link times; returns final edge loads."""
    loads = {e: 0.0 for e in G.edges(keys=True)}
    for it in range(1, iterations + 1):
        _, paths = routes_to(G, dest, "travel_time", closed_zones)
        target = {e: 0.0 for e in loads}
        for o in origins:
            if o in paths:
                for e in path_edges(G, paths[o], "travel_time", closed_zones):
                    target[e] += vehicles
        step = 1.0 / it
        for e in loads:
            loads[e] += step * (target[e] - loads[e])
            d = G.edges[e]
            d["travel_time"] = d["t0"] * (1 + BPR_ALPHA * (loads[e] / d["capacity_vph"]) ** BPR_BETA)
    return loads


def reset_times(G):
    for _, _, d in G.edges(data=True):
        d["travel_time"] = d["t0"]


def evaluate(G, origins, dest, closed_zones):
    """Free-flow time (s) per origin with the given zones closed; None = stranded."""
    times, _ = routes_to(G, dest, "t0", closed_zones)
    return {o: times.get(o) for o in origins}


# --- map -------------------------------------------------------------------------

STATUS_STYLE = {  # origin status -> (colour, legend text)
    "unaffected": ("#8c8c8c", "Origin, normal route stays dry"),
    "rerouted": ("#f28e2b", "Origin, must detour around flooding"),
    "stranded": ("#000000", "Origin, stranded (no dry route)"),
}


def edge_xy(G, u, v, k):
    """Edge drawn along its real shape when OSM provides one."""
    geom = G.edges[u, v, k].get("geometry")
    if geom is not None:
        xs, ys = geom.xy
        return list(xs), list(ys)
    return [G.nodes[u]["x"], G.nodes[v]["x"]], [G.nodes[u]["y"], G.nodes[v]["y"]]


def place_label(ax, G, edges, text, color, placed, weight="normal", size=7):
    """Label a street on its longest visible edge, rotated along the road."""
    import matplotlib.patheffects as pe
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    best = None
    for e in edges:
        xs, ys = edge_xy(G, *e)
        mx, my = (xs[0] + xs[-1]) / 2, (ys[0] + ys[-1]) / 2
        if not (x0 < mx < x1 and y0 < my < y1):
            continue
        if best is None or G.edges[e]["length"] > best[0]:
            best = (G.edges[e]["length"], xs, ys, mx, my)
    if best is None:
        return
    _, xs, ys, mx, my = best
    (px0, py0), (px1, py1) = ax.transData.transform([(xs[0], ys[0]), (xs[-1], ys[-1])])
    pmx, pmy = ax.transData.transform((mx, my))
    if any((pmx - a) ** 2 + (pmy - b) ** 2 < 45 ** 2 for a, b in placed):
        return  # would overlap an existing label
    placed.append((pmx, pmy))
    angle = math.degrees(math.atan2(py1 - py0, px1 - px0))
    if angle > 90:
        angle -= 180
    elif angle < -90:
        angle += 180
    ax.text(mx, my, text, color=color, fontsize=size, fontweight=weight, rotation=angle,
            rotation_mode="anchor", ha="center", va="center", zorder=8,
            path_effects=[pe.withStroke(linewidth=3, foreground="white")])


def plot_map(G, df, loads, flooded, crit_df, school_pt, zones, path, n_detour_labels=6):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(11, 10))
    fig.patch.set_facecolor("white")
    for u, v, k in G.edges(keys=True):
        xs, ys = edge_xy(G, u, v, k)
        ax.plot(xs, ys, color="#d4d4d4", lw=0.7, zorder=1)

    vmax = max(loads.values()) if loads and max(loads.values()) > 0 else 1
    used = [(e, val) for e, val in loads.items() if val > 0]
    for e, val in used:
        xs, ys = edge_xy(G, *e)
        ax.plot(xs, ys, color="#1f77b4", lw=1 + 6 * val / vmax, alpha=0.75,
                solid_capstyle="round", zorder=3)
    for e in flooded:
        xs, ys = edge_xy(G, *e)
        ax.plot(xs, ys, color="#d62728", lw=5, solid_capstyle="round", zorder=4)
    for status, (color, _) in STATUS_STYLE.items():
        grp = df[df["status"] == status]
        ax.scatter(grp["lon"], grp["lat"], s=22, color=color, edgecolor="white",
                   linewidth=0.5, zorder=5)
    ax.scatter([school_pt[1]], [school_pt[0]], marker="*", s=420, color="gold",
               edgecolor="black", zorder=7)
    ax.annotate(SCHOOL_NAME, (school_pt[1], school_pt[0]), xytext=(10, 10),
                textcoords="offset points", fontsize=9, fontweight="bold", zorder=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=0.6))

    # zoom to the affected area: origins, school, flooded and used roads
    xs = list(df["lon"]) + [school_pt[1]]
    ys = list(df["lat"]) + [school_pt[0]]
    for e in [e for e, _ in used] + list(flooded):
        ex, ey = edge_xy(G, *e)
        xs += ex
        ys += ey
    pad_x = (max(xs) - min(xs)) * 0.15 + 0.002
    pad_y = (max(ys) - min(ys)) * 0.15 + 0.002
    ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
    ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y)
    ax.set_aspect(1 / math.cos(math.radians(school_pt[0])))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.apply_aspect()

    # road names: flooded streets, busiest detour roads, then major context roads
    by_name = {}
    for u, v, k, d in G.edges(keys=True, data=True):
        name = first(d.get("name"))
        if name:
            by_name.setdefault(name, []).append((u, v, k))
    placed = []
    flooded_names = []
    for e in flooded:
        name = first(G.edges[e].get("name"))
        if name and name not in flooded_names:
            flooded_names.append(name)
    for name in flooded_names:
        place_label(ax, G, [e for e in flooded if first(G.edges[e].get("name")) == name],
                    f"{name} (FLOODED)", "#b22222", placed, weight="bold", size=8)
    used_set = {e for e, _ in used}
    detour_names = [n for n in (crit_df["road"].tolist() if len(crit_df) else [])
                    if n in by_name and n not in flooded_names][:n_detour_labels]
    for name in detour_names:
        place_label(ax, G, [e for e in by_name[name] if e in used_set] or by_name[name],
                    name, "#164f7a", placed, weight="bold", size=8)
    major = {"motorway", "trunk", "primary", "secondary", "tertiary"}
    for name, edges in sorted(by_name.items(), key=lambda kv: -sum(G.edges[e]["length"] for e in kv[1])):
        if name in flooded_names or name in detour_names:
            continue
        if any(first(G.edges[e].get("highway")) in major for e in edges):
            place_label(ax, G, edges, name, "#555555", placed)
    # a few residential street names for orientation
    for name, edges in sorted(by_name.items(), key=lambda kv: -sum(G.edges[e]["length"] for e in kv[1]))[:60]:
        if name not in flooded_names and name not in detour_names:
            place_label(ax, G, edges, name, "#777777", placed, size=6)

    handles = [
        Line2D([], [], color="#d62728", lw=5, label="Flooded road (closed to traffic)"),
        Line2D([], [], color="#1f77b4", lw=1.5, alpha=0.75, label="Evacuation route, few vehicles"),
        Line2D([], [], color="#1f77b4", lw=7, alpha=0.75, label="Evacuation route, many vehicles"),
        Line2D([], [], color="#d4d4d4", lw=1.5, label="Other roads"),
    ] + [Line2D([], [], marker="o", ls="", color=c, markeredgecolor="white", markersize=7, label=t)
         for c, t in STATUS_STYLE.values()] + [
        Line2D([], [], marker="*", ls="", color="gold", markeredgecolor="black", markersize=15,
               label=f"{SCHOOL_NAME} (destination)"),
    ]
    counts = df["status"].value_counts()
    ax.set_title(f"Flood rerouting to {SCHOOL_NAME}\n"
                 f"{', '.join(z['name'] for z in zones)} closed  |  "
                 f"{counts.get('rerouted', 0)} origins rerouted, {counts.get('stranded', 0)} stranded, "
                 f"{counts.get('unaffected', 0)} unaffected", fontsize=11)
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.text(0.5, 0.075, "Line width = number of evacuating vehicles on that road "
             "(congestion-aware assignment). Road names: red = flooded, blue = busiest detour roads.",
             ha="center", fontsize=8, color="#444444")
    fig.subplots_adjust(bottom=0.14, top=0.93, left=0.02, right=0.98)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    if "ipykernel" in sys.modules:
        # show the saved PNG inline; works even if the kernel's matplotlib backend is "Agg"
        from IPython.display import Image, display
        display(Image(filename=path, width=900))


# --- main --------------------------------------------------------------------------

def main(argv=None):
    """argv: list of options, e.g. main(["--vehicles-per-origin", 15]) from a notebook."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--osm-file", default=None,
                    help="downloaded OpenStreetMap .osm file to use instead of downloading")
    ap.add_argument("--radius-m", type=float, default=DOWNLOAD_RADIUS_M,
                    help=f"download roads within this distance of the school (default {DOWNLOAD_RADIUS_M})")
    ap.add_argument("--school-latlon", type=float, nargs=2, metavar=("LAT", "LON"), default=SCHOOL_LATLON,
                    help=f"coordinates of {SCHOOL_NAME} (default {SCHOOL_LATLON})")
    ap.add_argument("--flood-circle", type=float, nargs=3, action="append", default=[],
                    metavar=("LAT", "LON", "RADIUS_M"), help="add a circular flood zone (repeatable)")
    ap.add_argument("--only-circles", action="store_true",
                    help="ignore the built-in street-based zones and use only --flood-circle zones")
    ap.add_argument("--affected-radius-m", type=float, default=400,
                    help="origins = intersections within this distance of a flooded road")
    ap.add_argument("--vehicles-per-origin", type=float, default=10,
                    help="vehicles leaving from each origin intersection (congestion model)")
    ap.add_argument("--assign-iterations", type=int, default=8, help="MSA iterations (0 = skip congestion)")
    ap.add_argument("--out", default="flood_rerouting_results", help="output folder")
    ap.add_argument("--no-plot", action="store_true")
    if argv is not None:
        argv = [str(a) for a in argv]
    args, _ = ap.parse_known_args(argv)  # ignore Jupyter's "-f kernel.json"

    global T0
    T0 = time.time()
    run_date = dt.datetime.now().astimezone()
    ox.settings.use_cache = True
    ox.settings.log_console = False
    ox.settings.requests_timeout = 90      # give up on a stuck server sooner, try the next one
    ox.settings.overpass_rate_limit = False  # don't sit waiting for a slot on a busy server

    zones = [] if args.only_circles else [dict(z) for z in FLOOD_ZONES]
    for i, (lat, lon, r) in enumerate(args.flood_circle, 1):
        zones.append({"name": f"Flood circle {i}", "lat": lat, "lon": lon, "radius_m": r})
    if not zones:
        raise SystemExit("No flood zones defined.")

    # 1. network --------------------------------------------------------------
    school_pt = tuple(args.school_latlon)
    if args.osm_file:
        G = load_osm_file(args.osm_file)
        if min(dist_m(G, n, *school_pt) for n in G.nodes) > 300:
            print(f"WARNING: {SCHOOL_NAME} is outside {args.osm_file}; export a larger area.")
    else:
        G = download_network(school_pt, args.radius_m)
    G = add_travel_times_and_capacity(G)
    step(f"network ready: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # 2. school ---------------------------------------------------------------
    main_cc = max(nx.strongly_connected_components(G), key=len)
    school = min(main_cc, key=lambda n: dist_m(G, n, *school_pt))
    print(f"{SCHOOL_NAME}: ({school_pt[0]:.6f}, {school_pt[1]:.6f}) -> node {school} "
          f"({dist_m(G, school, *school_pt):.0f} m away)")

    # 3. flooding and origins ---------------------------------------------------
    step("marking flooded roads")
    hits = mark_flooded(G, zones)
    flooded = [e for es in hits.values() for e in es]
    if not flooded:
        raise SystemExit("No roads matched the flood zones; check the names or use --flood-circle.")
    all_zones = {z["name"] for z in zones}
    mids = [edge_mid(G, u, v) for u, v, _ in flooded]
    origin_zone = {}
    for n in G.nodes:
        dz = [(ox.distance.great_circle(G.nodes[n]["y"], G.nodes[n]["x"], *m), G.edges[e]["zone"])
              for m, e in zip(mids, flooded)]
        dmin, zname = min(dz)
        if dmin <= args.affected_radius_m and n != school:
            origin_zone[n] = zname
    origins = sorted(origin_zone)
    print(f"Origins (intersections within {args.affected_radius_m:.0f} m of flooding): {len(origins)}")

    # 4. routing: normal vs flooded (free flow) -----------------------------------
    step("routing every origin to the school")
    t_norm, p_norm = routes_to(G, school, "t0", set())
    t_fl, p_fl = routes_to(G, school, "t0", all_zones)

    rows = []
    for o in origins:
        if o not in t_norm:
            continue  # cannot reach the school even when dry (edge of download area)
        e_n = path_edges(G, p_norm[o], "t0", set())
        L_n, T_n, _ = path_stats(G, e_n)
        row = {"origin_node": o, "lat": G.nodes[o]["y"], "lon": G.nodes[o]["x"],
               "nearest_flood_zone": origin_zone[o],
               "normal_distance_mi": L_n / M_PER_MI, "normal_time_min": T_n / 60,
               "normal_route": street_sequence(G, e_n),
               "normal_route_uses_flooded_road": any(G.edges[e].get("zone") for e in e_n)}
        if o in t_fl:
            e_f = path_edges(G, p_fl[o], "t0", all_zones)
            L_f, T_f, _ = path_stats(G, e_f)
            row.update(status="rerouted" if row["normal_route_uses_flooded_road"] else "unaffected",
                       flooded_distance_mi=L_f / M_PER_MI, flooded_time_min=T_f / 60,
                       extra_distance_mi=(L_f - L_n) / M_PER_MI, extra_time_min=(T_f - T_n) / 60,
                       flooded_route=street_sequence(G, e_f))
        else:
            row.update(status="stranded")
        rows.append(row)
    df = pd.DataFrame(rows)
    origins = list(df["origin_node"])

    # 5a. congestion-aware assignment -----------------------------------------
    step(f"congestion-aware assignment ({args.assign_iterations} iterations)")
    loads = {}
    if args.assign_iterations > 0:
        reachable = list(df.loc[df["status"] != "stranded", "origin_node"])
        loads = assign_with_congestion(G, reachable, school, all_zones,
                                       args.vehicles_per_origin, args.assign_iterations)
        t_cg, p_cg = routes_to(G, school, "travel_time", all_zones)
        cg_time, cg_route = {}, {}
        for o in reachable:
            e_c = path_edges(G, p_cg[o], "travel_time", all_zones)
            cg_time[o] = path_stats(G, e_c)[2] / 60
            cg_route[o] = street_sequence(G, e_c)
        df["congested_time_min"] = df["origin_node"].map(cg_time)
        df["congested_route"] = df["origin_node"].map(cg_route)

    # 5b. critical detour roads -------------------------------------------------
    crit = {}
    for e, v in loads.items():
        if v > 0:
            d = G.edges[e]
            name = first(d.get("name", "(unnamed)")) or "(unnamed)"
            c = crit.setdefault(name, {"road": name, "peak_vehicles_per_hour": 0, "capacity_vph": 0,
                                       "segments_used": 0})
            if v > c["peak_vehicles_per_hour"]:
                c.update(peak_vehicles_per_hour=round(v), capacity_vph=d["capacity_vph"])
            c["segments_used"] += 1
    crit_df = pd.DataFrame(crit.values())
    if len(crit_df):
        crit_df["peak_volume_to_capacity"] = (crit_df["peak_vehicles_per_hour"] / crit_df["capacity_vph"]).round(2)
        crit_df = crit_df.sort_values("peak_vehicles_per_hour", ascending=False)

    # 5c. mitigation priority: reopen one zone at a time ---------------------------
    step("ranking flood zones for mitigation")
    base = evaluate(G, origins, school, all_zones)
    base_stranded = sum(base[o] is None for o in origins)
    mit = []
    for z in sorted(all_zones):
        tt = evaluate(G, origins, school, all_zones - {z})
        # delay saved only where a dry route exists either way; freed origins are counted separately
        saved = sum(base[o] - tt[o] for o in origins if base[o] is not None and tt[o] is not None) / 60
        mit.append({"reopen_zone": z,
                    "origins_freed_from_stranding": base_stranded - sum(tt[o] is None for o in origins),
                    "total_delay_saved_min": round(saved, 1),
                    "delay_saved_vehicle_hours": round(saved * args.vehicles_per_origin / 60, 1)})
    mit_df = pd.DataFrame(mit).sort_values(["origins_freed_from_stranding", "total_delay_saved_min"],
                                           ascending=False)

    # 6. report -----------------------------------------------------------------------
    n_st = (df["status"] == "stranded").sum()
    rr = df[df["status"] == "rerouted"]
    summary = pd.DataFrame([
        {"Measure": "Origin intersections analysed", "Value": len(df)},
        {"Measure": "Normally use a flooded road (rerouted)", "Value": len(rr)},
        {"Measure": "Stranded (no dry route to school)", "Value": int(n_st)},
        {"Measure": "Average normal trip, rerouted origins", "Value": f"{rr['normal_time_min'].mean():.1f} min, "
                                                                     f"{rr['normal_distance_mi'].mean():.2f} mi" if len(rr) else "-"},
        {"Measure": "Average flooded trip, rerouted origins", "Value": f"{rr['flooded_time_min'].mean():.1f} min, "
                                                                      f"{rr['flooded_distance_mi'].mean():.2f} mi" if len(rr) else "-"},
        {"Measure": "Average additional travel", "Value": f"+{rr['extra_time_min'].mean():.1f} min, "
                                                          f"+{rr['extra_distance_mi'].mean():.2f} mi" if len(rr) else "-"},
        {"Measure": "Worst additional travel", "Value": f"+{rr['extra_time_min'].max():.1f} min" if len(rr) else "-"},
    ])
    if "congested_time_min" in df and len(rr):
        summary.loc[len(summary)] = {"Measure": "Average flooded trip with everyone leaving at once",
                                     "Value": f"{df.loc[rr.index, 'congested_time_min'].mean():.1f} min "
                                              f"({args.vehicles_per_origin:g} vehicles per origin)"}

    print("\n=== Flood rerouting to", SCHOOL_NAME, "===")
    print(summary.to_string(index=False))
    if len(rr):
        print("\nMost affected origins:")
        cols = ["origin_node", "nearest_flood_zone", "normal_time_min", "flooded_time_min", "extra_time_min"]
        print(rr.sort_values("extra_time_min", ascending=False)[cols].head(5).round(1).to_string(index=False))
        top = rr.sort_values("extra_time_min", ascending=False).iloc[0]
        print(f"\nExample detour from node {top['origin_node']}:\n  normal : {top['normal_route']}\n"
              f"  flooded: {top['flooded_route']}")
    if len(crit_df):
        print("\nCritical detour roads (keep clear / traffic control):")
        print(crit_df.head(8).to_string(index=False))
    print("\nMitigation priority (reopen one flood zone):")
    print(mit_df.to_string(index=False))

    os.makedirs(args.out, exist_ok=True)
    summary.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    df.to_csv(os.path.join(args.out, "origin_routes.csv"), index=False)
    crit_df.to_csv(os.path.join(args.out, "critical_detour_roads.csv"), index=False)
    mit_df.to_csv(os.path.join(args.out, "mitigation_priority.csv"), index=False)
    record = {
        "destination": {"name": SCHOOL_NAME, "latlon": school_pt, "osm_node": int(school)},
        "flood_zones": [{k: v for k, v in z.items()} for z in zones],
        "blocked_segments": {z: len(es) for z, es in hits.items()},
        "measurement_source": f"OpenStreetMap drive network via OSMnx {ox.__version__} (Overpass API)",
        "measurement_date": run_date.isoformat(timespec="seconds"),
        "assumptions": {
            "speeds": "OSM maxspeed where tagged, else highway-class mean / DEFAULT_SPEEDS_MPH",
            "flooded_roads": "fully impassable in both directions",
            "origins": f"intersections within {args.affected_radius_m} m of a flooded road",
            "congestion": f"BPR alpha={BPR_ALPHA}, beta={BPR_BETA}; {args.vehicles_per_origin} veh/origin over 1 h; "
                          f"MSA iterations={args.assign_iterations}",
        },
    }
    with open(os.path.join(args.out, "run_record.json"), "w") as f:
        json.dump(record, f, indent=2, default=str)

    if not args.no_plot:
        plot_map(G, df, loads, flooded, crit_df, school_pt, zones,
                 os.path.join(args.out, "flood_rerouting_map.png"))

    print(f"\nOutputs written to {os.path.abspath(args.out)}/")
    return df


if __name__ == "__main__":
    main()
