"""
Simulated closure of Dickson Avenue: residents' trips to three schools
(Hanahan / North Charleston, SC).

Question
    If Dickson Avenue is closed along its whole length (both directions), how do
    trips from the surrounding homes change for
      - Hanahan High School,
      - George R. Fishburne Elementary School,
      - Matilda F. Dunston Elementary School?
    Normal routes are compared with closure routes on travel distance,
    estimated travel time, and schools that become unreachable; a comparison
    table shows whether the impact depends on which school people go to.

Method
    1. Road network: a downloaded OpenStreetMap .osm file (--osm-file), or a
       download of all drivable roads around Dickson Avenue.
    2. Closure: every road segment named Dickson Avenue is removed.
    3. Residents: every intersection within --resident-radius-m of Dickson
       Avenue (the streets whose homes use it). Street ends that can only be
       reached along Dickson Avenue itself (e.g. a dead-end stretch) are
       reported separately. Individual driveways are not in the road network:
       homes fronting the closed street are represented by its cross streets.
    4. Schools: found by name in the map data (amenity=school); any school
       not found must be given with --school "Name" LAT LON.
    5. For each school, one reverse Dijkstra (fastest free-flow route) with all
       roads open and one with Dickson Avenue closed.
    6. Through traffic: the Dickson Ave AADT (1,900 vehicles/day) times the
       detour between its two ends gives an upper-bound daily cost.

Data
    Road geometry, names, speed limits and school locations: OpenStreetMap
    (openstreetmap.org) via OSMnx. Missing speed limits are filled by road type.
    Travel times are free-flow estimates (no signals, turns or congestion).

Requirements
    pip install osmnx networkx pandas matplotlib

Usage
    python dickson_closure_schools.py --osm-file map.osm
    Jupyter:  %run dickson_closure_schools.py --osm-file map.osm
              or paste into a cell, then  main(["--osm-file", "map.osm"])
    If a school is missing from the map data:
              --school "Hanahan High School" LAT LON   (repeatable)
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

CLOSED_STREET = "Dickson Avenue"
DICKSON_CENTER = (32.9046, -80.0028)  # Allison Ave - Dickson Ave, mid-corridor
DOWNLOAD_RADIUS_M = 4000
# The schools lie north-west (Hanahan High, Fishburne Elementary) of Dickson Ave; Dunston
# Elementary's position must come from the map data or --school. Export generously.
EXPORT_BOX = (32.880, 32.930, -80.040, -79.980)  # min lat, max lat, min lon, max lon
DICKSON_AADT = 1900  # vehicles/day on Dickson Avenue (user-supplied count)
SCHOOLS = [  # label, words that must all appear in the OSM name (lower case)
    ("Hanahan High School", ("hanahan", "high")),
    ("George R. Fishburne Elementary School", ("fishburne",)),
    ("Matilda F. Dunston Elementary School", ("dunston",)),
]
MIN_NODES = 150

DEFAULT_SPEEDS_MPH = {
    "motorway": 60, "trunk": 50, "primary": 45, "secondary": 40,
    "tertiary": 35, "unclassified": 30, "residential": 25,
    "living_street": 15, "service": 15,
}
DRIVABLE = {
    "motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link",
    "secondary", "secondary_link", "tertiary", "tertiary_link", "unclassified",
    "residential", "living_street", "service", "road",
}
MAJOR = {"motorway", "trunk", "primary", "secondary", "tertiary"}
OVERPASS_URLS = [
    "https://overpass-api.de/api",
    "https://overpass.kumi.systems/api",
    "https://overpass.private.coffee/api",
    "https://maps.mail.ru/osm/tools/overpass/api",
]
M_PER_MI = 1609.344
T0 = time.time()


def step(msg):
    print(f"[{time.time() - T0:6.0f} s] {msg}", flush=True)



# --- names and geometry -------------------------------------------------------------

def norm(name):
    """Lower-case a street name and expand common suffix abbreviations."""
    s = str(name).lower().replace(".", "").strip()
    for short, full in ((" rd", " road"), (" dr", " drive"), (" ave", " avenue"),
                        (" av", " avenue"), (" st", " street")):
        if s.endswith(short):
            s = s[: -len(short)] + full
    return s


def first(value):
    return value[0] if isinstance(value, list) else value


def on_street(data, street):
    names = data.get("name", [])
    names = names if isinstance(names, list) else [names]
    return any(norm(street) in norm(n) for n in names if n)


def dist_m(G, n, lat, lon):
    return ox.distance.great_circle(lat, lon, G.nodes[n]["y"], G.nodes[n]["x"])


def node_street(G, n):
    """Readable name of an intersection: the streets that meet there."""
    names = []
    for _, _, d in list(G.out_edges(n, data=True)) + list(G.in_edges(n, data=True)):
        name = first(d.get("name"))
        if name and name not in names:
            names.append(name)
    return " & ".join(names[:2]) if names else f"node {n}"


def edge_xy(G, u, v, k):
    geom = G.edges[u, v, k].get("geometry")
    if geom is not None:
        xs, ys = geom.xy
        return list(xs), list(ys)
    return [G.nodes[u]["x"], G.nodes[v]["x"]], [G.nodes[u]["y"], G.nodes[v]["y"]]



# --- network ----------------------------------------------------------------------------

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
    # keep a node wherever the street name changes, so merged segments never carry two names
    G = ox.simplify_graph(G, edge_attrs_differ=["name"])
    return G.subgraph(max(nx.weakly_connected_components(G), key=len)).copy()


def download_network(center, radius_m):
    step(f"downloading roads within {radius_m:.0f} m of {center} (slow the first time, cached after)")
    ox.settings.use_cache = True
    ox.settings.requests_timeout = 90
    ox.settings.overpass_rate_limit = False
    last = None
    for url in OVERPASS_URLS * 2:
        ox.settings.overpass_url = url
        try:
            G = ox.graph_from_point(center, dist=radius_m, network_type="drive", simplify=False)
            return ox.simplify_graph(G, edge_attrs_differ=["name"])
        except Exception as e:  # busy server, timeout, bad gateway, ...
            last = e
            print(f"  {url} failed ({type(e).__name__}); trying the next server")
    raise RuntimeError("Every Overpass server failed; use --osm-file or try later.") from last


def add_travel_times(G):
    G = ox.add_edge_speeds(G, hwy_speeds={k: v * 1.609344 for k, v in DEFAULT_SPEEDS_MPH.items()},
                           fallback=25 * 1.609344)
    return ox.add_edge_travel_times(G)



# --- routing ----------------------------------------------------------------------------

def routes_to(G, targets, blocked):
    """Fastest route from every node to the nearest of `targets` (one reverse Dijkstra),
    avoiding blocked edges."""
    def w(u, v, dd):  # reversed graph: the original edge is v -> u
        costs = [d["travel_time"] for k, d in dd.items() if (v, u, k) not in blocked]
        return min(costs) if costs else None
    times, paths = nx.multi_source_dijkstra(G.reverse(copy=False), set(targets), weight=w)
    return times, {n: p[::-1] for n, p in paths.items()}


def path_edges(G, path, blocked):
    out = []
    for u, v in zip(path[:-1], path[1:]):
        k = min((d["travel_time"], k) for k, d in G[u][v].items() if (u, v, k) not in blocked)[1]
        out.append((u, v, k))
    return out


def street_sequence(G, edges):
    seq = []
    for e in edges:
        name = first(G.edges[e].get("name")) or "(unnamed)"
        if not seq or seq[-1][0] != name:
            seq.append([name, 0.0])
        seq[-1][1] += G.edges[e]["length"]
    return " -> ".join(f"{n} ({m / M_PER_MI:.2f} mi)" for n, m in seq)



# --- schools ------------------------------------------------------------------------------

def find_schools(args, center):
    """Locate each school by name in the OSM data (amenity=school), or use --school overrides."""
    given = {n.lower(): (n, float(a), float(b)) for n, a, b in args.school}
    found_osm = []
    try:
        gdf = (ox.features_from_xml(args.osm_file, tags={"amenity": "school"}) if args.osm_file
               else ox.features_from_point(center, tags={"amenity": "school"}, dist=args.radius_m))
        for _, r in gdf.iterrows():
            if "name" in r and isinstance(r["name"], str):
                pt = r.geometry.representative_point()
                found_osm.append((r["name"], pt.y, pt.x))
    except Exception as e:  # no schools in the data, or server failure
        print(f"  no schools found in the map data ({type(e).__name__})")
    out = []
    for label, keywords in SCHOOLS:
        override = next((v for k, v in given.items() if all(w in k for w in keywords)), None)
        if override:
            out.append({"name": label, "lat": override[1], "lon": override[2], "source": "given (--school)"})
            continue
        hits = [s for s in found_osm if all(w in s[0].lower() for w in keywords)]
        if hits:
            name, lat, lon = hits[0]
            out.append({"name": label, "lat": lat, "lon": lon, "source": f"OSM: '{name}'"})
        else:
            print(f"WARNING: {label} not found in the map data. Add it with "
                  f'--school "{label}" LAT LON (right-click it in Google Maps to copy them).')
    # any extra --school entries that are not one of the three
    for k, (n, a, b) in given.items():
        if not any(all(w in k for w in kw) for _, kw in SCHOOLS):
            out.append({"name": n, "lat": a, "lon": b, "source": "given (--school)"})
    return out


# --- map ----------------------------------------------------------------------------------

def plot_map(G, od, closed, school, example, path):
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.lines import Line2D

    sub = od[od["school"] == school["name"]]
    fig, ax = plt.subplots(figsize=(11, 10))
    for u, v, k, d in G.edges(keys=True, data=True):
        major = first(d.get("highway")) in MAJOR
        xs, ys = edge_xy(G, u, v, k)
        ax.plot(xs, ys, color="#a8a8a8" if major else "#d4d4d4", lw=1.4 if major else 0.7, zorder=1)
    for e in closed:
        xs, ys = edge_xy(G, *e)
        ax.plot(xs, ys, color="#d62728", lw=5, solid_capstyle="round", zorder=4)
    if example is not None:
        for edges, color, style, z in ((example["normal_edges"], "#1f77b4", "-", 5),
                                       (example["closure_edges"], "#ff7f0e", "--", 6)):
            for e in edges:
                xs, ys = edge_xy(G, *e)
                ax.plot(xs, ys, color=color, lw=3.5, ls=style, zorder=z)
    style = {"unchanged": ("#9e9e9e", 16), "longer": ("#ff7f0e", 28), "unreachable": ("#000000", 40)}
    for status, (color, size) in style.items():
        grp = sub[sub["status"] == status]
        ax.scatter(grp["lon"], grp["lat"], s=size, color=color, edgecolor="white", linewidth=0.5, zorder=7)
    ax.scatter([school["lon"]], [school["lat"]], marker="*", s=420, color="gold", edgecolor="black", zorder=9)
    ax.annotate(school["name"], (school["lon"], school["lat"]), xytext=(9, 9), textcoords="offset points",
                fontsize=9, fontweight="bold", zorder=10,
                path_effects=[pe.withStroke(linewidth=3, foreground="white")])

    xs, ys = list(sub["lon"]) + [school["lon"]], list(sub["lat"]) + [school["lat"]]
    for e in closed + (example["normal_edges"] + example["closure_edges"] if example else []):
        ex, ey = edge_xy(G, *e)
        xs += ex
        ys += ey
    px, py = (max(xs) - min(xs)) * 0.06 + 0.002, (max(ys) - min(ys)) * 0.06 + 0.002
    ax.set_xlim(min(xs) - px, max(xs) + px)
    ax.set_ylim(min(ys) - py, max(ys) + py)
    ax.set_aspect(1 / math.cos(math.radians(sum(ys) / len(ys))))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.apply_aspect()

    # street names: one per street, on its longest visible segment
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    best = {}
    for u, v, k, d in G.edges(keys=True, data=True):
        name = first(d.get("name"))
        ex, ey = edge_xy(G, u, v, k)
        mx, my = (ex[0] + ex[-1]) / 2, (ey[0] + ey[-1]) / 2
        if name and x0 < mx < x1 and y0 < my < y1 and (name not in best or d["length"] > best[name][0]):
            best[name] = (d["length"], ex, ey, mx, my, first(d.get("highway")) in MAJOR)
    placed = []
    order = sorted(best.items(), key=lambda kv: (norm(CLOSED_STREET) not in norm(kv[0]), not kv[1][5], -kv[1][0]))
    for name, (_, ex, ey, mx, my, major) in order:
        (a0, b0), (a1, b1), (am, bm) = ax.transData.transform([(ex[0], ey[0]), (ex[-1], ey[-1]), (mx, my)])
        if any((am - a) ** 2 + (bm - b) ** 2 < 42 ** 2 for a, b in placed):
            continue
        placed.append((am, bm))
        ang = math.degrees(math.atan2(b1 - b0, a1 - a0))
        ang = ang - 180 if ang > 90 else ang + 180 if ang < -90 else ang
        key = norm(CLOSED_STREET) in norm(name)
        ax.text(mx, my, f"{name} (CLOSED)" if key else name, fontsize=8 if key else (6.5 if major else 5.5),
                fontweight="bold" if key else "normal", color="#b22222" if key else "#444444",
                rotation=ang, rotation_mode="anchor", ha="center", va="center", zorder=11,
                path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

    handles = [Line2D([], [], color="#d62728", lw=5, label=f"{CLOSED_STREET} closed (both directions)")]
    if example is not None:
        handles += [Line2D([], [], color="#1f77b4", lw=3.5, label=f"Normal route (from {example['label']})"),
                    Line2D([], [], color="#ff7f0e", lw=3.5, ls="--", label="Route with Dickson Ave closed")]
    handles += [
        Line2D([], [], marker="o", ls="", color="#9e9e9e", label="Resident origin: trip unchanged"),
        Line2D([], [], marker="o", ls="", color="#ff7f0e", label="Resident origin: trip longer"),
        Line2D([], [], marker="o", ls="", color="#000000", label="Resident origin: cannot reach the school"),
        Line2D([], [], marker="*", ls="", color="gold", markeredgecolor="black", markersize=15,
               label=f"{school['name']} (destination)"),
    ]
    counts = sub["status"].value_counts()
    ax.set_title(f"{CLOSED_STREET} closed: trips to {school['name']}\n"
                 f"{counts.get('longer', 0)} longer, {counts.get('unreachable', 0)} cannot reach, "
                 f"{counts.get('unchanged', 0)} unchanged", fontsize=11)
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8.5, frameon=False)
    fig.subplots_adjust(bottom=0.15, top=0.93, left=0.02, right=0.98)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    if "ipykernel" in sys.modules:
        from IPython.display import Image, display
        display(Image(filename=path, width=900))


# --- main -----------------------------------------------------------------------------------

def main(argv=None):
    """argv: e.g. main(["--osm-file", "map.osm"]) from a notebook."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--osm-file", default=None, help="downloaded OpenStreetMap .osm file")
    ap.add_argument("--radius-m", type=float, default=DOWNLOAD_RADIUS_M,
                    help="download radius around Dickson Ave when no --osm-file is given")
    ap.add_argument("--school", nargs=3, action="append", default=[], metavar=("NAME", "LAT", "LON"),
                    help="school location if it is not in the map data (repeatable)")
    ap.add_argument("--resident-radius-m", type=float, default=400,
                    help="residents = intersections within this distance of Dickson Ave")
    ap.add_argument("--dickson-aadt", type=float, default=DICKSON_AADT,
                    help=f"vehicles/day on Dickson Ave (default {DICKSON_AADT})")
    ap.add_argument("--out", default="dickson_closure_schools_results", help="output folder")
    ap.add_argument("--no-plot", action="store_true")
    argv = sys.argv[1:] if argv is None else [str(a) for a in argv]
    args, _ = ap.parse_known_args(argv)  # ignores Jupyter's "-f kernel.json"

    global T0
    T0 = time.time()
    run_date = dt.datetime.now().astimezone()

    # 1. network
    G = load_osm_file(args.osm_file) if args.osm_file else download_network(DICKSON_CENTER, args.radius_m)
    G = add_travel_times(G)
    step(f"network ready: {G.number_of_nodes()} intersections, {G.number_of_edges()} road segments")
    if G.number_of_nodes() < MIN_NODES:
        print(f"WARNING: only {G.number_of_nodes()} intersections; the map is too small to contain detour "
              f"streets. Export a larger area (e.g. lat {EXPORT_BOX[0]}..{EXPORT_BOX[1]}, "
              f"lon {EXPORT_BOX[2]}..{EXPORT_BOX[3]}).")

    # 2. closure: every Dickson Avenue segment, both directions
    closed = [(u, v, k) for u, v, k, d in G.edges(keys=True, data=True) if on_street(d, CLOSED_STREET)]
    if not closed:
        raise SystemExit(f"{CLOSED_STREET} is not in the road network; export an area that includes it.")
    blocked = set(closed)
    dickson_nodes = {n for e in closed for n in e[:2]}
    closed_mi = sum(G.edges[e]["length"] for e in closed) / 2 / M_PER_MI  # two-way roads appear twice
    step(f"{CLOSED_STREET} closed: {len({frozenset(e[:2]) for e in closed})} segments, {closed_mi:.2f} mi")

    # 3. residents and schools
    pts = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in dickson_nodes]
    origins = [n for n in G.nodes
               if min(ox.distance.great_circle(G.nodes[n]["y"], G.nodes[n]["x"], a, b) for a, b in pts)
               <= args.resident_radius_m]
    # locations whose only road out is Dickson Ave (e.g. a dead-end stretch of it)
    dickson_only = [n for n in origins if all(on_street(G.edges[e], CLOSED_STREET)
                                              for e in list(G.out_edges(n, keys=True)) + list(G.in_edges(n, keys=True)))]
    schools = find_schools(args, DICKSON_CENTER)
    if not schools:
        raise SystemExit('No schools located; add them with --school "Name" LAT LON.')
    for s in schools:
        s["node"] = min(G.nodes, key=lambda n: dist_m(G, n, s["lat"], s["lon"]))
        gap = dist_m(G, s["node"], s["lat"], s["lon"])
        s["snap_m"] = round(gap)
        print(f"  {s['name']}: ({s['lat']:.5f}, {s['lon']:.5f}) from {s['source']}, nearest road {gap:.0f} m"
              + ("  <- WARNING: far from any road in the map; export a larger area" if gap > 300 else ""))
    step(f"{len(origins)} resident origins within {args.resident_radius_m:.0f} m of {CLOSED_STREET} "
         f"({len(dickson_only)} reachable only via {CLOSED_STREET})")

    # 4. normal vs closure routes
    rows, examples = [], {}
    for s in schools:
        t_n, p_n = routes_to(G, [s["node"]], set())
        t_c, p_c = routes_to(G, [s["node"]], blocked)
        for o in origins:
            if o not in t_n or o == s["node"]:
                continue  # unreachable even with all roads open (edge of the map)
            e_n = path_edges(G, p_n[o], set())
            row = {"school": s["name"], "origin_node": o, "origin": node_street(G, o),
                   "lat": G.nodes[o]["y"], "lon": G.nodes[o]["x"],
                   "uses_dickson_normally": any(e in blocked for e in e_n),
                   "normal_distance_mi": sum(G.edges[e]["length"] for e in e_n) / M_PER_MI,
                   "normal_time_min": t_n[o] / 60, "normal_route": street_sequence(G, e_n)}
            if o in t_c:
                e_c = path_edges(G, p_c[o], blocked)
                row.update(closure_distance_mi=sum(G.edges[e]["length"] for e in e_c) / M_PER_MI,
                           closure_time_min=t_c[o] / 60, closure_route=street_sequence(G, e_c))
                row["extra_distance_mi"] = row["closure_distance_mi"] - row["normal_distance_mi"]
                row["extra_time_min"] = row["closure_time_min"] - row["normal_time_min"]
                row["status"] = "longer" if row["extra_time_min"] > 0.05 else "unchanged"
                if row["status"] == "longer" and (s["name"] not in examples
                                                  or row["extra_time_min"] > examples[s["name"]]["extra"]):
                    examples[s["name"]] = {"extra": row["extra_time_min"], "normal_edges": e_n,
                                           "closure_edges": e_c, "label": node_street(G, o)}
            else:
                row["status"] = "unreachable"
            rows.append(row)
    od = pd.DataFrame(rows)
    if od.empty:
        raise SystemExit("No resident origin can reach any school even with all roads open; "
                         "export a larger area that includes the schools.")

    # 5. tables
    tables, compare = [], []
    for s in schools:
        sub = od[od["school"] == s["name"]]
        both = sub[sub["status"] != "unreachable"]
        hit = both[both["status"] == "longer"]
        unreachable = int((sub["status"] == "unreachable").sum())
        tables.append(pd.DataFrame([
            {"School": s["name"], "Condition": "All roads open",
             "Avg distance": f"{both['normal_distance_mi'].mean():.2f} mi",
             "Avg travel time": f"{both['normal_time_min'].mean():.1f} min",
             "Residents who can reach it": len(sub)},
            {"School": s["name"], "Condition": f"{CLOSED_STREET} closed",
             "Avg distance": f"{both['closure_distance_mi'].mean():.2f} mi",
             "Avg travel time": f"{both['closure_time_min'].mean():.1f} min",
             "Residents who can reach it": len(sub) - unreachable},
            {"School": s["name"], "Condition": "Difference",
             "Avg distance": f"{round(both['extra_distance_mi'].mean(), 2) + 0.0:+.2f} mi",
             "Avg travel time": f"{round(both['extra_time_min'].mean(), 1) + 0.0:+.1f} min",
             "Residents who can reach it": f"{unreachable} cannot reach it"},
        ]))
        compare.append({"School": s["name"], "Residents": len(sub),
                        "Normally use Dickson Ave": int(sub["uses_dickson_normally"].sum()),
                        "Trips longer": len(hit), "Cannot reach": unreachable,
                        "Avg extra time, affected trips (min)": round(hit["extra_time_min"].mean(), 2) + 0.0 if len(hit) else 0.0,
                        "Max extra time (min)": round(hit["extra_time_min"].max(), 2) + 0.0 if len(hit) else 0.0,
                        "Avg extra distance, affected trips (mi)": round(hit["extra_distance_mi"].mean(), 2) + 0.0 if len(hit) else 0.0})
    results = pd.concat(tables, ignore_index=True)
    comparison = pd.DataFrame(compare)

    # 6. Dickson Ave traffic (AADT): end-to-end detour as an upper bound per diverted vehicle
    # ends = the two farthest-apart points where Dickson Ave meets another street (a dead-end
    # tail carries no through traffic)
    joins = [n for n in dickson_nodes if not all(on_street(G.edges[e], CLOSED_STREET) for e in
                                                 list(G.out_edges(n, keys=True)) + list(G.in_edges(n, keys=True)))]
    ends = max(((a, b) for a in joins for b in joins),
               key=lambda ab: dist_m(G, ab[0], G.nodes[ab[1]]["y"], G.nodes[ab[1]]["x"]))
    aadt_rows = []
    for a, b in (ends, ends[::-1]):
        try:
            tn = nx.shortest_path_length(G, a, b, weight="travel_time")
            ln = nx.shortest_path_length(G, a, b, weight="length")
            H = G.copy()
            H.remove_edges_from(closed)
            tc = nx.shortest_path_length(H, a, b, weight="travel_time")
            lc = nx.shortest_path_length(H, a, b, weight="length")
            extra_t, extra_d = (tc - tn) / 60, (lc - ln) / M_PER_MI
        except nx.NetworkXNoPath:
            extra_t = extra_d = float("nan")
        veh = args.dickson_aadt / 2
        aadt_rows.append({"trip": f"{node_street(G, a)} -> {node_street(G, b)}", "vehicles_per_day": round(veh),
                          "extra_min_per_vehicle": round(extra_t, 2), "extra_mi_per_vehicle": round(extra_d, 2),
                          "extra_vehicle_hours_per_day": round(veh * extra_t / 60, 1),
                          "extra_vehicle_miles_per_day": round(veh * extra_d, 1)})
    aadt = pd.DataFrame(aadt_rows)

    # 7. report
    print(f"\n=== {CLOSED_STREET} closed: residents' trips to schools ===")
    print(f"Residents: {len(origins)} intersections within {args.resident_radius_m:.0f} m of {CLOSED_STREET}. "
          "Averages use residents who can reach the school in both cases.\n")
    print(results.to_string(index=False))
    for s in schools:
        sub = od[od["school"] == s["name"]]
        longer = sub[sub["status"] == "longer"].sort_values("extra_time_min", ascending=False)
        print(f"\n{s['name']}: {len(longer)} trips longer, {int((sub['status'] == 'unreachable').sum())} "
              f"cannot reach, {int((sub['status'] == 'unchanged').sum())} unchanged")
        if len(longer):
            print(longer[["origin", "normal_distance_mi", "closure_distance_mi", "normal_time_min",
                          "closure_time_min", "extra_time_min"]].head(5).round(2).to_string(index=False))
            top = longer.iloc[0]
            print(f"  Most affected: from {top['origin']}\n    normal : {top['normal_route']}\n"
                  f"    closed : {top['closure_route']}")
    if dickson_only:
        print(f"\n{len(dickson_only)} location(s) can only be reached along {CLOSED_STREET} itself "
              "(e.g. a dead-end stretch); with the street closed, nobody there can drive anywhere.")

    print("\n=== Does the impact depend on the school? ===")
    print(comparison.to_string(index=False))
    if len(comparison) > 1:
        for col, label in (("Trips longer", "Most trips made longer"), ("Cannot reach", "Most residents cut off"),
                           ("Max extra time (min)", "Largest single delay")):
            top = comparison.loc[comparison[col].idxmax()]
            same = (comparison[col] == top[col]).all()
            print(f"  {label}: " + ("same for every school" if same else f"{top['School']} ({top[col]})"))

    print(f"\n=== Through traffic on {CLOSED_STREET} (AADT {args.dickson_aadt:,.0f} vehicles/day) ===")
    print(aadt.to_string(index=False))
    if aadt["extra_vehicle_hours_per_day"].isna().any():
        print("Total: unknown - no detour between the ends of Dickson Ave inside this map.")
    else:
        print(f"Total: {aadt['extra_vehicle_hours_per_day'].sum():.1f} extra vehicle-hours/day, "
              f"{aadt['extra_vehicle_miles_per_day'].sum():.0f} extra vehicle-miles/day.")
    print("Upper bound: assumes every Dickson Ave vehicle drives its full length, split evenly by direction; "
          "free-flow speeds.")

    os.makedirs(args.out, exist_ok=True)
    results.to_csv(os.path.join(args.out, "results_table.csv"), index=False)
    comparison.to_csv(os.path.join(args.out, "school_comparison.csv"), index=False)
    od.to_csv(os.path.join(args.out, "resident_school_routes.csv"), index=False)
    aadt.to_csv(os.path.join(args.out, "dickson_aadt_impact.csv"), index=False)
    with open(os.path.join(args.out, "run_record.json"), "w") as f:
        json.dump({
            "scenario": f"{CLOSED_STREET} closed along its whole length, both directions",
            "closed_length_mi": round(closed_mi, 3),
            "residents": f"{len(origins)} intersections within {args.resident_radius_m} m of {CLOSED_STREET}",
            "residents_only_via_dickson": len(dickson_only),
            "schools": [{k: s[k] for k in ("name", "lat", "lon", "source", "snap_m")} for s in schools],
            "dickson_aadt": args.dickson_aadt,
            "data_source": (f"OpenStreetMap file {args.osm_file}" if args.osm_file else
                            "OpenStreetMap via Overpass API") + f", OSMnx {ox.__version__}",
            "speeds": "OSM maxspeed where tagged, else road-type mean / DEFAULT_SPEEDS_MPH; free flow",
            "run_date": run_date.isoformat(timespec="seconds"),
        }, f, indent=2)
    if not args.no_plot:
        for i, s in enumerate(schools, 1):
            slug = "".join(ch if ch.isalnum() else "_" for ch in s["name"].lower()).strip("_")[:40]
            plot_map(G, od, closed, s, examples.get(s["name"]),
                     os.path.join(args.out, f"dickson_closed_map_{i}_{slug}.png"))
    print(f"\nOutputs written to {os.path.abspath(args.out)}/")
    return results, comparison, od, aadt


if __name__ == "__main__":
    main()
