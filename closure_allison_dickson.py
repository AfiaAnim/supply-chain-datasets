"""
Simulated closure at Allison Avenue - Dickson Avenue (Hanahan / North Charleston, SC).

Question
    If the Allison Ave - Dickson Ave intersection is closed (flooding, crash,
    work zone), how do trips from the surrounding streets change?
    Normal routes are compared with closure routes on
      - travel distance,
      - estimated travel time,
      - destinations that become unreachable (and streets that are cut off).

Method
    1. Road network: a downloaded OpenStreetMap .osm file (--osm-file), or a
       download of all drivable roads around Hanahan Middle School.
    2. Closure: the node where Allison Avenue meets Dickson Avenue. By default
       every approach is closed (nobody drives through or turns there);
       --close dickson / --close allison closes only that street's approaches.
    3. Origins: every intersection within --area-radius-m of the closure.
       Destinations: Hanahan Middle School plus any --destination you add.
    4. For each destination, one reverse Dijkstra (fastest free-flow route)
       with all roads open and one with the closure, giving every origin's
       normal and closure route, distance and time.
    5. Cut-off streets: intersections that can no longer be driven to AND out
       of the main road network once the closure is in place.

Data
    Road geometry, names, speed limits: OpenStreetMap (openstreetmap.org),
    via OSMnx. Missing speed limits are filled by road type (DEFAULT_SPEEDS_MPH).
    Travel times are free-flow estimates (no signals, turns or congestion).

Requirements
    pip install osmnx networkx pandas matplotlib

Usage
    python closure_allison_dickson.py --osm-file map.osm
    python closure_allison_dickson.py                      # downloads the roads instead
    Jupyter:  %run closure_allison_dickson.py --osm-file map.osm
              or paste into a cell, then  main(["--osm-file", "map.osm"])
    Options:
      --close intersection|dickson|allison    what is closed (default intersection)
      --closure-latlon LAT LON                pick the intersection yourself
      --destination "Name" LAT LON            add a destination (repeatable)
      --area-radius-m 800                     size of the origin area
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

CLOSURE_STREETS = ("Allison Avenue", "Dickson Avenue")
SCHOOL = ("Hanahan Middle School", 32.90752, -79.994)
DOWNLOAD_RADIUS_M = 3500  # around the school; covers the Allison/Dickson area
# The intersection is at about (32.9046, -80.0028); this box holds it, the school and the
# side streets drivers would use to go around it. Smaller maps have no detours.
EXPORT_BOX = (32.895, 32.915, -80.015, -79.985)  # min lat, max lat, min lon, max lon
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
    G = ox.simplify_graph(G)
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
            return ox.graph_from_point(center, dist=radius_m, network_type="drive")
        except Exception as e:  # busy server, timeout, bad gateway, ...
            last = e
            print(f"  {url} failed ({type(e).__name__}); trying the next server")
    raise RuntimeError("Every Overpass server failed; use --osm-file or try later.") from last


def add_travel_times(G):
    G = ox.add_edge_speeds(G, hwy_speeds={k: v * 1.609344 for k, v in DEFAULT_SPEEDS_MPH.items()},
                           fallback=25 * 1.609344)
    return ox.add_edge_travel_times(G)


# --- closure ----------------------------------------------------------------------------

def find_intersection(G, street_a, street_b, near=None, max_gap_m=60):
    """Node(s) where street_a meets street_b; a complex junction can be several nodes."""
    na = {n for u, v, d in G.edges(data=True) if on_street(d, street_a) for n in (u, v)}
    nb = {n for u, v, d in G.edges(data=True) if on_street(d, street_b) for n in (u, v)}
    if not na or not nb:
        missing = [s for s, ns in ((street_a, na), (street_b, nb)) if not ns]
        raise SystemExit(f"{' and '.join(missing)} not found in the road network. "
                         "Check the spelling, export a larger area, or pass --closure-latlon LAT LON.")
    shared = sorted(na & nb)
    if near is not None:
        pool = shared or sorted(na)
        centre = min(pool, key=lambda n: dist_m(G, n, *near))
    elif shared:
        centre = shared[0]
        far = [n for n in shared if dist_m(G, n, G.nodes[centre]["y"], G.nodes[centre]["x"]) > max_gap_m]
        if far:
            print(f"NOTE: {street_a} meets {street_b} at more than one place:")
            for n in shared:
                print(f"    ({G.nodes[n]['y']:.6f}, {G.nodes[n]['x']:.6f})")
            print("  Using the first; pass --closure-latlon LAT LON to choose another.")
    else:
        gap, centre = min((dist_m(G, a, G.nodes[b]["y"], G.nodes[b]["x"]), a) for a in na for b in nb)
        print(f"NOTE: {street_a} and {street_b} do not share a junction in OSM; closest points are "
              f"{gap:.0f} m apart. Using ({G.nodes[centre]['y']:.6f}, {G.nodes[centre]['x']:.6f}).")
    cy, cx = G.nodes[centre]["y"], G.nodes[centre]["x"]
    # divided roads / slip lanes: every shared node close to the centre is the same junction
    return [n for n in (shared or [centre]) if dist_m(G, n, cy, cx) <= max_gap_m] or [centre]


def closed_edges(G, nodes, mode):
    """Edges removed by the closure."""
    out = []
    for u, v, k, d in G.edges(keys=True, data=True):
        if u not in nodes and v not in nodes:
            continue
        if mode == "intersection" or on_street(d, f"{mode.capitalize()} Avenue"):
            out.append((u, v, k))
    return out


# --- routing ----------------------------------------------------------------------------

def routes_to(G, dest, blocked):
    """Fastest route from every node to dest (one reverse Dijkstra), avoiding blocked edges."""
    def w(u, v, dd):  # reversed graph: the original edge is v -> u
        costs = [d["travel_time"] for k, d in dd.items() if (v, u, k) not in blocked]
        return min(costs) if costs else None
    times, paths = nx.single_source_dijkstra(G.reverse(copy=False), dest, weight=w)
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


# --- map --------------------------------------------------------------------------------------

def plot_map(G, od, cut, closure_nodes, removed, destinations, example, path):
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(11, 10))
    for u, v, k, d in G.edges(keys=True, data=True):
        major = first(d.get("highway")) in MAJOR
        xs, ys = edge_xy(G, u, v, k)
        ax.plot(xs, ys, color="#a8a8a8" if major else "#d4d4d4", lw=1.4 if major else 0.7, zorder=1)
    for e in removed:
        xs, ys = edge_xy(G, *e)
        ax.plot(xs, ys, color="#d62728", lw=5, solid_capstyle="round", zorder=4)
    cx = [G.nodes[n]["x"] for n in closure_nodes]
    cy = [G.nodes[n]["y"] for n in closure_nodes]
    ax.scatter(cx, cy, marker="X", s=260, color="#d62728", edgecolor="black", zorder=8)

    if example is not None:
        for edges, color, style, z in ((example["normal_edges"], "#1f77b4", "-", 5),
                                       (example["closure_edges"], "#ff7f0e", "--", 6)):
            for e in edges:
                xs, ys = edge_xy(G, *e)
                ax.plot(xs, ys, color=color, lw=3.5, ls=style, zorder=z)

    # origins coloured by what the closure does to their trip to the first destination
    first_dest = destinations[0][0]
    sub = od[od["destination"] == first_dest]
    style = {"unchanged": ("#9e9e9e", 18), "longer": ("#ff7f0e", 30), "unreachable": ("#000000", 40)}
    for status, (color, size) in style.items():
        grp = sub[sub["status"] == status]
        ax.scatter(grp["lon"], grp["lat"], s=size, color=color, edgecolor="white", linewidth=0.5, zorder=7)
    if len(cut):
        ax.scatter(cut["lon"], cut["lat"], marker="s", s=40, facecolor="none", edgecolor="black",
                   linewidth=1.2, zorder=7)
    for name, lat, lon in destinations:
        ax.scatter([lon], [lat], marker="*", s=380, color="gold", edgecolor="black", zorder=9)
        ax.annotate(name, (lon, lat), xytext=(9, 9), textcoords="offset points", fontsize=9,
                    fontweight="bold", zorder=10, path_effects=[pe.withStroke(linewidth=3, foreground="white")])

    # frame: origin area plus destinations and the example routes
    xs, ys = list(sub["lon"]) + [d[2] for d in destinations], list(sub["lat"]) + [d[1] for d in destinations]
    if example is not None:
        for e in example["normal_edges"] + example["closure_edges"]:
            ex, ey = edge_xy(G, *e)
            xs += ex
            ys += ey
    px, py = (max(xs) - min(xs)) * 0.08 + 0.002, (max(ys) - min(ys)) * 0.08 + 0.002
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
    focus = {CLOSURE_STREETS[0], CLOSURE_STREETS[1]}
    order = sorted(best.items(), key=lambda kv: (not any(norm(f) in norm(kv[0]) for f in focus),
                                                 not kv[1][5], -kv[1][0]))
    for name, (_, ex, ey, mx, my, major) in order:
        (a0, b0), (a1, b1), (am, bm) = ax.transData.transform([(ex[0], ey[0]), (ex[-1], ey[-1]), (mx, my)])
        if any((am - a) ** 2 + (bm - b) ** 2 < 42 ** 2 for a, b in placed):
            continue
        placed.append((am, bm))
        ang = math.degrees(math.atan2(b1 - b0, a1 - a0))
        ang = ang - 180 if ang > 90 else ang + 180 if ang < -90 else ang
        key = any(norm(f) in norm(name) for f in focus)
        ax.text(mx, my, name, fontsize=8 if key else (6.5 if major else 5.5),
                fontweight="bold" if key else "normal", color="#b22222" if key else "#444444",
                rotation=ang, rotation_mode="anchor", ha="center", va="center", zorder=11,
                path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

    handles = [
        Line2D([], [], marker="X", ls="", color="#d62728", markeredgecolor="black", markersize=13,
               label="Closed intersection (Allison Ave - Dickson Ave)"),
        Line2D([], [], color="#d62728", lw=5, label="Closed approaches"),
    ]
    if example is not None:
        handles += [Line2D([], [], color="#1f77b4", lw=3.5, label=f"Normal route ({example['label']})"),
                    Line2D([], [], color="#ff7f0e", lw=3.5, ls="--", label="Closure route (same trip)")]
    handles += [
        Line2D([], [], marker="o", ls="", color="#9e9e9e", label=f"Origin: trip to {first_dest} unchanged"),
        Line2D([], [], marker="o", ls="", color="#ff7f0e", label="Origin: trip longer because of closure"),
        Line2D([], [], marker="o", ls="", color="#000000", label="Origin: destination unreachable"),
        Line2D([], [], marker="s", ls="", markerfacecolor="none", markeredgecolor="black",
               label="Cut off from the road network"),
        Line2D([], [], marker="*", ls="", color="gold", markeredgecolor="black", markersize=15, label="Destination"),
    ]
    counts = sub["status"].value_counts()
    ax.set_title(f"Simulated closure: Allison Ave - Dickson Ave\nTrips to {first_dest}: "
                 f"{counts.get('longer', 0)} longer, {counts.get('unreachable', 0)} unreachable, "
                 f"{counts.get('unchanged', 0)} unchanged", fontsize=11)
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8.5, frameon=False)
    fig.subplots_adjust(bottom=0.17, top=0.93, left=0.02, right=0.98)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    if "ipykernel" in sys.modules:
        from IPython.display import Image, display
        display(Image(filename=path, width=900))


# --- main -------------------------------------------------------------------------------------

def main(argv=None):
    """argv: e.g. main(["--osm-file", "map.osm"]) from a notebook."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--osm-file", default=None, help="downloaded OpenStreetMap .osm file")
    ap.add_argument("--radius-m", type=float, default=DOWNLOAD_RADIUS_M,
                    help="download radius around the school when no --osm-file is given")
    ap.add_argument("--close", choices=["intersection", "dickson", "allison"], default="intersection",
                    help="close every approach (default) or only Dickson Ave / Allison Ave approaches")
    ap.add_argument("--closure-latlon", type=float, nargs=2, metavar=("LAT", "LON"), default=None,
                    help="coordinates of the intersection to close (if the name lookup picks the wrong one)")
    ap.add_argument("--destination", nargs=3, action="append", default=[], metavar=("NAME", "LAT", "LON"),
                    help='add a destination, e.g. --destination "Fire station" 32.91 -79.99')
    ap.add_argument("--area-radius-m", type=float, default=800,
                    help="origins = intersections within this distance of the closure")
    ap.add_argument("--out", default="closure_allison_dickson_results", help="output folder")
    ap.add_argument("--no-plot", action="store_true")
    argv = sys.argv[1:] if argv is None else [str(a) for a in argv]
    args, _ = ap.parse_known_args(argv)  # ignores Jupyter's "-f kernel.json"

    global T0
    T0 = time.time()
    run_date = dt.datetime.now().astimezone()
    destinations = [SCHOOL] + [(n, float(a), float(b)) for n, a, b in args.destination]

    # 1. network
    G = load_osm_file(args.osm_file) if args.osm_file else download_network(SCHOOL[1:], args.radius_m)
    G = add_travel_times(G)
    step(f"network ready: {G.number_of_nodes()} intersections, {G.number_of_edges()} road segments")
    if G.number_of_nodes() < MIN_NODES:
        print(f"WARNING: only {G.number_of_nodes()} intersections. The map is too small to contain detour "
              "streets, so trips through the closure will look 'unreachable' when in reality drivers "
              f"would go around. Export a larger area (e.g. lat {EXPORT_BOX[0]}..{EXPORT_BOX[1]}, "
              f"lon {EXPORT_BOX[2]}..{EXPORT_BOX[3]}) before using the results.")

    # 2. closure
    closure_nodes = find_intersection(G, *CLOSURE_STREETS, near=args.closure_latlon)
    removed = closed_edges(G, set(closure_nodes), args.close)
    blocked = set(removed)
    c_lat = sum(G.nodes[n]["y"] for n in closure_nodes) / len(closure_nodes)
    c_lon = sum(G.nodes[n]["x"] for n in closure_nodes) / len(closure_nodes)
    step(f"closure at {node_street(G, closure_nodes[0])} ({c_lat:.6f}, {c_lon:.6f}): "
         f"{len({frozenset(e[:2]) for e in removed})} approach segment(s) closed ({args.close})")

    # 3. origins and destinations
    origins = [n for n in G.nodes if n not in closure_nodes and dist_m(G, n, c_lat, c_lon) <= args.area_radius_m]
    dest_nodes = []
    for name, lat, lon in destinations:
        dn = min(G.nodes, key=lambda n: dist_m(G, n, lat, lon))
        if dist_m(G, dn, lat, lon) > 300:
            print(f"WARNING: {name} is {dist_m(G, dn, lat, lon):.0f} m from the nearest road in the network; "
                  "export a larger area.")
        dest_nodes.append((name, dn))
    step(f"{len(origins)} origin intersections within {args.area_radius_m:.0f} m; "
         f"{len(dest_nodes)} destination(s)")

    # 4. normal vs closure routes
    rows, examples = [], {}
    for name, dn in dest_nodes:
        t_n, p_n = routes_to(G, dn, set())
        t_c, p_c = routes_to(G, dn, blocked)
        for o in origins:
            if o not in t_n or o == dn:
                continue  # not reachable even with all roads open (edge of the map)
            e_n = path_edges(G, p_n[o], set())
            row = {"destination": name, "origin_node": o, "origin": node_street(G, o),
                   "lat": G.nodes[o]["y"], "lon": G.nodes[o]["x"],
                   "normal_distance_mi": sum(G.edges[e]["length"] for e in e_n) / M_PER_MI,
                   "normal_time_min": t_n[o] / 60, "normal_route": street_sequence(G, e_n)}
            if o in t_c:
                e_c = path_edges(G, p_c[o], blocked)
                row.update(closure_distance_mi=sum(G.edges[e]["length"] for e in e_c) / M_PER_MI,
                           closure_time_min=t_c[o] / 60, closure_route=street_sequence(G, e_c))
                row["extra_distance_mi"] = row["closure_distance_mi"] - row["normal_distance_mi"]
                row["extra_time_min"] = row["closure_time_min"] - row["normal_time_min"]
                row["status"] = "longer" if row["extra_time_min"] > 0.05 else "unchanged"
                if row["status"] == "longer" and (name not in examples
                                                  or row["extra_time_min"] > examples[name]["extra"]):
                    examples[name] = {"extra": row["extra_time_min"], "normal_edges": e_n,
                                      "closure_edges": e_c, "label": f"{node_street(G, o)} to {name}"}
            else:
                row["status"] = "unreachable"
            rows.append(row)
    od = pd.DataFrame(rows)

    # 5. cut-off streets: lose two-way connection to the main network
    main_before = max(nx.strongly_connected_components(G), key=len)
    H = G.copy()
    H.remove_edges_from(removed)
    main_after = max(nx.strongly_connected_components(H), key=len)
    cut_nodes = [n for n in main_before if n not in main_after and n not in closure_nodes]
    cut = pd.DataFrame([{"node": n, "location": node_street(G, n), "lat": G.nodes[n]["y"],
                         "lon": G.nodes[n]["x"]} for n in cut_nodes])

    # 6. results tables
    tables = []
    for name, _ in dest_nodes:
        sub = od[od["destination"] == name]
        both = sub[sub["status"] != "unreachable"]
        unreachable = int((sub["status"] == "unreachable").sum())
        tables.append(pd.DataFrame([
            {"Destination": name, "Condition": "All roads open",
             "Avg distance": f"{both['normal_distance_mi'].mean():.2f} mi",
             "Avg travel time": f"{both['normal_time_min'].mean():.1f} min",
             "Origins that can reach it": len(sub)},
            {"Destination": name, "Condition": "Allison-Dickson closed",
             "Avg distance": f"{both['closure_distance_mi'].mean():.2f} mi",
             "Avg travel time": f"{both['closure_time_min'].mean():.1f} min",
             "Origins that can reach it": len(sub) - unreachable},
            {"Destination": name, "Condition": "Difference",
             "Avg distance": f"{both['extra_distance_mi'].mean() + 0:+.2f} mi",
             "Avg travel time": f"{both['extra_time_min'].mean() + 0:+.1f} min",
             "Origins that can reach it": f"{unreachable} become unreachable"},
        ]))
    results = pd.concat(tables, ignore_index=True)

    print(f"\n=== Simulated closure: Allison Ave - Dickson Ave ({args.close}) ===")
    print(f"Origins: {len(origins)} intersections within {args.area_radius_m:.0f} m. "
          "Averages use origins that can reach the destination in both cases.\n")
    print(results.to_string(index=False))
    for name, _ in dest_nodes:
        sub = od[od["destination"] == name]
        print(f"\n{name}: {int((sub['status'] == 'longer').sum())} trips longer, "
              f"{int((sub['status'] == 'unreachable').sum())} unreachable, "
              f"{int((sub['status'] == 'unchanged').sum())} unchanged")
        longer = sub[sub["status"] == "longer"].sort_values("extra_time_min", ascending=False)
        if len(longer):
            print(longer[["origin", "normal_distance_mi", "closure_distance_mi", "normal_time_min",
                          "closure_time_min", "extra_time_min"]].head(5).round(2).to_string(index=False))
            top = longer.iloc[0]
            print(f"  Most affected: from {top['origin']}\n    normal : {top['normal_route']}\n"
                  f"    closure: {top['closure_route']}")
        unreach = sub[sub["status"] == "unreachable"]
        if len(unreach):
            print("  Unreachable from:", "; ".join(unreach["origin"].head(10)))
    print(f"\nCut off from the road network by the closure: {len(cut)} intersection(s)"
          + (": " + "; ".join(cut["location"].head(10)) if len(cut) else ""))

    os.makedirs(args.out, exist_ok=True)
    results.to_csv(os.path.join(args.out, "results_table.csv"), index=False)
    od.to_csv(os.path.join(args.out, "origin_destination_routes.csv"), index=False)
    cut.to_csv(os.path.join(args.out, "cut_off_locations.csv"), index=False)
    with open(os.path.join(args.out, "run_record.json"), "w") as f:
        json.dump({
            "scenario": f"closure of Allison Avenue - Dickson Avenue ({args.close})",
            "closure_nodes": [{"osm_node": int(n), "lat": G.nodes[n]["y"], "lon": G.nodes[n]["x"]}
                              for n in closure_nodes],
            "closed_approach_segments": len({frozenset(e[:2]) for e in removed}),
            "origins": f"{len(origins)} intersections within {args.area_radius_m} m of the closure",
            "destinations": [{"name": n, "lat": a, "lon": b} for n, a, b in destinations],
            "data_source": (f"OpenStreetMap file {args.osm_file}" if args.osm_file else
                            "OpenStreetMap via Overpass API") + f", OSMnx {ox.__version__}",
            "speeds": "OSM maxspeed where tagged, else road-type mean / DEFAULT_SPEEDS_MPH; free flow",
            "run_date": run_date.isoformat(timespec="seconds"),
        }, f, indent=2)
    if not args.no_plot and len(od):
        plot_map(G, od, cut, closure_nodes, removed, destinations,
                 examples.get(destinations[0][0]), os.path.join(args.out, "closure_map.png"))
    print(f"\nOutputs written to {os.path.abspath(args.out)}/")
    return results, od, cut


if __name__ == "__main__":
    main()
