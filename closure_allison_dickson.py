"""
Simulated closure at Allison Avenue - Dickson Avenue (Hanahan / North Charleston, SC).

Question
    If the Allison Ave - Dickson Ave intersection is closed (flooding, crash,
    work zone), how do trips from the surrounding streets change?
    Normal routes are compared with closure routes on
      - travel distance,
      - estimated travel time,
      - destinations that become unreachable (and streets that are cut off),
    for three kinds of destination, to see whether the consequences depend on
    where people need to go:
      - Hanahan Middle School,
      - the nearest healthcare facility (hospital / clinic / doctors in the
        OpenStreetMap data, or one you name),
      - a connection to Remount Road (reaching any point on Remount Road).

Method
    1. Road network: a downloaded OpenStreetMap .osm file (--osm-file), or a
       download of all drivable roads around Hanahan Middle School.
    2. Closure: the node where Allison Avenue meets Dickson Avenue. By default
       every approach is closed (nobody drives through or turns there);
       --close dickson / --close allison closes only that street's approaches.
       Movement check: every movement through the intersection (each way in
       x each way out: through, left, right) is listed and verified to be
       blocked, both in the closed network and by routing each movement and
       confirming the route no longer passes through the junction.
    3. Origins: every intersection within --area-radius-m of the closure.
    4. For each destination, one reverse Dijkstra (fastest free-flow route)
       with all roads open and one with the closure, giving every origin's
       normal and closure route, distance and time. For Remount Road the
       search starts from every Remount Road node, so each trip goes to its
       best access point, which is reported.
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
      --healthcare "Name" LAT LON             use this healthcare facility instead of the nearest in OSM
      --destination "Name" LAT LON            add another destination (repeatable)
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
EXPORT_BOX = (32.885, 32.915, -80.020, -79.985)  # min lat, max lat, min lon, max lon (incl. Remount Rd)
REMOUNT = "Remount Road"
DICKSON_AADT = 1900  # vehicles/day on Dickson Avenue (user-supplied count)
HEALTHCARE_TAGS = {"amenity": ["hospital", "clinic", "doctors"], "healthcare": True}
JUNCTION_RADIUS_M = 25  # a route passing this close to the closed junction counts as going through it
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


def compass(bearing):
    return ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][int((bearing + 22.5) % 360 // 45)]


def movement_check(G, cluster, removed, mode):
    """Every movement through the junction (way in x way out) and whether it is blocked.

    A movement counts as blocked when the closed network has no open in-edge + out-edge pair
    for it. For a full intersection closure the fastest route between its two legs must also
    avoid the junction entirely (not pass through or within JUNCTION_RADIUS_M of it). For a
    one-street closure, routes may still cross the junction on the street left open.
    """
    cy = sum(G.nodes[n]["y"] for n in cluster) / len(cluster)
    cx = sum(G.nodes[n]["x"] for n in cluster) / len(cluster)
    removed = set(removed)
    H = G.copy()
    H.remove_edges_from(removed)

    def leg_info(e, inbound):
        xs, ys = edge_xy(G, *e)
        # point on the leg next to the junction gives the leg's direction from the junction
        px, py = (xs[-2], ys[-2]) if inbound else (xs[1], ys[1])
        return first(G.edges[e].get("name")) or "(unnamed)", ox.bearing.calculate_bearing(cy, cx, py, px)

    legs_in, legs_out = {}, {}
    for c in cluster:
        for u, _, k in G.in_edges(c, keys=True):
            if u not in cluster:
                legs_in.setdefault(u, []).append((u, c, k))
        for _, v, k in G.out_edges(c, keys=True):
            if v not in cluster:
                legs_out.setdefault(v, []).append((c, v, k))
    street_of_mode = None if mode == "intersection" else f"{mode.capitalize()} Avenue"

    rows = []
    for u, ins in legs_in.items():
        for v, outs in legs_out.items():
            if u == v:
                continue  # U-turn back onto the same leg
            s_in, b_in = leg_info(ins[0], True)
            s_out, b_out = leg_info(outs[0], False)
            heading_in, heading_out = (b_in + 180) % 360, b_out  # direction of travel
            turn = (heading_out - heading_in + 540) % 360 - 180
            kind = ("through" if abs(turn) < 35 else "right turn" if 35 <= turn <= 145
                    else "left turn" if -145 <= turn <= -35 else "sharp turn")
            expected = street_of_mode is None or any(norm(street_of_mode) in norm(x) for x in (s_in, s_out))
            open_in = any(e not in removed for e in ins)
            open_out = any(e not in removed for e in outs)
            model_blocked = not (open_in and open_out)
            normal_len = min(G.edges[e]["length"] for e in ins) + min(G.edges[e]["length"] for e in outs)
            normal_t = min(G.edges[e]["travel_time"] for e in ins) + min(G.edges[e]["travel_time"] for e in outs)
            plen = pt = None
            try:
                path = nx.shortest_path(H, u, v, weight="travel_time")
                inner = path[1:-1]
                through = any(n in cluster or dist_m(G, n, cy, cx) <= JUNCTION_RADIUS_M for n in inner)
                best = [min(H[a][b].values(), key=lambda d: d["travel_time"]) for a, b in zip(path[:-1], path[1:])]
                plen = sum(d["length"] for d in best)
                pt = sum(d["travel_time"] for d in best)
                route = (("PASSES THROUGH junction, " if mode == "intersection" else "detour via open street, ")
                         if through else "detour ") + f"{plen / M_PER_MI:.2f} mi"
            except nx.NetworkXNoPath:
                through, route = False, "no route (map edge or dead end)"
            blocked = model_blocked and not (through and mode == "intersection")
            rows.append({"from": f"{s_in} ({compass(b_in)} leg)", "to": f"{s_out} ({compass(b_out)} leg)",
                         "movement": kind, "should_be_blocked": expected, "blocked_in_model": blocked,
                         "normal_movement_mi": round(normal_len / M_PER_MI, 2),
                         "normal_movement_min": round(normal_t / 60, 2),
                         "closure_movement_mi": round(plen / M_PER_MI, 2) if plen is not None else None,
                         "closure_movement_min": round(pt / 60, 2) if pt is not None else None,
                         "with_closure": "still open (as intended)" if not expected and not blocked else route,
                         "check": "PASS" if blocked == expected else "FAIL"})
    leftover = [e for e in G.edges(keys=True) if (e[0] in cluster or e[1] in cluster) and e not in removed]
    return pd.DataFrame(rows), leftover


def find_healthcare(args, c_lat, c_lon):
    """Nearest hospital / clinic / doctors in the OSM data, or the one given with --healthcare."""
    if args.healthcare:
        name, lat, lon = args.healthcare
        return name, float(lat), float(lon)
    try:
        gdf = (ox.features_from_xml(args.osm_file, tags=HEALTHCARE_TAGS) if args.osm_file
               else ox.features_from_point(SCHOOL[1:], tags=HEALTHCARE_TAGS, dist=args.radius_m))
    except Exception as e:  # none in the data, or server failure
        print(f"  no healthcare facility found in the map data ({type(e).__name__})")
        return None
    found = []
    for _, r in gdf.iterrows():
        pt = r.geometry.representative_point()
        kind = next((str(r[c]) for c in ("amenity", "healthcare") if c in r and isinstance(r[c], str)), "healthcare")
        name = r["name"] if "name" in r and isinstance(r["name"], str) else f"Unnamed {kind}"
        found.append((ox.distance.great_circle(c_lat, c_lon, pt.y, pt.x), f"{name} ({kind})", pt.y, pt.x))
    if not found:
        return None
    found.sort()
    print("  healthcare facilities in the map data (nearest first):")
    for d, name, lat, lon in found[:5]:
        print(f"    {name}: {d:.0f} m from the closure ({lat:.5f}, {lon:.5f})")
    return found[0][1], found[0][2], found[0][3]


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


# --- map --------------------------------------------------------------------------------------

def plot_map(G, od, cut, closure_nodes, removed, dest, example, path):
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

    # origins coloured by what the closure does to their trip to this destination
    first_dest = dest["name"]
    sub = od[od["destination"] == first_dest]
    style = {"unchanged": ("#9e9e9e", 18), "longer": ("#ff7f0e", 30), "unreachable": ("#000000", 40)}
    for status, (color, size) in style.items():
        grp = sub[sub["status"] == status]
        ax.scatter(grp["lon"], grp["lat"], s=size, color=color, edgecolor="white", linewidth=0.5, zorder=7)
    if len(cut):
        ax.scatter(cut["lon"], cut["lat"], marker="s", s=40, facecolor="none", edgecolor="black",
                   linewidth=1.2, zorder=7)
    xs, ys = list(sub["lon"]), list(sub["lat"])
    if dest["kind"] == "road":
        for e in dest["road_edges"]:
            ex, ey = edge_xy(G, *e)
            ax.plot(ex, ey, color="gold", lw=6, alpha=0.8, solid_capstyle="round", zorder=2)
    else:
        ax.scatter([dest["lon"]], [dest["lat"]], marker="*", s=380, color="gold", edgecolor="black", zorder=9)
        ax.annotate(dest["name"], (dest["lon"], dest["lat"]), xytext=(9, 9), textcoords="offset points",
                    fontsize=9, fontweight="bold", zorder=10,
                    path_effects=[pe.withStroke(linewidth=3, foreground="white")])
        xs.append(dest["lon"])
        ys.append(dest["lat"])

    # frame: origin area plus the destination and the example routes
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
        Line2D([], [], color="gold", lw=6, label=f"Destination: {dest['name']}") if dest["kind"] == "road" else
        Line2D([], [], marker="*", ls="", color="gold", markeredgecolor="black", markersize=15,
               label=f"Destination: {dest['name']}"),
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
    ap.add_argument("--healthcare", nargs=3, default=None, metavar=("NAME", "LAT", "LON"),
                    help="healthcare facility to use (default: nearest hospital/clinic/doctors in the OSM data)")
    ap.add_argument("--destination", nargs=3, action="append", default=[], metavar=("NAME", "LAT", "LON"),
                    help='add another destination, e.g. --destination "Fire station" 32.91 -79.99')
    ap.add_argument("--dickson-aadt", type=float, default=DICKSON_AADT,
                    help=f"annual average daily traffic on Dickson Ave (default {DICKSON_AADT})")
    ap.add_argument("--area-radius-m", type=float, default=800,
                    help="origins = intersections within this distance of the closure")
    ap.add_argument("--out", default="closure_allison_dickson_results", help="output folder")
    ap.add_argument("--no-plot", action="store_true")
    argv = sys.argv[1:] if argv is None else [str(a) for a in argv]
    args, _ = ap.parse_known_args(argv)  # ignores Jupyter's "-f kernel.json"

    global T0
    T0 = time.time()
    run_date = dt.datetime.now().astimezone()

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

    # 2b. check that every intended movement through the junction is blocked
    moves, leftover = movement_check(G, closure_nodes, removed, args.close)
    n_block = int(moves["should_be_blocked"].sum()) if len(moves) else 0
    n_fail = int((moves["check"] == "FAIL").sum()) if len(moves) else 0
    print(f"\n=== Movement check: {len(moves)} movements through the intersection, "
          f"{n_block} should be blocked ({args.close} closure) ===")
    if len(moves):
        print(moves.to_string(index=False))
    if args.close == "intersection" and leftover:
        print(f"FAIL: {len(leftover)} road segment(s) still touch the closed junction: {leftover}")
        n_fail += len(leftover)
    print(f"Movement check: {'PASS' if n_fail == 0 else 'FAIL'} - "
          + ("every intended movement is blocked." if n_fail == 0 else f"{n_fail} problem(s); see above.")
          + (" Note: OSM turn restrictions are not modelled." if args.close != "intersection" else ""))

    # 2c. daily cost for Dickson Avenue traffic (AADT)
    aadt_rows = []
    thru = moves[(moves["movement"] == "through") & moves["from"].str.startswith(CLOSURE_STREETS[1])
                 & moves["to"].str.startswith(CLOSURE_STREETS[1])] if len(moves) else moves
    for _, m in thru.iterrows():
        veh = args.dickson_aadt / max(len(thru), 1)  # AADT split evenly by direction
        if not m["should_be_blocked"]:
            extra_t, extra_d, note = 0.0, 0.0, "not affected (Dickson Ave stays open)"
        elif pd.isna(m["closure_movement_min"]):
            extra_t = extra_d = float("nan")
            note = "no detour inside this map"
        else:
            extra_t = m["closure_movement_min"] - m["normal_movement_min"]
            extra_d = m["closure_movement_mi"] - m["normal_movement_mi"]
            note = "detours around the closure"
        aadt_rows.append({"direction": f"{m['from']} -> {m['to']}", "vehicles_per_day": round(veh),
                          "extra_min_per_vehicle": round(extra_t, 2),
                          "extra_mi_per_vehicle": round(extra_d, 2),
                          "extra_vehicle_hours_per_day": round(veh * extra_t / 60, 1),
                          "extra_vehicle_miles_per_day": round(veh * extra_d, 1), "note": note})
    aadt = pd.DataFrame(aadt_rows)
    if len(aadt):
        print(f"\n=== Daily impact on Dickson Ave traffic (AADT {args.dickson_aadt:,.0f} vehicles/day) ===")
        print(aadt.to_string(index=False))
        if aadt["extra_vehicle_hours_per_day"].isna().any():
            print("Total: unknown - at least one direction has no detour inside this map "
                  "(export a larger area).")
        else:
            print(f"Total: {aadt['extra_vehicle_hours_per_day'].sum():.1f} extra vehicle-hours/day, "
                  f"{aadt['extra_vehicle_miles_per_day'].sum():.0f} extra vehicle-miles/day.")
        print("Assumes all Dickson Ave traffic passes straight through the intersection, split evenly "
              "by direction, at free-flow speeds.")

    # 3. origins and destinations
    origins = [n for n in G.nodes if n not in closure_nodes and dist_m(G, n, c_lat, c_lon) <= args.area_radius_m]
    points = [SCHOOL]
    hc = find_healthcare(args, c_lat, c_lon)
    if hc:
        points.append(hc)
    else:
        print("WARNING: no healthcare facility found; pass --healthcare NAME LAT LON to add one.")
    points += [(n, float(a), float(b)) for n, a, b in args.destination]
    dests = []
    for name, lat, lon in points:
        dn = min(G.nodes, key=lambda n: dist_m(G, n, lat, lon))
        if dist_m(G, dn, lat, lon) > 300:
            print(f"WARNING: {name} is {dist_m(G, dn, lat, lon):.0f} m from the nearest road in the network; "
                  "export a larger area.")
        dests.append({"name": name, "kind": "point", "lat": lat, "lon": lon, "targets": [dn]})
    remount_edges = [(u, v, k) for u, v, k, d in G.edges(keys=True, data=True) if on_street(d, REMOUNT)]
    if remount_edges:
        dests.append({"name": f"{REMOUNT} (any access point)", "kind": "road", "road_edges": remount_edges,
                      "targets": sorted({n for e in remount_edges for n in e[:2]})})
    else:
        print(f"WARNING: {REMOUNT} is not in the map; export an area reaching it (see EXPORT_BOX).")
    step(f"{len(origins)} origin intersections within {args.area_radius_m:.0f} m; "
         f"{len(dests)} destination(s): " + "; ".join(d["name"] for d in dests))

    # 4. normal vs closure routes
    rows, examples = [], {}
    for dest in dests:
        name = dest["name"]
        t_n, p_n = routes_to(G, dest["targets"], set())
        t_c, p_c = routes_to(G, dest["targets"], blocked)
        for o in origins:
            if o not in t_n or o in dest["targets"]:
                continue  # not reachable even with all roads open (edge of the map), or already there
            e_n = path_edges(G, p_n[o], set())
            row = {"destination": name, "origin_node": o, "origin": node_street(G, o),
                   "lat": G.nodes[o]["y"], "lon": G.nodes[o]["x"],
                   "normal_distance_mi": sum(G.edges[e]["length"] for e in e_n) / M_PER_MI,
                   "normal_time_min": t_n[o] / 60, "normal_route": street_sequence(G, e_n),
                   "normal_arrives_at": node_street(G, p_n[o][-1])}
            if o in t_c:
                e_c = path_edges(G, p_c[o], blocked)
                row.update(closure_distance_mi=sum(G.edges[e]["length"] for e in e_c) / M_PER_MI,
                           closure_time_min=t_c[o] / 60, closure_route=street_sequence(G, e_c),
                           closure_arrives_at=node_street(G, p_c[o][-1]))
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
    tables, compare = [], []
    for dest in dests:
        name = dest["name"]
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
             "Avg distance": f"{round(both['extra_distance_mi'].mean(), 2) + 0.0:+.2f} mi",
             "Avg travel time": f"{round(both['extra_time_min'].mean(), 1) + 0.0:+.1f} min",
             "Origins that can reach it": f"{unreachable} become unreachable"},
        ]))
        hit = both[both["status"] == "longer"]
        compare.append({"Destination": name, "Origins": len(sub), "Trips longer": len(hit),
                        "Unreachable": unreachable,
                        "Avg extra time, affected trips (min)": round(hit["extra_time_min"].mean(), 2) + 0.0 if len(hit) else 0.0,
                        "Max extra time (min)": round(hit["extra_time_min"].max(), 2) + 0.0 if len(hit) else 0.0,
                        "Avg extra distance, affected trips (mi)": round(hit["extra_distance_mi"].mean(), 2) + 0.0 if len(hit) else 0.0})
    results = pd.concat(tables, ignore_index=True)
    comparison = pd.DataFrame(compare)

    print(f"\n=== Simulated closure: Allison Ave - Dickson Ave ({args.close}) ===")
    print(f"Origins: {len(origins)} intersections within {args.area_radius_m:.0f} m. "
          "Averages use origins that can reach the destination in both cases.\n")
    print(results.to_string(index=False))
    for dest in dests:
        name = dest["name"]
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
            if dest["kind"] == "road":
                moved = longer[longer["normal_arrives_at"] != longer["closure_arrives_at"]]
                print(f"  {len(moved)} of these trips join {REMOUNT} at a different access point with the closure.")
        unreach = sub[sub["status"] == "unreachable"]
        if len(unreach):
            print("  Unreachable from:", "; ".join(unreach["origin"].head(10)))
    print(f"\nCut off from the road network by the closure: {len(cut)} intersection(s)"
          + (": " + "; ".join(cut["location"].head(10)) if len(cut) else ""))

    print("\n=== Does the impact depend on the destination? ===")
    print(comparison.to_string(index=False))
    if len(comparison) > 1:
        # report the leader on each measure; a single "worst" ranking would hide trade-offs
        for col, label in (("Trips longer", "Most trips made longer"), ("Unreachable", "Most origins cut off"),
                           ("Max extra time (min)", "Largest single delay")):
            top = comparison.loc[comparison[col].idxmax()]
            same = (comparison[col] == top[col]).all()
            print(f"  {label}: " + ("same for every destination" if same
                                     else f"{top['Destination']} ({top[col]})"))

    os.makedirs(args.out, exist_ok=True)
    results.to_csv(os.path.join(args.out, "results_table.csv"), index=False)
    comparison.to_csv(os.path.join(args.out, "destination_comparison.csv"), index=False)
    moves.to_csv(os.path.join(args.out, "movement_check.csv"), index=False)
    aadt.to_csv(os.path.join(args.out, "dickson_aadt_impact.csv"), index=False)
    od.to_csv(os.path.join(args.out, "origin_destination_routes.csv"), index=False)
    cut.to_csv(os.path.join(args.out, "cut_off_locations.csv"), index=False)
    with open(os.path.join(args.out, "run_record.json"), "w") as f:
        json.dump({
            "scenario": f"closure of Allison Avenue - Dickson Avenue ({args.close})",
            "closure_nodes": [{"osm_node": int(n), "lat": G.nodes[n]["y"], "lon": G.nodes[n]["x"]}
                              for n in closure_nodes],
            "closed_approach_segments": len({frozenset(e[:2]) for e in removed}),
            "origins": f"{len(origins)} intersections within {args.area_radius_m} m of the closure",
            "dickson_aadt": args.dickson_aadt,
            "dickson_extra_vehicle_hours_per_day": (float(aadt["extra_vehicle_hours_per_day"].sum())
                                                    if len(aadt) and not aadt["extra_vehicle_hours_per_day"].isna().any()
                                                    else None),
            "movement_check": {"movements": len(moves), "should_be_blocked": n_block,
                               "result": "PASS" if n_fail == 0 else "FAIL"},
            "destinations": [{k: v for k, v in d.items() if k in ("name", "kind", "lat", "lon")}
                             for d in dests],
            "data_source": (f"OpenStreetMap file {args.osm_file}" if args.osm_file else
                            "OpenStreetMap via Overpass API") + f", OSMnx {ox.__version__}",
            "speeds": "OSM maxspeed where tagged, else road-type mean / DEFAULT_SPEEDS_MPH; free flow",
            "run_date": run_date.isoformat(timespec="seconds"),
        }, f, indent=2)
    if not args.no_plot and len(od):
        for i, dest in enumerate(dests, 1):
            slug = "".join(ch if ch.isalnum() else "_" for ch in dest["name"].lower()).strip("_")[:40]
            plot_map(G, od, cut, closure_nodes, removed, dest, examples.get(dest["name"]),
                     os.path.join(args.out, f"closure_map_{i}_{slug}.png"))
    print(f"\nOutputs written to {os.path.abspath(args.out)}/")
    return results, comparison, od, cut, moves, aadt


if __name__ == "__main__":
    main()
