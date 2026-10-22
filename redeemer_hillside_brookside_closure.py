"""
Simulated closures of Redeemer Road, Hillside Drive and Brookside Drive
(Hanahan / North Charleston, SC): residents' trips to three schools.

Question
    If Redeemer Road, Hillside Drive or Brookside Drive is closed (each on its
    own, and all three together), how do trips from the surrounding homes change
    for
      - Hanahan High School,
      - George R. Fishburne Elementary School,
      - Matilda F. Dunston Elementary School?
    Normal routes are compared with closure routes on travel distance,
    estimated travel time and schools that become unreachable. Comparison
    tables show whether the impact depends on which street closes and on which
    school people go to, and whether closing all three compounds the effect.

Method
    1. Road network: a downloaded OpenStreetMap .osm file (--osm-file), or a
       download of all drivable roads around the three streets.
    2. Scenarios: each street closed along its whole length (both directions),
       then all three together. Streets missing from the map are reported
       with similar names, in case OSM spells them differently.
    3. Residents: every intersection within --resident-radius-m of any of the
       streets; the same set is used in every scenario so results compare.
       Individual driveways are not in the road network.
    4. Schools: found by name in the map data (amenity=school); any school
       not found must be given with --school "Name" LAT LON.
    5. For each school, one reverse Dijkstra (fastest free-flow route) with all
       roads open and one per scenario.
    6. Cut-off locations: intersections that lose their two-way connection to
       the main road network in a scenario (e.g. streets whose only way out is
       the closed street).

Data
    Road geometry, names, speed limits and school locations: OpenStreetMap
    (openstreetmap.org) via OSMnx. Missing speed limits are filled by road type.
    Travel times are free-flow estimates (no signals, turns or congestion).

Requirements
    pip install osmnx networkx pandas matplotlib

Usage
    python redeemer_hillside_brookside_closure.py --osm-file map.osm
    Jupyter:  %run redeemer_hillside_brookside_closure.py --osm-file map.osm
              or paste into a cell, then  main(["--osm-file", "map.osm"])
    Options:
      --street "Hillside Drive"            close other streets instead (repeatable)
      --school "Name" LAT LON              school missing from the map data (repeatable)
      --resident-radius-m 400              size of the resident area
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

STREETS = ["Redeemer Road", "Hillside Drive", "Brookside Drive"]
AREA_CENTER = (32.905, -80.010)  # Hanahan / Berkeley Hills / Highland Park
DOWNLOAD_RADIUS_M = 4000
# Hillside Dr and Brookside Dr lie in Highland Park / Berkeley Hills near Hanahan High and
# Fishburne Elementary; export generously so every street and school is inside the map.
EXPORT_BOX = (32.880, 32.930, -80.040, -79.980)  # min lat, max lat, min lon, max lon
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



# --- streets and schools -------------------------------------------------------------------

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


def similar_streets(G, street):
    """Street names in the map that share the street's first word (spelling help)."""
    word = norm(street).split()[0]
    names = {first(d.get("name")) for _, _, d in G.edges(data=True)}
    return sorted(n for n in names if n and word in norm(n))


# --- map ----------------------------------------------------------------------------------

STATUS_RANK = {"unchanged": 0, "longer": 1, "unreachable": 2}


def plot_map(G, od, scenario, closed, schools, example, path):
    """One scenario: closed streets, each resident coloured by its worst outcome over all schools."""
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.lines import Line2D

    sub = od[od["scenario"] == scenario["name"]]
    worst = (sub.assign(rank=sub["status"].map(STATUS_RANK))
             .sort_values("rank").groupby("origin_node").last().reset_index())
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
        grp = worst[worst["status"] == status]
        ax.scatter(grp["lon"], grp["lat"], s=size, color=color, edgecolor="white", linewidth=0.5, zorder=7)
    xs, ys = list(worst["lon"]), list(worst["lat"])
    for s in schools:
        ax.scatter([s["lon"]], [s["lat"]], marker="*", s=380, color="gold", edgecolor="black", zorder=9)
        ax.annotate(s["name"], (s["lon"], s["lat"]), xytext=(9, 9), textcoords="offset points", fontsize=8.5,
                    fontweight="bold", zorder=10, path_effects=[pe.withStroke(linewidth=3, foreground="white")])
        xs.append(s["lon"])
        ys.append(s["lat"])
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

    # street names: one per street, on its longest visible segment; closed streets first, in red
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    best = {}
    for u, v, k, d in G.edges(keys=True, data=True):
        name = first(d.get("name"))
        ex, ey = edge_xy(G, u, v, k)
        mx, my = (ex[0] + ex[-1]) / 2, (ey[0] + ey[-1]) / 2
        if name and x0 < mx < x1 and y0 < my < y1 and (name not in best or d["length"] > best[name][0]):
            best[name] = (d["length"], ex, ey, mx, my, first(d.get("highway")) in MAJOR)
    is_closed = lambda n: any(norm(s) in norm(n) for s in scenario["streets"])  # noqa: E731
    placed = []
    for name, (_, ex, ey, mx, my, major) in sorted(best.items(),
                                                   key=lambda kv: (not is_closed(kv[0]), not kv[1][5], -kv[1][0])):
        (a0, b0), (a1, b1), (am, bm) = ax.transData.transform([(ex[0], ey[0]), (ex[-1], ey[-1]), (mx, my)])
        if any((am - a) ** 2 + (bm - b) ** 2 < 42 ** 2 for a, b in placed):
            continue
        placed.append((am, bm))
        ang = math.degrees(math.atan2(b1 - b0, a1 - a0))
        ang = ang - 180 if ang > 90 else ang + 180 if ang < -90 else ang
        key = is_closed(name)
        ax.text(mx, my, f"{name} (CLOSED)" if key else name, fontsize=8 if key else (6.5 if major else 5.5),
                fontweight="bold" if key else "normal", color="#b22222" if key else "#444444",
                rotation=ang, rotation_mode="anchor", ha="center", va="center", zorder=11,
                path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

    handles = [Line2D([], [], color="#d62728", lw=5, label="Closed: " + ", ".join(scenario["streets"]))]
    if example is not None:
        handles += [Line2D([], [], color="#1f77b4", lw=3.5,
                           label=f"Normal route ({example['label']})"),
                    Line2D([], [], color="#ff7f0e", lw=3.5, ls="--", label="Route with the closure")]
    handles += [
        Line2D([], [], marker="o", ls="", color="#9e9e9e", label="Resident: no school trip affected"),
        Line2D([], [], marker="o", ls="", color="#ff7f0e", label="Resident: at least one school trip longer"),
        Line2D([], [], marker="o", ls="", color="#000000", label="Resident: cannot reach at least one school"),
        Line2D([], [], marker="*", ls="", color="gold", markeredgecolor="black", markersize=15, label="School"),
    ]
    c = worst["status"].value_counts()
    ax.set_title(f"Scenario: {scenario['name']}\nResidents: {c.get('longer', 0)} with a longer school trip, "
                 f"{c.get('unreachable', 0)} cut off from a school, {c.get('unchanged', 0)} unaffected", fontsize=11)
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
                    help="download radius when no --osm-file is given")
    ap.add_argument("--street", action="append", default=None,
                    help=f"street to close (repeatable; default: {', '.join(STREETS)})")
    ap.add_argument("--school", nargs=3, action="append", default=[], metavar=("NAME", "LAT", "LON"),
                    help="school location if it is not in the map data (repeatable)")
    ap.add_argument("--resident-radius-m", type=float, default=400,
                    help="residents = intersections within this distance of a closed street")
    ap.add_argument("--out", default="redeemer_hillside_brookside_results", help="output folder")
    ap.add_argument("--no-plot", action="store_true")
    argv = sys.argv[1:] if argv is None else [str(a) for a in argv]
    args, _ = ap.parse_known_args(argv)  # ignores Jupyter's "-f kernel.json"

    global T0
    T0 = time.time()
    run_date = dt.datetime.now().astimezone()
    streets = args.street or STREETS

    # 1. network
    G = load_osm_file(args.osm_file) if args.osm_file else download_network(AREA_CENTER, args.radius_m)
    G = add_travel_times(G)
    step(f"network ready: {G.number_of_nodes()} intersections, {G.number_of_edges()} road segments")
    if G.number_of_nodes() < MIN_NODES:
        print(f"WARNING: only {G.number_of_nodes()} intersections; the map is too small to contain detour "
              f"streets. Export a larger area (e.g. lat {EXPORT_BOX[0]}..{EXPORT_BOX[1]}, "
              f"lon {EXPORT_BOX[2]}..{EXPORT_BOX[3]}).")

    # 2. streets and scenarios: each street alone, then all together
    street_edges = {}
    for st in streets:
        edges = [(u, v, k) for u, v, k, d in G.edges(keys=True, data=True) if on_street(d, st)]
        if edges:
            street_edges[st] = edges
            mi = sum(G.edges[e]["length"] for e in edges) / 2 / M_PER_MI  # two-way roads appear twice
            print(f"  {st}: {len({frozenset(e[:2]) for e in edges})} segments, {mi:.2f} mi")
        else:
            alt = similar_streets(G, st)
            print(f"WARNING: {st} is not in the map" + (f"; similar names: {alt}" if alt else "")
                  + ". Check the spelling (--street) or export a larger area.")
    if not street_edges:
        raise SystemExit("None of the streets is in the road network.")
    scenarios = [{"name": f"{st} closed", "streets": [st], "edges": e} for st, e in street_edges.items()]
    if len(street_edges) > 1:
        scenarios.append({"name": "All closed together", "streets": list(street_edges),
                          "edges": [e for es in street_edges.values() for e in es]})

    # 3. residents (same set for every scenario, so results compare) and schools
    nodes_of = {st: {n for e in es for n in e[:2]} for st, es in street_edges.items()}
    origins, near_street = [], {}
    for n in G.nodes:
        d_best, st_best = min((min(dist_m(G, m, G.nodes[n]["y"], G.nodes[n]["x"]) for m in ns), st)
                              for st, ns in nodes_of.items())
        if d_best <= args.resident_radius_m:
            origins.append(n)
            near_street[n] = st_best
    centre = (sum(G.nodes[n]["y"] for n in origins) / len(origins),
              sum(G.nodes[n]["x"] for n in origins) / len(origins))
    schools = find_schools(args, centre)
    if not schools:
        raise SystemExit('No schools located; add them with --school "Name" LAT LON.')
    for s in schools:
        s["node"] = min(G.nodes, key=lambda n: dist_m(G, n, s["lat"], s["lon"]))
        gap = dist_m(G, s["node"], s["lat"], s["lon"])
        s["snap_m"] = round(gap)
        print(f"  {s['name']}: ({s['lat']:.5f}, {s['lon']:.5f}) from {s['source']}, nearest road {gap:.0f} m"
              + ("  <- WARNING: far from any road in the map; export a larger area" if gap > 300 else ""))
    step(f"{len(origins)} resident origins within {args.resident_radius_m:.0f} m of the streets; "
         f"{len(scenarios)} scenarios x {len(schools)} schools")

    # 4. routing: normal once per school, then once per scenario
    main_before = max(nx.strongly_connected_components(G), key=len)
    rows, examples, cut_rows = [], {}, []
    for sc in scenarios:
        blocked = set(sc["edges"])
        H = G.copy()
        H.remove_edges_from(sc["edges"])
        main_after = max(nx.strongly_connected_components(H), key=len)
        for n in main_before - main_after:
            cut_rows.append({"scenario": sc["name"], "node": n, "location": node_street(G, n),
                             "lat": G.nodes[n]["y"], "lon": G.nodes[n]["x"]})
    for s in schools:
        t_n, p_n = routes_to(G, [s["node"]], set())
        for sc in scenarios:
            blocked = set(sc["edges"])
            t_c, p_c = routes_to(G, [s["node"]], blocked)
            for o in origins:
                if o not in t_n or o == s["node"]:
                    continue  # unreachable even with all roads open (edge of the map)
                e_n = path_edges(G, p_n[o], set())
                row = {"scenario": sc["name"], "school": s["name"], "origin_node": o,
                       "origin": node_street(G, o), "near_street": near_street[o],
                       "lat": G.nodes[o]["y"], "lon": G.nodes[o]["x"],
                       "uses_closed_street_normally": any(e in blocked for e in e_n),
                       "normal_distance_mi": sum(G.edges[e]["length"] for e in e_n) / M_PER_MI,
                       "normal_time_min": t_n[o] / 60, "normal_route": street_sequence(G, e_n)}
                if o in t_c:
                    e_c = path_edges(G, p_c[o], blocked)
                    row.update(closure_distance_mi=sum(G.edges[e]["length"] for e in e_c) / M_PER_MI,
                               closure_time_min=t_c[o] / 60, closure_route=street_sequence(G, e_c))
                    row["extra_distance_mi"] = row["closure_distance_mi"] - row["normal_distance_mi"]
                    row["extra_time_min"] = row["closure_time_min"] - row["normal_time_min"]
                    row["status"] = "longer" if row["extra_time_min"] > 0.05 else "unchanged"
                    key = sc["name"]
                    if row["status"] == "longer" and (key not in examples
                                                      or row["extra_time_min"] > examples[key]["extra"]):
                        examples[key] = {"extra": row["extra_time_min"], "normal_edges": e_n, "closure_edges": e_c,
                                         "label": f"{node_street(G, o)} to {s['name']}"}
                else:
                    row["status"] = "unreachable"
                rows.append(row)
    od = pd.DataFrame(rows)
    cut = pd.DataFrame(cut_rows)
    if od.empty:
        raise SystemExit("No resident origin can reach any school even with all roads open; "
                         "export a larger area that includes the schools.")

    # 5. tables
    detail = []
    for sc in scenarios:
        for s in schools:
            sub = od[(od["scenario"] == sc["name"]) & (od["school"] == s["name"])]
            both = sub[sub["status"] != "unreachable"]
            hit = both[both["status"] == "longer"]
            detail.append({
                "Scenario": sc["name"], "School": s["name"], "Residents": len(sub),
                "Normally use closed street": int(sub["uses_closed_street_normally"].sum()),
                "Trips longer": len(hit), "Cannot reach": int((sub["status"] == "unreachable").sum()),
                "Avg normal time (min)": round(both["normal_time_min"].mean(), 2),
                "Avg closure time (min)": round(both["closure_time_min"].mean(), 2),
                "Avg extra time, affected trips (min)": round(hit["extra_time_min"].mean(), 2) + 0.0 if len(hit) else 0.0,
                "Max extra time (min)": round(hit["extra_time_min"].max(), 2) + 0.0 if len(hit) else 0.0,
                "Avg extra distance, affected trips (mi)": round(hit["extra_distance_mi"].mean(), 2) + 0.0 if len(hit) else 0.0,
            })
    detail = pd.DataFrame(detail)
    pivots = {col: detail.pivot(index="Scenario", columns="School", values=col)
              .reindex([sc["name"] for sc in scenarios])
              for col in ("Trips longer", "Cannot reach", "Max extra time (min)")}

    # 6. report
    print("\n=== Street closures: residents' trips to schools ===")
    print(f"Residents: {len(origins)} intersections within {args.resident_radius_m:.0f} m of "
          f"{', '.join(street_edges)} (same set in every scenario). Averages use residents who can "
          "reach the school in both cases.\n")
    print(detail.to_string(index=False))
    for col, table in pivots.items():
        print(f"\n--- {col} (scenario x school) ---")
        print(table.to_string())
    for sc in scenarios:
        ex = examples.get(sc["name"])
        sub = od[(od["scenario"] == sc["name"]) & (od["status"] == "longer")]
        if ex is not None and len(sub):
            top = sub.sort_values("extra_time_min", ascending=False).iloc[0]
            print(f"\n{sc['name']}: largest delay +{top['extra_time_min']:.2f} min, {top['origin']} to "
                  f"{top['school']}\n    normal : {top['normal_route']}\n    closed : {top['closure_route']}")
        n_cut = int((cut["scenario"] == sc["name"]).sum()) if len(cut) else 0
        if n_cut:
            print(f"  {n_cut} location(s) cut off from the road network: "
                  + "; ".join(cut[cut["scenario"] == sc["name"]]["location"].head(8)))

    print("\n=== Does the impact depend on the street, and on the school? ===")
    by_street = detail.groupby("Scenario", sort=False)[["Trips longer", "Cannot reach", "Max extra time (min)"]] \
        .agg({"Trips longer": "sum", "Cannot reach": "sum", "Max extra time (min)": "max"})
    by_school = detail[detail["Scenario"] != "All closed together"].groupby("School", sort=False)[
        ["Trips longer", "Cannot reach", "Max extra time (min)"]] \
        .agg({"Trips longer": "sum", "Cannot reach": "sum", "Max extra time (min)": "max"})
    print("By scenario (summed over schools):")
    print(by_street.to_string())
    print("\nBy school (summed over the single-street scenarios):")
    print(by_school.to_string())
    singles = by_street.drop(index="All closed together", errors="ignore")
    if len(singles) > 1:
        for col, label in (("Trips longer", "Most trips made longer"), ("Cannot reach", "Most trips cut off"),
                           ("Max extra time (min)", "Largest single delay")):
            same = (singles[col] == singles[col].max()).all()
            print(f"  {label}: " + ("same for every street" if same
                                     else f"{singles[col].idxmax()} ({singles[col].max()})"))
    if "All closed together" in by_street.index and len(singles) > 1:
        combo, total = by_street.loc["All closed together", "Trips longer"], singles["Trips longer"].sum()
        print(f"  Together vs separately: {combo} trips longer with all closed, vs {total} summed over "
              f"single closures ({'the closures compound' if combo > total else 'little interaction' if combo == total else 'overlapping effects'}).")

    os.makedirs(args.out, exist_ok=True)
    detail.to_csv(os.path.join(args.out, "scenario_school_results.csv"), index=False)
    for col, table in pivots.items():
        fname = "pivot_" + "".join(ch if ch.isalnum() else "_" for ch in col.lower()).strip("_") + ".csv"
        table.to_csv(os.path.join(args.out, fname))
    od.to_csv(os.path.join(args.out, "resident_school_routes.csv"), index=False)
    cut.to_csv(os.path.join(args.out, "cut_off_locations.csv"), index=False)
    with open(os.path.join(args.out, "run_record.json"), "w") as f:
        json.dump({
            "scenarios": [{"name": sc["name"], "streets": sc["streets"],
                           "closed_segments": len({frozenset(e[:2]) for e in sc["edges"]})} for sc in scenarios],
            "streets_not_found": [st for st in streets if st not in street_edges],
            "residents": f"{len(origins)} intersections within {args.resident_radius_m} m of the streets",
            "schools": [{k: s[k] for k in ("name", "lat", "lon", "source", "snap_m")} for s in schools],
            "data_source": (f"OpenStreetMap file {args.osm_file}" if args.osm_file else
                            "OpenStreetMap via Overpass API") + f", OSMnx {ox.__version__}",
            "speeds": "OSM maxspeed where tagged, else road-type mean / DEFAULT_SPEEDS_MPH; free flow",
            "run_date": run_date.isoformat(timespec="seconds"),
        }, f, indent=2)
    if not args.no_plot:
        for i, sc in enumerate(scenarios, 1):
            slug = "".join(ch if ch.isalnum() else "_" for ch in sc["name"].lower()).strip("_")[:40]
            plot_map(G, od, sc, sc["edges"], schools, examples.get(sc["name"]),
                     os.path.join(args.out, f"scenario_{i}_{slug}.png"))
    print(f"\nOutputs written to {os.path.abspath(args.out)}/")
    return detail, pivots, od, cut


if __name__ == "__main__":
    main()
