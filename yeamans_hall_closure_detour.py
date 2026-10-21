"""
Yeamans Hall Road corridor: baseline route and simulated segment closure.

Study corridor
    Start : Yeamans Hall Road x Brookside Drive   (Hanahan / North Charleston, SC)
    End   : Yeamans Hall Road x Dickson Avenue
    Network: drivable OpenStreetMap roads within a buffer around the corridor,
             so nearby connecting streets are available for detours.

Steps
    1. Download the drivable road network around the corridor (OSMnx / Overpass).
    2. Locate the two intersection nodes from the street names on the graph.
    3. Baseline: fastest route (by travel time) with all roads open.
    4. Close one Yeamans Hall Road segment inside the corridor (both directions)
       and recompute the fastest alternative route.
    5. Write the results table, the measurement record (route, source, date),
       and a map of both routes.

Requirements
    pip install osmnx networkx pandas matplotlib

Usage (Jupyter: install with  %pip install osmnx networkx pandas matplotlib
       then  %run yeamans_hall_closure_detour.py --close-segment 2
       or paste this file into a cell; it calls main() itself)
    python yeamans_hall_closure_detour.py
    python yeamans_hall_closure_detour.py --close-segment 2      # pick a segment by index
    python yeamans_hall_closure_detour.py --list-segments        # show segments and exit
    python yeamans_hall_closure_detour.py --buffer-m 2500 --out results/

Notes on travel time
    OSMnx assigns each edge its OSM `maxspeed` where tagged; otherwise the mean
    speed for that highway class in the graph, falling back to DEFAULT_SPEEDS_MPH.
    Times are free-flow estimates (no signals, turns, or congestion). If you also
    measure the baseline with Google Maps / a field drive, pass it with
    --observed-distance-mi / --observed-time-min so it is recorded alongside.
"""

import argparse
import datetime as dt
import json
import os
import sys
import time

import networkx as nx
import osmnx as ox
import pandas as pd

CORRIDOR = "Yeamans Hall Road"
START_CROSS = "Brookside Drive"
END_CROSS = "Dickson Avenue"
PLACES = ["Hanahan, South Carolina, USA", "North Charleston, South Carolina, USA"]

# Geocoder queries, used only to center the download and as a fallback when the
# street-name lookup on the graph fails.
START_QUERY = f"{CORRIDOR} & {START_CROSS}, Hanahan, SC, USA"
END_QUERY = f"{CORRIDOR} & {END_CROSS}, North Charleston, SC, USA"
STREET_QUERIES = [f"{CORRIDOR}, North Charleston, SC, USA",
                  f"{START_CROSS}, Hanahan, SC, USA",
                  f"{END_CROSS}, North Charleston, SC, USA"]

# Fallback free-flow speeds (mph) for highway classes without a maxspeed tag.
DEFAULT_SPEEDS_MPH = {
    "motorway": 60, "trunk": 50, "primary": 45, "secondary": 40,
    "tertiary": 35, "unclassified": 30, "residential": 25,
    "living_street": 15, "service": 15,
}

M_PER_MI = 1609.344

# Public Overpass servers, tried in order; the main one is often "at capacity".
OVERPASS_URLS = [
    "https://overpass-api.de/api",
    "https://overpass.kumi.systems/api",
    "https://overpass.private.coffee/api",
    "https://maps.mail.ru/osm/tools/overpass/api",
]


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


def geocode(query):
    try:
        lat, lon = ox.geocode(query)
        return lat, lon
    except Exception:
        return None


def overpass_retry(fn, *args, rounds=3, wait_s=10, **kwargs):
    """Call an OSMnx download, rotating through Overpass servers with backoff."""
    last = None
    for r in range(rounds):
        for url in OVERPASS_URLS:
            ox.settings.overpass_url = url
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


def download_network(buffer_m, start_pt=None, end_pt=None):
    """Drivable network around the corridor, buffered to include detour streets.

    start_pt / end_pt are (lat, lon) overrides; otherwise the intersections are
    geocoded. Each street is also geocoded on its own so the download covers all
    three streets even when an intersection lookup fails.
    """
    start_pt = start_pt or geocode(START_QUERY)
    end_pt = end_pt or geocode(END_QUERY)
    pts = [p for p in (start_pt, end_pt) if p]
    if len(pts) < 2:
        for q in STREET_QUERIES:
            p = geocode(q)
            print(f"  geocode '{q}': {p if p else 'not found'}")
            if p:
                pts.append(p)
    if pts:
        lat = sum(p[0] for p in pts) / len(pts)
        lon = sum(p[1] for p in pts) / len(pts)
        reach = max(ox.distance.great_circle(lat, lon, *p) for p in pts)
        dist = reach + buffer_m
        print(f"Downloading drive network: center=({lat:.5f}, {lon:.5f}), radius={dist:.0f} m")
        G = overpass_retry(ox.graph_from_point, (lat, lon), dist=dist, network_type="drive", simplify=True)
    else:
        print(f"Geocoding failed; downloading drive network for {PLACES}")
        G = overpass_retry(ox.graph_from_place, PLACES, network_type="drive")
    return G, (start_pt, end_pt)


def add_travel_times(G):
    G = ox.add_edge_speeds(G, hwy_speeds={k: v * 1.609344 for k, v in DEFAULT_SPEEDS_MPH.items()},
                           fallback=25 * 1.609344)
    return ox.add_edge_travel_times(G)


def street_key(street):
    """Distinctive part of a street name: 'Brookside Drive' -> 'brookside'."""
    words = norm(street).split()
    return " ".join(words[:-1]) if len(words) > 1 else words[0]


def street_nodes(G, street):
    """Nodes on edges whose name contains the street's distinctive part."""
    key = street_key(street)
    nodes = set()
    for u, v, d in G.edges(data=True):
        if any(key in n for n in edge_names(d)):
            nodes.update((u, v))
    return nodes


def similar_names(G, street):
    first = street_key(street).split()[0]
    found = set()
    for _, _, d in G.edges(data=True):
        found.update(n for n in edge_names(d) if first in n)
    return sorted(found)


def intersection_node(G, street_a, street_b, fallback_point=None, max_gap_m=150):
    """Node on street_a where it meets street_b.

    1. a node shared by both streets (true junction);
    2. otherwise the street_a node closest to any street_b node, if within
       max_gap_m (covers divided roads / slightly offset junctions in OSM);
    3. otherwise the node nearest the geocoded/override point.
    """
    na, nb = street_nodes(G, street_a), street_nodes(G, street_b)
    shared = na & nb
    if shared:
        cands = sorted(shared)
        if len(cands) > 1 and fallback_point:
            lat, lon = fallback_point
            cands.sort(key=lambda n: ox.distance.great_circle(
                lat, lon, G.nodes[n]["y"], G.nodes[n]["x"]))
        return cands[0], "street-name match on OSM graph"
    if na and nb:
        gap, best = min(
            (ox.distance.great_circle(G.nodes[a]["y"], G.nodes[a]["x"],
                                      G.nodes[b]["y"], G.nodes[b]["x"]), a)
            for a in na for b in nb)
        if gap <= max_gap_m:
            return best, f"closest {street_a} node to {street_b} ({gap:.0f} m apart)"
    if fallback_point:
        lat, lon = fallback_point
        # prefer a corridor node; plain search avoids nearest_nodes' scikit-learn dependency
        pool = na or G.nodes
        node = min(pool, key=lambda n: ox.distance.great_circle(lat, lon, G.nodes[n]["y"], G.nodes[n]["x"]))
        return node, "nearest corridor node to geocoded/given point"
    raise RuntimeError(
        f"Could not find the intersection of {street_a} and {street_b}.\n"
        f"  Streets in the downloaded network like '{street_a}': {similar_names(G, street_a) or 'none'}\n"
        f"  Streets in the downloaded network like '{street_b}': {similar_names(G, street_b) or 'none'}\n"
        "  Fix: right-click the intersection in Google Maps to copy its coordinates and pass them, e.g.\n"
        "  main(['--start-latlon', LAT, LON, '--end-latlon', LAT, LON])  (or %run ... --start-latlon LAT LON)")


def route_stats(G, route):
    """Total length (m), travel time (s) and per-edge rows for a node route."""
    rows, total_len, total_tt = [], 0.0, 0.0
    for u, v in zip(route[:-1], route[1:]):
        # pick the parallel edge that the router would have used
        k, d = min(G[u][v].items(), key=lambda kv: kv[1]["travel_time"])
        total_len += d["length"]
        total_tt += d["travel_time"]
        name = d.get("name", "(unnamed)")
        rows.append({"u": u, "v": v, "key": k,
                     "name": ", ".join(name) if isinstance(name, list) else name,
                     "length_m": d["length"], "speed_kph": d["speed_kph"],
                     "travel_time_s": d["travel_time"]})
    return total_len, total_tt, rows


def street_sequence(rows):
    """Collapse consecutive edges with the same street name into a readable route."""
    seq = []
    for r in rows:
        if not seq or seq[-1][0] != r["name"]:
            seq.append([r["name"], 0.0])
        seq[-1][1] += r["length_m"]
    return " -> ".join(f"{n} ({m / M_PER_MI:.2f} mi)" for n, m in seq)


def corridor_segments(rows):
    """Baseline edges that lie on Yeamans Hall Road (candidates for closure)."""
    key = street_key(CORRIDOR)
    return [r for r in rows if any(key in norm(x) for x in str(r["name"]).split(", "))]


def close_segment(G, u, v):
    """Copy of G with the segment u-v removed in both directions."""
    H = G.copy()
    for a, b in ((u, v), (v, u)):
        if H.has_edge(a, b):
            H.remove_edges_from([(a, b, k) for k in list(H[a][b])])
    return H


def fmt_mi(m):
    return f"{m / M_PER_MI:.2f} mi"


def fmt_min(s):
    return f"{s / 60:.1f} min"


def main(argv=None):
    """argv: list of options, e.g. main(["--close-segment", "2"]) from a notebook."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--buffer-m", type=float, default=2000,
                    help="network buffer around the corridor in metres (default 2000)")
    ap.add_argument("--close-segment", type=int, default=None,
                    help="index of the Yeamans Hall Road segment to close (default: middle one)")
    ap.add_argument("--list-segments", action="store_true", help="list closable segments and exit")
    ap.add_argument("--start-latlon", type=float, nargs=2, metavar=("LAT", "LON"), default=None,
                    help="override coordinates of Yeamans Hall Rd x Brookside Dr")
    ap.add_argument("--end-latlon", type=float, nargs=2, metavar=("LAT", "LON"), default=None,
                    help="override coordinates of Yeamans Hall Rd x Dickson Ave")
    ap.add_argument("--observed-distance-mi", type=float, default=None,
                    help="baseline distance measured externally (e.g. Google Maps), for the record")
    ap.add_argument("--observed-time-min", type=float, default=None,
                    help="baseline time measured externally, for the record")
    ap.add_argument("--observed-source", default=None, help="name of the external source")
    ap.add_argument("--out", default="yeamans_hall_results", help="output folder")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--overpass-url", default=None, help="Overpass server to try first")
    # parse_known_args: ignore the extra "-f kernel.json" Jupyter passes when pasted into a cell
    if argv is not None:
        argv = [str(a) for a in argv]  # allow numbers, e.g. ["--start-latlon", 32.9, -79.99]
    args, _ = ap.parse_known_args(argv)

    run_date = dt.datetime.now().astimezone()
    ox.settings.use_cache = True
    ox.settings.log_console = False
    ox.settings.requests_timeout = 300
    if args.overpass_url:
        OVERPASS_URLS.insert(0, args.overpass_url.rstrip("/"))

    # 1. network ------------------------------------------------------------
    G, (start_pt, end_pt) = download_network(args.buffer_m, args.start_latlon, args.end_latlon)
    G = add_travel_times(G)
    print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # 2. intersections ------------------------------------------------------
    orig, orig_how = intersection_node(G, CORRIDOR, START_CROSS, start_pt)
    dest, dest_how = intersection_node(G, CORRIDOR, END_CROSS, end_pt)
    print(f"Start node {orig} ({G.nodes[orig]['y']:.6f}, {G.nodes[orig]['x']:.6f}) via {orig_how}")
    print(f"End node   {dest} ({G.nodes[dest]['y']:.6f}, {G.nodes[dest]['x']:.6f}) via {dest_how}")

    # 3. baseline -----------------------------------------------------------
    base_route = nx.shortest_path(G, orig, dest, weight="travel_time")
    base_len, base_tt, base_rows = route_stats(G, base_route)
    print(f"\nBaseline route: {street_sequence(base_rows)}")

    segs = corridor_segments(base_rows)
    if not segs:
        sys.exit("Baseline route does not use Yeamans Hall Road; check the intersection nodes.")
    if args.list_segments:
        print("\nClosable Yeamans Hall Road segments on the baseline route:")
        for i, s in enumerate(segs):
            print(f"  [{i}] {s['u']} -> {s['v']}  {s['length_m']:.0f} m  {s['speed_kph'] / 1.609344:.0f} mph")
        return

    # 4. closure ------------------------------------------------------------
    idx = len(segs) // 2 if args.close_segment is None else args.close_segment
    if not 0 <= idx < len(segs):
        sys.exit(f"--close-segment must be between 0 and {len(segs) - 1}")
    seg = segs[idx]
    G_closed = close_segment(G, seg["u"], seg["v"])
    try:
        alt_route = nx.shortest_path(G_closed, orig, dest, weight="travel_time")
    except nx.NetworkXNoPath:
        sys.exit("No detour exists inside the downloaded network; increase --buffer-m.")
    alt_len, alt_tt, alt_rows = route_stats(G_closed, alt_route)
    u_xy = (G.nodes[seg["u"]]["y"], G.nodes[seg["u"]]["x"])
    v_xy = (G.nodes[seg["v"]]["y"], G.nodes[seg["v"]]["x"])
    print(f"\nClosed segment [{idx}]: {seg['u']} {u_xy} -> {seg['v']} {v_xy}, {seg['length_m']:.0f} m")
    print(f"Detour route: {street_sequence(alt_rows)}")

    # 5. results ------------------------------------------------------------
    table = pd.DataFrame([
        {"Condition": "All roads open", "Distance": fmt_mi(base_len), "Estimated travel time": fmt_min(base_tt)},
        {"Condition": "Selected segment closed", "Distance": fmt_mi(alt_len), "Estimated travel time": fmt_min(alt_tt)},
        {"Condition": "Additional travel", "Distance": "+" + fmt_mi(alt_len - base_len),
         "Estimated travel time": "+" + fmt_min(alt_tt - base_tt)},
    ])
    pct_d = 100 * (alt_len - base_len) / base_len
    pct_t = 100 * (alt_tt - base_tt) / base_tt

    print("\n" + table.to_string(index=False))
    print(f"(Detour adds {pct_d:.0f}% distance and {pct_t:.0f}% time.)")

    os.makedirs(args.out, exist_ok=True)
    table.to_csv(os.path.join(args.out, "results_table.csv"), index=False)
    pd.DataFrame(base_rows).to_csv(os.path.join(args.out, "baseline_route_edges.csv"), index=False)
    pd.DataFrame(alt_rows).to_csv(os.path.join(args.out, "detour_route_edges.csv"), index=False)

    record = {
        "study_corridor": f"{CORRIDOR} between {START_CROSS} and {END_CROSS}",
        "start": {"intersection": f"{CORRIDOR} x {START_CROSS}", "osm_node": int(orig),
                  "lat": G.nodes[orig]["y"], "lon": G.nodes[orig]["x"], "located_by": orig_how},
        "end": {"intersection": f"{CORRIDOR} x {END_CROSS}", "osm_node": int(dest),
                "lat": G.nodes[dest]["y"], "lon": G.nodes[dest]["x"], "located_by": dest_how},
        "measurement_source": (f"OpenStreetMap drive network via OSMnx {ox.__version__} "
                               "(Overpass API); fastest path by free-flow travel time"),
        "measurement_date": run_date.isoformat(timespec="seconds"),
        "network_buffer_m": args.buffer_m,
        "speed_assumptions": "OSM maxspeed where tagged, else highway-class mean / DEFAULT_SPEEDS_MPH",
        "baseline": {"route": street_sequence(base_rows), "distance_mi": round(base_len / M_PER_MI, 3),
                     "travel_time_min": round(base_tt / 60, 2), "nodes": [int(n) for n in base_route]},
        "closed_segment": {"index": idx, "u": int(seg["u"]), "v": int(seg["v"]),
                           "u_latlon": u_xy, "v_latlon": v_xy, "length_m": round(seg["length_m"], 1),
                           "directions_closed": "both"},
        "detour": {"route": street_sequence(alt_rows), "distance_mi": round(alt_len / M_PER_MI, 3),
                   "travel_time_min": round(alt_tt / 60, 2), "nodes": [int(n) for n in alt_route]},
        "additional_travel": {"distance_mi": round((alt_len - base_len) / M_PER_MI, 3),
                              "travel_time_min": round((alt_tt - base_tt) / 60, 2),
                              "distance_pct": round(pct_d, 1), "time_pct": round(pct_t, 1)},
    }
    if args.observed_distance_mi is not None or args.observed_time_min is not None:
        record["baseline_observed"] = {"distance_mi": args.observed_distance_mi,
                                       "travel_time_min": args.observed_time_min,
                                       "source": args.observed_source,
                                       "date": run_date.date().isoformat()}
    with open(os.path.join(args.out, "measurement_record.json"), "w") as f:
        json.dump(record, f, indent=2)

    if not args.no_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = ox.plot_graph_routes(G, [base_route, alt_route], route_colors=["tab:blue", "tab:red"],
                                       route_linewidths=[5, 3], node_size=0, bgcolor="white",
                                       edge_color="#bbbbbb", show=False, close=False)
        ax.plot([u_xy[1], v_xy[1]], [u_xy[0], v_xy[0]], color="black", lw=6, ls=":", zorder=5)
        ax.set_title("Yeamans Hall Rd: baseline (blue) vs detour (red); closed segment dotted")
        fig.savefig(os.path.join(args.out, "routes_map.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)

    print(f"\nOutputs written to {os.path.abspath(args.out)}/")


if __name__ == "__main__":
    main()
