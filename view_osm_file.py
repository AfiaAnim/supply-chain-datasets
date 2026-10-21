"""
View an OpenStreetMap XML (.osm) file in Python.

Loads the file with OSMnx, keeps only drivable streets (an .osm export from
openstreetmap.org also contains buildings, footpaths, rivers, ...), then:
  - prints a summary: size, road types, longest streets,
  - draws a static map with street names (optionally highlighting streets),
  - writes an interactive map (zoom, hover a road to see its name) if folium
    is installed.

Requirements
    pip install osmnx matplotlib            (interactive map: pip install folium mapclassify)

Usage
    python view_osm_file.py map.osm
    python view_osm_file.py map.osm --highlight "Sellers Road" "Dickson Avenue"
    Jupyter:  %run view_osm_file.py map.osm --highlight "Sellers Road"
              or paste into a cell, then  G = main(["map.osm"])
"""

import argparse
import importlib.util
import math
import os
import sys

import networkx as nx
import osmnx as ox

DRIVABLE = {
    "motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link",
    "secondary", "secondary_link", "tertiary", "tertiary_link", "unclassified",
    "residential", "living_street", "service", "road",
}
MAJOR = {"motorway", "trunk", "primary", "secondary", "tertiary"}


def first(value):
    return value[0] if isinstance(value, list) else value


def load_drive_graph(path):
    """Drivable street network from an .osm file (buildings, footpaths etc. removed)."""
    G = ox.graph_from_xml(path, simplify=False, retain_all=True)
    drop = [(u, v, k) for u, v, k, d in G.edges(keys=True, data=True)
            if first(d.get("highway")) not in DRIVABLE
            or first(d.get("access")) in {"no", "private"}
            or first(d.get("service")) in {"parking_aisle", "driveway"}]
    G.remove_edges_from(drop)
    G.remove_nodes_from([n for n in list(G.nodes) if G.degree(n) == 0])
    if G.number_of_edges() == 0:
        raise SystemExit(f"No drivable roads found in {path}.")
    # street lengths before simplification, while every segment has a single name
    lengths = {}
    for _, _, d in G.edges(data=True):
        name = first(d.get("name")) or "(unnamed)"
        lengths[name] = lengths.get(name, 0) + d["length"] / (1 if d.get("oneway") else 2)
    ys = [d["y"] for _, d in G.nodes(data=True)]
    xs = [d["x"] for _, d in G.nodes(data=True)]
    G = ox.simplify_graph(G)
    biggest = max(nx.weakly_connected_components(G), key=len)  # drop disconnected fragments
    G = G.subgraph(biggest).copy()
    G.graph.update(street_lengths=lengths, bounds=(min(ys), max(ys), min(xs), max(xs)))
    return G


def summary(G):
    lengths, kinds = G.graph["street_lengths"], {}
    for _, _, d in G.edges(data=True):
        for kind in (d["highway"] if isinstance(d.get("highway"), list) else [d.get("highway")]):
            kinds[kind] = kinds.get(kind, 0) + 1
    lat0, lat1, lon0, lon1 = G.graph["bounds"]
    print(f"Intersections/ends: {G.number_of_nodes()}, road segments: {G.number_of_edges()}")
    print(f"Area: lat {lat0:.5f}..{lat1:.5f}, lon {lon0:.5f}..{lon1:.5f}")
    print("Road types:", ", ".join(f"{k} {v}" for k, v in sorted(kinds.items(), key=lambda kv: -kv[1])))
    print("Longest streets:")
    for name, m in sorted(lengths.items(), key=lambda kv: -kv[1])[:15]:
        print(f"  {name:35s} {m / 1609.344:5.2f} mi")
    return lengths


def street_edges(G, street):
    s = street.lower()
    return [(u, v, k) for u, v, k, d in G.edges(keys=True, data=True)
            if any(s in str(n).lower() for n in (d.get("name") if isinstance(d.get("name"), list)
                                                  else [d.get("name")]))]


def edge_xy(G, u, v, k):
    geom = G.edges[u, v, k].get("geometry")
    if geom is not None:
        xs, ys = geom.xy
        return list(xs), list(ys)
    return [G.nodes[u]["x"], G.nodes[v]["x"]], [G.nodes[u]["y"], G.nodes[v]["y"]]


def plot_static(G, highlight, path):
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe

    fig, ax = plt.subplots(figsize=(11, 10))
    for u, v, k, d in G.edges(keys=True, data=True):
        major = first(d.get("highway")) in MAJOR
        xs, ys = edge_xy(G, u, v, k)
        ax.plot(xs, ys, color="#666666" if major else "#b5b5b5", lw=1.6 if major else 0.7, zorder=1)
    colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]
    handles = []
    for i, street in enumerate(highlight):
        edges = street_edges(G, street)
        if not edges:
            print(f"  '{street}' not found in this file")
            continue
        for e in edges:
            xs, ys = edge_xy(G, *e)
            ax.plot(xs, ys, color=colors[i % len(colors)], lw=4, zorder=3)
        handles.append(plt.Line2D([], [], color=colors[i % len(colors)], lw=4, label=street))

    lat0 = sum(d["y"] for _, d in G.nodes(data=True)) / G.number_of_nodes()
    ax.set_aspect(1 / math.cos(math.radians(lat0)))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.apply_aspect()

    # one label per street, on its longest segment, rotated along the road
    best = {}
    for u, v, k, d in G.edges(keys=True, data=True):
        name = first(d.get("name"))
        if name and (name not in best or d["length"] > best[name][0]):
            best[name] = (d["length"], (u, v, k), first(d.get("highway")) in MAJOR)
    placed = []
    for name, (length, e, major) in sorted(best.items(), key=lambda kv: (not kv[1][2], -kv[1][0])):
        xs, ys = edge_xy(G, *e)
        mx, my = (xs[0] + xs[-1]) / 2, (ys[0] + ys[-1]) / 2
        (px0, py0), (px1, py1), (pmx, pmy) = ax.transData.transform([(xs[0], ys[0]), (xs[-1], ys[-1]), (mx, my)])
        if any((pmx - a) ** 2 + (pmy - b) ** 2 < 40 ** 2 for a, b in placed):
            continue
        placed.append((pmx, pmy))
        angle = math.degrees(math.atan2(py1 - py0, px1 - px0))
        angle = angle - 180 if angle > 90 else angle + 180 if angle < -90 else angle
        ax.text(mx, my, name, fontsize=7 if major else 5.5, color="#222222" if major else "#555555",
                rotation=angle, rotation_mode="anchor", ha="center", va="center", zorder=4,
                path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])
    if handles:
        ax.legend(handles=handles, loc="lower right", fontsize=9)
    ax.set_title(f"Drivable streets in {os.path.basename(path).replace('_map.png', '')}")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    if "ipykernel" in sys.modules:
        from IPython.display import Image, display
        display(Image(filename=path, width=900))


def plot_interactive(G, highlight, path):
    if importlib.util.find_spec("folium") is None:  # GeoDataFrame.explore needs folium
        print("Interactive map skipped (pip install folium mapclassify to enable).")
        return None
    edges = ox.graph_to_gdfs(G, nodes=False).reset_index()
    for col in ("name", "highway"):
        edges[col] = edges[col].apply(lambda x: ", ".join(map(str, x)) if isinstance(x, list) else x)
    m = edges.explore(color="#555555", tooltip=["name", "highway"], style_kwds={"weight": 2},
                      tiles="CartoDB positron")
    for street, color in zip(highlight, ["red", "blue", "green", "purple", "orange"]):
        sub = edges[edges["name"].fillna("").str.lower().str.contains(street.lower(), regex=False)]
        if len(sub):
            sub.explore(m=m, color=color, tooltip=["name", "highway"], style_kwds={"weight": 6})
    m.save(path)
    print(f"Interactive map: {path} (open in a browser; hover a road to see its name)")
    return m


def main(argv=None):
    """argv: e.g. main(["map.osm", "--highlight", "Sellers Road"]) from a notebook."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("osm_file", nargs="?", help="path to the downloaded .osm file")
    ap.add_argument("--highlight", nargs="*", default=[], help="street names to colour")
    argv = sys.argv[1:] if argv is None else [str(a) for a in argv]
    if "-f" in argv:  # Jupyter passes "-f kernel.json"; drop it so it is not read as the .osm file
        i = argv.index("-f")
        argv = argv[:i] + argv[i + 2:]
    args, _ = ap.parse_known_args(argv)

    if not args.osm_file:
        print('Give the .osm file, e.g.  main(["map.osm"])  or  %run view_osm_file.py map.osm')
        return None
    G = load_drive_graph(args.osm_file)
    summary(G)
    stem = os.path.splitext(args.osm_file)[0]
    plot_static(G, args.highlight, stem + "_map.png")
    print(f"Static map: {stem}_map.png")
    m = plot_interactive(G, args.highlight, stem + "_map.html")
    if m is not None and "ipykernel" in sys.modules:
        from IPython.display import display
        display(m)
    return G


if __name__ == "__main__":
    main()
