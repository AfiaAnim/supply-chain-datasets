"""
Street closures with congestion: InfraRisk's traffic assignment on the Hanahan road network.

What it adds to the closure simulations
    The other closure scripts route every trip on empty roads. This one loads
    traffic onto the network, so detour streets slow down as diverted cars fill
    them. It uses the road-transport model of InfraRisk (Balakrishnan &
    Cassottana, 2022): a static traffic assignment in which travel time on each
    road follows the BPR function t = t0 * (1 + 0.15 (flow / capacity)^4) and
    drivers settle on routes no one can improve (user equilibrium, Frank-Wolfe).

Method
    1. Road network from a downloaded OpenStreetMap .osm file, read with the
       helpers of brookside_wedgewood_rembert_closure.py (keep both files in the
       same folder). Schools are found by name and reached at their entrances.
    2. Export to InfraRisk's transport format (TNTP): transpo_net.tntp,
       transpo_trips.tntp, transpo_node.tntp in <out>/infrarisk_transport/.
       Each school is a zone joined to its entrances. This folder can also be
       used as the transport network of a full InfraRisk IntegratedNetwork.
    3. Trips in the morning peak hour (vehicles/hour):
         - school trips: every resident intersection near the closed streets
           sends --trips-per-school vehicles to each school,
         - through traffic: for each --aadt STREET VALUE, VALUE x peak-hour
           factor, split evenly between the two directions along that street.
    4. Scenarios: all roads open, each street closed alone, all closed together.
       A closed road is failed the way InfraRisk does it (free-flow time set to
       9999 min). Trips with no open route at all are removed and reported as
       unreachable instead of being forced through the closure.
    5. For every scenario: user-equilibrium assignment, then each resident's
       congested travel time to each school, through-traffic travel times,
       total vehicle-hours, and the streets that absorb the diverted traffic.

Engines
    --engine infrarisk (default): InfraRisk's own Network class and
        userEquilibrium("FW", ...) - the same call InfraRisk's recovery module
        makes. Two adjustments for street-level networks, both result-neutral:
        the all-or-nothing step only searches from origins that have trips
        (stock InfraRisk searches from every node and prints a line per trip),
        and the throwaway per-link sort key is kept short while the file loads.
    --engine builtin: a compact implementation of the same algorithm (BPR
        costs, Frank-Wolfe with line search) for cross-checking, or when
        InfraRisk is not installed.

Installing InfraRisk (only the transport model is used; it needs just pandas)
    git clone https://github.com/srijithbalakrishnan/dreaminsg-integrated-model.git
    then pass  --infrarisk-path dreaminsg-integrated-model
    (or install the whole package: pip install -e dreaminsg-integrated-model,
    which also pulls in pandapower, wntr, pyomo, rasterio, ...)

Usage
    python infrarisk_closure_assignment.py --osm-file map.osm --infrarisk-path dreaminsg-integrated-model
    Jupyter:  %run infrarisk_closure_assignment.py --osm-file map.osm --infrarisk-path dreaminsg-integrated-model
    Options:
      --street "Brookside Drive"     streets to close (repeatable; default: Brookside, Wedgewood, Rembert)
      --aadt "Dickson Avenue" 1900   daily traffic on a street, added as through traffic (repeatable)
      --trips-per-school 1.5         peak-hour vehicles from each resident intersection to each school
      --engine builtin               run without InfraRisk
"""

import argparse
import contextlib
import datetime as dt
import heapq
import importlib
import io
import json
import math
import os
import sys
import time

import networkx as nx
import pandas as pd

import brookside_wedgewood_rembert_closure as base  # map loading, schools, entrances
from brookside_wedgewood_rembert_closure import first, node_street, on_street, step

CAPACITY_VPH_PER_LANE = {
    "motorway": 1900, "trunk": 1600, "primary": 1200, "secondary": 900,
    "tertiary": 700, "unclassified": 500, "residential": 500,
    "living_street": 300, "service": 300,
}
DEFAULT_AADT = {"Dickson Avenue": 1900}  # vehicles/day (user-supplied count)
PEAK_HOUR_FACTOR = 0.10                 # share of daily traffic in the peak hour
FAILED_FFT = 9999                       # InfraRisk's free-flow time for a failed link (min)
BPR_ALPHA, BPR_BETA = 0.15, 4
CONNECTOR = {"capacity": 99999, "length_mi": 0.0001, "fft_min": 0.01}


# --- export to InfraRisk (TNTP) ------------------------------------------------------------

def link_capacity(d):
    try:
        lanes = max(1, int(float(first(d.get("lanes", 1)))))
    except (TypeError, ValueError):
        lanes = 1
    if not d.get("oneway", False) and lanes > 1:
        lanes = max(1, lanes // 2)  # OSM `lanes` counts both directions on two-way roads
    return CAPACITY_VPH_PER_LANE.get(first(d.get("highway")), 500) * lanes


def export_tntp(G, schools, folder):
    """Write transpo_net/_node.tntp; return node and link name maps."""
    os.makedirs(folder, exist_ok=True)
    node_id = {n: f"T_J{i}" for i, n in enumerate(sorted(G.nodes), 1)}
    zone_id = {s["name"]: f"T_J{len(node_id) + i}" for i, s in enumerate(schools, 1)}
    links = []  # (link id, tail, head, capacity, length mi, fft min, speed mph, type, osm edge or None)
    for j, (u, v, k, d) in enumerate(G.edges(keys=True, data=True), 1):
        links.append((f"T_L{j}", node_id[u], node_id[v], link_capacity(d), d["length"] / base.M_PER_MI,
                      d["travel_time"] / 60, d["speed_kph"] / 1.609344, 1, (u, v, k)))
    for s in schools:  # centroid connectors: entrance -> school zone (inbound only, so no shortcut)
        for n in s["targets"]:
            links.append((f"T_L{len(links) + 1}", node_id[n], zone_id[s["name"]], CONNECTOR["capacity"],
                          CONNECTOR["length_mi"], CONNECTOR["fft_min"], 0, 2, None))
    n_nodes = len(node_id) + len(zone_id)
    with open(os.path.join(folder, "transpo_net.tntp"), "w") as f:
        f.write(f"<NUMBER OF ZONES> {n_nodes}\n<NUMBER OF NODES> {n_nodes}\n<FIRST THRU NODE> T_J1\n"
                f"<NUMBER OF LINKS> {len(links)}\n<END OF METADATA>\n\n\n"
                "~\tInit node\tTerm node\tCapacity\tLength\tFree Flow Time\tB\tPower\tSpeed limit\tToll\tType\tid\t;\n")
        for lid, a, b, cap, ln, fft, spd, typ, _ in links:
            f.write(f"{a}\t{b}\t{cap:.0f}\t{ln:.6f}\t{fft:.6f}\t{BPR_ALPHA}\t{BPR_BETA}\t{spd:.1f}\t0\t{typ}\t{lid}\t;\n")
    with open(os.path.join(folder, "transpo_node.tntp"), "w") as f:
        f.write("Node\tX\tY\t;\n")
        for n, name in node_id.items():
            f.write(f"{name}\t{G.nodes[n]['x']:.7f}\t{G.nodes[n]['y']:.7f}\t;\n")
        for s in schools:
            f.write(f"{zone_id[s['name']]}\t{s['lon']:.7f}\t{s['lat']:.7f}\t;\n")
    return node_id, zone_id, links, n_nodes


def write_trips(demand, n_zones, path):
    """demand: {(origin name, destination name): vehicles/hour}."""
    by_origin = {}
    for (o, d), q in demand.items():
        if q > 0 and o != d:
            by_origin.setdefault(o, []).append((d, q))
    total = sum(q for ods in by_origin.values() for _, q in ods)
    with open(path, "w") as f:
        f.write(f"<NUMBER OF ZONES> {n_zones}\n<TOTAL OD FLOW> {total:.4f}\n<END OF METADATA>\n\n\n")
        for o, ods in by_origin.items():
            f.write(f"Origin {o}\n")
            for i in range(0, len(ods), 5):
                f.write("    " + "    ".join(f"{d} :    {q:.4f};" for d, q in ods[i:i + 5]) + "\n")
            f.write("\n")


# --- engines ------------------------------------------------------------------------------------

class InfraRiskEngine:
    """InfraRisk's transportation Network: load TNTP files, fail links, run user equilibrium."""

    def __init__(self, folder, max_iter, gap):
        from infrarisk.src.physical.transportation.network import Network

        class StreetNetwork(Network):
            """InfraRisk Network with two result-neutral adjustments for street-level networks."""
            _reading = False

            @property
            def numLinks(self):
                # Link.__init__ builds sortKey = tail * numLinks + head from *string* node names;
                # report 1 while the file loads so that throwaway key stays short.
                return 1 if self._reading else self._num_links

            @numLinks.setter
            def numLinks(self, value):
                self._num_links = value

            def readNetworkFile(self, networkFileName):
                self._reading = True
                try:
                    super().readNetworkFile(networkFileName)
                finally:
                    self._reading = False

            def allOrNothing(self):
                # same loading as InfraRisk's allOrNothing, but only from origins that have trips
                flows = {ij: 0 for ij in self.link}
                by_origin = {}
                for od in self.ODpair.values():
                    if od.demand > 0:
                        by_origin.setdefault(od.origin, []).append(od)
                for origin, ods in by_origin.items():
                    backlink, _ = self.shortestPath(origin)
                    for od in ods:
                        cur = od.destination
                        while cur != origin:
                            flows[backlink[cur]] += od.demand
                            cur = self.link[backlink[cur]].tail
                return flows

        self.folder, self.max_iter, self.gap = folder, max_iter, gap
        with contextlib.redirect_stdout(io.StringIO()):
            self.net = StreetNetwork(os.path.join(folder, "transpo_net.tntp"),
                                     os.path.join(folder, "transpo_trips.tntp"),
                                     os.path.join(folder, "transpo_node.tntp"))
        self._paths = {}

    def run(self, closed_links, drop_od):
        net = self.net
        for lid in closed_links:  # InfraRisk's fail_transpo_link
            net.link[lid].freeFlowTime = FAILED_FFT
        for key in [k for k, od in net.ODpair.items() if (od.origin, od.destination) in drop_od]:
            del net.ODpair[key]
        for link in net.link.values():
            link.flow = 0
            link.updateCost()
        if any(od.demand > 0 for od in net.ODpair.values()):
            with contextlib.redirect_stdout(io.StringIO()):  # InfraRisk prints every iteration
                net.userEquilibrium("FW", self.max_iter, self.gap, net.averageExcessCost)
        for link in net.link.values():
            link.updateCost()
        self._paths = {}

    def flow(self, lid):
        return self.net.link[lid].flow

    def cost(self, lid):
        return self.net.link[lid].cost

    def route(self, origin, dest):
        """(travel time in minutes, [link ids]) on the equilibrium costs."""
        if origin not in self._paths:
            self._paths[origin] = self.net.shortestPath(origin)
        backlink, cost = self._paths[origin]
        if cost[dest] >= FAILED_FFT:
            return None, []
        path, cur = [], dest
        while cur != origin:
            path.append(backlink[cur])
            cur = self.net.link[backlink[cur]].tail
        return cost[dest], path[::-1]


class BuiltinEngine:
    """Same model (BPR costs, Frank-Wolfe user equilibrium) read from the same TNTP files."""

    def __init__(self, folder, max_iter, gap):
        self.max_iter, self.gap = max_iter, gap
        self.link, self.out = {}, {}
        with open(os.path.join(folder, "transpo_net.tntp")) as f:
            lines = f.read().splitlines()
        body = lines[next(i for i, l in enumerate(lines) if "<END OF METADATA>" in l) + 1:]
        for line in body:
            line = line.split("~")[0].strip()
            if not line:
                continue
            p = line.split()
            assert len(p) >= 12 and p[11] == ";", f"bad TNTP link line: {line}"
            self.link[p[10]] = {"tail": p[0], "head": p[1], "cap": float(p[2]), "len": float(p[3]),
                                "fft": float(p[4]), "fft0": float(p[4]), "a": float(p[5]), "b": float(p[6]),
                                "flow": 0.0}
            self.out.setdefault(p[0], []).append(p[10])
        self.od = {}
        with open(os.path.join(folder, "transpo_trips.tntp")) as f:
            lines = f.read().splitlines()
        origin = None
        for line in lines[next(i for i, l in enumerate(lines) if "<END OF METADATA>" in l) + 1:]:
            p = line.split("~")[0].split()
            if not p:
                continue
            if p[0] == "Origin":
                origin = p[1]
                continue
            assert len(p) % 3 == 0, f"bad TNTP demand line: {line}"
            for i in range(0, len(p), 3):
                assert p[i + 1] == ":" and p[i + 2].endswith(";"), f"bad TNTP demand line: {line}"
                self.od[(origin, p[i])] = float(p[i + 2][:-1])
        self._paths = {}

    def _cost(self, L, flow=None):
        x = L["flow"] if flow is None else flow
        return L["fft"] * (1 + L["a"] * (max(x, 0) / L["cap"]) ** L["b"])

    def _tree(self, origin):
        dist, back, heap = {origin: 0.0}, {}, [(0.0, origin)]
        while heap:
            d, n = heapq.heappop(heap)
            if d > dist.get(n, math.inf):
                continue
            for lid in self.out.get(n, ()):
                L = self.link[lid]
                nd = d + self._cost(L)
                if nd < dist.get(L["head"], math.inf):
                    dist[L["head"]], back[L["head"]] = nd, lid
                    heapq.heappush(heap, (nd, L["head"]))
        return dist, back

    def _aon(self):
        y = {lid: 0.0 for lid in self.link}
        by_origin = {}
        for (o, d), q in self.od.items():
            by_origin.setdefault(o, []).append((d, q))
        for o, ods in by_origin.items():
            _, back = self._tree(o)
            for d, q in ods:
                cur = d
                while cur != o:
                    y[back[cur]] += q
                    cur = self.link[back[cur]]["tail"]
        return y

    def run(self, closed_links, drop_od):
        for lid in closed_links:
            self.link[lid]["fft"] = FAILED_FFT
        self.od = {k: q for k, q in self.od.items() if k not in drop_od and q > 0}
        for L in self.link.values():
            L["flow"] = 0.0
        if self.od:
            x = self._aon()
            for lid, v in x.items():
                self.link[lid]["flow"] = v
            demand = sum(self.od.values())
            for _ in range(self.max_iter):
                y = self._aon()
                tstt = sum(L["flow"] * self._cost(L) for L in self.link.values())
                sptt = sum(y[lid] * self._cost(L) for lid, L in self.link.items())
                if (tstt - sptt) / demand < self.gap:  # average excess cost, as InfraRisk
                    break
                lo, hi = 0.0, 1.0  # line search on the Beckmann objective
                for _ in range(30):
                    mid = (lo + hi) / 2
                    g = sum(self._cost(L, L["flow"] + mid * (y[lid] - L["flow"])) * (y[lid] - L["flow"])
                            for lid, L in self.link.items())
                    lo, hi = (mid, hi) if g < 0 else (lo, mid)
                lam = (lo + hi) / 2
                for lid, L in self.link.items():
                    L["flow"] += lam * (y[lid] - L["flow"])
        self._paths = {}

    def flow(self, lid):
        return self.link[lid]["flow"]

    def cost(self, lid):
        return self._cost(self.link[lid])

    def route(self, origin, dest):
        if origin not in self._paths:
            self._paths[origin] = self._tree(origin)
        dist, back = self._paths[origin]
        if dest not in dist or dist[dest] >= FAILED_FFT:
            return None, []
        path, cur = [], dest
        while cur != origin:
            path.append(back[cur])
            cur = self.link[back[cur]]["tail"]
        return dist[dest], path[::-1]


# --- map ------------------------------------------------------------------------------------

def plot_flow_change(G, edge_of, delta, closed_edges, schools, title, path):
    """Change in peak-hour flow per road: orange = more traffic, blue = less, red = closed."""
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(11, 10))
    for u, v, k in G.edges(keys=True):
        xs, ys = base.edge_xy(G, u, v, k)
        ax.plot(xs, ys, color="#d4d4d4", lw=0.7, zorder=1)
    vmax = max([abs(x) for x in delta.values()] + [1])
    xs_all, ys_all = [], []
    for lid, dv in delta.items():
        if abs(dv) < 1:
            continue
        xs, ys = base.edge_xy(G, *edge_of[lid])
        ax.plot(xs, ys, color="#ff7f0e" if dv > 0 else "#1f77b4", lw=1 + 6 * abs(dv) / vmax, alpha=0.8,
                solid_capstyle="round", zorder=3)
        xs_all += xs
        ys_all += ys
    for e in closed_edges:
        xs, ys = base.edge_xy(G, *e)
        ax.plot(xs, ys, color="#d62728", lw=4, zorder=4)
        xs_all += xs
        ys_all += ys
    for s in schools:
        ax.scatter([s["lon"]], [s["lat"]], marker="*", s=320, color="gold", edgecolor="black", zorder=6)
        ax.annotate(s["name"], (s["lon"], s["lat"]), xytext=(8, 8), textcoords="offset points", fontsize=8,
                    fontweight="bold", zorder=7, path_effects=[pe.withStroke(linewidth=3, foreground="white")])
        xs_all.append(s["lon"])
        ys_all.append(s["lat"])
    if xs_all:
        px, py = (max(xs_all) - min(xs_all)) * 0.08 + 0.002, (max(ys_all) - min(ys_all)) * 0.08 + 0.002
        ax.set_xlim(min(xs_all) - px, max(xs_all) + px)
        ax.set_ylim(min(ys_all) - py, max(ys_all) + py)
        ax.set_aspect(1 / math.cos(math.radians(sum(ys_all) / len(ys_all))))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.apply_aspect()
    # label the roads with the largest increase, skipping labels that would overlap
    gains = {}
    for lid, dv in delta.items():
        name = first(G.edges[edge_of[lid]].get("name"))
        if name and dv > gains.get(name, (0, None))[0]:
            gains[name] = (dv, lid)
    placed = []
    for name, (dv, lid) in sorted(gains.items(), key=lambda kv: -kv[1][0])[:8]:
        xs, ys = base.edge_xy(G, *edge_of[lid])
        mx, my = xs[len(xs) // 2], ys[len(ys) // 2]
        px_, py_ = ax.transData.transform((mx, my))
        if any((px_ - a) ** 2 + (py_ - b) ** 2 < 70 ** 2 for a, b in placed):
            continue
        placed.append((px_, py_))
        ax.text(mx, my, f"{name} +{dv:.0f}", fontsize=7.5, fontweight="bold", ha="center", va="center",
                color="#a04000", zorder=8, path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])
    handles = [Line2D([], [], color="#d62728", lw=4, label="Closed"),
               Line2D([], [], color="#ff7f0e", lw=4, label="More traffic (width = vehicles/hour added)"),
               Line2D([], [], color="#1f77b4", lw=4, label="Less traffic"),
               Line2D([], [], marker="*", ls="", color="gold", markeredgecolor="black", markersize=14, label="School")]
    ax.set_title(title, fontsize=11)
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8.5, frameon=False)
    fig.subplots_adjust(bottom=0.12, top=0.92, left=0.02, right=0.98)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    if "ipykernel" in sys.modules:
        from IPython.display import Image, display
        display(Image(filename=path, width=900))


# --- main -----------------------------------------------------------------------------------

def main(argv=None):
    """argv: e.g. main(["--osm-file", "map.osm", "--infrarisk-path", "dreaminsg-integrated-model"])."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--osm-file", default=None,
                    help="downloaded OpenStreetMap .osm file (default: map.osm in the current folder)")
    ap.add_argument("--street", action="append", default=None,
                    help=f"street to close (repeatable; default: {', '.join(base.STREETS)})")
    ap.add_argument("--school", nargs=3, action="append", default=[], metavar=("NAME", "LAT", "LON"))
    ap.add_argument("--access", nargs=3, action="append", default=[], metavar=("NAME", "LAT", "LON"))
    ap.add_argument("--resident-radius-m", type=float, default=400)
    ap.add_argument("--trips-per-school", type=float, default=1.5,
                    help="peak-hour vehicles from each resident intersection to each school")
    ap.add_argument("--aadt", nargs=2, action="append", default=None, metavar=("STREET", "VEHICLES_PER_DAY"),
                    help="daily traffic on a street, loaded as through traffic (repeatable; "
                         "default: Dickson Avenue 1900 if it is in the map)")
    ap.add_argument("--peak-hour-factor", type=float, default=PEAK_HOUR_FACTOR)
    ap.add_argument("--engine", choices=["infrarisk", "builtin"], default="infrarisk")
    ap.add_argument("--infrarisk-path", default=None, help="folder of the cloned InfraRisk repository")
    ap.add_argument("--max-iter", type=int, default=50, help="Frank-Wolfe iterations")
    ap.add_argument("--gap", type=float, default=0.001, help="stop when average excess cost < gap (min)")
    ap.add_argument("--out", default="infrarisk_closure_results")
    ap.add_argument("--no-plot", action="store_true")
    argv = sys.argv[1:] if argv is None else [str(a) for a in argv]
    args, _ = ap.parse_known_args(argv)
    if not args.osm_file and os.path.exists("map.osm"):
        args.osm_file = "map.osm"
        print(f"Using map.osm from {os.getcwd()}")
    if not args.osm_file or not os.path.exists(args.osm_file):
        print(("No road map given. " if not args.osm_file else f"Map file not found: {args.osm_file}. ")
              + f"Current folder: {os.getcwd()}\n"
              'Run it with the map file, e.g.\n'
              '  main(["--osm-file", "map.osm", "--infrarisk-path", "dreaminsg-integrated-model"])\n'
              '  %run infrarisk_closure_assignment.py --osm-file map.osm --infrarisk-path dreaminsg-integrated-model')
        return None
    if args.engine == "infrarisk":
        if args.infrarisk_path:
            sys.path.insert(0, os.path.abspath(args.infrarisk_path))
        try:
            importlib.import_module("infrarisk.src.physical.transportation.network")
        except ImportError:
            raise SystemExit("InfraRisk not found. Clone it and pass its folder:\n"
                             "  git clone https://github.com/srijithbalakrishnan/dreaminsg-integrated-model.git\n"
                             "  --infrarisk-path dreaminsg-integrated-model\n"
                             "or run with --engine builtin.")
    base.T0 = time.time()
    run_date = dt.datetime.now().astimezone()
    streets = args.street or base.STREETS
    aadt = ({s: float(v) for s, v in args.aadt} if args.aadt else dict(DEFAULT_AADT))
    # base helpers read these option names
    args.radius_m, args.school, args.access = base.DOWNLOAD_RADIUS_M, args.school, args.access

    # 1. network and schools (via the closure-simulation helpers)
    schools = base.find_schools(args, base.AREA_CENTER)
    if not schools:
        raise SystemExit('No schools located; add them with --school "Name" LAT LON.')
    G = base.load_osm_file(args.osm_file, keep_areas=[s["grounds"].buffer(base.GROUNDS_BUFFER_DEG) for s in schools])
    G = base.add_travel_times(G)
    for s in schools:
        gates = [(float(a), float(b)) for n, a, b in args.access
                 if all(w in n.lower() for w in next((kw for lab, kw in base.SCHOOLS if lab == s["name"]),
                                                     tuple(s["name"].lower().split())))]
        s["targets"], s["access_method"] = base.school_entrances(G, s, gates)
    ground = {e for e in G.edges(keys=True) if any(base.on_grounds(G, e, s["grounds"]) for s in schools)}
    G.remove_edges_from(ground)  # school driveways only locate entrances; never cut-throughs
    keep = max(nx.strongly_connected_components(G), key=len)
    G = G.subgraph(keep).copy()
    for s in schools:
        s["targets"] = [n for n in s["targets"] if n in keep] or [
            min(keep, key=lambda n: base.dist_m(G, n, s["lat"], s["lon"]))]
    step(f"road network: {G.number_of_nodes()} intersections, {G.number_of_edges()} road segments; "
         f"{len(schools)} schools")

    street_edges = {}
    for st in streets:
        es = [(u, v, k) for u, v, k, d in G.edges(keys=True, data=True) if on_street(d, st)]
        if es:
            street_edges[st] = es
        else:
            alt = base.similar_streets(G, st)
            print(f"WARNING: {st} is not in the map" + (f"; similar names: {alt}" if alt else ""))
    if not street_edges:
        raise SystemExit("None of the streets to close is in the road network.")

    # 2. export to InfraRisk format
    folder = os.path.join(args.out, "infrarisk_transport")
    node_id, zone_id, links, n_zones = export_tntp(G, schools, folder)
    lid_of_edge = {L[8]: L[0] for L in links if L[8] is not None}
    length_of = {L[0]: L[4] for L in links}
    edge_of = {v: k for k, v in lid_of_edge.items()}
    name_of_node = {v: k for k, v in node_id.items()}

    # 3. peak-hour trips
    nodes_of = {n for es in street_edges.values() for e in es for n in e[:2]}
    entrances = {n for s in schools for n in s["targets"]}
    residents = [n for n in G.nodes if n not in entrances and
                 min(base.dist_m(G, m, G.nodes[n]["y"], G.nodes[n]["x"]) for m in nodes_of) <= args.resident_radius_m]
    demand = {(node_id[o], zone_id[s["name"]]): args.trips_per_school for o in residents for s in schools}
    through = []
    for st, daily in aadt.items():
        es = [(u, v, k) for u, v, k, d in G.edges(keys=True, data=True) if on_street(d, st)]
        if not es:
            print(f"  AADT for {st} ignored: street not in the map")
            continue
        sn = {n for e in es for n in e[:2]}
        joins = [n for n in sn if not all(on_street(G.edges[e], st) for e in
                                           list(G.out_edges(n, keys=True)) + list(G.in_edges(n, keys=True)))] or list(sn)
        a, b = max(((x, y) for x in joins for y in joins),
                   key=lambda p: base.dist_m(G, p[0], G.nodes[p[1]]["y"], G.nodes[p[1]]["x"]))
        q = daily * args.peak_hour_factor / 2
        for o, d in ((a, b), (b, a)):
            demand[(node_id[o], node_id[d])] = demand.get((node_id[o], node_id[d]), 0) + q
            through.append({"street": st, "from": node_street(G, o), "to": node_street(G, d),
                            "o": node_id[o], "d": node_id[d], "vehicles_per_hour": q})
    write_trips(demand, n_zones, os.path.join(folder, "transpo_trips.tntp"))
    step(f"InfraRisk files written to {folder}/ ({len(links)} links); {len(residents)} resident intersections, "
         f"{sum(demand.values()):.0f} peak-hour vehicles")

    # 4. scenarios
    scenarios = [{"name": "All roads open", "streets": [], "edges": []}]
    scenarios += [{"name": f"{st} closed", "streets": [st], "edges": es} for st, es in street_edges.items()]
    if len(street_edges) > 1:
        scenarios.append({"name": "All closed together", "streets": list(street_edges),
                          "edges": [e for es in street_edges.values() for e in es]})
    Engine = InfraRiskEngine if args.engine == "infrarisk" else BuiltinEngine
    results, od_rows, flows = {}, [], {}
    for sc in scenarios:
        t0 = time.time()
        H = G.copy()
        H.remove_edges_from(sc["edges"])
        can_reach = {}  # destination -> every node with an open route to it
        for d in {d for _, d in demand}:
            targets = (next(s["targets"] for s in schools if zone_id[s["name"]] == d) if d in zone_id.values()
                       else [name_of_node[d]])
            can_reach[d] = set(targets).union(*(nx.ancestors(H, t) for t in targets))
        drop = {(o, d) for (o, d) in demand if name_of_node[o] not in can_reach[d]}
        eng = Engine(folder, args.max_iter, args.gap)
        eng.run([lid_of_edge[e] for e in sc["edges"]], drop)
        flows[sc["name"]] = {L[0]: eng.flow(L[0]) for L in links if L[8] is not None}
        tstt = sum(eng.flow(L[0]) * eng.cost(L[0]) for L in links) / 60
        results[sc["name"]] = {"tstt": tstt, "dropped": drop}
        for o in residents:
            for s in schools:
                tt, path = ((None, []) if (node_id[o], zone_id[s["name"]]) in drop
                            else eng.route(node_id[o], zone_id[s["name"]]))
                dist = sum(length_of[lid] for lid in path) if path else None
                od_rows.append({"scenario": sc["name"], "school": s["name"], "origin": node_street(G, o),
                                "origin_node": o, "time_min": tt, "distance_mi": dist})
        for t in through:
            tt, _ = (None, []) if (t["o"], t["d"]) in drop else eng.route(t["o"], t["d"])
            t[sc["name"]] = tt
        step(f"{sc['name']}: assignment done in {time.time() - t0:.0f} s, {tstt:.1f} vehicle-hours, "
             f"{len(drop)} trip type(s) with no open route")

    # 5. results
    od = pd.DataFrame(od_rows)
    base_od = od[od["scenario"] == "All roads open"].set_index(["school", "origin_node"])
    summary = []
    for sc in scenarios[1:]:
        cur = od[od["scenario"] == sc["name"]].set_index(["school", "origin_node"])
        for s in schools:
            b, c = base_od.loc[s["name"]], cur.loc[s["name"]]
            both = b["time_min"].notna() & c["time_min"].notna()
            extra = (c["time_min"] - b["time_min"])[both]
            summary.append({"Scenario": sc["name"], "School": s["name"], "Residents": int(b["time_min"].notna().sum()),
                            "Avg time open (min)": round(b["time_min"][both].mean(), 2),
                            "Avg time closed (min)": round(c["time_min"][both].mean(), 2),
                            "Trips longer": int((extra > 0.05).sum()),
                            "Cannot reach": int((b["time_min"].notna() & c["time_min"].isna()).sum()),
                            "Max extra (min)": round(extra.max(), 2) + 0.0 if len(extra) else 0.0})
    summary = pd.DataFrame(summary)
    network = pd.DataFrame([{"Scenario": sc["name"],
                             "Vehicle-hours in peak hour": round(results[sc["name"]]["tstt"], 1),
                             "Extra vs open": round(results[sc["name"]]["tstt"] - results["All roads open"]["tstt"], 1),
                             "Vehicles/hour with no open route": round(sum(demand[k] for k in results[sc["name"]]["dropped"]), 1)}
                            for sc in scenarios])
    thr = pd.DataFrame([{"street": t["street"], "trip": f"{t['from']} -> {t['to']}",
                         "vehicles_per_hour": round(t["vehicles_per_hour"], 1),
                         **{sc["name"]: (round(t[sc["name"]], 2) if t[sc["name"]] is not None else "no route")
                            for sc in scenarios}} for t in through])
    detour_rows = []
    for sc in scenarios[1:]:
        best = {}
        for lid, f in flows[sc["name"]].items():
            dv = f - flows["All roads open"][lid]
            d = G.edges[edge_of[lid]]
            name = first(d.get("name")) or "(unnamed)"
            if dv > best.get(name, {"dv": 0})["dv"]:
                best[name] = {"dv": dv, "flow": f, "cap": link_capacity(d)}
        for name, b in sorted(best.items(), key=lambda kv: -kv[1]["dv"])[:8]:
            detour_rows.append({"Scenario": sc["name"], "Road": name, "Added vehicles/hour": round(b["dv"], 1),
                                "Flow (veh/h)": round(b["flow"], 1), "Volume/capacity": round(b["flow"] / b["cap"], 2)})
    detours = pd.DataFrame(detour_rows)

    engine_name = ("InfraRisk transportation model (Network.userEquilibrium, Frank-Wolfe)"
                   if args.engine == "infrarisk" else "built-in Frank-Wolfe (same BPR model)")
    print(f"\n=== Congested closure analysis - {engine_name} ===")
    print(f"Peak hour: {args.trips_per_school:g} veh/h from each of {len(residents)} resident intersections to each "
          f"school; through traffic from AADT x {args.peak_hour_factor:g}: "
          + (", ".join(f"{s} {v:,.0f}/day" for s, v in aadt.items()) or "none"))
    print("\n--- Network ---")
    print(network.to_string(index=False))
    if (network["Vehicles/hour with no open route"] > 0).any():
        print("Note: vehicles with no open route cannot travel, so they drop out of the vehicle-hours; "
              "a lower total there is not a benefit.")
    print("\n--- Residents' trips to schools (congested travel time) ---")
    print(summary.to_string(index=False))
    if len(thr):
        print("\n--- Through traffic, travel time in minutes per scenario ---")
        print(thr.to_string(index=False))
    if len(detours):
        print("\n--- Roads absorbing the diverted traffic ---")
        print(detours.to_string(index=False))

    os.makedirs(args.out, exist_ok=True)
    network.to_csv(os.path.join(args.out, "network_summary.csv"), index=False)
    summary.to_csv(os.path.join(args.out, "school_trips_summary.csv"), index=False)
    od.to_csv(os.path.join(args.out, "resident_school_times.csv"), index=False)
    thr.to_csv(os.path.join(args.out, "through_traffic.csv"), index=False)
    detours.to_csv(os.path.join(args.out, "detour_roads.csv"), index=False)
    for sc in scenarios[1:]:
        pd.DataFrame({"link_id": [lid_of_edge[e] for e in sc["edges"]],
                      "street": [first(G.edges[e].get("name")) for e in sc["edges"]]}).to_csv(
            os.path.join(folder, "closed_links_" + "".join(ch if ch.isalnum() else "_" for ch in sc["name"].lower()) + ".csv"),
            index=False)
    with open(os.path.join(args.out, "run_record.json"), "w") as f:
        json.dump({"engine": engine_name, "osm_file": args.osm_file, "run_date": run_date.isoformat(timespec="seconds"),
                   "infrarisk_files": folder, "failed_link_free_flow_time_min": FAILED_FFT,
                   "bpr": {"alpha": BPR_ALPHA, "beta": BPR_BETA}, "capacity_vph_per_lane": CAPACITY_VPH_PER_LANE,
                   "trips_per_school": args.trips_per_school, "aadt": aadt, "peak_hour_factor": args.peak_hour_factor,
                   "frank_wolfe": {"max_iter": args.max_iter, "gap_min": args.gap},
                   "schools": [{"name": s["name"], "access_method": s["access_method"],
                                "entrances": [node_street(G, n) for n in s["targets"]]} for s in schools]},
                  f, indent=2)
    if not args.no_plot:
        for i, sc in enumerate(scenarios[1:], 1):
            delta = {lid: flows[sc["name"]][lid] - flows["All roads open"][lid] for lid in flows[sc["name"]]}
            extra = results[sc["name"]]["tstt"] - results["All roads open"]["tstt"]
            slug = "".join(ch if ch.isalnum() else "_" for ch in sc["name"].lower()).strip("_")[:40]
            plot_flow_change(G, edge_of, delta, sc["edges"], schools,
                             f"{sc['name']}: change in peak-hour traffic ({engine_name.split(' (')[0]})\n"
                             f"{extra:+.1f} vehicle-hours in the peak hour",
                             os.path.join(args.out, f"flow_change_{i}_{slug}.png"))
    print(f"\nOutputs written to {os.path.abspath(args.out)}/")
    return network, summary, od, thr, detours


if __name__ == "__main__":
    main()
