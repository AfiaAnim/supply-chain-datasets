"""Window-size scan for the tide analysis. Usage: python tide_window_scan.py <folder with the tide xlsx> <out.csv>
Then: python tide_window_scan_figure.py <out.csv> <figure.png>"""
import os, sys, time, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tide_tda_analysis.py")).read()
g = {"__name__": "x"}; exec(src[:src.index("# === Run ===")], g)
from sklearn.ensemble import HistGradientBoostingRegressor
events = []
for p in g["find_workbooks"](sys.argv[1]):
    ev = g["load_event"](p); g["prepare"](ev); events.append(ev)
fq = g["freq"]
FEATS = {"VR H1": g["vr_h1"], "Alpha H1": g["alpha_h1"], "FastTDA": lambda pc: g["fasttda_fit"](pc)[0],
         "BVG minor": lambda pc: g["bvg"](pc)[0], "BVG ratio": lambda pc: g["bvg"](pc)[1]}
memory = ["tide", "tide_1h", "tide_2h", "tide_3h", "slope_1h", "tide_max_6h"]
start_h = 30.0                                     # common start: every window size scored on the same hours
out = []
for win_h in [6, 9, 12.4, 15, 18.5, 24.8, 30]:
    for tau_h in [0.5, 1.5]:
        t0 = time.time()
        w, tau = int(round(win_h * fq)), max(1, int(round(tau_h * fq)))
        if w - tau < 8:
            continue
        g["s"] = 1
        rows = []
        for ev in events:
            mem = g["tide_memory"](ev)
            for e in range(max(w - 1, int(start_h * fq)), len(ev["h"])):
                cl = g["clouds_of"](ev, e, w, tau)
                r = {"Event": ev["name"], "Type": ev["type"], "Hour": ev["h"][e], "Ditch_depth_m": ev["depth"][e],
                     **mem.iloc[e].to_dict()}
                for c in ("tide", "loop"):
                    for f, fun in FEATS.items():
                        r[f"{c}: {f}"] = fun(cl[c])
                rows.append(r)
        t = pd.DataFrame(rows)
        res = {"window_h": win_h, "tau_h": tau_h, "windows": len(t)}
        for f in FEATS:
            for ty in ("Regular Tide", "King Tide"):
                m = t.Type == ty
                res[f"rho tide {f} ({ty.split()[0]})"] = t.loc[m, f"tide: {f}"].corr(t.loc[m, "Ditch_depth_m"], method="spearman")
            cyc = t.assign(c=(t.Hour // 12.42).astype(int)).groupby(["Event", "c"]).agg(v=(f"loop: {f}", "max"), ty=("Type", "first"))
            a, b = cyc.loc[cyc.ty == "Regular Tide", "v"].dropna(), cyc.loc[cyc.ty == "King Tide", "v"].dropna()
            res[f"sep loop {f}"] = (a.median() - b.median()) / np.sqrt((a.var() + b.var()) / 2)
        res["FastTDA fail loop %"] = 100 * t["loop: FastTDA"].isna().mean()
        res["FastTDA fail tide %"] = 100 * t["tide: FastTDA"].isna().mean()
        blocks = t["Event"] + (t["Hour"] // 24).astype(int).astype(str)
        y = t["Ditch_depth_m"].to_numpy()
        for name, cols in [("GB memory", memory), ("GB memory+TDA", memory + [f"tide: {f}" for f in FEATS])]:
            pr = np.full(len(t), np.nan)
            X = t[cols].fillna(0)
            for b in blocks.unique():
                te = (blocks == b).to_numpy()
                gb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                                   min_samples_leaf=20, random_state=0).fit(X[~te], y[~te])
                pr[te] = gb.predict(X[te])
            res[f"RMSE {name}"] = np.sqrt(np.mean((pr - y) ** 2))
        res["TDA gain %"] = 100 * (1 - res["RMSE GB memory+TDA"] / res["RMSE GB memory"])
        out.append(res)
        print(f"window {win_h:5.1f} h tau {tau_h} h: {len(t)} windows, gain {res['TDA gain %']:+.1f}%, "
              f"rho BVG {res['rho tide BVG minor (Regular)']:+.2f}/{res['rho tide BVG minor (King)']:+.2f}, "
              f"sep loop BVG {res['sep loop BVG minor']:.2f}, VR {res['sep loop VR H1']:.2f} ({time.time() - t0:.0f} s)", flush=True)
pd.DataFrame(out).to_csv(sys.argv[2], index=False)
