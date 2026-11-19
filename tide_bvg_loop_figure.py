import sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
src = open("/home/user/supply-chain-datasets/tide_tda_analysis.py").read()
g = {"__name__": "x"}; exec(src[:src.index("# === Run ===")], g)
evs = {}
for p in g["find_workbooks"](sys.argv[1]):
    ev = g["load_event"](p); g["prepare"](ev); evs[ev["type"]] = ev
w = 60
def windows(ev):
    out = []
    for e in range(w - 1 + 160, len(ev["h"])):                 # after the first ~40 h
        lt, ld = g["scale"](ev["tide"][e-w+1:e+1]), g["scale"](ev["depth"][e-w+1:e+1])
        out.append((g["bvg"](np.column_stack([lt, ld]))[0], e))
    return out
coupled = min(windows(evs["King Tide"]))
decoupled = max(windows(evs["Regular Tide"]))
fig, axes = plt.subplots(2, 2, figsize=(11, 8), gridspec_kw={"height_ratios": [1, 1.25]})
for col, (title, ev, (bmin, e), c) in enumerate([("Ditch coupled to the tide (King tide)", evs["King Tide"], coupled, "#1B7F8E"),
                                                  ("Ditch disconnected (Regular tide)", evs["Regular Tide"], decoupled, "#E07A2F")]):
    sl = slice(e - w + 1, e + 1)
    ax = axes[0, col]
    ax.plot(ev["h"][sl] - ev["h"][sl][0], ev["tide"][sl], color="#0B2E3C", lw=1.6, label="tide WSE (m)")
    ax.plot(ev["h"][sl] - ev["h"][sl][0], ev["depth"][sl], color=c, lw=1.8, label="ditch depth (m)")
    ax.axhline(0.5, color="#8a8984", lw=0.8, ls=":")
    ax.set_title(title, fontsize=12, fontweight="bold", color=c)
    ax.set_xlabel("hours in the 15 h window")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax = axes[1, col]
    lt, ld = g["scale"](ev["tide"][sl]), g["scale"](ev["depth"][sl])
    pc = np.column_stack([lt, ld])
    ax.plot(lt, ld, ".", color=c, ms=6)
    m, r, lam, vec = g["bvg"](pc)
    ang = np.linspace(0, 2 * np.pi, 200)
    circ = np.stack([2 * np.sqrt(lam[1]) * np.cos(ang), 2 * np.sqrt(lam[0]) * np.sin(ang)])
    gx = pc.mean(axis=0)[:, None] + vec[:, [1, 0]] @ circ
    ax.plot(gx[0], gx[1], color="#0B2E3C", lw=2, ls="--")
    ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6); ax.set_aspect("equal")
    ax.set_xlabel("tide (scaled in window)"); ax.set_ylabel("ditch depth (scaled in window)")
    ax.set_title(f"Gaussian ellipse: minor axis {m:.2f}, ratio {r:.2f}", fontsize=11)
    for a in axes[:, col]:
        a.grid(color="#e4e3df", lw=0.6); a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(sys.argv[2], dpi=170)
print("coupled", coupled[0], "decoupled", decoupled[0])
