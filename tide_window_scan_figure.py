import sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
d = pd.read_csv(sys.argv[1])
fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
for tau, ls, mk in [(0.5, "-", "o"), (1.5, "--", "s")]:
    q = d[d.tau_h == tau]
    axes[0].plot(q.window_h, q["rho tide BVG minor (Regular)"], ls, marker=mk, color="#1B7F8E", label=f"Regular tide, τ {tau} h")
    axes[0].plot(q.window_h, q["rho tide BVG minor (King)"], ls, marker=mk, color="#E07A2F", label=f"King tide, τ {tau} h")
    axes[2].plot(q.window_h, q["TDA gain %"], ls, marker=mk, color="#0B2E3C", label=f"τ {tau} h")
q = d[d.tau_h == 0.5]
axes[1].plot(q.window_h, q["sep loop BVG minor"], "-o", color="#2a78d6", label="Gaussian ellipse (minor axis)")
axes[1].plot(q.window_h, q["sep loop VR H1"], "-o", color="#6250d6", label="VR H1")
axes[1].plot(q.window_h, q["sep loop Alpha H1"], "-o", color="#1baf7a", label="Alpha H1")
titles = ["Gaussian ellipse (tide) vs ditch depth: Spearman ρ\n(more negative = stronger link)",
          "Regular vs King tide separation, tide–ditch loop\n(Cohen d, per tidal cycle)",
          "Depth prediction: gain from adding all tide TDA\nfeatures to gradient boosting (%)"]
for ax, t in zip(axes, titles):
    for x, lab in [(12.42, "1 tide"), (24.84, "1 day")]:
        ax.axvline(x, color="#b8b7b2", lw=1, ls=":")
        ax.text(x, 1.01, lab, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8, color="#8a8984")
    ax.axvspan(14, 19.5, color="#F2B33D", alpha=0.12, lw=0)
    ax.set_title(t, fontsize=10, fontweight="bold", pad=16)
    ax.set_xlabel("window length (h)")
    ax.grid(color="#e4e3df", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
axes[2].axhline(0, color="#3d3d3a", lw=0.8)
fig.suptitle("What changes with the window size (both events, scored on the same hours; shaded = 15-18.5 h)", fontweight="bold")
fig.tight_layout()
fig.savefig(sys.argv[2], dpi=150)
