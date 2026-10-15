%matplotlib inline
# ===== Capacitance & Force vs Displacement - run AFTER the main cell (uses its trials, out_dir) =====
CFD_RELIABLE_ONLY = True                # only trials with a reliable SEC-MTS alignment
CFD_EVERY = 2                           # plot every n-th sample (20 Hz -> 10 Hz) to keep it light

use = [tr for tr in trials if tr["offset_ok"]] if CFD_RELIABLE_ONLY else trials
fig, axes = plt.subplots(2, len(use), figsize=(4.2 * len(use), 7.6), squeeze=False, sharey="row")
rows = []
for j, tr in enumerate(use):
    disp = tr["targets"]["Displacement_m"] * 1000          # mm
    force = tr["targets"]["Force_N"]
    cap = tr["x"] - np.median(tr["x"])                    # pF, zeroed per trial (baselines differ)
    t = tr["t"]
    k = slice(None, None, CFD_EVERY)
    for ax, y, lab in ((axes[0, j], cap, "Capacitance change (pF)"), (axes[1, j], force, "Force (N)")):
        sc = ax.scatter(disp[k], y[k], c=t[k], cmap="viridis", s=3, lw=0)
        r = np.corrcoef(disp, y)[0, 1]
        fit = np.polyfit(disp, y, 1)
        xx = np.linspace(disp.min(), disp.max(), 2)
        ax.plot(xx, np.polyval(fit, xx), color="#eb6834", lw=1.8,
                label=f"linear fit: R2 = {r ** 2:.2f}\nslope = {fit[0]:.4g} per mm")
        ax.set_xlabel("Displacement (mm)")
        ax.grid(color="#e4e3df", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="best", frameon=False, fontsize=8)
        if j == 0:
            ax.set_ylabel(lab)
    axes[0, j].set_title(tr["name"], fontweight="bold")
    rc, rf = np.corrcoef(disp, cap)[0, 1], np.corrcoef(disp, force)[0, 1]
    rows.append({"Trial": tr["name"],
                 "Cap~Disp r": rc, "Cap~Disp R2": rc ** 2,
                 "Cap slope (pF/mm)": np.polyfit(disp, cap, 1)[0],
                 "Force~Disp r": rf, "Force~Disp R2": rf ** 2,
                 "Stiffness (N/mm)": np.polyfit(disp, force, 1)[0],
                 "Cap~Force r": np.corrcoef(force, cap)[0, 1]})
fig.colorbar(sc, ax=axes, fraction=0.015, pad=0.02, label="Time (s)")
fig.suptitle("SEC capacitance and MTS force vs MTS displacement (colour = time, loops = hysteresis)",
             fontweight="bold")
png = os.path.join(out_dir, "SEC_MTS_cap_force_vs_disp.png")
fig.savefig(png, dpi=150, bbox_inches="tight")
print(f"Saved: {png}")
plt.show()
cfd = pd.DataFrame(rows)
print(cfd.round(4).to_string(index=False))
save_workbook(os.path.join(out_dir, "SEC_MTS_cap_force_vs_disp.xlsx"), {"Summary": cfd})
