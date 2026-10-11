"""
Visualize the FastTDA fine-tuning results produced by trial_fasttda_finetune.py.

Figures:
    1. Signal (raw + filtered capacitance) and FastTDA (2b) over time for one profile
    2. FastTDA over time for every window/delay combination (one panel per window length)
    3. Grid overview heatmaps: invalid-fit rate and mean FastTDA for each window x delay
    4. Example windows: Takens point cloud with the fitted FastTDA ellipse

Run it after trial_fasttda_finetune.py. Figures are shown and saved as PNGs next to the
results file.

Requirements:
    pip install numpy pandas scipy scikit-learn matplotlib openpyxl
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import MinMaxScaler


# ============================================================
# USER CONFIGURATION
# ============================================================

CSV_PATH = r"C:\Users\aphya\Downloads\trial_01.csv"
RESULTS_XLSX = None          # None = <trial>_FastTDA_FineTune.xlsx next to the CSV

TIME_COL = "time_s"
SIGNAL_COL = "capacitance_filtered_pF"   # the column the analysis used
RAW_COL = "capacitance_raw_pF"           # shown for reference in figure 1 (None to hide)

PROFILE = None               # e.g. "W133_T5"; None = profile with the fewest invalid fits
EXAMPLE_TIMES = None         # e.g. [10, 30, 50] (seconds, window end); None = 3 spread-out windows
SAVE_FIGS = True             # save PNGs next to the results file


# ============================================================
# STYLE (validated default palette, light surface)
# ============================================================

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
NEUTRAL = "#a3a29c"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_2,
    "axes.titlecolor": TEXT,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": TEXT_2,
    "ytick.color": TEXT_2,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "lines.linewidth": 1.5,
    "font.size": 9,
})


# ============================================================
# HELPERS (same definitions as trial_fasttda_finetune.py)
# ============================================================

def to_local_path(path: str) -> str:
    path = str(path).strip().strip('"')
    if os.name != "nt" and len(path) > 2 and path[1] == ":" and path[2] in "\\/":
        return f"/mnt/{path[0].lower()}/{path[3:].replace(chr(92), '/')}"
    return path


def load_csv(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        first = f.readline()
    counts = {d: first.count(d) for d in (",", ";", "\t")}
    sep = max(counts, key=counts.get) if max(counts.values()) > 0 else r"\s+"
    df = pd.read_csv(path, sep=sep, engine="python", encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, [c for c in df.columns if not c.startswith("Unnamed")]]
    return df.apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any").reset_index(drop=True)


def minmax_minus1_plus1(x):
    return MinMaxScaler(feature_range=(-1, 1)).fit_transform(np.asarray(x, float).reshape(-1, 1)).ravel()


def takens_points(x, delay, dim, stride):
    span = len(x) - (dim - 1) * delay - 1
    if span < 0:
        return np.empty((0, dim))
    n_points = span // stride + 1
    first = span % stride
    last = first + (n_points - 1) * stride
    return np.column_stack([x[first + d * delay: last + d * delay + 1: stride] for d in range(dim)])


def fit_conic(X, Y):
    """Same SVD-LSE fit as FastTDA; returns (A, B, C, D, E, F, mean_x, mean_y, scale) or None."""
    mx, my = np.mean(X), np.mean(Y)
    Xc, Yc = X - mx, Y - my
    max_range = max(np.max(np.abs(Xc)), np.max(np.abs(Yc)))
    if not np.isfinite(max_range) or max_range == 0:
        return None
    scale = 1.5 / max_range
    Xc, Yc = Xc * scale, Yc * scale
    G = np.column_stack([Xc**2 - Yc**2, Xc * Yc, Xc, Yc, np.ones_like(Xc)])
    U, S, Vt = np.linalg.svd(G, full_matrices=False)
    rcond = np.finfo(float).eps * max(G.shape)
    Sinv = np.where(S > rcond * S[0], 1.0 / S, 0.0)
    A, B, D, E, F = (Vt.T * Sinv) @ (U.T @ (-(Yc**2)))
    C = 1.0 - A
    if 4.0 * A * C - B**2 <= 0.0:
        return None
    return A, B, C, D, E, F, mx, my, scale


def save(fig, out_dir, name, top=1.0):
    fig.tight_layout(rect=(0, 0, 1, top))
    if SAVE_FIGS:
        path = os.path.join(out_dir, name)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.show()


# ============================================================
# FIGURES
# ============================================================

def fig_signal_and_fasttda(df_csv, prof_df, prof_name, out_dir):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1]})
    t = df_csv[TIME_COL]
    if RAW_COL and RAW_COL in df_csv and RAW_COL != SIGNAL_COL:
        ax1.plot(t, df_csv[RAW_COL], color=NEUTRAL, lw=0.8, label=RAW_COL)
    ax1.plot(t, df_csv[SIGNAL_COL], color=SERIES[0], lw=1.5, label=SIGNAL_COL)
    ax1.set_ylabel("Capacitance (pF)")
    ax1.set_title("Capacitance signal")
    ax1.legend(loc="upper right", ncol=2)

    ax2.plot(prof_df["Time"], prof_df["FastTDA"], color=SERIES[0], lw=1.5, label="FastTDA (2b)")
    bad = prof_df[~prof_df["FastTDA_FitOK"].astype(bool)]
    if len(bad):
        y0 = np.nanmin(prof_df["FastTDA"])
        ax2.plot(bad["Time"], np.full(len(bad), y0), "|", color=SERIES[1], ms=8, mew=1,
                 label=f"invalid ellipse fit ({len(bad)} windows)")
    ax2.set_ylabel("FastTDA minor axis 2b")
    ax2.set_xlabel("Time (s), window end")
    ax2.set_title(f"FastTDA over time - profile {prof_name}")
    ax2.legend(loc="upper right")
    save(fig, out_dir, "fig1_signal_fasttda.png")


def fig_all_profiles(sheets, summary, out_dir):
    windows = sorted(summary["Window_Size"].unique())
    delays = sorted(summary["Takens_Delay"].unique())
    color_of = {d: SERIES[i % len(SERIES)] for i, d in enumerate(delays)}   # fixed per delay
    fig, axes = plt.subplots(len(windows), 1, figsize=(11, 2.6 * len(windows)),
                             sharex=True, sharey=True, squeeze=False)
    # Robust shared y-range: rare fit spikes would otherwise flatten every line
    allv = np.concatenate([sheets[p]["FastTDA"].to_numpy(float) for p in summary["Profile"]])
    allv = allv[np.isfinite(allv)]
    ymax = np.percentile(allv, 99.5) * 1.1
    n_clipped = int((allv > ymax).sum())
    for ax, w in zip(axes[:, 0], windows):
        rows = summary[summary["Window_Size"] == w]
        for _, r in rows.sort_values("Takens_Delay").iterrows():
            d = r["Takens_Delay"]
            df = sheets[r["Profile"]]
            ax.plot(df["Time"], df["FastTDA"], color=color_of[d], lw=1.2,
                    label=f"delay {r['Delay_s']:.3f} s ({d} samples)")
        ax.set_title(f"Window {rows['Window_s'].iloc[0]:.2f} s ({w} samples)", loc="left")
        ax.set_ylabel("FastTDA 2b")
    axes[0, 0].set_ylim(0, ymax)
    axes[-1, 0].set_xlabel("Time (s), window end")
    note = f" ({n_clipped} spikes above {ymax:.2f} not shown)" if n_clipped else ""
    fig.suptitle(f"FastTDA for every window x delay combination{note}", color=TEXT,
                 fontweight="bold", y=0.995)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(delays), bbox_to_anchor=(0.5, 0.965))
    save(fig, out_dir, "fig2_all_profiles.png", top=0.94)


def _heatmap(ax, matrix, row_labels, col_labels, title, fmt):
    im = ax.imshow(matrix, cmap=SEQ_CMAP, aspect="auto")
    ax.set_xticks(range(len(col_labels)), col_labels)
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.set_xlabel("Takens delay")
    ax.set_ylabel("Window length")
    ax.set_title(title)
    ax.grid(False)
    lo, hi = np.nanmin(matrix), np.nanmax(matrix)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if np.isnan(v):
                continue
            dark = (v - lo) / (hi - lo + 1e-12) > 0.55
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=9,
                    color="#ffffff" if dark else TEXT)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def fig_grid_overview(summary, out_dir):
    s = summary.copy()
    s["Invalid_pct"] = 100 * s["FastTDA_Invalid_Count"] / s["Windows"]
    windows = sorted(s["Window_Size"].unique())
    delays = sorted(s["Takens_Delay"].unique())
    wl = [f"{s.loc[s.Window_Size == w, 'Window_s'].iloc[0]:.2f} s" for w in windows]
    dl = [f"{s.loc[s.Takens_Delay == d, 'Delay_s'].iloc[0]:.3f} s" for d in delays]
    inv = s.pivot(index="Window_Size", columns="Takens_Delay", values="Invalid_pct").loc[windows, delays]
    mean = s.pivot(index="Window_Size", columns="Takens_Delay", values="FastTDA_mean").loc[windows, delays]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    _heatmap(ax1, inv.to_numpy(), wl, dl, "Invalid ellipse fits (% of windows)", "{:.1f}%")
    _heatmap(ax2, mean.to_numpy(), wl, dl, "Mean FastTDA 2b", "{:.3f}")
    save(fig, out_dir, "fig3_grid_overview.png")


def fig_embedding_examples(df_csv, prof_df, window_size, delay, stride, prof_name, out_dir):
    t = df_csv[TIME_COL].to_numpy(float)
    x = df_csv[SIGNAL_COL].to_numpy(float)
    if EXAMPLE_TIMES:
        ends = [int(np.argmin(np.abs(t - te))) + 1 for te in EXAMPLE_TIMES]
    else:
        ends = [int(len(x) * f) for f in (0.2, 0.5, 0.8)]
    ends = [min(max(e, window_size), len(x)) for e in ends]

    fig, axes = plt.subplots(1, len(ends), figsize=(4 * len(ends), 4.2), squeeze=False)
    for ax, end in zip(axes[0], ends):
        pc = takens_points(minmax_minus1_plus1(x[end - window_size:end]), delay, 2, stride)
        ax.plot(pc[:, 0], pc[:, 1], color=GRID, lw=0.8, zorder=1)
        ax.scatter(pc[:, 0], pc[:, 1], s=14, color=SERIES[0], edgecolor=SURFACE, lw=0.6,
                   zorder=2, label="embedded points")
        conic = fit_conic(pc[:, 0], pc[:, 1])
        row = prof_df.iloc[(prof_df["Time"] - t[end - 1]).abs().argmin()]
        if conic is not None and bool(row["FastTDA_FitOK"]):
            A, B, C, D, E, F, mx, my, sc = conic
            gx, gy = np.meshgrid(np.linspace(-1.3, 1.3, 400), np.linspace(-1.3, 1.3, 400))
            u, v = (gx - mx) * sc, (gy - my) * sc
            ax.contour(gx, gy, A*u**2 + B*u*v + C*v**2 + D*u + E*v + F, levels=[0],
                       colors=SERIES[1], linewidths=2, zorder=3)
            ax.plot([], [], color=SERIES[1], lw=2, label="fitted ellipse")
            title = f"t = {t[end - 1]:.1f} s,  2b = {row['FastTDA']:.3f}"
        else:
            title = f"t = {t[end - 1]:.1f} s,  ellipse fit invalid"
        ax.set_title(title, fontsize=10)
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_aspect("equal")
        ax.set_xlabel("x(t)")
        ax.set_ylabel(f"x(t + \u03c4),  \u03c4 = {delay} samples")
    axes[0, 0].legend(loc="lower right")
    fig.suptitle(f"Takens embeddings and FastTDA ellipses - profile {prof_name}",
                 color=TEXT, fontweight="bold")
    save(fig, out_dir, "fig4_embedding_examples.png")


# ============================================================
# MAIN
# ============================================================

def main():
    csv_path = to_local_path(CSV_PATH)
    tag = os.path.splitext(os.path.basename(csv_path))[0]
    xlsx = to_local_path(RESULTS_XLSX) if RESULTS_XLSX else os.path.join(
        os.path.dirname(csv_path), f"{tag}_FastTDA_FineTune.xlsx")
    for p in (csv_path, xlsx):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Not found: {p}\nRun trial_fasttda_finetune.py first.")

    df_csv = load_csv(csv_path)
    book = pd.read_excel(xlsx, sheet_name=None)
    summary = book.pop("Summary")
    sheets = book
    out_dir = os.path.dirname(xlsx)

    if PROFILE:
        prof_name = PROFILE
    else:
        s = summary.assign(rate=summary["FastTDA_Invalid_Count"] / summary["Windows"])
        prof_name = s.sort_values(["rate", "Window_Size"]).iloc[0]["Profile"]
    if prof_name not in sheets:
        raise KeyError(f"PROFILE {prof_name!r} not in results. Available: {list(sheets)}")
    meta = summary.set_index("Profile").loc[prof_name]
    print(f"Profile shown in figures 1 and 4: {prof_name} "
          f"(window {meta['Window_s']:.2f} s, delay {meta['Delay_s']:.3f} s)")

    fig_signal_and_fasttda(df_csv, sheets[prof_name], prof_name, out_dir)
    fig_all_profiles(sheets, summary, out_dir)
    fig_grid_overview(summary, out_dir)
    fig_embedding_examples(df_csv, sheets[prof_name], int(meta["Window_Size"]),
                           int(meta["Takens_Delay"]), int(meta["Takens_Stride"]), prof_name, out_dir)


if __name__ == "__main__":
    main()
