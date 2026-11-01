"""
Figures for the spatial fidelity map.

    python plot_map.py --model siglip2 --spherical --cf-dir result/cf/1M_preview7

Writes PNGs into the map's own directory, under figures/:

  map_<pairing>.png        THE MAP -- landmarks projected to 2D, coloured by
                           the cell's mean CF, one panel per configuration
  occupancy_<pairing>.png  B queries per cell, and where the floors cut
  baselines.png            SM vs GA vs DO on held-out C (needs h5_preview.json)
  calibration.png          predicted vs observed CF on C
  h1_error_vs_distance.png SM error against distance-to-landmark and boundary gap

PROJECTION
----------
The landmarks live in 1152-d. The scatter uses the first two principal
components of the centroid set, fitted per index modality so both pairings that
share an index share a projection and can be compared panel to panel. PCA is a
linear shadow: neighbouring points on screen are genuinely close in the
embedding, but distant points may be folded together, so read these for
structure and contiguity, not for metric distance. The explained-variance
fraction is printed on each figure so the shadow's thickness is never implicit.

COLOUR
------
CF is a magnitude, so the map uses one hue light-to-dark (blue, the reference
palette's sequential default) on a fixed 0-1 domain, NOT per-panel autoscaling:
the whole point is comparing configurations against each other, and a per-panel
scale would paint a config that never exceeds CF 0.3 the same as one that never
drops below 0.9. Cells with no B queries are drawn in the surface's grey, not
in the ramp, so "no data" can never be mistaken for "low fidelity".
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")                       # no display on a compute node
import matplotlib.pyplot as plt             # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize   # noqa: E402
from matplotlib.lines import Line2D         # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from map_config import (                                               # noqa: E402
    DEFAULT_CF, DEFAULT_KMEANS, DEFAULT_OUT, DEFAULT_PILOT, PAIRINGS,
    layout, load_query_geometry, resolve_configs, resolve_pairings,
)

# Reference palette, sequential blue ramp 100->700 (see dataviz palette.md).
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
             "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
             "#0d366b"]
CF_CMAP = LinearSegmentedColormap.from_list("cf_blue", BLUE_RAMP)
NO_DATA = "#d8d7d2"                         # empty cell -- deliberately off-ramp

# Diverging pair from the same palette: blue <-> red about a neutral grey. Used
# only for correlation, which has a real zero and a sign; magnitude anywhere
# else in these figures stays on the single-hue ramp.
DIVERGING = LinearSegmentedColormap.from_list(
    "corr", ["#104281", "#2a78d6", "#9ec5f4", "#f0efec",
             "#f3a6a5", "#e34948", "#a32020"])

# Categorical slots 1-3, which are the three that validate on the all-pairs
# list; GA / DO / SM is exactly three series, so no folding is needed.
SERIES = {"GA": "#2a78d6", "DO": "#eb6834", "SM": "#1baf7a"}

# The reference palette's full categorical order. Used for the per-config line
# charts, which are line marks and so are graded on the ADJACENT pairlist, where
# all eight slots validate -- unlike the scatter, which is all-pairs and capped
# at three. Assigned in fixed catalogue order and never cycled, so a
# configuration keeps its colour across every figure.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def config_colour(cfg: str, catalogue: list) -> str:
    """Colour follows the configuration itself, keyed to its position in the
    model's full catalogue rather than in whatever subset a pairing happens to
    have. A pairing missing ivfpq_high must not repaint the other six."""
    return CATEGORICAL[catalogue.index(cfg) % len(CATEGORICAL)]

INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e1"


def style(ax):
    """Recessive axes: the marks carry the chart, not the frame."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=3)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)


def pca2(X: np.ndarray):
    """First two PCs of the centroid set, plus the variance they explain."""
    Xc = X - X.mean(axis=0)
    # SVD rather than an eigendecomposition of the covariance: 1024x1152 is
    # small, and it avoids forming a 1152x1152 matrix to throw away.
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    coords = U[:, :2] * S[:2]
    var = (S ** 2) / (S ** 2).sum()
    return coords, float(var[:2].sum())


# --------------------------------------------------------------------------
def plot_map(lay, pairing: str, figdir: Path, meta: dict) -> None:
    """The map itself: one scatter panel per configuration."""
    spec = PAIRINGS[pairing]
    stats = np.load(lay.pairing_dir(pairing) / "cell_stats.npz", allow_pickle=False)
    configs = stats["configs"].tolist()
    cents = np.load(lay.centroids(spec["index_modality"]))
    coords, evr = pca2(cents)

    n_b = stats["n"][0]                     # identical across configs by design
    ncol = 4
    nrow = int(np.ceil((len(configs) + 1) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.8 * nrow),
                             constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    norm = Normalize(0.0, 1.0)

    for ax, cfg in zip(axes, configs):
        i = configs.index(cfg)
        cf = stats["cf_mean"][i]
        seen = np.isfinite(cf)
        # Empty cells first and underneath, so a real value is never hidden.
        ax.scatter(coords[~seen, 0], coords[~seen, 1], s=6, c=NO_DATA,
                   linewidths=0, zorder=1)
        sc = ax.scatter(coords[seen, 0], coords[seen, 1],
                        c=cf[seen], cmap=CF_CMAP, norm=norm,
                        s=4 + 22 * (n_b[seen] / max(n_b.max(), 1)) ** 0.5,
                        linewidths=0.3, edgecolors="white", zorder=2)
        ax.set_title(f"{cfg}   mean CF {np.nanmean(cf):.3f}",
                     fontsize=9, color=INK, pad=6)
        style(ax)
        ax.set_xticklabels([]); ax.set_yticklabels([])

    # Key panel rather than a floating overlay, so nothing covers the data.
    # The colourbar is horizontal and sits inside the spare panel: attaching it
    # to that axis with location="left" squeezed it into a sliver.
    lax = axes[len(configs)]
    lax.axis("off")
    cax = lax.inset_axes([0.06, 0.60, 0.80, 0.07])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cb.set_label("cell mean CF@100", fontsize=8, color=INK2)
    cb.ax.tick_params(labelsize=7, colors=INK2)
    cb.outline.set_visible(False)
    lax.legend(handles=[
        Line2D([], [], marker="o", ls="", mfc=NO_DATA, mec="none", ms=6,
               label="no B queries (not low CF)"),
        Line2D([], [], marker="o", ls="", mfc="#9ec5f4", mec="white", ms=4,
               label="few B queries"),
        Line2D([], [], marker="o", ls="", mfc="#9ec5f4", mec="white", ms=9,
               label="many B queries"),
    ], loc="upper left", bbox_to_anchor=(0.02, 0.45), frameon=False,
        fontsize=8, labelcolor=INK2)
    for ax in axes[len(configs) + 1:]:
        ax.axis("off")

    fig.suptitle(
        f"Spatial fidelity map - {pairing}   "
        f"({lay.model}, K={lay.k}, {'spherical' if lay.spherical else 'L2'})\n"
        f"landmarks in PCA(2) of {spec['index_modality']} space, "
        f"{evr:.0%} of variance; marker size = B queries in cell",
        fontsize=11, color=INK)
    fig.savefig(figdir / f"map_{pairing}.png", dpi=150,
                facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(lay, pairing: str, figdir: Path, floor: int) -> None:
    """
    Three heatmap views of the same table.

    (a) LANDMARK x CONFIG. The served map drawn literally: one row per cell,
        one column per configuration, colour = mean CF. Rows are sorted by the
        cell's mean CF across configurations, which turns P2 into something you
        can see: if every column shared one ordering, the sorted image would be
        a smooth gradient in all columns at once and keying per configuration
        would be pointless. Columns that stay mottled after the sort are
        configurations that disagree with the consensus about which
        neighbourhoods are hard.

    (b) SPATIAL BINNING. The PCA scatter rebinned to a regular grid and
        averaged, which is the closest thing to a conventional heat map of the
        embedding space. Bins are B-count weighted so a cell holding 3 queries
        does not outvote one holding 300. Empty bins take the surface grey.

    (c) CONFIG x CONFIG agreement. Spearman between configurations over their
        per-cell means, restricted to reliable cells. This is H3's question
        asked directly: near-1 everywhere means one shared column would do.
    """
    stats = np.load(lay.pairing_dir(pairing) / "cell_stats.npz", allow_pickle=False)
    cfgs = stats["configs"].tolist()
    cf = stats["cf_mean"]                      # [n_configs, K]
    n_b = stats["n"][0]
    spec = PAIRINGS[pairing]

    occupied = n_b > 0
    reliable = n_b >= floor

    fig = plt.figure(figsize=(16.5, 5.4), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.15, 1.0])

    # ---- (a) landmark x config ----
    ax = fig.add_subplot(gs[0, 0])
    sub = cf[:, occupied]
    order = np.argsort(np.nanmean(sub, axis=0))[::-1]
    im = ax.imshow(sub[:, order].T, aspect="auto", cmap=CF_CMAP,
                   vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(len(cfgs)))
    ax.set_xticklabels(cfgs, rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel(f"landmark cell, sorted by mean CF  "
                  f"({int(occupied.sum())} occupied)", fontsize=8.5, color=INK2)
    ax.set_title("(a) the served map: cell x configuration",
                 fontsize=9.5, color=INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("cell mean CF@100", fontsize=8, color=INK2)
    cb.ax.tick_params(labelsize=7, colors=INK2)
    cb.outline.set_visible(False)

    # ---- (b) spatial binning ----
    ax = fig.add_subplot(gs[0, 1])
    coords, evr = pca2(np.load(lay.centroids(spec["index_modality"])))
    mean_cf = np.nanmean(cf, axis=0)
    ok = occupied & np.isfinite(mean_cf)
    # 1024 landmarks spread over a wedge: a fine grid leaves most bins empty and
    # reads as confetti rather than a field. 22 bins puts a few landmarks in
    # each occupied bin, which is what makes the average mean anything.
    nb = 22
    xe = np.linspace(coords[:, 0].min(), coords[:, 0].max(), nb + 1)
    ye = np.linspace(coords[:, 1].min(), coords[:, 1].max(), nb + 1)
    w = n_b[ok].astype(float)
    num, _, _ = np.histogram2d(coords[ok, 0], coords[ok, 1], [xe, ye],
                               weights=w * mean_cf[ok])
    den, _, _ = np.histogram2d(coords[ok, 0], coords[ok, 1], [xe, ye], weights=w)
    with np.errstate(invalid="ignore", divide="ignore"):
        grid = np.where(den > 0, num / den, np.nan)
    cmap = CF_CMAP.copy()
    cmap.set_bad(NO_DATA)
    im = ax.imshow(grid.T, origin="lower", cmap=cmap, vmin=0, vmax=1,
                   extent=[xe[0], xe[-1], ye[0], ye[-1]],
                   aspect="auto", interpolation="nearest")
    ax.set_title(f"(b) mean CF over the embedding, PCA(2) ({evr:.0%} var)",
                 fontsize=9.5, color=INK)
    ax.set_xticklabels([]); ax.set_yticklabels([])
    style(ax)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("mean CF@100 across configs", fontsize=8, color=INK2)
    cb.ax.tick_params(labelsize=7, colors=INK2)
    cb.outline.set_visible(False)

    # ---- (c) config x config agreement ----
    ax = fig.add_subplot(gs[0, 2])
    m = len(cfgs)
    R = np.eye(m)
    for i in range(m):
        for j in range(i + 1, m):
            good = reliable & np.isfinite(cf[i]) & np.isfinite(cf[j])
            if good.sum() > 2:
                R[i, j] = R[j, i] = spearmanr(cf[i][good], cf[j][good]).statistic
    # Correlation carries polarity, so this is the one diverging scale in the
    # figure: blue<->red about a neutral grey zero, on a fixed -1..1 domain so
    # "they all agree" cannot be manufactured by autoscaling.
    im = ax.imshow(R, cmap=DIVERGING, vmin=-1, vmax=1, interpolation="nearest")
    ax.set_xticks(range(m)); ax.set_yticks(range(m))
    ax.set_xticklabels(cfgs, rotation=45, ha="right", fontsize=7.5)
    ax.set_yticklabels(cfgs, fontsize=7.5)
    # Three decimals off the diagonal: ivfpq_low and oivfpq_mod share every IVF
    # and PQ parameter and differ only by the OPQ rotation, so they land close
    # enough that ".2f" prints 1.00 and invites the reader to conclude the two
    # columns are duplicates. They are not -- only 19% of their per-query CF
    # values agree.
    for i in range(m):
        for j in range(m):
            txt = "1" if i == j else f"{R[i, j]:.3f}".lstrip("0")
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.2,
                    color="white" if abs(R[i, j]) > 0.55 else INK)
    off = R[~np.eye(m, dtype=bool)]
    ax.set_title(f"(c) do configs agree? Spearman, reliable cells\n"
                 f"off-diagonal mean {off.mean():.2f}, min {off.min():.2f}",
                 fontsize=9.5, color=INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.ax.tick_params(labelsize=7, colors=INK2)
    cb.outline.set_visible(False)

    fig.suptitle(f"Spatial fidelity map, heatmap views - {pairing}   "
                 f"({lay.model}, K={lay.k}, floor {floor})",
                 fontsize=11.5, color=INK)
    fig.savefig(figdir / f"heatmap_{pairing}.png", dpi=150,
                facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_occupancy(lay, pairing: str, figdir: Path, meta: dict) -> None:
    """Where the reliability floors actually cut."""
    stats = np.load(lay.pairing_dir(pairing) / "cell_stats.npz", allow_pickle=False)
    n_b = stats["n"][0]
    floors = [int(f) for f in stats["floors"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.6), constrained_layout=True)

    occupied = n_b[n_b > 0]
    ax1.hist(occupied, bins=np.logspace(0, np.log10(max(occupied.max(), 10)), 40),
             color="#2a78d6", edgecolor="white", linewidth=0.4)
    ax1.set_xscale("log")
    for f, c in zip(floors, ["#eb6834", "#e34948", "#4a3aa7"]):
        ax1.axvline(f, color=c, lw=1.6, ls="--",
                    label=f"floor {f} ({int((n_b < f).sum())} cells below)")
    ax1.set_xlabel("B queries in cell (log)", fontsize=9, color=INK2)
    ax1.set_ylabel("cells", fontsize=9, color=INK2)
    ax1.set_title(f"Cell occupancy - {pairing}", fontsize=10, color=INK)
    ax1.legend(frameon=False, fontsize=7.5, labelcolor=INK2)
    style(ax1)

    # The deployment-relevant view: not how many cells are thin, but how much
    # of B they actually hold. These answer differently, which is the point.
    order = np.sort(occupied)
    ax2.plot(np.arange(len(order)) / len(order),
             np.cumsum(order) / order.sum(), color="#2a78d6", lw=2)
    ax2.set_xlabel("fraction of occupied cells (sparsest first)",
                   fontsize=9, color=INK2)
    ax2.set_ylabel("fraction of B covered", fontsize=9, color=INK2)
    ax2.set_title("The thin cells hold little of B", fontsize=10, color=INK)
    ax2.axvline(0.10, color=INK2, lw=1, ls=":")
    ax2.annotate("sparsest decile", xy=(0.10, 0.02), xytext=(0.14, 0.10),
                 fontsize=7.5, color=INK2)
    style(ax2)

    fig.savefig(figdir / f"occupancy_{pairing}.png", dpi=150,
                facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_baselines(lay, figdir: Path, h5: dict) -> None:
    """SM vs GA vs DO, MAE on held-out C, grouped by pairing."""
    pairings = [p for p, v in h5["pairings"].items() if v]
    fig, axes = plt.subplots(1, len(pairings), figsize=(4.1 * len(pairings), 4.0),
                             constrained_layout=True, sharey=True)
    axes = np.atleast_1d(axes)

    for ax, p in zip(axes, pairings):
        pc = h5["pairings"][p]
        cfgs = list(pc)
        x = np.arange(len(cfgs))
        w = 0.26
        for j, (name, colour) in enumerate(SERIES.items()):
            vals = [pc[c][name]["MAE"] for c in cfgs]
            # 2px surface gap between adjacent bars: drawn as a white edge.
            ax.bar(x + (j - 1) * w, vals, w * 0.92, label=name, color=colour,
                   edgecolor="white", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(cfgs, rotation=45, ha="right", fontsize=7.5)
        ax.set_title(p.replace("_query_", " q -> ").replace("_index", " idx"),
                     fontsize=9, color=INK)
        style(ax)
    axes[0].set_ylabel("MAE vs true CF@100 on C  (lower is better)",
                       fontsize=9, color=INK2)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK2)

    fig.suptitle(f"Held-out prediction error - {lay.model}, K={lay.k}   "
                 f"(SM = the map, GA = one number per config, DO = density only)",
                 fontsize=11, color=INK)
    fig.savefig(figdir / "baselines.png", dpi=150,
                facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_calibration(lay, figdir: Path, h5: dict, floor: int,
                     catalogue: list) -> None:
    """Predicted vs observed on C -- S5.7 asks for this, not only correlation."""
    pairings = [p for p, v in h5["pairings"].items() if v]
    fig, axes = plt.subplots(1, len(pairings), figsize=(3.6 * len(pairings), 3.7),
                             constrained_layout=True, sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    edges = np.linspace(0, 1, 11)
    mid = (edges[:-1] + edges[1:]) / 2

    for ax, p in zip(axes, pairings):
        stats = np.load(lay.pairing_dir(p) / "cell_stats.npz", allow_pickle=False)
        cfgs = stats["configs"].tolist()
        gc = load_query_geometry(lay, p, "C")
        ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls="--", zorder=1)

        for i, cfg in enumerate(cfgs):
            f = lay.cf_vector(p, "C", cfg)
            # h5_preview.json can be stale relative to cell_stats.npz -- e.g.
            # built before T8 finished the last configs -- so a config present
            # in the map is not guaranteed to be present in the H5 comparison.
            # Skip it here rather than crash; compare_baselines.py is what
            # regenerates h5_preview.json against the current CF directory.
            if not f.exists() or cfg not in h5["pairings"].get(p, {}):
                continue
            true = np.load(f).astype(np.float64)
            ac = gc["assign"][:len(true)]
            pred = stats["cf_mean"][i][ac].astype(np.float64)
            ga = h5["pairings"][p][cfg]["ga"]
            bad = (stats["n"][i][ac] < floor) | ~np.isfinite(pred)
            pred = np.where(bad, ga, pred)

            idx = np.clip(np.digitize(pred, edges) - 1, 0, 9)
            obs = np.array([true[idx == b].mean() if (idx == b).any() else np.nan
                            for b in range(10)])
            ax.plot(mid, obs, lw=2, alpha=0.9, label=cfg,
                    color=config_colour(cfg, catalogue))

        ax.set_title(p.replace("_query_", " q -> ").replace("_index", " idx"),
                     fontsize=9, color=INK)
        ax.set_xlabel("predicted CF", fontsize=9, color=INK2)
        style(ax)
    axes[0].set_ylabel("observed CF on C", fontsize=9, color=INK2)
    # Identity is never colour-alone: a legend is always present for >=2 series.
    axes[-1].legend(frameon=False, fontsize=7, labelcolor=INK2,
                    loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle(f"Calibration - {lay.model}, K={lay.k}   "
                 f"(one line per configuration; dashed = perfect)",
                 fontsize=11, color=INK)
    fig.savefig(figdir / "calibration.png", dpi=150,
                facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_h1(lay, figdir: Path, h5: dict, floor: int, catalogue: list) -> None:
    """
    H1's diagnostic: does SM's error grow with distance-to-landmark, and with
    boundary proximity? P1 predicts yes to both.
    """
    pairings = [p for p, v in h5["pairings"].items() if v]
    fig, axes = plt.subplots(2, len(pairings), figsize=(3.6 * len(pairings), 6.4),
                             constrained_layout=True)
    axes = np.atleast_2d(axes)

    for col, p in enumerate(pairings):
        stats = np.load(lay.pairing_dir(p) / "cell_stats.npz", allow_pickle=False)
        cfgs = stats["configs"].tolist()
        gc = load_query_geometry(lay, p, "C")

        for row, (key, label) in enumerate(
                [("d1", "distance to nearest landmark"),
                 ("gap", "boundary gap  (d2 - d1; small = near a wall)")]):
            ax = axes[row, col]
            for i, cfg in enumerate(cfgs):
                f = lay.cf_vector(p, "C", cfg)
                if not f.exists() or cfg not in h5["pairings"].get(p, {}):
                    continue
                true = np.load(f).astype(np.float64)
                m = len(true)
                ac = gc["assign"][:m]
                pred = stats["cf_mean"][i][ac].astype(np.float64)
                ga = h5["pairings"][p][cfg]["ga"]
                bad = (stats["n"][i][ac] < floor) | ~np.isfinite(pred)
                pred = np.where(bad, ga, pred)
                err = np.abs(pred - true)

                v = gc[key][:m]
                # Deciles of the variable, so each point carries equal support
                # -- fixed-width bins would put most of C in one bar.
                qs = np.quantile(v, np.linspace(0, 1, 11))
                qs[-1] += 1e-9
                b = np.clip(np.digitize(v, qs) - 1, 0, 9)
                ax.plot([v[b == k].mean() for k in range(10)],
                        [err[b == k].mean() for k in range(10)],
                        lw=2, alpha=0.9, label=cfg if (row == 0 and col == 0) else None,
                        color=config_colour(cfg, catalogue))
            if row == 0:
                ax.set_title(p.replace("_query_", " q -> ").replace("_index", " idx"),
                             fontsize=9, color=INK)
            ax.set_xlabel(label, fontsize=8, color=INK2)
            if col == 0:
                ax.set_ylabel("mean |SM error| on C", fontsize=9, color=INK2)
            style(ax)

    axes[0, 0].legend(frameon=False, fontsize=7, labelcolor=INK2, ncol=2,
                      loc="upper left")
    fig.suptitle(f"H1 diagnostic - {lay.model}, K={lay.k}   "
                 f"(P1 predicts error rises with distance, and falls with gap)",
                 fontsize=11, color=INK)
    fig.savefig(figdir / "h1_error_vs_distance.png", dpi=150,
                facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="siglip2")
    ap.add_argument("--k", type=int, default=1024)
    ap.add_argument("--spherical", action="store_true")
    ap.add_argument("--pairing", nargs="+", default=["all"])
    ap.add_argument("--pilot-dir", type=Path, default=DEFAULT_PILOT)
    ap.add_argument("--kmeans-dir", type=Path, default=DEFAULT_KMEANS)
    ap.add_argument("--cf-dir", type=Path, default=DEFAULT_CF)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    lay = layout(args.model, args.k, args.spherical, pilot=args.pilot_dir,
                 kmeans=args.kmeans_dir, cf=args.cf_dir, out=args.out_dir)
    meta = json.load(open(lay.dest / "map_meta.json"))
    floor = int(meta["reliability_floor"])
    pairings = [p for p in resolve_pairings(args.pairing) if p in meta["pairings"]]

    figdir = lay.dest / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    print(f"figures -> {figdir}\n")

    for p in pairings:
        plot_map(lay, p, figdir, meta)
        plot_heatmap(lay, p, figdir, floor)
        plot_occupancy(lay, p, figdir, meta)
        print(f"  map_{p}.png, heatmap_{p}.png, occupancy_{p}.png")

    h5_path = lay.dest / "h5_preview.json"
    if h5_path.exists():
        h5 = json.load(open(h5_path))
        catalogue = resolve_configs(args.model, "all")
        plot_baselines(lay, figdir, h5)
        plot_calibration(lay, figdir, h5, floor, catalogue)
        plot_h1(lay, figdir, h5, floor, catalogue)
        print("  baselines.png, calibration.png, h1_error_vs_distance.png")
    else:
        print("  (no h5_preview.json -- run compare_baselines.py for the "
              "baseline, calibration and H1 figures)")


if __name__ == "__main__":
    main()
