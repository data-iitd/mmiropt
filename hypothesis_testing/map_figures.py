"""
The map, drawn three ways: what it says, where it is wrong, and where it lies.

T9 already draws the first of these -- landmarks projected to 2D, one panel per
configuration, coloured by the cell's mean CF. That figure answers "where does
this index degrade". It is copied into T10's tree by ``import_t9_artifacts.py``
rather than redrawn.

This module adds the two T10 can draw and T9 could not, because both need the
held-out labels:

  error   mean |prediction - truth| over the C queries in each cell
  bias    mean (prediction - truth), signed

HOW EACH IS COMPUTED
--------------------
For configuration c and cell l, let Q_l be the C queries assigned to l. Every
one of them receives the SAME prediction -- m, the cell's mean CF over B, or GA
if the cell fell under the reliability floor. Then

    error[l] = mean over q in Q_l of | m - CF(q) |
    bias[l]  = mean over q in Q_l of ( m - CF(q) )

Because the prediction is constant within a cell, error[l] is exactly the mean
absolute deviation of the held-out truth around the number the map serves --
which is the quantity premise P1 claims should be small. It is not a residual
from a fitted line; there is no line.

WHY BIAS IS PLOTTED SEPARATELY
------------------------------
|error| cannot distinguish a cell where the map promises 0.85 and delivers 0.70
from one where it promises 0.70 and delivers 0.85. Those are not equally bad. A
fidelity-floor policy consumes the prediction as a promise: a cell with
POSITIVE bias over-promises, so the policy clears tau on paper, runs a
configuration that is too cheap, and silently returns worse results than it
believes. Under-promising merely wastes latency. Bias is therefore drawn on a
diverging scale centred at zero, red for the dangerous direction, and it is the
panel to read before trusting the map anywhere.

TWO PROJECTIONS, AND WHY BOTH ARE DRAWN
---------------------------------------
The landmarks live in 1152 dimensions and every 2-D view of them is a lie. The
two available lie about different things, so both are written and neither is
authoritative:

  pca    A linear shadow. Distances and DENSITY survive -- a crowded patch is
         genuinely crowded. What it gets wrong is folding: two regions far apart
         in 1152-d can land on top of each other. It reports how much it hid
         (the explained-variance fraction on every figure), which is 15% in
         SigLIP 2's image space and 34% in its text space. Default, and the one
         T9's CF maps use, so the two tasks' figures can be laid side by side.

  tsne   A rearrangement. NEIGHBOURHOODS survive and nothing else does:
         distances between distant points, blob sizes, gaps between blobs, and
         above all DENSITY are meaningless, because t-SNE equalises density by
         construction. That last point is why it is not the default here --
         density is the independent variable of this entire study (DO predicts
         fidelity from it, P1 is a claim about sparse regions), and a view that
         paints sparse and dense identically invites exactly the wrong reading.
         It is the better view for the narrow question "is the error
         concentrated or scattered", and it is stochastic, so the layout is
         cached and seeded to keep re-runs comparable.

Neither is evidence. The evidence for "is this quantity spatially structured"
is Moran's I above, computed on the k-NN graph in the full 1152 dimensions with
no projection involved at all. The pictures illustrate; the statistic decides.

Cells that no C query reached are drawn in the surface grey rather than in the
colour ramp, so "not graded" can never be misread as "graded and fine".
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "spatialmap"))

from ht_config import (PAIRINGS, add_common_args, load_block,               # noqa: E402
                       resolve_from_args, write_csv, write_json)
from predictors import sm_on_c                                             # noqa: E402
# T9's projection and palette, imported so the two tasks' figures line up.
from plot_map import CF_CMAP, DIVERGING, INK, INK2, NO_DATA, pca2          # noqa: E402


def knn_graph(centroids: np.ndarray, k: int = 10):
    """Row-normalised k-nearest-neighbour weights over the landmarks, built in
    the FULL embedding space -- never in a projection.

    This is the point of the whole exercise: Moran's I asks whether neighbouring
    cells have similar values, and "neighbouring" must mean neighbouring in the
    1152 dimensions the corpus actually lives in. Deriving neighbours from the
    2-D picture would measure the projection, not the data.
    """
    import faiss

    x = np.ascontiguousarray(centroids.astype(np.float32))
    index = faiss.IndexFlatL2(x.shape[1])
    index.add(x)
    _, idx = index.search(x, k + 1)
    return idx[:, 1:]                     # drop self, which is always column 0


def morans_i(values: np.ndarray, neighbours: np.ndarray, n_perm: int = 999,
             seed: int = 0) -> dict:
    """Spatial autocorrelation of a per-cell quantity, with a permutation test.

    Moran's I answers, in one number and with no projection involved, the
    question both map figures are being asked to answer by eye:

        I ~ 0    the quantity is spatially random across the tiling
        I > 0    neighbouring cells resemble each other -- it is CLUSTERED
        I < 0    neighbouring cells differ -- a chequerboard, which for CF or
                 error would mean the tiling is cutting through structure

    Applied to CF it tests P1's premise directly: fidelity must vary smoothly
    for a cell average to represent its neighbourhood at all. Applied to the
    map's ERROR it tests something the aggregate metrics cannot see -- whether
    what the map gets wrong is still spatially organised. Structured residual
    error means signal the tiling failed to capture, and points at H4 (finer K)
    rather than at H0 being false. Unstructured residual means the spatial
    signal has been extracted and what remains is within-cell noise, which is
    the ceiling H0's ICC already quantified.

    Significance comes from a permutation test rather than the analytic
    normality formula: cell values here are means over wildly unequal sample
    sizes (T9 measured cells holding 2 B queries next to cells holding 400), so
    the assumptions behind the closed-form variance do not hold.
    """
    v = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(v)
    if ok.sum() < 30:
        return {"I": float("nan"), "n_cells": int(ok.sum()),
                "note": "too few graded cells"}

    # Restrict the graph to graded cells. An ungraded neighbour is dropped from
    # the row rather than treated as a zero, which would pull every value near
    # an empty region toward the mean and manufacture negative autocorrelation.
    def restrict(sub: np.ndarray):
        pos = -np.ones(len(v), dtype=np.int64)
        pos[sub] = np.arange(len(sub))
        nb = pos[neighbours[sub]]
        return nb, nb >= 0

    sub = np.nonzero(ok)[0]
    nb, valid = restrict(sub)
    sub = sub[valid.any(axis=1)]          # drop cells left with no neighbour
    nb, valid = restrict(sub)             # renumber against the kept set
    x = v[sub]
    n = len(x)
    if n < 30:
        return {"I": float("nan"), "n_cells": n, "note": "too few after pruning"}

    w = valid.astype(np.float64)
    w /= np.maximum(w.sum(axis=1, keepdims=True), 1.0)   # row-normalised

    def compute(z: np.ndarray) -> float:
        denom = float((z ** 2).sum())
        if denom <= 0:
            return float("nan")
        lag = np.einsum("ij,ij->i", w, np.where(valid, z[np.clip(nb, 0, None)], 0.0))
        return float(n / w.sum() * float((z * lag).sum()) / denom)

    z = x - x.mean()
    obs = compute(z)

    rng = np.random.default_rng(seed)
    null = np.array([compute(rng.permutation(z)) for _ in range(n_perm)])
    # one-sided: we are asking whether it is MORE clustered than chance
    p = float((1 + np.sum(null >= obs)) / (n_perm + 1))
    return {
        "I": obs,
        "expected_under_null": float(-1.0 / (n - 1)),
        "permutation_mean": float(null.mean()),
        "permutation_std": float(null.std()),
        "z_score": float((obs - null.mean()) / null.std()) if null.std() > 0 else float("nan"),
        "p_value": p,
        "n_cells": n, "k_neighbours": int(neighbours.shape[1]),
        "clustered": bool(p < 0.01 and obs > 0),
    }


def project(centroids: np.ndarray, method: str, cache: Path, seed: int = 0):
    """2-D layout of the landmarks, cached so every panel and every re-run of
    this script share one picture.

    Caching is not just for speed -- t-SNE is stochastic, so without a cached
    layout two figures of the same pairing would not be comparable, and a
    re-run would silently move every point.
    """
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        z = np.load(cache)
        return z["coords"], float(z["evr"])

    if method == "pca":
        coords, evr = pca2(centroids)
    elif method == "tsne":
        from sklearn.manifold import TSNE
        # cosine, because T5 fitted the tiling spherically and the centroids are
        # unit-norm; Euclidean on unit vectors is a monotone function of cosine
        # but the perplexity calibration is cleaner in the metric the space uses.
        coords = TSNE(n_components=2, metric="cosine", init="pca",
                      perplexity=30.0, random_state=seed,
                      ).fit_transform(centroids.astype(np.float64))
        # There is no variance-explained for t-SNE. Reported as nan rather than
        # omitted, so the absence is visible on the figure instead of implied.
        evr = float("nan")
    else:
        raise ValueError(f"unknown projection {method!r}")
    np.savez(cache, coords=coords, evr=evr)
    return coords, evr


def per_cell_grades(block, config: str, floor: int, K: int):
    """error, bias, and the C-query count, per cell.

    Returns arrays of length K with nan where no C query landed, so the plot
    can grey those cells out instead of colouring them as if they were graded.
    """
    pred, _ = sm_on_c(block, config, floor)
    true = block.cf_c[config]
    assign = block.geom_c["assign"]
    resid = pred - true

    n = np.bincount(assign, minlength=K).astype(np.float64)
    s_abs = np.bincount(assign, weights=np.abs(resid), minlength=K)
    s_sgn = np.bincount(assign, weights=resid, minlength=K)

    error = np.full(K, np.nan)
    bias = np.full(K, np.nan)
    seen = n > 0
    error[seen] = s_abs[seen] / n[seen]
    bias[seen] = s_sgn[seen] / n[seen]
    return error, bias, n


def _panel_grid(n_panels: int, ncol: int = 4):
    import matplotlib.pyplot as plt
    nrow = int(np.ceil(n_panels / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.8 * nrow),
                             constrained_layout=True)
    return fig, np.atleast_1d(axes).ravel()


def plot_quantity(block, coords, evr, values, counts, kind: str, path: Path,
                  floor: int, scale: str = "panel", method: str = "pca") -> None:
    """One scatter panel per configuration, coloured by ``values[cfg]``.

    SCALING -- the choice that decides whether this figure shows anything
    ------------------------------------------------------------------
    ``scale="shared"`` puts every panel on one colour range. That makes
    configurations comparable to each other, and it is what a CF map wants.
    It is the WRONG default here: cell error runs from 0.142 for hnsw_low down
    to 0.017 for hnsw_high in the same pairing, so a shared 0-0.14 ramp renders
    the accurate configurations as a uniform pale wash with no visible
    structure at all. Cross-configuration comparison is already available, and
    more precisely, from ``h1_per_cell_grades.csv`` and the printed means.

    ``scale="panel"`` (the default) gives each panel its own range, stretched
    to the 2nd-98th percentile of that configuration's own cells so a couple of
    outliers cannot flatten the rest. What the picture is for is WHERE inside a
    given configuration the map fails, and that only shows up when the range
    matches the data.

    The honesty cost of per-panel scaling is that identical colours in two
    panels mean different numbers. It is paid two ways: every panel gets its
    OWN colourbar rather than one shared bar that would be a lie, and each
    panel title prints the range its colours actually span.

    COLOUR
    ------
    A perceptually uniform ramp (viridis for error, magnitude only) rather than
    a single-hue light-to-dark, because the errors bunch tightly and a
    light-to-dark blue spends most of its visible contrast on a range the data
    does not occupy. Bias keeps the diverging map centred on zero, since its
    sign is the whole point.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, TwoSlopeNorm

    configs = block.configs
    fig, axes = _panel_grid(len(configs))
    cmap = plt.get_cmap("viridis") if kind == "error" else DIVERGING
    label = ("mean |pred - truth|" if kind == "error"
             else "mean (pred - truth)")

    def limits(v: np.ndarray):
        """Colour range for one panel (or, for shared, for everything)."""
        finite = v[np.isfinite(v)]
        if finite.size == 0:
            return (0.0, 1.0)
        if kind == "error":
            lo, hi = np.percentile(finite, [2, 98])
            return (float(lo), float(max(hi, lo + 1e-4)))
        lim = float(np.percentile(np.abs(finite), 98))
        return (-max(lim, 1e-4), max(lim, 1e-4))

    if scale == "shared":
        pooled = np.concatenate([values[c] for c in configs])
        shared_lo, shared_hi = limits(pooled)

    size = 4 + 26 * (counts / max(counts.max(), 1)) ** 0.5
    for ax, cfg in zip(axes, configs):
        v = values[cfg]
        seen = np.isfinite(v)
        lo, hi = (shared_lo, shared_hi) if scale == "shared" else limits(v)
        norm = (Normalize(lo, hi) if kind == "error"
                else TwoSlopeNorm(0.0, lo, hi))

        ax.scatter(coords[~seen, 0], coords[~seen, 1], s=6, c=NO_DATA,
                   linewidths=0, zorder=1)
        # Draw order is not cosmetic here. 1,024 markers over a 2-D shadow
        # overlap heavily, and whatever is drawn last is what the reader sees.
        # Sorting by magnitude puts the WORST cells on top, so a bad cell can
        # never be hidden underneath a good one -- the opposite convention
        # would quietly flatter the map.
        idx = np.nonzero(seen)[0]
        idx = idx[np.argsort(np.abs(v[idx]) if kind == "bias" else v[idx])]
        sc = ax.scatter(coords[idx, 0], coords[idx, 1], c=v[idx],
                        cmap=cmap, norm=norm, s=size[idx],
                        linewidths=0.25, edgecolors="white", zorder=2)

        stat = (f"mean {np.nanmean(v):.3f}" if kind == "error"
                else f"mean {np.nanmean(v):+.4f}, "
                     f"{100 * np.nanmean(v > 0):.0f}% over")
        ax.set_title(f"{cfg}   {stat}\nscale {lo:+.3f} to {hi:+.3f}",
                     fontsize=8.5, color=INK, pad=5)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        if scale == "panel":
            cb = fig.colorbar(sc, ax=ax, shrink=0.72, pad=0.02)
            cb.ax.tick_params(labelsize=6, colors=INK2)
            cb.outline.set_visible(False)

    for ax in axes[len(configs):]:
        ax.axis("off")
    if scale == "shared":
        cb = fig.colorbar(sc, ax=axes[:len(configs)].tolist(), label=label,
                          shrink=0.6)
        cb.outline.set_visible(False)

    what = ("where the map is WRONG" if kind == "error"
            else "where the map LIES, and in which direction "
                 "(red = OVER-promises)")
    note = ("each panel on its OWN colour scale, printed above it -- the same "
            "colour in two panels is NOT the same number"
            if scale == "panel" else
            "one colour scale across all panels")
    if method == "pca":
        proj = (f"landmarks in PCA(2) of {block.index_modality} space, "
                f"{evr:.0%} of variance retained -- distances and DENSITY are "
                f"meaningful, but distant points may be folded together")
    else:
        proj = (f"landmarks in t-SNE(2) of {block.index_modality} space -- "
                f"NEIGHBOURHOODS are meaningful; distances, blob sizes and "
                f"APPARENT DENSITY are NOT (t-SNE equalises density by "
                f"construction, so this view cannot be read for sparse vs dense)")
    fig.suptitle(
        f"{what} - {block.pairing}   ({block.model}, K={block.K}, floor {floor})\n"
        f"{label} per cell; {note}\n"
        f"{proj}; marker size = C queries in cell; grey = no C query landed",
        fontsize=10.5, color=INK)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def run(args) -> dict:
    lay, ht, floor, pairings, configs, meta = resolve_from_args(args)
    dest = ht.ensure("H1")           # these are H1's spatial companion
    figdir = dest / "figures"

    report = {
        "created": datetime.now().isoformat(),
        "artifact": "per-cell grades of the served map on held-out C",
        "model": lay.model, "tiling": lay.variant, "floor": floor,
        "error": "mean |prediction - truth| over the C queries in the cell",
        "bias": "mean (prediction - truth); POSITIVE = the map over-promises, "
                "which is the direction a fidelity-floor policy is harmed by",
        "projection": "first two principal components of the centroid set; "
                      "read for structure, never for metric distance",
        "morans_i": "spatial autocorrelation of each per-cell quantity over a "
                    "k-NN landmark graph built in the FULL embedding space, "
                    "never in a projection; permutation-tested. I>0 = "
                    "neighbouring cells resemble each other. On CF it tests "
                    "P1's smoothness premise directly; on the map's ERROR it "
                    "tests whether structure the tiling failed to capture "
                    "remains, which points at H4 rather than at H0 being false",
        "colour_scale": "per-panel, 2nd-98th percentile, each panel carrying "
                        "its own colourbar and printed range; the "
                        "*_sharedscale.png variant puts every panel on one "
                        "range instead, which is comparable but washes out the "
                        "low-error configurations",
        "pairings": {},
    }
    rows = []

    methods = list(getattr(args, "projection", ["pca", "tsne"]))
    scale = getattr(args, "scale", "panel")
    knn_k = int(getattr(args, "knn", 10))
    n_perm = int(getattr(args, "perm", 999))
    seed = int(getattr(args, "seed", 0))
    moran_rows = []

    for pairing in pairings:
        block = load_block(lay, pairing, configs, floor)
        if not block.configs:
            continue
        cents = np.load(lay.centroids(block.index_modality))

        errors, biases, counts = {}, {}, None
        for cfg in block.configs:
            e, b, n = per_cell_grades(block, cfg, floor, block.K)
            errors[cfg], biases[cfg] = e, b
            counts = n                       # identical across configurations
            for l in np.nonzero(n > 0)[0]:
                rows.append([pairing, cfg, int(l), int(n[l]),
                             f"{block.cell_mean[block.cfg_row(cfg), l]:.5f}",
                             f"{int(block.cell_n[block.cfg_row(cfg), l])}",
                             f"{e[l]:.5f}", f"{b[l]:.5f}"])

        # ---- the number, computed in the real space, no projection ---------
        nb = knn_graph(cents, k=knn_k)
        moran = {}
        print(f"\n  {pairing}   Moran's I over the {knn_k}-NN landmark graph "
              f"in full {cents.shape[1]}-d space")
        print(f"    {'config':<12} {'I(CF)':>8} {'I(error)':>9} {'I(bias)':>8}   "
              f"reading")
        for cfg in block.configs:
            r = block.cfg_row(cfg)
            cf_cells = np.where(block.cell_n[r] > 0, block.cell_mean[r], np.nan)
            m_cf = morans_i(cf_cells, nb, n_perm, seed)
            m_err = morans_i(errors[cfg], nb, n_perm, seed)
            m_bias = morans_i(biases[cfg], nb, n_perm, seed)
            moran[cfg] = {"cf": m_cf, "error": m_err, "bias": m_bias}
            # The residual is the interesting one: structure left in the error
            # is spatial signal the tiling did not capture.
            reading = ("residual error still clustered -- signal the tiling "
                       "missed (see H4)" if m_err.get("clustered")
                       else "residual error spatially random -- the spatial "
                            "signal has been extracted")
            moran_rows.append([
                pairing, cfg, block.families[cfg], knn_k,
                f"{m_cf.get('I', float('nan')):.4f}",
                f"{m_cf.get('p_value', float('nan')):.4f}",
                f"{m_err.get('I', float('nan')):.4f}",
                f"{m_err.get('p_value', float('nan')):.4f}",
                f"{m_bias.get('I', float('nan')):.4f}",
                f"{m_bias.get('p_value', float('nan')):.4f}",
                "yes" if m_err.get("clustered") else "no"])
            print(f"    {cfg:<12} {m_cf.get('I', float('nan')):>8.4f} "
                  f"{m_err.get('I', float('nan')):>9.4f} "
                  f"{m_bias.get('I', float('nan')):>8.4f}   {reading}")

        # ---- the pictures, one set per projection ---------------------------
        projections = {}
        for method in methods:
            coords, evr = project(
                cents, method,
                dest / "projections" / f"{block.index_modality}_{method}.npz",
                seed=seed)
            projections[method] = float(evr)
            plot_quantity(block, coords, evr, errors, counts, "error",
                          figdir / f"map_error_{pairing}_{method}.png",
                          floor, scale, method)
            plot_quantity(block, coords, evr, biases, counts, "bias",
                          figdir / f"map_bias_{pairing}_{method}.png",
                          floor, scale, method)
            if scale == "panel":
                plot_quantity(block, coords, evr, errors, counts, "error",
                              figdir / f"map_error_{pairing}_{method}"
                                       f"_sharedscale.png",
                              floor, "shared", method)

        over = {c: float(np.nanmean(biases[c] > 0)) for c in block.configs}
        report["pairings"][pairing] = {
            "K": block.K, "n_c": block.n_c,
            "projections": projections,
            "cells_with_no_c_query": int((counts == 0).sum()),
            "mean_cell_error": {c: float(np.nanmean(errors[c]))
                                for c in block.configs},
            "mean_cell_bias": {c: float(np.nanmean(biases[c]))
                               for c in block.configs},
            "fraction_of_cells_over_promising": over,
            "morans_i": moran,
            "n_configs_with_clustered_residual_error": sum(
                1 for c in moran if moran[c]["error"].get("clustered")),
        }
        for cfg in block.configs:
            print(f"    {cfg:<12} cell error {np.nanmean(errors[cfg]):.4f}  "
                  f"cell bias {np.nanmean(biases[cfg]):+.4f}  "
                  f"over-promising in {over[cfg]:.0%} of cells")

    write_csv(dest / "h1_per_cell_grades.csv",
              ["pairing", "config", "landmark", "n_c_queries", "prediction",
               "n_b_queries", "mean_abs_error", "mean_bias"], rows)
    write_csv(dest / "h1_spatial_autocorrelation.csv",
              ["pairing", "config", "family", "k_neighbours", "I_cf", "p_cf",
               "I_error", "p_error", "I_bias", "p_bias",
               "residual_error_clustered"], moran_rows)
    write_json(dest / "h1_map_figures.json", report)
    print(f"\n  wrote {dest / 'h1_spatial_autocorrelation.csv'}")
    print(f"  wrote {figdir}/map_{{error,bias}}_<pairing>_{{{','.join(methods)}}}.png")
    return report


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--projection", nargs="+", default=["pca", "tsne"],
                    choices=("pca", "tsne"),
                    help="2-D layouts to draw; both by default. PCA preserves "
                         "density and reports what it hides; t-SNE preserves "
                         "neighbourhoods and destroys density")
    ap.add_argument("--knn", type=int, default=10,
                    help="neighbours per landmark for Moran's I, in FULL "
                         "embedding space")
    ap.add_argument("--perm", type=int, default=999,
                    help="permutations for Moran's I significance")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", choices=("panel", "shared"), default="panel",
                    help="panel: each configuration gets its own colour range, "
                         "stretched 2nd-98th percentile, with its own colourbar "
                         "-- shows structure. shared: one range for all panels "
                         "-- comparable but washes out the accurate configs")
    args = ap.parse_args()
    print(f"map figures  model={args.model}  error and bias per cell "
          f"(colour scale: {args.scale}, projections: {args.projection})")
    run(args)


if __name__ == "__main__":
    main()
