"""
H4 -- landmark resolution sweep.

    Vary K (the number of landmarks) from coarse to fine, and report the
    resulting accuracy, storage, and lookup-cost trade-off. Accuracy should
    improve with finer tiling up to the point where individual landmarks no
    longer have enough nearby sampled queries to estimate fidelity reliably,
    after which per-cell variance should increase and accuracy should degrade
    or plateau.                                                  (draft S5.5)

THE TWO FORCES, AND WHY THE ANSWER IS A U
-----------------------------------------
Finer tiling shrinks the IRREDUCIBLE error: each cell describes a smaller, more
homogeneous neighbourhood, so its mean fits its members better. H0's ICC rises
with K for that reason. But the same fixed pool of B queries is being sliced
thinner, so each mean is built from fewer samples and the ESTIMATION error
grows. Total error is the sum, and it is U-shaped. Locating the bottom of that
U is what turns "we picked K=1024" into a rule someone else can apply.

The interesting part is that T9 already forecast where the U bottoms out, and it
is not the same place in all four pairings. At K=1024 the sparsest tenth of
occupied cells holds 54 and 28 B queries in the same-modal pairings but 4 and 2
in the cross-modal ones -- unusable at any reliability floor. So the cross-modal
tilings are expected to want a COARSER K than the same-modal ones, and if that
shows up as two different U-bottoms it is a finding about the modality gap, not
a tuning detail.

ONE KIND OF TILING ONLY
-----------------------
Every K in the sweep is a fresh k-means over the index set A, produced by T5's
``build_landmarks.py`` and assigned by its ``occupancy_check.py``. A K with no
such tiling on disk is SKIPPED and named in the report -- it is never
substituted, approximated, or interpolated. A resolution sweep whose points
were produced by different procedures would report the difference between the
procedures as if it were a difference in resolution.

WHAT IS RECOMPUTED HERE, AND WHY
--------------------------------
Cell means are rebuilt from B's labels at every K, including K=1024 where T9's
map already holds them. Two reasons: the sweep must treat every K identically
or a difference in method would masquerade as a difference in resolution; and
recomputing at 1024 gives a free check against T9's stored ``cell_stats.npz``,
which is asserted (``--check-1024``, on by default) rather than assumed.

Nothing in this sweep re-runs a search or touches an index. CF labels are
properties of (query, configuration) and do not depend on the tiling at all --
only the ASSIGNMENT of queries to cells does. That is why a K sweep costs
k-means and arithmetic rather than another exact-KNN pass.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np

from ht_config import (PAIRINGS, add_common_args, ht_layout, layout,
                       load_query_geometry, load_role_ids, open_map_meta,
                       resolve_configs, resolve_pairings, write_csv, write_json)
from metrics import fmt, regression_metrics

DEFAULT_SWEEP = (64, 128, 256, 512, 1024, 2048, 4096)


# ==========================================================================
# Serving a map at an arbitrary K, from assignments alone
# ==========================================================================
def cell_means(assign_b: np.ndarray, cf_b: np.ndarray, K: int
               ) -> Tuple[np.ndarray, np.ndarray]:
    n = np.bincount(assign_b, minlength=K).astype(np.int64)
    s = np.bincount(assign_b, weights=cf_b, minlength=K)
    mean = np.zeros(K, dtype=np.float64)
    nz = n > 0
    mean[nz] = s[nz] / n[nz]
    return mean, n


def serve(assign_c: np.ndarray, mean: np.ndarray, n: np.ndarray,
          ga: float, floor: int) -> Tuple[np.ndarray, np.ndarray]:
    thin = n < floor
    m = mean.copy()
    m[thin] = ga
    return m[assign_c], thin[assign_c]


# ==========================================================================
# Where a given K's assignments come from
# ==========================================================================
def real_assignments(args, k: int, pairing: str
                     ) -> Optional[Dict[str, np.ndarray]]:
    """T5's own assignments at K, if ``occupancy_check.py`` has been run there.

    Returns None when that tiling does not exist, and the caller skips the K
    rather than substituting anything for it.

    Note this needs only T5, not T9: H4 does not require the served map to have
    been built at every K, because it rebuilds cell means from B itself. So the
    ladder is ``build_landmarks.py`` then ``occupancy_check.py``, and nothing
    more.
    """
    lay = layout(args.model, k, args.spherical, pilot=args.pilot_dir,
                 kmeans=args.kmeans_dir, cf=args.cf_dir, out=args.map_dir)
    spec = PAIRINGS[pairing]
    path = lay.query_nn(spec["index_modality"], "B", spec["query_modality"])
    if not path.exists():
        return None
    try:
        gb = load_query_geometry(lay, pairing, "B")
        gc = load_query_geometry(lay, pairing, "C")
    except FileNotFoundError:
        return None
    return {"assign_b": gb["assign"], "assign_c": gc["assign"], "K": k}


# ==========================================================================
# The sweep
# ==========================================================================
def run(args) -> dict:
    base_lay = layout(args.model, args.k, args.spherical, pilot=args.pilot_dir,
                      kmeans=args.kmeans_dir, cf=args.cf_dir, out=args.map_dir)
    meta = open_map_meta(base_lay)
    floor = args.floor if args.floor is not None else int(meta["reliability_floor"])
    pairings = [p for p in resolve_pairings(args.pairing) if p in meta["pairings"]]
    configs = resolve_configs(args.model, args.configs)
    ht = ht_layout(args.model, args.out_dir)
    dest = ht.ensure("H4")
    ks = sorted(set(int(k) for k in args.sweep))

    report = {
        "created": datetime.now().isoformat(),
        "hypothesis": "H4",
        "question": "how fine should the tiling be, and where does per-cell "
                    "starvation overtake the gain from finer cells",
        "model": args.model, "source_tiling": base_lay.variant, "floor": floor,
        "sweep": ks, "measured_on": "C (held-out); cell means rebuilt from B at "
                                    "every K by the identical procedure",
        "tiling_source": "fresh k-means over A at every K (T5 build_landmarks "
                         "then occupancy_check); a K without one is skipped, "
                         "never substituted",
        "expected_shape": "U (or a plateau): irreducible error falls with K, "
                          "estimation error rises as B is sliced thinner",
        "skipped_k": {},
        "pairings": {},
    }
    rows = []

    for pairing in pairings:
        spec = PAIRINGS[pairing]
        base = real_assignments(args, args.k, pairing)
        if base is None:
            print(f"  skip {pairing}: no assignments at the source K={args.k}")
            continue

        cf_b, cf_c, live = {}, {}, []
        for cfg in configs:
            fb = base_lay.cf_vector(pairing, "B", cfg)
            fc = base_lay.cf_vector(pairing, "C", cfg)
            if fb.exists() and fc.exists():
                cf_b[cfg] = np.load(fb).astype(np.float64)
                cf_c[cfg] = np.load(fc).astype(np.float64)
                live.append(cfg)
        if not live:
            continue
        ga = {c: float(cf_b[c].mean()) for c in live}

        print(f"\n  {pairing}   ({len(live)} configs, "
              f"B {len(base['assign_b']):,} / C {len(base['assign_c']):,})")
        print(f"    {'K':>6} {'MAE':>8} {'skill':>8} {'rho':>8} "
              f"{'backoff':>8} {'empty':>7} {'minB':>6} {'medB':>6} "
              f"{'rows':>8}")

        per_k, skipped = {}, []
        for k in ks:
            tiling = real_assignments(args, k, pairing)
            if tiling is None:
                skipped.append(k)
                continue

            ab, ac, K = tiling["assign_b"], tiling["assign_c"], tiling["K"]
            maes, skills, rhos, backoffs = [], [], [], []
            counts_seen = None
            for cfg in live:
                mean, n = cell_means(ab, cf_b[cfg], K)
                if counts_seen is None:
                    counts_seen = n
                pred, backed = serve(ac, mean, n, ga[cfg], floor)
                m = regression_metrics(pred, cf_c[cfg], ga=ga[cfg])
                maes.append(m["MAE"]); skills.append(m["skill"])
                rhos.append(m["spearman"]); backoffs.append(float(backed.mean()))

                if k == args.k and args.check_1024:
                    _check_against_t9(base_lay, pairing, cfg, mean, n)

            occupied = counts_seen[counts_seen > 0]
            entry = {
                "K": K,
                "MAE": float(np.mean(maes)),
                "skill": float(np.nanmean(skills)),
                "spearman": float(np.nanmean(rhos)),
                "backoff_rate": float(np.mean(backoffs)),
                "empty_cells": int((counts_seen == 0).sum()),
                "min_b_per_occupied_cell": int(occupied.min()) if len(occupied) else 0,
                "median_b_per_occupied_cell": float(np.median(occupied))
                if len(occupied) else 0.0,
                "decile_cutoff_b_per_cell": float(np.quantile(occupied, 0.1))
                if len(occupied) else 0.0,
                # storage and lookup, the other two axes S5.5 asks for
                "served_rows": int(K * len(live)),
                "lookup_scan_points": int(K),
            }
            per_k[str(k)] = entry
            rows.append([pairing, K, len(live),
                         f"{entry['MAE']:.5f}", f"{entry['skill']:.5f}",
                         f"{entry['spearman']:.5f}", f"{entry['backoff_rate']:.5f}",
                         entry["empty_cells"], entry["min_b_per_occupied_cell"],
                         entry["median_b_per_occupied_cell"],
                         entry["decile_cutoff_b_per_cell"],
                         entry["served_rows"], entry["lookup_scan_points"]])
            print(f"    {K:>6} {entry['MAE']:>8.5f} "
                  f"{entry['skill']:>8.4f} {fmt(entry['spearman'],4):>8} "
                  f"{entry['backoff_rate']:>8.3f} {entry['empty_cells']:>7} "
                  f"{entry['min_b_per_occupied_cell']:>6} "
                  f"{entry['median_b_per_occupied_cell']:>6.0f} "
                  f"{entry['served_rows']:>8}")

        if skipped:
            report["skipped_k"][pairing] = skipped
            print(f"    !! no k-means tiling on disk for K = {skipped}; these "
                  f"points are ABSENT from the sweep, not approximated.")
            print(f"       run: landmarks/build_landmarks.py --model "
                  f"{args.model} --modality both --spherical --k "
                  f"{' '.join(str(k) for k in skipped)}")

        if per_k:
            best = max(per_k.values(), key=lambda e: e["skill"])
            ks_seen = [e["K"] for e in per_k.values()]
            shape = _shape(per_k)
            report["pairings"][pairing] = {
                "same_modality": bool(spec["same_modality"]),
                "n_configs": len(live), "per_k": per_k,
                "best_K_by_skill": best["K"],
                "best_skill": best["skill"],
                "source_K_skill": per_k.get(str(args.k), {}).get("skill"),
                "shape": shape,
                "k_range_tested": [min(ks_seen), max(ks_seen)],
                "k_requested_but_absent": skipped,
            }
            print(f"    -> best K = {best['K']} (skill {best['skill']:.4f}); "
                  f"curve looks {shape}")

    write_csv(dest / "h4_sweep.csv",
              ["pairing", "K", "n_configs", "MAE", "skill",
               "spearman", "backoff_rate", "empty_cells", "min_B_per_cell",
               "median_B_per_cell", "decile_cutoff_B_per_cell", "served_rows",
               "lookup_scan_points"], rows)
    write_json(dest / "h4_report.json", report)
    _plot(report, dest / "figures" / "h4_resolution_sweep.png")
    print(f"\n  wrote {dest / 'h4_report.json'}")
    return report


def _shape(per_k: Dict[str, dict]) -> str:
    """Name the curve: the H4 finding is the SHAPE, not the winning K."""
    items = sorted(per_k.values(), key=lambda e: e["K"])
    s = [e["skill"] for e in items]
    if len(s) < 3:
        return "too-few-points"
    peak = int(np.argmax(s))
    span = max(s) - min(s)
    if span < 0.02:
        return "flat (resolution barely matters over this range)"
    if peak == 0:
        return "monotone-decreasing (already past the bottom of the U -- try coarser)"
    if peak == len(s) - 1:
        return "monotone-improving (never reached starvation -- try finer)"
    return f"U-shaped, bottom at K={items[peak]['K']}"


def _check_against_t9(lay, pairing: str, cfg: str, mean: np.ndarray,
                      n: np.ndarray) -> None:
    """Assert the recomputed cell means reproduce T9's stored map exactly.

    If this fires, the sweep and the served map disagree about what "the cell
    mean at K=1024" is, which would make every other K in the sweep suspect
    too. Cheap to check, and it fails loudly rather than shifting a curve.
    """
    p = lay.pairing_dir(pairing) / "cell_stats.npz"
    if not p.exists():
        return
    z = np.load(p, allow_pickle=False)
    names = [str(c) for c in z["configs"].tolist()]
    if cfg not in names:
        return
    r = names.index(cfg)
    stored_n = z["n"][r].astype(np.int64)
    stored_mean = z["cf_mean"][r].astype(np.float64)
    if not np.array_equal(stored_n, n):
        raise AssertionError(
            f"H4 recomputed cell counts differ from T9's for {pairing}/{cfg}: "
            f"the sweep is not rebuilding the same map T9 serves")
    occupied = n > 0
    if not np.allclose(stored_mean[occupied], mean[occupied], atol=1e-6):
        raise AssertionError(
            f"H4 recomputed cell means differ from T9's for {pairing}/{cfg}")


def _plot(report: dict, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pairings = list(report["pairings"])
    if not pairings:
        return
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.6))
    cmap = plt.get_cmap("tab10")
    for i, pairing in enumerate(pairings):
        pk = sorted(report["pairings"][pairing]["per_k"].values(),
                    key=lambda e: e["K"])
        ks = [e["K"] for e in pk]
        color = cmap(i % 10)
        label = pairing.replace("_query_", " q / ").replace("_index", " idx")
        axes[0, 0].plot(ks, [e["skill"] for e in pk], marker="o", color=color,
                        label=label)
        axes[0, 1].plot(ks, [e["MAE"] for e in pk], marker="o", color=color)
        axes[1, 0].plot(ks, [e["median_b_per_occupied_cell"] for e in pk],
                        marker="o", color=color)
        axes[1, 0].plot(ks, [e["decile_cutoff_b_per_cell"] for e in pk],
                        marker="s", ls="--", color=color, alpha=0.6)
        axes[1, 1].plot(ks, [e["backoff_rate"] for e in pk], marker="o",
                        color=color)

    for ax, title, ylab in (
            (axes[0, 0], "skill vs GA  (higher is better)", "mean skill"),
            (axes[0, 1], "MAE  (lower is better)", "mean MAE"),
            (axes[1, 0], "B queries per occupied cell\n(circle = median, "
                         "dashed square = bottom-decile cutoff)", "queries"),
            (axes[1, 1], "share of C served by GA backoff", "backoff rate")):
        ax.set_xscale("log", base=2)
        ax.set_xlabel("landmarks K")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.25)
    axes[1, 0].set_yscale("log")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("H4 -- landmark resolution sweep. Every point is a fresh "
                 "k-means over A at that K.", fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--sweep", nargs="+", type=int, default=list(DEFAULT_SWEEP),
                    help="landmark counts to evaluate; a K with no k-means "
                         "tiling on disk is skipped and named in the report, "
                         "never approximated")
    ap.add_argument("--no-check-1024", dest="check_1024", action="store_false",
                    help="skip the assertion that recomputed cell means "
                         "reproduce T9's stored map at the source K")
    ap.set_defaults(check_1024=True)
    args = ap.parse_args()
    print(f"H4  model={args.model}  landmark resolution sweep over {args.sweep}")
    run(args)


if __name__ == "__main__":
    main()
