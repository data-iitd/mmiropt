"""
H2 -- boundary stress test.

    Partition held-out queries into interior (nearest landmark much closer than
    second-nearest) and boundary (comparably close) subsets, and report SM
    accuracy separately for each. P1's failure mode predicts a visible accuracy
    gap between the two subsets; this directly tests the boundary-degradation
    concern rather than letting it average out and disappear inside an
    aggregate metric.                                            (draft S5.3)

H1 AND H2 ARE NOT THE SAME TEST
-------------------------------
H1 fits a trend across all queries; a real but modest slope can coexist with a
perfectly usable map. H2 asks the deployment question instead: is there a
population of queries the map serves BADLY, identifiable at query time from
geometry alone? That is sharper, and it is actionable in a way a slope is not
-- a system that can recognise its boundary queries can refuse to trust the map
for them, which is a policy; "error rises with the ratio" is not.

THE SPLIT
---------
On ``ratio = d1/d2``, not on the raw gap. The four pairings sit at very
different absolute distances from their landmarks (T9 measured d1 ~ 0.39-0.71
same-modal against ~ 1.34-1.36 cross-modal), so a fixed gap threshold would
classify almost every cross-modal query as a boundary query for reasons of
scale rather than geometry. The ratio is scale-free.

Two splits are reported, and they answer different questions:

  quantile   bottom third of the ratio vs top third, WITHIN each pairing.
             Balanced by construction, so the two subsets have comparable
             statistical power and the comparison is fair. This is the headline.

  absolute   ratio >= 0.9 is "boundary" everywhere. Unbalanced -- in some
             pairings almost every query qualifies -- but it means the same
             thing in all four, which the quantile split does not. It is what
             says whether the pairings differ in how boundary-like they are,
             rather than only in how their own boundary-most third behaves.

The middle third is excluded from the quantile comparison on purpose: including
it would blunt the contrast the test exists to expose, and it is reported
separately so nothing is hidden.

WHAT COUNTS AS A FAILURE
------------------------
A gap in MAE alone is not enough, because MAE is not comparable across
populations with different CF spread -- if boundary queries simply have more
variable true CF, every predictor including GA does worse there. So the test is
run on SKILL relative to GA computed WITHIN each subset. That asks whether the
MAP specifically degrades at boundaries, rather than whether boundaries are
just harder for everyone.
"""
from __future__ import annotations

import argparse
from datetime import datetime

import numpy as np

from ht_config import (add_common_args, load_block, resolve_from_args,
                       write_csv, write_json)
from metrics import fmt, regression_metrics
from predictors import sm_on_c

ABS_BOUNDARY = 0.90       # ratio at or above this is "on the wall"


def _subset_metrics(pred, true, mask) -> dict:
    """Metrics within one subset, with GA recomputed INSIDE that subset.

    The within-subset GA is the point: comparing the map's boundary MAE against
    the global GA would conflate "the map is worse here" with "this population
    is harder for anything".
    """
    if mask.sum() < 30:
        return {"n": int(mask.sum()), "insufficient": True}
    p, t = pred[mask], true[mask]
    m = regression_metrics(p, t, ga=float(t.mean()))
    m["true_mean"] = float(t.mean())
    m["true_std"] = float(t.std())
    return m


def run(args) -> dict:
    lay, ht, floor, pairings, configs, meta = resolve_from_args(args)
    dest = ht.ensure("H2")

    report = {
        "created": datetime.now().isoformat(),
        "hypothesis": "H2",
        "question": "do queries near a cell wall get a measurably worse map "
                    "than interior queries",
        "model": lay.model, "tiling": lay.variant, "floor": floor,
        "measured_on": "C (held-out)",
        "split_variable": "ratio = d1/d2 in (0,1]; 1.0 = exactly on the wall",
        "splits": {
            "quantile": "interior = bottom third of ratio, boundary = top third, "
                        "within each pairing; middle third excluded and reported "
                        "separately",
            "absolute": f"boundary = ratio >= {ABS_BOUNDARY}, same threshold in "
                        f"all four pairings",
        },
        "comparison_metric": "skill vs a GA recomputed WITHIN each subset, so a "
                             "gap means the map degrades rather than the subset "
                             "being harder for every predictor",
        "pass_rule": "P1's boundary failure mode is CONFIRMED for a block when "
                     "interior skill exceeds boundary skill by more than 0.05",
        "pairings": {},
    }
    rows = []

    for pairing in pairings:
        block = load_block(lay, pairing, configs, floor)
        if not block.configs:
            continue
        ratio = block.ratio("C")
        q33, q67 = np.quantile(ratio, [1 / 3, 2 / 3])
        interior_q = ratio <= q33
        boundary_q = ratio >= q67
        middle_q = ~(interior_q | boundary_q)
        boundary_a = ratio >= ABS_BOUNDARY
        interior_a = ~boundary_a

        print(f"\n  {pairing}   ratio terciles at {q33:.4f} / {q67:.4f};  "
              f"{boundary_a.mean():.1%} of C at ratio >= {ABS_BOUNDARY}")
        print(f"    {'config':<12} | {'interior':>18} | {'boundary':>18} | "
              f"{'gap':>16}")
        print(f"    {'':<12} | {'MAE   skill':>18} | {'MAE   skill':>18} | "
              f"{'dMAE  dskill':>16}  P1 fail mode")

        per_config = {}
        for cfg in block.configs:
            pred, backed = sm_on_c(block, cfg, floor)
            true = block.cf_c[cfg]

            qi = _subset_metrics(pred, true, interior_q)
            qb = _subset_metrics(pred, true, boundary_q)
            qm = _subset_metrics(pred, true, middle_q)
            ai = _subset_metrics(pred, true, interior_a)
            ab = _subset_metrics(pred, true, boundary_a)

            def gap(a, b, key):
                if a.get("insufficient") or b.get("insufficient"):
                    return float("nan")
                return float(a[key] - b[key])

            d_skill = gap(qi, qb, "skill")
            d_mae = gap(qb, qi, "MAE")      # boundary minus interior: positive = worse
            confirmed = bool(np.isfinite(d_skill) and d_skill > 0.05)

            per_config[cfg] = {
                "family": block.families[cfg],
                "quantile": {"interior": qi, "middle": qm, "boundary": qb,
                             "delta_skill_interior_minus_boundary": d_skill,
                             "delta_mae_boundary_minus_interior": d_mae},
                "absolute": {"interior": ai, "boundary": ab,
                             "delta_skill_interior_minus_boundary":
                                 gap(ai, ab, "skill"),
                             "delta_mae_boundary_minus_interior":
                                 gap(ab, ai, "MAE"),
                             "boundary_fraction": float(boundary_a.mean())},
                "backoff_rate_interior": float(backed[interior_q].mean()),
                "backoff_rate_boundary": float(backed[boundary_q].mean()),
                "p1_failure_mode_confirmed": confirmed,
            }
            rows.append([
                pairing, cfg, block.families[cfg],
                int(interior_q.sum()), int(boundary_q.sum()),
                f"{qi.get('MAE', float('nan')):.4f}",
                f"{qi.get('skill', float('nan')):.4f}",
                f"{qb.get('MAE', float('nan')):.4f}",
                f"{qb.get('skill', float('nan')):.4f}",
                f"{d_mae:.4f}", f"{d_skill:.4f}",
                f"{ab.get('MAE', float('nan')):.4f}",
                f"{ab.get('skill', float('nan')):.4f}",
                "CONFIRMED" if confirmed else "not-seen"])
            print(f"    {cfg:<12} | {fmt(qi.get('MAE'),4):>8} "
                  f"{fmt(qi.get('skill'),4):>9} | {fmt(qb.get('MAE'),4):>8} "
                  f"{fmt(qb.get('skill'),4):>9} | {fmt(d_mae,4):>7} "
                  f"{fmt(d_skill,4):>8}  "
                  f"{'CONFIRMED' if confirmed else 'not-seen'}")

        n_conf = sum(v["p1_failure_mode_confirmed"] for v in per_config.values())
        report["pairings"][pairing] = {
            "n_c": block.n_c, "same_modality": block.same_modality,
            "ratio_tercile_cuts": [float(q33), float(q67)],
            "ratio_mean": float(ratio.mean()),
            "frac_at_or_above_abs_threshold": float(boundary_a.mean()),
            "per_config": per_config,
            "n_configs": len(per_config),
            "n_boundary_degradation_confirmed": n_conf,
            "verdict": ("boundary-degradation-visible" if n_conf > len(per_config) / 2
                        else "partial" if n_conf else "no-boundary-effect"),
        }

    write_csv(dest / "h2_boundary.csv",
              ["pairing", "config", "family", "n_interior", "n_boundary",
               "interior_MAE", "interior_skill", "boundary_MAE", "boundary_skill",
               "delta_MAE", "delta_skill", "abs_boundary_MAE",
               "abs_boundary_skill", "p1_failure_mode"], rows)
    write_json(dest / "h2_report.json", report)
    _plot(report, dest / "figures" / "h2_interior_vs_boundary.png")
    print(f"\n  wrote {dest / 'h2_report.json'}")
    return report


def _plot(report: dict, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pairings = list(report["pairings"])
    if not pairings:
        return
    fig, axes = plt.subplots(1, len(pairings), figsize=(4.6 * len(pairings), 4.4),
                             squeeze=False)
    for ax, pairing in zip(axes[0], pairings):
        pc = report["pairings"][pairing]["per_config"]
        names = list(pc)
        x = np.arange(len(names))
        inter = [pc[c]["quantile"]["interior"].get("skill", np.nan) for c in names]
        bound = [pc[c]["quantile"]["boundary"].get("skill", np.nan) for c in names]
        ax.bar(x - 0.2, inter, 0.4, label="interior third", color="#31a354")
        ax.bar(x + 0.2, bound, 0.4, label="boundary third", color="#de2d26")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=70, ha="right", fontsize=7)
        ax.set_ylabel("skill vs within-subset GA")
        ax.set_title(pairing.replace("_query_", " q / ").replace("_index", " idx"),
                     fontsize=9)
        ax.legend(fontsize=7)
    fig.suptitle("H2 -- P1's failure mode predicts the red bars sit below the "
                 "green ones", fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    args = ap.parse_args()
    print(f"H2  model={args.model}  interior vs boundary queries")
    run(args)


if __name__ == "__main__":
    main()
