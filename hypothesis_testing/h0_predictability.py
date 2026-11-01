"""
H0 -- is candidate fidelity predictable from location at all?

    H0 (working hypothesis). A query's candidate fidelity under configuration A
    is predictable, to a useful approximation, from the local density of the
    corpus around the query's location alone -- without running A, and without
    computing the query's own exact top-k.                     (draft S4.1)

H5 asks whether the map as built beats the baselines. H0 asks the prior
question: is there anything in location to find, whatever any particular
estimator manages to extract? The two answers come apart in a way that matters,
and separating them is the whole point of running this first:

    signal present, captured    ICC high, skill ~ ICC     H0 holds, map works
    signal present, not captured ICC high, skill << ICC   H0 holds, map is
                                                          UNDERFED -- the fix is
                                                          more B queries or
                                                          coarser K (H4), not
                                                          abandoning the idea
    no signal                    ICC ~ 0                  H0 fails here; H5 will
                                                          confirm, and no tuning
                                                          rescues it

Reporting only H5 would collapse the middle row into the bottom one and draw
the wrong conclusion from it.

HOW THE CEILING IS MEASURED
---------------------------
ICC(1), the one-way random-effects intraclass correlation, computed on the TRUE
CF of the held-out C queries grouped by their landmark cell. It is the share of
CF variance that lies BETWEEN cells rather than within them -- i.e. exactly the
share a perfect map of this tiling could ever explain.

The luck correction is not optional. Group means built from finite samples
differ from each other even when the grouping is meaningless, so a raw
1 - within/total rises with the number of cells for free. ICC(1) subtracts that
expectation via ``(MSB - MSW) / (MSB + (n0-1) MSW)``, which is why a genuinely
uninformative tiling scores 0 rather than something reassuringly positive. A
slightly negative value is sampling noise around zero and is reported as-is
rather than clipped, so the reader can see it happened.

Measured on C, not B: cell means were fitted to B, so a decomposition on B
would be partly in-sample. This is the honest ceiling.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from typing import Dict, Tuple

import numpy as np

from ht_config import (Block, add_common_args, load_block, resolve_from_args,
                       write_csv, write_json)
from metrics import fmt, regression_metrics
from predictors import do_features, predict_do, sm_on_c


def icc1(values: np.ndarray, groups: np.ndarray, n_groups: int
         ) -> Dict[str, float]:
    """One-way random-effects ICC of ``values`` grouped by ``groups``.

    Returns the ICC together with the raw variance components, because the
    components are what make a surprising ICC interpretable -- a near-zero ICC
    on a config whose CF barely varies at all (hnsw_high) is a different fact
    from a near-zero ICC on one with a wide spread.
    """
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups)
    n = len(values)
    counts = np.bincount(groups, minlength=n_groups).astype(np.float64)
    occupied = counts > 0
    k = int(occupied.sum())
    if n < 3 or k < 2:
        return {"icc": float("nan"), "n": int(n), "n_cells": k}

    sums = np.bincount(groups, weights=values, minlength=n_groups)
    means = np.zeros(n_groups)
    means[occupied] = sums[occupied] / counts[occupied]
    grand = values.mean()

    ss_between = float(np.sum(counts[occupied] * (means[occupied] - grand) ** 2))
    ss_within = float(np.sum((values - means[groups]) ** 2))
    df_b, df_w = k - 1, n - k
    if df_w <= 0:
        return {"icc": float("nan"), "n": int(n), "n_cells": k}

    msb, msw = ss_between / df_b, ss_within / df_w
    # n0: the "effective" cell size for unbalanced groups. With equal counts it
    # reduces to the common count; with wildly unequal ones -- which is exactly
    # the cross-modal case -- it is pulled well below the mean.
    n0 = (n - float(np.sum(counts[occupied] ** 2)) / n) / df_b
    denom = msb + (n0 - 1.0) * msw
    return {
        "icc": float((msb - msw) / denom) if denom > 0 else float("nan"),
        "msb": msb, "msw": msw, "n0": float(n0),
        "between_var_share_uncorrected": float(ss_between / (ss_between + ss_within))
        if ss_between + ss_within > 0 else float("nan"),
        "cf_std": float(values.std()),
        "n": int(n), "n_cells": k,
    }


def verdict(icc: float, skill: float) -> str:
    """The three-way call the docstring's table sets out."""
    if not np.isfinite(icc) or icc < 0.02:
        return "no-signal"
    if not np.isfinite(skill):
        return "no-signal"
    if skill >= 0.5 * icc:
        return "captured"
    return "underfed"


def run(args) -> dict:
    lay, ht, floor, pairings, configs, meta = resolve_from_args(args)
    dest = ht.ensure("H0")

    report = {
        "created": datetime.now().isoformat(),
        "hypothesis": "H0",
        "question": "is CF predictable from location at all, before asking "
                    "whether this particular map extracts it",
        "model": lay.model, "tiling": lay.variant, "floor": floor,
        "measured_on": "C (held-out)",
        "icc_definition": "ICC(1), one-way random effects, luck-corrected: "
                          "(MSB - MSW) / (MSB + (n0-1) MSW)",
        "verdict_rule": "captured if skill >= 0.5*ICC; underfed if 0 < skill < "
                        "0.5*ICC; no-signal if ICC < 0.02",
        "pairings": {},
    }
    rows = []

    for pairing in pairings:
        block = load_block(lay, pairing, configs, floor)
        if not block.configs:
            continue
        X = do_features(lay, block)
        assign_c = block.geom_c["assign"]
        print(f"\n  {pairing}   (C queries: {block.n_c:,}, K={block.K})")
        print(f"    {'config':<12} {'CF sd':>7} {'ICC':>7} {'skill SM':>9} "
              f"{'skill DO':>9} {'captured':>9}  verdict")

        per_config = {}
        for cfg in block.configs:
            true_c = block.cf_c[cfg]
            comp = icc1(true_c, assign_c, block.K)
            sm_pred, backed = sm_on_c(block, cfg, floor)
            do_pred = predict_do(X, block, cfg)[1]

            m_sm = regression_metrics(sm_pred, true_c, ga=block.ga[cfg])
            m_do = regression_metrics(do_pred, true_c, ga=block.ga[cfg])
            captured = (m_sm["skill"] / comp["icc"]
                        if np.isfinite(comp["icc"]) and comp["icc"] > 0
                        else float("nan"))
            v = verdict(comp["icc"], m_sm["skill"])

            per_config[cfg] = {
                "family": block.families[cfg], "ga": block.ga[cfg],
                "icc": comp, "skill_SM": m_sm["skill"], "skill_DO": m_do["skill"],
                "MAE_SM": m_sm["MAE"], "spearman_SM": m_sm["spearman"],
                "captured_fraction_of_ceiling": captured,
                "backoff_rate": float(backed.mean()), "verdict": v,
            }
            rows.append([pairing, cfg, block.families[cfg], block.n_c,
                         comp["n_cells"], f"{comp['cf_std']:.4f}",
                         f"{comp['icc']:.4f}", f"{m_sm['skill']:.4f}",
                         f"{m_do['skill']:.4f}", f"{captured:.4f}",
                         f"{backed.mean():.4f}", v])
            print(f"    {cfg:<12} {fmt(comp['cf_std'],4):>7} {fmt(comp['icc'],4):>7} "
                  f"{fmt(m_sm['skill'],4):>9} {fmt(m_do['skill'],4):>9} "
                  f"{fmt(captured,3):>9}  {v}")

        verdicts = [v["verdict"] for v in per_config.values()]
        report["pairings"][pairing] = {
            "K": block.K, "n_c": block.n_c,
            "same_modality": block.same_modality,
            "per_config": per_config,
            "n_no_signal": verdicts.count("no-signal"),
            "n_underfed": verdicts.count("underfed"),
            "n_captured": verdicts.count("captured"),
            "median_icc": float(np.nanmedian(
                [v["icc"]["icc"] for v in per_config.values()])),
            "median_skill": float(np.nanmedian(
                [v["skill_SM"] for v in per_config.values()])),
        }

    write_csv(dest / "h0_icc_table.csv",
              ["pairing", "config", "family", "n_c", "n_cells", "cf_std", "icc",
               "skill_SM", "skill_DO", "captured_fraction", "backoff_rate",
               "verdict"], rows)
    write_json(dest / "h0_report.json", report)
    _plot(report, dest / "figures" / "h0_icc_vs_skill.png")
    print(f"\n  wrote {dest / 'h0_report.json'}")
    print(f"  wrote {dest / 'h0_icc_table.csv'}")
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
        icc = [pc[c]["icc"]["icc"] for c in names]
        skill = [pc[c]["skill_SM"] for c in names]
        ax.bar(x - 0.2, icc, 0.4, label="ICC (ceiling)", color="#9ecae1")
        ax.bar(x + 0.2, skill, 0.4, label="skill of SM (realised)", color="#3182bd")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=70, ha="right", fontsize=7)
        ax.set_title(pairing.replace("_query_", " q / ").replace("_index", " idx"),
                     fontsize=9)
        ax.set_ylabel("share of CF variance")
        ax.legend(fontsize=7)
    fig.suptitle("H0 -- how much of CF location could explain (ICC), and how "
                 "much the map does (skill), on held-out C", fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    args = ap.parse_args()
    print(f"H0  model={args.model}  is CF predictable from location at all?")
    run(args)


if __name__ == "__main__":
    main()
