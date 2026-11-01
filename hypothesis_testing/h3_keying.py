"""
H3 -- keyed versus shared-across-configurations ablation.

    Compare SM ((l, Aj)-keyed) against a variant keyed by landmark alone and
    shared across index families, on at least two structurally different
    families (e.g. HNSW and IVFPQ) built over the same embedding. P2 predicts
    the shared variant should perform visibly worse, since it conflates two
    different degradation mechanisms at the same location.       (draft S5.4)

WHAT IS ACTUALLY AT STAKE
-------------------------
The obvious stake is storage: a per-configuration map is K x m cells, a shared
one is K. But the scientific stake is larger and points the other way. P2 is
the claim that HNSW loses fidelity through graph connectivity and hub effects
while IVFPQ loses it through quantisation distortion and hard cell-boundary
exclusion -- DIFFERENT PHYSICS, so different places hurt. If that holds, the
per-configuration keying is not a storage choice but a necessity, and
collapsing to one column would not cost a little precision, it would cancel
signal: a cell that is poison for HNSW and painless for IVFPQ averages to
nothing.

If it does not hold, that is equally publishable and the paper is a different
one -- "a single spatial map governs all index families", ~100x smaller, with
this ablation as the headline rather than a supporting check.

FOUR RUNGS, NOT TWO
-------------------
The draft compares keyed against shared. Two rungs cannot separate the three
different things that could be true, so the ladder is:

    SM_shared         one deviation column of K numbers for all 9 configs,
                      re-levelled to each config's own mean
    SM_shared_scaled  the SAME column expressed as unit-scale SHAPE, with each
                      config fitting its own level AND amplitude (2 numbers)
    SM_family         one column per family (HNSW / IVFPQ / O-IVFPQ)
    SM_config         one column per configuration -- the proposal

The scaled rung was added after the first full run, and it is the one that
carries the argument. That run found SM_config beating SM_shared everywhere
(+0.10 to +0.69 skill) while the per-cell deviation columns correlated about as
strongly ACROSS families as within them (text->image: 0.673 within, 0.659
across). Both cannot be read as P2 support: configurations were agreeing about
WHICH cells are hard, which is precisely what P2 says they should not do.

The confound was amplitude. A shared column carrying raw deviations is
dominated by the configurations with the widest CF spread, and image->text's
ivfpq_mod -- true CF standard deviation 0.019 -- scored skill -4.72 under it.
The column was not pointing at the wrong cells; it was pointing at the right
cells far too loudly. So SM_shared could lose either because the geography is
genuinely per-configuration (P2 true) or merely because the magnitude is
(P2 false), and the two-rung ladder cannot tell them apart.

SM_shared_scaled removes the amplitude explanation. What remains of the gap
after rescaling is geography that genuinely differs per configuration, and the
verdict is read off THAT residual rather than off the raw gap:

    residual large, family blocks     P2 confirmed
    residual large, no family blocks  P2 confirmed weakly -- per-config
                                      geography is real but does not organise
                                      by family, so P2's physical story
                                      (graph connectivity vs quantisation) is
                                      not the mechanism
    residual ~ 0                      P2 FALSE: the win was amplitude. One
                                      shared shape column plus 2 numbers per
                                      config reproduces the map, it shrinks
                                      from K x m to K + 2m, and this ablation
                                      becomes the headline
    shared beats config               starvation, not disproof -- read H4

THE SECOND PIECE OF EVIDENCE
----------------------------
Accuracy alone can hide the mechanism. Two configurations can score identically
while disagreeing completely about WHICH cells are hard. So the per-cell
deviation-from-GA columns are correlated pairwise across all configurations and
reported as a matrix, sorted by family. P2 true looks like warm blocks on the
diagonal (each family has its own geography); P2 false looks like one uniformly
warm block (one shared difficulty geography). That picture, not the MAE table,
is what distinguishes "sharing loses a little precision" from "sharing cancels
the signal".
"""
from __future__ import annotations

import argparse
from datetime import datetime
from itertools import combinations

import numpy as np

from ht_config import (add_common_args, load_block, resolve_from_args,
                       write_csv, write_json)
from metrics import fmt, regression_metrics
from predictors import PER_CONFIG_PARAMS, KEYINGS, sm_on_c


def deviation_matrix(block, floor: int):
    """[n_configs, K] of each configuration's cell deviation from its own GA,
    with under-populated and empty cells masked out as nan.

    Deviations rather than raw means because raw means differ across
    configurations by their global level (0.43 for hnsw_low against 0.92 for
    hnsw_mod), and correlating levels would report agreement that is entirely
    about the catalogue's fidelity spread rather than about geography.
    """
    n_cfg, K = block.cell_mean.shape
    M = np.full((n_cfg, K), np.nan)
    for r, cfg in enumerate(block.configs):
        n = block.cell_n[r]
        ok = n >= floor
        M[r, ok] = block.cell_mean[r, ok] - block.ga[cfg]
    return M


def pairwise_corr(M: np.ndarray):
    """Pearson correlation between every pair of deviation columns, over the
    cells where BOTH are estimated. Pairwise-complete rather than dropping any
    cell missing anywhere: with 286 empty cells in image->text, a complete-case
    matrix would be computed on a small and unrepresentative remainder.
    """
    n = M.shape[0]
    C = np.full((n, n), np.nan)
    overlap = np.zeros((n, n), dtype=int)
    for i in range(n):
        C[i, i] = 1.0
        for j in range(i + 1, n):
            ok = np.isfinite(M[i]) & np.isfinite(M[j])
            overlap[i, j] = overlap[j, i] = int(ok.sum())
            if ok.sum() >= 10 and np.std(M[i, ok]) > 0 and np.std(M[j, ok]) > 0:
                C[i, j] = C[j, i] = float(np.corrcoef(M[i, ok], M[j, ok])[0, 1])
    return C, overlap


def run(args) -> dict:
    lay, ht, floor, pairings, configs, meta = resolve_from_args(args)
    dest = ht.ensure("H3")

    report = {
        "created": datetime.now().isoformat(),
        "hypothesis": "H3",
        "question": "must the map be keyed by (landmark, configuration), or "
                    "does one shared column of landmark difficulty suffice",
        "model": lay.model, "tiling": lay.variant, "floor": floor,
        "measured_on": "C (held-out)",
        "rungs": {
            "SM_shared": "one deviation column for all configurations, "
                         "re-levelled to each configuration's GA",
            "SM_shared_scaled": "the same column as unit-scale SHAPE, with a "
                                "per-configuration level and amplitude fitted "
                                "on B (2 numbers per configuration)",
            "SM_family": "one deviation column per index family, re-levelled",
            "SM_config": "one column per (landmark, configuration) -- proposed",
        },
        "levelling_note": "the shared and family rungs carry DEVIATIONS from "
                          "each configuration's own global mean, not pooled raw "
                          "means; pooling raw means would lose to GA on level "
                          "alone and prove nothing about geography",
        "pass_rule": "P2 is judged on the RESIDUAL gap SM_config - "
                     "SM_shared_scaled, not on SM_config - SM_shared: the "
                     "latter is inflated by per-configuration amplitude, which "
                     "is not what P2 claims. Residual > 0.05 supports P2; "
                     "residual ~ 0 means the win was amplitude and one shared "
                     "shape column suffices. Family block structure in the "
                     "deviation correlations decides whether P2's stated "
                     "mechanism (family physics) is the one operating.",
        "pairings": {},
    }
    rows, corr_rows = [], []

    for pairing in pairings:
        block = load_block(lay, pairing, configs, floor)
        if not block.configs:
            continue
        M = deviation_matrix(block, floor)
        C, overlap = pairwise_corr(M)
        fams = [block.families[c] for c in block.configs]

        within, across = [], []
        for i, j in combinations(range(len(block.configs)), 2):
            if not np.isfinite(C[i, j]):
                continue
            (within if fams[i] == fams[j] else across).append(C[i, j])
            corr_rows.append([pairing, block.configs[i], block.configs[j],
                              fams[i], fams[j],
                              "within" if fams[i] == fams[j] else "across",
                              overlap[i, j], f"{C[i, j]:.4f}"])

        print(f"\n  {pairing}")
        print(f"    {'config':<12} {'family':<8} | " +
              " | ".join(f"{k:>16}" for k in KEYINGS))
        print(f"    {'':<12} {'':<8} | " +
              " | ".join(f"{'MAE   skill':>16}" for _ in KEYINGS))

        per_config, skills = {}, {k: [] for k in KEYINGS}
        for cfg in block.configs:
            true = block.cf_c[cfg]
            entry = {"family": block.families[cfg], "ga": block.ga[cfg]}
            line = []
            for keying in KEYINGS:
                pred, backed = sm_on_c(block, cfg, floor, keying)
                m = regression_metrics(pred, true, ga=block.ga[cfg])
                m["backoff_rate"] = float(backed.mean())
                entry[keying] = m
                skills[keying].append(m["skill"])
                line.append(f"{fmt(m['MAE'],4):>7} {fmt(m['skill'],4):>8}")
            entry["config_minus_shared_skill"] = float(
                entry["SM_config"]["skill"] - entry["SM_shared"]["skill"])
            entry["config_minus_family_skill"] = float(
                entry["SM_config"]["skill"] - entry["SM_family"]["skill"])
            per_config[cfg] = entry
            rows.append([pairing, cfg, block.families[cfg]]
                        + [f"{entry[k][m]:.4f}" for k in KEYINGS
                           for m in ("MAE", "skill")]
                        + [f"{entry['config_minus_shared_skill']:.4f}",
                           f"{entry['config_minus_family_skill']:.4f}"])
            print(f"    {cfg:<12} {block.families[cfg]:<8} | " + " | ".join(line))

        mean_skill = {k: float(np.nanmean(skills[k])) for k in KEYINGS}
        w = float(np.mean(within)) if within else float("nan")
        a = float(np.mean(across)) if across else float("nan")
        margin = mean_skill["SM_config"] - mean_skill["SM_shared"]
        # The load-bearing quantity: what survives once the shared column is
        # allowed its own amplitude per configuration. Whatever remains here is
        # geography that genuinely differs per configuration -- which is what
        # P2 claims. Everything the scaled rung recovers was magnitude, not
        # location, and says nothing about P2 either way.
        residual = mean_skill["SM_config"] - mean_skill["SM_shared_scaled"]
        recovered = (1.0 - residual / margin) if margin > 1e-9 else float("nan")
        blocky = bool(np.isfinite(w) and np.isfinite(a) and (w - a) > 0.15)

        if margin < -0.02:
            v = "starvation (shared beats config -- read H4, trust the family rung)"
        elif margin <= 0.02:
            v = "P2-false-at-this-scale (one shared column suffices)"
        elif residual > 0.05 and blocky:
            v = "P2-confirmed (per-configuration geography survives rescaling)"
        elif residual > 0.05:
            v = ("P2-confirmed-weakly (per-configuration geography survives "
                 "rescaling, but no family block structure)")
        else:
            v = ("P2-false: the per-config win was AMPLITUDE, not geography -- "
                 "one shared shape column plus 2 numbers per config recovers it")

        report["pairings"][pairing] = {
            "n_c": block.n_c, "same_modality": block.same_modality,
            "per_config": per_config,
            "mean_skill": mean_skill,
            "config_minus_shared_skill": float(margin),
            "config_minus_shared_scaled_skill": float(residual),
            "fraction_of_gap_recovered_by_rescaling": float(recovered),
            "config_minus_family_skill": float(
                mean_skill["SM_config"] - mean_skill["SM_family"]),
            "storage_numbers_per_pairing": {
                k: int(block.K * (1 if k != "SM_family" else 3)
                       + PER_CONFIG_PARAMS[k] * len(block.configs))
                   if k != "SM_config" else int(block.K * len(block.configs))
                for k in KEYINGS},
            "deviation_correlation": {
                "mean_within_family": w, "mean_across_family": a,
                "block_structure": blocky,
                "n_within_pairs": len(within), "n_across_pairs": len(across),
            },
            "verdict": v,
        }
        print(f"    -> mean skill  shared {mean_skill['SM_shared']:.4f}  "
              f"family {mean_skill['SM_family']:.4f}  "
              f"config {mean_skill['SM_config']:.4f}")
        print(f"    -> deviation corr  within-family {fmt(w,3)}  "
              f"across-family {fmt(a,3)}")
        print(f"    -> gap config-shared {margin:+.4f}; after rescaling "
              f"{residual:+.4f}  ({fmt(recovered,2)} of the gap was amplitude)")
        print(f"    -> {v}")

        _plot_corr(block, C, dest / "figures" /
                   f"h3_config_correlation_{pairing}.png", pairing)

    header = (["pairing", "config", "family"]
              + [f"{k}_{m}" for k in KEYINGS for m in ("MAE", "skill")]
              + ["config_minus_shared_skill", "config_minus_family_skill"])
    write_csv(dest / "h3_keying.csv", header, rows)
    write_csv(dest / "h3_deviation_correlations.csv",
              ["pairing", "config_a", "config_b", "family_a", "family_b",
               "relation", "cells_overlapping", "pearson_r"], corr_rows)
    write_json(dest / "h3_report.json", report)
    _plot_ladder(report, dest / "figures" / "h3_keying_ladder.png")
    print(f"\n  wrote {dest / 'h3_report.json'}")
    return report


def _plot_corr(block, C, path, pairing) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = sorted(range(len(block.configs)),
                   key=lambda i: (block.families[block.configs[i]], i))
    names = [block.configs[i] for i in order]
    Cs = C[np.ix_(order, order)]
    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    im = ax.imshow(Cs, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=70, ha="right", fontsize=7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    for i in range(len(names)):
        for j in range(len(names)):
            if np.isfinite(Cs[i, j]):
                ax.text(j, i, f"{Cs[i,j]:.2f}", ha="center", va="center",
                        fontsize=5.5,
                        color="white" if abs(Cs[i, j]) > 0.55 else "black")
    fig.colorbar(im, ax=ax, label="correlation of per-cell deviation from GA")
    ax.set_title(f"H3 -- do configurations agree about geography?\n{pairing}\n"
                 "warm blocks on the diagonal = P2 true (per-family geography)",
                 fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_ladder(report: dict, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pairings = list(report["pairings"])
    if not pairings:
        return
    fig, ax = plt.subplots(figsize=(1.9 * len(pairings) + 3.2, 4.2))
    x = np.arange(len(pairings))
    colors = {"SM_shared": "#bdbdbd", "SM_shared_scaled": "#9ecae1",
              "SM_family": "#6baed6", "SM_config": "#08519c"}
    w = 0.8 / len(KEYINGS)
    for i, keying in enumerate(KEYINGS):
        vals = [report["pairings"][p]["mean_skill"][keying] for p in pairings]
        ax.bar(x + (i - (len(KEYINGS) - 1) / 2) * w, vals, w, label=keying,
               color=colors.get(keying, "#777777"))
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("_query_", " q / ").replace("_index", " idx")
                        for p in pairings], rotation=15, fontsize=8)
    ax.set_ylabel("mean skill vs GA, over configurations")
    ax.set_title("H3 -- the keying ladder. The load-bearing gap is "
                 "config vs shared+scale, not config vs shared.", fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    args = ap.parse_args()
    print(f"H3  model={args.model}  keyed vs shared across configurations")
    run(args)


if __name__ == "__main__":
    main()
