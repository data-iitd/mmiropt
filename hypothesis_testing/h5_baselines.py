"""
H5 -- does location help at all?  THE GO/NO-GO GATE.

    Compare SM directly against GA and DO on the same held-out query set. This
    is the most fundamental check: if SM does not clearly outperform GA, the
    entire premise that location predicts fidelity better than a single global
    number is unsupported, independent of any of the more detailed diagnoses in
    H1-H4.                                                       (draft S5.6)

This script decides whether the pilot scales to 7M. The draft is explicit about
how a negative result is to be read (S9): "if H5 shows SM does not beat GA ...
the right conclusion is that H0 does not hold at the tested scale or corpus,
NOT that the map needs more tuning." That reading is encoded in the verdict
below rather than left to the reader.

WHAT IS REPORTED, AND WHY EACH IS NEEDED
----------------------------------------
MAE and Spearman            S5.7's headline pair.

Calibration curves          "correlation alone can look favorable while the
                            estimator is still poorly calibrated, and a
                            downstream fidelity-floor policy would need the
                            latter, not only the former" (S5.7). Binned
                            predicted-vs-observed, plus the count-weighted
                            deviation from the diagonal.

Threshold accuracy at tau   The map's actual job is a decision, and MAE charges
                            error everywhere while only error NEAR tau can flip
                            a decision.

Realized CF regret at tau   Accuracy still treats every wrong call alike. Regret
                            asks what the wrong call COST: the true fidelity of
                            the configuration the map would run, against the one
                            an oracle would run, under the identical
                            cheapest-clearing policy. A wrong pick between two
                            near-tied configurations is then penalised far less
                            than a genuine miss.

THE COST BASIS
--------------
The regret policy needs an ordering over configurations by cost. It uses T7's
MEASURED queries-per-second for this pairing on role C -- the throughput
recorded while running the very searches being graded -- rather than a rank
assigned by hand. The full cost model (latency + storage + energy) is deferred
to the parent work; latency is the component actually measured here, and every
report states that this is what "cheapest" meant.

DO IS DELIBERATELY OVER-EQUIPPED
--------------------------------
S5.1 specifies DO as a density score combined with GA "via a simple fitted
offset". This implementation gives it three corpus-geometry features and a full
OLS fit instead. If SM still beats it, that margin is not an artefact of
handicapping the rival explanation -- which is the point of having the rival at
all.
"""
from __future__ import annotations

import argparse
from datetime import datetime

import numpy as np

from ht_config import (TAUS, add_common_args, cost_basis_note, load_block,
                       resolve_from_args, write_csv, write_json)
from metrics import (calibration_curve, fmt, realized_regret,
                     regression_metrics, threshold_metrics,
                     CellWeights, boot_ci, skill_replicates, wilcoxon_paired)

#: draft main.tex l.449: configurations whose true fidelity has a standard deviation
#: below 0.03 in a direction are excluded from skill-based averages (measured on Q_test).
FLAT_SD = 0.03
from predictors import KEYINGS, all_predictions_on_c

#: The margin SM must clear over GA, in skill, for the gate to open. Skill is
#: 1 - MSE(SM)/MSE(GA), so 0.05 means "removes at least 5% of GA's squared
#: error". Chosen as a floor for "clearly outperform" (S5.6) rather than
#: statistical significance: at ~50,000 held-out queries almost any positive
#: margin is significant, and significance is not the question -- usefulness is.
GATE_SKILL_MARGIN = 0.05

PREDICTORS = ("GA", "DO", "SM_shared", "SM_shared_scaled",
              "SM_family", "SM_config")


def run(args) -> dict:
    global PREDICTORS
    lay, ht, floor, pairings, configs, meta = resolve_from_args(args)
    import ht_config as _htc
    if _htc.DO_PROTOCOL:
        PREDICTORS = ("GA", "DO", "DO_shared", "SM_shared", "SM_shared_scaled",
                      "SM_family", "SM_config")
    dest = ht.ensure("H5")

    report = {
        "created": datetime.now().isoformat(),
        "hypothesis": "H5",
        "question": "does the spatial map beat GA and DO on held-out queries",
        "role": "GO/NO-GO GATE before scaling the 1M pilot to 7M",
        "model": lay.model, "tiling": lay.variant, "floor": floor,
        "measured_on": "C (held-out); every predictor fitted on B only",
        "taus": list(TAUS),
        "gate_rule": f"PASS when SM_config's skill vs GA exceeds "
                     f"{GATE_SKILL_MARGIN} in every pairing; a FAIL is to be "
                     f"read as H0 not holding at this scale or corpus, not as "
                     f"the map needing tuning (draft S9)",
        "cost_basis": cost_basis_note(),
        "do_note": ("DO per the protocol (l.108): one density score per landmark, "
                    "the mean L2 distance to its 100 nearest corpus vectors, as a "
                    "fitted offset on GA -- 'DO' one slope per configuration, "
                    "'DO_shared' one slope for all" if _htc.DO_PROTOCOL else
                    "DO uses three corpus-only density features and a full OLS "
                    "fit, which is more generous than S5.1's 'simple fitted "
                    "offset'; S5.1's own definition (mean distance to a "
                    "landmark's k nearest corpus points) is not materialised "
                    "upstream"),
        "pairings": {},
    }
    rows, calib_rows, thr_rows, regret_rows = [], [], [], []
    boot_r = int(getattr(args, "boot_r", 0) or 0)
    boot_reps = {}

    for pairing in pairings:
        block = load_block(lay, pairing, configs, floor)
        if not block.configs:
            continue
        preds = all_predictions_on_c(lay, block, floor)
        missing_cost = [c for c in block.configs if c not in block.costs]

        print(f"\n  {pairing}   (C: {block.n_c:,})")
        print(f"    {'config':<12} " +
              " ".join(f"{p:>16}" for p in PREDICTORS))
        print(f"    {'':<12} " +
              " ".join(f"{'MAE  skill':>16}" for _ in PREDICTORS))

        per_config = {}
        cw = CellWeights(block.geom_c["assign"], R=boot_r) if boot_r else None
        for cfg in block.configs:
            true = block.cf_c[cfg]
            entry = {"family": block.families[cfg], "ga": block.ga[cfg],
                     "true_mean": float(true.mean()),
                     "true_std": float(true.std()),
                     "cost_seconds_per_query": block.costs.get(cfg)}
            line = []
            for name in PREDICTORS:
                p = preds[name][cfg]
                m = regression_metrics(p, true, ga=block.ga[cfg])
                m["calibration"] = calibration_curve(p, true)
                m["thresholds"] = {str(t): threshold_metrics(p, true, t)
                                   for t in TAUS}
                if name in KEYINGS:
                    m["backoff_rate"] = float(preds[f"backoff_{name}"][cfg].mean())
                entry[name] = m
                line.append(f"{fmt(m['MAE'],4):>7} {fmt(m['skill'],4):>8}")

                for i in range(len(m["calibration"]["bin"])):
                    calib_rows.append([
                        pairing, cfg, name, m["calibration"]["bin"][i],
                        m["calibration"]["n"][i],
                        f"{m['calibration']['pred_mean'][i]:.5f}",
                        f"{m['calibration']['true_mean'][i]:.5f}",
                        f"{m['calibration']['true_std'][i]:.5f}"])
                for t in TAUS:
                    th = m["thresholds"][str(t)]
                    thr_rows.append([pairing, cfg, name, t,
                                     f"{th['accuracy']:.4f}",
                                     f"{th['precision']:.4f}",
                                     f"{th['recall']:.4f}",
                                     f"{th['base_rate']:.4f}"])

            entry["flat"] = bool(true.std() < FLAT_SD)
            if cw is not None:
                # draft l.478: 95% CIs, resampling cells; l.482: Wilcoxon on the
                # per-query squared error (map vs GA in RQ2, keyed vs shared in RQ3)
                ga_pred = preds["GA"][cfg]
                for name in PREDICTORS:
                    if name == "GA":
                        continue
                    reps = skill_replicates(cw, preds[name][cfg], ga_pred, true)
                    entry[name]["skill_ci"] = boot_ci(reps, entry[name]["skill"])
                    boot_reps[f"{pairing}|{name}|{cfg}"] = reps
                se = {n_: (preds[n_][cfg] - true) ** 2 for n_ in ("GA", "SM_config", "SM_shared")}
                entry["tests"] = {
                    "SM_config_vs_GA": wilcoxon_paired(se["SM_config"], se["GA"]),
                    "SM_config_vs_SM_shared": wilcoxon_paired(se["SM_config"], se["SM_shared"]),
                }
                entry["keyed_minus_shared_skill_ci"] = boot_ci(
                    boot_reps[f"{pairing}|SM_config|{cfg}"] - boot_reps[f"{pairing}|SM_shared|{cfg}"],
                    entry["SM_config"]["skill"] - entry["SM_shared"]["skill"])
            entry["SM_vs_GA_mae_drop"] = float(
                1.0 - entry["SM_config"]["MAE"] / entry["GA"]["MAE"]
            ) if entry["GA"]["MAE"] > 0 else float("nan")
            entry["SM_vs_DO_mae_drop"] = float(
                1.0 - entry["SM_config"]["MAE"] / entry["DO"]["MAE"]
            ) if entry["DO"]["MAE"] > 0 else float("nan")
            per_config[cfg] = entry

            rows.append([pairing, cfg, block.families[cfg],
                         f"{true.mean():.4f}", f"{true.std():.4f}"]
                        + [f"{entry[p][m]:.4f}" for p in PREDICTORS
                           for m in ("MAE", "spearman", "skill")]
                        + [f"{entry['SM_config'].get('backoff_rate', 0.0):.4f}",
                           f"{entry['SM_vs_GA_mae_drop']:.4f}",
                           f"{entry['SM_vs_DO_mae_drop']:.4f}"])
            print(f"    {cfg:<12} " + " ".join(line))

        # ---- the cross-configuration decision: regret at each tau ----------
        regret = {}
        for name in PREDICTORS:
            regret[name] = {
                str(t): realized_regret(preds[name], block.cf_c, block.costs, t,
                                        configs=block.configs)
                for t in TAUS}
            for t in TAUS:
                r = regret[name][str(t)]
                if r.get("n_configs"):
                    regret_rows.append([
                        pairing, name, t, f"{r['disagreement_rate']:.4f}",
                        f"{r['mean_regret']:.5f}", f"{r['mean_shortfall']:.5f}",
                        f"{r['p95_shortfall']:.5f}",
                        f"{r['mean_cf_map_pick']:.4f}",
                        f"{r['mean_cf_oracle_pick']:.4f}",
                        f"{r['cost_ratio_map_over_oracle']:.4f}",
                        f"{r['frac_escalated_map']:.4f}"])

        skills = {p: float(np.nanmean([per_config[c][p]["skill"]
                                       for c in per_config]))
                  for p in PREDICTORS}
        nonflat = [c for c in per_config if not per_config[c]["flat"]]
        mean_skill_nonflat = {p: float(np.nanmean([per_config[c][p]["skill"] for c in nonflat]))
                              if nonflat else float("nan") for p in PREDICTORS}
        mean_skill_nonflat_ci = {}
        if cw is not None and nonflat:
            for p_ in PREDICTORS:
                if p_ == "GA":
                    continue
                reps = np.nanmean(np.stack([boot_reps[f"{pairing}|{p_}|{c}"] for c in nonflat]), axis=0)
                boot_reps[f"{pairing}|{p_}|MEAN_NONFLAT"] = reps
                mean_skill_nonflat_ci[p_] = boot_ci(reps, mean_skill_nonflat[p_])
        beats_ga = skills["SM_config"] > GATE_SKILL_MARGIN
        beats_do = skills["SM_config"] > skills["DO"]
        n_beat = sum(per_config[c]["SM_config"]["skill"]
                     > per_config[c]["GA"]["skill"] + GATE_SKILL_MARGIN
                     for c in per_config)

        report["pairings"][pairing] = {
            "n_c": block.n_c, "same_modality": block.same_modality,
            "per_config": per_config,
            "mean_skill": skills,
            "mean_skill_nonflat": mean_skill_nonflat,
            "mean_skill_nonflat_ci": mean_skill_nonflat_ci,
            "flat_configs": [c for c in per_config if per_config[c]["flat"]],
            "flat_rule": f"true CF std on Q_test < {FLAT_SD} (draft l.449)",
            "n_configs": len(per_config),
            "n_configs_sm_beats_ga": int(n_beat),
            "sm_beats_ga": bool(beats_ga),
            "sm_beats_do": bool(beats_do),
            "regret": regret,
            "configs_without_measured_cost": missing_cost,
            "verdict": ("PASS" if beats_ga and beats_do else
                        "PASS-but-DO-ties-or-wins" if beats_ga else "FAIL"),
        }
        print(f"    -> mean skill  GA 0.0000  DO {skills['DO']:.4f}  "
              f"SM {skills['SM_config']:.4f}   "
              f"({n_beat}/{len(per_config)} configs clear the "
              f"{GATE_SKILL_MARGIN} margin)")
        for t in TAUS:
            g, s = regret["GA"][str(t)], regret["SM_config"][str(t)]
            if g.get("n_configs"):
                print(f"       tau={t}: shortfall  GA {g['mean_shortfall']:.4f} "
                      f"-> SM {s['mean_shortfall']:.4f}   "
                      f"disagreement {s['disagreement_rate']:.1%}   "
                      f"cost x{s['cost_ratio_map_over_oracle']:.2f} vs oracle")

    # ---- the gate ---------------------------------------------------------
    verdicts = {p: v["verdict"] for p, v in report["pairings"].items()}
    all_pass = bool(verdicts) and all(v.startswith("PASS") for v in verdicts.values())
    strict_pass = bool(verdicts) and all(v == "PASS" for v in verdicts.values())
    report["gate"] = {
        "verdicts_by_pairing": verdicts,
        "sm_beats_ga_everywhere": all_pass,
        "sm_beats_ga_and_do_everywhere": strict_pass,
        "decision": ("GO" if strict_pass else
                     "GO-WITH-CAVEAT" if all_pass else "NO-GO"),
        "reading": ("SM clears GA by the required margin in every modality "
                    "configuration; the premise is supported at 1M and the "
                    "pilot may scale to 7M."
                    if strict_pass else
                    "SM clears GA everywhere but does not clearly separate from "
                    "DO in at least one pairing: location helps, but generic "
                    "corpus density may account for much of it, which weakens "
                    "the case for the per-configuration labelling pass."
                    if all_pass else
                    "SM does not clearly beat GA in at least one modality "
                    "configuration. Per draft S9 the conclusion is that H0 does "
                    "not hold at this scale or corpus -- NOT that the map needs "
                    "more tuning. Do not proceed to 7M on this evidence."),
    }

    write_csv(dest / "h5_metrics.csv",
              ["pairing", "config", "family", "true_mean", "true_std"]
              + [f"{p}_{m}" for p in PREDICTORS
                 for m in ("MAE", "spearman", "skill")]
              + ["SM_backoff_rate", "SM_vs_GA_mae_drop", "SM_vs_DO_mae_drop"],
              rows)
    write_csv(dest / "h5_calibration.csv",
              ["pairing", "config", "predictor", "bin", "n", "pred_mean",
               "true_mean", "true_std"], calib_rows)
    write_csv(dest / "h5_threshold_decisions.csv",
              ["pairing", "config", "predictor", "tau", "accuracy", "precision",
               "recall", "base_rate"], thr_rows)
    write_csv(dest / "h5_regret.csv",
              ["pairing", "predictor", "tau", "disagreement_rate", "mean_regret",
               "mean_shortfall", "p95_shortfall", "mean_cf_map_pick",
               "mean_cf_oracle_pick", "cost_ratio_vs_oracle", "frac_escalated"],
              regret_rows)
    if boot_reps:
        np.savez(dest / "h5_boot_replicates.npz", **boot_reps)
        report["bootstrap"] = {"R": boot_r, "resample": "cells (draft l.478)",
                               "replicates_file": "h5_boot_replicates.npz"}
    write_json(dest / "h5_report.json", report)
    _plot(report, dest / "figures")

    print(f"\n  GATE: {report['gate']['decision']}")
    print(f"  {report['gate']['reading']}")
    print(f"\n  wrote {dest / 'h5_report.json'}")
    return report


def _plot(report: dict, figdir) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir.mkdir(parents=True, exist_ok=True)
    pairings = list(report["pairings"])
    if not pairings:
        return

    # ---- skill ladder -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(2.0 * len(pairings) + 3.6, 4.2))
    x = np.arange(len(pairings))
    colors = {"GA": "#d9d9d9", "DO": "#fdae6b", "DO_shared": "#fdd0a2", "SM_shared": "#c6dbef",
              "SM_shared_scaled": "#9ecae1", "SM_family": "#6baed6",
              "SM_config": "#08519c"}
    for i, p in enumerate(PREDICTORS):
        vals = [report["pairings"][q]["mean_skill"][p] for q in pairings]
        w = 0.84 / len(PREDICTORS)
        ax.bar(x + (i - (len(PREDICTORS) - 1) / 2) * w, vals, w, label=p, color=colors[p])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("_query_", " q / ").replace("_index", " idx")
                        for p in pairings], rotation=15, fontsize=8)
    ax.set_ylabel("mean skill vs GA")
    ax.set_title(f"H5 -- the gate.  decision: {report['gate']['decision']}",
                 fontsize=11)
    ax.legend(fontsize=7, ncol=6)
    fig.tight_layout()
    fig.savefig(figdir / "h5_skill_ladder.png", dpi=140)
    plt.close(fig)

    # ---- calibration ------------------------------------------------------
    fig, axes = plt.subplots(1, len(pairings), figsize=(4.4 * len(pairings), 4.2),
                             squeeze=False)
    for ax, pairing in zip(axes[0], pairings):
        pc = report["pairings"][pairing]["per_config"]
        cmap = plt.get_cmap("viridis")
        for i, (cfg, v) in enumerate(pc.items()):
            cal = v["SM_config"]["calibration"]
            ax.plot(cal["pred_mean"], cal["true_mean"], marker="o", ms=3, lw=1.1,
                    color=cmap(i / max(len(pc) - 1, 1)), label=cfg)
        ax.plot([0, 1], [0, 1], "k--", lw=0.9)
        ax.set_xlabel("predicted CF@100 (binned)", fontsize=8)
        ax.set_ylabel("observed CF@100", fontsize=8)
        ax.set_title(pairing.replace("_query_", " q / ").replace("_index", " idx"),
                     fontsize=9)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, ncol=2)
    fig.suptitle("H5 -- calibration of SM on held-out C. On the dashed line = "
                 "the predicted number is believable, not only well ordered.",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(figdir / "h5_calibration.png", dpi=140)
    plt.close(fig)

    # ---- regret vs tau ----------------------------------------------------
    fig, axes = plt.subplots(1, len(pairings), figsize=(4.0 * len(pairings), 3.8),
                             squeeze=False)
    for ax, pairing in zip(axes[0], pairings):
        reg = report["pairings"][pairing]["regret"]
        for name, color in (("GA", "#d9d9d9"), ("DO", "#fdae6b"),
                            ("SM_config", "#08519c")):
            xs, ys = [], []
            for t in TAUS:
                r = reg[name][str(t)]
                if r.get("n_configs"):
                    xs.append(t)
                    ys.append(r["mean_shortfall"])
            ax.plot(xs, ys, marker="o", lw=1.4, color=color, label=name)
        ax.set_xlabel("fidelity floor  tau", fontsize=8)
        ax.set_ylabel("mean realized CF shortfall", fontsize=8)
        ax.set_title(pairing.replace("_query_", " q / ").replace("_index", " idx"),
                     fontsize=9)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)
    fig.suptitle("H5 -- what a wrong pick actually costs, under the "
                 "cheapest-clearing policy", fontsize=11)
    fig.tight_layout()
    fig.savefig(figdir / "h5_regret_vs_tau.png", dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    args = ap.parse_args()
    print(f"H5  model={args.model}  SM vs GA vs DO -- the go/no-go gate")
    run(args)


if __name__ == "__main__":
    main()
