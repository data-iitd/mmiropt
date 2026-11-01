"""
H1 -- error-versus-distance diagnostic.

    Plot SM's prediction error against q's distance to l(q), and separately
    against the gap between distances to the first- and second-nearest
    landmarks (a proxy for boundary proximity). P1 predicts error should grow
    with both quantities; a flat relationship would indicate the map is not
    capturing meaningful spatial structure at all, regardless of its aggregate
    accuracy.                                                    (draft S5.2)

WHAT THIS TEST IS FOR
---------------------
H5 can come out positive for the wrong reason. If the map beats GA because
cells happen to line up with something unrelated -- a lucky split, an artefact
of how the corpus was assembled -- the win is real but unexplained, and an
unexplained win is fragile: there is no reason to expect it to survive another
model, another corpus, or the 7M scale. P1 makes a specific physical claim
(fidelity varies SMOOTHLY across the space, so a cell's average describes its
neighbourhood), and a real mechanism signs its own work by predicting when it
will FAIL, not only when it will succeed:

    far from the centre  -> the cell mean describes territory the query is not
                            in            -> error should rise with d1
    close to a wall      -> the query's true neighbourhood straddles two cells,
                            and the map forces it into one -> error should rise
                            as the query approaches a boundary

Nothing new is computed from indexes or embeddings. Per test query this joins
three things already on disk: SM's prediction (T9's map), the true CF (T8's
labels for C), and d1/d2 (T5's assignment, carried into T9's geometry files).

TWO PROXIES FOR BOUNDARY PROXIMITY, AND WHY BOTH
------------------------------------------------
``gap = d2 - d1`` is the draft's own wording. It is in the units of the
pairing's distances, and those units differ by roughly 3x between the
same-modal and cross-modal pairings, so a gap threshold does not mean the same
thing across the 2x2 -- and the draft's own figures show cross-modal gaps ~10x
smaller than same-modal ones for that reason alone.

``ratio = d1 / d2`` in (0, 1] is scale-free: 1.0 is exactly on the wall
regardless of how far away the wall is. Both are reported; the ratio is the one
H2 splits on. Where the two disagree, that disagreement IS the scale effect and
is worth reading rather than hiding.

SERVED VERSUS ALL
-----------------
Queries whose cell fell under the reliability floor are served GA, not a cell
mean. Their error therefore has nothing to do with distance-to-landmark by
construction, and leaving them in would dilute exactly the relationship being
tested -- toward flat, i.e. toward falsely rejecting P1. The headline is
computed on SERVED queries; the all-queries version is reported alongside so
the choice is visible rather than assumed.
"""
from __future__ import annotations

import argparse
from datetime import datetime

import numpy as np

from ht_config import (add_common_args, load_block, resolve_from_args,
                       write_csv, write_json)
from metrics import CellBootstrap, binned_profile, boot_ci, fmt, spearman_fast, trend
from predictors import sm_on_c

BINS = 10


def _boot_one(job):
    """One config's cell-bootstrap CIs of Spearman(error, d1) and (error, d1/d2)."""
    cells, d1, ratio, err, R = job
    cb = CellBootstrap(cells, R=R)
    reps = {"d1": [], "ratio": []}
    for idx in cb.draws():
        reps["d1"].append(spearman_fast(d1[idx], err[idx]))
        reps["ratio"].append(spearman_fast(ratio[idx], err[idx]))
    return {ax: boot_ci(reps[ax], spearman_fast({"d1": d1, "ratio": ratio}[ax], err))
            for ax in ("d1", "ratio")} | {"n_cells": cb.n_cells}


def run(args) -> dict:
    lay, ht, floor, pairings, configs, meta = resolve_from_args(args)
    dest = ht.ensure("H1")

    report = {
        "created": datetime.now().isoformat(),
        "hypothesis": "H1",
        "question": "does SM's prediction error grow with distance-to-landmark "
                    "and with boundary proximity, as P1 predicts",
        "model": lay.model, "tiling": lay.variant, "floor": floor,
        "measured_on": "C (held-out)",
        "error": "absolute error |SM prediction - true CF@100|",
        "boundary_proxies": {
            "gap": "d2 - d1, the draft's wording; in the pairing's own distance units",
            "ratio": "d1 / d2 in (0,1]; scale-free, 1.0 = on the cell wall",
        },
        "population": "served queries only (cell population >= floor); the "
                      "all-queries variant is reported as *_all",
        "pass_rule": ("P1 supported for a block when the 95% cell-resampled bootstrap "
                      f"CI (R = {args.boot_r}) of Spearman(error, d1) AND of "
                      "Spearman(error, d1/d2) both lie above 0 (user, 2026-10-19; "
                      "queries in one cell share an estimate, so per-query p-values "
                      "overstate the evidence)" if getattr(args, "boot_r", 0) else
                      "P1 supported for a block when Spearman(error, d1) > 0 and "
                      "Spearman(error, ratio) > 0, both with p < 0.01"),
        "pairings": {},
    }
    trend_rows, profile_rows = [], []

    for pairing in pairings:
        block = load_block(lay, pairing, configs, floor)
        if not block.configs:
            continue
        d1 = block.geom_c["d1"].astype(np.float64)
        gap = block.geom_c["gap"].astype(np.float64)
        ratio = block.ratio("C")

        print(f"\n  {pairing}   (C: {block.n_c:,}, d1 mean {d1.mean():.3f}, "
              f"gap mean {gap.mean():.4f})")
        print(f"    {'config':<12} {'served':>7} | {'rho(e,d1)':>10} {'slope':>8} | "
              f"{'rho(e,ratio)':>12} {'slope':>8} | {'rho(e,-gap)':>11} | P1")

        per_config = {}
        boot = {}
        if getattr(args, "boot_r", 0):
            from multiprocessing import Pool
            jobs, names = [], []
            for cfg in block.configs:
                pred, backed = sm_on_c(block, cfg, floor)
                s_ = ~backed
                err_ = np.abs(pred - block.cf_c[cfg])
                jobs.append((block.geom_c["assign"][s_], d1[s_], ratio[s_], err_[s_], args.boot_r))
                names.append(cfg)
            with Pool(min(len(jobs), 9)) as pool:
                boot = dict(zip(names, pool.map(_boot_one, jobs)))
        for cfg in block.configs:
            pred, backed = sm_on_c(block, cfg, floor)
            err = np.abs(pred - block.cf_c[cfg])
            served = ~backed

            def block_trends(mask):
                if mask.sum() < 10:
                    return None
                return {
                    "d1": trend(d1[mask], err[mask]),
                    "ratio": trend(ratio[mask], err[mask]),
                    # Negated so the sign convention matches the others: a
                    # POSITIVE number always means "error rises as the query
                    # gets closer to a boundary".
                    "neg_gap": trend(-gap[mask], err[mask]),
                }

            t_served = block_trends(served)
            t_all = block_trends(np.ones_like(served, dtype=bool))
            if t_served is None:
                continue

            passes_p = bool(
                t_served["d1"]["spearman"] > 0 and t_served["d1"]["p_value"] < 0.01
                and t_served["ratio"]["spearman"] > 0
                and t_served["ratio"]["p_value"] < 0.01)
            b = boot.get(cfg)
            passes = (bool(b["d1"]["lo"] > 0 and b["ratio"]["lo"] > 0) if b is not None
                      else passes_p)

            prof_d1 = binned_profile(d1[served], err[served], BINS)
            prof_ratio = binned_profile(ratio[served], err[served], BINS)
            per_config[cfg] = {
                "family": block.families[cfg],
                "n_served": int(served.sum()),
                "served_fraction": float(served.mean()),
                "mean_abs_error_served": float(err[served].mean()),
                "trend_served": t_served, "trend_all": t_all,
                "profile_d1": prof_d1, "profile_ratio": prof_ratio,
                "p1_supported": passes,
                "p1_supported_by_per_query_p": passes_p,
                "cell_bootstrap": b,
            }
            trend_rows.append([
                pairing, cfg, block.families[cfg], int(served.sum()),
                f"{err[served].mean():.4f}",
                f"{t_served['d1']['spearman']:.4f}",
                f"{t_served['d1']['p_value']:.3e}",
                f"{t_served['d1']['slope_per_sd']:.5f}",
                f"{t_served['ratio']['spearman']:.4f}",
                f"{t_served['ratio']['p_value']:.3e}",
                f"{t_served['ratio']['slope_per_sd']:.5f}",
                f"{t_served['neg_gap']['spearman']:.4f}",
                "PASS" if passes else "FAIL",
                "" if b is None else f"{b['d1']['lo']:.4f}",
                "" if b is None else f"{b['d1']['hi']:.4f}",
                "" if b is None else f"{b['ratio']['lo']:.4f}",
                "" if b is None else f"{b['ratio']['hi']:.4f}",
                "" if b is None else b["n_cells"]])
            for axis, prof in (("d1", prof_d1), ("ratio", prof_ratio)):
                for i in range(len(prof["bin"])):
                    profile_rows.append([
                        pairing, cfg, axis, prof["bin"][i], prof["n"][i],
                        f"{prof['x_mean'][i]:.5f}", f"{prof['y_mean'][i]:.5f}",
                        f"{prof['y_sem'][i]:.5f}"])
            print(f"    {cfg:<12} {served.mean():>6.1%} | "
                  f"{fmt(t_served['d1']['spearman'],4):>10} "
                  f"{fmt(t_served['d1']['slope_per_sd'],5):>8} | "
                  f"{fmt(t_served['ratio']['spearman'],4):>12} "
                  f"{fmt(t_served['ratio']['slope_per_sd'],5):>8} | "
                  f"{fmt(t_served['neg_gap']['spearman'],4):>11} | "
                  f"{'PASS' if passes else 'FAIL'}")

        n_pass = sum(v["p1_supported"] for v in per_config.values())
        report["pairings"][pairing] = {
            "n_c": block.n_c, "same_modality": block.same_modality,
            "d1_mean": float(d1.mean()), "gap_mean": float(gap.mean()),
            "ratio_mean": float(ratio.mean()),
            "per_config": per_config,
            "n_configs": len(per_config), "n_p1_supported": n_pass,
            "verdict": ("supported" if n_pass == len(per_config) and per_config
                        else "mixed" if n_pass else "not-supported"),
        }

    write_csv(dest / "h1_trends.csv",
              ["pairing", "config", "family", "n_served", "mean_abs_error",
               "rho_err_d1", "p_d1", "slope_d1_per_sd", "rho_err_ratio",
               "p_ratio", "slope_ratio_per_sd", "rho_err_neg_gap", "p1",
               "ci_d1_lo", "ci_d1_hi", "ci_ratio_lo", "ci_ratio_hi", "n_cells"],
              trend_rows)
    write_csv(dest / "h1_binned_profiles.csv",
              ["pairing", "config", "axis", "bin", "n", "x_mean",
               "error_mean", "error_sem"], profile_rows)
    write_json(dest / "h1_report.json", report)
    _plot(report, dest / "figures")
    print(f"\n  wrote {dest / 'h1_report.json'}")
    return report


def _plot(report: dict, figdir) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir.mkdir(parents=True, exist_ok=True)
    pairings = list(report["pairings"])
    for axis, xlabel in (("profile_d1", "distance to nearest landmark  d1"),
                         ("profile_ratio", "boundary proximity  d1 / d2")):
        if not pairings:
            return
        fig, axes = plt.subplots(1, len(pairings),
                                 figsize=(4.4 * len(pairings), 4.0), squeeze=False)
        for ax, pairing in zip(axes[0], pairings):
            pc = report["pairings"][pairing]["per_config"]
            cmap = plt.get_cmap("viridis")
            for i, (cfg, v) in enumerate(pc.items()):
                prof = v[axis]
                ax.errorbar(prof["x_mean"], prof["y_mean"], yerr=prof["y_sem"],
                            marker="o", ms=3, lw=1.2, capsize=2,
                            color=cmap(i / max(len(pc) - 1, 1)), label=cfg)
            ax.set_xlabel(xlabel, fontsize=8)
            ax.set_ylabel("mean |prediction error|", fontsize=8)
            ax.set_title(pairing.replace("_query_", " q / ").replace("_index", " idx"),
                         fontsize=9)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=6, ncol=2)
        fig.suptitle("H1 -- P1 predicts these curves rise. Flat means the map "
                     "is not capturing spatial structure.", fontsize=11)
        fig.tight_layout()
        name = "h1_error_vs_d1.png" if axis == "profile_d1" else "h1_error_vs_boundary.png"
        fig.savefig(figdir / name, dpi=140)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    args = ap.parse_args()
    print(f"H1  model={args.model}  does error grow with distance and boundary "
          f"proximity?")
    run(args)


if __name__ == "__main__":
    main()
