"""
H5 preview -- SM vs GA vs DO on the held-out C queries.

    python compare_baselines.py --model siglip2 --spherical --cf-dir result/cf/1M_preview7

This is T10's experiment, run here because T9 has everything it needs on disk
and the answer is the first real check on whether the map is worth anything.
T10 owns the full version (calibration curves, H1/H2 splits, the K sweep); this
is the headline comparison only.

THE THREE PREDICTORS
--------------------
GA   The config's mean CF over B. One number for every query. R^2 = 0 by
     construction -- the bar any location-aware signal must clear.

DO   Landmark-keyed but label-free in its shape: a corpus-geometry density
     score per landmark, mapped into CF units by a fit on B. Paper S5.1 defines
     the score as "mean distance to its k nearest corpus points", which no
     upstream artifact currently holds, so this uses three corpus-only
     quantities T5 did store:

         log10 cell population        (cell_counts_A)
         mean member distance         (nn_A: mean d1 of the cell's A points)
         cell tightness               (cell_tightness_A, spherical fits only)

     Fitted per config by ordinary least squares on B. That is MORE generous
     than the paper's "simple fitted offset" -- three features and a full
     linear fit rather than one offset -- deliberately, so that SM beating DO
     is not an artifact of handicapping DO.

SM   The map: the cell's mean CF over B, with backoff to GA for cells holding
     fewer than the floor. Cells with no B queries at all also fall back to GA;
     they are counted separately in the report.

EVALUATION
----------
All three are fitted on B and scored on C, which the map never saw. Metrics are
S5.7's: MAE and Spearman, plus the threshold-decision accuracy that S5.7 argues
is the metric that actually matters, swept over tau in {0.7, 0.8, 0.9}.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from map_config import (                                               # noqa: E402
    DEFAULT_CF, DEFAULT_KMEANS, DEFAULT_OUT, DEFAULT_PILOT, PAIRINGS,
    layout, load_query_geometry, load_role_ids, resolve_configs,
    resolve_pairings, write_json,
)

TAUS = (0.7, 0.8, 0.9)


def density_features(lay, landmark_modality: str, K: int) -> np.ndarray:
    """
    Corpus-only per-landmark density, standardised. No CF labels touched.

    Column 0 is log population, column 1 the mean distance from a cell's corpus
    members to its centroid (spread, i.e. inverse density per unit volume), and
    column 2 cell tightness where T5 stored it.
    """
    tile = lay.tiling(landmark_modality)
    counts = np.load(tile / "cell_counts_A.npy").astype(np.float64)

    z = np.load(tile / "nn_A.npz")
    a_assign = z["assign"]
    # nn_A holds SQUARED L2, as everywhere in T5.
    a_d1 = np.sqrt(np.maximum(z["d1"].astype(np.float64), 0.0))
    n = np.bincount(a_assign, minlength=K).astype(np.float64)
    spread = np.zeros(K)
    nz = n > 0
    spread[nz] = np.bincount(a_assign, weights=a_d1, minlength=K)[nz] / n[nz]

    cols = [np.log10(np.maximum(counts, 1.0)), spread]
    tpath = tile / "cell_tightness_A.npy"
    if tpath.exists():
        cols.append(np.load(tpath).astype(np.float64))

    X = np.column_stack(cols)
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd < 1e-12] = 1.0
    return (X - mu) / sd


def fit_do(X_cells: np.ndarray, assign_b: np.ndarray, cf_b: np.ndarray) -> np.ndarray:
    """OLS of B's CF on its landmark's density features, with an intercept.

    The intercept alone reproduces GA exactly, so DO can never do worse than GA
    on the data it was fitted to -- which is the point: it isolates whether
    label-free density adds anything over the global mean."""
    A = np.column_stack([np.ones(len(assign_b)), X_cells[assign_b]])
    coef, *_ = np.linalg.lstsq(A, cf_b.astype(np.float64), rcond=None)
    return coef


# --------------------------------------------------------------------------
# DO as the protocol defines it (spatial-map-protocol.tex l.108; user 2026-10-19)
# --------------------------------------------------------------------------
DO_NN = 100


def density_score(lay, landmark_modality: str, K: int, nn: int = DO_NN) -> np.ndarray:
    """Per-landmark mean L2 distance to its nn nearest corpus vectors
    (landmarks/landmark_density.py). Label-free."""
    f = lay.tiling(landmark_modality) / f"density_A_k{nn}.npy"
    if not f.exists():
        raise FileNotFoundError(f"{f} missing -- run landmarks/landmark_density.py first")
    s = np.load(f).astype(np.float64)
    if s.shape != (K,):
        raise ValueError(f"{f}: shape {s.shape}, expected ({K},)")
    return s


def fit_do_offsets(score_cells: np.ndarray, assign_b: np.ndarray,
                   cf_b: dict, ga: dict) -> tuple:
    """GA_j + offset(z), the protocol's 'simple fitted offset'. z is the cell's
    density score standardised over the B queries (so the offset averages to 0
    over B and DO keeps each configuration's GA level). Fitted on B by least
    squares, in two forms (user, 2026-10-19: report both):
      per-config   pred_j = GA_j + beta_j * z      (one slope per configuration)
      shared       pred_j = GA_j + beta   * z      (one slope for all of them)
    Returns (z_cells, {config: beta_j}, beta_shared)."""
    zq = score_cells[assign_b]
    mu, sd = float(zq.mean()), float(zq.std())
    sd = sd if sd > 1e-12 else 1.0
    z_cells = (score_cells - mu) / sd
    zb = z_cells[assign_b]
    zz = float((zb * zb).sum())
    beta = {c: float((zb * (cf_b[c] - ga[c])).sum() / zz) for c in cf_b}
    beta_shared = float(sum((zb * (cf_b[c] - ga[c])).sum() for c in cf_b) / (len(cf_b) * zz))
    return z_cells, beta, beta_shared


def metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    err = pred - true
    out = {
        "MAE": float(np.abs(err).mean()),
        "RMSE": float(np.sqrt((err ** 2).mean())),
        "bias": float(err.mean()),
    }
    # Spearman is undefined if a predictor is constant, which GA always is.
    if np.ptp(pred) > 0:
        out["spearman"] = float(spearmanr(pred, true).statistic)
    else:
        out["spearman"] = float("nan")
    for t in TAUS:
        out[f"acc@{t}"] = float(((pred >= t) == (true >= t)).mean())
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="siglip2")
    ap.add_argument("--k", type=int, default=1024)
    ap.add_argument("--spherical", action="store_true")
    ap.add_argument("--pairing", nargs="+", default=["all"])
    ap.add_argument("--configs", default="all")
    ap.add_argument("--floor", type=int, default=None,
                    help="SM backoff floor; defaults to the map's primary floor")
    ap.add_argument("--pilot-dir", type=Path, default=DEFAULT_PILOT)
    ap.add_argument("--kmeans-dir", type=Path, default=DEFAULT_KMEANS)
    ap.add_argument("--cf-dir", type=Path, default=DEFAULT_CF)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--do-protocol", action="store_true",
                    help="v2: DO = the protocol's density score (mean distance to the "
                         "100 nearest corpus vectors) + offset on GA; adds DO_shared")
    args = ap.parse_args()

    lay = layout(args.model, args.k, args.spherical, pilot=args.pilot_dir,
                 kmeans=args.kmeans_dir, cf=args.cf_dir, out=args.out_dir)
    meta = json.load(open(lay.dest / "map_meta.json"))
    floor = args.floor if args.floor is not None else int(meta["reliability_floor"])
    pairings = [p for p in resolve_pairings(args.pairing) if p in meta["pairings"]]

    b_ids, c_ids = load_role_ids(lay, "B"), load_role_ids(lay, "C")
    print(f"H5 preview  model={lay.model}  tiling={lay.variant}  floor={floor}")
    print(f"  fitted on B ({len(b_ids):,}), scored on held-out C ({len(c_ids):,})\n")

    report = {"created": datetime.now().isoformat(), "model": lay.model,
              "K": lay.k, "spherical": lay.spherical, "floor": floor,
              "taus": list(TAUS), "cf_source": str(lay.cf_dir()),
              "do_note": ("DO per the protocol: mean distance to the 100 nearest corpus "
                          "vectors per landmark, as an offset on GA; 'DO' one slope per "
                          "config, 'DO_shared' one slope for all" if args.do_protocol else
                          "density features substitute for S5.1's 'mean distance "
                          "to k nearest corpus points', which is not on disk"),
              "pairings": {}}

    for pairing in pairings:
        spec = PAIRINGS[pairing]
        K = int(meta["pairings"][pairing]["K"])
        configs = [c for c in resolve_configs(args.model, args.configs)
                   if c in meta["pairings"][pairing]["configs"]]

        gb = load_query_geometry(lay, pairing, "B")
        gc = load_query_geometry(lay, pairing, "C")
        X = None if args.do_protocol else density_features(lay, spec["index_modality"], K)
        if args.do_protocol:
            have = [c for c in configs if lay.cf_vector(pairing, "B", c).exists()]
            cfb_all = {c: np.load(lay.cf_vector(pairing, "B", c)).astype(np.float64) for c in have}
            z_cells, beta, beta_sh = fit_do_offsets(
                density_score(lay, spec["index_modality"], K), gb["assign"],
                cfb_all, {c: float(v.mean()) for c, v in cfb_all.items()})

        stats = np.load(lay.pairing_dir(pairing) / "cell_stats.npz",
                        allow_pickle=False)
        cfg_index = {c: i for i, c in enumerate(stats["configs"].tolist())}

        print(f"  {pairing}   (C queries: {len(c_ids):,})")
        print(f"    {'config':12s} {'GA':>18s} {'DO':>18s} {'SM':>18s}   "
              f"{'SM vs GA':>9s}")
        print(f"    {'':12s} {'MAE   rho':>18s} {'MAE   rho':>18s} "
              f"{'MAE   rho':>18s}   {'MAE drop':>9s}")

        per_config = {}
        for cfg in configs:
            fb = lay.cf_vector(pairing, "B", cfg)
            fc = lay.cf_vector(pairing, "C", cfg)
            if not (fb.exists() and fc.exists()):
                continue
            cf_b = np.load(fb).astype(np.float64)
            cf_c = np.load(fc).astype(np.float64)
            ab, ac = gb["assign"][:len(cf_b)], gc["assign"][:len(cf_c)]

            # ---- GA ----
            ga = float(cf_b.mean())
            p_ga = np.full(len(cf_c), ga)

            # ---- DO ----
            if args.do_protocol:
                p_do = np.clip(ga + beta[cfg] * z_cells[ac], 0.0, 1.0)
                p_do_sh = np.clip(ga + beta_sh * z_cells[ac], 0.0, 1.0)
            else:
                coef = fit_do(X, ab, cf_b)
                p_do = np.clip(
                    np.column_stack([np.ones(len(ac)), X[ac]]) @ coef, 0.0, 1.0)

            # ---- SM ----
            i = cfg_index[cfg]
            cell_mean = stats["cf_mean"][i]
            cell_n = stats["n"][i]
            p_sm = cell_mean[ac].astype(np.float64)
            # Two distinct reasons to back off, counted separately: a cell too
            # thin to trust, and a cell no B query ever reached (nan mean).
            thin = cell_n[ac] < floor
            unseen = ~np.isfinite(p_sm)
            p_sm = np.where(thin | unseen, ga, p_sm)

            m_ga, m_do, m_sm = (metrics(p, cf_c) for p in (p_ga, p_do, p_sm))
            drop = (m_ga["MAE"] - m_sm["MAE"]) / m_ga["MAE"] if m_ga["MAE"] else 0.0

            per_config[cfg] = {
                "cf_c_mean": float(cf_c.mean()), "ga": ga,
                "frac_backed_off": float((thin | unseen).mean()),
                "frac_unseen_cell": float(unseen.mean()),
                "GA": m_ga, "DO": m_do, "SM": m_sm,
                "SM_vs_GA_mae_drop": float(drop),
                "SM_vs_DO_mae_drop": float(
                    (m_do["MAE"] - m_sm["MAE"]) / m_do["MAE"]) if m_do["MAE"] else 0.0,
            }
            if args.do_protocol:
                per_config[cfg]["DO_shared"] = metrics(p_do_sh, cf_c)
            print(f"    {cfg:12s} {m_ga['MAE']:6.4f} {m_ga['spearman']:6.3f}"
                  f"   {m_do['MAE']:6.4f} {m_do['spearman']:6.3f}"
                  f"   {m_sm['MAE']:6.4f} {m_sm['spearman']:6.3f}"
                  f"   {drop:+8.1%}")

        report["pairings"][pairing] = per_config
        print()

    write_json(lay.dest / "h5_preview.json", report)

    # ---- headline ----
    print("  " + "-" * 70)
    print(f"  {'pairing':24s} {'GA MAE':>8s} {'DO MAE':>8s} {'SM MAE':>8s} "
          f"{'SM vs GA':>9s} {'SM vs DO':>9s}")
    for pairing, pc in report["pairings"].items():
        if not pc:
            continue
        g = np.mean([v["GA"]["MAE"] for v in pc.values()])
        d = np.mean([v["DO"]["MAE"] for v in pc.values()])
        s = np.mean([v["SM"]["MAE"] for v in pc.values()])
        print(f"  {pairing:24s} {g:8.4f} {d:8.4f} {s:8.4f} "
              f"{(g - s) / g:+9.1%} {(d - s) / d:+9.1%}"
              + (f"   DO_shared MAE {np.mean([v['DO_shared']['MAE'] for v in pc.values()]):.4f}"
                 if args.do_protocol else ""))
    print(f"\n  wrote {lay.dest / 'h5_preview.json'}")


if __name__ == "__main__":
    main()
