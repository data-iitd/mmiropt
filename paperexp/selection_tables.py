"""
Paper experiments, part 1 of 2: per-query plan selection, from labels that exist.

    python paperexp/selection_tables.py --model siglip2        (inside a PBS job)

Fills the paper's RQ1 (headroom), RQ3 (rank flips), RQ4 (selection) and the
Rec./Viol. columns of RQ5's K sweep. RQ2, RQ3's keyed-vs-shared table and the
rest of RQ5 are read straight from T10's stored results by
writeup/7M_cleaned/make_paper_tables.py and are not recomputed here, except as
the cross-checks listed below.

NO SEARCH OF ANY KIND. Every input was written by an earlier task:

    T8   fidelity labels     result/cf/7M_cleaned/<model>/cf_<pairing>_{B,C}_<config>.npy
    T5   landmark assignment result/kmeans/7M_cleaned/<model>/<modality>/K*_spherical/
                             queries/nn_{B,C}_<query modality>q.npz
    cost controlled bench    result/throughput/7M_cleaned/<model>/throughput.json
    T9   served map (K=2500) result/spatial_map/7M_cleaned/<model>/K02500_spherical/
                             <pairing>/cell_stats.npz  -- cross-check only
    T10  H4 sweep            result/hypothesis_testing/7M_cleaned/<model>/H4/h4_sweep.csv
                             -- cross-check only

Output: result/paper_experiments/7M_cleaned/<model>/selection.json, and nothing
else. Role B is the paper's construction sample Q; role C is Q_test.

DEFINITIONS (paper notation; F = fidelity, k = 100)
---------------------------------------------------
Fidelity is handled as the integer overlap |Top-k_KNN ∩ Top-k_A| in 0..100, and
"A meets tau" is overlap >= round(100 tau). That is exact: no float32 value can
sit a rounding error below the target (the defect H5 had before 2026-09-29).

Configurations are ordered cheapest first by the controlled benchmark's seconds
per query. Ties in any argmax resolve to the cheaper configuration.

  static plan   chosen on Q: the cheapest A whose coverage over Q is >= rho;
                if none reaches rho, the A with the largest coverage.
                A map with K = 1 selects exactly this plan (checked below).
  optimum       per query in Q_test: the cheapest A with F(q,A) >= tau; if none,
                the A with the highest F(q,A) ("none" queries).
  cell tables   from Q at tiling K: n_i, mu_ij = mean F, pi_ij(tau) = share of
                the cell's queries meeting tau. A cell with n_i < q_min = 10
                serves the whole-sample values mu_bar_j, pi_bar_j instead.
  mean rule     cheapest A with F_hat(q,A) = mu_{l(q)j} >= tau, else argmax F_hat.
  coverage rule cheapest A with pi_hat(q,A;tau) >= rho, else argmax pi_hat.

  viol          share of Q_test whose chosen A has F < tau
  short         mean (tau - F) over those violated queries
  cost          mean seconds per query of the chosen A (search only; the
                landmark lookup c_est is added by the table generator from
                paperexp/bench_lookup.py's measurement)
  rec           (c_static - c) / (c_static - c_opt): share of the optimum's
                saving over the static plan that a selector recovers
  ECE           calibration of the coverage estimate: over every (query, A)
                pair in Q_test, pi_hat against the outcome 1[F >= tau], in 10
                equal-count bins -- hypothesis_testing/metrics.calibration_curve
  rank flips    over cells with n_i >= q_min at K = 2500: the share where
                mu_a - mu_b > 0.1 ("a wins") and where mu_b - mu_a > 0.1

CROSS-CHECKS, written into the output
-------------------------------------
  1. every stored fidelity is an exact multiple of 1/100
  2. at K = 2500 the rebuilt cell counts and means equal T9's cell_stats.npz
  3. at every K the rebuilt mean-estimate skill and backoff rate equal H4's
     h4_sweep.csv to 1e-4
  4. at K = 1 the coverage rule picks the static plan for every query
A failed check stops the run: numbers from a mismatched join are not written.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "spatialmap"))
sys.path.insert(0, str(REPO_ROOT / "indexes"))
sys.path.insert(0, str(REPO_ROOT / "hypothesis_testing"))

from map_config import (PAIRINGS, config_family, layout,             # noqa: E402
                        load_query_geometry, resolve_configs,
                        cf_histogram, coverage_from_histogram)
from metrics import calibration_curve                                  # noqa: E402
from metrics import (CellWeights, boot_ci, mcnemar,                    # noqa: E402
                     resampled_matched_optimum, skill_replicates, wilcoxon_paired)

SCALE = "7M_cleaned"
# The run's tree and K values. Defaults are v1 (result/, K = 2,500 main, the
# v1 sweep); the v2 jobs set SFM_RESULTS_ROOT=result_v2, SFM_KS=<v2 sweep> and
# SFM_K_MAIN=<the main K chosen from H4>. Every paperexp script that imports
# these names reads them from here, so one setting moves all of them.
R = REPO_ROOT / os.environ.get("SFM_RESULTS_ROOT", "result")
KS = tuple(int(k) for k in os.environ.get("SFM_KS", "1250,2500,5000,10000,20000").split(","))
K_MAIN = int(os.environ.get("SFM_K_MAIN", "2500"))
# Which landmark / map tree: SCALE by default; the seed robustness runs use
# e.g. SFM_TILING_TAG=7M_cleaned_seed1 (result_v2/kmeans/<tag>, spatial_map/<tag>).
TILING_TAG = os.environ.get("SFM_TILING_TAG", SCALE)
# Which T8 label tree: SCALE by default; the single-index catalog (main.tex
# sec:small-catalog) uses SFM_CF_TAG=7M_cleaned_single (result_v2/cf/<tag>).
CF_TAG = os.environ.get("SFM_CF_TAG", SCALE)
# The T9 map tree (spatial_map/<tag>) checked against at K_MAIN, if it differs from the
# tiling's: the fidelity@10 run (v3_k10, 2026-10-30) has its own map built from its own
# labels on the unchanged tiling, SFM_MAP_TAG=7M_cleaned_k10 with SFM_CF_TAG=7M_cleaned_k10.
MAP_TAG = os.environ.get("SFM_MAP_TAG", TILING_TAG)
# k of the stored fidelity, a label only: the overlap scale stays 0..100, and a
# fidelity@10 vector (multiples of 0.1) is exact on it.
FID_K = int(os.environ.get("SFM_FID_K", "100"))
# Where T10 wrote H4 (v2 writes one tree per map K): <dir>/<model>/H4/h4_sweep.csv
HT_DIR = Path(os.environ["SFM_HT_DIR"]) if os.environ.get("SFM_HT_DIR") else None
TAUS = (0.70, 0.80, 0.90)
RHOS = (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99)
Q_MIN = 10
FLIP_MARGIN = 0.10
ECE_BINS = 10
CHECK_TOL = 1e-4
FLAT_SD = 0.03        # draft l.449: excluded from skill-based averages
V2_STATS = False      # --v2-stats: set in main()
EST = None            # estimate.json (c_est), loaded in main() under --v2-stats
BOOT_R_V2 = 1000      # user, 2026-10-19
RHO_MAIN = 0.90       # draft l.435: the reliability level of the map and the static plan


def auc_score(score: np.ndarray, y: np.ndarray):
    """AUC of a score for a binary outcome (Mann-Whitney, ties averaged); None if
    every query meets tau or none does (draft l.457: such configurations excluded)."""
    from scipy.stats import rankdata
    y = np.asarray(y, bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return None
    r = rankdata(score)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ece_restricted(p: np.ndarray, y: np.ndarray, bins: int, lo: float = 0.05, hi: float = 0.95):
    """ECE over (query, configuration) pairs with lo < p < hi, equal-width bins
    (draft l.457); None if no pair is in the open interval."""
    keep = (p > lo) & (p < hi)
    if not keep.any():
        return None, 0
    p, y = p[keep], y[keep].astype(np.float64)
    b = np.minimum((p * bins).astype(np.int64), bins - 1)
    n = np.bincount(b, minlength=bins)
    gap = np.abs(np.bincount(b, weights=p, minlength=bins) - np.bincount(b, weights=y, minlength=bins))
    return float(gap.sum() / n.sum()), int(keep.sum())


def v2_stats(tau, t, order, cost, ov_c, f_c, cov_b, cells, pis, est, pairing, boot_r):
    """The draft's statistics at the main K (user decisions 2026-10-19): RQ2's
    coverage-estimate columns and RQ4's selection numbers, each with a 95%
    cell-resampled bootstrap CI, plus the draft's paired tests. Rec. is against
    the v2 matched optimum (misses charged the static configuration)."""
    from matched_violation import matched_optimum_static_charge
    n, _mu, fhat, _thin, _ab, ac = cells[K_MAIN]
    P = pis[K_MAIN]
    n_c, m = ov_c.shape
    rows = np.arange(n_c)
    y = ov_c >= t
    cw = CellWeights(ac, R=boot_r)
    out = {"K": K_MAIN, "tau": tau, "boot": {"R": boot_r, "resample": "cells of the main tiling"}}

    # ---- RQ2: coverage estimate (draft l.457) ----
    per_cfg, reps_ok = {}, []
    for j, c in enumerate(order):
        yj = y[:, j].astype(np.float64)
        pj = P[:, j]
        base = np.full(n_c, cov_b[j])
        bg = float(((base - yj) ** 2).mean())
        bss = 1.0 - float(((pj - yj) ** 2).mean()) / bg if bg > 0 else float("nan")
        reps = skill_replicates(cw, pj, base, yj)
        flat = bool(f_c[:, j].std() < FLAT_SD)
        per_cfg[c] = {"brier_skill": bss, "brier_skill_ci": boot_ci(reps, bss),
                      "auc": auc_score(pj, y[:, j]), "flat": flat}
        if not flat and np.isfinite(bss):
            reps_ok.append((bss, reps))
    ece, n_pairs = ece_restricted(P.ravel(), y.ravel(), ECE_BINS)
    aucs = [v["auc"] for v in per_cfg.values() if v["auc"] is not None and not v["flat"]]   # non-flat only (2026-10-29)
    out["coverage_estimate"] = {
        "per_config": per_cfg,
        "mean_brier_skill_nonflat": (float(np.mean([b for b, _ in reps_ok])) if reps_ok else None),
        "mean_brier_skill_nonflat_ci": (boot_ci(np.nanmean(np.stack([r for _, r in reps_ok]), axis=0),
                                                float(np.mean([b for b, _ in reps_ok])))
                                        if reps_ok else None),
        "mean_auc": float(np.mean(aucs)) if aucs else None,
        "n_auc_configs": len(aucs),
        "ece_restricted": ece, "ece_pairs": n_pairs, "ece_bins": ECE_BINS,
        "ece_rule": "pairs with 0.05 < pi_hat < 0.95, equal-width bins (draft l.457)",
    }

    # ---- RQ4: selection at rho = 0.9 (draft l.459-468) ----
    s, reached = static_plan(cov_b, RHO_MAIN)
    c_s = float(cost[s])
    cest = float(est["pairings"][pairing]["batched_steps_1_3"][str(K_MAIN)]["s_per_query"])
    ok = y
    none = ~ok.any(axis=1)
    first = np.argmax(ok, axis=1)
    cstar = np.where(none, c_s, cost[first])
    static_miss = ov_c[:, s] < t
    mo = matched_optimum_static_charge(ov_c, cost, t, int(static_miss.sum()), s)
    mo_r = resampled_matched_optimum(cw, cstar, none, static_miss, c_s) if mo["feasible"] else None
    nq = cw.totals(np.ones(n_c))
    midx, _ = pick(fhat >= tau - 1e-9, fhat)
    cidx, _ = pick(P >= RHO_MAIN - 1e-12, P)
    rules = {"static": np.full(n_c, s, dtype=np.int64), "mean_rule": midx, "coverage_rule": cidx}
    R_ = {"static": {"config": order[s], "reached_rho_on_Q": reached, "cost_s": c_s},
          "c_est_s": cest, "matched_optimum": mo, "rules": {}}
    shortfall = {}
    for name, idx in rules.items():
        f = ov_c[rows, idx]
        miss = f < t
        shortfall[name] = np.maximum(tau - f / 100.0, 0.0)
        add = 0.0 if name == "static" else cest
        cq = cost[idx]
        c_bar = float(cq.mean())
        rel = (add + c_bar) / c_s
        rel_r = (add + cw.totals(cq) / nq) / c_s
        viol = float(miss.mean())
        viol_r = cw.totals(miss) / nq
        e = {"cost_s_search": c_bar, "cost_s_total": add + c_bar,
             "relative_cost": rel, "relative_cost_ci": boot_ci(rel_r, rel),
             "violation_rate": viol, "violation_rate_ci": boot_ci(viol_r, viol),
             "shortfall_mean_violated": (float(shortfall[name][miss].mean()) if miss.any() else 0.0)}
        if name != "static" and mo["feasible"]:
            gap = c_s - mo["cost_s"]
            rec = (c_s - (add + c_bar)) / gap if gap > 1e-15 else None
            gap_r = c_s - mo_r
            with np.errstate(divide="ignore", invalid="ignore"):
                rec_r = np.where(gap_r > 1e-15, (c_s - (add + cw.totals(cq) / nq)) / gap_r, np.nan)
            e["rec_matched"] = rec
            e["rec_matched_ci"] = boot_ci(rec_r, rec) if rec is not None else None
            e["optimum_gap_share"] = gap / c_s      # the 5% rule is applied by the table generator
        if name != "static":
            e["mcnemar_vs_static"] = mcnemar(miss, static_miss)
        R_["rules"][name] = e
    R_["shortfall_coverage_vs_mean_wilcoxon_pratt"] = wilcoxon_paired(
        shortfall["coverage_rule"], shortfall["mean_rule"], zero_method="pratt")
    out["selection_rho_0.90"] = R_
    return out


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


# ==========================================================================
# inputs
# ==========================================================================
def lay_at(model: str, k: int):
    return layout(model, k, True, pilot=R / SCALE, kmeans=R / "kmeans" / TILING_TAG,
                  cf=R / "cf" / CF_TAG, out=R / "spatial_map" / MAP_TAG)


def bench_costs(model: str, cost_dir: Path | None = None) -> dict:
    """Seconds per query per (pairing, config), from a controlled-throughput tree.

    Default tree: result/throughput/7M_cleaned (FAISS 1.14.1 faiss-gpu-cu12, CPU
    code built without AVX2). result/throughput/7M_cleaned_avx2 holds the same
    benchmark under FAISS 1.15.1 faiss-cpu (AVX2); see result/RESULT_TREES.md.
    """
    path = (Path(cost_dir) if cost_dir else R / "throughput" / SCALE) / model / "throughput.json"
    doc = json.load(open(path))
    if not doc.get("complete"):
        raise RuntimeError(f"{path} is incomplete -- finish the benchmark first")
    return doc["cost_s_per_query"]


def load_overlap(lay, pairing: str, role: str, order: list, n_expected: int) -> np.ndarray:
    """[n, m] int16 overlap counts in `order` (cheapest first)."""
    cols = []
    for c in order:
        v = np.load(lay.cf_vector(pairing, role, c)).astype(np.float64)
        if len(v) != n_expected:
            raise ValueError(f"{pairing}/{role}/{c}: {len(v)} rows, role has {n_expected}")
        ov = np.rint(v * 100.0)
        worst = float(np.abs(ov / 100.0 - v).max())
        if worst > 1e-6:
            raise ValueError(f"{pairing}/{role}/{c}: fidelity is not a multiple of "
                             f"1/100 (off by {worst:g}) -- not a CF@100 vector")
        cols.append(ov.astype(np.int16))
    return np.column_stack(cols)


# ==========================================================================
# the selectors
# ==========================================================================
def pick(ok: np.ndarray, score: np.ndarray):
    """First (cheapest) column where ok; else the argmax of score (first max,
    i.e. the cheaper one on ties). Returns (index, none-mask)."""
    idx = np.argmax(ok, axis=1)
    none = ~ok.any(axis=1)
    if none.any():
        idx[none] = np.argmax(score[none], axis=1)
    return idx, none


def static_plan(coverage: np.ndarray, rho: float):
    ok = np.nonzero(coverage >= rho - 1e-12)[0]
    if len(ok):
        return int(ok[0]), True
    return int(np.argmax(coverage)), False


def cell_tables(assign_b: np.ndarray, ov_b: np.ndarray, K: int):
    """Cell counts and mean fidelity per (cell, config), from Q."""
    n = np.bincount(assign_b, minlength=K).astype(np.float64)
    nz = n > 0
    mu = np.zeros((K, ov_b.shape[1]))
    for j in range(ov_b.shape[1]):
        s = np.bincount(assign_b, weights=ov_b[:, j].astype(np.float64), minlength=K)
        mu[nz, j] = s[nz] / n[nz] / 100.0
    return n, mu


def cell_coverage(assign_b: np.ndarray, ov_b: np.ndarray, K: int, n: np.ndarray, t: int):
    """pi(tau) per (cell, config), READ OFF THE MAP'S CF HISTOGRAM as the draft
    defines the map (main.tex Step 4: 20 bins of width 0.05). For a tau on a bin
    edge -- 0.70, 0.80, 0.90 -- it must equal the direct count exactly, and that
    is asserted, so the histogram can never silently change a result."""
    nz = n > 0
    pi = np.zeros((K, ov_b.shape[1]))
    for j in range(ov_b.shape[1]):
        hist = cf_histogram(assign_b, ov_b[:, j] / 100.0, K)
        pi[:, j] = coverage_from_histogram(hist, t / 100.0)
        if t % 5 == 0:
            c = np.bincount(assign_b, weights=(ov_b[:, j] >= t).astype(np.float64), minlength=K)
            direct = np.zeros(K)
            direct[nz] = c[nz] / n[nz]
            if not np.allclose(pi[:, j], direct, rtol=0, atol=1e-12):
                raise RuntimeError(f"histogram coverage != direct count at t={t}, config {j}")
    return pi


def evaluate(idx: np.ndarray, ov_c: np.ndarray, cost: np.ndarray, t: int) -> dict:
    rows = np.arange(len(idx))
    f = ov_c[rows, idx].astype(np.int64)
    viol = f < t
    return {
        "cost_s": float(cost[idx].mean()),
        "viol": float(viol.mean()),
        "short_on_viol": float(((t - f[viol]) / 100.0).mean()) if viol.any() else 0.0,
        "mean_fidelity": float(f.mean() / 100.0),
        "pick_share": (np.bincount(idx, minlength=ov_c.shape[1]) / len(idx)).tolist(),
    }


def with_rec(ev: dict, c_static: float, c_opt: float) -> dict:
    gap = c_static - c_opt
    ev["cost_rel_static"] = ev["cost_s"] / c_static
    ev["rec"] = (c_static - ev["cost_s"]) / gap if gap > 1e-15 else None
    return ev


# ==========================================================================
# cross-checks against what is already on disk
# ==========================================================================
def check_t9(lay, pairing: str, order: list, n: np.ndarray, mu: np.ndarray) -> dict:
    z = np.load(lay.pairing_dir(pairing) / "cell_stats.npz", allow_pickle=False)
    stored = [str(c) for c in z["configs"].tolist()]
    worst_mu, worst_n = 0.0, 0
    for j, c in enumerate(order):
        r = stored.index(c)
        sn = z["n"][r].astype(np.int64)
        worst_n = max(worst_n, int(np.abs(sn - n.astype(np.int64)).max()))
        occ = sn > 0
        worst_mu = max(worst_mu, float(np.abs(z["cf_mean"][r][occ] - mu[occ, j]).max()))
    ok = worst_n == 0 and worst_mu < 1e-5
    if not ok:
        raise RuntimeError(f"{pairing}: rebuilt K={K_MAIN} cells differ from T9's "
                           f"cell_stats.npz (max |dn|={worst_n}, max |dmu|={worst_mu:g})")
    return {"max_abs_count_diff": worst_n, "max_abs_mean_diff": worst_mu, "ok": ok}


def h4_rows(model: str) -> dict:
    p = (HT_DIR or R / "hypothesis_testing" / SCALE) / model / "H4" / "h4_sweep.csv"
    with open(p) as f:
        return {(r["pairing"], int(r["K"])): r for r in csv.DictReader(f)}


# ==========================================================================
# one pairing
# ==========================================================================
def run_pairing(model: str, pairing: str, configs: list, costs: dict, h4: dict,
                assign_dir: Path | None = None) -> dict:
    """assign_dir: Q_test cells from another lookup (paperexp/hnsw_assign.py) instead of
    the stored exact assignment; the map itself is still built from Q's exact cells."""
    lay = lay_at(model, K_MAIN)
    order = sorted(configs, key=lambda c: (costs[c], configs.index(c)))
    cost = np.array([costs[c] for c in order], dtype=np.float64)
    n_b = len(np.load(lay.role_ids("B")))
    n_c = len(np.load(lay.role_ids("C")))
    ov_b = load_overlap(lay, pairing, "B", order, n_b)
    ov_c = load_overlap(lay, pairing, "C", order, n_c)
    m = len(order)
    log(f"  {pairing}: Q {n_b:,}  Q_test {n_c:,}  configs (cheapest first) {order}")

    mu_bar = ov_b.mean(axis=0) / 100.0
    f_c = ov_c / 100.0
    out = {"cost_order": order, "cost_s_per_query": cost.tolist(),
           "families": [config_family(model, c) for c in order],
           "n_Q": n_b, "n_Q_test": n_c, "checks": {}, "by_tau": {}, "by_K": {}}

    # ---- per-K cell tables (tau-independent parts) --------------------------
    tilings = {1: (np.zeros(n_b, np.int64), np.zeros(n_c, np.int64))}
    for k in KS:
        lk = lay_at(model, k)
        ab = load_query_geometry(lk, pairing, "B")["assign"].astype(np.int64)
        ac = (np.load(assign_dir / f"K{k}" / f"{pairing}_C.npy").astype(np.int64) if assign_dir
              else load_query_geometry(lk, pairing, "C")["assign"].astype(np.int64))
        if len(ab) != n_b or len(ac) != n_c:
            raise ValueError(f"{pairing} K={k}: assignment lengths {len(ab)}/{len(ac)} "
                             f"!= roles {n_b}/{n_c}")
        tilings[k] = (ab, ac)

    cells = {}
    for k, (ab, ac) in tilings.items():
        K = max(k, 1)
        n, mu = cell_tables(ab, ov_b, K)
        thin = n < Q_MIN
        mu_hat = np.where(thin[:, None], mu_bar[None, :], mu)
        fhat = mu_hat[ac]                                   # [n_c, m]
        skills = []
        for j in range(m):
            mse = float(((fhat[:, j] - f_c[:, j]) ** 2).mean())
            mse_ga = float(((mu_bar[j] - f_c[:, j]) ** 2).mean())
            skills.append(1.0 - mse / mse_ga if mse_ga > 0 else 0.0)
        thin_rate = float(thin[ac].mean())
        entry = {"skill": float(np.nanmean(skills)), "thin_rate": thin_rate,
                 "empty_cells": int((n == 0).sum())}
        if k in KS:                                         # reproduce H4
            r = h4[(pairing, k)]
            ds, db = abs(entry["skill"] - float(r["skill"])), abs(thin_rate - float(r["backoff_rate"]))
            if (ds > CHECK_TOL or db > CHECK_TOL) and assign_dir is None:
                raise RuntimeError(f"{pairing} K={k}: skill/backoff {entry['skill']:.5f}/"
                                   f"{thin_rate:.5f} differ from H4 {r['skill']}/"
                                   f"{r['backoff_rate']}")
            entry["h4_check"] = {"skill_h4": float(r["skill"]), "backoff_h4": float(r["backoff_rate"]),
                                 "abs_diff_skill": ds, "abs_diff_backoff": db,
                                 "ok": ds <= CHECK_TOL and db <= CHECK_TOL}
        if k == K_MAIN:
            out["checks"]["t9_cell_stats_K2500"] = check_t9(lay, pairing, order, n, mu)
        out["by_K"][str(k)] = entry
        cells[k] = (n, mu_hat, fhat, thin, ab, ac)

    # ---- rank flips at K_MAIN -------------------------------------------------
    n, mu_hat, _, thin, _, _ = cells[K_MAIN]
    well = ~thin
    flips = {}
    for a, b in combinations(range(m), 2):
        d = mu_hat[well, a] - mu_hat[well, b]
        flips[f"{order[a]}|{order[b]}"] = {
            "a": order[a], "b": order[b], "n_cells": int(well.sum()),
            "a_wins_cells": int((d > FLIP_MARGIN).sum()),
            "b_wins_cells": int((d < -FLIP_MARGIN).sum()),
            "cost_ratio": float(max(cost[a], cost[b]) / min(cost[a], cost[b])),
        }
    out["rank_flips_K2500"] = flips
    out["checks"]["overlap_is_integer"] = True

    # ---- per tau: optimum, static, mean rule, coverage rule --------------------
    for tau in TAUS:
        t = int(round(tau * 100))
        suff_c = ov_c >= t
        opt_idx, none = pick(suff_c, ov_c)
        c_opt = float(cost[opt_idx].mean())
        cov_b = (ov_b >= t).mean(axis=0)
        pi_bar = cov_b
        T = {"t_overlap": t, "optimum": evaluate(opt_idx, ov_c, cost, t),
             "none_share": float(none.mean()), "by_rho": {}, "by_K": {}}
        T["optimum"]["cost_s"] = c_opt

        pis = {}
        for k, (n_k, _, fhat, thin_k, ab, ac) in cells.items():
            pi = cell_coverage(ab, ov_b, max(k, 1), n_k, t)
            pi_hat = np.where(thin_k[:, None], pi_bar[None, :], pi)
            pis[k] = pi_hat[ac]                             # [n_c, m]
            midx, _ = pick(fhat >= tau - 1e-9, fhat)
            T["by_K"][str(k)] = {"mean_rule": evaluate(midx, ov_c, cost, t)}
            if k == K_MAIN:
                cal = calibration_curve(pis[k].ravel(), suff_c.ravel().astype(np.float64),
                                        bins=ECE_BINS)
                T["by_K"][str(k)]["coverage_ece"] = cal["ece"]
                T["by_K"][str(k)]["coverage_calibration"] = {
                    kk: cal[kk] for kk in ("n", "pred_mean", "true_mean")}

        for rho in RHOS:
            s, reached = static_plan(cov_b, rho)
            c_static = float(cost[s])
            st_idx = np.full(n_c, s, dtype=np.int64)
            rel = cost[opt_idx] / c_static
            served = ~none
            R_ = {
                "static": {**with_rec(evaluate(st_idx, ov_c, cost, t), c_static, c_opt),
                           "config": order[s], "reached_rho_on_Q": reached,
                           "coverage_on_Q": float(cov_b[s]),
                           "serves_on_Q_test": float(suff_c[:, s].mean())},
                "optimum_rel_cost": c_opt / c_static,
                "headroom_buckets": {   # RQ1: cost of each query's cheapest sufficient plan
                    "le_one_third": float((served & (rel <= 1 / 3 + 1e-12)).mean()),
                    "one_third_to_one": float((served & (rel > 1 / 3 + 1e-12) & (rel < 1 - 1e-12)).mean()),
                    "equal_one": float((served & (np.abs(rel - 1) <= 1e-12)).mean()),
                    "above_one": float((served & (rel > 1 + 1e-12)).mean()),
                    "none": float(none.mean()),
                },
                "by_K": {},
            }
            for k in cells:
                fhat = cells[k][2]
                midx, _ = pick(fhat >= tau - 1e-9, fhat)
                cidx, cnone = pick(pis[k] >= rho - 1e-12, pis[k])
                if k == 1 and not np.all(cidx == s):
                    raise RuntimeError(f"{pairing} tau={tau} rho={rho}: K=1 coverage "
                                       f"rule did not reproduce the static plan")
                R_["by_K"][str(k)] = {
                    "mean_rule": with_rec(evaluate(midx, ov_c, cost, t), c_static, c_opt),
                    "coverage_rule": {**with_rec(evaluate(cidx, ov_c, cost, t), c_static, c_opt),
                                      "fallback_share": float(cnone.mean())},
                }
            T["by_rho"][f"{rho:.2f}"] = R_
        if V2_STATS:
            T["v2"] = v2_stats(tau, t, order, cost, ov_c, f_c, cov_b, cells, pis, EST,
                               pairing, BOOT_R_V2)
            # K sweep (RQ5): Rec. of both rules against the v2 matched optimum at every K
            mo_c = T["v2"]["selection_rho_0.90"]["matched_optimum"]
            R9 = T["by_rho"]["0.90"]
            s9 = order.index(R9["static"]["config"])
            c_s9 = float(cost[s9])
            for k_, ent in R9["by_K"].items():
                if k_ == "1":
                    continue
                ce = float(EST["pairings"][pairing]["batched_steps_1_3"][k_]["s_per_query"])
                for rn in ("mean_rule", "coverage_rule"):
                    tot = ce + ent[rn]["cost_s"]
                    gap = (c_s9 - mo_c["cost_s"]) if mo_c["feasible"] else None
                    ent[rn]["cost_s_total_v2"] = tot
                    ent[rn]["rec_matched"] = ((c_s9 - tot) / gap if gap is not None and gap > 1e-15
                                              else None)
                    ent[rn]["optimum_gap_share"] = (gap / c_s9 if gap is not None else None)
        out["checks"][f"k1_equals_static_tau{tau:.2f}"] = True
        out["by_tau"][f"{tau:.2f}"] = T
        main = T["by_rho"]["0.90"]
        log(f"    tau={tau:.2f}: optimum {c_opt * 1e6:,.0f} us/q, none {none.mean():.1%}; "
            f"static(rho=0.90) {main['static']['config']}; coverage rule K={K_MAIN} "
            f"viol {main['by_K'][str(K_MAIN)]['coverage_rule']['viol']:.1%} "
            f"rec {main['by_K'][str(K_MAIN)]['coverage_rule']['rec']}")
    return out


# ==========================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True,
                    choices=["siglip2", "qwen3vl", "openclip", "jinav4"])
    ap.add_argument("--out-dir", type=Path, default=R / "paper_experiments" / SCALE)
    ap.add_argument("--cost-dir", type=Path, default=None,
                    help="controlled-throughput tree (<dir>/<model>/throughput.json); "
                         "default result/throughput/7M_cleaned")
    ap.add_argument("--assign-dir", type=Path, default=None,
                    help="Q_test cells from paperexp/hnsw_assign.py (<dir>/K<k>/<pairing>_C.npy)")
    ap.add_argument("--out-name", default="selection.json")
    ap.add_argument("--v2-stats", action="store_true",
                    help="v2: at the main K add RQ2's coverage columns and RQ4's numbers "
                         "with cell-bootstrap CIs, the draft's tests, and Rec. against the "
                         "v2 matched optimum (needs estimate.json in --out-dir/<model>)")
    ap.add_argument("--boot-r", type=int, default=1000)
    ap.add_argument("--allow-login-node", action="store_true")
    args = ap.parse_args()
    global V2_STATS, EST, BOOT_R_V2
    if args.v2_stats:
        V2_STATS, BOOT_R_V2 = True, args.boot_r
        ep = args.out_dir / args.model / "estimate.json"
        EST = json.loads(ep.read_text())
        if not EST.get("complete"):
            raise SystemExit(f"{ep} is incomplete -- run bench_estimate.py first")
    if not os.environ.get("PBS_JOBID") and not args.allow_login_node:
        raise SystemExit("Refusing to run outside a PBS job -- submit "
                         "paperexp/jobs/paperexp_selection_7m.pbs")

    t0 = time.time()
    configs = resolve_configs(args.model, "all")
    costs = bench_costs(args.model, args.cost_dir)
    h4 = h4_rows(args.model)
    log(f"selection experiments -- model={args.model}  scale={SCALE}  "
        f"K={list(KS)}  tau={list(TAUS)}  rho={list(RHOS)}  q_min={Q_MIN}")

    result = {
        "task": "paper experiments part 1: plan selection (RQ1, RQ3 rank flips, RQ4, RQ5 K sweep)",
        "model": args.model, "scale": SCALE,
        "created": datetime.now().isoformat(),
        "host": os.uname().nodename, "pbs_jobid": os.environ.get("PBS_JOBID"),
        "parameters": {"K_sweep": list(KS), "K_main": K_MAIN, "taus": list(TAUS),
                       "rhos": list(RHOS), "q_min": Q_MIN, "flip_margin": FLIP_MARGIN,
                       "ece_bins": ECE_BINS, "k": FID_K},
        "inputs": {
            "labels": str(R / "cf" / SCALE / args.model),
            "assignments": str(R / "kmeans" / SCALE / args.model),
            "costs": str((args.cost_dir or R / "throughput" / SCALE) / args.model / "throughput.json"),
            "t9_crosscheck": str(R / "spatial_map" / SCALE / args.model / "K02500_spherical"),
            "h4_crosscheck": str(R / "hypothesis_testing" / SCALE / args.model / "H4" / "h4_sweep.csv"),
            "q_test_cells": str(args.assign_dir) if args.assign_dir else "exact (stored assignment)",
        },
        "definitions": __doc__.split("DEFINITIONS")[1].split("CROSS-CHECKS")[0].strip(),
        "pairings": {},
    }
    for p in sorted(PAIRINGS):
        result["pairings"][p] = run_pairing(args.model, p, configs, costs[p], h4, args.assign_dir)

    dest = args.out_dir / args.model
    dest.mkdir(parents=True, exist_ok=True)
    tmp = dest / (args.out_name + ".tmp")
    tmp.write_text(json.dumps(result, indent=1))
    tmp.replace(dest / args.out_name)
    log(f"wrote {dest / args.out_name} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
