"""
Paper experiments, part 4: plan selection held to the static plan's violation
rate -- the goal stated in Section 2.4:

    "The goal is a selector with low average cost, c_est + c(A_hat(q)), and a
     violation rate no higher than that of the static plan."

    python paperexp/matched_violation.py --model siglip2        (inside a PBS job)

The RQ4 table compares selectors at one fixed rho, where their violation rates
differ from the static plan's. This compares them at MATCHED violation: each
selector's one free parameter is tuned so that it violates no more often than
the static plan, and then its cost is compared.

NO SEARCH. Same inputs as selection_tables.py (T8 labels, T5 assignments at
every K, controlled costs), plus estimate.json for c_est.

PROTOCOL (no tuning on the test set)
------------------------------------
  Q is split once, seed 0, into Q_build (80%) and Q_cal (20%).
  Q_build  builds the map: n_i, pi_ij(tau) per cell, q_min = 10 backoff; and
           chooses the static plan at rho_s = 0.9 exactly as in the RQ4 table.
  Q_cal    measures the static plan's violation v_s, and tunes each selector's
           parameter: the cheapest setting (cost measured on Q_cal) whose
           violation on Q_cal is <= v_s. If no setting reaches v_s the selector
           is recorded as infeasible for that direction.
  Q_test   is touched once, to evaluate every tuned selector and the static plan.

SELECTORS (configurations ordered cheapest first; ties go to the cheaper one)
  coverage rule   cheapest A_j with pi_hat(q, A_j; tau) >= rho, else argmax
                  pi_hat. rho tuned on a grid 0.500, 0.505, ..., 1.000.
  cost-aware rule argmin_j  c(A_j) + lambda * (1 - pi_hat(q, A_j; tau)):
                  each miss is priced at lambda seconds, so the violation budget
                  goes where it saves the most. lambda tuned on 10^-6 .. 10^1 s
                  (141 log-spaced values).
  Both map selectors pay c_est: Steps 1-3 measured by bench_estimate.py
  (batched, at the same K).

FIXED CAP (--viol-cap v): instead of the static plan's violation, each selector is
tuned to violate on at most a fraction v of Q_cal, and the bounds follow:
  static at cap    the cheapest static deployment chosen on Q_build whose violation
                   there is <= v: one configuration, or a random split of the queries
                   between two (the lower convex hull of the nine), costed and
                   violation-counted on Q_test (the split's expected values)
  matched optimum  as below, with the violation budget floor(v * |Q_test|)
Costs stay relative to the rho = 0.9 static plan.

BOUNDS, on Q_test
  optimum          the per-query optimum of Section 2.4 (zero avoidable violations)
  matched optimum  the cheapest assignment that knows every F(q, A) and violates
                   on no more queries than the static plan does on Q_test: queries
                   no configuration serves violate anyway and go to the cheapest
                   configuration; the remaining violation budget is spent on the
                   queries whose cheapest sufficient plan saves the most by being
                   replaced with the cheapest configuration. It is the lower bound
                   for any selector at the static plan's violation rate.

V2 PROTOCOL (--v2-protocol; user decisions of 2026-10-19)
  tuning      5-fold cross-fitting on ALL of Q (seed 0) instead of one 80/20
              split: every Q query gets an out-of-fold pi_hat from a map built
              on the other four folds; rho / lambda are tuned on all of Q with
              those, to the static plan's violation on Q; the FINAL map is built
              on all of Q and applied once to Q_test. The static plan is chosen
              on all of Q (draft main.tex l.440, l.442).
  matched     queries the matched optimum leaves unserved (unservable ones, and
  optimum     the extra misses) are charged the STATIC plan's configuration --
              what a static deployment pays for them -- not the cheapest one.
              The extra misses go to the queries whose cheapest sufficient
              configuration costs most above the static one.
  feasibility "feasible": false and cost null when even the optimum cannot stay
              within the miss budget (more unservable queries than the budget).

Output: result/paper_experiments/7M_cleaned/<model>/matched.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
for sub in ("paperexp", "indexes", "spatialmap", "hypothesis_testing"):
    sys.path.insert(0, str(REPO_ROOT / sub))

from map_config import PAIRINGS, load_query_geometry, resolve_configs  # noqa: E402
from metrics import (CellWeights, boot_ci, mcnemar,                     # noqa: E402
                     resampled_matched_optimum)
BOOT_R = 1000           # user, 2026-10-19 (draft l.478)
from selection_tables import (KS, K_MAIN, Q_MIN, R, SCALE, bench_costs,  # noqa: E402
                              cell_coverage, cell_tables, lay_at,
                              load_overlap, pick, static_plan)

TAUS = (0.70, 0.80, 0.90)
RHO_STATIC = 0.90
CAL_FRACTION = 0.20
SPLIT_SEED = 0
N_FOLDS = 5
RHO_GRID = np.round(np.arange(0.500, 1.0001, 0.005), 3)
LAM_GRID = np.logspace(-6, 1, 141)


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def viol_cost(idx: np.ndarray, ov: np.ndarray, cost: np.ndarray, t: int):
    f = ov[np.arange(len(idx)), idx]
    return float((f < t).mean()), float(cost[idx].mean())


def cost_aware(P: np.ndarray, cost: np.ndarray, lam: float) -> np.ndarray:
    return np.argmin(cost[None, :] + lam * (1.0 - P), axis=1)


def tune(values, choose, P_cal, ov_cal, cost, t, v_target):
    """Cheapest setting (cost on Q_cal) whose Q_cal violation <= v_target."""
    best, trace = None, []
    for v in values:
        idx = choose(P_cal, v)
        vi, co = viol_cost(idx, ov_cal, cost, t)
        trace.append((float(v), vi, co))
        if vi <= v_target + 1e-12 and (best is None or co < best[2]):
            best = (float(v), vi, co)
    return best, trace


def static_at_cap(ov_q: np.ndarray, ov_c: np.ndarray, cost: np.ndarray, order: list,
                  t: int, cap: float) -> dict:
    """Cheapest static deployment with violation <= cap on ov_q (Q_build): one configuration,
    or a random split between two; evaluated on ov_c (Q_test)."""
    vq, vc = (ov_q < t).mean(axis=0), (ov_c < t).mean(axis=0)
    m = len(cost)
    single = [j for j in range(m) if vq[j] <= cap + 1e-12]
    js = min(single, key=lambda j: (cost[j], j)) if single else None
    out = {"single": None if js is None else {
        "config": order[js], "cost_s": float(cost[js]), "viol_build": float(vq[js]),
        "viol_test": float(vc[js])}}
    best = None if js is None else (float(cost[js]), js, js, 0.0)
    for a in range(m):
        for b in range(m):
            if vq[a] < cap < vq[b]:
                w = float((cap - vq[a]) / (vq[b] - vq[a]))          # share sent to b
                c = (1 - w) * cost[a] + w * cost[b]
                if best is None or c < best[0]:
                    best = (float(c), a, b, w)
    if best is None:
        out["mix"] = None
        return out
    c, a, b, w = best
    out["mix"] = {"a": order[a], "b": order[b], "share_b": w, "cost_s": c,
                  "viol_test": float((1 - w) * vc[a] + w * vc[b])}
    return out


def matched_optimum_static_charge(ov_c: np.ndarray, cost: np.ndarray, t: int, budget: int,
                                  s: int) -> dict:
    """v2: misses are charged the static plan's configuration s (user, 2026-10-19)."""
    ok = ov_c >= t
    none = ~ok.any(axis=1)
    first = np.argmax(ok, axis=1)
    cstar = np.where(none, cost[s], cost[first])
    total = cstar.sum()
    extra = budget - int(none.sum())
    if extra > 0:
        saving = np.where(none, 0.0, cstar - cost[s])      # > 0 only above the static cost
        top = np.sort(saving)[::-1][:extra]
        total -= top[top > 0].sum()
    feasible = int(none.sum()) <= budget
    return {"cost_s": float(total / len(cstar)) if feasible else None,
            "cost_s_if_unconstrained": float(total / len(cstar)),
            "violations": int(max(budget, none.sum())), "unservable": int(none.sum()),
            "budget": int(budget), "feasible": bool(feasible),
            "missed_charged": "static plan configuration"}


def oof_coverage(ab: np.ndarray, ov_b: np.ndarray, k: int, t: int, folds: np.ndarray,
                 ab_lookup: np.ndarray | None = None) -> np.ndarray:
    """Out-of-fold pi_hat [n_Q, m]: each fold scored with a map (q_min backoff to
    the training folds' coverage) built on the other folds only. ab_lookup: the cells
    the held-out fold is LOOKED UP in (an online lookup, e.g. HNSW over the landmarks);
    the maps are always built with the exact cells ab."""
    ab_lookup = ab if ab_lookup is None else ab_lookup
    P = np.zeros(ov_b.shape, dtype=np.float64)
    for f in range(N_FOLDS):
        tr, te = folds != f, folds == f
        n, _ = cell_tables(ab[tr], ov_b[tr], k)
        pi = cell_coverage(ab[tr], ov_b[tr], k, n, t)
        bar = (ov_b[tr] >= t).mean(axis=0)
        pi_hat = np.where((n < Q_MIN)[:, None], bar[None, :], pi)
        P[te] = pi_hat[ab_lookup[te]]
    return P


def matched_optimum(ov_c: np.ndarray, cost: np.ndarray, t: int, budget: int) -> dict:
    ok = ov_c >= t
    none = ~ok.any(axis=1)
    first = np.argmax(ok, axis=1)
    cstar = np.where(none, cost[0], cost[first])          # none -> cheapest (violates anyway)
    extra = budget - int(none.sum())
    total = cstar.sum()
    if extra > 0:
        saving = np.where(none, 0.0, cstar - cost[0])
        top = np.sort(saving)[::-1][:extra]
        total -= top.sum()
    return {"cost_s": float(total / len(cstar)), "violations": int(max(budget, none.sum())),
            "unservable": int(none.sum()), "budget": int(budget)}


def run_pairing(model, pairing, configs, costs, est, assign_dir=None, viol_cap=None,
                v2: bool = False) -> dict:
    """assign_dir: Q_cal and Q_test cells from another lookup (paperexp/hnsw_assign.py),
    since both stand in for online queries; Q_build keeps its exact cells."""
    if v2:
        return run_pairing_v2(model, pairing, configs, costs, est, viol_cap, assign_dir)
    lay = lay_at(model, K_MAIN)
    order = sorted(configs, key=lambda c: (costs[c], configs.index(c)))
    cost = np.array([costs[c] for c in order], dtype=np.float64)
    n_b = len(np.load(lay.role_ids("B")))
    n_c = len(np.load(lay.role_ids("C")))
    ov_b = load_overlap(lay, pairing, "B", order, n_b)
    ov_c = load_overlap(lay, pairing, "C", order, n_c)

    rng = np.random.default_rng(SPLIT_SEED)
    cal = np.zeros(n_b, dtype=bool)
    cal[rng.choice(n_b, size=int(round(CAL_FRACTION * n_b)), replace=False)] = True
    build = ~cal
    log(f"  {pairing}: Q_build {build.sum():,}  Q_cal {cal.sum():,}  Q_test {n_c:,}")

    out = {"cost_order": order, "cost_s_per_query": cost.tolist(),
           "n_build": int(build.sum()), "n_cal": int(cal.sum()), "n_test": n_c, "by_tau": {}}
    tilings = {}
    for k in KS:
        lk = lay_at(model, k)
        ab = load_query_geometry(lk, pairing, "B")["assign"].astype(np.int64)
        if assign_dir:
            ab_on = np.load(assign_dir / f"K{k}" / f"{pairing}_B.npy").astype(np.int64)
            ac = np.load(assign_dir / f"K{k}" / f"{pairing}_C.npy").astype(np.int64)
        else:
            ab_on = ab
            ac = load_query_geometry(lk, pairing, "C")["assign"].astype(np.int64)
        tilings[k] = (ab, ac, ab_on)

    for tau in TAUS:
        t = int(round(tau * 100))
        # static plan: chosen on Q_build, exactly as the RQ4 table chooses it on Q
        cov_build = (ov_b[build] >= t).mean(axis=0)
        s, reached = static_plan(cov_build, RHO_STATIC)
        s_full, _ = static_plan((ov_b >= t).mean(axis=0), RHO_STATIC)
        v_s_cal = float((ov_b[cal, s] < t).mean())
        v_s_test = float((ov_c[:, s] < t).mean())
        c_static = float(cost[s])

        ok_c = ov_c >= t
        opt_idx, _ = pick(ok_c, ov_c)
        c_opt = float(cost[opt_idx].mean())
        v_budget = v_s_test if viol_cap is None else viol_cap
        mo = matched_optimum(ov_c, cost, t, int(np.floor(v_budget * n_c + 1e-9)))
        target = v_s_cal if viol_cap is None else viol_cap

        T = {"static": {"config": order[s], "reached_rho_on_Q_build": reached,
                        "same_as_full_Q_choice": s == s_full,
                        "viol_cal": v_s_cal, "viol_test": v_s_test, "cost_s": c_static},
             "optimum": {"cost_s": c_opt},
             "matched_optimum": mo, "by_K": {}}
        if viol_cap is not None:
            T["viol_cap"] = viol_cap
            T["static_at_cap"] = static_at_cap(ov_b[build], ov_c, cost, order, t, viol_cap)

        for k, (ab, ac, ab_on) in tilings.items():
            n, _ = cell_tables(ab[build], ov_b[build], k)
            pi = cell_coverage(ab[build], ov_b[build], k, n, t)
            pi_bar = cov_build
            thin = n < Q_MIN
            pi_hat = np.where(thin[:, None], pi_bar[None, :], pi)
            P_cal, P_test = pi_hat[ab_on[cal]], pi_hat[ac]
            ov_cal = ov_b[cal]
            cest = est["pairings"][pairing]["batched_steps_1_3"][str(k)]["s_per_query"]

            entry = {"c_est_s": cest, "thin_rate_test": float(thin[ac].mean())}
            for name, grid, choose in (
                    ("coverage_rule", RHO_GRID, lambda P, v: pick(P >= v - 1e-12, P)[0]),
                    ("cost_aware_rule", LAM_GRID, lambda P, v: cost_aware(P, cost, v))):
                best, trace = tune(grid, choose, P_cal, ov_cal, cost, t, target)
                if best is None:
                    # infeasible on Q_cal: report the setting with the fewest violations
                    lo = min(trace, key=lambda x: (x[1], x[2]))
                    idx = choose(P_test, lo[0])
                    vt, ct = viol_cost(idx, ov_c, cost, t)
                    entry[name] = {"feasible_on_cal": False, "param": lo[0], "viol_cal": lo[1],
                                   "viol_test": vt, "cost_s_search": ct,
                                   "cost_s_total": ct + cest}
                    continue
                idx = choose(P_test, best[0])
                vt, ct = viol_cost(idx, ov_c, cost, t)
                total = ct + cest
                entry[name] = {
                    "feasible_on_cal": True, "param": best[0], "viol_cal": best[1],
                    "viol_test": vt, "cost_s_search": ct, "cost_s_total": total,
                    "cost_rel_static": total / c_static,
                    "no_higher_viol_than_static_on_test": vt <= v_s_test + 1e-12,
                    "within_cap_on_test": None if viol_cap is None else vt <= viol_cap + 1e-12,
                    "cheaper_than_static": total < c_static,
                    "rec_vs_matched_optimum": ((c_static - total) / (c_static - mo["cost_s"])
                                               if c_static - mo["cost_s"] > 1e-15 else None),
                    "pick_share": (np.bincount(idx, minlength=len(order)) / len(idx)).tolist(),
                }
            T["by_K"][str(k)] = entry
        out["by_tau"][f"{tau:.2f}"] = T
        e = T["by_K"][str(K_MAIN)]
        log(f"    tau={tau:.2f} static {order[s]} viol cal/test {v_s_cal:.2%}/{v_s_test:.2%} | "
            + " | ".join(f"{nm}: " + (f"param {e[nm]['param']:.3g} viol {e[nm]['viol_test']:.2%} "
                                      f"cost {e[nm]['cost_s_total'] / c_static:.2f}x"
                                      if e[nm]["feasible_on_cal"] else "infeasible")
                         for nm in ("coverage_rule", "cost_aware_rule"))
            + f" | matched optimum {mo['cost_s'] / c_static:.2f}x")
    return out


def run_pairing_v2(model, pairing, configs, costs, est, viol_cap=None, assign_dir=None) -> dict:
    """The v2 protocol (see the docstring): cross-fitted tuning on all of Q, final
    map on all of Q, matched optimum with misses charged the static configuration.
    assign_dir (2026-10-31; before this it was silently ignored under --v2-protocol):
    the cells of an online lookup (paperexp/hnsw_assign.py) for the queries that stand
    in for online ones -- the held-out fold in tuning and Q_test. Every map is still
    built with the exact cells; the bootstrap still resamples the exact Q_test cells."""
    lay = lay_at(model, K_MAIN)
    order = sorted(configs, key=lambda c: (costs[c], configs.index(c)))
    cost = np.array([costs[c] for c in order], dtype=np.float64)
    n_b = len(np.load(lay.role_ids("B")))
    n_c = len(np.load(lay.role_ids("C")))
    ov_b = load_overlap(lay, pairing, "B", order, n_b)
    ov_c = load_overlap(lay, pairing, "C", order, n_c)
    folds = np.random.default_rng(SPLIT_SEED).permutation(n_b) % N_FOLDS
    log(f"  {pairing}: Q {n_b:,} in {N_FOLDS} folds  Q_test {n_c:,}  (v2 protocol)")
    out = {"cost_order": order, "cost_s_per_query": cost.tolist(), "protocol": "v2",
           "tuning_set": f"all of Q, {N_FOLDS}-fold cross-fitted pi_hat (seed {SPLIT_SEED})",
           "n_Q": n_b, "n_test": n_c, "by_tau": {}}
    tilings = {k: (load_query_geometry(lay_at(model, k), pairing, "B")["assign"].astype(np.int64),
                   load_query_geometry(lay_at(model, k), pairing, "C")["assign"].astype(np.int64))
               for k in KS}
    online = {}                                   # k -> (B lookup cells, C lookup cells)
    for k, (ab, ac) in tilings.items():
        if assign_dir:
            ob = np.load(assign_dir / f"K{k}" / f"{pairing}_B.npy").astype(np.int64)
            oc = np.load(assign_dir / f"K{k}" / f"{pairing}_C.npy").astype(np.int64)
            if len(ob) != len(ab) or len(oc) != len(ac):
                raise ValueError(f"{pairing} K={k}: lookup cells {len(ob)}/{len(oc)} != {len(ab)}/{len(ac)}")
            online[k] = (ob, oc)
        else:
            online[k] = (ab, ac)
    if assign_dir:
        out["online_cells"] = str(assign_dir)
        out["online_cell_agreement"] = {str(k): {"B": float((online[k][0] == ab).mean()),
                                                 "C": float((online[k][1] == ac).mean())}
                                        for k, (ab, ac) in tilings.items()}
    for tau in TAUS:
        t = int(round(tau * 100))
        cov_q = (ov_b >= t).mean(axis=0)
        s, reached = static_plan(cov_q, RHO_STATIC)
        v_s_q = float((ov_b[:, s] < t).mean())
        v_s_test = float((ov_c[:, s] < t).mean())
        c_static = float(cost[s])
        opt_idx, _ = pick(ov_c >= t, ov_c)
        v_budget = v_s_test if viol_cap is None else viol_cap
        mo = matched_optimum_static_charge(ov_c, cost, t, int(np.floor(v_budget * n_c + 1e-9)), s)
        target = v_s_q if viol_cap is None else viol_cap
        T = {"static": {"config": order[s], "reached_rho_on_Q": reached, "viol_Q": v_s_q,
                        "viol_test": v_s_test, "cost_s": c_static},
             "optimum": {"cost_s": float(cost[opt_idx].mean())},
             "matched_optimum": mo, "by_K": {}}
        mo_cheap = None
        if viol_cap is not None:
            T["viol_cap"] = viol_cap
            T["static_at_cap"] = static_at_cap(ov_b, ov_c, cost, order, t, viol_cap)
            # main.tex l.878 (user, 2026-10-27): the optimum at <= 1 - rho is the cheapest
            # assignment missing at most viol_cap of Q_test, a missed query run on the
            # CHEAPEST configuration (a true lower bound). matched_optimum above stays
            # the A3 static-charged version for the matched comparison.
            bud = int(np.floor(viol_cap * n_c + 1e-9))
            mo_cheap = matched_optimum(ov_c, cost, t, bud)
            mo_cheap.update(feasible=bool(mo_cheap["unservable"] <= bud),
                            missed_charged="cheapest configuration")
            T["optimum_cap_cheapest"] = mo_cheap
        for k, (ab, ac) in tilings.items():
            ob, oc = online[k]
            P_tune = oof_coverage(ab, ov_b, k, t, folds, ab_lookup=ob)
            n, _ = cell_tables(ab, ov_b, k)                      # the final map: all of Q
            pi = cell_coverage(ab, ov_b, k, n, t)
            thin = n < Q_MIN
            pi_hat = np.where(thin[:, None], cov_q[None, :], pi)
            P_test = pi_hat[oc]
            cest = est["pairings"][pairing]["batched_steps_1_3"][str(k)]["s_per_query"]
            entry = {"c_est_s": cest, "thin_rate_test": float(thin[oc].mean())}
            boot = None
            if k == K_MAIN:          # draft l.478: CIs by resampling cells of the main tiling
                cw = CellWeights(ac, R=BOOT_R)
                nq = cw.totals(np.ones(n_c))
                static_miss = ov_c[:, s] < t
                okc = ov_c >= t
                nonec = ~okc.any(axis=1)
                cstar = np.where(nonec, c_static, cost[np.argmax(okc, axis=1)])
                mo_r = (resampled_matched_optimum(cw, cstar, nonec, static_miss, c_static)
                        if mo["feasible"] else None)
                boot = (cw, nq, static_miss, mo_r)
            for name, grid, choose in (
                    ("coverage_rule", RHO_GRID, lambda P, v: pick(P >= v - 1e-12, P)[0]),
                    ("cost_aware_rule", LAM_GRID, lambda P, v: cost_aware(P, cost, v))):
                best, trace = tune(grid, choose, P_tune, ov_b, cost, t, target)
                feasible = best is not None
                if not feasible:
                    best = min(trace, key=lambda x: (x[1], x[2]))
                idx = choose(P_test, best[0])
                vt, ct = viol_cost(idx, ov_c, cost, t)
                total = ct + cest
                gap = c_static - mo["cost_s"] if mo["cost_s"] is not None else None
                entry[name] = {
                    "feasible_on_tuning": feasible, "param": best[0], "viol_tuning": best[1],
                    "viol_test": vt, "cost_s_search": ct, "cost_s_total": total,
                    "cost_rel_static": total / c_static,
                    "no_higher_viol_than_static_on_test": vt <= v_s_test + 1e-12,
                    "within_cap_on_test": None if viol_cap is None else vt <= viol_cap + 1e-12,
                    "cheaper_than_static": total < c_static,
                    "rec_vs_matched_optimum": ((c_static - total) / gap
                                               if gap is not None and gap > 1e-15 else None),
                    "rec_vs_cap_optimum": (((c_static - total) / (c_static - mo_cheap["cost_s"]))
                                           if mo_cheap is not None and mo_cheap["feasible"]
                                           and c_static - mo_cheap["cost_s"] > 1e-15 else None),
                    "pick_share": (np.bincount(idx, minlength=len(order)) / len(idx)).tolist(),
                }
                if boot is not None:
                    cw, nq, static_miss, mo_r = boot
                    miss = ov_c[np.arange(n_c), idx] < t
                    cq = cost[idx]
                    tot_r = cest + cw.totals(cq) / nq
                    ent = entry[name]
                    ent["cost_rel_static_ci"] = boot_ci(tot_r / c_static, total / c_static)
                    ent["viol_test_ci"] = boot_ci(cw.totals(miss) / nq, vt)
                    if mo_r is not None and ent["rec_vs_matched_optimum"] is not None:
                        with np.errstate(divide="ignore", invalid="ignore"):
                            rr = np.where(c_static - mo_r > 1e-15,
                                          (c_static - tot_r) / (c_static - mo_r), np.nan)
                        ent["rec_vs_matched_optimum_ci"] = boot_ci(rr, ent["rec_vs_matched_optimum"])
                    ent["mcnemar_vs_static"] = mcnemar(miss, static_miss)
            T["by_K"][str(k)] = entry
        out["by_tau"][f"{tau:.2f}"] = T
        e = T["by_K"][str(K_MAIN)]
        log(f"    tau={tau:.2f} static {order[s]} viol Q/test {v_s_q:.2%}/{v_s_test:.2%} | "
            + " | ".join(f"{nm}: viol {e[nm]['viol_test']:.2%} cost {e[nm]['cost_rel_static']:.2f}x"
                         + ("" if e[nm]["feasible_on_tuning"] else " (infeasible)")
                         for nm in ("coverage_rule", "cost_aware_rule"))
            + (f" | matched optimum {mo['cost_s'] / c_static:.2f}x" if mo["feasible"]
               else " | matched optimum infeasible"))
    return out


def shortfall(idx: np.ndarray, ov: np.ndarray, t: int) -> float:
    """Mean shortfall (t - F)/100 over the queries that miss the target."""
    f = ov[np.arange(len(idx)), idx].astype(np.int64)
    miss = f < t
    return float(((t - f[miss]) / 100.0).mean()) if miss.any() else 0.0


def reliability_bins(p: np.ndarray, y: np.ndarray, bins: int = 10, lo: float = 0.05,
                     hi: float = 0.95) -> dict:
    """Reliability-diagram data over pairs with lo < p < hi, the same equal-width
    bins as selection_tables.ece_restricted (draft l.457, l.676)."""
    keep = (p > lo) & (p < hi)
    p, y = p[keep], y[keep].astype(np.float64)
    b = np.minimum((p * bins).astype(np.int64), bins - 1)
    n = np.bincount(b, minlength=bins)
    sp = np.bincount(b, weights=p, minlength=bins)
    sy = np.bincount(b, weights=y, minlength=bins)
    with np.errstate(divide="ignore", invalid="ignore"):
        mp, my = np.where(n > 0, sp / n, np.nan), np.where(n > 0, sy / n, np.nan)
    ece = float(np.abs(sp - sy).sum() / n.sum()) if n.sum() else None
    return {"n": n.tolist(), "mean_pred": [None if np.isnan(v) else float(v) for v in mp],
            "observed": [None if np.isnan(v) else float(v) for v in my],
            "ece": ece, "pairs": int(keep.sum())}


def curves_pairing(model, pairing, configs, costs, est, sel_json, matched_json) -> dict:
    """--curves (user, 2026-10-20): what run_pairing_v2 computes but does not keep, at
    the main K only and with the same v2 protocol, so the RQ4 trade-off figure
    (draft l.790) and the reliability diagram (draft l.676) can be drawn:
      tuning   the grid trace on Q with 5-fold out-of-fold pi_hat (miss rate, cost)
      test     every grid value applied with the FINAL map to Q_test (miss rate,
               search cost, total cost incl. c_est, shortfall on misses)
      chosen   the tuned value as in matched.json, plus its shortfall
      reliab.  binned pi_hat vs observed rate on Q_test, pooled and per config
    Cross-checks: chosen values against matched.json, ECE against selection.json."""
    lay = lay_at(model, K_MAIN)
    order = sorted(configs, key=lambda c: (costs[c], configs.index(c)))
    cost = np.array([costs[c] for c in order], dtype=np.float64)
    n_b = len(np.load(lay.role_ids("B")))
    n_c = len(np.load(lay.role_ids("C")))
    ov_b = load_overlap(lay, pairing, "B", order, n_b)
    ov_c = load_overlap(lay, pairing, "C", order, n_c)
    folds = np.random.default_rng(SPLIT_SEED).permutation(n_b) % N_FOLDS
    ab = load_query_geometry(lay, pairing, "B")["assign"].astype(np.int64)
    ac = load_query_geometry(lay, pairing, "C")["assign"].astype(np.int64)
    cest = est["pairings"][pairing]["batched_steps_1_3"][str(K_MAIN)]["s_per_query"]
    log(f"  {pairing}: curves at K={K_MAIN}  Q {n_b:,}  Q_test {n_c:,}")
    out = {"K": K_MAIN, "cost_order": order, "cost_s_per_query": cost.tolist(),
           "c_est_s": cest, "by_tau": {}}
    for tau in TAUS:
        t = int(round(tau * 100))
        tk = f"{tau:.2f}"
        cov_q = (ov_b >= t).mean(axis=0)
        s, _ = static_plan(cov_q, RHO_STATIC)
        v_s_q = float((ov_b[:, s] < t).mean())
        st_idx = np.full(n_c, s, dtype=np.int64)
        P_tune = oof_coverage(ab, ov_b, K_MAIN, t, folds)
        n, _ = cell_tables(ab, ov_b, K_MAIN)
        pi = cell_coverage(ab, ov_b, K_MAIN, n, t)
        P_test = np.where((n < Q_MIN)[:, None], cov_q[None, :], pi)[ac]
        T = {"static": {"config": order[s], "cost_s": float(cost[s]),
                        "viol_test": float((ov_c[:, s] < t).mean()),
                        "shortfall_test": shortfall(st_idx, ov_c, t)}}
        ref = matched_json["pairings"][pairing]["by_tau"][tk]["by_K"][str(K_MAIN)]
        for name, grid, choose in (
                ("coverage_rule", RHO_GRID, lambda P, v: pick(P >= v - 1e-12, P)[0]),
                ("cost_aware_rule", LAM_GRID, lambda P, v: cost_aware(P, cost, v))):
            best, trace = tune(grid, choose, P_tune, ov_b, cost, t, v_s_q)
            feasible = best is not None
            if not feasible:
                best = min(trace, key=lambda x: (x[1], x[2]))
            test = []
            for v in grid:
                idx = choose(P_test, float(v))
                vt, ct = viol_cost(idx, ov_c, cost, t)
                test.append([float(v), vt, ct, ct + cest, shortfall(idx, ov_c, t)])
            idx = choose(P_test, best[0])
            vt, ct = viol_cost(idx, ov_c, cost, t)
            T[name] = {
                "grid_fields": ["param", "viol", "cost_s_search", "cost_s_total", "shortfall"],
                "tuning_trace_Q_oof": [[v, vi, co] for v, vi, co in trace],
                "test_curve": test,
                "chosen": {"param": best[0], "feasible_on_tuning": feasible,
                           "viol_test": vt, "cost_s_total": ct + cest,
                           "shortfall_test": shortfall(idx, ov_c, t),
                           "matches_matched_json": bool(
                               abs(best[0] - ref[name]["param"]) < 1e-12
                               and abs(vt - ref[name]["viol_test"]) < 1e-12)},
            }
        y = ov_c >= t
        rel = {"pooled": reliability_bins(P_test.ravel(), y.ravel()),
               "per_config": {c: reliability_bins(P_test[:, j], y[:, j])
                              for j, c in enumerate(order)}}
        e_sel = sel_json["pairings"][pairing]["by_tau"][tk]["v2"]["coverage_estimate"]["ece_restricted"]
        e_new = rel["pooled"]["ece"]
        rel["ece_selection_json"] = e_sel
        rel["ece_matches_selection_json"] = bool(
            e_sel is not None and e_new is not None and abs(e_sel - e_new) < 1e-9)
        T["reliability"] = rel
        out["by_tau"][tk] = T
        log(f"    tau={tk} static {order[s]} | chosen match matched.json: "
            + ", ".join(f"{nm} {T[nm]['chosen']['matches_matched_json']}"
                        for nm in ("coverage_rule", "cost_aware_rule"))
            + f" | ECE {e_new:.4f} vs selection.json {e_sel:.4f} "
              f"match {rel['ece_matches_selection_json']}")
    return out


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
                    help="Q_cal / Q_test cells from paperexp/hnsw_assign.py")
    ap.add_argument("--out-name", default="matched.json")
    ap.add_argument("--viol-cap", type=float, default=None,
                    help="tune to this violation rate instead of the static plan's")
    ap.add_argument("--v2-protocol", action="store_true",
                    help="cross-fitted tuning on all of Q, matched optimum charging misses "
                         "the static configuration, feasibility flags (see V2 PROTOCOL)")
    ap.add_argument("--configs", default="all",
                    help="comma-separated subset of the catalog (smaller-catalog runs); "
                         "static plan, optimum and rules are then computed within it")
    ap.add_argument("--extra-configs", default="",
                    help="comma-separated configurations outside the catalogue (labels in the "
                         "SFM_CF_TAG tree, costs in --cost-dir), appended to --configs")
    ap.add_argument("--curves", action="store_true",
                    help="v2 only: store the tuning traces, the Q_test trade-off curves of "
                         "both rules and reliability bins at the main K, in "
                         "matched_curves.json (needs matched.json and selection.json)")
    ap.add_argument("--allow-login-node", action="store_true")
    args = ap.parse_args()
    if args.curves and not args.v2_protocol:
        raise SystemExit("--curves needs --v2-protocol")
    if args.curves and args.out_name == "matched.json":
        args.out_name = "matched_curves.json"          # never overwrite matched.json
    if not os.environ.get("PBS_JOBID") and not args.allow_login_node:
        raise SystemExit("Refusing to run outside a PBS job -- submit "
                         "paperexp/jobs/paperexp_matched_7m.pbs")

    t0 = time.time()
    est_path = args.out_dir / args.model / "estimate.json"
    est = json.loads(est_path.read_text())
    if not est.get("complete"):
        raise SystemExit(f"{est_path} is incomplete -- run bench_estimate.py first")
    configs = resolve_configs(args.model, "all")
    if args.configs != "all":                      # a smaller catalog (sec:small-catalog)
        keep = args.configs.split(",")
        unknown = [c for c in keep if c not in configs]
        if unknown:
            raise SystemExit(f"--configs: unknown {unknown}; known {configs}")
        configs = [c for c in configs if c in keep]
    costs = bench_costs(args.model, args.cost_dir)
    if args.extra_configs:                         # union catalog (main.tex Table 9, M.R. 2026-10-27)
        extra = args.extra_configs.split(",")
        missing = [c for c in extra for p in costs if c not in costs[p]]
        if missing or set(extra) & set(configs):
            raise SystemExit(f"--extra-configs: not in the cost table or duplicate: {missing or extra}")
        configs = configs + extra
    result = {
        "task": "paper experiments part 4: selection at matched violation (Section 2.4 goal)",
        "model": args.model, "scale": SCALE, "created": datetime.now().isoformat(),
        "host": os.uname().nodename, "pbs_jobid": os.environ.get("PBS_JOBID"),
        "parameters": {"taus": list(TAUS), "rho_static": RHO_STATIC, "K_sweep": list(KS),
                       "q_min": Q_MIN, "cal_fraction": CAL_FRACTION, "split_seed": SPLIT_SEED,
                       "rho_grid": [float(RHO_GRID[0]), float(RHO_GRID[-1]), len(RHO_GRID)],
                       "lambda_grid_s": [float(LAM_GRID[0]), float(LAM_GRID[-1]), len(LAM_GRID)]},
        "inputs": {"c_est": str(est_path),
                   "costs": str((args.cost_dir or R / "throughput" / SCALE) / args.model / "throughput.json"),
                   "online_cells": str(args.assign_dir) if args.assign_dir else "exact (stored assignment)"},
        "protocol": (__doc__.split("V2 PROTOCOL")[1] if args.v2_protocol
                     else __doc__.split("PROTOCOL")[1]).split("Output:")[0].strip(),
        "v2_protocol": bool(args.v2_protocol),
        "pairings": {},
    }
    from selection_tables import CF_TAG
    result["catalog"] = {"configs": configs, "cf_tree": str(R / "cf" / CF_TAG),
                         "costs": str(args.cost_dir) if args.cost_dir else None}
    if args.curves:
        result["task"] = "paper experiments part 4b: RQ4 trade-off curves and reliability bins"
        sel = json.loads((args.out_dir / args.model / "selection.json").read_text())
        mat = json.loads((args.out_dir / args.model / "matched.json").read_text())
        for p in sorted(PAIRINGS):
            result["pairings"][p] = curves_pairing(args.model, p, configs, costs[p], est, sel, mat)
    else:
        for p in sorted(PAIRINGS):
            result["pairings"][p] = run_pairing(args.model, p, configs, costs[p], est,
                                                args.assign_dir, args.viol_cap,
                                                v2=args.v2_protocol)

    dest = args.out_dir / args.model
    tmp = dest / (args.out_name + ".tmp")
    tmp.write_text(json.dumps(result, indent=1))
    tmp.replace(dest / args.out_name)
    log(f"wrote {dest / args.out_name} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
