"""
v2 paper experiments part 5: the draft's remaining analyses (user, 2026-10-20:
"compute and plot everything that is required for the v2"), per model at the main
K (SFM_K_MAIN), from the stored labels, assignments and costs. Nothing is re-run
upstream; everything reuses selection_tables / matched_violation, so every
definition is the one the tables use.

  labels      per (direction, tau, configuration): share of Q_test meeting tau, mean
              fidelity; the share of queries whose cheapest sufficient configuration
              is each configuration ("none" when no configuration serves)
              -> RQ1 l.565, intro l.111, catalog appendix l.423, dominated l.401
  geometry    distance of Q_test queries to their nearest landmark (d1), per
              direction -> RQ2 modality-gap measure l.678
  boundary    interior / boundary thirds of d1/d2 (H2's quantile split): MAE, the
              variance of true fidelity, skill against the subset's own mean
              -> RQ2 l.682 (cross-checked against H2's interior/boundary MAE)
  shared_sel  selection with the SHARED map (RQ3 outline): mean rule on
              mu_bar_j + offset_i (the draft's shared map, l.443); coverage and budget
              rules on pi_bar_j + coverage offset_i, offset_i = mean_j(pi_ij - pi_bar_j)
              -- the coverage analogue of the draft's offset, NOT defined in the draft
  sample      construction-sample size (1%, 10%, 100% of Q, seed 0) and q_min in
              {1, 10, 50} at 100% (RQ5 outline): skill, back-off, static plan,
              coverage rule (rho = 0.9) and budget rule (5-fold cross-fitted on the
              subsample), all evaluated on Q_test. A practitioner with the subsample
              also chooses the static plan on it, so the static plan is refitted.
  shortfall   per-query shortfall of the coverage vs the mean rule, Wilcoxon-Pratt
              per direction and pooled per (model, tau) -> l.359's pooled tests

Checks (fail loudly): the 100% / q_min = 10 variant reproduces selection.json (skill,
coverage rule) and matched.json (budget rule); the keyed budget rule recomputed here
equals matched.json; the boundary split reproduces H2's counts and MAE.

Output: <out-dir>/<model>/extras.json
"""
from __future__ import annotations

import argparse
import csv
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
from metrics import wilcoxon_paired                                     # noqa: E402
from selection_tables import (K_MAIN, Q_MIN, R, SCALE, bench_costs,      # noqa: E402
                              cell_coverage, cell_tables, lay_at, load_overlap,
                              pick, static_plan)
from matched_violation import (LAM_GRID, N_FOLDS, SPLIT_SEED, TAUS,      # noqa: E402
                               cost_aware, matched_optimum_static_charge, oof_coverage,
                               tune, viol_cost)

RHO = 0.90
FLAT_SD = 0.03                       # draft l.449, as h5_baselines / the tables
FRACTIONS = (0.01, 0.10, 1.00)
QMINS = (1, 10, 50)
TOL = 1e-9


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def keyed_map(ab, ov, K, t, qmin):
    """Means and coverage per cell, backed off to the sample's global values."""
    n, mu = cell_tables(ab, ov, K)
    thin = n < qmin
    mubar = ov.mean(axis=0) / 100.0
    cov = (ov >= t).mean(axis=0)
    pi = cell_coverage(ab, ov, K, n, t)
    return {"n": n, "thin": thin, "mubar": mubar, "cov": cov,
            "mu": np.where(thin[:, None], mubar[None, :], mu),
            "pi": np.where(thin[:, None], cov[None, :], pi)}


def oof_cov(ab, ov, K, t, folds, qmin, shared=False):
    """Out-of-fold pi_hat on the sample (as matched_violation.oof_coverage, with a
    q_min argument and an optional shared-offset form)."""
    P = np.zeros(ov.shape, dtype=np.float64)
    for f in range(N_FOLDS):
        tr, te = folds != f, folds == f
        M = keyed_map(ab[tr], ov[tr], K, t, qmin)
        pi = M["pi"]
        if shared:
            off = np.where(M["thin"], 0.0, (pi - M["cov"][None, :]).mean(axis=1))
            pi = np.clip(M["cov"][None, :] + off[:, None], 0.0, 1.0)
        P[te] = pi[ab[te]]
    return P


def rule_eval(idx, ov_c, cost, t, cest, c_static, mo_cost):
    vt, ct = viol_cost(idx, ov_c, cost, t)
    tot = ct + cest
    gap = None if mo_cost is None else c_static - mo_cost
    return {"viol": vt, "cost_rel": tot / c_static,
            "rec": (c_static - tot) / gap if gap is not None and gap > 1e-15 else None}


def tune_lam(P, ov, cost, t, target):
    """matched_violation's tuning with its fallback: if no lambda meets the target on
    the tuning data, the setting with the fewest violations (flagged infeasible)."""
    best, trace = tune(LAM_GRID, lambda P_, v: cost_aware(P_, cost, v), P, ov, cost, t, target)
    if best is None:
        return min(trace, key=lambda x: (x[1], x[2]))[0], False
    return best[0], True


def shortfall_vec(idx, ov_c, t):
    f = ov_c[np.arange(len(idx)), idx].astype(np.int64)
    return np.maximum(t - f, 0) / 100.0


def h2_rows(model):
    p = Path(os.environ["SFM_HT_DIR"]) / model / "H2" / "h2_boundary.csv"
    with open(p) as fh:
        return {(r["pairing"], r["config"]): r for r in csv.DictReader(fh)}


def run_pairing(model, pairing, configs, costs, est, sel, mat, h2, short_pool):
    lay = lay_at(model, K_MAIN)
    K = K_MAIN
    order = sorted(configs, key=lambda c: (costs[c], configs.index(c)))
    cost = np.array([costs[c] for c in order], dtype=np.float64)
    n_b = len(np.load(lay.role_ids("B")))
    n_c = len(np.load(lay.role_ids("C")))
    ov_b = load_overlap(lay, pairing, "B", order, n_b)
    ov_c = load_overlap(lay, pairing, "C", order, n_c)
    gB = load_query_geometry(lay, pairing, "B")
    gC = load_query_geometry(lay, pairing, "C")
    ab, ac = gB["assign"].astype(np.int64), gC["assign"].astype(np.int64)
    F = ov_c / 100.0
    flat = F.std(axis=0) < FLAT_SD
    cest = est["pairings"][pairing]["batched_steps_1_3"][str(K)]["s_per_query"]
    out = {"cost_order": order, "cost_s_per_query": cost.tolist(), "c_est_s": cest,
           "flat_configs": [c for c, f in zip(order, flat) if f], "n_Q": n_b, "n_test": n_c}
    log(f"  {pairing}: Q {n_b:,}  Q_test {n_c:,}  K {K}")

    # ---- labels -------------------------------------------------------------------
    lab = {"mean_fidelity": dict(zip(order, F.mean(axis=0).tolist())), "by_tau": {}}
    for tau in TAUS:
        t = int(round(tau * 100))
        ok = ov_c >= t
        none = ~ok.any(axis=1)
        first = np.argmax(ok, axis=1)
        share = np.bincount(first[~none], minlength=len(order)) / n_c
        lab["by_tau"][f"{tau:.2f}"] = {
            "share_meeting": dict(zip(order, ok.mean(axis=0).tolist())),
            "cheapest_sufficient_share": dict(zip(order, share.tolist())),
            "none_share": float(none.mean())}
    out["labels"] = lab

    # ---- geometry --------------------------------------------------------------------
    out["geometry"] = {r: {"d1_mean": float(g["d1"].mean()), "d1_median": float(np.median(g["d1"])),
                           "ratio_mean": float((g["d1"] / g["d2"]).mean())}
                       for r, g in (("B", gB), ("C", gC))}

    # ---- keyed map on all of Q (the paper's map) --------------------------------------
    t70 = 70
    M = keyed_map(ab, ov_b, K, t70, Q_MIN)
    pred = M["mu"][ac]

    # ---- boundary (H2's quantile split of d1/d2) ---------------------------------------
    r = gC["d1"].astype(np.float64) / gC["d2"].astype(np.float64)
    lo, hi = np.quantile(r, 1 / 3), np.quantile(r, 2 / 3)
    bnd = {}
    for name, mask in (("interior", r <= lo), ("boundary", r >= hi)):
        e = pred[mask] - F[mask]
        var = F[mask].var(axis=0)
        mse = (e ** 2).mean(axis=0)
        bnd[name] = {"n": int(mask.sum()),
                     "per_config": {c: {"mae": float(np.abs(e[:, j]).mean()), "var_F": float(var[j]),
                                        "skill": float(1 - mse[j] / var[j]) if var[j] > 0 else None}
                                    for j, c in enumerate(order)}}
    for j, c in enumerate(order):
        h = h2[(pairing, c)]
        for name, key in (("interior", "interior_MAE"), ("boundary", "boundary_MAE")):
            assert int(h[f"n_{name}"]) == bnd[name]["n"], (pairing, c, name, "H2 count")
            assert abs(float(h[key]) - bnd[name]["per_config"][c]["mae"]) < 6e-5, \
                (pairing, c, name, "H2 MAE", h[key], bnd[name]["per_config"][c]["mae"])
    out["boundary"] = bnd

    # ---- selection: keyed vs shared; shortfall; sample size ----------------------------
    folds = np.random.default_rng(SPLIT_SEED).permutation(n_b) % N_FOLDS
    out["shared_sel"], out["shortfall"], out["sample"] = {}, {}, {}
    for tau in TAUS:
        t = int(round(tau * 100))
        tk = f"{tau:.2f}"
        Mk = keyed_map(ab, ov_b, K, t, Q_MIN)
        s, _ = static_plan(Mk["cov"], RHO)
        c_static = float(cost[s])
        v_s_q = float((ov_b[:, s] < t).mean())
        v_s_test = float((ov_c[:, s] < t).mean())
        mo = matched_optimum_static_charge(ov_c, cost, t, int(np.floor(v_s_test * n_c + 1e-9)), s)
        ev = lambda idx: rule_eval(idx, ov_c, cost, t, cest, c_static, mo["cost_s"])   # noqa: E731

        # keyed
        P_k = Mk["pi"][ac]
        mean_k, _ = pick(Mk["mu"][ac] >= tau - 1e-9, Mk["mu"][ac])
        cov_k, _ = pick(P_k >= RHO - 1e-12, P_k)
        lam_k, _ = tune_lam(oof_coverage(ab, ov_b, K, t, folds), ov_b, cost, t, v_s_q)
        bud_k = cost_aware(P_k, cost, lam_k)
        ref = mat["pairings"][pairing]["by_tau"][tk]["by_K"][str(K)]["cost_aware_rule"]
        e_bk = ev(bud_k)
        assert abs(lam_k - ref["param"]) < TOL and abs(e_bk["viol"] - ref["viol_test"]) < TOL, \
            (pairing, tk, "keyed budget vs matched.json")
        sref = sel["pairings"][pairing]["by_tau"][tk]["v2"]["selection_rho_0.90"]["rules"]
        e_ck = ev(cov_k)
        assert abs(e_ck["cost_rel"] - sref["coverage_rule"]["relative_cost"]) < TOL, \
            (pairing, tk, "coverage vs selection.json")
        # shared
        off_mu = np.where(Mk["thin"], 0.0, (Mk["mu"] - Mk["mubar"][None, :]).mean(axis=1))
        mu_sh = (Mk["mubar"][None, :] + off_mu[:, None])[ac]
        off_pi = np.where(Mk["thin"], 0.0, (Mk["pi"] - Mk["cov"][None, :]).mean(axis=1))
        P_sh = np.clip((Mk["cov"][None, :] + off_pi[:, None])[ac], 0.0, 1.0)
        mean_sh, _ = pick(mu_sh >= tau - 1e-9, mu_sh)
        cov_sh, _ = pick(P_sh >= RHO - 1e-12, P_sh)
        lam_sh, feas_sh = tune_lam(oof_cov(ab, ov_b, K, t, folds, Q_MIN, shared=True), ov_b, cost, t, v_s_q)
        out["shared_sel"][tk] = {
            "static": {"config": order[s], "viol_test": v_s_test},
            "matched_optimum_rel": None if mo["cost_s"] is None else mo["cost_s"] / c_static,
            "keyed": {"mean_rule": ev(mean_k), "coverage_rule": e_ck, "budget_rule": e_bk},
            "shared": {"mean_rule": ev(mean_sh), "coverage_rule": ev(cov_sh),
                       "budget_rule": {**ev(cost_aware(P_sh, cost, lam_sh)),
                                       "lambda": lam_sh, "feasible_on_tuning": feas_sh}}}

        # shortfall, coverage vs mean rule (keyed)
        sc, sm = shortfall_vec(cov_k, ov_c, t), shortfall_vec(mean_k, ov_c, t)
        short_pool.setdefault(tk, []).append((sc, sm))
        out["shortfall"][tk] = {"coverage_mean": float(sc.mean()), "mean_mean": float(sm.mean()),
                                "pratt": wilcoxon_paired(sc, sm, zero_method="pratt")}

        # sample size / q_min
        S = {}
        for frac in FRACTIONS:
            for qmin in (QMINS if frac == 1.0 else (Q_MIN,)):
                rng = np.random.default_rng(SPLIT_SEED)
                sub = np.sort(rng.permutation(n_b)[:int(round(frac * n_b))]) if frac < 1 else np.arange(n_b)
                a_s, o_s = ab[sub], ov_b[sub]
                Ms = keyed_map(a_s, o_s, K, t, qmin)
                pr = Ms["mu"][ac]
                mse = ((pr - F) ** 2).mean(axis=0)
                mse_ga = ((Ms["mubar"][None, :] - F) ** 2).mean(axis=0)
                skill = 1 - mse / mse_ga
                ss, _ = static_plan(Ms["cov"], RHO)
                cs = float(cost[ss])
                vs_q = float((o_s[:, ss] < t).mean())
                vs_t = float((ov_c[:, ss] < t).mean())
                mo_s = matched_optimum_static_charge(ov_c, cost, t, int(np.floor(vs_t * n_c + 1e-9)), ss)
                Ps = Ms["pi"][ac]
                ci, _ = pick(Ps >= RHO - 1e-12, Ps)
                fs = np.random.default_rng(SPLIT_SEED).permutation(len(sub)) % N_FOLDS
                lam_s, feas_s = tune_lam(oof_cov(a_s, o_s, K, t, fs, qmin), o_s, cost, t, vs_q)
                bi = cost_aware(Ps, cost, lam_s)
                evs = lambda idx: rule_eval(idx, ov_c, cost, t, cest, cs, mo_s["cost_s"])  # noqa: E731
                key = f"f{frac:g}_q{qmin}"
                S[key] = {"fraction": frac, "q_min": qmin, "n_sample": int(len(sub)),
                          "skill_all": float(skill.mean()),
                          "skill_nonflat": float(skill[~flat].mean()) if (~flat).any() else None,
                          "backoff_test": float(Ms["thin"][ac].mean()),
                          "static": {"config": order[ss], "viol_test": vs_t},
                          "matched_optimum_rel": None if mo_s["cost_s"] is None else mo_s["cost_s"] / cs,
                          "coverage_rule": evs(ci), "budget_rule": {**evs(bi), "lambda": lam_s,
                                                          "feasible_on_tuning": feas_s}}
        full = S[f"f1_q{Q_MIN}"]
        assert abs(full["coverage_rule"]["cost_rel"] - e_ck["cost_rel"]) < TOL
        assert abs(full["budget_rule"]["cost_rel"] - e_bk["cost_rel"]) < TOL
        if tau == TAUS[0]:
            sk = sel["pairings"][pairing]["by_K"][str(K)]["skill"]
            assert abs(full["skill_all"] - sk) < 1e-6, (pairing, "skill vs selection.json", full["skill_all"], sk)
        out["sample"][tk] = S
        log(f"    tau={tk}: static {order[s]}; shared cov {out['shared_sel'][tk]['shared']['coverage_rule']['cost_rel']:.2f}x"
            f" vs keyed {e_ck['cost_rel']:.2f}x; 1% skill {S['f0.01_q10']['skill_all']:.3f}"
            f" / 100% {full['skill_all']:.3f}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=["siglip2", "qwen3vl", "openclip", "jinav4"])
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--cost-dir", type=Path, required=True)
    args = ap.parse_args()
    if not os.environ.get("PBS_JOBID"):
        raise SystemExit("Refusing to run outside a PBS job -- paperexp/jobs/paperexp_extras_v2.pbs")
    t0 = time.time()
    d = args.out_dir / args.model
    est = json.loads((d / "estimate.json").read_text())
    sel = json.loads((d / "selection.json").read_text())
    mat = json.loads((d / "matched.json").read_text())
    configs = resolve_configs(args.model, "all")
    costs = bench_costs(args.model, args.cost_dir)
    h2 = h2_rows(args.model)
    res = {"task": "paper experiments part 5: remaining draft analyses (extras_v2.py)",
           "model": args.model, "scale": SCALE, "K": K_MAIN, "created": datetime.now().isoformat(),
           "host": os.uname().nodename, "pbs_jobid": os.environ.get("PBS_JOBID"),
           "parameters": {"rho": RHO, "q_min": Q_MIN, "flat_sd": FLAT_SD, "fractions": FRACTIONS,
                          "q_mins": QMINS, "taus": TAUS, "folds": N_FOLDS, "seed": SPLIT_SEED},
           "definitions": __doc__.split("Checks")[0].strip(), "pairings": {}}
    pool = {}
    for p in sorted(PAIRINGS):
        res["pairings"][p] = run_pairing(args.model, p, configs, costs[p], est, sel, mat, h2, pool)
    res["shortfall_pooled"] = {
        tk: wilcoxon_paired(np.concatenate([a for a, _ in v]), np.concatenate([b for _, b in v]),
                            zero_method="pratt") | {
            "coverage_mean": float(np.concatenate([a for a, _ in v]).mean()),
            "mean_mean": float(np.concatenate([b for _, b in v]).mean())}
        for tk, v in pool.items()}
    tmp = d / "extras.json.tmp"
    tmp.write_text(json.dumps(res, indent=1))
    tmp.replace(d / "extras.json")
    log(f"wrote {d / 'extras.json'} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
