"""Same-plan rate of the HNSW landmark lookup under all three rules (user 2026-10-31).
Map: built from Q (B) with the EXACT cells, K=10,000, k=100 labels, q_min back-off (as the
paper). For every Q_test (C) query, the rule's pick from its exact cell vs from its HNSW
cell (hnsw_assign output). Rules: mean (mu_hat >= tau), coverage (pi_hat >= 0.9), budget
argmin c_j + lambda (1 - pi_hat_j) with the stored lambda: at the static plan's rate
(matched.json) and at the 10% requirement (matched_cap10.json). Costs: the exclusive v2
benchmark (c_est is the same for every configuration, so it does not change the pick)."""
import json, os, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]
for s in ("paperexp", "indexes", "knn", "spatialmap"):
    sys.path.insert(0, str(REPO / s))
from selection_tables import lay_at, load_overlap, bench_costs, cell_tables, cell_coverage, pick, Q_MIN
from map_config import resolve_configs, load_query_geometry
from matched_violation import cost_aware
K = 10000; PE = REPO / 'result_v2/paper_experiments'
TREES = [t for t in ("7M_cleaned_lmhnsw", "7M_cleaned_lmhnsw_uniform")
         if all((PE / t / f"K{K}" / m / "hnsw_assign.json").exists() for m in ("siglip2", "openclip", "jinav4", "qwen3vl"))]
DIRS = ["image_query_image_index", "image_query_text_index", "text_query_image_index", "text_query_text_index"]
out = {"doc": __doc__, "trees": TREES, "rows": []}
for m in ("siglip2", "openclip", "jinav4", "qwen3vl"):
    cfgs = resolve_configs(m, "all"); costs = bench_costs(m, REPO / "result_v2/throughput/7M_cleaned")
    mat = json.load(open(PE / "7M_cleaned" / f"K{K}" / m / "matched.json"))["pairings"]
    cap = json.load(open(PE / "7M_cleaned" / f"K{K}" / m / "matched_cap10.json"))["pairings"]
    lay = lay_at(m, K)
    n_b = len(np.load(lay.role_ids("B")))
    for p in DIRS:
        order = sorted(cfgs, key=lambda c: (costs[p][c], cfgs.index(c)))
        assert order == mat[p]["cost_order"] == cap[p]["cost_order"]
        cost = np.array([costs[p][c] for c in order])
        ov_b = load_overlap(lay, p, "B", order, n_b)
        ab = load_query_geometry(lay, p, "B")["assign"].astype(np.int64)
        ac = load_query_geometry(lay, p, "C")["assign"].astype(np.int64)
        n, mu = cell_tables(ab, ov_b, K); thin = n < Q_MIN
        mu_hat = np.where(thin[:, None], (ov_b.mean(axis=0) / 100.0)[None, :], mu)
        hc = {t: np.load(PE / t / f"K{K}" / m / "hnsw_assign" / f"K{K}" / f"{p}_C.npy").astype(np.int64) for t in TREES}
        for tau in ("0.70", "0.80", "0.90"):
            tf = float(tau); ti = int(round(tf * 100))
            pi = cell_coverage(ab, ov_b, K, n, ti)
            pi_hat = np.where(thin[:, None], (ov_b >= ti).mean(axis=0)[None, :], pi)
            lam_s = mat[p]["by_tau"][tau]["by_K"][str(K)]["cost_aware_rule"]["param"]
            lam_c = cap[p]["by_tau"][tau]["by_K"][str(K)]["cost_aware_rule"]["param"]
            rules = {"mean": lambda a: pick(mu_hat[a] >= tf - 1e-9, mu_hat[a])[0],
                     "coverage": lambda a: pick(pi_hat[a] >= 0.9 - 1e-12, pi_hat[a])[0],
                     "budget_static": lambda a: cost_aware(pi_hat[a], cost, lam_s),
                     "budget_cap": lambda a: cost_aware(pi_hat[a], cost, lam_c)}
            ex = {r: f(ac) for r, f in rules.items()}
            for t in TREES:
                row = {"model": m, "pairing": p, "tau": tau, "tree": t,
                       "same_cell": float((hc[t] == ac).mean()), "n": int(len(ac))}
                for r, f in rules.items():
                    row["same_plan_" + r] = float((f(hc[t]) == ex[r]).mean())
                out["rows"].append(row)
                print(m, p[:22], tau, t[-8:], " ".join(f"{k}={v:.4f}" for k, v in row.items() if k.startswith("same")), flush=True)
json.dump(out, open(Path(__file__).parent / "same_plan.json", "w"), indent=1)
