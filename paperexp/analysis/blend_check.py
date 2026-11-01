"""Check of 'final1' item 9: cheapest location-blind blend of <= 2 configurations
(matched_violation.static_at_cap: chosen on Q, evaluated on Q_test), (a) at the static plan's
violation rate on Q, vs the static plan; (b) at 1 - rho = 10%, vs the budget rule at the
requirement (cost incl. c_est). tau = 0.7, K = 10,000, main v2 results."""
import json, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]
for s in ("paperexp", "indexes", "spatialmap"):
    sys.path.insert(0, str(REPO / s))
from selection_tables import lay_at, load_overlap, bench_costs
from matched_violation import static_at_cap
from map_config import resolve_configs
PE = REPO / "result_v2/paper_experiments/7M_cleaned/K10000"
DIRS = ["image_query_image_index", "image_query_text_index", "text_query_image_index", "text_query_text_index"]
rows = []
for m in ("siglip2", "openclip", "jinav4", "qwen3vl"):
    cfgs = resolve_configs(m, "all"); costs = bench_costs(m, REPO / "result_v2/throughput/7M_cleaned")
    cap = json.load(open(PE / m / "matched_cap10.json"))["pairings"]
    lay = lay_at(m, 10000); nb = len(np.load(lay.role_ids("B"))); nc = len(np.load(lay.role_ids("C")))
    for p in DIRS:
        order = sorted(cfgs, key=lambda c: (costs[p][c], cfgs.index(c))); cost = np.array([costs[p][c] for c in order])
        ob = load_overlap(lay, p, "B", order, nb); oc = load_overlap(lay, p, "C", order, nc)
        T = cap[p]["by_tau"]["0.70"]; st = T["static"]; vsq = st["viol_Q"]
        a = static_at_cap(ob, oc, cost, order, 70, vsq)
        best_a = min([x for x in (a["single"], a["mix"]) if x], key=lambda x: x["cost_s"]) if (a["single"] or a["mix"]) else None
        b = static_at_cap(ob, oc, cost, order, 70, 0.10)
        best_b = min([x for x in (b["single"], b["mix"]) if x], key=lambda x: x["cost_s"]) if (b["single"] or b["mix"]) else None
        bud = T["by_K"]["10000"]["cost_aware_rule"]
        stored = T.get("static_at_cap", {})
        rows.append({"model": m, "pairing": p, "static_reached": st["reached_rho_on_Q"], "static_viol_Q": vsq,
                     "blend_at_static_rel": best_a["cost_s"] / st["cost_s"] if best_a else None,
                     "blend_at_static": best_a,
                     "blend_at_cap_rel_static": best_b["cost_s"] / st["cost_s"] if best_b else None,
                     "blend_at_cap_viol_test": best_b["viol_test"] if best_b else None,
                     "budget_cap_rel_static": bud["cost_rel_static"], "budget_feasible": bud["feasible_on_tuning"],
                     "budget_vs_blend": (bud["cost_s_total"] / best_b["cost_s"]) if best_b else None,
                     "stored_mix_cost": (stored.get("mix") or {}).get("cost_s"), "my_mix_cost": (b["mix"] or {}).get("cost_s")})
        r = rows[-1]
        print(m, p[:22], {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k not in ("model", "pairing", "blend_at_static")}, flush=True)
json.dump(rows, open(Path(__file__).parent / "blend.json", "w"), indent=1)
