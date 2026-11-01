"""
Assemble the T10 deliverable: one Markdown report per model, plus the go/no-go
record as a machine-readable file.

Reads only the per-hypothesis JSON each h*.py wrote. It computes nothing of its
own, so the report cannot say something the underlying artifacts do not.

THE GATE IS H5'S ALONE
----------------------
H0-H4 diagnose; H5 decides. The draft is unambiguous (S5.6, S9): if SM does not
clearly beat GA, the conclusion is that H0 does not hold at this scale or
corpus, not that the map needs tuning. So the decision line is copied from H5's
own verdict rather than re-derived here from a blend of all five, which would
let a strong H1 quietly rescue a weak H5.

H0-H4 do change what a PASS MEANS, and the report says so in the caveats block:
a pass with a flat H1 is a win the mechanism does not explain, and a pass whose
H4 curve is still monotone-improving at the finest K tested is a pilot that
never found its own resolution ceiling.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from ht_config import add_common_args, ht_layout, write_json


def _load(ht, name: str, filename: str) -> Optional[dict]:
    p = ht.h(name) / filename
    return json.load(open(p)) if p.exists() else None


def _pairing_label(p: str) -> str:
    return p.replace("_query_", " q -> ").replace("_index", " idx")


def build(ht) -> str:
    R = {
        "H0": _load(ht, "H0", "h0_report.json"),
        "H1": _load(ht, "H1", "h1_report.json"),
        "H2": _load(ht, "H2", "h2_report.json"),
        "H3": _load(ht, "H3", "h3_report.json"),
        "H4": _load(ht, "H4", "h4_report.json"),
        "H5": _load(ht, "H5", "h5_report.json"),
    }
    L = []
    a = L.append
    model = next((r["model"] for r in R.values() if r), ht.model)
    tiling = next((r.get("tiling") for r in R.values() if r and r.get("tiling")), "?")

    a(f"# T10 — Hypothesis testing: {model}")
    a("")
    a(f"*Generated {datetime.now().isoformat(timespec='seconds')} — "
      f"tiling `{tiling}`, all numbers on the held-out role C unless stated.*")
    a("")
    a("Spatial fidelity maps: verification of H0 through H5 on the 1M pilot. "
      "Every predictor is fitted on B (map-construction queries) and scored on "
      "C (test queries), which no construction step touched. H0–H4 diagnose the "
      "mechanism; **H5 alone is the go/no-go gate** before scaling to 7M.")
    a("")

    # ---- the decision, first ---------------------------------------------
    if R["H5"]:
        g = R["H5"]["gate"]
        a("## Decision")
        a("")
        a(f"> ### {g['decision']}")
        a(">")
        a(f"> {g['reading']}")
        a("")
        a("| pairing | verdict |")
        a("|---|---|")
        for p, v in g["verdicts_by_pairing"].items():
            a(f"| {_pairing_label(p)} | {v} |")
        a("")
    else:
        a("## Decision\n\n**H5 has not been run — no gate call available.**\n")

    # ---- H5 ---------------------------------------------------------------
    if R["H5"]:
        a("## H5 — does location help at all?")
        a("")
        a(f"*{R['H5']['question']}. Fitted on B, scored on C. "
          f"Backoff floor {R['H5']['floor']}.*")
        a("")
        a("Mean skill vs GA over the catalogue (skill = 1 − MSE/MSE(GA); GA is "
          "0 by construction):")
        a("")
        a("| pairing | DO | SM shared | SM shared+scale | SM family | "
          "**SM config** | configs where SM clears GA | verdict |")
        a("|---|---|---|---|---|---|---|---|")
        for p, v in R["H5"]["pairings"].items():
            s = v["mean_skill"]
            a(f"| {_pairing_label(p)} | {s['DO']:.3f} | {s['SM_shared']:.3f} | "
              f"{s['SM_shared_scaled']:.3f} | {s['SM_family']:.3f} | "
              f"**{s['SM_config']:.3f}** | "
              f"{v['n_configs_sm_beats_ga']}/{v['n_configs']} | {v['verdict']} |")
        a("")
        a("Decision metrics at each fidelity floor τ, under the "
          "cheapest-clearing policy (cost = T7's measured queries/second):")
        a("")
        a("| pairing | τ | GA shortfall | DO shortfall | SM shortfall | "
          "SM disagreement | SM cost vs oracle |")
        a("|---|---|---|---|---|---|---|")
        for p, v in R["H5"]["pairings"].items():
            for t in R["H5"]["taus"]:
                ga, do, sm = (v["regret"]["GA"][str(t)], v["regret"]["DO"][str(t)],
                              v["regret"]["SM_config"][str(t)])
                if not sm.get("n_configs"):
                    continue
                a(f"| {_pairing_label(p)} | {t} | {ga['mean_shortfall']:.4f} | "
                  f"{do['mean_shortfall']:.4f} | **{sm['mean_shortfall']:.4f}** | "
                  f"{sm['disagreement_rate']:.1%} | "
                  f"×{sm['cost_ratio_map_over_oracle']:.2f} |")
        a("")
        a("*Shortfall is the true CF given up by following the map's pick "
          "instead of the oracle's; lower is better. Cost ratio above 1 means "
          "the map buys its fidelity by escalating to slower configurations.*")
        a("")
        a("Full per-configuration numbers: `H5/h5_metrics.csv`, "
          "`H5/h5_calibration.csv`, `H5/h5_threshold_decisions.csv`, "
          "`H5/h5_regret.csv`. Figures: `H5/figures/`.")
        a("")

    # ---- H0 ---------------------------------------------------------------
    if R["H0"]:
        a("## H0 — is there any signal in location to find?")
        a("")
        a("ICC is the ceiling: the share of held-out CF variance lying between "
          "cells rather than within them, luck-corrected. Skill is what the map "
          "actually extracts. The two coming apart is the difference between "
          "*no signal* and *underfed*.")
        a("")
        a("| pairing | median ICC | median skill | captured | underfed | no signal |")
        a("|---|---|---|---|---|---|")
        for p, v in R["H0"]["pairings"].items():
            a(f"| {_pairing_label(p)} | {v['median_icc']:.3f} | "
              f"{v['median_skill']:.3f} | {v['n_captured']} | "
              f"{v['n_underfed']} | {v['n_no_signal']} |")
        a("")

    # ---- H1 ---------------------------------------------------------------
    if R["H1"]:
        a("## H1 — does error rise with distance and with boundary proximity?")
        a("")
        a("P1's mechanism claim, tested by its own failure predictions. "
          "Correlations of |error| against distance-to-landmark and against "
          "d1/d2, on served (non-backed-off) queries.")
        a("")
        a("| pairing | mean d1 | configs supporting P1 | verdict |")
        a("|---|---|---|---|")
        for p, v in R["H1"]["pairings"].items():
            a(f"| {_pairing_label(p)} | {v['d1_mean']:.3f} | "
              f"{v['n_p1_supported']}/{v['n_configs']} | {v['verdict']} |")
        a("")

    # ---- H2 ---------------------------------------------------------------
    if R["H2"]:
        a("## H2 — boundary stress test")
        a("")
        a("Interior (bottom third of d1/d2) against boundary (top third), each "
          "scored against a GA recomputed inside that subset, so a gap means "
          "the map degrades rather than the population being harder for "
          "everything.")
        a("")
        a("| pairing | share of C at d1/d2 ≥ 0.9 | configs showing degradation | "
          "verdict |")
        a("|---|---|---|---|")
        for p, v in R["H2"]["pairings"].items():
            a(f"| {_pairing_label(p)} | {v['frac_at_or_above_abs_threshold']:.1%} | "
              f"{v['n_boundary_degradation_confirmed']}/{v['n_configs']} | "
              f"{v['verdict']} |")
        a("")

    # ---- H3 ---------------------------------------------------------------
    if R["H3"]:
        a("## H3 — must the map be keyed per configuration?")
        a("")
        a("P2 says HNSW and IVFPQ fail in different neighbourhoods for "
          "different physical reasons. The shared and family rungs carry each "
          "cell's *deviation* from the configuration's own mean, re-levelled "
          "per configuration, so the rungs differ only in how finely geography "
          "is keyed.")
        a("")
        a("| pairing | shared | shared+scale | family | config | "
          "raw gap | **residual after rescaling** | dev. corr within / across | "
          "verdict |")
        a("|---|---|---|---|---|---|---|---|---|")
        for p, v in R["H3"]["pairings"].items():
            s, d = v["mean_skill"], v["deviation_correlation"]
            a(f"| {_pairing_label(p)} | {s['SM_shared']:.3f} | "
              f"{s['SM_shared_scaled']:.3f} | {s['SM_family']:.3f} | "
              f"{s['SM_config']:.3f} | "
              f"{v['config_minus_shared_skill']:+.3f} | "
              f"**{v['config_minus_shared_scaled_skill']:+.3f}** | "
              f"{d['mean_within_family']:.2f} / {d['mean_across_family']:.2f} | "
              f"{v['verdict']} |")
        a("")
        a("*P2 is judged on the **residual**, not the raw gap. The raw gap is "
          "inflated by per-configuration amplitude — a shared column carrying "
          "raw deviations is dominated by the widest-spread configurations and "
          "overwhelms narrow ones regardless of whether it points at the right "
          "cells. `shared+scale` is the same single column of K numbers "
          "expressed as unit-scale shape, with each configuration fitting its "
          "own level and amplitude (2 numbers). Whatever gap survives that is "
          "geography that genuinely differs per configuration, which is what P2 "
          "claims.*")
        a("")

    # ---- H4 ---------------------------------------------------------------
    if R["H4"]:
        a("## H4 — landmark resolution sweep")
        a("")
        a(f"K requested: {R['H4']['sweep']}. Every evaluated point is a fresh "
          f"k-means over the index set A at that K.")
        skipped = R["H4"].get("skipped_k") or {}
        if skipped:
            missing = sorted({k for v in skipped.values() for k in v})
            a("")
            a(f"**Absent from the sweep: K = {missing}.** No k-means tiling "
              f"exists on disk for these, and nothing was substituted for them. "
              f"Build them with `landmarks/build_landmarks.py` then "
              f"`landmarks/occupancy_check.py` and re-run H4 before reading the "
              f"curve as complete.")
        a("")
        a("| pairing | best K | skill there | skill at source K | curve shape |")
        a("|---|---|---|---|---|")
        for p, v in R["H4"]["pairings"].items():
            src = v.get("source_K_skill")
            a(f"| {_pairing_label(p)} | {v['best_K_by_skill']} | "
              f"{v['best_skill']:.3f} | "
              f"{('%.3f' % src) if src is not None else '—'} | {v['shape']} |")
        a("")

    # ---- caveats ----------------------------------------------------------
    a("## Caveats on reading the gate")
    a("")
    caveats = []
    if R["H1"]:
        weak = [p for p, v in R["H1"]["pairings"].items()
                if v["verdict"] == "not-supported"]
        if weak:
            caveats.append(
                f"H1 is flat in {', '.join(_pairing_label(p) for p in weak)}: any "
                f"win there is real but **unexplained by smoothness**, so there "
                f"is no reason to expect it to transfer to another model or to "
                f"7M. Investigate what those cells actually captured before "
                f"leaning on the result.")
    if R["H4"]:
        open_ended = [p for p, v in R["H4"]["pairings"].items()
                      if "monotone-improving" in v["shape"]]
        if open_ended:
            caveats.append(
                f"H4's curve is still improving at the finest K tested in "
                f"{', '.join(_pairing_label(p) for p in open_ended)}: the pilot "
                f"never reached the bottom of its U, so the reported K is a "
                f"floor on the useful resolution, not the optimum.")
        thin = [p for p, v in R["H4"]["pairings"].items() if len(v["per_k"]) < 3]
        if thin:
            caveats.append(
                f"H4 has fewer than three K on disk for "
                f"{', '.join(_pairing_label(p) for p in thin)}, which is not "
                f"enough points to call a curve shape. Build the missing "
                f"tilings and re-run H4 before quoting a K.")
    if R["H0"]:
        underfed = [p for p, v in R["H0"]["pairings"].items()
                    if v["n_underfed"] > v["n_captured"]]
        if underfed:
            caveats.append(
                f"H0 reports most configurations **underfed** in "
                f"{', '.join(_pairing_label(p) for p in underfed)} — the signal "
                f"is there but B cannot estimate it at this K. That is an "
                f"argument for coarser tiling or more B queries, not against H0.")
    if R["H5"]:
        losers = []
        for p, v in R["H5"]["pairings"].items():
            for t in R["H5"]["taus"]:
                ga, sm = v["regret"]["GA"][str(t)], v["regret"]["SM_config"][str(t)]
                if (ga.get("n_configs") and
                        sm["mean_shortfall"] > ga["mean_shortfall"] + 1e-9):
                    losers.append(f"{_pairing_label(p)} at tau={t}")
        if losers:
            caveats.append(
                "**The gate and the decision metric disagree.** The gate reads "
                "skill only, but SM picks WORSE configurations than GA under the "
                "cheapest-clearing policy in: " + "; ".join(losers) + ". Better "
                "CF regression is not translating into better configuration "
                "selection there, and S5.7 argues the decision is the metric "
                "that matters. The gate rule has not been changed to account "
                "for this — read this line alongside the decision above.")
        ties = [p for p, v in R["H5"]["pairings"].items() if not v["sm_beats_do"]]
        if ties:
            caveats.append(
                f"DO matches or beats SM in {', '.join(_pairing_label(p) for p in ties)}. "
                f"Generic corpus density explains as much as the labelled map "
                f"does there, which undercuts the case for paying for the "
                f"per-configuration exact-KNN labelling pass in that pairing.")
    if not caveats:
        caveats.append("None raised by H0–H4.")
    for c in caveats:
        a(f"- {c}")
    a("")

    a("## Provenance")
    a("")
    a("- Predictors, keying rungs and backoff: `predictors.py`")
    a("- Metric definitions, including realized regret's policy: `metrics.py`")
    # H5 records which cost source it used; reports from before --cost-dir
    # existed carry no such field and were all costed from T7.
    a("- Cost basis for τ decisions: "
      + ((R["H5"] or {}).get("cost_basis")
         or "T7's measured `queries_per_second`, role C"))
    a("- T9 artifacts that preceded this task: `imported_from_t9/` "
      "(copies; T9's own tree is untouched)")
    a("")
    return "\n".join(L)


def run(args) -> dict:
    ht = ht_layout(args.model, args.out_dir)
    ht.dest.mkdir(parents=True, exist_ok=True)
    text = build(ht)
    (ht.dest / "REPORT.md").write_text(text)

    h5 = _load(ht, "H5", "h5_report.json")
    gate = {
        "created": datetime.now().isoformat(),
        "model": ht.model,
        "task": "T10 verification, H0-H5",
        "gate": h5["gate"] if h5 else {"decision": "NOT-RUN"},
        "source": "H5/h5_report.json",
        "note": "H5 alone decides. H0-H4 qualify how a pass should be read; "
                "they cannot overturn it in either direction.",
    }
    write_json(ht.dest / "gonogo.json", gate)
    print(f"  wrote {ht.dest / 'REPORT.md'}")
    print(f"  wrote {ht.dest / 'gonogo.json'}   decision: "
          f"{gate['gate'].get('decision')}")
    return gate


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
