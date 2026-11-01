"""
v2 single-index catalog (main.tex sec:small-catalog, user 2026-10-26): ONE HNSW
index per model and modality -- the build of hnsw_high (M = 16, efC = 40) --
searched at three efSearch values, so per-query selection adds no memory.

Creates, and never touches the main tree:
  result_v2/indexes/7M_cleaned_single/<model>/<modality>/
      hnsw_{low,mod,high}.faiss       symlinks to ../7M_cleaned/.../hnsw_high.faiss
      hnsw_{low,mod,high}.build.json  copies of hnsw_high.build.json (+ a note)
      <ivf configs>.build.json        copies of the main ones -- configs_v2 reads
                                      all nine build files; no IVF index is linked
  result_v2/param_check/7M_cleaned/single_index_params.json
      final_params_chosen.json with the three HNSW search values replaced.

The names hnsw_low/mod/high are kept so that every v2 script (approx_knn.py,
cf_compute.py, bench_throughput.py) runs unchanged; inside this tree they mean
"the hnsw_high build at efSearch low/mod/high". Keep the outputs in *_single trees.

efSearch rule (user, 2026-10-26: "the same for a given model ... lets keep it that
way"): one value per model for all four directions, the first value of the 5.8M
search-parameter sweep (result_v2/param_check/7M_cleaned/<model>/<modality>/
hnsw_high.csv, 5,000 role-B queries) whose mean fidelity over the model's four
directions reaches 0.70 / 0.80 / 0.85. Directions that do not reach a target stay
below it.

    .venv-faisscuvs/bin/python knn/setup_single_index_v2.py          (light: links + JSON)
"""
from __future__ import annotations

import csv
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
V2 = REPO / "result_v2"
MAIN = V2 / "indexes" / "7M_cleaned"
SINGLE = V2 / "indexes" / "7M_cleaned_single"
CHOSEN = V2 / "param_check" / "7M_cleaned" / "final_params_chosen.json"
OUT_PARAMS = V2 / "param_check" / "7M_cleaned" / "single_index_params.json"
MODELS = ("siglip2", "openclip", "qwen3vl", "jinav4")
CONFIGS = [f"{f}_{t}" for f in ("hnsw", "ivfpq", "oivfpq") for t in ("low", "mod", "high")]
TARGETS = {"hnsw_low": 0.70, "hnsw_mod": 0.80, "hnsw_high": 0.85}
DIRS = {("image", "same"): "i->i", ("image", "cross"): "t->i",
        ("text", "same"): "t->t", ("text", "cross"): "i->t"}


def sweep(model: str) -> dict:
    """efSearch -> {direction: mean CF} from the 5.8M check of hnsw_high."""
    F = {}
    for x in ("image", "text"):
        for r in csv.DictReader(open(V2 / "param_check" / "7M_cleaned" / model / x / "hnsw_high.csv")):
            v = int(r["value"])
            F.setdefault(v, {})[DIRS[(x, "same")]] = float(r["cf_same"])
            F[v][DIRS[(x, "cross")]] = float(r["cf_cross"])
    return F


def main():
    chosen = json.loads(CHOSEN.read_text())
    params = json.loads(json.dumps(chosen["params"]))
    picks = {}
    for m in MODELS:
        F = sweep(m)
        vals = sorted(v for v in F if len(F[v]) == 4)
        mean4 = {v: sum(F[v].values()) / 4 for v in vals}
        picks[m] = {}
        for c, t in TARGETS.items():
            v = next((x for x in vals if mean4[x] >= t - 1e-12), None)
            if v is None:
                raise SystemExit(f"{m}: no efSearch reaches mean fidelity {t}")
            picks[m][c] = {"efSearch": v, "mean_cf_4dirs": round(mean4[v], 6),
                           "cf_per_direction": {d: round(f, 6) for d, f in F[v].items()}}
            params[c][m] = {"value": v, "param": "efSearch", "rule": f"single index: first sweep value "
                            f"with mean CF over 4 directions >= {t}", "index_file": "hnsw_high build"}
    for m in MODELS:
        for x in ("image", "text"):
            src, dst = MAIN / m / x, SINGLE / m / x
            dst.mkdir(parents=True, exist_ok=True)
            hb = json.loads((src / "hnsw_high.build.json").read_text())
            for c in CONFIGS:
                bj = dst / f"{c}.build.json"
                if c.startswith("hnsw"):
                    link = dst / f"{c}.faiss"
                    if link.is_symlink() or link.exists():
                        assert os.path.realpath(link) == os.path.realpath(src / "hnsw_high.faiss"), link
                    else:
                        link.symlink_to(src / "hnsw_high.faiss")
                    b = dict(hb, config=c, single_index_note=(
                        f"single-index variant: {c} = the hnsw_high build of {src} searched at "
                        f"efSearch {picks[m][c]['efSearch']}"))
                    bj.write_text(json.dumps(b, indent=2))
                else:
                    shutil.copyfile(src / f"{c}.build.json", bj)
    OUT_PARAMS.write_text(json.dumps({
        "task": "v2 single-index catalog: hnsw_high build at three efSearch values per model",
        "created": datetime.now().isoformat(), "derived_from": str(CHOSEN),
        "rule": __doc__.split("efSearch rule")[1].split("    .venv")[0].strip(),
        "index_root": str(SINGLE), "picks": picks, "n_queries": chosen.get("n_queries"),
        "role": chosen.get("role"), "params": params}, indent=1))
    for m in MODELS:
        print(m, {c: p["efSearch"] for c, p in picks[m].items()})
    print(f"wrote {OUT_PARAMS} and {SINGLE}")


def combine():
    """--combine (user, 2026-10-27, option A): the single index's HIGH value becomes the
    main catalog's hnsw_high (the same build at its usual efSearch), so that its static
    plan can reach 90% wherever the full catalog's hnsw_high does; low and mod stay as
    chosen above. Nothing is re-run and nothing existing is changed; this writes
      result_v2/cf/7M_cleaned_single_b/<m>/      links: low/mod -> 7M_cleaned_single,
                                                        high    -> 7M_cleaned (main T8)
      result_v2/throughput/7M_cleaned_single_b/<m>/throughput.json
          low/mod from 7M_cleaned_single, high from 7M_cleaned_single_ref -- the main
          hnsw_high re-timed in the SAME job and node as low/mod
      result_v2/param_check/7M_cleaned/single_index_params_b.json"""
    pairs = ("image_query_image_index", "image_query_text_index",
             "text_query_image_index", "text_query_text_index")
    cfroot, tproot = V2 / "cf", V2 / "throughput"
    sp = json.loads(OUT_PARAMS.read_text())
    main_p = json.loads(CHOSEN.read_text())["params"]
    for m in MODELS:
        d = cfroot / "7M_cleaned_single_b" / m
        d.mkdir(parents=True, exist_ok=True)
        for p in pairs:
            for r in ("B", "C"):
                for c, tree in (("hnsw_low", "7M_cleaned_single"), ("hnsw_mod", "7M_cleaned_single"),
                                ("hnsw_high", "7M_cleaned")):
                    f = f"cf_{p}_{r}_{c}.npy"
                    src, link = cfroot / tree / m / f, d / f
                    assert src.exists(), src
                    if link.is_symlink() or link.exists():
                        assert os.path.realpath(link) == os.path.realpath(src), link
                    else:
                        link.symlink_to(src)
        lo = json.loads((tproot / "7M_cleaned_single" / m / "throughput.json").read_text())
        hi = json.loads((tproot / "7M_cleaned_single_ref" / m / "throughput.json").read_text())
        assert lo.get("complete") and hi.get("complete"), m
        cost = {p: {"hnsw_low": lo["cost_s_per_query"][p]["hnsw_low"],
                    "hnsw_mod": lo["cost_s_per_query"][p]["hnsw_mod"],
                    "hnsw_high": hi["cost_s_per_query"][p]["hnsw_high"]} for p in pairs}
        out = tproot / "7M_cleaned_single_b" / m
        out.mkdir(parents=True, exist_ok=True)
        (out / "throughput.json").write_text(json.dumps({
            "task": "single-index catalog costs, option A (high = main hnsw_high)",
            "model": m, "complete": True, "created": datetime.now().isoformat(),
            "sources": {"hnsw_low, hnsw_mod": str(tproot / "7M_cleaned_single" / m / "throughput.json"),
                        "hnsw_high": str(tproot / "7M_cleaned_single_ref" / m / "throughput.json")},
            "same_node": lo["environment"]["host"] == hi["environment"]["host"],
            "hosts": [lo["environment"]["host"], hi["environment"]["host"]],
            "cost_s_per_query": cost}, indent=1))
        sp["params"]["hnsw_high"][m] = dict(main_p["hnsw_high"][m], note="option A: the main hnsw_high")
        sp["picks"][m]["hnsw_high"] = {"efSearch": main_p["hnsw_high"][m]["value"], "source": "main catalog hnsw_high"}
    sp["task"] += " -- option A: high = main hnsw_high"
    sp["created_b"] = datetime.now().isoformat()
    (OUT_PARAMS.parent / "single_index_params_b.json").write_text(json.dumps(sp, indent=1))
    print("combined: cf/7M_cleaned_single_b, throughput/7M_cleaned_single_b, single_index_params_b.json")


def rescale():
    """--rescale (2026-10-27, to check main.tex sec:small-catalog, whose text puts the
    single-index costs "on the scale of the main benchmark through hnsw_high"): every
    single_b cost of a direction x (main-benchmark hnsw_high / same-node hnsw_high).
    Writes result_v2/throughput/7M_cleaned_single_c/<m>/throughput.json only; the labels
    are the single_b links (cf/7M_cleaned_single_b)."""
    tp = V2 / "throughput"
    for m in MODELS:
        b = json.loads((tp / "7M_cleaned_single_b" / m / "throughput.json").read_text())
        main_c = json.loads((tp / "7M_cleaned" / m / "throughput.json").read_text())["cost_s_per_query"]
        ref = json.loads((tp / "7M_cleaned_single_ref" / m / "throughput.json").read_text())["cost_s_per_query"]
        f = {p: main_c[p]["hnsw_high"] / ref[p]["hnsw_high"] for p in b["cost_s_per_query"]}
        cost = {p: {c: v * f[p] for c, v in cs.items()} for p, cs in b["cost_s_per_query"].items()}
        out = tp / "7M_cleaned_single_c" / m
        out.mkdir(parents=True, exist_ok=True)
        (out / "throughput.json").write_text(json.dumps({
            "task": "single-index costs (option A) rescaled to the main benchmark through hnsw_high",
            "model": m, "complete": True, "created": datetime.now().isoformat(),
            "source": str(tp / "7M_cleaned_single_b" / m / "throughput.json"),
            "factor_main_over_same_node_hnsw_high": f, "cost_s_per_query": cost}, indent=1))
    print("rescaled: throughput/7M_cleaned_single_c")


UNION_EXTRA = {"hsi_low": "hnsw_low", "hsi_mod": "hnsw_mod"}   # union name -> single-index name


def union():
    """--union (Maya Ramanath, 2026-10-27, main.tex Table 9): the 9 configurations of the main
    catalog plus the single index's low and mod (its high IS the main hnsw_high), 11 in all.
    Writes only new trees:
      result_v2/cf/7M_cleaned_union/<m>/   links: 9 -> cf/7M_cleaned, hsi_low/hsi_mod ->
                                           cf/7M_cleaned_single (hnsw_low/hnsw_mod files)
      result_v2/throughput/7M_cleaned_union/<m>/throughput.json
          the 9 main-benchmark costs + hsi_low/hsi_mod from 7M_cleaned_single_c (rescaled to
          the main benchmark, as Table 9's single-index columns)"""
    pairs = ("image_query_image_index", "image_query_text_index",
             "text_query_image_index", "text_query_text_index")
    cfr, tp = V2 / "cf", V2 / "throughput"
    for m in MODELS:
        d = cfr / "7M_cleaned_union" / m
        d.mkdir(parents=True, exist_ok=True)
        links = [(f"cf_{p}_{r}_{c}.npy", cfr / "7M_cleaned" / m / f"cf_{p}_{r}_{c}.npy")
                 for p in pairs for r in ("B", "C") for c in CONFIGS]
        links += [(f"cf_{p}_{r}_{u}.npy", cfr / "7M_cleaned_single" / m / f"cf_{p}_{r}_{c}.npy")
                  for p in pairs for r in ("B", "C") for u, c in UNION_EXTRA.items()]
        for name, src in links:
            assert src.exists(), src
            link = d / name
            if link.is_symlink() or link.exists():
                assert os.path.realpath(link) == os.path.realpath(src), link
            else:
                link.symlink_to(src)
        main_c = json.loads((tp / "7M_cleaned" / m / "throughput.json").read_text())["cost_s_per_query"]
        sc = json.loads((tp / "7M_cleaned_single_c" / m / "throughput.json").read_text())["cost_s_per_query"]
        cost = {p: dict({c: main_c[p][c] for c in CONFIGS},
                        **{u: sc[p][c] for u, c in UNION_EXTRA.items()}) for p in pairs}
        out = tp / "7M_cleaned_union" / m
        out.mkdir(parents=True, exist_ok=True)
        (out / "throughput.json").write_text(json.dumps({
            "task": "union catalog costs: main 9 + single-index low/mod (rescaled to the main benchmark)",
            "model": m, "complete": True, "created": datetime.now().isoformat(),
            "sources": {"9 main": str(tp / "7M_cleaned" / m / "throughput.json"),
                        "hsi_low, hsi_mod": str(tp / "7M_cleaned_single_c" / m / "throughput.json")},
            "cost_s_per_query": cost}, indent=1))
    print("union: cf/7M_cleaned_union, throughput/7M_cleaned_union (11 configurations)")


if __name__ == "__main__":
    import sys
    (union() if "--union" in sys.argv else rescale() if "--rescale" in sys.argv
     else combine() if "--combine" in sys.argv else main())
