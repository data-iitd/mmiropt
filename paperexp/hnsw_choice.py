"""
Which HNSW setting serves the landmark lookup, per direction -- one rule, shared by
paperexp/hnsw_assign.py and writeup/7M_cleaned/make_paper_tables.py. Stdlib only.

Input: a split_lookup_variants[_K<k>].json from split_single_query.py --hnsw-sweep.

Rule: one HNSW graph per landmark set, so one M per set (image landmarks serve
i->i and t->i; text landmarks serve i->t and t->t). Per direction, the fastest
efSearch (single-query Step 1) whose chosen configuration matches the exact
scan's for >= AGREE of the benchmark queries; where no setting does, the exact
scan ("flat"). M is the one that minimises the mean single-query Step 1 over the
set's two directions.
"""
from __future__ import annotations

AGREE = 0.999
BY_INDEX = (("image_query_image_index", "text_query_image_index"),    # image landmarks
            ("image_query_text_index", "text_query_text_index"))      # text landmarks


def step1_us(pairing: dict, name: str) -> float:
    v = pairing["flat"] if name == "flat" else pairing["hnsw"][name]
    return v["single_1"]["step1_search"]["mean_us"]


def choose(sweep: dict, agree: float = AGREE) -> dict:
    """{pairing: "flat" | "M<m>_ef<ef>"} for one model's sweep."""
    P, out = sweep["pairings"], {}
    for dirs in BY_INDEX:
        best = None
        for M in sorted({v["M"] for v in P[dirs[0]]["hnsw"].values()}):
            pick = {}
            for p in dirs:
                ok = [(step1_us(P[p], nm), nm) for nm, v in P[p]["hnsw"].items()
                      if v["M"] == M and v["pick_agreement"] >= agree]
                pick[p] = min(ok)[1] if ok else "flat"
            t = sum(step1_us(P[p], nm) for p, nm in pick.items()) / len(pick)
            if best is None or t < best[0]:
                best = (t, pick)
        out.update(best[1])
    return out


def choose_uniform(sweep: dict, agree: float = AGREE, max_fail: int = 2) -> dict:
    """{pairing: "M<m>_ef<ef>"}, ONE setting for all four directions of a model (user
    2026-10-31): the fastest mean single-query Step 1 over the four directions among the
    settings whose pick agreement is >= agree in all but at most max_fail directions.
    No exact-scan fallback; "flat" everywhere only if no setting qualifies."""
    P = sweep["pairings"]
    names = list(P[next(iter(P))]["hnsw"])
    ok = [nm for nm in names
          if sum(P[p]["hnsw"][nm]["pick_agreement"] < agree for p in P) <= max_fail]
    if not ok:
        return {p: "flat" for p in P}
    best = min(ok, key=lambda nm: sum(step1_us(P[p], nm) for p in P) / len(P))
    return {p: best for p in P}


def parse(name: str):
    """"M16_ef64" -> (16, 64); "flat" -> None."""
    if name == "flat":
        return None
    M, ef = name[1:].split("_ef")
    return int(M), int(ef)


def sweep_name(k: int) -> str:
    return "split_lookup_variants.json" if k == 2500 else f"split_lookup_variants_K{k}.json"
