"""
Assign the canonical integer ID for one run, and freeze it.

Every artifact upstream of here keys records by string pair-id
(worker_N/shard_NNNNN/NNNNNNNNN). FAISS addresses vectors by int64, so each
pair needs an integer -- and the SAME integer has to mean the same pair in the
index (T4), the exact-KNN labels (T6), the approximate-KNN results (T7), the CF
table (T8) and the landmark assignment (T9).

WHY THIS IS ONE FILE AND NOT PER-INDEX
    T8 computes CF by intersecting one exact top-k against each configuration's
    approximate top-k. That single exact pass is reused by all 8 configurations
    -- which only works if every index returns IDs in the same space. If each
    index numbered its own rows, the intersection would still produce a number,
    just a wrong one, with no error raised. So: one ID space per run, shared by
    all 8 configurations and all 4 modality pairings.

    An ID names a PAIR. Which vector that is -- the image or the caption -- is
    decided by the configuration's index modality, not by the ID.

WHY IT IS SCOPED TO ONE RUN
    The 1M pilot and the eventual 7M run share nothing: the 7M rebuilds every
    index, re-runs exact-KNN, and uses a different A/B/C split (A >= 6M, B
    ~250-300k, C ~100k -- not a scaled copy of the pilot's). So each run gets
    its own dense 0..N-1 space, tagged with `scope`. Downstream code should
    refuse to mix ID spaces whose scope tags differ -- a pilot ID silently read
    as a 7M ID is exactly the desync this file exists to prevent.

ORDERING
    Ascending (worker, shard, name) -- the corpus's own order, parsed
    numerically rather than lexicographically. Identity is deliberately
    independent of role: re-running the split with a different seed changes the
    role column and leaves every integer untouched.

    python assign_ids.py
"""

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "siglip2"

_PID = re.compile(r"worker_(\d+)/shard_(\d+)/(\d+)$")


def corpus_order_key(pair_id: str):
    m = _PID.match(pair_id)
    if not m:
        raise ValueError(f"unparseable pair id: {pair_id!r}")
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL,
                     help="reference space the split was drawn in; sets the default paths")
    ap.add_argument("--split", type=Path, default=None,
                     help="A/B/C split CSV (pair_id,image_stratum,text_stratum,role)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--meta", type=Path, default=None)
    ap.add_argument("--scope", type=str, default=None,
                     help="tag naming this ID space; IDs from different scopes must never be mixed")
    ap.add_argument("--emit-role-ids", action="store_true", default=True,
                     help="also write {A,B,C}_ids.npy for direct use by FAISS / the KNN passes")
    args = ap.parse_args()

    pilot = REPO_ROOT / "result/pilot_1M" / args.model
    if args.split is None:
        args.split = pilot / "abc_split.csv"
    if args.out is None:
        args.out = pilot / "pair_index.csv"
    if args.meta is None:
        args.meta = pilot / "pair_index_meta.json"
    if args.scope is None:
        args.scope = f"pilot_1M/{args.model}"

    print(f"reading {args.split} ...")
    rows = []
    with open(args.split) as f:
        for r in csv.DictReader(f):
            rows.append((r["pair_id"], int(r["image_stratum"]), int(r["text_stratum"]), r["role"]))
    print(f"  {len(rows):,} pairs")

    rows.sort(key=lambda r: corpus_order_key(r[0]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    by_role = {"A": [], "B": [], "C": []}
    with open(args.out, "w") as f:
        f.write("int_id,pair_id,image_stratum,text_stratum,role\n")
        for n, (pid, i, t, role) in enumerate(rows):
            f.write(f"{n},{pid},{i},{t},{role}\n")
            by_role[role].append(n)

    if args.emit_role_ids:
        for role, ids in by_role.items():
            p = args.out.parent / f"{role}_ids.npy"
            np.save(p, np.asarray(ids, dtype=np.int64))
            print(f"  {role}: {len(ids):,} ids -> {p.name}")

    ranges = {r: ([int(v[0]), int(v[-1])] if v else None) for r, v in by_role.items()}
    args.meta.write_text(json.dumps({
        "created": datetime.now().isoformat(),
        "scope": args.scope,
        "source_split": str(args.split),
        "n_pairs": len(rows),
        "id_space": [0, len(rows) - 1],
        "ordering": "ascending (worker, shard, name), parsed numerically",
        "role_counts": {r: len(v) for r, v in by_role.items()},
        "role_id_ranges": ranges,
        "contract": (
            "int_id names a PAIR within this scope. Which vector it refers to -- "
            "image or caption -- is set by the configuration's index modality, not "
            "by the ID. One ID space serves all 8 index configurations and all 4 "
            "modality pairings, which is what lets the single exact-KNN pass (T6) "
            "be reused by every configuration in T8."
        ),
        "warning": (
            f"These IDs are scoped to '{args.scope}'. Never mix them with another "
            f"run's ID space (e.g. a full-7M run): the two are both dense from 0, "
            f"so a mismatch produces wrong pairings rather than an error."
        ),
        "faiss_note": (
            "Build every index with add_with_ids / IndexIDMap using these ints. "
            "A's ids are non-contiguous within the space (B and C leave gaps); "
            "that is intended, matching the exclusion mask's gap discipline."
        ),
    }, indent=2))

    print(f"wrote {len(rows):,} rows -> {args.out}")
    print(f"      meta -> {args.meta}")
    print(f"  id space: 0..{len(rows)-1}  scope={args.scope!r}")


if __name__ == "__main__":
    main()
