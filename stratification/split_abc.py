"""
T3 -- partition pairs into the three disjoint roles, within each joint stratum.

    A  index set             ~85%
    B  map-construction queries ~10%
    C  held-out test queries     ~5%

Per stratum: integer floors at 85/10/5, then each leftover unit is handed out
at random weighted by the same ratios (unbiased -- the global split lands on
85/10/5 to within a few hundredths of a percent).

That alone starves the tail: for any stratum under 20 pairs, C's exact share
is below 1, so all three floors round C to zero and it can only be reached
through the remainder lottery at p=0.05. So a repair follows: if a stratum has
at least 3 pairs and B or C came out empty, one pair moves over from A. This
costs ~28 pairs corpus-wide and is the only thing that makes the paper's
"guaranteed representation in all three sets" literally true. Strata of 1-2
pairs cannot be split three ways and go to A, where they still contribute the
sparse geometry that CF is measured against.

Runs on any labelled pair list, so it serves both the 1M pilot and, later, the
full corpus:

    python split_abc.py                                   # the 1M pilot
    python split_abc.py --image-labels ... --text-labels ... \
        --out ../result/full_7M/abc_split.csv             # the whole corpus
"""

import argparse
import random
from collections import Counter
from datetime import datetime
from pathlib import Path

from strata_io import (
    open_pairs, count_strata, largest_remainder, shuffled_sequence, Dealer, write_meta,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "siglip2"

ROLES = ("A", "B", "C")
RATIOS = (0.85, 0.10, 0.05)
MIN_SPLITTABLE = 3


def allocate(n: int, rng: random.Random, ratios=RATIOS):
    """
    Returns (a, b, c, repaired, unsplittable) for one stratum of n pairs.
    """
    if n < MIN_SPLITTABLE:
        return n, 0, 0, False, True

    a, b, c = largest_remainder(n, ratios, rng)

    repaired = False
    if c == 0 and a > 0:
        a, c = a - 1, 1
        repaired = True
    if b == 0 and a > 0:
        a, b = a - 1, 1
        repaired = True
    return a, b, c, repaired, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL,
                     help="reference space the pilot was drawn in; sets the default paths")
    ap.add_argument("--input", type=Path, default=None,
                     help="labelled pair CSV (pair_id,image_stratum,text_stratum)")
    ap.add_argument("--image-labels", type=Path, default=None,
                     help="alternative input: the per-modality label CSVs (full corpus, no sampling)")
    ap.add_argument("--text-labels", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--meta", type=Path, default=None)
    args = ap.parse_args()

    pilot = REPO_ROOT / "result/pilot_1M" / args.model
    if args.out is None:
        args.out = pilot / "abc_split.csv"
    if args.meta is None:
        args.meta = pilot / "abc_split_meta.json"

    if args.image_labels and args.text_labels:
        args.input = None   # strata_io.open_pairs then reads the label pair
    elif args.input is None:
        args.input = pilot / "sample.csv"

    print("pass 1: counting pairs per joint stratum ...")
    counts = count_strata(open_pairs(args))
    total = sum(counts.values())
    print(f"  {len(counts)} strata, {total:,} pairs")

    rng = random.Random(args.seed)
    seqs, n_repaired, n_unsplittable = {}, 0, 0
    planned = Counter()
    for k in sorted(counts):
        a, b, c, repaired, unsplittable = allocate(counts[k], rng)
        n_repaired += repaired
        n_unsplittable += unsplittable
        planned.update({"A": a, "B": b, "C": c})
        seqs[k] = shuffled_sequence({"A": a, "B": b, "C": c}, rng)
    dealer = Dealer(seqs)

    print(f"  planned  A={planned['A']:,} B={planned['B']:,} C={planned['C']:,}")
    print(f"  strata repaired (B or C was empty): {n_repaired}")
    print(f"  strata too small to split (<{MIN_SPLITTABLE} pairs, all to A): {n_unsplittable}")

    print("pass 2: writing the split ...")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    realized = Counter()
    with open(args.out, "w") as f:
        f.write("pair_id,image_stratum,text_stratum,role\n")
        for pid, i, t in open_pairs(args):
            role = dealer.take((i, t))
            realized[role] += 1
            f.write(f"{pid},{i},{t},{role}\n")
    dealer.assert_exhausted()

    empty_c = sum(1 for k in counts if seqs[k].count("C") == 0)
    empty_b = sum(1 for k in counts if seqs[k].count("B") == 0)

    for r in ROLES:
        print(f"  {r}: {realized[r]:,} ({100*realized[r]/total:.3f}%)")
    print(f"  strata with no C: {empty_c}   with no B: {empty_b}")

    write_meta(args.meta, {
        "created": datetime.now().isoformat(),
        "seed": args.seed,
        "source": str(args.input) if args.input else f"{args.image_labels} + {args.text_labels}",
        "pairs": total,
        "strata": len(counts),
        "ratios": dict(zip(ROLES, RATIOS)),
        "realized": {r: realized[r] for r in ROLES},
        "realized_pct": {r: round(100 * realized[r] / total, 4) for r in ROLES},
        "strata_repaired": n_repaired,
        "strata_unsplittable": n_unsplittable,
        "strata_without_C": empty_c,
        "strata_without_B": empty_b,
        "method": (
            "Per stratum: integer floors at 85/10/5, leftover units assigned at "
            "random weighted by the same ratios. Role order within a stratum is "
            "a seeded shuffle, so role never correlates with corpus order. "
            "Split is at the pair level: a pair's image and caption always take "
            "the same role, so no query's partner is ever in the index."
        ),
        "deviation": (
            f"{n_repaired} strata had B or C land empty under stochastic rounding "
            f"and had one pair moved over from A, so every stratum with >=3 pairs "
            f"appears in all three roles. {n_unsplittable} strata hold 1-2 pairs "
            f"and cannot be split three ways; they go to A."
        ),
    })
    print(f"wrote {total:,} rows -> {args.out}")
    print(f"      meta -> {args.meta}")


if __name__ == "__main__":
    main()
