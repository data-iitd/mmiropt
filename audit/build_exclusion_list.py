"""
The canonical list of pairs no downstream stage should touch.

WHY A LIST AND NOT A DELETION
-----------------------------
T1 established the discipline and T3 restated it: a damaged record is FLAGGED,
never dropped. Both modalities still write the vector and keep the row, because
dropping a row shifts every row after it and every later artifact then pairs
with the wrong image -- silently, with no error anywhere. Exclusion is applied
at sampling time from one canonical list, and no artifact is ever renumbered.
``result/excluded_pairs.txt`` (the 353 T1/T2 anomalies) is that list for
unreadable images and empty captions. This is the same thing for duplicates,
and it is deliberately a SEPARATE file: the two populations are found by
different means, mean different things, and one of them may later be revised.

THE THREE CATEGORIES, AND WHY THE ORDER MATTERS
-----------------------------------------------
    placeholder      the image is a failed download -- one stock graphic
                     standing in for many unrelated records. The caption is
                     real but orphaned: it describes an image the corpus does
                     not contain. The whole record is corrupt.   -> DELETE ALL

    repeated pair    image and caption both duplicated, as a unit. Nothing is
                     corrupt; the same advert simply appears on 250 pages. But
                     250 identical rows in A make the exact top-100 arbitrary
                     for anything nearby.                        -> KEEP ONE

    boilerplate      the caption is duplicated ("Front Cover", "Patent
    caption          Drawing") but the images are genuinely different books
                     and patents. The record is fine.            -> KEEP ALL

Order matters because the categories overlap. About 1% of the rows in the
13,796-member placeholder group ALSO have a duplicated caption, purely by
chance. A per-row rule would see "both sides duplicated" and route them to
keep-one. They are failed downloads and must be deleted. So placeholder
membership is decided at GROUP level and takes precedence over everything
else; only what survives that is considered for the keep-one rule.

WHAT MAKES A GROUP A PLACEHOLDER GROUP
--------------------------------------
That most of its members carry DIFFERENT captions. One image under many
unrelated captions is the failed-download signature; one image under one
caption is an advert that repeats.

WHICH GROUPS QUALIFY
--------------------
The top 5% of groups by size, per modality: image ``>= 12`` members (p95 of
image group size is 11) and text ``>= 6`` (p95 is 5). This is not a tuning
knob -- it is the point at which a group is large enough to matter, and the
justification is arithmetic. CF@100 compares two 100-slot lists, so a group
with n members in A can corrupt at most n of those 100 slots. Below the
threshold, groups hold 2-4 members in A and so perturb at most 4% of a label.
Above it sit every group that can corrupt a whole label: all 8 image groups
and all 10 text groups with >= 100 members in A clear it.

Measured consequence: the exclusion changes SigLIP 2 i->i gate skill from
0.5693 to 0.4095, against 0.4073 when all 1,177 placeholder groups are
excluded instead of 74. The 22% of duplicated pairs left behind move the
result by 0.002, which is what makes the threshold safe rather than merely
convenient.

WHICH ONE IS KEPT
-----------------
The lowest int_id in the unit. Deterministic, so a re-run cannot silently
change which record survives, and independent of role -- meaning the keeper
may be a B or C row. Roles were assigned before any of this was known, and
re-deriving them here would make the split depend on the duplicate analysis.
The count removed per role is reported instead of hidden.

    ../.venv/bin/python build_exclusion_list.py
    ../.venv/bin/python build_exclusion_list.py --image-min-size 12 --text-min-size 6
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GROUPS = REPO_ROOT / "result/duplicate_groups"
DEFAULT_PILOT = REPO_ROOT / "result/pilot_1M/siglip2"
DEFAULT_OUT = REPO_ROOT / "result/duplicate_exclusions"


# ==========================================================================
# The loader every downstream stage should use
# ==========================================================================
def load_exclusions(path: Path = None) -> set:
    """pair_ids to skip, as a set(). Mirrors load_vectors.load_included_shards:
    read the canonical file, never re-derive the population."""
    path = Path(path or (DEFAULT_OUT / "duplicate_exclusions.txt"))
    if not path.exists():
        return set()
    return {ln.strip() for ln in open(path) if ln.strip()
            and not ln.startswith("#")}


def run(args):
    base = args.groups_dir / args.model / args.scope
    g_img = np.load(base / "image" / "row_group.npy")
    g_txt = np.load(base / "text" / "row_group.npy")
    # pair_id is the key at every scope. int_ids are assigned by T3 per split,
    # so at corpus scope there are none -- and the whole point of running this
    # BEFORE T2 is that the split does not exist yet.
    pids = np.load(base / "image" / "row_pair_id.npy")   # S30 bytes
    pids2 = np.load(base / "text" / "row_pair_id.npy")
    if not np.array_equal(pids, pids2):
        raise RuntimeError("image/text row_pair_id.npy disagree; the join here "
                           "is positional and would be wrong.")
    ids = pids                      # the identifier, whatever the scope

    roles = np.full(len(pids), "-")
    if args.scope == "pilot" and (args.pilot_dir / "pair_index.csv").exists():
        role = {r["pair_id"]: r["role"]
                for r in csv.DictReader(open(args.pilot_dir / "pair_index.csv"))}
        roles = np.array([role.get(p.decode(), "-") for p in pids])

    # ---- 1. which image groups are placeholder groups --------------------
    n_img = int(g_img.max()) + 1
    n_txt = int(g_txt.max()) + 1
    size = np.bincount(g_img[g_img >= 0], minlength=n_img)
    size_t = np.bincount(g_txt[g_txt >= 0], minlength=n_txt)
    uniq_cap = np.bincount(g_img[(g_img >= 0) & (g_txt < 0)], minlength=n_img)
    frac_uniq = np.divide(uniq_cap, np.maximum(size, 1))
    is_ph = (frac_uniq >= args.placeholder_threshold) & (size >= args.image_min_size)
    ph_groups = set(np.nonzero(is_ph)[0].tolist())
    print(f"  image: {len(ph_groups):,} of {n_img:,} groups are placeholder groups "
          f"(>= {args.placeholder_threshold:.0%} unique captions, "
          f"size >= {args.image_min_size})")
    print(f"  text:  {int((size_t >= args.text_min_size).sum()):,} of {n_txt:,} "
          f"groups clear size >= {args.text_min_size}")

    # ---- 2. assign every row, placeholder first --------------------------
    is_placeholder = np.isin(g_img, list(ph_groups))

    # Repeated-pair units among what survives. A unit is ONE object that
    # appears in both groupings, so it qualifies when either side clears its
    # own threshold -- requiring both would silently drop units that are large
    # on one side and small on the other.
    survivor = ~is_placeholder
    qualifies = ((g_img >= 0) & (size[np.maximum(g_img, 0)] >= args.image_min_size)) | \
                ((g_txt >= 0) & (size_t[np.maximum(g_txt, 0)] >= args.text_min_size))
    both = survivor & (g_img >= 0) & (g_txt >= 0) & qualifies
    key = np.where(both, g_img.astype(np.int64) * 10_000_000 + g_txt, -1)

    drop_rep = np.zeros(len(ids), bool)
    kept_rep = np.zeros(len(ids), bool)
    idx = np.nonzero(both)[0]
    if len(idx):
        order = idx[np.lexsort((ids[idx], key[idx]))]   # by unit, then pair_id
        k = key[order]
        first = np.r_[True, k[1:] != k[:-1]]
        sizes = np.diff(np.r_[np.nonzero(first)[0], len(k)])
        multi = np.repeat(sizes > 1, sizes)
        drop_rep[order[multi & ~first]] = True          # all but the lowest id
        kept_rep[order[multi & first]] = True

    excluded = is_placeholder | drop_rep
    reason = np.where(is_placeholder, "placeholder",
                      np.where(drop_rep, "repeated_pair", ""))

    # ---- 3. write -------------------------------------------------------
    dest = args.out_dir
    dest.mkdir(parents=True, exist_ok=True)
    sel = np.nonzero(excluded)[0]
    sel = sel[np.argsort(ids[sel])]

    # No comment header: T2's stratify_kmeans.load_excluded_pairs takes EVERY
    # non-blank line as a pair_id, so a '#' line would enter the set.
    dup = sorted(p.decode() for p in ids[sel])
    with open(dest / "duplicate_exclusions.txt", "w") as f:
        f.write("\n".join(dup) + "\n")

    # The file T2 actually consumes: duplicates + the T1/T2 anomalies that fall
    # inside the corpus cut. Asserted disjoint rather than assumed -- they are
    # today only because load_corpus dropped the anomalies before grouping.
    anom = {l.strip() for l in open(args.anomalies) if l.strip()} \
        if Path(args.anomalies).exists() else set()
    overlap = anom & set(dup)
    if overlap:
        raise RuntimeError(f"{len(overlap)} pair_ids are in BOTH the anomaly "
                           f"list and the duplicate list, e.g. {sorted(overlap)[:3]}")
    combined = sorted(anom | set(dup))
    with open(dest / "exclusions_combined.txt", "w") as f:
        f.write("\n".join(combined) + "\n")

    with open(dest / "duplicate_exclusions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "role", "reason",
                    "image_group", "text_group", "image_group_size"])
        for j in sel:
            w.writerow([ids[j].decode(), roles[j], reason[j],
                        int(g_img[j]), int(g_txt[j]),
                        int(size[g_img[j]]) if g_img[j] >= 0 else 0])

    by_reason = {r: int((reason == r).sum())
                 for r in ("placeholder", "repeated_pair")}
    counts = None
    if args.scope == "pilot":
        counts = {r: {x: int((excluded & (roles == x) & (reason == r)).sum())
                      for x in "ABC"}
                  for r in by_reason}
    meta = {
        "created": datetime.now().isoformat(),
        "task": "audit -- canonical duplicate exclusion list",
        "scope": args.scope,
        "model_groups_derived_from": args.model,
        "placeholder_threshold": args.placeholder_threshold,
        "image_min_size": args.image_min_size,
        "text_min_size": args.text_min_size,
        "threshold_basis": "top 5% of groups by size, per modality (p95 of "
                           "group size at this scope)",
        "n_rows_considered": int(len(ids)),
        "n_excluded": int(excluded.sum()),
        "by_reason": by_reason,
        "by_reason_by_role": counts,
        "n_placeholder_groups": len(ph_groups),
        "n_text_groups_qualifying": int((size_t >= args.text_min_size).sum()),
        "n_repeated_pair_units_kept": int(kept_rep.sum()),
        "keep_rule": "lowest pair_id in the unit",
        "precedence": "placeholder (group level) beats repeated_pair (row level)",
        "not_renumbered": (
            "This is a mask. No artifact is rebuilt, no id is reassigned."),
        "anomaly_list": str(args.anomalies),
        "n_anomalies_unioned": len(anom),
        "n_combined": len(combined),
        "combined_file": "exclusions_combined.txt -- what T2 consumes",
    }
    print(f"\n  excluded {int(excluded.sum()):,} of {len(ids):,} pairs "
          f"({excluded.sum()/len(ids):.2%})")
    for r, c in by_reason.items():
        line = f"    {r:<15} {c:>8,}"
        if counts:
            line += (f"   A {counts[r]['A']:>7,}  B {counts[r]['B']:>6,}"
                     f"  C {counts[r]['C']:>6,}")
        print(line)
    print(f"    {'+ T1/T2 anomalies':<15} {len(anom):>8,}")
    print(f"    {'= combined':<15} {len(combined):>8,}")
    json.dump(meta, open(dest / "meta.json", "w"), indent=2)
    print(f"  wrote {dest}/exclusions_combined.txt")
    return meta

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="siglip2")
    ap.add_argument("--scope", default="corpus", choices=["pilot", "corpus"])
    ap.add_argument("--anomalies", type=Path,
                    default=REPO_ROOT / "result/excluded_pairs.txt")
    ap.add_argument("--placeholder-threshold", type=float, default=0.5,
                    help="share of an image group's members needing a UNIQUE "
                         "caption for the group to count as failed downloads")
    ap.add_argument("--image-min-size", type=int, default=8,
                    help="minimum image-group size to qualify; 12 is the top "
                         "5%% of image groups (p95 size = 11)")
    ap.add_argument("--text-min-size", type=int, default=6,
                    help="minimum text-group size to qualify; 6 is the top "
                         "5%% of text groups (p95 size = 5)")
    ap.add_argument("--groups-dir", type=Path, default=DEFAULT_GROUPS)
    ap.add_argument("--pilot-dir", type=Path, default=DEFAULT_PILOT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    print("build_exclusion_list")
    run(args)


if __name__ == "__main__":
    main()
