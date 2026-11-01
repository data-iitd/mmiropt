"""
Put content behind the duplicate groups, so they can be adjudicated.

``duplicate_groups.py`` finds groups GEOMETRICALLY -- rows whose embeddings are
the same vector to float32 precision. That says two rows are indistinguishable
to the model. It does NOT say why, and the why is what decides whether a group
should be excluded:

    a scrape failure      one placeholder image, many unrelated captions
                          -> the exact top-100 is undefined, exclude
    a repeated record     the same (image, caption) pair posted N times
                          -> also undefined, but a real corpus property
    caption degeneracy    one generic caption ("Cover", "No Image"), many
                          genuinely different images
                          -> breaks the text-side labels only
    a tight real cluster  distinct items the encoder cannot separate
                          -> the ground truth is fine, the INDEX is being
                             tested, and excluding it would be cheating

Only the last must be kept, and no geometric test can find it. So this script
pulls each group's images and captions out of the LAION tars and lays them
side by side for a human to flag.

WHY BOTH MODALITIES ARE SHOWN FOR EVERY GROUP
---------------------------------------------
Every pilot row is a PAIR. A group found on the image side is a set of pairs
whose IMAGES coincide -- and the interesting question is immediately what their
CAPTIONS look like. If the captions are all different, one image is standing in
for many unrelated records and the group is a scrape failure. If the captions
coincide too, the whole record repeats. The contact sheet therefore always
shows the image and its caption together, whichever modality defined the group:

    image group  ->  identical pictures across the row, captions vary
    text group   ->  identical captions across the row, pictures vary

That contrast is the finding, and it is visible at a glance rather than
argued from a statistic.

THE STRUCTURAL PRE-CLASSIFICATION
---------------------------------
Before any tar is opened, each group is cross-tabulated against the OTHER
modality's grouping, which is free -- both are already on disk. A group whose
members almost all lack a partner-side duplicate is a one-sided duplication;
a group whose members all share one partner group is a repeated record. This
is emitted as ``structural_class`` so the flagging pass can concentrate on
what the structure cannot settle, rather than eyeballing 2,138 rows.

COST
----
One tar per group -- the single shard holding the most of its members -- at
~17 MB each. A 40-group catalogue streams ~700 MB. No GPU, no embeddings, no
corpus matrix.

    cd mmir_optimizer/audit
    ../.venv/bin/python duplicate_catalogue.py --modality image --top 40
    ../.venv/bin/python duplicate_catalogue.py --modality text  --top 40
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import tarfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LAION = REPO_ROOT.parent / "laion/worker1_20/tar1"
DEFAULT_GROUPS = REPO_ROOT / "result/duplicate_groups"
DEFAULT_PILOT = REPO_ROOT / "result/pilot_1M/siglip2"

OTHER = {"image": "text", "text": "image"}


# ==========================================================================
# Inputs
# ==========================================================================
def load_side(groups_dir: Path, model: str, scope: str, modality: str):
    """(group id per row, pair_id per row). pair_id is the key at every scope --
    int_ids are assigned by T3 per split and do not exist at corpus scope."""
    d = groups_dir / model / scope / modality
    return (np.load(d / "row_group.npy"),
            np.load(d / "row_pair_id.npy").astype(str))


def load_roles(pair_index_csv: Path) -> dict:
    """pair_id -> role. Pilot scope only; returns {} when the split does not
    exist, and the A/B/C columns are then omitted rather than written as 0."""
    if not Path(pair_index_csv).exists():
        return {}
    return {r["pair_id"]: r["role"]
            for r in csv.DictReader(open(pair_index_csv))}


def classify(members_partner: np.ndarray, modality: str) -> tuple:
    """(structural_class, evidence) from the partner modality's group ids.

    ``members_partner`` is the OTHER side's group id for each member, -1 where
    that member has no duplicate on the other side.
    """
    n = len(members_partner)
    n_none = int((members_partner < 0).sum())
    present = members_partner[members_partner >= 0]
    top_share = 0.0
    if len(present):
        top_share = Counter(present.tolist()).most_common(1)[0][1] / n
    other = OTHER[modality]
    if n_none / n >= 0.8:
        return (f"one_{modality}_many_{other}",
                f"{n_none}/{n} members have no {other}-side duplicate")
    if top_share >= 0.8:
        return ("repeated_record",
                f"{top_share:.0%} of members share one {other} group")
    return ("mixed", f"{n_none}/{n} unpaired, largest {other} group {top_share:.0%}")


# ==========================================================================
# Content
# ==========================================================================
def _read_shard(task):
    """Worker: pull the wanted members out of one tar. Must be top-level and
    take plain data, because ProcessPoolExecutor pickles it."""
    laion, w, s, names, n_img, n_cap = task
    tp = Path(laion) / w / f"{s}.tar"
    imgs, caps = [], []
    if not tp.exists():
        return w, s, imgs, caps, False
    want = set(names)
    with tarfile.open(tp, "r|") as tf:
        for m in tf:
            st = Path(m.name).stem
            if st not in want:
                continue
            if m.name.endswith(".jpg") and len(imgs) < n_img:
                imgs.append((st, tf.extractfile(m).read()))
            elif m.name.endswith(".json") and len(caps) < n_cap:
                try:
                    caps.append((st, json.loads(
                        tf.extractfile(m).read()).get("caption", "")))
                except Exception:
                    pass
            if len(imgs) >= n_img and len(caps) >= n_cap:
                break                      # tar is sequential; stop scanning
    return w, s, imgs, caps, True


def plan_group(member_pids, n_shards: int):
    """The (worker, shard) -> member-names reads one group needs. No I/O.

    Not one tar per member: a scrape-failure group's members are scattered over
    the whole corpus and opening a 17 MB archive for each would dominate the
    runtime for no extra information. But not one tar either -- drawing every
    sample from a single shard means every sample is a near neighbour in crawl
    order, which is exactly the population most likely to share a caption by
    accident. Spreading over the richest ``n_shards`` keeps the read bounded
    while making the caption variety an honest sample rather than an artifact
    of where we happened to look.
    """
    by = defaultdict(list)
    for pid in member_pids:
        w, s, n = pid.split("/")
        by[(w, s)].append(n)
    return sorted(by.items(), key=lambda kv: -len(kv[1]))[:n_shards]


# ==========================================================================
# Run
# ==========================================================================
def run(args):
    model, mod = args.model, args.modality
    grp, pair_of = load_side(args.groups_dir, model, args.scope, mod)
    pgrp, pids = load_side(args.groups_dir, model, args.scope, OTHER[mod])
    if not np.array_equal(pair_of, pids):
        raise RuntimeError(
            "image and text row_pair_id.npy disagree -- the cross-modal "
            "classification below joins them POSITIONALLY and would be wrong.")
    roles = load_roles(args.pilot_dir / "pair_index.csv") if args.scope == "pilot" else {}

    n_groups = int(grp.max()) + 1
    top = min(args.top, n_groups)
    print(f"  {mod}: {n_groups:,} groups; cataloguing the largest {top}")

    # Plan every tar read first, then execute them all in one pool. The reads
    # are latency-bound on a shared filesystem (~0.5 MB/s per stream), not
    # bandwidth-bound, so concurrency is worth roughly its width; doing them
    # group-by-group took ~90 s each and would have run for two hours.
    # Tasks are keyed by their own INDEX, never by (worker, shard). Two groups
    # routinely want different members of the same shard -- group 11 and group
    # 19 both draw from worker_0/shard_00354 -- and keying the results by shard
    # silently served one group's images to the other. The samples still looked
    # plausible, which is exactly what made it dangerous.
    plans, tasks = {}, []
    for g in range(top):
        picks = plan_group(pair_of[grp == g], args.shards_per_group)
        plans[g] = []
        for (w, s), names in picks:
            plans[g].append(len(tasks))
            tasks.append((str(args.laion), w, s, names,
                          args.samples, args.samples * 4))
    print(f"  {len(tasks)} shard reads over {args.workers} workers ...")

    from concurrent.futures import ProcessPoolExecutor
    fetched = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for t, (w, s, imgs, caps, ok) in enumerate(ex.map(_read_shard, tasks)):
            fetched[t] = (w, s, imgs, caps, ok)

    dest = args.groups_dir / model / args.scope / mod
    rows, panels = [], []
    for g in range(top):
        m = grp == g
        member_pids = pair_of[m]
        cls, why = classify(pgrp[m], mod)
        imgs, caps, seen = [], [], []
        for t in plans[g]:
            w, s, gi, gc, ok = fetched[t]
            imgs.extend(gi); caps.extend(gc)
            seen.append(f"{w}/{s}" + ("" if ok else " (missing)"))
        # Every sample must actually belong to this group. Cheap, and the only
        # thing that distinguishes "these images differ because the group is
        # heterogeneous" from "these images differ because they are not all
        # from this group" -- a contact sheet cannot tell you which.
        member_stems = {pid.rsplit("/", 1)[1] for pid in member_pids}
        stray = [st for st, _ in imgs if st not in member_stems]
        if stray:
            raise RuntimeError(
                f"group {g}: {len(stray)} sampled image(s) are not members "
                f"({stray[:3]}). The sheet would show one group's pictures "
                f"under another's label.")
        caps = [(st, c) for st, c in caps if st in member_stems]

        imgs, shard = imgs[:args.samples], " + ".join(seen)
        texts = [c for _, c in caps]
        distinct = len(set(t.strip().lower() for t in texts if t.strip()))
        row = {"group_id": g, "n_total": int(m.sum())}
        if roles:                      # pilot scope only; T3 assigns roles
            r = [roles.get(pid, "?") for pid in member_pids]
            row.update({"n_A": r.count("A"), "n_B": r.count("B"),
                        "n_C": r.count("C")})
        row.update({
            "structural_class": cls,
            "structural_evidence": why,
            "sampled_shard": shard or "",
            "n_captions_sampled": len(texts),
            "n_distinct_captions": distinct,
            "rep_pair_id": str(member_pids[0]),
            "sample_captions": " | ".join(t[:60] for t in texts[:5]),
            "FLAG": "",          # <- you fill this in
        })
        rows.append(row)
        panels.append((g, int(m.sum()), cls, imgs, dict(caps)))
        print(f"    group {g:>4}  n={int(m.sum()):>6}  {cls:<28} "
              f"{distinct} distinct captions of {len(texts)} sampled")

    out_csv = dest / f"catalogue_top{top}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  wrote {out_csv}")

    if not args.no_figure:
        # Paginated: one tall strip of 40 groups is not something anyone can
        # actually read. Pages of --per-page keep each sheet to a screenful.
        per = max(1, args.per_page)
        for p0 in range(0, len(panels), per):
            page = panels[p0:p0 + per]
            n_page = p0 // per + 1
            _contact_sheet(page, dest / f"catalogue_top{top}_p{n_page}.png",
                           mod, args.samples,
                           subtitle=f"groups {page[0][0]}-{page[-1][0]} "
                                    f"(page {n_page} of "
                                    f"{(len(panels) + per - 1) // per})")

    json.dump({
        "created": datetime.now().isoformat(),
        "model": model, "modality": mod, "top": top,
        "samples_per_group": args.samples,
        "flagging": ("Fill the FLAG column with one of: scrape_failure, "
                     "repeated_record, caption_degeneracy, real_cluster. "
                     "Only real_cluster is kept in the labels."),
        "structural_class_is_a_hint": (
            "Computed from the two groupings alone, before any image was "
            "looked at. It narrows the eyeballing; it does not replace it."),
    }, open(dest / f"catalogue_top{top}_meta.json", "w"), indent=2)
    return rows


def _contact_sheet(panels, path: Path, modality: str, n_samples: int,
                   subtitle: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    panels = [p for p in panels if p[3]]
    if not panels:
        print("  no images recovered; skipping contact sheet")
        return
    nrow = len(panels)
    fig, axes = plt.subplots(nrow, n_samples,
                             figsize=(n_samples * 2.1 + 3.0, nrow * 2.35),
                             squeeze=False)
    fig.suptitle(
        f"SigLIP 2 {modality}-side duplicate groups, pilot (A+B+C)   {subtitle}\n"
        f"each row is one group; the {modality} is what coincides, "
        f"so watch the {OTHER[modality]} vary (or not)",
        fontsize=12, y=0.995)

    for row, (g, n, cls, imgs, capmap) in enumerate(panels):
        for k in range(n_samples):
            ax = axes[row][k]
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor("0.8")
            if k < len(imgs):
                st, raw = imgs[k]
                try:
                    ax.imshow(Image.open(io.BytesIO(raw)).convert("RGB"))
                except Exception:
                    ax.axis("off")
                cap = capmap.get(st, "")
                ax.set_title(cap[:38] if cap else "(no caption)",
                             fontsize=5.8, pad=2)
            else:
                ax.axis("off")
        axes[row][0].set_ylabel(f"grp {g}\nn={n:,}\n{cls}", fontsize=7,
                                rotation=0, ha="right", va="center", labelpad=52)
    fig.tight_layout(rect=[0.03, 0.01, 1, 0.965])
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  wrote {path}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="siglip2")
    ap.add_argument("--modality", default="image", choices=["image", "text"])
    ap.add_argument("--scope", default="corpus", choices=["pilot", "corpus"])
    ap.add_argument("--top", type=int, default=40,
                    help="largest N groups to catalogue; ~17 MB of tar read each")
    ap.add_argument("--samples", type=int, default=6)
    ap.add_argument("--shards-per-group", type=int, default=3,
                    help="distinct shards to draw a group's samples from, so "
                         "caption variety is not an artifact of crawl order")
    ap.add_argument("--per-page", type=int, default=10,
                    help="groups per contact sheet; a 40-row strip is unreadable")
    ap.add_argument("--workers", type=int, default=16,
                    help="concurrent tar reads; the shard filesystem is "
                         "latency-bound, so this buys close to linear speedup")
    ap.add_argument("--laion", type=Path, default=DEFAULT_LAION)
    ap.add_argument("--groups-dir", type=Path, default=DEFAULT_GROUPS)
    ap.add_argument("--pilot-dir", type=Path, default=DEFAULT_PILOT)
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()
    print(f"duplicate_catalogue  model={args.model}  modality={args.modality}")
    run(args)


if __name__ == "__main__":
    main()
