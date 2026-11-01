#!/usr/bin/env python3
"""
Combine the image and caption anomaly catalogues into one exclusion list.

WHY A UNION
    An image-text pair is only usable if BOTH halves are. A 1x1 pixel with a
    good caption is as useless as a good photo with no alt text, so the two
    catalogues have to be merged before anything samples from the corpus.

    Neither modality DROPS its anomalies - both write the vector and keep the
    row, which is what makes image and text row-aligned. Exclusion is therefore
    a decision made at sampling time, not at generation time, and this file is
    what that decision reads.

KEYING
    Both catalogues name records by file:

        worker_0/shard_00130/001306322.jpg      (image)
        worker_0/shard_00130/001306322.json     (caption)

    Stripping the extension gives the PAIR id, which is what the two have in
    common and what a sampler filters on:

        worker_0/shard_00130/001306322

    The worker/shard prefix is load-bearing - workers reuse the same id space,
    so 001000002 exists in several workers as different records.

SCOPE
    Restricted to the comparison corpus: the shard set covered by every model in
    both modalities. The image catalogue is wider than that (clip336 embedded
    748 shards the other three never saw), so those extra shards are dropped
    here - a pair no model can score is not worth listing.

WHAT IS NOT HERE
    Long captions. Every model has a fixed context window, so clipping the tail
    is normal operation for the whole family rather than damage. Flagging it
    would bury the ~400 records that genuinely carry nothing under ~76,000 that
    carry plenty, and excluding those would cost ~0.95% of the corpus and bias
    it toward short captions.

OUTPUT
    result/anomalous_pairs.json    one object per pair, with whichever side(s)
                                   are degenerate and why
    result/excluded_pairs.txt      the same ids, one per line, for a sampler to
                                   read straight into a set()

INPUTS - regenerate these first if the corpus changed:
    python scan_anomalies.py --workers 48
    python scan_caption_anomalies.py --workers 48
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import Dict, Set

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGE_ANOM = REPO_ROOT / "result/anomalous_images.json"
DEFAULT_CAPTION_ANOM = REPO_ROOT / "result/anomalous_captions.json"
DEFAULT_TEXT_ROOT = REPO_ROOT / "result/embeddings_text"
DEFAULT_OUT = REPO_ROOT / "result/anomalous_pairs.json"
DEFAULT_LIST = REPO_ROOT / "result/excluded_pairs.txt"
CANONICAL_MODEL = "flava_full"


def log(msg: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def pair_id(name: str) -> str:
    """worker_0/shard_00130/001306322.jpg -> worker_0/shard_00130/001306322"""
    return name.rsplit(".", 1)[0]


def corpus_shards(text_root: Path, model: str) -> Set[str]:
    return {f"{p.parent.name}/{p.stem}"
            for p in (text_root / model).glob("worker_*/*.npy")}


def load(path: Path, key: str) -> list:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing - run the corresponding scan_*.py first")
    rows = json.load(open(path))
    return [r for r in rows if key in r]


def main():
    ap = argparse.ArgumentParser(description="Union the image and caption anomaly lists.")
    ap.add_argument("--image-anomalies", type=str, default=str(DEFAULT_IMAGE_ANOM))
    ap.add_argument("--caption-anomalies", type=str, default=str(DEFAULT_CAPTION_ANOM))
    ap.add_argument("--text-root", type=str, default=str(DEFAULT_TEXT_ROOT))
    ap.add_argument("--model", type=str, default=CANONICAL_MODEL,
                    help="Whose shard set defines the corpus.")
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--list-out", type=str, default=str(DEFAULT_LIST))
    ap.add_argument("--verify", action="store_true",
                    help="Confirm every pair id really appears in the corpus "
                         "names, so the ids will match at sampling time.")
    args = ap.parse_args()

    text_root = Path(args.text_root)
    shards = corpus_shards(text_root, args.model)
    log(f"corpus: {len(shards):,} shards (from {args.model})")

    img_rows = load(Path(args.image_anomalies), "image_name")
    cap_rows = load(Path(args.caption_anomalies), "caption_name")
    log(f"image anomalies  : {len(img_rows):,}")
    log(f"caption anomalies: {len(cap_rows):,}")

    pairs: Dict[str, Dict] = {}
    dropped_out_of_corpus = 0

    def shard_of(pid: str) -> str:
        w, s, _ = pid.split("/")
        return f"{w}/{s}"

    for r in img_rows:
        pid = pair_id(r["image_name"])
        if shard_of(pid) not in shards:
            dropped_out_of_corpus += 1
            continue
        e = pairs.setdefault(pid, {"pair": pid, "shard": shard_of(pid),
                                   "image": None, "caption": None, "url": None})
        e["image"] = {"reason": r["reason"],
                      "width": r.get("width"), "height": r.get("height")}
        e["url"] = e["url"] or r.get("url")

    for r in cap_rows:
        pid = pair_id(r["caption_name"])
        if shard_of(pid) not in shards:
            dropped_out_of_corpus += 1
            continue
        e = pairs.setdefault(pid, {"pair": pid, "shard": shard_of(pid),
                                   "image": None, "caption": None, "url": None})
        e["caption"] = {"reason": r["reason"], "chars": r.get("chars"),
                        "text": r.get("caption")}
        e["url"] = e["url"] or r.get("url")

    rows = [pairs[k] for k in sorted(pairs)]

    if args.verify:
        # A pair id that no names.json contains would silently never match at
        # sampling time, which is exactly the kind of failure worth catching now.
        by_shard: Dict[str, Set[str]] = {}
        for r in rows:
            by_shard.setdefault(r["shard"], set()).add(r["pair"])
        missing = []
        for sh, want in by_shard.items():
            w, s = sh.split("/")
            names = json.loads((text_root / args.model / w / f"{s}.names.json").read_text())
            have = {pair_id(n) for n in names}
            missing += sorted(want - have)
        if missing:
            log(f"VERIFY FAILED: {len(missing)} ids absent from the corpus, e.g. {missing[:3]}")
            return 1
        log(f"verified: all {len(rows)} ids present in {args.model}'s names lists")

    out_path, list_path = Path(args.out), Path(args.list_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(out_path, "w"), indent=2)
    with open(list_path, "w") as f:
        for r in rows:
            f.write(r["pair"] + "\n")

    both = sum(1 for r in rows if r["image"] and r["caption"])
    img_only = sum(1 for r in rows if r["image"] and not r["caption"])
    cap_only = sum(1 for r in rows if r["caption"] and not r["image"])
    total_rows = sum(len(json.loads((text_root / args.model / sh.split('/')[0] /
                                     f"{sh.split('/')[1]}.names.json").read_text()))
                     for sh in sorted(shards)) if args.verify else None

    log("=" * 70)
    log(f"excluded pairs   : {len(rows):,}")
    log(f"  image only     : {img_only:,}")
    log(f"  caption only   : {cap_only:,}")
    log(f"  both           : {both:,}")
    if dropped_out_of_corpus:
        log(f"outside corpus   : {dropped_out_of_corpus:,} (not listed)")
    if total_rows:
        log(f"corpus pairs     : {total_rows:,}  -> excluding {100.0*len(rows)/total_rows:.4f}%")
    for tag, src in (("image", "image"), ("caption", "caption")):
        c = Counter(r[src]["reason"].split("_")[0] for r in rows if r[src])
        if c:
            log(f"{tag} reasons: {dict(c)}")
    log(f"written: {out_path}")
    log(f"         {list_path}")
    log("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
