#!/usr/bin/env python3
"""
Catalogue degenerate images across the LAION shards, independently of any model.

WHY THIS IS SEPARATE FROM THE GENERATORS
    Whether an image is degenerate is a property of the FILE, not of the model
    that embeds it - the check reads PIL's size before any transform runs. So
    one list applies to all four models, and keeping four identical copies
    inside the generators would be redundant.

    It also covers shards that were embedded before the generators learned to
    record anomalies, which no amount of re-running the generators would fix
    short of redoing 32M embeddings.

WHAT COUNTS AS DEGENERATE (identical thresholds to the generators)
    any side < 16px, or a long:short ratio > 20:1, or unreadable dimensions.

    Note these are NOT the same as the "channel dimension is ambiguous" warning
    transformers emits: that only fires when a shape is literally ambiguous
    (a side of exactly 1 or 3), so it misses things like a 12x256 sliver.

SCOPE
    By default the shard list is derived from what was actually embedded - the
    union of shard files under result/embeddings_image/*/ - so the catalogue lines up
    exactly with the vectors on disk. --min-worker/--max-worker scans a raw
    range instead.

CPU only: reads JPEG headers, never decodes pixels and never touches a GPU, so
it is safe to run alongside an embedding job.
"""

import io
import os
import json
import time
import tarfile
import argparse
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = REPO_ROOT.parent / "laion/worker1_20/tar1"
DEFAULT_EMBED_ROOT = REPO_ROOT / "result/embeddings_image"
DEFAULT_OUT = REPO_ROOT / "result/anomalous_images.json"

# Must stay in step with the generators.
MIN_DIM = 16
MAX_ASPECT = 20.0

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}


def log(msg: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def classify_anomaly(w: int, h: int) -> Optional[str]:
    if w <= 0 or h <= 0:
        return "zero_dimension"
    if min(w, h) < MIN_DIM:
        return f"tiny_side_{min(w, h)}px"
    if max(w, h) / min(w, h) > MAX_ASPECT:
        return f"extreme_aspect_{max(w, h) / min(w, h):.0f}to1"
    return None


def scan_shard(job: Tuple[str, str, str]) -> Tuple[str, int, List[Dict]]:
    """
    Scan one shard. Returns (tar_key, images_seen, anomalies).

    Headers only - Image.open parses the JPEG header without decoding pixels,
    so this is I/O bound rather than CPU bound. The sibling .json is read only
    for images that turn out to be anomalous, to attach the source URL.
    """
    worker_name, tar_path_s, tar_key = job
    tar_path = Path(tar_path_s)
    stem_of_shard = tar_path.stem
    sizes: Dict[str, Tuple[str, int, int]] = {}
    json_members: Dict[str, tarfile.TarInfo] = {}
    seen = 0

    try:
        with tarfile.open(tar_path, "r") as tf:
            for m in tf:
                if not m.isfile():
                    continue
                p = Path(m.name)
                stem = p.stem
                if p.suffix.lower() in IMAGE_EXTENSIONS:
                    seen += 1
                    try:
                        raw = tf.extractfile(m).read()
                        with Image.open(io.BytesIO(raw)) as im:
                            sizes[stem] = (p.name, im.size[0], im.size[1])
                    except Exception as e:
                        sizes[stem] = (p.name, -1, -1)
                elif p.suffix.lower() == ".json":
                    json_members[stem] = m

            out: List[Dict] = []
            for stem, (fname, w, h) in sizes.items():
                reason = classify_anomaly(w, h) if w > 0 else "unreadable_header"
                if not reason:
                    continue
                row = {"image_name": f"{worker_name}/{stem_of_shard}/{fname}",
                       "width": w, "height": h, "reason": reason}
                mi = json_members.get(stem)
                if mi is not None:
                    try:
                        md = json.loads(tf.extractfile(mi).read().decode("utf-8", "replace"))
                        row["url"] = md.get("url")
                        row["json_width"] = md.get("width")
                        row["json_height"] = md.get("height")
                    except Exception:
                        pass
                out.append(row)
    except Exception as e:
        return tar_key, 0, [{"image_name": tar_key, "width": 0, "height": 0,
                             "reason": f"shard_unreadable: {type(e).__name__}"}]

    return tar_key, seen, out


def shards_from_embeddings(embed_root: Path, data_root: Path) -> List[Tuple[str, str, str]]:
    """Exactly the shards that produced vectors, so the catalogue matches them."""
    keys = set()
    for npy in embed_root.glob("*/worker_*/*.npy"):
        keys.add(f"{npy.parent.name}/{npy.stem}")
    jobs = []
    for key in sorted(keys):
        worker_name, shard = key.split("/")
        tp = data_root / worker_name / f"{shard}.tar"
        if tp.exists():
            jobs.append((worker_name, str(tp), f"{worker_name}/{shard}.tar"))
    return jobs


def shards_from_range(data_root: Path, lo: int, hi: int) -> List[Tuple[str, str, str]]:
    jobs = []
    for wd in sorted(data_root.glob("worker_*"), key=lambda p: int(p.name.split("_")[1])):
        idx = int(wd.name.split("_")[1])
        if not (lo <= idx <= hi):
            continue
        for tp in sorted(wd.glob("*.tar")):
            jobs.append((wd.name, str(tp), f"{wd.name}/{tp.name}"))
    return jobs


def main():
    ap = argparse.ArgumentParser(description="Catalogue degenerate LAION images.")
    ap.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT))
    ap.add_argument("--embed-root", type=str, default=str(DEFAULT_EMBED_ROOT),
                    help="Derive the shard list from vectors already written here.")
    ap.add_argument("--min-worker", type=int, default=None,
                    help="Scan a raw worker range instead of --embed-root.")
    ap.add_argument("--max-worker", type=int, default=None)
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--workers", type=int, default=48)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_path = Path(args.out)
    state_path = out_path.with_suffix(".state.json")

    if args.min_worker is not None:
        jobs = shards_from_range(data_root, args.min_worker,
                                 args.max_worker if args.max_worker is not None else 999)
        scope = f"workers {args.min_worker}..{args.max_worker}"
    else:
        jobs = shards_from_embeddings(Path(args.embed_root), data_root)
        scope = f"shards embedded under {args.embed_root}"

    done, anomalies, seen_total = set(), [], 0
    if state_path.exists():
        try:
            st = json.load(open(state_path))
            done = set(st.get("scanned", []))
            seen_total = int(st.get("images_seen", 0))
            anomalies = json.load(open(out_path)) if out_path.exists() else []
            log(f"Resume: {len(done)} shards scanned, {len(anomalies)} anomalies")
        except Exception:
            done, anomalies, seen_total = set(), [], 0

    jobs = [j for j in jobs if j[2] not in done]
    log(f"Scope: {scope}")
    log(f"Shards to scan: {len(jobs)}  (workers={args.workers})")
    if not jobs:
        log("Nothing to do.")
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    n = 0

    with mp.Pool(processes=args.workers) as pool:
        for tar_key, seen, rows in pool.imap_unordered(scan_shard, jobs, chunksize=4):
            done.add(tar_key)
            seen_total += seen
            n += 1
            if rows:
                anomalies.extend(rows)
                for r in rows:
                    log(f"  {r['image_name']}  {r['width']}x{r['height']}  {r['reason']}")
            if n % 200 == 0 or n == len(jobs):
                json.dump(anomalies, open(out_path, "w"), indent=2)
                json.dump({"scanned": sorted(done), "images_seen": seen_total,
                           "updated_at": datetime.now().isoformat()},
                          open(state_path, "w"))
                el = time.time() - start
                log(f"[{n}/{len(jobs)}] {seen_total:,} images seen, "
                    f"{len(anomalies)} anomalies, {n/el:.1f} shards/s")

    json.dump(anomalies, open(out_path, "w"), indent=2)
    json.dump({"scanned": sorted(done), "images_seen": seen_total,
               "updated_at": datetime.now().isoformat()}, open(state_path, "w"))

    log("=" * 70)
    log(f"Scanned {len(done):,} shards, {seen_total:,} images")
    log(f"Anomalies: {len(anomalies):,}  ({100.0*len(anomalies)/max(1,seen_total):.4f}%)")
    log(f"Written to {out_path}")
    from collections import Counter
    for reason, c in Counter(r["reason"].split("_")[0] for r in anomalies).most_common():
        log(f"   {reason:<16} {c}")
    log("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
