#!/usr/bin/env python3
"""
Catalogue degenerate captions across the LAION shards, independently of any model.

WHY THIS IS SEPARATE FROM THE GENERATORS
    Whether a caption is degenerate is a property of the FILE, not of the model
    that embeds it - the check reads the "caption" string before any tokeniser
    runs. So one list applies to all four models, and keeping four identical
    copies inside the generators would be redundant.

    This is the text twin of scan_anomalies.py, and it exists for the same
    reason: one canonical catalogue rather than four copies that could drift.

WHAT COUNTS AS DEGENERATE (identical thresholds to the generators)
    missing         the record has no "caption" key, or "caption": null
    unparseable     the .json will not parse (img2dataset writes zero-byte ones)
    empty           present but whitespace only
    tiny            fewer than 3 non-space characters

    A LONG caption is deliberately not listed. Every model here has a fixed
    context window, so clipping the tail is normal operation for the whole
    family rather than damage.

    A null caption is NOT a broken record. The image side embedded that pair, so
    the generators embed the empty string and flag it here; dropping the row
    would shift every later row of the shard out of alignment.

WHAT IS NOT DEGENERATE
    A caption that is merely odd - a filename, a price, a stock-photo boilerplate
    string - still tokenises to something the model can use, so it is not
    flagged. This catalogue is about captions the embedding cannot represent,
    not captions that are low quality.

SCOPE
    By default the shard list is derived from what was actually embedded - the
    union of shard files under result/embeddings_text/*/ - so the catalogue
    lines up exactly with the vectors on disk. If the text runs have not
    happened yet, point --embed-root at result/embeddings_image instead, or use
    --min-worker/--max-worker to scan a raw range.

CPU only: reads the .json members, never loads a model and never touches a GPU,
so it is safe to run alongside an embedding job. It also skips the .jpg members
entirely - tarfile seeks past their payload rather than reading it - so this is
far lighter than the image scan.
"""

import json
import time
import tarfile
import argparse
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import List, Dict, Tuple, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = REPO_ROOT.parent / "laion/worker1_20/tar1"
DEFAULT_EMBED_ROOT = REPO_ROOT / "result/embeddings_text"
DEFAULT_OUT = REPO_ROOT / "result/anomalous_captions.json"

# Must stay in step with the generators.
MIN_CHARS = 3

CAPTION_KEY = "caption"


def log(msg: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


# Byte-identical to the generators' classify_anomaly, so the two never disagree
# about what counts as degenerate.
def classify_anomaly(text) -> Optional[str]:
    if text is None:
        return "missing_caption"
    if not isinstance(text, str):
        return f"non_string_caption_{type(text).__name__}"
    s = text.strip()
    if not s:
        return "empty_caption"
    if len(s) < MIN_CHARS:
        return f"tiny_caption_{len(s)}chars"
    return None


def scan_shard(job: Tuple[str, str, str]) -> Tuple[str, int, List[Dict]]:
    """
    Scan one shard. Returns (tar_key, captions_seen, anomalies).

    Only .json members are read; tarfile seeks past the image payloads, so the
    bytes actually pulled off disk are a small fraction of the shard.
    """
    worker_name, tar_path_s, tar_key = job
    tar_path = Path(tar_path_s)
    stem_of_shard = tar_path.stem
    seen = 0
    out: List[Dict] = []

    try:
        with tarfile.open(tar_path, "r") as tf:
            for m in tf:
                if not m.isfile() or Path(m.name).suffix.lower() != ".json":
                    continue
                seen += 1
                fname = Path(m.name).name
                name = f"{worker_name}/{stem_of_shard}/{fname}"
                try:
                    raw = tf.extractfile(m).read()
                    md = json.loads(raw.decode("utf-8", "replace"))
                except Exception as e:
                    # img2dataset occasionally writes a zero-byte .json even
                    # though the image landed. The generators embed the empty
                    # string for these rather than dropping the row, so the
                    # reason string here is the one they record.
                    out.append({"caption_name": name, "chars": -1,
                                "reason": "unparseable_json",
                                "detail": f"{type(e).__name__}, {len(raw)} bytes",
                                "caption": None})
                    continue

                text = md.get(CAPTION_KEY) if isinstance(md, dict) else None
                reason = classify_anomaly(text)
                if not reason:
                    continue
                out.append({
                    "caption_name": name,
                    "chars": len(text.strip()) if isinstance(text, str) else 0,
                    "reason": reason,
                    "caption": (text[:200] if isinstance(text, str) else None),
                    "url": md.get("url"),
                })
    except Exception as e:
        return tar_key, 0, [{"caption_name": tar_key, "chars": -1,
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
    ap = argparse.ArgumentParser(description="Catalogue degenerate LAION captions.")
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
            seen_total = int(st.get("captions_seen", 0))
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
                    log(f"  {r['caption_name']}  {r['chars']} chars  {r['reason']}")
            if n % 200 == 0 or n == len(jobs):
                json.dump(anomalies, open(out_path, "w"), indent=2)
                json.dump({"scanned": sorted(done), "captions_seen": seen_total,
                           "updated_at": datetime.now().isoformat()},
                          open(state_path, "w"))
                el = time.time() - start
                log(f"[{n}/{len(jobs)}] {seen_total:,} captions seen, "
                    f"{len(anomalies)} anomalies, {n/el:.1f} shards/s")

    json.dump(anomalies, open(out_path, "w"), indent=2)
    json.dump({"scanned": sorted(done), "captions_seen": seen_total,
               "updated_at": datetime.now().isoformat()}, open(state_path, "w"))

    log("=" * 70)
    log(f"Scanned {len(done):,} shards, {seen_total:,} captions")
    log(f"Anomalies: {len(anomalies):,}  ({100.0*len(anomalies)/max(1,seen_total):.4f}%)")
    log(f"Written to {out_path}")
    for reason, c in Counter(r["reason"].split("_")[0] for r in anomalies).most_common():
        log(f"   {reason:<16} {c}")
    log("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
