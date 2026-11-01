#!/usr/bin/env python3
"""
Merge an 8-way shard-strided embedding run (<model>_part0 .. _part7) into one
canonical model directory, in the same worker_*/shard_*.npy + .names.json
layout the single-process generators produce.

WHY THIS EXISTS
    siglip2, qwen3vl, and jinav4 are each being embedded by 8 concurrent
    processes, one GPU each, partitioned with --shard-stride 8 --shard-index i
    (i = 0..7) over the SAME worker_0..worker_20 range - the approach the
    embeddingsgenerator README recommends after the clip336 split (which
    partitioned by worker range instead, and silently left a 546-shard hole).
    Striding guarantees the eight parts' shard sets are disjoint and their
    union is the whole range, so merging them is: combine the directory
    trees, and treat any overlap as a bug, not something to resolve quietly.

WHAT THIS DOES NOT DO
    It never renumbers or reassigns rows. Every shard file keeps its own
    worker_N/shard_NNNNN identity; merging only moves it to sit under one
    parent directory instead of eight. Row order inside a shard, and which
    shard a record belongs to, is unchanged - this is exactly the row-
    alignment invariant every downstream script (starting with the A/B/C
    split) depends on.

SAFETY
    - Collision check: a shard key ("worker_N/shard_NNNNN") present in more
      than one part aborts before any write, since stride partitioning should
      make that impossible - if it happens, something upstream is wrong and
      merging anyway would silently pick one copy over the other.
    - Non-empty destination: refuses to merge into a destination directory
      that already holds any worker_*/*.npy, unless --force is given. This
      guards against exactly the situation found on this host: a stale,
      abandoned single-process siglip2 run (716,105/8,000,000 images, stopped
      2026-09-04) sitting at the same canonical path
      (siglip2_so400m_patch16_512) that the part-based run's merge would also
      target.
    - Default link mode is a hardlink, not a copy: zero extra disk, and the
      part directories are left intact (nothing is deleted) unless
      --then-remove-parts is passed explicitly, after a successful, verified
      merge.

LEDGER
    Does not shell out to finalize.py --rebuild: that command is documented as
    image-only as written (it writes total_images, which the text generators'
    resume logic reads as total_texts), and this script needs to be correct
    for both modalities. Instead it rebuilds progress.json / run_summary.json
    itself, reading only .npy headers (never the arrays) and using the
    modality-correct key.

USAGE
    # dry run - report collisions / stale-destination issues, write nothing
    python merge_sharded_parts.py --model siglip2 --modality image --verify-only

    # real merge, once all 8 parts are done (or to merge whatever exists so far)
    python merge_sharded_parts.py --model siglip2 --modality image

    # after spot-checking the merged dir, reclaim the part directories' disk
    python merge_sharded_parts.py --model siglip2 --modality image --then-remove-parts
"""

import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

# Canonical output-dir name per model - must match each gen_<model>.py's
# MODEL_KEY exactly, since that is what every later script will look for.
MODEL_KEYS = {
    "siglip2": "siglip2_so400m_patch16_512",
    "qwen3vl": "qwen3_vl_embedding_2b",
    "jinav4": "jina_embeddings_v4",
    "openclip": "openclip_vit_l14_laion2b",
}

ANOMALY_FILE = {"image": "anomalous_images.json", "text": "anomalous_captions.json"}
ANOMALY_NAME_KEY = {"image": "image_name", "text": "caption_name"}
LEDGER_TOTAL_KEY = {"image": "total_images", "text": "total_texts"}


def log(msg: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def shard_key(npy_path: Path) -> str:
    """.../worker_3/shard_00512.npy -> worker_3/shard_00512"""
    return f"{npy_path.parent.name}/{npy_path.stem}"


def collect_part(part_dir: Path) -> Dict[str, Tuple[Path, Path]]:
    """shard_key -> (npy_path, names_json_path) for one part directory."""
    out = {}
    for npy in sorted(part_dir.glob("worker_*/shard_*.npy")):
        names = npy.with_suffix(".names.json")
        out[shard_key(npy)] = (npy, names)
    return out


def npy_header_rows(path: Path) -> int:
    try:
        return int(np.load(path, mmap_mode="r").shape[0])
    except Exception:
        return -1


def find_parts(embed_root: Path, model: str, explicit: List[str]) -> List[Path]:
    if explicit:
        return [Path(p) for p in explicit]
    return sorted(embed_root.glob(f"{model}_part*"),
                  key=lambda p: p.name)


def load_anomalies(part_dirs: List[Path], fname: str) -> list:
    seen = set()
    merged = []
    for pd in part_dirs:
        p = pd / fname
        if not p.exists():
            continue
        for row in json.loads(p.read_text()):
            key = row.get("image_name") or row.get("caption_name")
            if key in seen:
                continue
            seen.add(key)
            merged.append(row)
    return merged


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(MODEL_KEYS),
                     help="Which sharded model run to merge.")
    ap.add_argument("--modality", required=True, choices=["image", "text"])
    ap.add_argument("--embed-root", type=str, default=None,
                     help="Defaults to result/embeddings_{modality}.")
    ap.add_argument("--parts", nargs="*", default=None,
                     help="Explicit part directory paths. Default: "
                          "<embed-root>/<model>_part* sorted by name.")
    ap.add_argument("--dest", type=str, default=None,
                     help="Merge destination. Default: <embed-root>/<canonical model key>.")
    ap.add_argument("--link-mode", choices=["hardlink", "copy"], default="hardlink")
    ap.add_argument("--force", action="store_true",
                     help="Allow merging into a destination that already "
                          "contains worker_*/*.npy files. Off by default: "
                          "a non-empty destination is far more likely to be "
                          "stale leftover content than something safe to mix in.")
    ap.add_argument("--then-remove-parts", action="store_true",
                     help="After a verified, successful merge, delete the "
                          "source part directories. Off by default.")
    ap.add_argument("--verify-only", action="store_true",
                     help="Report collisions / row-count problems / a "
                          "non-empty destination, write nothing.")
    args = ap.parse_args()

    modality = args.modality
    embed_root = Path(args.embed_root) if args.embed_root else \
        REPO_ROOT / f"result/embeddings_{modality}"
    dest = Path(args.dest) if args.dest else embed_root / MODEL_KEYS[args.model]

    part_dirs = find_parts(embed_root, args.model, args.parts)
    if not part_dirs:
        log(f"no part directories found matching {args.model}_part* under {embed_root}")
        return 1
    log(f"parts found: {[p.name for p in part_dirs]}")

    # --- collect + collision check -----------------------------------------
    owner: Dict[str, str] = {}
    merged: Dict[str, Tuple[Path, Path]] = {}
    collisions: Dict[str, List[str]] = defaultdict(list)
    for pd in part_dirs:
        part_map = collect_part(pd)
        log(f"  {pd.name:<20} {len(part_map):>6,} shards")
        for key, paths in part_map.items():
            if key in owner:
                collisions[key] = [owner[key], pd.name]
            else:
                owner[key] = pd.name
                merged[key] = paths

    if collisions:
        log(f"ABORT: {len(collisions)} shard(s) claimed by more than one part "
            f"- stride partitioning should make this impossible.")
        for k, parts in list(collisions.items())[:10]:
            log(f"    {k}  ->  {parts}")
        log("Not writing anything. Investigate the --shard-stride/--shard-index "
            "values each part was actually launched with before re-running.")
        return 1

    total_rows_claimed = len(merged)
    log(f"total shards across all parts (no collisions): {total_rows_claimed:,}")

    # --- row-count sanity: every .npy needs a matching .names.json ----------
    bad = []
    for key, (npy, names) in merged.items():
        if not names.exists():
            bad.append((key, "missing .names.json"))
            continue
        n_rows = npy_header_rows(npy)
        n_names = len(json.loads(names.read_text()))
        if n_rows < 0:
            bad.append((key, "unreadable .npy"))
        elif n_rows != n_names:
            bad.append((key, f".npy has {n_rows} rows, .names.json has {n_names}"))
    if bad:
        log(f"ABORT: {len(bad)} shard(s) failed the .npy/.names.json row-count check.")
        for k, why in bad[:10]:
            log(f"    {k}: {why}")
        return 1
    log("row-count check: every shard's .npy matches its .names.json")

    # --- destination safety check -------------------------------------------
    dest_has_content = dest.exists() and any(dest.glob("worker_*/shard_*.npy"))
    if dest_has_content and not args.force:
        existing = sorted(dest.glob("worker_*/shard_*.npy"))
        log(f"ABORT: destination {dest} already contains {len(existing)} shard "
            f"file(s) - refusing to merge into it without --force.")
        log("This is exactly the situation found for siglip2_so400m_patch16_512: "
            "a stale, abandoned single-process run sitting at the canonical path. "
            "Move or remove that content deliberately first (or pass --dest to "
            "target a scratch directory instead), then re-run.")
        return 1

    if args.verify_only:
        log("verify-only: no collisions, no row-count problems"
            + (", destination already has content (would need --force)" if dest_has_content else ""))
        log(f"would write {total_rows_claimed:,} shards to {dest}")
        return 0

    # --- link (or copy) into the destination --------------------------------
    dest.mkdir(parents=True, exist_ok=True)
    linked, copied = 0, 0
    for key, (npy, names) in merged.items():
        worker, stem = key.split("/")
        out_dir = dest / worker
        out_dir.mkdir(exist_ok=True)
        for src, dst_name in ((npy, f"{stem}.npy"), (names, f"{stem}.names.json")):
            dst = out_dir / dst_name
            if dst.exists():
                continue
            if args.link_mode == "hardlink":
                try:
                    dst.hardlink_to(src)
                    linked += 1
                    continue
                except OSError:
                    pass  # fall through to copy, e.g. cross-device
            shutil.copy2(src, dst)
            copied += 1
    log(f"linked {linked:,} file(s), copied {copied:,} file(s) into {dest}")

    # --- merge anomaly catalogues -------------------------------------------
    anomaly_fname = ANOMALY_FILE[modality]
    anomalies = load_anomalies(part_dirs, anomaly_fname)
    (dest / anomaly_fname).write_text(json.dumps(anomalies, indent=2))
    log(f"merged anomaly catalogue: {len(anomalies)} entries -> {dest / anomaly_fname}")

    # --- rebuild the ledger, modality-aware (does not reuse finalize.py's
    #     --rebuild, which is documented image-only) ---------------------------
    total_rows = sum(npy_header_rows(npy) for npy, _ in merged.values())
    dims = {int(np.load(npy, mmap_mode="r").shape[1]) for npy, _ in merged.values()}
    per_worker: Dict[str, int] = defaultdict(int)
    for key, (npy, _) in merged.items():
        per_worker[key.split("/")[0]] += npy_header_rows(npy)

    total_key = LEDGER_TOTAL_KEY[modality]
    progress = {
        "completed_tars": sorted(f"{k}.tar" for k in merged),
        total_key: total_rows,
        "total_failed": 0,
        "merged_from_parts": [p.name for p in part_dirs],
        "merged_at": datetime.now().isoformat(),
        "note": "written by merge_sharded_parts.py; supersedes any per-part ledger",
    }
    (dest / "progress.json").write_text(json.dumps(progress, indent=2))

    run_summary = {
        "merged_at": datetime.now().isoformat(),
        "merged_from_parts": [p.name for p in part_dirs],
        total_key: total_rows,
        "completed_shards": len(merged),
        "per_worker": dict(per_worker),
        "embedding_dim": sorted(dims)[0] if dims else None,
        "dtype": "float32",
        "normalised": True,
        "anomalies": len(anomalies),
    }
    (dest / "run_summary.json").write_text(json.dumps(run_summary, indent=2))
    log(f"wrote progress.json / run_summary.json  ({total_rows:,} rows, dims={sorted(dims)})")

    if args.then_remove_parts:
        for pd in part_dirs:
            shutil.rmtree(pd)
            log(f"removed {pd}")
    else:
        log("part directories left in place (pass --then-remove-parts to reclaim disk "
            "once you've spot-checked the merge)")

    log("=" * 70)
    log(f"merge complete: {dest}")
    log("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
