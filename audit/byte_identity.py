"""
How much of each duplicate group is literally the same file?

``duplicate_groups.py`` groups rows whose EMBEDDINGS coincide. That is the
level at which the harm to CF@100 happens, but it cannot distinguish

    the corpus literally contains one file many times
        -> a crawl or dedup failure upstream of everything here
    the corpus contains many re-encodings of one picture
        -> different bytes, different JPEG tables, same content
    the corpus contains genuinely different pictures the encoder cannot
    separate
        -> the data is fine and the INDEX is what is being tested

and those three want different responses. This measures which is which.

THREE HASHES, BECAUSE "IDENTICAL" IS THREE DIFFERENT QUESTIONS
--------------------------------------------------------------
raw       sha256 of the file's bytes exactly as stored. This is the strict
          reading of "byte identical". It answers "is this the same file?"
          and nothing else: re-saving a JPEG at the same quality changes it.

pixels    sha256 of the decoded array. Catches the same image wrapped in a
          different container or with different EXIF, which ``raw`` misses.
          It does NOT catch re-encoding -- JPEG is lossy, so a re-encode
          differs in the low bits of nearly every pixel.

dhash     64-bit difference hash of a 9x8 greyscale reduction. Robust to
          re-encoding, resizing and mild compression, so two copies of one
          picture that took different paths through the crawl land on the
          same value. This is the one that matches what a human means by
          "the same image".

Reporting all three is the point. If a group is one value under ``dhash``
but many under ``raw``, it is re-encodings of one picture -- which is what a
scrape failure looks like, since the placeholder is served fresh by each
site. If it is many values under ``dhash`` too, the pictures genuinely
differ and only the caption was boilerplate.

WHY THIS READS ALMOST THE WHOLE TREE
------------------------------------
The members of a placeholder group are scattered across the crawl -- the
13,796-member group touches 2,860 of 3,236 shards -- so there is no targeted
subset to read. Given a full pass is unavoidable, every grouped row is hashed
rather than only the large groups: same I/O, complete answer.

    qsub -I -q standard -P mimr.spons -N mmir_hash \
         -l select=1:ncpus=128 -l walltime=02:00:00

    cd mmir_optimizer/audit
    unset http_proxy https_proxy
    ../.venv/bin/python byte_identity.py --workers 32

Login-node-safe smoke test over a handful of shards, writes to _smoke/:

    ../.venv/bin/python byte_identity.py --max-shards 4 --workers 4
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import tarfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LAION = REPO_ROOT.parent / "laion/worker1_20/tar1"
DEFAULT_GROUPS = REPO_ROOT / "result/duplicate_groups"
DEFAULT_PILOT = REPO_ROOT / "result/pilot_1M/siglip2"


def dhash64(img) -> str:
    """64-bit difference hash: 9x8 greyscale, compare each pixel to its right
    neighbour. Survives re-encoding and resizing, which is exactly what
    separates 'the same picture' from 'the same file'."""
    g = np.asarray(img.convert("L").resize((9, 8)), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).ravel()
    return f"{int(''.join('1' if b else '0' for b in bits), 2):016x}"


def _hash_shard(task):
    """Worker: hash every wanted member of one tar. Top-level + plain data so
    ProcessPoolExecutor can pickle it."""
    from PIL import Image
    laion, w, s, names = task
    tp = Path(laion) / w / f"{s}.tar"
    out = []
    if not tp.exists():
        return w, s, out, False
    want = set(names)
    imgs, caps = {}, {}
    with tarfile.open(tp, "r|") as tf:
        for m in tf:
            st = Path(m.name).stem
            if st not in want:
                continue
            if m.name.endswith(".jpg"):
                imgs[st] = tf.extractfile(m).read()
            elif m.name.endswith(".json"):
                try:
                    caps[st] = json.loads(
                        tf.extractfile(m).read()).get("caption", "")
                except Exception:
                    caps[st] = ""
    for st in want:
        raw = imgs.get(st)
        rec = {"stem": st,
               "raw": "", "pixels": "", "dhash": "", "bytes": 0,
               "w": 0, "h": 0,
               "cap": hashlib.sha256(
                   caps.get(st, "").strip().lower().encode()).hexdigest()[:16]}
        if raw:
            rec["raw"] = hashlib.sha256(raw).hexdigest()[:16]
            rec["bytes"] = len(raw)
            try:
                im = Image.open(io.BytesIO(raw)).convert("RGB")
                rec["w"], rec["h"] = im.size
                rec["pixels"] = hashlib.sha256(
                    np.asarray(im).tobytes()).hexdigest()[:16]
                rec["dhash"] = dhash64(im)
            except Exception:
                pass
        out.append(rec)
    return w, s, out, True


def run(args):
    smoke = args.max_shards > 0
    if not smoke and not os.environ.get("PBS_JOBID") and not args.allow_login_node:
        sys.exit(
            "refusing to run on what looks like a login node: this streams "
            "~48 GiB of tar. Submit a job, or use --max-shards for a smoke test.")

    pair, roles = {}, {}
    with open(args.pilot_dir / "pair_index.csv") as f:
        for r in csv.DictReader(f):
            pair[int(r["int_id"])] = r["pair_id"]
            roles[int(r["int_id"])] = r["role"]

    # Every row that is in ANY group, either modality -- one pass serves both.
    want_ids = set()
    side = {}
    for mod in ("image", "text"):
        g = np.load(args.groups_dir / args.model / mod / "row_group.npy")
        ids = np.load(args.groups_dir / args.model / mod / "row_int_id.npy")
        side[mod] = (g, ids)
        want_ids |= set(ids[g >= 0].tolist())
    print(f"  {len(want_ids):,} grouped rows to hash")

    by_shard = defaultdict(list)
    for i in want_ids:
        w, s, n = pair[int(i)].split("/")
        by_shard[(w, s)].append(n)
    shards = sorted(by_shard)
    if smoke:
        shards = shards[:args.max_shards]
        print(f"  SMOKE: {len(shards)} shards only")
    print(f"  {len(shards):,} shards, {args.workers} workers "
          f"(~{len(shards) * 17 / 1024:.1f} GiB)")

    from concurrent.futures import ProcessPoolExecutor
    tasks = [(str(args.laion), w, s, by_shard[(w, s)]) for (w, s) in shards]
    stem_to_id = {pair[int(i)].rsplit("/", 1)[1]: int(i) for i in want_ids}

    recs = {}
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for w, s, out, ok in ex.map(_hash_shard, tasks, chunksize=4):
            for rec in out:
                iid = stem_to_id.get(rec["stem"])
                if iid is not None:
                    recs[iid] = rec
            done += 1
            if done % 200 == 0:
                print(f"    {done:,}/{len(shards):,} shards", flush=True)

    dest = (args.groups_dir / "_smoke" if smoke else args.groups_dir) / args.model
    dest.mkdir(parents=True, exist_ok=True)
    with open(dest / "row_hashes.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["int_id", "pair_id", "role", "raw_sha", "pixel_sha",
                     "dhash", "bytes", "width", "height", "caption_sha"])
        for iid in sorted(recs):
            r = recs[iid]
            wr.writerow([iid, pair[iid], roles[iid], r["raw"], r["pixels"],
                         r["dhash"], r["bytes"], r["w"], r["h"], r["cap"]])
    print(f"  wrote {dest / 'row_hashes.csv'}  ({len(recs):,} rows)")

    # ---- per-group summary ------------------------------------------------
    for mod in ("image", "text"):
        g, ids = side[mod]
        n_g = int(g.max()) + 1
        rows = []
        for gid in range(n_g):
            member = ids[g == gid]
            got = [recs[int(i)] for i in member if int(i) in recs]
            if not got:
                continue
            def uniq(k):
                v = [r[k] for r in got if r[k]]
                return len(set(v)), (max(v.count(x) for x in set(v)) / len(v)
                                     if v else 0.0)
            n_raw, sh_raw = uniq("raw")
            n_pix, sh_pix = uniq("pixels")
            n_dh, sh_dh = uniq("dhash")
            n_cap, sh_cap = uniq("cap")
            rows.append([gid, len(member), len(got),
                         n_raw, f"{sh_raw:.3f}", n_pix, f"{sh_pix:.3f}",
                         n_dh, f"{sh_dh:.3f}", n_cap, f"{sh_cap:.3f}"])
        out = dest / f"group_identity_{mod}.csv"
        with open(out, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["group_id", "n_members", "n_hashed",
                         "n_raw_sha", "largest_raw_share",
                         "n_pixel_sha", "largest_pixel_share",
                         "n_dhash", "largest_dhash_share",
                         "n_caption_sha", "largest_caption_share"])
            wr.writerows(rows)
        print(f"  wrote {out}  ({len(rows):,} groups)")

    json.dump({
        "created": datetime.now().isoformat(),
        "model": args.model, "smoke": bool(smoke),
        "n_rows_hashed": len(recs), "n_shards_read": len(shards),
        "hashes": {
            "raw_sha": "sha256 of the stored file bytes -- strict byte identity",
            "pixel_sha": "sha256 of the decoded RGB array -- same image, "
                         "different container; does NOT survive re-encoding",
            "dhash": "64-bit difference hash -- survives re-encoding and "
                     "resizing; this is 'the same picture' as a human means it",
            "caption_sha": "sha256 of the caption, stripped and lowercased",
        },
        "reading": (
            "one dhash but many raw_sha => re-encodings of one picture, the "
            "signature of a placeholder served fresh by each site. many dhash "
            "=> the pictures genuinely differ and only the caption coincided."),
    }, open(dest / "byte_identity_meta.json", "w"), indent=2)
    return recs


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="siglip2")
    ap.add_argument("--laion", type=Path, default=DEFAULT_LAION)
    ap.add_argument("--groups-dir", type=Path, default=DEFAULT_GROUPS)
    ap.add_argument("--pilot-dir", type=Path, default=DEFAULT_PILOT)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--max-shards", type=int, default=0,
                    help="smoke test: read only the first N shards, write to _smoke/")
    ap.add_argument("--allow-login-node", action="store_true")
    args = ap.parse_args()
    print("byte_identity")
    run(args)


if __name__ == "__main__":
    main()
