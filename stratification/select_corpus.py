"""
Defines the canonical 7M-pair corpus as a fixed, ordered set of shard keys,
soft-capped at 7,000,000 rows. Whole shards only -- never split a shard, so
the final count lands at or just under the cap, not exactly on it.

Generation produced 8,003,932 rows (3236 shards) on BOTH the image and text
side, verified shard-for-shard identical (zero row-count mismatches across
all 3236 shards). One row-count table therefore applies to both modalities,
and this manifest is the single place that decides which ~1M rows of the
generated 8M get held out of the 7M pilot corpus.

Ordering is (worker_id, shard_id) ascending -- deterministic, so re-running
this script reproduces an identical cut. This is a corpus-definition
decision, not a T2-only concern: T3 (A/B/C split), T4 (index creation), T5
(landmarks), and T6 (exact-KNN) must all read result/corpus_shard_manifest.json
rather than re-deriving their own shard subset, exactly like the 353-pair
exclusion mask is decided once and applied identically everywhere.

Output: result/corpus_shard_manifest.json
"""

import json
import re
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent  # .../mmir_optimizer
IMAGE_ROOT = REPO_ROOT / "result/embeddings_image/siglip2_so400m_patch16_512"
OUT_PATH = REPO_ROOT / "result/corpus_shard_manifest.json"
SOFT_LIMIT = 7_000_000

_KEY_RE = re.compile(r"worker_(\d+)/shard_(\d+)")


def _sort_key(rel_key: str):
    m = _KEY_RE.match(rel_key)
    return (int(m.group(1)), int(m.group(2)))


def main():
    files = sorted(
        IMAGE_ROOT.glob("worker_*/shard_*.npy"),
        key=lambda p: _sort_key(str(p.relative_to(IMAGE_ROOT)).removesuffix(".npy")),
    )

    included, excluded = [], []
    cum_rows, total_rows = 0, 0
    hit_cap = False
    for f in files:
        key = str(f.relative_to(IMAGE_ROOT)).removesuffix(".npy")
        n = int(np.load(f, mmap_mode="r").shape[0])
        total_rows += n
        # Stop at the FIRST shard that would overflow, rather than skipping it
        # and continuing -- otherwise a later, smaller shard could still slip
        # in and the cut would no longer be a clean prefix of the ordering.
        if hit_cap or cum_rows + n > SOFT_LIMIT:
            hit_cap = True
            excluded.append(key)
        else:
            included.append(key)
            cum_rows += n

    manifest = {
        "soft_limit": SOFT_LIMIT,
        "total_rows_generated": total_rows,
        "included_shards": included,
        "included_rows": cum_rows,
        "excluded_shards": excluded,
        "excluded_rows": total_rows - cum_rows,
        "ordering": "ascending (worker_id, shard_id)",
        "note": (
            "Row counts verified identical between image and text for every "
            "shard, so this manifest applies to both modalities unchanged. "
            "T3-T6 must read this file rather than re-deriving a corpus subset."
        ),
    }
    OUT_PATH.write_text(json.dumps(manifest, indent=2))

    print(f"included: {len(included)} shards, {cum_rows:,} rows (<= {SOFT_LIMIT:,})")
    print(f"excluded: {len(excluded)} shards, {total_rows - cum_rows:,} rows (tail beyond the cap)")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
