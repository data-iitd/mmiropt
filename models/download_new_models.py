#!/usr/bin/env python
"""Download the second wave of backbones into models/.

    siglip2  so400m-patch16-512      -> google/siglip2-so400m-patch16-512
    jinaclip jina-clip-v2            -> jinaai/jina-clip-v2  (+2 code repos)
    jinav4   jina-embeddings-v4      -> jinaai/jina-embeddings-v4
    qwen3vl  Qwen3-VL-Embedding-2B   -> Qwen/Qwen3-VL-Embedding-2B

TRANSFER SPEED - THE ONE SETTING THAT MATTERS
    The IITD proxy throttles ~170-350 KB/s PER CONNECTION, so a single-stream
    download of a 4.6GB file takes hours. This sets HF_XET_HIGH_PERFORMANCE=1,
    which makes Xet fetch a single file over many concurrent chunks - measured
    at roughly 5-10x the single-stream rate on this link.

    Do NOT reach for HF_HUB_ENABLE_HF_TRANSFER: this huggingface_hub version
    ignores it and prints a deprecation warning. hf_transfer is not used any
    more; Xet replaced it.

    max_workers only parallelises across FILES, which does nothing for a repo
    whose weight is one big safetensors - hence the Xet variable, not workers.

JINA NEEDS THREE REPOS, NOT ONE
    jina-clip-v2's config.json has
        auto_map.AutoModel = "jinaai/jina-clip-implementation--modeling_clip..."
    so the model CODE lives in a different repo, and its text tower pulls a
    third. Compute nodes have no network and run with HF_HUB_OFFLINE=1, so
    trust_remote_code can only use what is already on disk. Downloading
    jina-clip-v2 alone gives you weights that cannot be loaded there.

RESUMING
    Safe to re-run: snapshot_download skips files already complete. A partial
    Xet transfer is NOT resumed though - each attempt starts a fresh
    .incomplete blob and orphans the previous one, so an interrupted big file
    restarts from zero and the old partial is dead weight worth deleting.

Everything (weights + HF cache) stays under models/ so the project is relocatable.
"""

import os
import sys
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent

# Keep the HF cache inside the project rather than ~/.cache/huggingface.
os.environ.setdefault("HF_HOME", str(MODELS_DIR / ".hf_cache"))
# The transfer knob. See the module docstring - this is the whole ballgame on
# a throttled link, and it must be set BEFORE huggingface_hub is imported.
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
# Left over from an older attempt; this version ignores it and warns.
os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)

from huggingface_hub import snapshot_download  # noqa: E402

# (local subdir, repo_id, ignore_patterns, note)
TARGETS = [
    (
        "siglip2_so400m_patch16_512",
        "google/siglip2-so400m-patch16-512",
        ["*.msgpack", "*.h5"],
        "1.14B, 1152-d, image 512px / text 64 tokens",
    ),
    (
        "jina_clip_v2",
        "jinaai/jina-clip-v2",
        # onnx/ is ~9GB of duplicate weights for a runtime we do not use.
        ["onnx/*", "*.onnx*", "pytorch_model.bin"],
        "0.87B, 1024-d, image 512px / text 8192 tokens",
    ),
    (
        "jina_clip_v2/_code_impl",
        "jinaai/jina-clip-implementation",
        None,
        "modeling code jina-clip-v2's auto_map points at",
    ),
    (
        "jina_clip_v2/_code_xlmr",
        "jinaai/xlm-roberta-flash-implementation",
        None,
        "text-tower code pulled in turn by the above",
    ),
    (
        "jina_embeddings_v4",
        "jinaai/jina-embeddings-v4",
        None,
        "3.75B, 2048-d; code is self-contained in the repo",
    ),
    (
        "qwen3_vl_embedding_2b",
        "Qwen/Qwen3-VL-Embedding-2B",
        None,
        "2.13B, 2048-d, native-resolution images, last-token pooling",
    ),
]


def fetch(subdir, repo_id, ignore_patterns, note):
    dest = MODELS_DIR / subdir
    print(f"\n=== {repo_id} -> models/{subdir} ===", flush=True)
    print(f"    {note}", flush=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(dest),
        ignore_patterns=ignore_patterns,
        max_workers=8,
    )


def main():
    only = set(sys.argv[1:])
    if only:
        print(f"restricting to: {sorted(only)}")

    rc = 0
    for subdir, repo_id, ignore, note in TARGETS:
        if only and not any(o in subdir or o in repo_id for o in only):
            continue
        try:
            fetch(subdir, repo_id, ignore, note)
        except Exception as exc:
            print(f"FAILED {repo_id}: {type(exc).__name__}: {exc}", file=sys.stderr)
            rc = 1

    print("\nAll requested snapshots downloaded." if rc == 0
          else "\nSome downloads failed - re-run to retry (completed files are skipped).")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
