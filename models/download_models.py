#!/usr/bin/env python
"""Download the MMIR backbones into models/ as self-contained snapshots.

Architectures are pinned exactly as specified for the project:

  clip     ViT-L/14@336px               -> openai/clip-vit-large-patch14-336
  openclip ViT-L-14 / laion2b_s32b_b82k -> laion/CLIP-ViT-L-14-laion2B-s32B-b82K
  flava    facebook/flava-full          -> facebook/flava-full
  uniir    ViT-L-14-quickgelu           -> TIGER-Lab/UniIR (CLIP_SF large checkpoint)

UniIR note: clip_sf_large.pth is a fine-tune of OpenAI CLIP ViT-L/14 (224px,
QuickGELU) -- open_clip calls that exact architecture "ViT-L-14-quickgelu".
UniIR's own code builds the arch via OpenAI's `clip` package and then loads
this checkpoint over it, so the checkpoint is expected to carry the full set of
weights. inspect_uniir.py verifies that before we decide whether the separate
OpenAI base weights are needed at all.

Network note (Pragya): the IITD proxy throttles ~170-250 KB/s *per connection*
but scales near-linearly with parallel connections, so hf_transfer's chunked
parallel downloader is enabled -- it is worth ~4-5x here.

Everything (weights + HF cache) stays under models/ so the project is relocatable.
"""

import os
import sys
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent

# Keep the HF cache inside the project rather than ~/.cache/huggingface.
os.environ.setdefault("HF_HOME", str(MODELS_DIR / ".hf_cache"))

from huggingface_hub import snapshot_download  # noqa: E402

# (local subdir, repo_id, ignore_patterns, allow_patterns)
#
# This script fetches ONLY the small files -- configs, tokenizers, preprocessor
# json. The large weight files are handled by download_weights.sh, which
# chunk-parallelises them (worth ~5x on the throttled IITD proxy).
#
# Every weight extension is excluded below on purpose. snapshot_download tracks
# what it has fetched via metadata under local_dir/.cache/huggingface; files put
# there by the shell script carry no such metadata, so without these ignores it
# would re-download all 9.3GB.
WEIGHTS = ["*.bin", "*.pth", "*.safetensors", "*.h5", "*.msgpack", "*.pt"]

HF_TARGETS = [
    (
        "clip_vit_l14_336",
        "openai/clip-vit-large-patch14-336",
        WEIGHTS,
        None,
    ),
    (
        "flava_full",
        "facebook/flava-full",
        WEIGHTS,
        None,
    ),
    (
        "openclip_vit_l14_laion2b_s32b_b82k",
        "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
        # runs/ is training tensorboard logs.
        WEIGHTS + ["runs/*"],
        None,
    ),
    # TIGER-Lab/UniIR holds nothing but the two .pth checkpoints, so there is
    # no config to fetch -- clip_sf_large.pth is already in place from
    # download_weights.sh, and BLIP_FF is a different architecture we skip.
]


def fetch_hf(subdir, repo_id, ignore_patterns, allow_patterns):
    dest = MODELS_DIR / subdir
    print(f"\n=== {repo_id} -> models/{subdir} ===", flush=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(dest),
        ignore_patterns=ignore_patterns,
        allow_patterns=allow_patterns,
        max_workers=8,
    )


def main():
    for subdir, repo_id, ignore, allow in HF_TARGETS:
        try:
            fetch_hf(subdir, repo_id, ignore, allow)
        except Exception as exc:
            print(f"FAILED {repo_id}: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
    print("\nAll snapshots downloaded.")
    print("Next: python models/inspect_uniir.py  (decides if base weights are needed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
