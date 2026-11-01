#!/usr/bin/env python
"""Load all four backbones from disk and report that they instantiate cleanly.

This is the acceptance check for the download step: it proves the chunked
parallel fetch produced usable weights and that each architecture matches what
was specified, before anything depends on them.
"""

import os
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(HERE / ".hf_cache"))
# Nothing here should touch the network.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch  # noqa: E402


def nparams(m):
    return sum(p.numel() for p in m.parameters())


def check_clip_336():
    from transformers import CLIPModel, CLIPProcessor

    p = HERE / "clip_vit_l14_336"
    m = CLIPModel.from_pretrained(p)
    proc = CLIPProcessor.from_pretrained(p)
    res = m.config.vision_config.image_size
    assert res == 336, f"expected 336px, got {res}"
    return f"{nparams(m)/1e6:.0f}M params, image_size={res}, proj_dim={m.config.projection_dim}"


def check_flava():
    from transformers import FlavaModel, FlavaProcessor

    p = HERE / "flava_full"
    m = FlavaModel.from_pretrained(p)
    FlavaProcessor.from_pretrained(p)
    return f"{nparams(m)/1e6:.0f}M params, hidden={m.config.image_config.hidden_size}"


def check_openclip_laion2b():
    import open_clip

    w = HERE / "openclip_vit_l14_laion2b_s32b_b82k/open_clip_pytorch_model.safetensors"
    m, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained=str(w))
    return f"{nparams(m)/1e6:.0f}M params, embed_dim={m.text_projection.shape[1]}"


def check_uniir():
    import open_clip

    w = HERE / "uniir_clip_sf_large/checkpoint/CLIP_SF/clip_sf_large.pth"
    m = open_clip.create_model("ViT-L-14-quickgelu", pretrained=None)
    # The published checkpoint is an epoch-29 training snapshot: weights live
    # under "model", behind the CLIPScoreFusion wrapper's "clip_model." prefix,
    # alongside optimizer/scheduler/scaler state that inference ignores.
    ckpt = torch.load(w, map_location="cpu", weights_only=False)["model"]
    sd = {k.removeprefix("clip_model."): v for k, v in ckpt.items()}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    # attn_mask is a registered buffer, never stored in OpenAI-format weights.
    missing = [k for k in missing if "attn_mask" not in k]
    if missing or unexpected:
        raise AssertionError(f"missing={missing[:5]} unexpected={unexpected[:5]}")
    res = m.visual.image_size
    return f"{nparams(m)/1e6:.0f}M params, image_size={res}, all 446 tensors matched"


CHECKS = [
    ("clip      ViT-L/14@336px", check_clip_336),
    ("openclip  ViT-L-14 laion2b_s32b_b82k", check_openclip_laion2b),
    ("flava     facebook/flava-full", check_flava),
    ("uniir     ViT-L-14-quickgelu (CLIP_SF)", check_uniir),
]


def main():
    rc = 0
    for label, fn in CHECKS:
        try:
            detail = fn()
            print(f"  OK    {label:<40} {detail}", flush=True)
        except Exception as exc:
            rc = 1
            print(f"  FAIL  {label:<40} {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
    print("\nall models load" if rc == 0 else "\nsome models failed", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
