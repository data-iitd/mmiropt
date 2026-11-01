#!/usr/bin/env python3
"""
OpenCLIP image-normalisation check: how far do the image vectors, and the
exact neighbour lists built on them, move if images are normalised with the
mean/std the checkpoint was trained with?

WHY
    ViT-L-14 / laion2b_s32b_b82k was trained with mean = std = (0.5, 0.5, 0.5)
    (INCEPTION_MEAN/STD): open_clip's own tag table says so
    (pretrained.py, laion2b_s32b_b82k=_pcfg(..., mean=INCEPTION_MEAN,
    std=INCEPTION_STD)) and so does the checkpoint's open_clip_config.json.
    gen_openclip.py passes pretrained= a FILE PATH, and for a file path
    open_clip never looks the tag up (factory.py: "elif
    os.path.isfile(pretrained)" sets only checkpoint_path), so the transform
    keeps PreprocessCfg's default, the OpenAI mean/std. Every stored OpenCLIP
    image vector was made from inputs normalised the wrong way.

WHAT IT DOES (one process, one GPU, nothing but the report written)
    Same 84-shard sample as the Jina and SigLIP 2 checks (A and B rows).
    Each image is decoded once, exactly as T1 decoded it (PIL open ->
    convert("RGB") -> transform; a failed decode is the zero placeholder), and
    passed through two transforms that differ ONLY in Normalize:
        P  T1's transform, OpenAI mean/std  (S vs P = rerun noise floor)
        C  the same transform, mean = std = 0.5
    Both are encoded through T1's OpenClipEmbedder (fp16 autocast, batch 64).
    Compared with S, the stored image vector:
        - per-image cosine
        - exact top-k overlap, B queries vs the sample's A rows, for i2i, i2t
          and t2i (the stored OpenCLIP text vectors are unchanged; t2t does
          not involve an image vector)
        - Spearman of the 10th / 100th neighbour similarity across queries
        - paired retrieval inside B (caption <-> own image) recall@1/10

OUTPUT
    result/audit/openclip_imagenorm_check/report.json

RUN THIS IN A PBS JOB (audit/openclip_imagenorm_check.pbs), not on the login node.
"""

import io
import sys
import copy
import json
import time
import tarfile
import argparse
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from jina_prefix_check import log, stem_of, DATA_ROOT
import embed_check_common as K

REPO_ROOT = Path(__file__).resolve().parent.parent
TEXT_DIR = REPO_ROOT / "result/embeddings_text/openclip_vit_l14_laion2b"
IMAGE_DIR = REPO_ROOT / "result/embeddings_image/openclip_vit_l14_laion2b"
SAMPLE = REPO_ROOT / "result/audit/jina_prefix_check/sample.json"
DEFAULT_OUT = REPO_ROOT / "result/audit/openclip_imagenorm_check"
MODEL_CFG = REPO_ROOT / "models/openclip_vit_l14_laion2b_s32b_b82k/open_clip_config.json"

BATCH = 64            # T1 (openclip.log: "Batch size : 64", AMP True)
CORRECT = (0.5, 0.5, 0.5)


def read_shard_images(shard: str, rows):
    worker, name = shard.split("/")
    want = {stem_of(pid) for pid, _, _ in rows}
    got = {}
    with tarfile.open(DATA_ROOT / worker / f"{name}.tar", "r") as tf:
        for m in tf:
            if not m.isfile():
                continue
            p = Path(m.name)
            if p.suffix.lower() != ".jpg" or p.stem not in want:
                continue
            fh = tf.extractfile(m)
            got[p.stem] = fh.read() if fh else b""
    missing = want - set(got)
    if missing:
        raise RuntimeError(f"{shard}: {len(missing)} images not in tar")
    return [got[stem_of(pid)] for pid, _, _ in rows]


def main():
    ap = argparse.ArgumentParser(description="OpenCLIP image normalisation check.")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--sample", default=str(SAMPLE))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num-workers", type=int, default=24)
    args = ap.parse_args()

    import torch
    from torch.utils.data import Dataset, DataLoader
    from torchvision.transforms import Normalize
    from PIL import Image
    import open_clip
    from open_clip.pretrained import get_pretrained_cfg
    sys.path.insert(0, str(REPO_ROOT / "embeddingsgenerator"))
    import gen_openclip as T1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sample = json.load(open(args.sample, encoding="utf-8"))
    shards = sample["shards"]

    t0 = time.time()
    with ThreadPoolExecutor(8) as ex:
        per_shard = list(ex.map(lambda s: read_shard_images(s, sample["rows"][s]), shards))
    names, roles, blobs = [], [], []
    for s, imgs in zip(shards, per_shard):
        for (pid, _, role), b in zip(sample["rows"][s], imgs):
            names.append(pid); roles.append(role); blobs.append(b)
    del per_shard
    roles = np.array(roles)
    n = len(names)
    log(f"{n:,} images read from {len(shards)} shards in {time.time() - t0:.0f}s")

    emb = T1.OpenClipEmbedder(args.device, BATCH, True, T1.DEFAULT_WEIGHTS, args.num_workers)
    tf_P = emb.transform
    tf_C = copy.deepcopy(tf_P)
    norms = [t for t in tf_C.transforms if isinstance(t, Normalize)]
    if len(norms) != 1:
        raise RuntimeError(f"expected one Normalize in T1's transform, got {len(norms)}: {tf_P}")
    norms[0].mean, norms[0].std = list(CORRECT), list(CORRECT)

    tag = get_pretrained_cfg("ViT-L-14", "laion2b_s32b_b82k") or {}
    guard = {
        "open_clip_version": getattr(open_clip, "__version__", None),
        "transform_T1": repr(tf_P),
        "transform_correct": repr(tf_C),
        "tag_table_mean_std": [list(tag.get("mean", [])), list(tag.get("std", []))],
        "checkpoint_open_clip_config_preprocess_cfg":
            json.load(open(MODEL_CFG, encoding="utf-8")).get("preprocess_cfg"),
    }
    log(f"guard: {json.dumps(guard)}")

    class Pairs(Dataset):
        def __len__(self):
            return n

        def __getitem__(self, k):
            try:
                with Image.open(io.BytesIO(blobs[k])) as im:
                    rgb = im.convert("RGB")
                    return tf_P(rgb), tf_C(rgb), True
            except Exception:
                return emb.blank, emb.blank, False

    loader = DataLoader(Pairs(), batch_size=BATCH, num_workers=args.num_workers,
                        shuffle=False, drop_last=False, prefetch_factor=4)
    P, C, ok = [], [], []
    t0 = time.time()
    with torch.inference_mode():
        for xp, xc, good in loader:
            P.append(emb._encode_batch(xp).numpy())
            C.append(emb._encode_batch(xc).numpy())
            ok.append(good.numpy())
    P = np.concatenate(P).astype(np.float32)
    C = np.concatenate(C).astype(np.float32)
    ok = np.concatenate(ok)
    log(f"encoded {n:,} x2 in {time.time() - t0:.0f}s ({int((~ok).sum())} decode failures)")
    dim = emb.dim
    del emb, blobs, loader
    torch.cuda.empty_cache()

    S_t, S_i = K.join_stored(names, (TEXT_DIR, IMAGE_DIR), dim)
    log("stored text and image vectors joined by pair id")

    K.exact_fp32(torch)
    dev = torch.device(args.device)
    image = {"S": S_i, "P": P, "C": C}
    pairs = (("S", "P"), ("S", "C"), ("P", "C"))

    report = {
        "created": datetime.now().isoformat(),
        "sample": {"source": str(args.sample),
                   **{k: sample[k] for k in ("seed", "scope", "n_shards", "n_A", "n_B")}},
        "n": n, "n_A": int((roles == "A").sum()), "n_B": int((roles == "B").sum()),
        "decode_failures": int((~ok).sum()),
        "versions": {"S": "stored T1 image vector (OpenAI mean/std)",
                     "P": "re-embedded now with T1's transform (OpenAI mean/std)",
                     "C": "re-embedded now, same transform with mean = std = 0.5"},
        "guard": guard,
        "cosine_per_image": K.cosine_block(image, pairs),
    }
    log(f"cos S.P mean {report['cosine_per_image']['S_vs_P']['mean']:.6f}   "
        f"cos S.C mean {report['cosine_per_image']['S_vs_C']['mean']:.6f}")

    A = np.flatnonzero(roles == "A")
    B = np.flatnonzero(roles == "B")
    directions = {
        "i2i": (lambda v: image[v][B], lambda v: image[v][A]),
        "i2t": (lambda v: image[v][B], lambda v: S_t[A]),
        "t2i": (lambda v: S_t[B],      lambda v: image[v][A]),
    }
    report["knn"] = {"queries": "B", "corpus": "A rows of the sample",
                     "n_queries": int(len(B)), "n_corpus": int(len(A))}
    for dname, (qf, xf) in directions.items():
        block = K.knn_block(torch, dev, qf, xf, image, pairs)
        report["knn"][dname] = block
        log(f"{dname}: overlap@100 mean  S.P {block['S_vs_P']['overlap@100']['mean']:.4f}  "
            f"S.C {block['S_vs_C']['overlap@100']['mean']:.4f}")

    report["paired_retrieval_B"] = {"n": int(len(B))}
    for v in image:
        report["paired_retrieval_B"][v] = K.paired(torch, dev, S_t[B], image[v][B])
    log(f"paired B recall@1 t2i  S {report['paired_retrieval_B']['S']['t2i']['recall@1']:.4f}  "
        f"C {report['paired_retrieval_B']['C']['t2i']['recall@1']:.4f}")

    with open(out / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log(f"wrote {out / 'report.json'}")


if __name__ == "__main__":
    main()
