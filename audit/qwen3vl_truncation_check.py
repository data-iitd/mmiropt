#!/usr/bin/env python3
"""
Qwen3-VL text truncation check: how many corpus captions lost the END of the
chat template to the 512-token cap, and how much did their vectors suffer?

WHY
    gen_qwen3vl_text.py renders the chat template FIRST
        <|im_start|>system\\nRepresent the user's input.<|im_end|>\\n
        <|im_start|>user\\n{caption}<|im_end|>\\n<|im_start|>assistant\\n
    and only then tokenises with truncation=True, max_length=512. Truncation
    cuts from the right, so a caption long enough to push the rendered input
    past 512 tokens loses "<|im_end|>\\n<|im_start|>assistant\\n" - and the
    model's vector is the hidden state of the LAST token (1_Pooling:
    "lasttoken"). For those captions the pooled token is a caption word the
    model was never trained to pool from. Nobody has counted them.

WHAT IT DOES
    Stage 1 (CPU, every pair in the 7M split: A, B and C)
        Renders each caption exactly as T1 did and counts its untruncated token
        length. Reports the length distribution and every caption over the cap.
    Stage 2 (GPU, only the captions over the cap; up to --max-embed of them)
        Encodes each twice through T1's Qwen3VLTextEmbedder (bf16):
            T  max_text_len 512, as T1   (S vs T = rerun noise floor)
            F  no cap (model limit)       (the template stays intact)
        and compares with S, the stored vector: per-caption cosine,
        similarity to the pair's own stored image, and the own image's rank
        among the 84-shard sample's A images (the Jina check's sample; any
        affected pair is removed from that pool).

OUTPUT
    result/audit/qwen3vl_truncation_check/report.json

RUN THIS IN A PBS JOB (audit/qwen3vl_truncation_check.pbs), not on the login node.
"""

import os
import sys
import csv
import json
import time
import random
import tarfile
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from multiprocessing import get_context

import numpy as np

from jina_prefix_check import log, summ, caption_text, stem_of, DATA_ROOT, PAIR_INDEX
import embed_check_common as K

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "models/qwen3_vl_embedding_2b"
TEXT_DIR = REPO_ROOT / "result/embeddings_text/qwen3_vl_embedding_2b"
IMAGE_DIR = REPO_ROOT / "result/embeddings_image/qwen3_vl_embedding_2b"
SAMPLE = REPO_ROOT / "result/audit/jina_prefix_check/sample.json"
DEFAULT_OUT = REPO_ROOT / "result/audit/qwen3vl_truncation_check"

CAP = 512                                   # T1's max_text_len (all 4 parts)
INSTRUCTION = "Represent the user's input."  # T1's instruction (all 4 parts)
LEN_EDGES = (128, 256, 384, 448, 480, 512, 1024, 2048, 4096)

_ROWS = None     # shard -> [(pair_id, role)]; set before the pool forks
_PROC = None


def _conversation(text: str):
    """Verbatim gen_qwen3vl_text.Qwen3VLTextEmbedder._conversation."""
    return [
        {"role": "system", "content": [{"type": "text", "text": INSTRUCTION}]},
        {"role": "user", "content": [{"type": "text", "text": text}]},
    ]


def _init_worker():
    global _PROC
    from transformers import AutoProcessor
    _PROC = AutoProcessor.from_pretrained(str(MODEL_DIR))


def _count_shard(shard: str):
    rows = _ROWS[shard]
    worker, name = shard.split("/")
    pos = {stem_of(pid): i for i, (pid, _) in enumerate(rows)}
    caps = [None] * len(rows)
    with tarfile.open(DATA_ROOT / worker / f"{name}.tar", "r") as tf:
        for m in tf:
            if not m.isfile():
                continue
            p = Path(m.name)
            if p.suffix.lower() != ".json" or p.stem not in pos:
                continue
            fh = tf.extractfile(m)
            caps[pos[p.stem]] = caption_text(fh.read() if fh else b"")
    if any(c is None for c in caps):
        raise RuntimeError(f"{shard}: {sum(c is None for c in caps)} captions not in tar")
    rendered = _PROC.apply_chat_template(
        [_conversation(c) for c in caps], add_generation_prompt=True, tokenize=False)
    ids = _PROC(text=rendered, padding=False, truncation=False)["input_ids"]
    lens = np.fromiter((len(x) for x in ids), dtype=np.int64, count=len(ids))
    over = [(rows[i][0], rows[i][1], int(lens[i]), caps[i]) for i in np.flatnonzero(lens > CAP)]
    return shard, lens, over


def main():
    global _ROWS
    ap = argparse.ArgumentParser(description="Qwen3-VL text truncation check.")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--sample", default=str(SAMPLE))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--procs", type=int, default=30)
    ap.add_argument("--max-embed", type=int, default=20000)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------- stage 1
    rows = defaultdict(list)
    role_total = Counter()
    with open(PAIR_INDEX, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            pid, role = row[1], row[-1]
            rows[pid.rsplit("/", 1)[0]].append((pid, role))
            role_total[role] += 1
    _ROWS = rows
    shards = sorted(rows)
    log(f"pair_index: {sum(role_total.values()):,} pairs in {len(shards):,} shards {dict(role_total)}")

    t0 = time.time()
    all_lens, affected = [], []
    with get_context("fork").Pool(args.procs, initializer=_init_worker) as pool:
        for i, (shard, lens, over) in enumerate(
                pool.imap_unordered(_count_shard, shards, chunksize=2), 1):
            all_lens.append(lens)
            affected.extend(over)
            if i % 250 == 0 or i == len(shards):
                log(f"[{i}/{len(shards)}] shards counted, {len(affected):,} over the cap "
                    f"({time.time() - t0:.0f}s)")
    lens = np.concatenate(all_lens)
    affected.sort()
    aff_roles = Counter(a[1] for a in affected)

    report = {
        "created": datetime.now().isoformat(),
        "cap": CAP, "instruction": INSTRUCTION,
        "corpus": {"pairs": int(len(lens)), "by_role": dict(role_total)},
        "rendered_length": {
            **summ(lens), "p99": float(np.percentile(lens, 99)),
            "p99.9": float(np.percentile(lens, 99.9)), "p99.99": float(np.percentile(lens, 99.99)),
            "count_over": {str(e): int((lens > e).sum()) for e in LEN_EDGES},
        },
        "over_cap": {
            "count": len(affected),
            "fraction": len(affected) / len(lens),
            "by_role": dict(aff_roles),
            "fraction_of_role": {k: aff_roles.get(k, 0) / v for k, v in role_total.items()},
        },
    }
    log(f"over the cap: {len(affected):,} of {len(lens):,} "
        f"({100 * len(affected) / len(lens):.4f}%), by role {dict(aff_roles)}")

    if affected:
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(str(MODEL_DIR))
        rend = proc.apply_chat_template([_conversation(affected[0][3])],
                                        add_generation_prompt=True, tokenize=False)
        ids_t = proc(text=rend, truncation=True, max_length=CAP)["input_ids"][0]
        ids_f = proc(text=rend, truncation=False)["input_ids"][0]
        report["truncation_example"] = {
            "pair_id": affected[0][0], "full_length": len(ids_f), "t1_length": len(ids_t),
            "t1_input_tail": proc.tokenizer.decode(ids_t[-12:]),
            "full_input_tail": proc.tokenizer.decode(ids_f[-12:]),
        }
        log(f"example tail T1: {report['truncation_example']['t1_input_tail']!r}  "
            f"full: {report['truncation_example']['full_input_tail']!r}")

    with open(out / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(out / "over_cap_pairs.json", "w", encoding="utf-8") as f:
        json.dump([{"pair_id": a[0], "role": a[1], "rendered_tokens": a[2]} for a in affected], f)
    if not affected:
        log("nothing over the cap; stage 2 skipped")
        return

    # ---------------------------------------------------------- stage 2
    import torch
    sys.path.insert(0, str(REPO_ROOT / "embeddingsgenerator"))
    import gen_qwen3vl_text as T1

    # torch 2.13 routes the RoPE outer-product bmm ((B,M,1) @ (B,1,N)) to a
    # Triton kernel whose launcher compiles against Python.h, and the compute
    # nodes have no Python headers (job 106075 died on exactly that). An outer
    # product is one multiply per element, so cuBLAS gives the same bits.
    from torch._native import triton_utils as _native_triton
    _native_triton.deregister_op_overrides()
    report["torch_native_triton_overrides"] = "deregistered (no Python.h on compute nodes)"

    sub = affected if len(affected) <= args.max_embed else \
        sorted(random.Random(0).sample(affected, args.max_embed))
    names = [a[0] for a in sub]
    texts = [a[3] for a in sub]
    emb = T1.Qwen3VLTextEmbedder(args.device, 128, True, T1.DEFAULT_MODEL_DIR,
                                 num_workers=0, instruction=INSTRUCTION, max_text_len=CAP)

    def encode(cap, bs):
        emb.max_text_len = cap
        res = []
        with torch.inference_mode():
            for i in range(0, len(texts), bs):
                res.append(emb._encode_batch(texts[i:i + bs]).numpy())
        return np.concatenate(res).astype(np.float32)

    t0 = time.time()
    T = encode(CAP, 32)
    F = encode(emb.model_limit, 4)
    log(f"encoded {len(texts):,} over-cap captions x2 in {time.time() - t0:.0f}s")
    dim = emb.dim
    del emb
    torch.cuda.empty_cache()

    S_t, S_i = K.join_stored(names, (TEXT_DIR, IMAGE_DIR), dim)
    text = {"S": S_t, "T": T, "F": F}
    st2 = {"n_embedded": len(sub), "sampled": len(sub) < len(affected),
           "versions": {"S": "stored T1 vector (template end cut by the 512 cap)",
                        "T": "re-embedded now, cap 512 (T1 path)",
                        "F": "re-embedded now, no cap (template intact)"},
           "cosine_per_caption": K.cosine_block(text, (("S", "T"), ("S", "F"), ("T", "F"))),
           "own_image_similarity": {v: summ((text[v] * S_i).sum(1)) for v in text}}

    # Own image's rank among the sample's A images (affected pairs removed).
    K.exact_fp32(torch)
    dev = torch.device(args.device)
    sample = json.load(open(args.sample, encoding="utf-8"))
    skip = set(names)
    pool_names = [pid for s in sample["shards"] for pid, _, ro in sample["rows"][s]
                  if ro == "A" and pid not in skip]
    pool = K.join_stored(pool_names, (IMAGE_DIR,), dim)[0]
    Pg = torch.from_numpy(pool).to(dev)
    st2["rank_pool"] = {"source": str(args.sample), "n_images": len(pool_names)}
    for v in text:
        q = torch.from_numpy(text[v]).to(dev)
        own = torch.from_numpy((text[v] * S_i).sum(1)).to(dev)
        rank = torch.cat([((q[i:i + 2048] @ Pg.T) > own[i:i + 2048, None]).sum(1)
                          for i in range(0, len(q), 2048)]).cpu().numpy()
        st2["rank_pool"][v] = {"recall@1": float((rank < 1).mean()),
                               "recall@10": float((rank < 10).mean()),
                               "recall@100": float((rank < 100).mean()),
                               "median_rank": float(np.median(rank))}
    report["stage2"] = st2
    log(f"cos S.T mean {st2['cosine_per_caption']['S_vs_T']['mean']:.5f}  "
        f"S.F mean {st2['cosine_per_caption']['S_vs_F']['mean']:.5f} | own-image R@1 "
        f"S {st2['rank_pool']['S']['recall@1']:.4f}  F {st2['rank_pool']['F']['recall@1']:.4f}")

    with open(out / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log(f"wrote {out / 'report.json'}")


if __name__ == "__main__":
    main()
