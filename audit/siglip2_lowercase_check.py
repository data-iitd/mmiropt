#!/usr/bin/env python3
"""
SigLIP 2 lowercase check: how far do the caption vectors, their exact
neighbour lists and their text strata move if captions are lowercased before
tokenising, as SigLIP 2 was trained?

WHY
    The checkpoint's tokenizer_config.json says "do_lower_case": true but names
    "tokenizer_class": "GemmaTokenizer". AutoTokenizer therefore loads the Gemma
    tokenizer, whose tokenizer.json normalizer only replaces spaces - the flag
    is never applied. transformers' own Siglip2Tokenizer prepends
    normalizers.Lowercase() ("SigLIP2 training default: lowercase
    normalization"), but nothing selected it. So every stored SigLIP 2 caption
    vector was embedded from cased text. SigLIP 2 text vectors also define the
    25 text strata behind the A/B/C split, so a change here reaches the split.

WHAT IT DOES (one process, one GPU, nothing but the report written)
    Reuses the Jina check's sample (result/audit/jina_prefix_check/sample.json:
    84 random 7M shards, A and B rows only) so the two checks are comparable.
    Every caption is tokenised exactly as T1 did (AutoTokenizer, fixed padding
    to 64, truncation) and encoded through the SAME Siglip2TextEmbedder (fp16
    autocast) twice:
        P  as-is            (S vs P = rerun noise floor)
        L  text.lower()     (S vs L = the lowercase effect)
    and compares them with S, the stored T1 vector:
        - per-caption cosine
        - exact top-k neighbour overlap, B queries vs the sample's A rows,
          for t2t, t2i, i2t (i2i does not involve a caption vector)
        - Spearman of the 10th / 100th neighbour similarity across queries
        - paired retrieval inside B (caption <-> own image) recall@1/10
        - text stratum: nearest of the 25 stored centroids (what
          MiniBatchKMeans.predict does), S vs P vs L, and S vs the label the
          split actually used

    Guards: batched tokenising must equal T1's per-caption call, and
    str.lower() + Gemma is compared token-for-token with the installed
    Siglip2Tokenizer, so L is shown to be what that tokenizer produces.

OUTPUT
    result/audit/siglip2_lowercase_check/report.json

RUN THIS IN A PBS JOB (audit/siglip2_lowercase_check.pbs), not on the login node.
"""

import sys
import csv
import json
import time
import tarfile
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from jina_prefix_check import (log, summ, spearman, caption_text, stem_of,
                               DATA_ROOT, PAIR_INDEX)

REPO_ROOT = Path(__file__).resolve().parent.parent
TEXT_DIR = REPO_ROOT / "result/embeddings_text/siglip2_so400m_patch16_512"
IMAGE_DIR = REPO_ROOT / "result/embeddings_image/siglip2_so400m_patch16_512"
CENTROIDS = REPO_ROOT / "result/stratification/7M_cleaned/siglip2/text_stratum_centroids.npy"
SAMPLE = REPO_ROOT / "result/audit/jina_prefix_check/sample.json"
DEFAULT_OUT = REPO_ROOT / "result/audit/siglip2_lowercase_check"

BATCH = 512           # T1's batch (siglip2_txt.log: "Batch size : 512", AMP True)
KS = (10, 100)
CHUNK = 4096


def read_shard_captions(shard: str, rows):
    worker, name = shard.split("/")
    want = {stem_of(pid) for pid, _, _ in rows}
    caps = {}
    with tarfile.open(DATA_ROOT / worker / f"{name}.tar", "r") as tf:
        for m in tf:
            if not m.isfile():
                continue
            p = Path(m.name)
            if p.suffix.lower() != ".json" or p.stem not in want:
                continue
            fh = tf.extractfile(m)
            caps[p.stem] = caption_text(fh.read() if fh else b"")
    missing = want - set(caps)
    if missing:
        raise RuntimeError(f"{shard}: {len(missing)} captions not in tar")
    return [caps[stem_of(pid)] for pid, _, _ in rows]


def main():
    ap = argparse.ArgumentParser(description="SigLIP 2 lowercase check.")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--sample", default=str(SAMPLE))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    import torch
    sys.path.insert(0, str(REPO_ROOT / "embeddingsgenerator"))
    import gen_siglip2_text as T1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sample = json.load(open(args.sample, encoding="utf-8"))
    shards = sample["shards"]

    # ---- captions, in the sample's (pair-id sorted) order
    t0 = time.time()
    with ThreadPoolExecutor(8) as ex:
        per_shard = list(ex.map(lambda s: read_shard_captions(s, sample["rows"][s]), shards))
    names, roles, texts = [], [], []
    for s, caps in zip(shards, per_shard):
        for (pid, _, role), c in zip(sample["rows"][s], caps):
            names.append(pid); roles.append(role); texts.append(c)
    roles = np.array(roles)
    n = len(names)
    log(f"{n:,} captions read from {len(shards)} shards in {time.time() - t0:.0f}s")

    # ---- stored text-stratum labels, for the replication check
    want = set(names)
    stored_label = {}
    with open(PAIR_INDEX, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        col = header.index("text_stratum")
        for row in r:
            if row[1] in want:
                stored_label[row[1]] = int(row[col])
    stored_label = np.array([stored_label[x] for x in names])

    # ---- model and tokenisers
    emb = T1.Siglip2TextEmbedder(args.device, BATCH, True, T1.DEFAULT_MODEL_DIR, num_workers=0)
    tok, L_ctx = emb.tokenizer, emb.max_text_len

    def tokenize(t, batch):
        return t(batch, padding="max_length", truncation=True,
                 max_length=L_ctx, return_tensors="pt")["input_ids"]

    ids_P = tokenize(tok, texts)
    lowered = [t.lower() for t in texts]
    ids_L = tokenize(tok, lowered)

    guard = {"tokenizer_class_used_by_T1": type(tok).__name__,
             "context": L_ctx,
             "normalizer_used_by_T1": str(tok.backend_tokenizer.normalizer)
             if hasattr(tok, "backend_tokenizer") else None}
    m = min(2000, n)
    per_call = torch.stack([emb.transform(t) for t in texts[:m]])
    guard["batched_equals_T1_per_caption"] = bool(torch.equal(per_call, ids_P[:m]))
    if not guard["batched_equals_T1_per_caption"]:
        raise RuntimeError(f"batched tokenising differs from T1: {guard}")
    guard["frac_captions_with_uppercase"] = float(np.mean([t != t.lower() for t in texts]))
    guard["frac_token_ids_changed_by_lowercase"] = float((ids_P != ids_L).any(1).float().mean())
    try:
        from transformers import Siglip2Tokenizer
        tok2 = Siglip2Tokenizer.from_pretrained(str(T1.DEFAULT_MODEL_DIR))
        ids_O = tokenize(tok2, texts)
        guard["siglip2tokenizer_padding_side"] = tok2.padding_side
        guard["siglip2tokenizer_normalizer"] = str(tok2.backend_tokenizer.normalizer)
        guard["frac_lower_plus_gemma_equals_siglip2tokenizer"] = float(
            (ids_O == ids_L).all(1).float().mean())
    except Exception as e:  # recorded, not fatal: L stands on str.lower()
        guard["siglip2tokenizer_error"] = repr(e)
    log(f"guard: {guard}")

    # ---- encode, exactly T1's path (fp16 autocast, no attention mask)
    def encode(ids):
        res = []
        with torch.inference_mode():
            for i in range(0, len(ids), BATCH):
                res.append(emb._encode_batch(ids[i:i + BATCH]).numpy())
        return np.concatenate(res).astype(np.float32)

    t0 = time.time()
    P = encode(ids_P)
    L = encode(ids_L)
    log(f"encoded {n:,} x2 in {time.time() - t0:.0f}s")
    del emb
    torch.cuda.empty_cache()

    # ---- stored T1 vectors, joined by pair id
    by_shard = defaultdict(list)
    for i, pid in enumerate(names):
        by_shard[pid.rsplit("/", 1)[0]].append(i)
    S_t = np.empty_like(P)
    S_i = np.empty_like(P)
    for s, idxs in by_shard.items():
        worker, shard = s.split("/")
        for root, dst in ((TEXT_DIR, S_t), (IMAGE_DIR, S_i)):
            nm = json.load(open(root / worker / f"{shard}.names.json", encoding="utf-8"))
            mat = np.load(root / worker / f"{shard}.npy", mmap_mode="r")
            if mat.shape[0] != len(nm):
                raise RuntimeError(f"{root.name}/{s}: {mat.shape[0]} rows vs {len(nm)} names")
            pos = {stem_of(x): j for j, x in enumerate(nm)}
            dst[idxs] = mat[[pos[stem_of(names[i])] for i in idxs]]
    log("stored text and image vectors joined by pair id")

    # ---- analysis (exact fp32, TF32 off)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    dev = torch.device(args.device)
    text = {"S": S_t, "P": P, "L": L}

    report = {
        "created": datetime.now().isoformat(),
        "sample": {"source": str(args.sample),
                   **{k: sample[k] for k in ("seed", "scope", "n_shards", "n_A", "n_B")}},
        "n": n, "n_A": int((roles == "A").sum()), "n_B": int((roles == "B").sum()),
        "versions": {"S": "stored T1 vector (cased text, GemmaTokenizer)",
                     "P": "re-embedded now, cased text (T1 path)",
                     "L": "re-embedded now, text.lower() then the same tokenizer"},
        "guard": guard,
        "cosine_per_caption": {"S_vs_P": summ((S_t * P).sum(1)),
                               "S_vs_L": summ((S_t * L).sum(1)),
                               "P_vs_L": summ((P * L).sum(1))},
    }
    changed = (ids_P != ids_L).any(1).numpy()
    report["cosine_per_caption"]["S_vs_L_only_changed_captions"] = summ((S_t * L).sum(1)[changed])
    log(f"cos S.P mean {report['cosine_per_caption']['S_vs_P']['mean']:.6f}   "
        f"cos S.L mean {report['cosine_per_caption']['S_vs_L']['mean']:.6f}")

    # Text strata: nearest centroid by Euclidean distance, as km.predict does.
    C = np.load(CENTROIDS).astype(np.float32)
    def label(X):
        Cg = torch.from_numpy(C).to(dev)
        c2 = (Cg * Cg).sum(1)
        out_l = []
        for i in range(0, len(X), CHUNK):
            x = torch.from_numpy(np.ascontiguousarray(X[i:i + CHUNK])).to(dev)
            out_l.append(torch.argmin(c2[None, :] - 2 * x @ Cg.T, dim=1).cpu())
        return torch.cat(out_l).numpy()
    lab = {v: label(text[v]) for v in text}
    report["text_stratum"] = {
        "k": int(C.shape[0]),
        "S_matches_split_label": float((lab["S"] == stored_label).mean()),
        "S_vs_P_same": float((lab["S"] == lab["P"]).mean()),
        "S_vs_L_same": float((lab["S"] == lab["L"]).mean()),
        "S_vs_L_same_only_changed_captions": float((lab["S"] == lab["L"])[changed].mean()),
    }
    log(f"text stratum: S=split {report['text_stratum']['S_matches_split_label']:.4f}  "
        f"S=P {report['text_stratum']['S_vs_P_same']:.4f}  "
        f"S=L {report['text_stratum']['S_vs_L_same']:.4f}")

    A = np.flatnonzero(roles == "A")
    B = np.flatnonzero(roles == "B")
    kmax = max(KS)

    def topk(Q, X):
        Xg = torch.from_numpy(np.ascontiguousarray(X)).to(dev)
        D, I = [], []
        for i in range(0, len(Q), CHUNK):
            q = torch.from_numpy(np.ascontiguousarray(Q[i:i + CHUNK])).to(dev)
            d, ix = torch.topk(q @ Xg.T, kmax, dim=1)
            D.append(d.cpu()); I.append(ix.cpu())
        del Xg
        return torch.cat(D).numpy(), torch.cat(I).numpy()

    def overlap(Ia, Ib, k):
        a = torch.from_numpy(Ia[:, :k]).to(dev)
        b = torch.from_numpy(Ib[:, :k]).to(dev)
        res = []
        for i in range(0, len(a), 4096):
            res.append((a[i:i + 4096, :, None] == b[i:i + 4096, None, :]).any(2).sum(1).cpu())
        return torch.cat(res).numpy() / k

    directions = {
        "t2t": (lambda v: text[v][B], lambda v: text[v][A]),
        "t2i": (lambda v: text[v][B], lambda v: S_i[A]),
        "i2t": (lambda v: S_i[B],     lambda v: text[v][A]),
    }
    report["knn"] = {"queries": "B", "corpus": "A rows of the sample",
                     "n_queries": int(len(B)), "n_corpus": int(len(A))}
    for dname, (qf, xf) in directions.items():
        D, I = {}, {}
        for v in ("S", "P", "L"):
            D[v], I[v] = topk(qf(v), xf(v))
        block = {}
        for a, b in (("S", "P"), ("S", "L"), ("P", "L")):
            pair = {}
            for k in KS:
                ov = overlap(I[a], I[b], k)
                pair[f"overlap@{k}"] = summ(ov)
                pair[f"frac_overlap@{k}_eq_1"] = float((ov == 1.0).mean())
                pair[f"frac_overlap@{k}_lt_0.9"] = float((ov < 0.9).mean())
            pair["top1_same"] = float((I[a][:, 0] == I[b][:, 0]).mean())
            pair["spearman_sim@10"] = spearman(D[a][:, 9], D[b][:, 9])
            pair["spearman_sim@100"] = spearman(D[a][:, kmax - 1], D[b][:, kmax - 1])
            block[f"{a}_vs_{b}"] = pair
        block["mean_sim@100"] = {v: float(D[v][:, kmax - 1].mean()) for v in D}
        report["knn"][dname] = block
        log(f"{dname}: overlap@100 mean  S.P {block['S_vs_P']['overlap@100']['mean']:.4f}  "
            f"S.L {block['S_vs_L']['overlap@100']['mean']:.4f}")

    def paired(T, Im):
        Tg = torch.from_numpy(np.ascontiguousarray(T)).to(dev)
        Ig = torch.from_numpy(np.ascontiguousarray(Im)).to(dev)
        res = {}
        for name, Qg, Xg in (("t2i", Tg, Ig), ("i2t", Ig, Tg)):
            ranks = []
            for i in range(0, len(Qg), CHUNK):
                s = Qg[i:i + CHUNK] @ Xg.T
                own = s[torch.arange(s.shape[0], device=dev),
                        torch.arange(i, i + s.shape[0], device=dev)]
                ranks.append((s > own[:, None]).sum(1).cpu())
            r = torch.cat(ranks).numpy()
            res[name] = {"recall@1": float((r < 1).mean()), "recall@10": float((r < 10).mean())}
        return res

    report["paired_retrieval_B"] = {"n": int(len(B))}
    for v in ("S", "P", "L"):
        report["paired_retrieval_B"][v] = paired(text[v][B], S_i[B])
    log(f"paired B recall@1 t2i  S {report['paired_retrieval_B']['S']['t2i']['recall@1']:.4f}  "
        f"L {report['paired_retrieval_B']['L']['t2i']['recall@1']:.4f}")

    with open(out / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log(f"wrote {out / 'report.json'}")


if __name__ == "__main__":
    main()
