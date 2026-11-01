#!/usr/bin/env python3
"""
Jina v4 text-prefix check: how far do the caption vectors, and the exact
neighbour lists built on them, move if captions are embedded WITHOUT the
"Query: " prefix?

WHY
    gen_jinav4_text.py passes no prompt_name and its docstring calls that "no
    prefix". The model code does not agree: modeling_jina_embeddings_v4.py:442
    sets prompt_name = prompt_name or "query", and process_texts then writes
    f"{prefix}: {text}". So every stored caption vector is the embedding of
    "Query: <caption>". This job measures what dropping the prefix would change
    BEFORE anyone decides to re-embed 8M captions and rerun T2 onward.

WHAT IT DOES
    sample   Seeded random WHOLE shards from the 7M manifest, added until the
             sample holds >= --min-b B rows. Only A and B rows of the 7M split
             are kept; C stays untouched.
    encode   One process per GPU, over shards part::parts. Each caption is
             encoded twice through the SAME JinaV4TextEmbedder the T1 run used
             (task retrieval, batch 64, max_text_len 512, bf16):
               P  the default path -> "Query: <caption>"
               N  process_texts patched to drop the prefix -> "<caption>"
             A guard proves the patch works before any caption is encoded:
             N("Query: x") must equal P("x").
    analyze  Compares three versions of every sampled caption vector:
               S  the stored T1 vector
               P  re-embedded with the prefix  (S vs P = rerun noise floor)
               N  re-embedded without it       (S vs N = the prefix effect)
             - per-caption cosine S.P, S.N, P.N
             - exact top-k neighbour overlap at k = 10 and 100, B queries
               against the sample's A rows, for t2t, t2i and i2t (i2i does
               not involve a caption vector)
             - Spearman, across queries, of the 10th and 100th neighbour
               similarity (the local-density signal the fidelity map rides on)
             - paired retrieval inside B: caption -> own image and back,
               recall@1 / @10

    Search is exact fp32 matmul with TF32 OFF, so the noise floor measures
    the embedding rerun, not the search.

OUTPUT
    result/audit/jina_prefix_check/
        sample.json                      shards and rows that were sampled
        vectors/part<i>_{P,N}.npy        float32 [n_i, 2048], L2-normalised
        vectors/part<i>.json             pair ids, roles, guard, timing
        report.json                      every number analyze computes

RUN THIS IN A PBS JOB (audit/jina_prefix_check.pbs), not on the login node.
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
from collections import defaultdict

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT.parent / "laion/worker1_20/tar1"
MANIFEST = REPO_ROOT / "result/corpus_shard_manifest.json"
PAIR_INDEX = REPO_ROOT / "result/7M_cleaned/siglip2/pair_index.csv"
TEXT_DIR = REPO_ROOT / "result/embeddings_text/jina_embeddings_v4"
IMAGE_DIR = REPO_ROOT / "result/embeddings_image/jina_embeddings_v4"
DEFAULT_OUT = REPO_ROOT / "result/audit/jina_prefix_check"

BATCH = 64            # T1's DataLoader batch, which _encode_batch received
KS = (10, 100)
CHUNK = 2048          # query rows per GPU matmul


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def stem_of(name: str) -> str:
    """worker_0/shard_00000/000000001(.json|.jpg) -> 000000001"""
    return Path(name).name.split(".")[0]


def caption_text(raw: bytes) -> str:
    """Exactly the string T1 encoded (gen_jinav4_text.MultiShardDataset)."""
    try:
        meta = json.loads(raw.decode("utf-8"))
        caption = meta.get("caption") if isinstance(meta, dict) else None
    except Exception:
        caption = None
    return caption if isinstance(caption, str) else ""


# ---------------------------------------------------------------- sample

def cmd_sample(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    shards = list(json.load(open(MANIFEST, encoding="utf-8"))["included_shards"])
    order = shards[:]
    random.Random(args.seed).shuffle(order)
    cand = order[:args.max_candidates]
    cand_set = set(cand)

    rows = defaultdict(list)          # shard -> [(pair_id, int_id, role)]
    t0 = time.time()
    with open(PAIR_INDEX, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        if header[:2] != ["int_id", "pair_id"] or header[-1] != "role":
            raise RuntimeError(f"unexpected pair_index header {header}")
        for row in r:
            role = row[-1]
            if role not in ("A", "B"):
                continue
            shard = row[1].rsplit("/", 1)[0]
            if shard in cand_set:
                rows[shard].append((row[1], int(row[0]), role))
    log(f"pair_index scanned in {time.time() - t0:.1f}s")

    picked, n_a, n_b = [], 0, 0
    for s in cand:
        picked.append(s)
        n_a += sum(1 for _, _, ro in rows[s] if ro == "A")
        n_b += sum(1 for _, _, ro in rows[s] if ro == "B")
        if n_b >= args.min_b:
            break
    else:
        raise RuntimeError(f"{len(cand)} candidate shards give only {n_b} B rows")

    sample = {
        "created": datetime.now().isoformat(),
        "seed": args.seed,
        "min_b": args.min_b,
        "scope": "7M_cleaned/siglip2",
        "pair_index": str(PAIR_INDEX),
        "n_shards": len(picked),
        "n_A": n_a,
        "n_B": n_b,
        "shards": picked,
        # 9-digit zero-padded names, so string order == T1's filename order
        "rows": {s: sorted(rows[s]) for s in picked},
    }
    with open(out / "sample.json", "w", encoding="utf-8") as f:
        json.dump(sample, f)
    log(f"sample: {len(picked)} shards, A={n_a:,}  B={n_b:,}")


# ---------------------------------------------------------------- encode

def cmd_encode(args) -> None:
    import torch
    sys.path.insert(0, str(REPO_ROOT / "embeddingsgenerator"))
    import gen_jinav4_text as T1

    torch.set_num_threads(max(1, args.torch_threads))
    out = Path(args.out)
    vec_dir = out / "vectors"
    vec_dir.mkdir(parents=True, exist_ok=True)
    sample = json.load(open(out / "sample.json", encoding="utf-8"))
    my_shards = sample["shards"][args.part::args.parts]

    emb = T1.JinaV4TextEmbedder(
        args.device, BATCH, T1.DEFAULT_MODEL_DIR, num_workers=0,
        task=T1.DEFAULT_TASK, truncate_dim=None, prompt_name=None,
        max_text_len=T1.DEFAULT_MAX_TEXT_LEN,
    )

    # encode_text builds partial(self.processor.process_texts, prefix=...) on
    # every call, so patching the processor CLASS reaches it whatever wrapper
    # (PeftModel) sits in front. With drop=False the call is unchanged.
    proc = emb.model.processor
    cls = type(proc)
    orig = cls.process_texts
    state = {"drop": False}

    def process_texts(self, texts, max_length=None, prefix=None, padding=None):
        return orig(self, texts, max_length=max_length,
                    prefix=None if state["drop"] else prefix, padding=padding)

    cls.process_texts = process_texts

    def encode(texts, drop):
        state["drop"] = drop
        try:
            with torch.inference_mode():
                return emb._encode_batch(texts).numpy().astype(np.float32)
        finally:
            state["drop"] = False

    # Guard: the default path must render "Query: x", the patched one "x",
    # and N("Query: x") must be P("x") - i.e. the default adds exactly that.
    probe = "a red car parked on a street"

    def rendered(drop):
        state["drop"] = drop
        try:
            enc = proc.process_texts([probe], max_length=512, prefix="Query")
            return proc.tokenizer.decode(enc["input_ids"][0])
        finally:
            state["drop"] = False

    r_p, r_n = rendered(False), rendered(True)
    v_p = encode([probe], False)[0]
    v_manual = encode([f"Query: {probe}"], True)[0]
    v_n = encode([probe], True)[0]
    guard = {
        "rendered_default": r_p,
        "rendered_patched": r_n,
        "cos_default_vs_manual_query_prefix": float(v_p @ v_manual),
        "cos_default_vs_no_prefix": float(v_p @ v_n),
    }
    log(f"guard: {guard}")
    if not r_p.startswith("Query: ") or "Query" in r_n:
        raise RuntimeError(f"prefix patch did not take: {guard}")
    if guard["cos_default_vs_manual_query_prefix"] < 0.999:
        raise RuntimeError(f"default path is not 'Query: ' + text: {guard}")

    names, roles, P, N = [], [], [], []
    t0 = time.time()
    done = 0
    truncated = False
    for s in my_shards:
        if args.stop_at and time.time() > args.stop_at:
            truncated = True
            log(f"stop-at reached, {done}/{len(my_shards)} shards encoded")
            break
        worker, shard = s.split("/")
        want = {stem_of(pid): (pid, role) for pid, _, role in sample["rows"][s]}
        caps = {}
        with tarfile.open(DATA_ROOT / worker / f"{shard}.tar", "r") as tf:
            for m in tf:
                if not m.isfile():
                    continue
                p = Path(m.name)
                if p.suffix.lower() != ".json" or p.stem not in want:
                    continue
                fh = tf.extractfile(m)
                caps[p.stem] = caption_text(fh.read() if fh else b"")
        missing = sorted(set(want) - set(caps))
        if missing:
            raise RuntimeError(f"{s}: {len(missing)} captions not in tar, e.g. {missing[:3]}")

        stems = sorted(want)
        texts = [caps[x] for x in stems]
        for i in range(0, len(texts), BATCH):
            chunk = texts[i:i + BATCH]
            P.append(encode(chunk, False))
            N.append(encode(chunk, True))
        names += [want[x][0] for x in stems]
        roles += [want[x][1] for x in stems]
        done += 1
        if done % 4 == 0 or done == len(my_shards):
            el = time.time() - t0
            log(f"[{done}/{len(my_shards)}] {len(names):,} captions x2 "
                f"in {el:.0f}s ({2 * len(names) / el:.0f} enc/s)")

    P = np.concatenate(P) if P else np.zeros((0, emb.dim), np.float32)
    N = np.concatenate(N) if N else np.zeros((0, emb.dim), np.float32)
    np.save(vec_dir / f"part{args.part}_P.npy", P)
    np.save(vec_dir / f"part{args.part}_N.npy", N)
    meta = {
        "part": args.part, "parts": args.parts, "device": args.device,
        "shards_assigned": len(my_shards), "shards_encoded": done,
        "truncated_by_stop_at": truncated, "n": len(names),
        "seconds": time.time() - t0, "guard": guard,
        "names": names, "roles": roles,
    }
    with open(vec_dir / f"part{args.part}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)
    log(f"part {args.part} done: {len(names):,} captions, truncated={truncated}")


# ---------------------------------------------------------------- analyze

def summ(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    q = np.percentile(x, [0.1, 1, 5, 50])
    return {"mean": float(x.mean()), "min": float(x.min()),
            "p0.1": float(q[0]), "p1": float(q[1]), "p5": float(q[2]),
            "p50": float(q[3]), "max": float(x.max())}


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.empty(len(x)); rx[np.argsort(x, kind="stable")] = np.arange(len(x))
    ry = np.empty(len(y)); ry[np.argsort(y, kind="stable")] = np.arange(len(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def cmd_analyze(args) -> None:
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    dev = torch.device(args.device)

    out = Path(args.out)
    vec_dir = out / "vectors"
    sample = json.load(open(out / "sample.json", encoding="utf-8"))

    names, roles, P, N, parts = [], [], [], [], []
    for i in range(args.parts):
        meta = json.load(open(vec_dir / f"part{i}.json", encoding="utf-8"))
        names += meta["names"]
        roles += meta["roles"]
        P.append(np.load(vec_dir / f"part{i}_P.npy"))
        N.append(np.load(vec_dir / f"part{i}_N.npy"))
        parts.append({k: v for k, v in meta.items() if k not in ("names", "roles")})
    P, N = np.concatenate(P), np.concatenate(N)
    roles = np.array(roles)
    n = len(names)
    if P.shape[0] != n or N.shape[0] != n or len(set(names)) != n:
        raise RuntimeError(f"part files disagree: names {n}, P {P.shape}, N {N.shape}")
    log(f"loaded {n:,} re-embedded captions from {args.parts} parts")

    # Stored T1 vectors, joined by pair id (never by position).
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

    report = {
        "created": datetime.now().isoformat(),
        "sample": {k: sample[k] for k in ("seed", "min_b", "scope", "n_shards", "n_A", "n_B")},
        "encoded": {"n": n, "n_A": int((roles == "A").sum()), "n_B": int((roles == "B").sum()),
                    "parts": parts},
        "versions": {"S": "stored T1 vector ('Query: ' prefix, per the model code)",
                     "P": "re-embedded now, default path ('Query: <caption>')",
                     "N": "re-embedded now, prefix removed ('<caption>')"},
        "cosine_per_caption": {
            "S_vs_P": summ((S_t * P).sum(1)),
            "S_vs_N": summ((S_t * N).sum(1)),
            "P_vs_N": summ((P * N).sum(1)),
        },
    }
    log(f"cos S.P mean {report['cosine_per_caption']['S_vs_P']['mean']:.6f}   "
        f"cos S.N mean {report['cosine_per_caption']['S_vs_N']['mean']:.6f}")

    A = np.flatnonzero(roles == "A")
    B = np.flatnonzero(roles == "B")
    text = {"S": S_t, "P": P, "N": N}
    kmax = max(KS)

    def topk(Q: np.ndarray, X: np.ndarray):
        Xg = torch.from_numpy(np.ascontiguousarray(X)).to(dev)
        D, I = [], []
        for i in range(0, len(Q), CHUNK):
            q = torch.from_numpy(np.ascontiguousarray(Q[i:i + CHUNK])).to(dev)
            d, ix = torch.topk(q @ Xg.T, kmax, dim=1)
            D.append(d.cpu()); I.append(ix.cpu())
        del Xg
        return torch.cat(D).numpy(), torch.cat(I).numpy()

    def overlap(Ia: np.ndarray, Ib: np.ndarray, k: int) -> np.ndarray:
        a = torch.from_numpy(Ia[:, :k]).to(dev)
        b = torch.from_numpy(Ib[:, :k]).to(dev)
        res = []
        for i in range(0, len(a), 4096):
            hit = (a[i:i + 4096, :, None] == b[i:i + 4096, None, :]).any(2).sum(1)
            res.append(hit.cpu())
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
        for v in ("S", "P", "N"):
            D[v], I[v] = topk(qf(v), xf(v))
        block = {}
        for a, b in (("S", "P"), ("S", "N"), ("P", "N")):
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
            f"S.N {block['S_vs_N']['overlap@100']['mean']:.4f}  "
            f"P.N {block['P_vs_N']['overlap@100']['mean']:.4f}")

    # Paired retrieval inside B: caption i's own image is image i.
    def paired(T: np.ndarray, Im: np.ndarray) -> dict:
        Tg = torch.from_numpy(np.ascontiguousarray(T)).to(dev)
        Ig = torch.from_numpy(np.ascontiguousarray(Im)).to(dev)
        out = {}
        for name, Qg, Xg in (("t2i", Tg, Ig), ("i2t", Ig, Tg)):
            ranks = []
            for i in range(0, len(Qg), CHUNK):
                s = Qg[i:i + CHUNK] @ Xg.T
                own = s[torch.arange(s.shape[0], device=dev),
                        torch.arange(i, i + s.shape[0], device=dev)]
                ranks.append((s > own[:, None]).sum(1).cpu())
            r = torch.cat(ranks).numpy()
            out[name] = {"recall@1": float((r < 1).mean()),
                         "recall@10": float((r < 10).mean())}
        return out

    report["paired_retrieval_B"] = {"n": int(len(B))}
    for v in ("S", "P", "N"):
        report["paired_retrieval_B"][v] = paired(text[v][B], S_i[B])
    log(f"paired B recall@1 t2i  S {report['paired_retrieval_B']['S']['t2i']['recall@1']:.4f}  "
        f"N {report['paired_retrieval_B']['N']['t2i']['recall@1']:.4f}")

    with open(out / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log(f"wrote {out / 'report.json'}")


def main():
    ap = argparse.ArgumentParser(description="Jina v4 'Query: ' prefix check.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sample")
    s.add_argument("--out", default=str(DEFAULT_OUT))
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--min-b", type=int, default=20000)
    s.add_argument("--max-candidates", type=int, default=400)

    e = sub.add_parser("encode")
    e.add_argument("--out", default=str(DEFAULT_OUT))
    e.add_argument("--part", type=int, required=True)
    e.add_argument("--parts", type=int, required=True)
    e.add_argument("--device", default="cuda:0")
    e.add_argument("--torch-threads", type=int, default=4)
    e.add_argument("--stop-at", type=float, default=0.0,
                   help="Epoch seconds; take no new shard after this.")

    a = sub.add_parser("analyze")
    a.add_argument("--out", default=str(DEFAULT_OUT))
    a.add_argument("--parts", type=int, required=True)
    a.add_argument("--device", default="cuda:0")

    args = ap.parse_args()
    {"sample": cmd_sample, "encode": cmd_encode, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
