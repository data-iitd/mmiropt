"""
Shared pieces of the embedding input checks in audit/.

jina_prefix_check.py and siglip2_lowercase_check.py carry their own copies of
this code (their reports were produced by those copies, so they stay as they
are); the later checks import it from here instead.

Everything is exact fp32 matmul with TF32 OFF, so a noise floor measured with
these helpers is the embedding rerun, not the search.
"""

import json
from pathlib import Path
from collections import defaultdict

import numpy as np

from jina_prefix_check import summ, spearman, stem_of

KS = (10, 100)
CHUNK = 4096


def exact_fp32(torch) -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def join_stored(names, roots, dim):
    """
    Stored T1 vectors for `names` (pair ids), joined by pair id, never by
    position. Returns one float32 [n, dim] array per root, in `roots` order.
    """
    by_shard = defaultdict(list)
    for i, pid in enumerate(names):
        by_shard[pid.rsplit("/", 1)[0]].append(i)
    outs = [np.empty((len(names), dim), np.float32) for _ in roots]
    for s, idxs in by_shard.items():
        worker, shard = s.split("/")
        for root, dst in zip(roots, outs):
            nm = json.load(open(Path(root) / worker / f"{shard}.names.json", encoding="utf-8"))
            mat = np.load(Path(root) / worker / f"{shard}.npy", mmap_mode="r")
            if mat.shape[0] != len(nm):
                raise RuntimeError(f"{Path(root).name}/{s}: {mat.shape[0]} rows vs {len(nm)} names")
            pos = {stem_of(x): j for j, x in enumerate(nm)}
            dst[idxs] = mat[[pos[stem_of(names[i])] for i in idxs]]
    return outs


def cosine_block(versions: dict, pairs) -> dict:
    return {f"{a}_vs_{b}": summ((versions[a] * versions[b]).sum(1)) for a, b in pairs}


def topk(torch, dev, Q, X, k):
    Xg = torch.from_numpy(np.ascontiguousarray(X)).to(dev)
    D, I = [], []
    for i in range(0, len(Q), CHUNK):
        q = torch.from_numpy(np.ascontiguousarray(Q[i:i + CHUNK])).to(dev)
        d, ix = torch.topk(q @ Xg.T, k, dim=1)
        D.append(d.cpu()); I.append(ix.cpu())
    del Xg
    return torch.cat(D).numpy(), torch.cat(I).numpy()


def overlap(torch, dev, Ia, Ib, k):
    a = torch.from_numpy(Ia[:, :k]).to(dev)
    b = torch.from_numpy(Ib[:, :k]).to(dev)
    res = []
    for i in range(0, len(a), 4096):
        res.append((a[i:i + 4096, :, None] == b[i:i + 4096, None, :]).any(2).sum(1).cpu())
    return torch.cat(res).numpy() / k


def knn_block(torch, dev, qf, xf, versions, pairs) -> dict:
    """
    qf(v) / xf(v) give the query and corpus matrices for version v. Every
    version's exact top-100 is compared pairwise: overlap at k = 10 and 100,
    top-1 agreement, and Spearman of the 10th / 100th neighbour similarity.
    """
    kmax = max(KS)
    D, I = {}, {}
    for v in versions:
        D[v], I[v] = topk(torch, dev, qf(v), xf(v), kmax)
    block = {}
    for a, b in pairs:
        pair = {}
        for k in KS:
            ov = overlap(torch, dev, I[a], I[b], k)
            pair[f"overlap@{k}"] = summ(ov)
            pair[f"frac_overlap@{k}_eq_1"] = float((ov == 1.0).mean())
            pair[f"frac_overlap@{k}_lt_0.9"] = float((ov < 0.9).mean())
        pair["top1_same"] = float((I[a][:, 0] == I[b][:, 0]).mean())
        pair["spearman_sim@10"] = spearman(D[a][:, 9], D[b][:, 9])
        pair["spearman_sim@100"] = spearman(D[a][:, kmax - 1], D[b][:, kmax - 1])
        block[f"{a}_vs_{b}"] = pair
    block["mean_sim@100"] = {v: float(D[v][:, kmax - 1].mean()) for v in D}
    return block


def paired(torch, dev, T, Im) -> dict:
    """Row i of T and row i of Im are one pair: rank of the own partner."""
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
