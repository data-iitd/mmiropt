"""
The DO baseline's density score, as the protocol defines it (spatial-map-protocol.tex
l.108): for every landmark, the mean distance to its k nearest CORPUS points.

    python landmarks/landmark_density.py --model siglip2 --modality image \
        --k-list 500 1250 2500 5000 10000 20000 25000 40000 --nn 100   (inside a PBS job)

Label-free: reads only the corpus (role A) vectors and the landmarks. Exact k-NN of
the K landmarks over all of A on the GPU (FAISS flat L2), A streamed in chunks of
CHUNK vectors with a running top-k merge, so it never needs A resident on the GPU.

k = 100 (user, 2026-10-19): CF@100 asks whether an index finds a query's 100 nearest
neighbours, so the density that matters is the one at the scale of 100 neighbours.

Output, next to the tiling: <kmeans>/<model>/<modality>/K<K>_spherical/density_A_k<nn>.npy
(float64 [K], mean ORDINARY L2 distance -- the root of FAISS's squared distance)
+ density_A_k<nn>.json (provenance).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import faiss

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "indexes"))
sys.path.insert(0, str(REPO_ROOT / "landmarks"))

from index_configs import MODEL_DIMS                                       # noqa: E402
from load_vectors import load_included_shards, load_pair_index, load_role_matrix  # noqa: E402
from v2_dirs import MANIFEST, SPLIT_DIR, SPLIT_SCOPE, V2, embed_dir         # noqa: E402
from build_landmarks import out_dir                                        # noqa: E402

CHUNK = 1_000_000


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def knn_mean_dist(A: np.ndarray, L: np.ndarray, nn: int, res) -> np.ndarray:
    """Exact: mean L2 distance from each row of L to its nn nearest rows of A."""
    best = np.full((L.shape[0], nn), np.inf, dtype=np.float32)
    for s in range(0, A.shape[0], CHUNK):
        idx = faiss.GpuIndexFlatL2(res, A.shape[1])
        idx.add(np.ascontiguousarray(A[s:s + CHUNK]))
        D, _ = idx.search(L, nn)
        both = np.concatenate([best, D], axis=1)
        best = np.partition(both, nn - 1, axis=1)[:, :nn]
        del idx
    return np.sqrt(np.maximum(best.astype(np.float64), 0.0)).mean(axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=("siglip2", "openclip", "qwen3vl", "jinav4"))
    ap.add_argument("--modality", required=True, choices=("image", "text"))
    ap.add_argument("--k-list", type=int, nargs="+", required=True)
    ap.add_argument("--nn", type=int, default=100)
    ap.add_argument("--kmeans-dir", type=Path, default=V2 / "kmeans" / "7M_cleaned")
    ap.add_argument("--limit-a", type=int, default=0, help="SMOKE TEST: first N of A only")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if not os.environ.get("PBS_JOBID"):
        raise SystemExit("Refusing to run outside a PBS job (reads all of A, uses the GPU)")

    todo = []
    for k in args.k_list:
        dest = out_dir(args.kmeans_dir, args.model, args.modality, k, True)
        f = dest / f"density_A_k{args.nn}.npy"
        if f.exists() and not args.overwrite and not args.limit_a:
            log(f"K={k}: {f.name} exists, skipping")
            continue
        if not (dest / "centroids.npy").exists():
            raise SystemExit(f"missing {dest / 'centroids.npy'}")
        todo.append((k, dest))
    if not todo:
        return

    t0 = time.time()
    scope = json.load(open(SPLIT_DIR / "pair_index_meta.json"))["scope"]
    if scope != SPLIT_SCOPE:
        raise RuntimeError(f"{SPLIT_DIR}: scope {scope!r}, expected {SPLIT_SCOPE!r}")
    a_ids = np.load(SPLIT_DIR / "A_ids.npy").astype(np.int64)
    pair_to_int = load_pair_index(SPLIT_DIR / "pair_index.csv", role="A")
    A = load_role_matrix(args.model, args.modality, a_ids, pair_to_int, load_included_shards(MANIFEST),
                         expect_dim=MODEL_DIMS[args.model], root=embed_dir(args.model, args.modality))
    A = np.ascontiguousarray(A[:args.limit_a] if args.limit_a else A, dtype=np.float32)
    log(f"loaded A {A.shape} in {time.time() - t0:.0f}s")

    res = faiss.StandardGpuResources()
    for k, dest in todo:
        t1 = time.time()
        L = np.ascontiguousarray(np.load(dest / "centroids.npy"), dtype=np.float32)
        if L.shape[0] != k:
            raise RuntimeError(f"{dest}: {L.shape[0]} centroids, expected {k}")
        score = knn_mean_dist(A, L, args.nn, res)
        # second, independent check: plain numpy over all of A for 3 landmarks
        a2 = None
        for i in np.random.default_rng(k).choice(k, 3, replace=False):
            if a2 is None:
                a2 = np.einsum("ij,ij->i", A, A)
            d2 = a2 - 2.0 * (A @ L[i]) + float(L[i] @ L[i])
            ref = float(np.sqrt(np.maximum(np.partition(d2, args.nn - 1)[:args.nn], 0)).mean())
            if abs(ref - score[i]) > 1e-3:
                raise RuntimeError(f"K={k} landmark {i}: GPU {score[i]:.6f} vs numpy {ref:.6f}")
        suffix = f".smoke{args.limit_a}" if args.limit_a else ""
        np.save(dest / f"density_A_k{args.nn}{suffix}.npy", score)
        (dest / f"density_A_k{args.nn}{suffix}.json").write_text(json.dumps({
            "what": f"mean ordinary L2 distance from each landmark to its {args.nn} nearest "
                    f"corpus (role A) vectors -- the protocol's DO density score (l.108)",
            "model": args.model, "modality": args.modality, "K": k, "nn": args.nn,
            "n_A": int(A.shape[0]), "exact": True, "metric": "L2 (root taken)",
            "embedding_root": str(embed_dir(args.model, args.modality)),
            "split_dir": str(SPLIT_DIR), "faiss": faiss.__version__,
            "created": datetime.now().isoformat(), "pbs_jobid": os.environ.get("PBS_JOBID"),
            "summary": {"min": float(score.min()), "median": float(np.median(score)),
                        "max": float(score.max())}}, indent=1))
        log(f"K={k}: density median {np.median(score):.4f} [{score.min():.4f}, {score.max():.4f}] "
            f"in {time.time() - t1:.0f}s")
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
