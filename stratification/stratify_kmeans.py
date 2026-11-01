"""
Core k-means routines for T2 stratification (SigLIP 2 embeddings).

Data layout (matches embeddingsgenerator/gen_siglip2*.py output):
    <root>/worker_N/shard_M.npy          float32 [rows, dim], L2-normalised
    <root>/worker_N/shard_M.names.json   `rows` names, same order as the .npy

Identity is the PAIR ID: "worker_N/shard_M/<name>", extension stripped. Image
roots and the text root share this same id space by construction -- verified
against this corpus (3236/3236 shard keys identical, per-shard name order
identical).

This is the T2 stratification k-means (k~25) -- NOT the T5 landmark k-means
(K~1000). Do not reuse outputs of this module as landmarks.

Degenerate pairs are never dropped/recompacted: they're skipped during fit
and predict, and simply never get a row in the output label file.
"""

import json
from pathlib import Path
from typing import Iterable, Iterator, Optional, Tuple, List, Set

import numpy as np
from sklearn.cluster import MiniBatchKMeans
import joblib


def load_excluded_pairs(path: Path) -> Set[str]:
    """One pair-id per line, e.g. 'worker_0/shard_00032/000324669'."""
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def load_corpus_manifest(path: Path) -> Set[str]:
    """Shard keys (worker_N/shard_M) inside the canonical 7M soft-limit cut."""
    manifest = json.load(open(path))
    return set(manifest["included_shards"])


def _pair_id(name: str) -> str:
    """'worker_0/shard_00130/001306322.jpg' -> 'worker_0/shard_00130/001306322'"""
    return name.rsplit(".", 1)[0]


def _shard_key(root: Path, npy_path: Path) -> str:
    """'worker_0/shard_00130.npy' (relative to root) -> 'worker_0/shard_00130'"""
    return str(npy_path.relative_to(root)).removesuffix(".npy")


def iter_shards(roots: Iterable[Path], included_shards: Optional[Set[str]] = None) -> Iterator[Path]:
    """
    All shard .npy files under the given roots. If included_shards is given
    (the corpus manifest), only shards whose key is in that set are yielded --
    this is how the 7M soft-limit cut is enforced, on top of the exclusion
    mask which drops individual rows within a kept shard.
    """
    for root in roots:
        for f in sorted(root.glob("worker_*/shard_*.npy")):
            if included_shards is None or _shard_key(root, f) in included_shards:
                yield f


def load_shard(npy_path: Path) -> Tuple[List[str], np.ndarray]:
    names_path = npy_path.with_suffix("").with_suffix(".names.json")
    import json
    names = json.load(open(names_path))
    vecs = np.load(npy_path)
    assert len(names) == vecs.shape[0], f"{npy_path}: names/rows mismatch"
    pair_ids = [_pair_id(n) for n in names]
    return pair_ids, vecs


def fit_minibatch_kmeans(
    roots: Iterable[Path],
    excluded: Set[str],
    k: int,
    n_epochs: int = 5,
    seed: int = 0,
    included_shards: Optional[Set[str]] = None,
) -> MiniBatchKMeans:
    """
    Fit MiniBatchKMeans one shard at a time (natural batch = one shard's
    valid rows), shuffling shard order each epoch. Never materializes the
    full corpus in RAM.
    """
    shard_paths = list(iter_shards(roots, included_shards))
    assert shard_paths, f"no shards found under {list(roots)}"

    km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=10)
    rng = np.random.default_rng(seed)

    for epoch in range(n_epochs):
        order = rng.permutation(len(shard_paths))
        rows_seen = 0
        for i, idx in enumerate(order):
            pair_ids, vecs = load_shard(shard_paths[idx])
            keep = np.fromiter((pid not in excluded for pid in pair_ids), dtype=bool, count=len(pair_ids))
            batch = vecs[keep]
            if batch.shape[0] == 0:
                continue
            km.partial_fit(batch)
            rows_seen += batch.shape[0]
            if (i + 1) % 500 == 0:
                print(f"[epoch {epoch + 1}/{n_epochs}] {i + 1}/{len(shard_paths)} shards, {rows_seen:,} rows fit")
        print(f"[epoch {epoch + 1}/{n_epochs}] done: {rows_seen:,} rows")

    return km


def assign_and_save_labels(
    roots: Iterable[Path],
    excluded: Set[str],
    km: MiniBatchKMeans,
    out_csv: Path,
    included_shards: Optional[Set[str]] = None,
) -> Tuple[int, int]:
    """
    Predicts a cluster label for every valid row and streams (pair_id,label)
    to out_csv, one shard at a time. Returns (n_rows_written, n_excluded_seen).
    """
    n_written, n_excluded_seen = 0, 0
    with open(out_csv, "w") as f:
        f.write("pair_id,stratum\n")
        for npy_path in iter_shards(roots, included_shards):
            pair_ids, vecs = load_shard(npy_path)
            keep_mask = np.fromiter((pid not in excluded for pid in pair_ids), dtype=bool, count=len(pair_ids))
            n_excluded_seen += int((~keep_mask).sum())
            kept_ids = [pid for pid, k in zip(pair_ids, keep_mask) if k]
            if not kept_ids:
                continue
            labels = km.predict(vecs[keep_mask])
            for pid, lab in zip(kept_ids, labels):
                f.write(f"{pid},{lab}\n")
            n_written += len(kept_ids)
    return n_written, n_excluded_seen


def save_model(km: MiniBatchKMeans, out_prefix: Path) -> None:
    np.save(f"{out_prefix}_centroids.npy", km.cluster_centers_.astype(np.float32))
    joblib.dump(km, f"{out_prefix}_model.joblib")
