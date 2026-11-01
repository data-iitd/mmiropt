"""
T2 stratification k-means for SigLIP 2: two separate fits (image, text),
k~25 each, over the actual on-disk shard layout.

Defaults match the current merged corpus (result/embeddings_image and
result/embeddings_text, both under siglip2_so400m_patch16_512, 3236/3236
shard keys identical). --image-root accepts multiple paths in case image
embeddings ever get split back across parts.

Run with no args:
    python run_stratification.py
Or override any path/hyperparameter:
    python run_stratification.py --k 30 --image-root A B C --out-dir /somewhere
"""

import argparse
from pathlib import Path

from stratify_kmeans import (
    load_excluded_pairs,
    load_corpus_manifest,
    fit_minibatch_kmeans,
    assign_and_save_labels,
    save_model,
)

REPO_ROOT = Path(__file__).resolve().parent.parent  # .../mmir_optimizer

DEFAULT_MODEL = "siglip2"
MODEL_DIRS = {"siglip2": "siglip2_so400m_patch16_512"}   # add other models here

DEFAULT_EXCLUDED_PAIRS = REPO_ROOT / "result/excluded_pairs.txt"
DEFAULT_CORPUS_MANIFEST = REPO_ROOT / "result/corpus_shard_manifest.json"


def run_one_modality(roots, excluded_path, manifest_path, out_dir, k, n_epochs, seed, tag):
    print(f"=== stratification k-means: {tag} (k={k}) ===")
    excluded = load_excluded_pairs(excluded_path)
    included_shards = load_corpus_manifest(manifest_path) if manifest_path else None
    km = fit_minibatch_kmeans(roots, excluded, k=k, n_epochs=n_epochs, seed=seed, included_shards=included_shards)

    out_prefix = out_dir / f"{tag}_stratum"
    out_csv = out_dir / f"{tag}_stratum_labels.csv"
    n_written, n_excl = assign_and_save_labels(roots, excluded, km, out_csv, included_shards=included_shards)
    save_model(km, out_prefix)

    print(f"{tag}: {n_written:,} rows labelled, {n_excl:,} excluded rows skipped")
    print(f"  -> {out_csv}")
    print(f"  -> {out_prefix}_centroids.npy / _model.joblib")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL,
                     help="reference space; sets the embedding dirs and the output subdir")
    ap.add_argument("--image-root", nargs="+", type=Path, default=None,
                     help="one or more dirs, each containing worker_*/shard_*.npy")
    ap.add_argument("--text-root", nargs="+", type=Path, default=None)
    ap.add_argument("--excluded-pairs", type=Path, default=DEFAULT_EXCLUDED_PAIRS)
    ap.add_argument("--corpus-manifest", type=Path, default=DEFAULT_CORPUS_MANIFEST,
                     help="shard keys inside the 7M soft-limit cut (see select_corpus.py); "
                          "pass an empty string to disable the cap")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--k", type=int, default=25, help="stratification k (T2), NOT the T5 landmark K")
    ap.add_argument("--n-epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", choices=["image", "text", "both"], default="both")
    args = ap.parse_args()

    # Paths hang off --model unless given explicitly, so a second model is a
    # one-word change rather than four path arguments.
    emb_dir = MODEL_DIRS.get(args.model, args.model)
    if args.image_root is None:
        args.image_root = [REPO_ROOT / "result/embeddings_image" / emb_dir]
    if args.text_root is None:
        args.text_root = [REPO_ROOT / "result/embeddings_text" / emb_dir]
    if args.out_dir is None:
        args.out_dir = REPO_ROOT / "result/stratification" / args.model

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.corpus_manifest if str(args.corpus_manifest) else None

    if args.only in ("image", "both"):
        run_one_modality(args.image_root, args.excluded_pairs, manifest_path, args.out_dir,
                          args.k, args.n_epochs, args.seed, tag="image")
    if args.only in ("text", "both"):
        run_one_modality(args.text_root, args.excluded_pairs, manifest_path, args.out_dir,
                          args.k, args.n_epochs, args.seed, tag="text")


if __name__ == "__main__":
    main()
