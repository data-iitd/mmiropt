"""
T5: build the spatial fidelity map's tiling L -- a dedicated k-means over A.

    python build_landmarks.py --model siglip2 --modality both --k 1024

Depends only on T3 (the split existing). It needs nothing from T4's index
builds and nothing from T6's exact-KNN, so it runs in parallel with both.

WHAT THIS IS
    K landmark centroids per (model, modality), fitted over A ONLY, in that
    model's own embedding space, with K ~ sqrt(|A|).  At query time T9 finds a
    query's single nearest landmark by an O(K) flat scan and reads that cell's
    precomputed mean CF@k.  The scan is index-agnostic by construction: it
    touches no ANN structure at all.

WHY NOT AN INDEX'S COARSE QUANTISER
    The catalogue spans nlist of 8, 128, 4096 and 32768.  Borrowing any one of
    them would give different configs different zone grids, and H3 -- does a map
    keyed by (landmark, config) beat one keyed by landmark alone -- is only
    meaningful if `landmark` denotes the same geography in both arms.  So the
    tiling is built here, once per (model, modality), and every config is scored
    against it.

WHY NOT T2's STRATIFICATION K-MEANS
    Different object entirely.  T2 fits k=25 per modality over the whole 7M
    corpus to balance a sampling draw; it never appears in H1-H5.  This fits
    K~1000 over A only, to estimate CF per cell, and is swept in H4.  Neither
    may be substituted for the other.

WHY A only
    A is the searchable corpus, so a landmark is a statement about index
    geometry.  Fitting over B or C would tile query density instead, and would
    additionally leak held-out queries into the map's construction.

FAISS IS USED HERE, AND THAT IS NOT A CONTRADICTION
    faiss.Kmeans is Lloyd's algorithm with a flat exact-search assignment step.
    It builds no ANN index and shares nothing with the catalogue.  It is used
    because it is already a dependency, it is GPU-capable, and it minimises the
    same L2 objective index_configs.METRIC declares -- so "nearest landmark"
    and "nearest neighbour" agree about what near means.

    Whether centroids are renormalised to the unit sphere is set by --spherical,
    and it is not a cosmetic choice -- see fit_landmarks' docstring.  The short
    version: data is unit-norm but a k-means centroid is not, so raw-L2
    assignment carries a query-independent ||c||^2 bias toward loose clusters.
    In-modality it is harmless; cross-modally it collapses the tiling to a
    handful of cells.  --spherical removes it.  Both variants coexist on disk.

SUBSAMPLING IS DISABLED
    faiss.Kmeans defaults to max_points_per_centroid=256, which would silently
    fit K=1024 landmarks on 262,144 of A's 850,393 rows.  The sparse and
    boundary regions this study exists to characterise are exactly what such a
    subsample thins out, so max_points_per_centroid is set to ceil(|A|/K) --
    every row of A trains the tiling.

DEGENERATE PAIRS
    Not handled here.  The 353 excluded pairs were dropped at T2 labelling, so
    they hold no int_id and never entered A.  Their rows are still physically
    present in the .npy files at their original positions; we simply never ask
    for them.  Inherited, not re-implemented.
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import faiss

REPO_ROOT = Path(__file__).resolve().parent.parent

# T4's loader, imported rather than reimplemented.  The row-to-ID mapping is the
# one thing in this pipeline that fails silently, so T5 must establish it the
# same way T4 does -- not merely a compatible way.
sys.path.insert(0, str(REPO_ROOT / "indexes"))
from index_configs import METRIC, MODEL_DIMS                   # noqa: E402
from load_vectors import (                                        # noqa: E402
    load_included_shards, load_pair_index, load_role_matrix,
)

DEFAULT_PILOT = REPO_ROOT / "result/pilot_1M/siglip2"
DEFAULT_MANIFEST = REPO_ROOT / "result/corpus_shard_manifest.json"
DEFAULT_OUT = REPO_ROOT / "result/kmeans/1M"


def out_dir(root: Path, model: str, modality: str, k: int,
            spherical: bool = False) -> Path:
    """Spherical runs get their own directory so the two tilings coexist."""
    return root / model / modality / (f"K{k:05d}_spherical" if spherical
                                      else f"K{k:05d}")


def _fmt(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def fit_landmarks(vecs, k, seed, niter, nredo, use_gpu, spherical=False,
                  verbose=True):
    """
    Lloyd's k-means over the full matrix.  Returns (centroids, meta).

    METRIC is asserted rather than assumed: faiss.Kmeans is L2-only, so an L2
    tiling under an inner-product catalogue would be a silent mismatch between
    "nearest landmark" and "nearest neighbour".

    SPHERICAL, AND WHY IT MATTERS HERE
        The data is unit-norm, so ||x-y||^2 = 2 - 2<x,y> and L2 ordering equals
        cosine ordering -- but ONLY when both sides are unit-norm.  A plain
        k-means centroid is not: it is the arithmetic mean of its members, so
        spread members partially cancel and ||c|| measures how tight the cluster
        is (measured here: 0.465 to 1.000).  The assignment then reads

            argmin_c [ ||c||^2 - 2<q,c> ]
                       ^^^^^^^ query-INdependent bias favouring loose clusters

        In-modality that bias is swamped by <q,c>, which varies strongly.
        Cross-modally the modality gap makes <q,c> small AND nearly constant
        across every centroid, so the bias decides the argmin and every text
        query picks the single shortest image centroid.  Measured at K=1024:
        60 of 1024 image cells ever used, one holding 58,824 of 99,833 queries.

        spherical=True renormalises centroids each iteration, so ||c|| = 1, the
        bias term vanishes identically, and L2 and cosine agree exactly again.
        Note this needs no change to `assign` -- once centroids are unit-norm,
        the IndexFlatL2 scan below IS a cosine scan.

        It does not remove the modality gap itself, only the artifact stacked
        on top of it.  See landmarks/README.md.
    """
    if METRIC != "L2":
        raise RuntimeError(
            f"index_configs.METRIC is {METRIC!r}; faiss.Kmeans minimises L2. "
            f"Landmark assignment and index search would order differently."
        )

    n, d = vecs.shape
    if k >= n:
        raise ValueError(f"K={k} >= |A|={n}: every point would be its own landmark")

    # Use every row of A -- see the module docstring.
    mppc = int(math.ceil(n / k))

    km = faiss.Kmeans(
        d, k,
        niter=niter,
        nredo=nredo,
        seed=seed,
        spherical=spherical,
        min_points_per_centroid=1,
        max_points_per_centroid=mppc,
        gpu=use_gpu,
        verbose=verbose,
    )

    t0 = time.time()
    km.train(vecs)
    fit_s = time.time() - t0

    # faiss's python Kmeans wrapper already hands these back as numpy.
    centroids = np.ascontiguousarray(
        np.asarray(km.centroids).reshape(k, d), dtype=np.float32)
    obj = [float(x) for x in np.asarray(km.obj).ravel()]

    cn = np.linalg.norm(centroids, axis=1)
    if spherical and not np.allclose(cn, 1.0, atol=1e-3):
        raise RuntimeError(
            f"spherical=True but centroid norms are [{cn.min():.6f}, {cn.max():.6f}] "
            f"-- the ||c||^2 bias would still be present"
        )
    return centroids, {"fit_seconds": fit_s, "objective_trace": obj,
                       "max_points_per_centroid": mppc, "spherical": bool(spherical),
                       "centroid_norm": {"min": float(cn.min()), "max": float(cn.max()),
                                         "median": float(np.median(cn))}}


def assign(centroids, vecs, nn=1, use_gpu=False):
    """
    Exact flat search of `vecs` against the landmarks.  This is the same O(K)
    scan T9 performs at query time -- deliberately flat, deliberately not an
    index.  Returns (D, I) with squared L2 distances.
    """
    k, d = centroids.shape
    index = faiss.IndexFlatL2(d)
    if use_gpu and faiss.get_num_gpus() > 0:
        index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)
    index.add(centroids)
    return index.search(vecs, nn)


def cell_stats(counts, n_assigned):
    """Occupancy summary for one assignment histogram."""
    k = len(counts)
    nz = counts[counts > 0]
    p = counts / max(n_assigned, 1)
    nzp = p[p > 0]
    entropy = float(-(nzp * np.log(nzp)).sum())
    order = np.sort(counts)[::-1]
    return {
        "K": int(k),
        "n_assigned": int(n_assigned),
        "empty_cells": int((counts == 0).sum()),
        "min": int(counts.min()), "max": int(counts.max()),
        "mean": float(counts.mean()),
        "median": float(np.median(counts)),
        "p01": float(np.percentile(counts, 1)),
        "p99": float(np.percentile(counts, 99)),
        # sum(c^2)*k/n^2 -- 1.0 is perfectly even, higher is more lopsided.
        "imbalance_factor": float((counts.astype(np.float64) ** 2).sum() * k
                                  / max(n_assigned, 1) ** 2),
        "entropy_nats": entropy,
        "entropy_ratio": float(entropy / np.log(k)) if k > 1 else 0.0,
        "frac_in_top_10_cells": float(order[:10].sum() / max(n_assigned, 1)),
        "cells_holding_50pct": int(np.searchsorted(np.cumsum(order),
                                                   0.5 * n_assigned) + 1),
        "cells_below_30": int((counts < 30).sum()),
        "cells_below_50": int((counts < 50).sum()),
    }


def build_one(vecs, a_ids, model, modality, k, args, shared_meta):
    d = vecs.shape[1]
    dest = out_dir(Path(args.out_dir), model, modality, k, args.spherical)
    dest.mkdir(parents=True, exist_ok=True)

    print(f"\n[{modality}] K={k}  fitting over {vecs.shape[0]:,} x {d} "
          f"({'spherical/cosine' if args.spherical else 'L2'}) ...")
    centroids, fit_meta = fit_landmarks(
        vecs, k, args.seed, args.niter, args.nredo, args.gpu,
        spherical=args.spherical, verbose=args.verbose
    )
    print(f"[{modality}] K={k}  fitted in {_fmt(fit_meta['fit_seconds'])}")

    # Assign A to its own tiling.  Two neighbours, not one: d2-d1 is the
    # boundary-proximity feature H1/H2 are stated in terms of, and recomputing
    # it later would mean a second full scan.
    D, I = assign(centroids, vecs, nn=2, use_gpu=args.gpu)
    lab = I[:, 0].astype(np.int32)
    counts = np.bincount(lab, minlength=k).astype(np.int64)

    # Per-cell tightness, free from the scan just done.  Under spherical fitting
    # ||c|| is 1 for every cell, so the cluster-spread signal that ||c||^2 used
    # to carry is gone -- and H5's DO baseline ("knows a landmark's corpus
    # density but no fidelity labels") wants it.  Recovered from d1 via
    #   d^2 = 1 - 2<x,c> + ||c||^2   =>   <x,c> = (1 + ||c||^2 - d^2) / 2
    # then divided by ||c|| to give the mean cosine of members to their centroid.
    cn2 = (centroids.astype(np.float64) ** 2).sum(1)
    dots = (1.0 + cn2[lab] - D[:, 0].astype(np.float64)) / 2.0
    cos_to_c = dots / np.sqrt(np.maximum(cn2[lab], 1e-12))
    tightness = np.bincount(lab, weights=cos_to_c, minlength=k) / np.maximum(counts, 1)

    np.save(dest / "centroids.npy", centroids)
    np.save(dest / "assign_A.npy", lab)
    np.save(dest / "cell_counts_A.npy", counts)
    np.save(dest / "cell_tightness_A.npy", tightness.astype(np.float32))
    np.savez(dest / "nn_A.npz",
             assign=lab,
             d1=D[:, 0].astype(np.float32),
             d2=D[:, 1].astype(np.float32))

    stats = cell_stats(counts, len(lab))
    meta = {
        "created": datetime.now().isoformat(),
        "task": "T5 landmark creation",
        "model": model,
        "modality": modality,
        "K": int(k),
        "K_over_sqrt_A": float(k / math.sqrt(len(a_ids))),
        "dim": int(d),
        "metric": METRIC,
        "n_A": int(len(a_ids)),
        "fitted_over": "role A only",
        "seed": args.seed,
        "niter": args.niter,
        "nredo": args.nredo,
        "gpu": bool(args.gpu),
        "occupancy_A": stats,
        "cell_tightness_A": {
            "min": float(tightness.min()), "median": float(np.median(tightness)),
            "max": float(tightness.max()),
        },
        "files": {
            "centroids.npy": f"float32 [{k}, {d}] -- the deliverable",
            "assign_A.npy": "int32 [|A|], row i is the landmark of A_ids[i]",
            "cell_counts_A.npy": "int64 [K], corpus density per cell (the DO baseline's input)",
            "cell_tightness_A.npy": ("float32 [K], mean cosine of a cell's members to "
                                     "its centroid -- density per unit volume, the other "
                                     "half of DO's signal"),
            "nn_A.npz": "assign, d1, d2 (squared L2) for A -- d2-d1 is H1/H2's boundary gap",
        },
        "contract": (
            "assign_A.npy is aligned to A_ids.npy from the pilot dir, NOT to any "
            "shard order. Row i is the landmark of pilot int_id A_ids[i]. A "
            "landmark id is an index into centroids.npy and means nothing "
            "outside this (model, modality, K, spherical) tuple."
        ),
        "not_a_quantiser": (
            "These centroids are never an index's coarse quantiser and no "
            "index's quantiser is ever used as a landmark set. H3 compares an "
            "(landmark, config)-keyed map against a landmark-only one, which "
            "requires every config to be scored against this one tiling."
        ),
        **shared_meta,
        **fit_meta,
    }
    json.dump(meta, open(dest / "kmeans_meta.json", "w"), indent=2)

    print(f"[{modality}] K={k}  empty={stats['empty_cells']}  "
          f"min/med/max={stats['min']}/{stats['median']:.0f}/{stats['max']}  "
          f"imbalance={stats['imbalance_factor']:.3f}  -> {dest}")
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="siglip2")
    ap.add_argument("--modality", default="both", choices=["image", "text", "both"])
    ap.add_argument("--k", type=int, nargs="+", default=[1024],
                    help="landmark counts; several = the H4 sweep, one load of A")
    ap.add_argument("--spherical", action="store_true",
                    help="renormalise centroids each iteration so ||c||=1, making L2 "
                         "and cosine agree exactly. Writes to K<k>_spherical/, so the "
                         "L2 tiling is untouched and the two stay comparable.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--niter", type=int, default=25)
    ap.add_argument("--nredo", type=int, default=1)
    ap.add_argument("--threads", type=int, default=0, help="0 = leave FAISS default")
    ap.add_argument("--no-gpu", dest="gpu", action="store_false", default=None)
    ap.add_argument("--gpu", dest="gpu", action="store_true")
    ap.add_argument("--pilot-dir", default=str(DEFAULT_PILOT))
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--image-root", default=None,
                    help="image vector tree (default: the model's result/embeddings_image tree); "
                         "v2 passes its result_v2 tree (indexes/v2_dirs.py)")
    ap.add_argument("--text-root", default=None, help="as --image-root, for captions")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--limit-a", type=int, default=0,
                    help="smoke test: fit on the first N of A. Still reads every "
                         "shard, so this is a compute-node test, not a login-node one.")
    ap.add_argument("--synthetic", type=int, default=0,
                    help="login-node-safe smoke test: N random unit vectors instead "
                         "of A. Touches no embedding shard. Point --out-dir somewhere "
                         "disposable -- the output is meaningless as a tiling.")
    ap.add_argument("--allow-login-node", action="store_true",
                    help="required to run full size outside a PBS job")
    ap.add_argument("--quiet", dest="verbose", action="store_false", default=True)
    args = ap.parse_args()

    if args.gpu is None:
        args.gpu = faiss.get_num_gpus() > 0

    in_job = bool(os.environ.get("PBS_JOBID"))
    if not in_job and not (args.limit_a or args.synthetic) and not args.allow_login_node:
        sys.exit(
            "Refusing to fit full-size k-means outside a PBS job.\n"
            "  qsub -I -q standard -P mimr.spons -l select=1:ncpus=64:ngpus=1 "
            "-l walltime=02:00:00\n"
            "or smoke-test with  --limit-a 20000 --no-gpu --out-dir /tmp/t5_smoke"
        )
    if args.threads:
        faiss.omp_set_num_threads(args.threads)

    pilot = Path(args.pilot_dir)
    modalities = ["image", "text"] if args.modality == "both" else [args.modality]

    a_ids = np.load(pilot / "A_ids.npy")
    if args.synthetic:
        pair_to_int, included = {}, set()
    else:
        pair_to_int = load_pair_index(pilot / "pair_index.csv", role="A")
        if len(pair_to_int) != len(a_ids):
            raise RuntimeError(f"pair_index.csv has {len(pair_to_int):,} role-A rows "
                               f"but A_ids.npy has {len(a_ids):,}")
        included = load_included_shards(Path(args.manifest))

    print(f"T5 landmarks | model={args.model} | |A|={len(a_ids):,} | "
          f"sqrt|A|={math.sqrt(len(a_ids)):.0f} | K={args.k} | "
          f"gpu={args.gpu} | threads={faiss.omp_get_max_threads()}")

    shared = {
        "pilot_dir": str(pilot),
        "id_scope": json.load(open(pilot / "pair_index_meta.json"))["scope"],
        "manifest": str(args.manifest),
        "included_shards": len(included),
        "embedding_roots": {"image": args.image_root or "v1 default", "text": args.text_root or "v1 default"},
    }
    if args.limit_a:
        shared["SMOKE_TEST"] = f"fitted on the first {args.limit_a} rows of A only"
    if args.synthetic:
        shared["SMOKE_TEST"] = (f"synthetic random unit vectors ({args.synthetic}), "
                                f"no embedding shard was read -- NOT a usable tiling")

    summary = {}
    roots = {"image": args.image_root, "text": args.text_root}     # None = v1 default tree
    for modality in modalities:
        if args.synthetic:
            rng = np.random.default_rng(args.seed)
            d = MODEL_DIMS[args.model]
            vecs = rng.standard_normal((args.synthetic, d), dtype=np.float32)
            vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
            ids = a_ids[:args.synthetic]
            print(f"  SMOKE TEST: {args.synthetic:,} synthetic unit vectors, dim {d}")
        else:
            vecs = load_role_matrix(
                args.model, modality, a_ids, pair_to_int, included,
                expect_dim=MODEL_DIMS[args.model],
                verbose=args.verbose,
                root=roots[modality])
            ids = a_ids
            if args.limit_a:
                vecs, ids = vecs[:args.limit_a].copy(), a_ids[:args.limit_a]
                print(f"  SMOKE TEST: using {len(ids):,} of {len(a_ids):,} rows")

        for k in args.k:
            summary[f"{modality}/K{k:05d}"] = build_one(
                vecs, ids, args.model, modality, k, args, shared)
        del vecs

    print("\n=== T5 summary ===")
    for key, s in summary.items():
        print(f"  {key:>18}  empty={s['empty_cells']:>4}  "
              f"min={s['min']:>6}  median={s['median']:>9.1f}  max={s['max']:>7}  "
              f"imbalance={s['imbalance_factor']:.3f}")
    print("\nNext: occupancy_check.py -- the deliverable's query-side half.")


if __name__ == "__main__":
    main()
