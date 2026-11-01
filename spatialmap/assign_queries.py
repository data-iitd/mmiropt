"""
T9, part 1 of 2 -- per-query landmark geometry for B and C.

    python assign_queries.py --model siglip2 --spherical

For every query in B and C, under every pairing, this records

    assign  the nearest landmark  (the map's lookup key)
    d1      distance to it        (H1's "does error grow with distance?")
    d2      distance to the second-nearest landmark
    gap     d2 - d1               (H2's boundary-proximity splitter)

WHY THIS MOSTLY COPIES RATHER THAN COMPUTES
-------------------------------------------
T5's occupancy_check.py already assigned every B and C query to both tilings
and stored (assign, d1, d2) in ``queries/nn_{role}_{qmod}q.npz``. Recomputing
here would mean re-deriving a quantity that already exists, from centroids that
must be loaded with exactly matching --spherical and metric conventions. Any
drift between the two derivations is invisible: the map would be BUILT on one
assignment and GRADED on another, and every cell mean would be quietly averaging
the wrong queries.

So the default path reads T5's arrays, de-squares the distances, and adds the
gap column. ``--recompute`` exists for the case where T5's queries/ directory is
missing or a tiling was refitted, and it is verified against T5's copy whenever
both are present (``--check``), rather than silently preferred.

WHAT THIS IS NOT
----------------
Not the map. This is the geometry half only, split into its own script because
H1 and H2 in T10 need it per query, whereas the map itself is per cell and
throws the per-query detail away.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from map_config import (                                               # noqa: E402
    DEFAULT_CF, DEFAULT_KMEANS, DEFAULT_OUT, DEFAULT_PILOT, PAIRINGS, ROLES,
    layout, load_query_geometry, load_role_ids, resolve_pairings, write_json,
)


def recompute_geometry(lay, pairing: str, role: str, *, threads: int, gpu):
    """
    Fallback: assign queries to centroids here instead of reading T5's answer.

    Only reached under --recompute. Loads the role's vectors in the query
    modality and runs a 2-NN search against the tiling, which is the same
    operation T5's occupancy_check performs.
    """
    import faiss
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "indexes"))
    from load_vectors import load_role_matrix                          # noqa: E402

    spec = PAIRINGS[pairing]
    lmod, qmod = spec["index_modality"], spec["query_modality"]

    centroids = np.load(lay.centroids(lmod))
    if threads:
        faiss.omp_set_num_threads(threads)

    # The --spherical flag must describe how the tiling was actually fitted.
    # Assigning against raw-L2 centroids while calling them spherical (or the
    # reverse) yields a different tiling with no error raised.
    norms = np.linalg.norm(centroids, axis=1)
    is_sph = bool(np.allclose(norms, 1.0, atol=1e-3))
    if is_sph != lay.spherical:
        raise RuntimeError(
            f"{lay.centroids(lmod)}: centroid norms are "
            f"[{norms.min():.4f}, {norms.max():.4f}] i.e. "
            f"{'spherical' if is_sph else 'raw L2'}, but --spherical={lay.spherical}")

    vecs = load_role_matrix(
        model=lay.model, modality=qmod, role=role,
        pilot_dir=lay.pilot / lay.model,
        manifest=lay.pilot.parent / "corpus_shard_manifest.json",
    )
    index = faiss.IndexFlatL2(centroids.shape[1])
    if gpu:
        index = faiss.index_cpu_to_all_gpus(index)
    index.add(np.ascontiguousarray(centroids, dtype=np.float32))
    D, I = index.search(np.ascontiguousarray(vecs, dtype=np.float32), 2)

    d1 = np.sqrt(np.maximum(D[:, 0].astype(np.float64), 0.0)).astype(np.float32)
    d2 = np.sqrt(np.maximum(D[:, 1].astype(np.float64), 0.0)).astype(np.float32)
    return {"assign": I[:, 0].astype(np.int32), "d1": d1, "d2": d2,
            "gap": (d2 - d1).astype(np.float32)}


def summarise(geom, n_landmarks: int) -> dict:
    """Distribution facts H1/H2 will want, computed once and stored."""
    d1, gap, assign = geom["d1"], geom["gap"], geom["assign"]
    counts = np.bincount(assign, minlength=n_landmarks)
    return {
        "n_queries": int(len(assign)),
        "n_landmarks": int(n_landmarks),
        "empty_cells": int((counts == 0).sum()),
        "d1": {"mean": float(d1.mean()), "median": float(np.median(d1)),
               "p10": float(np.percentile(d1, 10)),
               "p90": float(np.percentile(d1, 90)),
               "max": float(d1.max())},
        "gap": {"mean": float(gap.mean()), "median": float(np.median(gap)),
                "p10": float(np.percentile(gap, 10)),
                "p90": float(np.percentile(gap, 90))},
        # A negative gap is impossible -- d2 is by construction the second
        # smallest of the same search, so this catches a corrupted or
        # column-swapped npz rather than a modelling problem.
        "negative_gaps": int((gap < 0).sum()),
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="siglip2")
    ap.add_argument("--k", type=int, default=1024, help="landmark count K of the tiling")
    ap.add_argument("--spherical", action="store_true",
                    help="use the spherical tiling (K01024_spherical). This is the "
                         "one T5's README says to build on.")
    ap.add_argument("--pairing", nargs="+", default=["all"])
    ap.add_argument("--roles", nargs="+", default=list(ROLES), choices=list(ROLES))
    ap.add_argument("--recompute", action="store_true",
                    help="assign here instead of reading T5's npz (needs vectors)")
    ap.add_argument("--check", action="store_true",
                    help="with --recompute, also compare against T5's stored copy")
    ap.add_argument("--pilot-dir", type=Path, default=DEFAULT_PILOT)
    ap.add_argument("--kmeans-dir", type=Path, default=DEFAULT_KMEANS)
    ap.add_argument("--cf-dir", type=Path, default=DEFAULT_CF)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--allow-login-node", action="store_true")
    args = ap.parse_args()

    # Reading T5's npz is a few MB of I/O and is login-node safe. Recomputing
    # loads the full role matrices and runs a FAISS search, which is not.
    if args.recompute and not os.environ.get("PBS_JOBID") and not args.allow_login_node:
        sys.exit("--recompute loads full role matrices; run it inside a PBS job "
                 "or pass --allow-login-node.")

    lay = layout(args.model, args.k, args.spherical, pilot=args.pilot_dir,
                 kmeans=args.kmeans_dir, cf=args.cf_dir, out=args.out_dir)
    pairings = resolve_pairings(args.pairing)

    print(f"T9 geometry  model={lay.model}  tiling={lay.variant}")
    print(f"  source: {'recomputed here' if args.recompute else 'T5 queries/*.npz'}")
    print(f"  dest:   {lay.dest / 'geometry'}\n")

    report = {"created": datetime.now().isoformat(), "task": "T9 query geometry",
              "model": lay.model, "K": lay.k, "spherical": lay.spherical,
              "source": "recomputed" if args.recompute else "T5",
              "pairings": {}}

    for pairing in pairings:
        spec = PAIRINGS[pairing]
        for role in args.roles:
            dest = lay.geometry(role, pairing)
            if dest.exists() and not args.overwrite:
                print(f"  {pairing:24s} {role}  exists, skipping (--overwrite to redo)")
                continue

            if args.recompute:
                geom = recompute_geometry(lay, pairing, role,
                                          threads=args.threads, gpu=args.gpu)
                if args.check and lay.query_nn(spec["index_modality"], role,
                                               spec["query_modality"]).exists():
                    ref = load_query_geometry(lay, pairing, role)
                    agree = float((ref["assign"] == geom["assign"]).mean())
                    print(f"  {pairing:24s} {role}  vs T5: "
                          f"{agree:.4%} of assignments agree")
                    if agree < 0.999:
                        raise RuntimeError(
                            f"{pairing}/{role}: recomputed assignment differs from "
                            f"T5's on {1 - agree:.2%} of queries. The map would be "
                            f"built on one tiling and graded on another.")
            else:
                geom = load_query_geometry(lay, pairing, role)

            ids = load_role_ids(lay, role)
            if len(ids) != len(geom["assign"]):
                raise RuntimeError(
                    f"{pairing}/{role}: {len(geom['assign']):,} assignments but "
                    f"{len(ids):,} ids in {lay.role_ids(role)}. Row i of the "
                    f"assignment must be the landmark of {role}_ids[i].")

            dest.parent.mkdir(parents=True, exist_ok=True)
            np.savez(dest, query_id=ids.astype(np.int64), **geom)

            s = summarise(geom, np.load(lay.centroids(spec["index_modality"])).shape[0])
            s["query_modality"] = spec["query_modality"]
            s["index_modality"] = spec["index_modality"]
            report["pairings"][f"{pairing}/{role}"] = s
            print(f"  {pairing:24s} {role}  n={s['n_queries']:>6,}  "
                  f"empty={s['empty_cells']:>4}  d1={s['d1']['mean']:.4f}  "
                  f"gap={s['gap']['mean']:.4f}")

    write_json(lay.dest / "geometry" / "geometry_report.json", report)
    print(f"\nwrote {lay.dest / 'geometry'}")


if __name__ == "__main__":
    main()
