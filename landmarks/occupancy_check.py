"""
T5 (second half): the occupancy check -- do queries spread across zones?

    python occupancy_check.py --model siglip2 --k 1024

The tiling is fitted over A, but the map is *read* by queries, so a tiling that
looks healthy on A can still be useless: if B and C collapse into a handful of
cells, then almost every cell's mean CF@k is estimated from nothing and the map
has no support where it will actually be queried.  That is what this measures.

WHAT IS ASSIGNED TO WHAT
    Four (query modality -> landmark space) combinations are reported, because
    the catalogue evaluates four pairings and a landmark space mirrors an INDEX
    modality, not a query one:

        image queries -> image landmarks     i->i
        text  queries -> text  landmarks     t->t
        text  queries -> image landmarks     t->i
        image queries -> text  landmarks     i->t

    The two cross-modal rows are expected to be lopsided.  Image and text
    occupy systematically offset regions of the joint space (the modality gap),
    so text queries land in whichever image landmarks sit nearest the text
    region.  That is a finding to carry into T9, not a failure of the tiling --
    so the PASS/FAIL verdict is applied to the same-modality rows only and the
    cross-modal rows are reported as evidence.

    Keying the cross-modal pairings by index modality vs. by query modality is
    an open T9 decision.  Both are written here so T9 can settle it on data
    rather than by rebuild.

WHAT IS ALSO WRITTEN, AND WHY HERE
    d1 (distance to nearest landmark) and d2 (to the second nearest) per query.
    H1 asks whether prediction error grows with d1 and with the d2-d1 gap;
    H2 splits boundary queries from interior ones on that same gap.  Both come
    free from the scan this check already performs, and recomputing them in T9
    would mean a second full pass.

C IS TOUCHED, BUT NOT USED
    Assigning C here reads its vectors and its geometry -- no CF label, no
    fitted parameter, nothing that could tune the map.  It is a sizing check.
    If that ever feels close to the line, drop --roles C: the tiling is
    unaffected.
"""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

import faiss

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "indexes"))
from index_configs import MODEL_DIMS                              # noqa: E402
from load_vectors import load_included_shards, load_role_matrix    # noqa: E402

from build_landmarks import assign, cell_stats, out_dir           # noqa: E402

DEFAULT_PILOT = REPO_ROOT / "result/pilot_1M/siglip2"
DEFAULT_MANIFEST = REPO_ROOT / "result/corpus_shard_manifest.json"
DEFAULT_OUT = REPO_ROOT / "result/kmeans/1M"

PAIRING = {("image", "image"): "i->i", ("text", "text"): "t->t",
           ("text", "image"): "t->i", ("image", "text"): "i->t"}


def load_pair_index_roles(pair_index_csv, roles):
    """pair_id -> int_id for several roles in ONE pass over the master file."""
    want = set(roles)
    out = {}
    with open(pair_index_csv) as f:
        for r in csv.DictReader(f):
            if r["role"] in want:
                out[r["pair_id"]] = int(r["int_id"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="siglip2")
    ap.add_argument("--k", type=int, nargs="+", default=[1024])
    ap.add_argument("--spherical", action="store_true",
                    help="grade the spherical tiling in K<k>_spherical/ instead of the "
                         "L2 one. No change to the scan itself: once ||c||=1 the "
                         "IndexFlatL2 search below IS a cosine search.")
    ap.add_argument("--roles", nargs="+", default=["B", "C"], choices=["B", "C"])
    ap.add_argument("--pilot-dir", default=str(DEFAULT_PILOT))
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--image-root", default=None,
                    help="image vector tree (default: the model's result/embeddings_image tree); "
                         "v2 passes its result_v2 tree (indexes/v2_dirs.py)")
    ap.add_argument("--text-root", default=None, help="as --image-root, for captions")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--no-gpu", dest="gpu", action="store_false", default=None)
    ap.add_argument("--gpu", dest="gpu", action="store_true")
    # A cell needs enough B queries for its mean CF to mean anything; the build
    # plan puts that floor at 30-50.
    ap.add_argument("--min-b-per-cell", type=int, default=30)
    ap.add_argument("--max-frac-top10", type=float, default=0.25,
                    help="fail if this fraction of queries lands in 10 cells")
    ap.add_argument("--synthetic", type=int, default=0,
                    help="login-node-safe smoke test: N random unit query vectors "
                         "per role instead of B/C. Reads no embedding shard; the "
                         "verdict it prints is meaningless.")
    args = ap.parse_args()

    if args.gpu is None:
        args.gpu = faiss.get_num_gpus() > 0
    if args.threads:
        faiss.omp_set_num_threads(args.threads)

    pilot = Path(args.pilot_dir)
    root = Path(args.out_dir)
    role_ids = {r: np.load(pilot / f"{r}_ids.npy") for r in args.roles}

    if args.synthetic:
        role_ids = {r: v[:args.synthetic] for r, v in role_ids.items()}
        print(f"T5 occupancy | SMOKE TEST: {args.synthetic:,} synthetic queries "
              f"per role, no shard read")
        rng = np.random.default_rng(0)
        qvecs = {}
        for qmod in ("image", "text"):
            qvecs[qmod] = {}
            for r in args.roles:
                v = rng.standard_normal((len(role_ids[r]), MODEL_DIMS[args.model]), dtype=np.float32)
                v /= np.linalg.norm(v, axis=1, keepdims=True)
                qvecs[qmod][r] = v
    else:
        included = load_included_shards(Path(args.manifest))
        pair_to_int = load_pair_index_roles(pilot / "pair_index.csv", args.roles)

        # One combined ascending id array, so each modality is read off disk once
        # rather than once per role.
        all_ids = np.sort(np.concatenate(list(role_ids.values())))
        if len(np.unique(all_ids)) != len(all_ids):
            raise RuntimeError("roles overlap at the pair level -- B and C must be disjoint")
        if len(pair_to_int) != len(all_ids):
            raise RuntimeError(f"pair_index.csv has {len(pair_to_int):,} rows for "
                               f"{args.roles} but the id arrays hold {len(all_ids):,}")

        print(f"T5 occupancy | model={args.model} | K={args.k} | roles={args.roles} "
              f"({', '.join(f'{r}={len(v):,}' for r, v in role_ids.items())}) "
              f"| gpu={args.gpu}")

        qvecs = {}
        for qmod in ("image", "text"):
            eroot = {"image": args.image_root, "text": args.text_root}[qmod]
            mat = load_role_matrix(
                args.model, qmod, all_ids, pair_to_int, included,
                expect_dim=MODEL_DIMS[args.model], verbose=True,
                root=Path(eroot) if eroot else None)
            qvecs[qmod] = {r: mat[np.searchsorted(all_ids, role_ids[r])]
                           for r in args.roles}
            del mat

    failures, overall = [], {}

    for k in args.k:
        for lmod in ("image", "text"):
            dest = out_dir(root, args.model, lmod, k, args.spherical)
            cpath = dest / "centroids.npy"
            if not cpath.exists():
                sys.exit(f"missing {cpath} -- run build_landmarks.py"
                         f"{' --spherical' if args.spherical else ''} first")
            centroids = np.load(cpath)
            if centroids.shape[0] != k:
                raise RuntimeError(f"{cpath} holds {centroids.shape[0]} centroids, "
                                   f"expected {k}")
            # The flag must match how the tiling was actually fitted, or we would
            # grade one variant while labelling it the other.
            cn = np.linalg.norm(centroids, axis=1)
            is_sph = bool(np.allclose(cn, 1.0, atol=1e-3))
            if is_sph != args.spherical:
                raise RuntimeError(
                    f"{cpath}: centroid norms are [{cn.min():.4f}, {cn.max():.4f}], "
                    f"i.e. {'spherical' if is_sph else 'raw L2'}, but --spherical="
                    f"{args.spherical}"
                )
            qdir = dest / "queries"
            qdir.mkdir(parents=True, exist_ok=True)

            report = {
                "created": datetime.now().isoformat(),
                "model": args.model, "landmark_modality": lmod, "K": int(k),
                "spherical": bool(args.spherical),
                "embedding_roots": {"image": args.image_root, "text": args.text_root},
                "corpus_A": json.load(open(dest / "kmeans_meta.json"))["occupancy_A"],
                "queries": {},
            }

            for qmod in ("image", "text"):
                pairing = PAIRING[(qmod, lmod)]
                same = qmod == lmod
                for role in args.roles:
                    v = qvecs[qmod][role]
                    D, I = assign(centroids, v, nn=2, use_gpu=args.gpu)
                    lab = I[:, 0].astype(np.int32)
                    d1, d2 = D[:, 0].astype(np.float32), D[:, 1].astype(np.float32)
                    counts = np.bincount(lab, minlength=k).astype(np.int64)

                    np.savez(qdir / f"nn_{role}_{qmod}q.npz",
                             assign=lab, d1=d1, d2=d2)
                    with open(qdir / f"occupancy_{role}_{qmod}q.csv", "w",
                              newline="") as f:
                        w = csv.writer(f)
                        w.writerow(["landmark", "n_queries"])
                        w.writerows(enumerate(counts.tolist()))

                    s = cell_stats(counts, len(lab))
                    s["pairing"] = pairing
                    s["query_modality"] = qmod
                    s["same_modality"] = same
                    # sqrt of squared L2, so these read as ordinary distances.
                    s["d1_mean"] = float(np.sqrt(np.maximum(d1, 0)).mean())
                    s["boundary_gap_mean"] = float(
                        (np.sqrt(np.maximum(d2, 0)) - np.sqrt(np.maximum(d1, 0))).mean())

                    tag = f"K={k:<5} {lmod:>5} landmarks <- {role} {qmod:>5} ({pairing})"
                    if same:
                        bad = []
                        if s["frac_in_top_10_cells"] > args.max_frac_top10:
                            bad.append(f"{s['frac_in_top_10_cells']:.1%} of queries "
                                       f"in 10 cells")
                        if role == "B" and s["cells_below_30"] > 0:
                            bad.append(f"{s['cells_below_30']} cells hold "
                                       f"<{args.min_b_per_cell} B queries")
                        s["verdict"] = "FAIL" if bad else "PASS"
                        s["notes"] = bad
                        if bad:
                            failures.append(f"{tag}: " + "; ".join(bad))
                    else:
                        s["verdict"] = "INFO"
                        s["notes"] = ["cross-modal: lopsided occupancy is the "
                                      "modality gap, not a tiling defect"]

                    report["queries"][f"{role}_{qmod}q"] = s
                    overall[f"K{k:05d}/{lmod}L/{role}_{qmod}q"] = s
                    print(f"  {tag}  empty={s['empty_cells']:>4} "
                          f"med={s['median']:>7.1f} max={s['max']:>6} "
                          f"top10={s['frac_in_top_10_cells']:>6.1%} "
                          f"[{s['verdict']}]")

            json.dump(report, open(dest / "occupancy_report.json", "w"), indent=2)

    # Variant-aware: a spherical run must not silently replace the L2 baseline
    # it is being compared against.
    summary_path = root / args.model / (
        "occupancy_summary_spherical.json" if args.spherical
        else "occupancy_summary.json")
    json.dump({"created": datetime.now().isoformat(), "model": args.model,
               "K_values": args.k, "spherical": bool(args.spherical),
               "roles": args.roles,
               "thresholds": {"min_b_per_cell": args.min_b_per_cell,
                              "max_frac_top10": args.max_frac_top10},
               "results": overall,
               "failures": failures}, open(summary_path, "w"), indent=2)

    print(f"\nwrote {summary_path}")
    if failures:
        print("\nFAIL -- same-modality occupancy checks:")
        for f in failures:
            print("  " + f)
        sys.exit(1)
    print("\nPASS -- queries spread across zones in both same-modality pairings.")


if __name__ == "__main__":
    main()
