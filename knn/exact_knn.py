"""
T6: exact-KNN labelling -- the ground truth every CF@k in T8 is measured against.

    python exact_knn.py --model siglip2 --pairing text_query_image_index

For every query in B and in C, the true top-100 neighbours over A, by brute-force
flat search.  This is the one computation the spatial-fidelity-map premise exists
to let deployment avoid -- but it still has to be paid for once, offline, because
without it there is nothing to call an approximate result wrong against.

DEPENDS ONLY ON T3
    The split, and nothing else.  No index from T4, no landmarks from T5, so all
    three run in parallel off the split.

THERE IS NO ANN INDEX HERE, AND THAT IS THE ENTIRE POINT
    IndexFlatL2 is an exhaustive scan: no training, no graph, no codebook, no
    nprobe.  It is not a ninth configuration and must never be treated as one.
    If any approximation crept in here, every CF@k in T8 would be an
    intersection of two approximate lists and would silently mean nothing.

METRIC
    index_configs.METRIC, asserted rather than assumed.  The vectors are
    unit-norm, so ||x-y||^2 = 2 - 2<x,y> and L2-argmin orders identically to
    cosine, cross-modally included.  CF@k intersects this top-k with T7's; if
    the two disagree about the metric the intersection still returns a number,
    just a meaningless one.

    Distances are written as FAISS returns them: SQUARED L2.  Same convention as
    T5's d1/d2.  Take the square root before reading one as a distance.

ONE RUN PER PAIRING, NOT PER CONFIG
    Exact top-k depends on the query vectors, the corpus vectors and the metric.
    It does not depend on any index.  So the four (query modality -> index
    modality) pairings need four runs between them, and all 8 configs at a given
    pairing reuse the same labels:

        image queries -> image index    i->i
        text  queries -> text  index    t->t
        text  queries -> image index    t->i     <- cross-modal
        image queries -> text  index    i->t     <- cross-modal

    A pairing's "index" side is A's vectors in the index modality -- exactly
    what T4 populated its 8 indexes with.

HOW THE IDS SURVIVE, AND WHY IT IS DONE THIS WAY
    IndexFlatL2 is searched bare and returns ROW POSITIONS into A's matrix; the
    canonical pilot int_ids are recovered by the gather A_ids[I].  That is
    correct because load_vectors guarantees the one invariant -- row i of the
    matrix is the vector for A_ids[i] -- and it is the same guarantee T4's
    add_with_ids rests on, established by the same loader rather than by a
    merely compatible one.

    IndexIDMap2 is deliberately NOT used: it does not survive index_cpu_to_gpu,
    and the gather is exact anyway.  The two things that could make the gather
    silently wrong are both checked before it happens -- FAISS returning -1 for
    a short result (which would index A_ids[-1] and yield a plausible wrong id
    with no error), and a broken row-to-ID pairing (caught by self-retrieval).

WHAT IS NOT IN THE INDEX
    A only.  B and C are never added.  A query that is also an index point
    retrieves itself at distance zero and inflates every fidelity number
    downstream; the split is at the pair level precisely so a query's partner is
    never in the corpus either.  Checked directly: no returned id may lie in
    B or C, and no query may retrieve its own int_id.

DEGENERATE PAIRS
    Not handled here.  The 353 excluded pairs were dropped at T2 labelling, so
    they hold no int_id and entered neither A nor B nor C.  Their rows are still
    physically present in the .npy files at their original positions; we simply
    never ask for them.  Inherited, not re-implemented.
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import faiss

REPO_ROOT = Path(__file__).resolve().parent.parent

# T4's loader, imported rather than reimplemented.  The row-to-ID mapping is the
# one thing in this pipeline that fails without an error message, so T6 must
# establish it the same way T4 does -- not merely a compatible way.
sys.path.insert(0, str(REPO_ROOT / "indexes"))
from index_configs import METRIC, MODEL_DIMS                      # noqa: E402
from load_vectors import (                                         # noqa: E402
    load_included_shards, load_role_matrix,
)

DEFAULT_PILOT = REPO_ROOT / "result/pilot_1M/siglip2"
DEFAULT_MANIFEST = REPO_ROOT / "result/corpus_shard_manifest.json"
DEFAULT_OUT = REPO_ROOT / "result/knn/1M"

MODALITIES = ("image", "text")

# Directory names are written out in full rather than abbreviated to t2i/i2t.
# The compact forms are ambiguous about which side is the query, and getting
# that backwards produces a complete, plausible, entirely wrong label set.
PAIRINGS = {
    f"{q}_query_{i}_index": {"query_modality": q, "index_modality": i,
                             "short": f"{q[0]}->{i[0]}", "same_modality": q == i}
    for i in MODALITIES for q in MODALITIES
}

FAISS_METRIC = {"L2": faiss.METRIC_L2, "IP": faiss.METRIC_INNER_PRODUCT}[METRIC]


def _fmt(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def out_dir(root: Path, pairing: str, model: str) -> Path:
    """result/knn/1M/<pairing>/<model>/ -- pairing first, model second."""
    return root / pairing / model


def load_pair_index_roles(pair_index_csv: Path, roles) -> dict:
    """pair_id -> int_id for several roles, in ONE pass over the master file."""
    want = set(roles)
    out = {}
    with open(pair_index_csv) as f:
        for r in csv.DictReader(f):
            if r["role"] in want:
                out[r["pair_id"]] = int(r["int_id"])
    return out


def to_search_index(flat, use_gpu, ngpu):
    """
    Move the flat scan onto GPUs if asked.  Sharding splits A across devices and
    FAISS merges the per-shard results; with successive_ids the returned labels
    stay global row positions into the single A matrix, which is what the
    A_ids gather requires.
    """
    if not use_gpu:
        return flat, "cpu"
    avail = faiss.get_num_gpus()
    if avail == 0:
        raise SystemExit(
            "no GPUs visible. Exact KNN over A is a GPU job -- run it inside a "
            "PBS job, or pass --no-gpu to force the CPU path."
        )
    n = avail if ngpu <= 0 else min(ngpu, avail)
    if n == 1:
        res = faiss.StandardGpuResources()
        return faiss.index_cpu_to_gpu(res, 0, flat), "gpu:1"
    co = faiss.GpuMultipleClonerOptions()
    co.shard = True                 # split A across devices, not replicate it
    co.useFloat16 = False           # exact means exact: fp32 distances
    return faiss.index_cpu_to_all_gpus(flat, co=co, ngpu=n), f"gpu:{n}-sharded"


def self_retrieval_check(index, vecs, n_probe=512):
    """
    The decisive check that the row-to-ID pairing survived.

    Re-query the index with rows that are already in it and require each to come
    back as itself.  On an exhaustive scan this is exact -- distance zero, rank
    one -- so a failure means the matrix and the id array disagree, which is the
    failure mode that otherwise produces plausible neighbours and a poisoned T8.

    An exact-duplicate vector elsewhere in A can legitimately tie at distance
    zero and be returned instead.  That is reported, not failed: the pairing is
    intact, the corpus merely contains a duplicate.
    """
    n = vecs.shape[0]
    rows = np.linspace(0, n - 1, min(n, n_probe)).astype(np.int64)
    D, I = index.search(np.ascontiguousarray(vecs[rows]), 1)
    d0 = D[:, 0]
    hit = I[:, 0] == rows
    zero = d0 <= 1e-4
    if not bool(zero.all()):
        bad = int((~zero).sum())
        raise RuntimeError(
            f"self-retrieval FAILED: {bad}/{len(rows)} probe rows did not find "
            f"themselves at distance ~0 (max {float(d0.max()):.6f}). The vector "
            f"matrix and A_ids are misaligned -- every label written from here "
            f"would be wrong with no error raised."
        )
    return {"probes": int(len(rows)),
            "exact_self_rank1": int(hit.sum()),
            "tied_duplicates": int((~hit).sum()),
            "max_self_distance": float(d0.max())}


def search_all(index, q, k_probe, batch, label=""):
    """Batched exhaustive search.  Returns (D, I) over row positions."""
    n = q.shape[0]
    D = np.empty((n, k_probe), dtype=np.float32)
    I = np.empty((n, k_probe), dtype=np.int64)
    t0 = time.time()
    for s in range(0, n, batch):
        e = min(s + batch, n)
        d, i = index.search(np.ascontiguousarray(q[s:e]), k_probe)
        D[s:e], I[s:e] = d, i
        done = e
        rate = done / max(time.time() - t0, 1e-9)
        print(f"    {label} {done:,}/{n:,}  {rate:,.0f} q/s  "
              f"eta {_fmt((n - done) / max(rate, 1e-9))}", end="\r", flush=True)
    print()
    return D, I, time.time() - t0


def grade(D, I, a_ids, q_ids, bc_ids, k):
    """
    Everything that must hold before these labels are allowed to be written.

    The -1 check comes first and is not optional: FAISS returns -1 for a short
    result, and A_ids[-1] is a perfectly valid gather that yields the last id in
    A. That would be a wrong label with no exception anywhere.
    """
    n = I.shape[0]
    if int(I.min()) < 0:
        raise RuntimeError(
            f"FAISS returned {int((I < 0).sum()):,} sentinel -1 labels -- fewer "
            f"than k neighbours were found. The A_ids gather would silently turn "
            f"each into A_ids[-1]."
        )
    if int(I.max()) >= len(a_ids):
        raise RuntimeError(f"row position {int(I.max())} out of range for |A|={len(a_ids):,}")

    # Sorted ascending is what makes "top-k" a prefix; a tiny fp tolerance
    # because sharded merges can reorder exact ties.
    dd = np.diff(D[:, :k], axis=1)
    if float(dd.min()) < -1e-3:
        raise RuntimeError(f"distances not non-decreasing (min delta {float(dd.min()):.6f})")

    ids = a_ids[I[:, :k]]

    # Nothing outside A may appear. By construction it cannot -- the gather only
    # reads A_ids -- so this is really a check on the split itself.
    leaked = np.intersect1d(ids.ravel(), bc_ids, assume_unique=False)
    if leaked.size:
        raise RuntimeError(
            f"{leaked.size:,} returned ids lie in B or C (e.g. {leaked[:5].tolist()}). "
            f"A and the query roles are not disjoint."
        )

    # A query retrieving itself would mean its pair is in the corpus, which the
    # pair-level split forbids -- and it inflates every fidelity number after.
    self_hit = int((ids == q_ids[:, None]).sum())
    if self_hit:
        raise RuntimeError(
            f"{self_hit:,} queries retrieved their own int_id: the query role "
            f"leaked into A."
        )

    # Boundary ties. If d[k-1] == d[k] the k-th place is arbitrary, so CF@k has
    # irreducible noise there that is not the index's fault. Cheap to measure
    # because k_probe already fetched one extra column; recorded for T8.
    ties = None
    if D.shape[1] > k:
        gap = D[:, k] - D[:, k - 1]
        ties = {"queries_with_boundary_tie": int((gap <= 1e-6).sum()),
                "frac": float((gap <= 1e-6).mean()),
                "median_gap": float(np.median(gap))}

    return ids, {"n_queries": int(n), "k": int(k),
                 "self_retrievals": 0, "leaked_bc_ids": 0,
                 "boundary_ties": ties,
                 "dist_min": float(D[:, 0].min()),
                 "dist_max_at_k": float(D[:, k - 1].max()),
                 "mean_dist_at_1": float(D[:, 0].mean()),
                 "mean_dist_at_k": float(D[:, k - 1].mean())}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="siglip2")
    ap.add_argument("--pairing", nargs="+", default=["text_query_image_index"],
                    choices=sorted(PAIRINGS) + ["all"],
                    help="query->index pairings to label (default: the cross-modal t->i)")
    ap.add_argument("--roles", nargs="+", default=["B", "C"], choices=["B", "C"])
    ap.add_argument("--k", type=int, default=100, help="neighbours to store per query")
    ap.add_argument("--k-probe", type=int, default=0,
                    help="neighbours to fetch; default k+1, the extra column "
                         "measuring boundary ties. Never stored.")
    ap.add_argument("--batch", type=int, default=8192, help="queries per search call")
    ap.add_argument("--pilot-dir", type=Path, default=DEFAULT_PILOT,
                    help="the SHARED pilot -- always siglip2/, see pilot_1M/README.md")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--image-root", type=Path, default=None,
                    help="image worker_*/shard_*.npy tree; default result/embeddings_image/<model>")
    ap.add_argument("--text-root", type=Path, default=None,
                    help="text worker_*/shard_*.npy tree; default result/embeddings_text/<model>")
    ap.add_argument("--gpu", dest="gpu", action="store_true", default=None)
    ap.add_argument("--no-gpu", dest="gpu", action="store_false")
    ap.add_argument("--ngpu", type=int, default=0, help="0 = every visible GPU")
    ap.add_argument("--threads", type=int, default=0, help="0 = leave FAISS default")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit-a", type=int, default=0, help="smoke test: first N of A")
    ap.add_argument("--limit-q", type=int, default=0, help="smoke test: first N queries")
    ap.add_argument("--synthetic", type=int, default=0,
                    help="login-node-safe smoke test: N random unit vectors for A "
                         "and N/8 per role. Reads no shard; the labels it writes "
                         "are meaningless, so use --out-dir /tmp/...")
    ap.add_argument("--allow-login-node", action="store_true",
                    help="run full size outside PBS. Don't.")
    args = ap.parse_args()

    pairings = sorted(PAIRINGS) if "all" in args.pairing else list(dict.fromkeys(args.pairing))
    k = args.k
    k_probe = args.k_probe or (k + 1)
    if k_probe < k:
        raise SystemExit(f"--k-probe {k_probe} < --k {k}")

    smoke = bool(args.synthetic or args.limit_a or args.limit_q)
    if not smoke and not os.environ.get("PBS_JOBID") and not args.allow_login_node:
        raise SystemExit(
            "Refusing to run full size outside a PBS job: this loads A (3.65 GiB "
            "over 2,875 shards) and scans it exhaustively. Submit an interactive "
            "job, or pass --synthetic / --limit-a for a smoke test."
        )
    if args.gpu is None:
        args.gpu = not smoke and faiss.get_num_gpus() > 0
    if args.threads:
        faiss.omp_set_num_threads(args.threads)

    pilot = args.pilot_dir
    roots = {"image": args.image_root, "text": args.text_root}   # None = the v1 default
    role_ids = {r: np.load(pilot / f"{r}_ids.npy") for r in args.roles}
    a_ids_full = np.load(pilot / "A_ids.npy")

    print("=" * 78)
    print(f"T6 exact-KNN labelling -- model={args.model}")
    pretty = ", ".join(f"{p} ({PAIRINGS[p]['short']})" for p in pairings)
    print(f"  pairings {pretty}")
    print(f"  roles    {', '.join(f'{r}={len(v):,}' for r, v in role_ids.items())}")
    print(f"  k        {k} stored, {k_probe} fetched")
    print(f"  metric   {METRIC} (rank-identical to cosine on unit-norm vectors)")
    print(f"  out      {args.out_dir}")
    print("=" * 78)

    # ---- assemble query vectors, one read per query modality ---------------
    # B and C are read together in a single ascending id array so a modality is
    # pulled off disk once rather than once per role.
    q_needed = sorted({PAIRINGS[p]["query_modality"] for p in pairings})
    i_needed = sorted({PAIRINGS[p]["index_modality"] for p in pairings})

    all_q_ids = np.sort(np.concatenate([role_ids[r] for r in args.roles]))
    if len(np.unique(all_q_ids)) != len(all_q_ids):
        raise RuntimeError("roles overlap at the pair level -- B and C must be disjoint")
    if np.intersect1d(all_q_ids, a_ids_full).size:
        raise RuntimeError("query ids intersect A -- the split is not disjoint")

    t_all = time.time()
    qvecs = {}
    if args.synthetic:
        rng = np.random.default_rng(0)
        n_syn_q = max(args.synthetic // 8, 1)
        role_ids = {r: v[:n_syn_q] for r, v in role_ids.items()}
        all_q_ids = np.sort(np.concatenate([role_ids[r] for r in args.roles]))
        print(f"  SMOKE TEST: synthetic A={args.synthetic:,}, "
              f"{n_syn_q:,} queries per role, no shard read")
        for qmod in q_needed:
            v = rng.standard_normal((len(all_q_ids), MODEL_DIMS[args.model]), dtype=np.float32)
            v /= np.linalg.norm(v, axis=1, keepdims=True)
            qvecs[qmod] = v
    else:
        included = load_included_shards(args.manifest)
        pair_to_int = load_pair_index_roles(pilot / "pair_index.csv", args.roles)
        if len(pair_to_int) != len(all_q_ids):
            raise RuntimeError(
                f"pair_index.csv has {len(pair_to_int):,} rows for {args.roles} "
                f"but the id arrays hold {len(all_q_ids):,}")
        for qmod in q_needed:
            print(f"\nloading queries: {qmod}")
            qvecs[qmod] = load_role_matrix(
                args.model, qmod, all_q_ids, pair_to_int, included,
                expect_dim=MODEL_DIMS.get(args.model), root=roots[qmod])

    # Row slices per role, aligned to the combined ascending array.
    role_slice = {r: np.searchsorted(all_q_ids, role_ids[r]) for r in args.roles}

    # ---- one index build per INDEX modality, reused by both its pairings ---
    for imod in i_needed:
        todo = [p for p in pairings if PAIRINGS[p]["index_modality"] == imod]
        dests = {p: out_dir(args.out_dir, p, args.model) for p in todo}
        if not args.overwrite:
            pending = [p for p in todo
                       if not all((dests[p] / f"knn_{r}_ids.npy").exists()
                                  for r in args.roles)]
            for p in set(todo) - set(pending):
                print(f"\n[{p}] complete, skipping (--overwrite to redo)")
            todo = pending
        if not todo:
            continue

        print(f"\n{'-' * 78}\nindex side: A / {imod}")
        t0 = time.time()
        if args.synthetic:
            rng = np.random.default_rng(1)
            a_ids = a_ids_full[:args.synthetic].copy()
            avecs = rng.standard_normal((len(a_ids), MODEL_DIMS[args.model]), dtype=np.float32)
            avecs /= np.linalg.norm(avecs, axis=1, keepdims=True)
        else:
            a_pair_to_int = load_pair_index_roles(pilot / "pair_index.csv", ["A"])
            a_ids = a_ids_full
            avecs = load_role_matrix(
                args.model, imod, a_ids, a_pair_to_int,
                load_included_shards(args.manifest),
                expect_dim=MODEL_DIMS.get(args.model), root=roots[imod])
            if args.limit_a:
                avecs, a_ids = avecs[:args.limit_a].copy(), a_ids[:args.limit_a].copy()
                print(f"  SMOKE TEST: |A| cut to {len(a_ids):,}")
        load_s = time.time() - t0
        d = avecs.shape[1]
        print(f"  A: {len(a_ids):,} x {d} in {_fmt(load_s)}  "
              f"(ids {a_ids[0]:,}..{a_ids[-1]:,}, sparse -- B and C leave the gaps)")

        if len(a_ids) < k:
            raise SystemExit(f"|A|={len(a_ids):,} < k={k}")

        flat = faiss.IndexFlatL2(d) if METRIC == "L2" else faiss.IndexFlat(d, FAISS_METRIC)
        flat.add(avecs)
        if flat.ntotal != len(a_ids):
            raise RuntimeError(f"ntotal {flat.ntotal:,} != |A| {len(a_ids):,}")
        index, where = to_search_index(flat, args.gpu, args.ngpu)
        print(f"  IndexFlat{METRIC} on {where}  ntotal={flat.ntotal:,}")

        selfcheck = self_retrieval_check(index, avecs)
        print(f"  self-retrieval OK: {selfcheck['exact_self_rank1']}/"
              f"{selfcheck['probes']} rank-1 self"
              + (f", {selfcheck['tied_duplicates']} tied duplicates"
                 if selfcheck["tied_duplicates"] else ""))

        for p in todo:
            qmod = PAIRINGS[p]["query_modality"]
            dest = dests[p]
            dest.mkdir(parents=True, exist_ok=True)
            print(f"\n[{p}]  {PAIRINGS[p]['short']}  "
                  f"{qmod} queries -> {imod} corpus")

            per_role, checks = {}, {}
            for r in args.roles:
                q = qvecs[qmod][role_slice[r]]
                q_ids = role_ids[r]
                if args.limit_q:
                    q, q_ids = q[:args.limit_q], q_ids[:args.limit_q]
                bc = np.sort(np.concatenate([role_ids[x] for x in args.roles]))

                D, I, search_s = search_all(index, q, k_probe, args.batch,
                                            label=f"{r}:")
                ids, stats = grade(D, I, a_ids, q_ids, bc, k)

                np.save(dest / f"knn_{r}_ids.npy", ids.astype(np.int64))
                np.save(dest / f"knn_{r}_dist.npy", D[:, :k].astype(np.float32))
                np.save(dest / f"knn_{r}_qids.npy", q_ids.astype(np.int64))

                stats["search_seconds"] = round(search_s, 2)
                stats["queries_per_second"] = round(len(q) / max(search_s, 1e-9), 1)
                per_role[r] = stats
                checks[r] = "PASS"
                tie = stats["boundary_ties"]
                print(f"    {r}: {len(q):,} queries in {_fmt(search_s)} "
                      f"({stats['queries_per_second']:,.0f} q/s)  "
                      f"d@1={stats['mean_dist_at_1']:.4f} "
                      f"d@{k}={stats['mean_dist_at_k']:.4f}"
                      + (f"  boundary ties {tie['frac']:.3%}" if tie else ""))

            json.dump({
                "task": "T6 exact-KNN labelling",
                "created": datetime.now().isoformat(),
                "model": args.model,
                "pairing": p,
                "pairing_short": PAIRINGS[p]["short"],
                "query_modality": qmod,
                "index_modality": imod,
                "same_modality": PAIRINGS[p]["same_modality"],
                "k": k, "k_probe": k_probe,
                "dim": int(d),
                "metric": METRIC,
                "metric_note": (
                    "Squared L2, as FAISS returns it -- same convention as T5's "
                    "d1/d2. Vectors are unit-norm so L2-argmin orders identically "
                    "to cosine, cross-modally included. T7 must search under this "
                    "same metric: CF@k intersects these lists with T7's, and a "
                    "mismatch yields a number rather than an error."
                ),
                "search": {
                    "structure": f"IndexFlat{METRIC}",
                    "exhaustive": True,
                    "device": where,
                    "batch": args.batch,
                    "a_load_seconds": round(load_s, 2),
                    "faiss_version": faiss.__version__,
                    "threads": faiss.omp_get_max_threads(),
                },
                "embedding_roots": {m: str(r) if r else "default (result/embeddings_*)"
                                    for m, r in roots.items()},
                "corpus": {
                    "role": "A", "n": int(len(a_ids)),
                    "id_min": int(a_ids[0]), "id_max": int(a_ids[-1]),
                    "source": str(pilot / "A_ids.npy"),
                    "scope": json.load(open(pilot / "pair_index_meta.json"))["scope"],
                },
                "roles": per_role,
                "checks": {
                    "self_retrieval": selfcheck,
                    "no_sentinel_labels": "PASS",
                    "distances_sorted": "PASS",
                    "no_B_or_C_ids_returned": "PASS",
                    "no_query_retrieved_itself": "PASS",
                    "per_role": checks,
                },
                "files": {
                    f"knn_{{{','.join(args.roles)}}}_ids.npy":
                        f"int64 [n_queries, {k}] -- canonical pilot int_ids, "
                        f"rank order, the deliverable",
                    f"knn_{{{','.join(args.roles)}}}_dist.npy":
                        f"float32 [n_queries, {k}] -- SQUARED L2, ascending",
                    f"knn_{{{','.join(args.roles)}}}_qids.npy":
                        "int64 [n_queries] -- row i of the two arrays above is "
                        "the result for this int_id",
                },
                "contract": (
                    "Row i of knn_<role>_ids.npy is the exact top-k of pilot "
                    "int_id knn_<role>_qids.npy[i], over role A's vectors in the "
                    f"{imod} modality. Ids are the same space T4's add_with_ids "
                    "used, which is what makes the T8 intersection meaningful. "
                    "Never reorder or recompact these arrays."
                ),
                "reuse": (
                    "All 8 index configurations at this pairing share these "
                    "labels. Exact top-k depends on the query vectors, the "
                    "corpus vectors and the metric -- not on any index -- so "
                    "recomputing per config would be waste, and computing it "
                    "twice risks two versions of ground truth."
                ),
                "not_an_ann_index": (
                    "IndexFlat is an exhaustive scan with no training, no graph "
                    "and no codebook. It is not a ninth catalogue entry and its "
                    "results are the reference T7's 8 configs are graded against."
                ),
                "degenerate_pairs": (
                    "The 353 excluded pairs were dropped at T2 labelling and hold "
                    "no int_id, so they entered neither A nor B nor C. Their rows "
                    "remain physically present in the .npy files at their original "
                    "positions; they are simply never requested. Inherited."
                ),
                "smoke": {"synthetic": args.synthetic or None,
                          "limit_a": args.limit_a or None,
                          "limit_q": args.limit_q or None} if smoke else None,
            }, open(dest / "exact_knn_meta.json", "w"), indent=2)
            print(f"    -> {dest}")

        del index, flat, avecs

    print(f"\ntotal wall {_fmt(time.time() - t_all)}")
    if smoke:
        print("SMOKE TEST -- the labels written above are not real ground truth.")


if __name__ == "__main__":
    main()
