"""
Post-build verification for T4. Run after build_indexes.py, before T7.

    python verify_indexes.py --model siglip2 --modality image

The deliverable is "9 built indexes, each queryable, each preserving the
original row-ID space". Only the first of those is self-evident from a file
existing; the other two are what this script actually checks.

The failure this guards against has no error message. If out[i] were paired
with the wrong ID at build time, every index would still build, still search,
still return plausible neighbours -- and every CF@k in T8 would be computed
against the wrong ground truth.

The gate is RECONSTRUCT, not self-retrieval. reconstruct(id) returns the vector
stored under that id, so comparing it to the vector we meant to store tests the
pairing directly, with nothing else in the path. Self-retrieval -- query with a
vector that is in the index, expect its own id back -- tests the same pairing
THROUGH graph traversal and cell pruning, so a perfectly correct index built at
a deliberately weak operating point fails it. hnsw_low (M=8, efSearch=16) has a
reference CF@100 of 50-56%; demanding it self-retrieve at 98% would be asking a
configuration that exists to be bad to behave well. It is kept as a reported
diagnostic with a floor that only trips on genuine scrambling.

Checks, per index:
  1. ntotal == |A|
  2. reconstruct       -- GATE. bit-exact for HNSW; for PQ, each reconstruction
                          must match its own original better than any other's
  3. self-retrieval    -- diagnostic; floor 0.5 vs chance of k/|A| ~ 1.2e-4
  4. no B or C ids     -- nothing outside A was ever added
  5. honest queries    -- a real held-out B query returns only A ids, and
                          never its own id (B is not in the index)
  6. search params     -- efSearch / nprobe survived serialisation
"""

import argparse
import json
from pathlib import Path

import numpy as np

import faiss

from index_configs import configs_for, MODEL_DIMS, METRIC
from load_vectors import load_selected

REPO_ROOT = Path(__file__).resolve().parent.parent


def read_search_param(index, cfg):
    """Pull the serialised search-time parameter back out of a loaded index."""
    if cfg.is_hnsw:
        inner = faiss.downcast_index(index.index)
        return "efSearch", int(inner.hnsw.efSearch)
    return "nprobe", int(faiss.extract_index_ivf(index).nprobe)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="siglip2")
    ap.add_argument("--modality", required=True, choices=["image", "text"])
    ap.add_argument("--pilot-dir", type=Path, default=None)
    ap.add_argument("--manifest", type=Path,
                    default=REPO_ROOT / "result/corpus_shard_manifest.json")
    ap.add_argument("--index-dir", type=Path, default=None)
    ap.add_argument("--n-probe-queries", type=int, default=512,
                    help="how many A rows to re-query for the self-retrieval test")
    ap.add_argument("--n-b-queries", type=int, default=256)
    ap.add_argument("--k", type=int, default=100, help="CF is defined at k=100")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pilot_dir = args.pilot_dir or (REPO_ROOT / "result/pilot_1M/siglip2")
    index_dir = args.index_dir or (REPO_ROOT / "result/indexes" / args.model / args.modality)
    catalogue = configs_for(args.model)

    print("=" * 78)
    print(f"T4 verification -- model={args.model} index modality={args.modality}")
    print(f"  indexes {index_dir}")
    print("=" * 78)

    # Verification needs a few hundred vectors, not all 850,393 of A -- the
    # id-set checks below run off the *_ids.npy arrays alone. So no full pass
    # over the corpus happens here.
    a_ids = np.load(pilot_dir / "A_ids.npy")
    a_set = set(a_ids.tolist())

    b_ids = np.load(pilot_dir / "B_ids.npy")
    c_ids = np.load(pilot_dir / "C_ids.npy")
    forbidden = set(b_ids.tolist()) | set(c_ids.tolist())
    print(f"  |A| {len(a_ids):,}   |B| {len(b_ids):,}   |C| {len(c_ids):,}")

    rng = np.random.default_rng(args.seed)

    # Probes: rows already in the index, with their known ids.
    probe_ids = np.sort(rng.choice(a_ids, size=min(args.n_probe_queries, len(a_ids)),
                                   replace=False))
    probe_vecs = np.ascontiguousarray(
        load_selected(args.model, args.modality, probe_ids,
                      pilot_dir / "pair_index.csv"))
    if probe_vecs.shape[1] != MODEL_DIMS[args.model]:
        raise RuntimeError(f"expected dim {MODEL_DIMS[args.model]}, "
                           f"got {probe_vecs.shape[1]}")

    # Honest queries: real B vectors, which are in no index.
    b_sub = np.sort(rng.choice(b_ids, size=min(args.n_b_queries, len(b_ids)),
                               replace=False))
    b_vecs = np.ascontiguousarray(
        load_selected(args.model, args.modality, b_sub,
                      pilot_dir / "pair_index.csv"))
    print(f"  probes: {len(probe_ids)} A rows, {len(b_sub)} B queries, k={args.k}\n")

    results, failures = [], []
    for cfg in catalogue:
        path = index_dir / f"{cfg.name}.faiss"
        if not path.exists():
            print(f"[{cfg.name}] MISSING {path.name}")
            failures.append(f"{cfg.name}: index file missing")
            continue

        index = faiss.read_index(str(path))
        r = {"name": cfg.name, "family": cfg.family, "ntotal": int(index.ntotal)}
        bad = []

        # 1. every A vector, and nothing else
        if index.ntotal != len(a_ids):
            bad.append(f"ntotal {index.ntotal:,} != |A| {len(a_ids):,}")

        # 5. search parameter survived write/read
        pname, pval = read_search_param(index, cfg)
        r["search_param"] = {pname: pval}
        want = cfg.search[pname]
        # FAISS clamps nprobe to nlist on read-back; that is expected for the
        # two configs whose nprobe is deliberately >= nlist.
        if pval != want and not (pname == "nprobe" and pval == min(want, cfg.build["nlist"])):
            bad.append(f"{pname} is {pval}, expected {want}")

        # 2. reconstruct -- THE ID-space gate.
        #
        # reconstruct(id) returns the vector stored under that id. Comparing it
        # to the vector we intended to store tests the pairing directly, with
        # nothing else in the path. Self-retrieval (below) tests the same thing
        # through graph traversal and cell pruning, so a correct index built at
        # a deliberately weak operating point fails it -- which says nothing
        # about the IDs. That distinction is why this is the gate and that is
        # only a diagnostic.
        try:
            if not cfg.is_hnsw:
                # NOT make_direct_map(): that builds a DirectMap::Array, which
                # requires ids to be exactly 0..ntotal-1 and throws
                # "direct map supported only for sequential ids" on ours. Our
                # ids ARE deliberately non-sequential -- the pilot int_ids,
                # sparse in 0..999,999 because B and C leave gaps -- so the
                # hashtable variant is the correct one.
                #
                # Worth noting what the Array failure actually proved: it fired
                # because the sparse id space survived the build. Had the index
                # silently renumbered its rows to 0..850,392 -- the exact defect
                # this check exists to catch -- Array would have succeeded.
                ivf = faiss.downcast_index(faiss.extract_index_ivf(index))
                ivf.set_direct_map_type(faiss.DirectMap.Hashtable)
            recon = np.vstack([index.reconstruct(int(i)) for i in probe_ids])
        except Exception as e:
            recon = None
            bad.append(f"reconstruct unavailable: {type(e).__name__}: {e}")

        if recon is not None:
            if cfg.is_hnsw:
                # IndexHNSWFlat stores full vectors, so this must be exact.
                exact = bool(np.array_equal(recon, probe_vecs))
                r["reconstruct_exact"] = exact
                if not exact:
                    worst = float(np.abs(recon - probe_vecs).max())
                    bad.append(f"reconstruct not bit-exact (max |diff| {worst:.3e}) "
                               f"-- vectors are stored against the wrong ids")
            else:
                # PQ decodes lossily, so require instead that each
                # reconstruction resembles ITS OWN original more than any other
                # probe's. A permuted id map fails this immediately: it would
                # score ~1/n, not ~1.
                n = len(probe_ids)
                sim = recon @ probe_vecs.T
                best = np.argmax(sim, axis=1)
                own = float(np.mean(best == np.arange(n)))
                cos = float(np.mean(np.sum(recon * probe_vecs, axis=1)))

                # LAION contains duplicate and near-duplicate images. Where two
                # originals are nearly the same vector, a lossy reconstruction
                # can land marginally closer to the twin -- a fact about the
                # corpus, not about the id mapping. So a mismatch only counts
                # against the index if the winner is NOT a near-duplicate of
                # the true original.
                mism = np.flatnonzero(best != np.arange(n))
                dup = np.array([
                    float(probe_vecs[i] @ probe_vecs[best[i]]) > 0.99 for i in mism
                ], dtype=bool) if len(mism) else np.zeros(0, dtype=bool)
                unexplained = int((~dup).sum())
                own_or_dup = float((n - unexplained) / n)

                r["reconstruct_matches_own"] = own
                r["reconstruct_own_or_duplicate"] = own_or_dup
                r["reconstruct_cosine"] = cos
                r["reconstruct_mismatches"] = int(len(mism))
                r["reconstruct_mismatches_are_duplicates"] = int(dup.sum())

                # Chance is 1/n ~ 0.002, so anything near 1 means the mapping
                # is intact. The floor is loose deliberately -- this gate is
                # about id integrity, not quantisation quality.
                if own_or_dup < 0.98:
                    bad.append(
                        f"reconstruct matches its own vector for only "
                        f"{own_or_dup:.3f} of probes ({unexplained} mismatches "
                        f"not explained by near-duplicates; chance is {1/n:.4f}) "
                        f"-- id map looks permuted")
                if cos < 0.5:
                    bad.append(f"reconstruct cosine {cos:.3f} too low even for PQ")

        # 3. self-retrieval -- DIAGNOSTIC ONLY, not a quality gate.
        #
        # Querying with a vector that is in the index and asking for its own id
        # back exercises the search output path, so it catches an index that
        # returns internal offsets instead of mapped ids. It does NOT measure
        # correctness: HNSW traverses approximately however exactly it stores,
        # so hnsw_low (M=8, efSearch=16, reference CF@100 50-56%) is expected to
        # miss itself often. The floor here only trips on genuine scrambling --
        # chance is k/|A|, about 1.2e-4.
        _, I = index.search(probe_vecs, args.k)
        rank1 = int((I[:, 0] == probe_ids).sum())
        in_topk = int(np.array([pid in row for pid, row in zip(probe_ids, I)]).sum())
        r["self_rank1"] = rank1 / len(probe_ids)
        r["self_in_topk"] = in_topk / len(probe_ids)
        r["self_chance"] = args.k / len(a_ids)
        if r["self_in_topk"] < 0.5:
            bad.append(f"self-retrieval in-top{args.k} {r['self_in_topk']:.3f} < 0.5 "
                       f"(chance {r['self_chance']:.2e}) -- id map may be scrambled")

        # Duplicate diagnostic: LAION repeats images, and a vector with more
        # than k exact duplicates cannot reliably find itself -- top-k fills
        # with equidistant twins. Distinguishes that from a real defect.
        D, _ = index.search(probe_vecs, 1)
        missed = np.array([pid not in row for pid, row in zip(probe_ids, I)])
        if missed.any():
            top1_is_tie = int(np.sum(np.abs(D[missed, 0]) < 1e-5))
            r["missed_probes"] = int(missed.sum())
            r["missed_with_zero_distance_top1"] = top1_is_tie

        # 3. nothing outside A is reachable
        returned = set(int(x) for x in I.ravel() if x >= 0)
        leaked = returned & forbidden
        outside = returned - a_set
        r["ids_leaked_from_BC"] = len(leaked)
        r["ids_outside_A"] = len(outside)
        if leaked:
            bad.append(f"{len(leaked)} returned ids belong to B/C: {sorted(leaked)[:5]}")
        if outside:
            bad.append(f"{len(outside)} returned ids are not in A: {sorted(outside)[:5]}")

        # 4. honest held-out queries
        _, IB = index.search(b_vecs, args.k)
        rb = set(int(x) for x in IB.ravel() if x >= 0)
        self_hits = int((IB == b_sub[:, None]).sum())
        r["b_ids_outside_A"] = len(rb - a_set)
        r["b_self_hits"] = self_hits
        if rb - a_set:
            bad.append(f"B query returned {len(rb - a_set)} non-A ids")
        if self_hits:
            bad.append(f"B query retrieved itself {self_hits}x -- B is in the index!")

        status = "ok" if not bad else "FAIL"
        if r.get("reconstruct_exact"):
            rec_str = "exact"
        elif "reconstruct_own_or_duplicate" in r:
            rec_str = (f"own={r['reconstruct_matches_own']:.3f} "
                       f"+dup={r['reconstruct_own_or_duplicate']:.3f} "
                       f"cos={r['reconstruct_cosine']:.3f}")
        else:
            rec_str = "unavailable"
        print(f"[{cfg.name}] {status}  ntotal={index.ntotal:,}  {pname}={pval}  "
              f"reconstruct[{rec_str}]  "
              f"self rank1={r['self_rank1']:.3f} top{args.k}={r['self_in_topk']:.3f}")
        for b in bad:
            print(f"    !! {b}")
            failures.append(f"{cfg.name}: {b}")

        r["status"] = status
        results.append(r)
        del index

    out = index_dir / "verify_report.json"
    out.write_text(json.dumps({
        "model": args.model, "index_modality": args.modality,
        "metric": METRIC, "k": args.k,
        "n_A": len(a_ids), "n_probe": len(probe_ids), "n_b_queries": len(b_sub),
        "seed": args.seed,
        "checks": results, "failures": failures,
        "verdict": "PASS" if not failures and len(results) == len(catalogue) else "FAIL",
    }, indent=2))

    print(f"\nreport -> {out}")
    if failures or len(results) != len(catalogue):
        print(f"VERDICT: FAIL ({len(failures)} problem(s), "
              f"{len(results)}/{len(catalogue)} indexes present)")
        raise SystemExit(1)
    print(f"VERDICT: PASS -- {len(results)}/{len(catalogue)} indexes queryable, "
          f"ID space intact")


if __name__ == "__main__":
    main()
