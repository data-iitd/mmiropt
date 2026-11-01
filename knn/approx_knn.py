"""
T7: approximate-KNN -- every query in B and C, through all 9 configurations.

    python approx_knn.py --model siglip2 --pairing all --threads 128

For each (query, configuration, pairing) the configuration's own top-100 over A.
T8 intersects these lists with T6's exact ones; nothing else in the pipeline
reads them.

DEPENDS ONLY ON T4
    The nine indexes existing, for both index modalities. It does NOT depend on
    T6: the query set and its order come from the split (T3), not from the
    labels. So T7 can run on a pairing whose exact-KNN pass is still going, and
    T8 picks the two up afterwards. Where T6 labels DO already exist, their
    qids are read and asserted equal to the split's -- row i must mean the same
    query in both files, because T8 zips them by position and never joins.

SEARCH PARAMETERS ARE NOT SET HERE, EVER
    FAISS serialises efSearch and nprobe into the index file, so a written index
    already carries exactly what index_configs assigned it and build_indexes
    stamped in. This script reads them back and records them; assigning them is
    how a config silently stops being the config it claims to be. If a value
    looks wrong, fix the catalogue and rebuild the index -- not the search.

    V2 EXCEPTION (--params-json, the second run in result_v2/): the v2 files
    carry only the build's starting value, and the user chose the search values
    at 5.8M afterwards (final_params_chosen.json; no rebuild, user 2026-10-14).
    So in v2 the chosen value is SET once after the load, read back, asserted
    and recorded with its source. The configs come from the v2 build.json files
    (indexes/v2_dirs.py configs_v2), the queries from the v2 embedding trees.

    python knn/approx_knn.py --model siglip2 --pairing all --configs hnsw_low \
        --params-json result_v2/param_check/7M_cleaned/final_params_chosen.json \
        --index-root result_v2/indexes/7M_cleaned --pilot-dir result_v2/7M_cleaned/siglip2 \
        --manifest result/corpus_shard_manifest.json \
        --knn-dir result_v2/knn/7M_cleaned --out-dir result_v2/approx_knn/7M_cleaned

QUERY CACHE (--query-cache DIR, v2 run)
    Every per-config job would otherwise read all 2,875 shards again for the
    same 1.02M query vectors (~23 min under load, most of a cheap job).
    --build-query-cache loads them once, exactly as below (load_role_matrix,
    all roles asked for, in all_q_ids order), writes
    DIR/<model>_<modality>_<roles>.npy (+ _ids.npy, .json) and exits. With
    --query-cache alone a job reads that file instead, after asserting its ids
    equal this run's all_q_ids and its shape and embedding root match; if the
    file is missing it reads the shards as before.

IDS
    No gather here, unlike T6. Every index was populated with add_with_ids using
    the canonical pilot int_ids (HNSW via IndexIDMap2, the IVF families
    natively), so the labels FAISS returns ARE those ints. The one thing that
    must not be waved through is -1: FAISS pads a short result with it, and
    unlike T6 an approximate search can genuinely come up short. -1 is stored as
    -1 -- never gathered, never renumbered -- and T8 counts it as a miss.

WHAT IS CHECKED BEFORE ANYTHING IS WRITTEN
    Every returned id lies in A; none lies in B or C; no query retrieves itself;
    no id appears twice in one list; distances are non-decreasing. A stale or
    misaligned list is still a valid int64 array of plausible ids -- it fails
    nowhere and poisons every CF in T8 -- so it is checked at the point of
    writing rather than trusted.

COST -- READ THIS BEFORE SIZING THE JOB
    This is NOT cheaper than T6 here, and that is a property of the catalogue
    rather than a bug. T6 scans A exhaustively but does it on an A100 in fp32;
    ivfpq_high and oivfpq_high probe 8192/32768 and 16384/32768 cells -- a
    quarter and a half of the corpus -- against 288/384-byte PQ codes, on CPU,
    because FAISS's GPU IVFPQ supports neither those code lengths nor an nprobe
    that large. Those two configs are ~85% of the wall time. See knn/README.md.
"""

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

# T4's loader and catalogue, imported rather than reimplemented. The row-to-ID
# mapping is the one thing in this pipeline that fails without an error message.
sys.path.insert(0, str(REPO_ROOT / "indexes"))
from index_configs import METRIC, MODEL_DIMS, configs_for, describe   # noqa: E402
from load_vectors import load_included_shards, load_role_matrix       # noqa: E402
from verify_indexes import read_search_param                          # noqa: E402

from exact_knn import (                                               # noqa: E402
    DEFAULT_MANIFEST, DEFAULT_PILOT, PAIRINGS, load_pair_index_roles,
)

DEFAULT_KNN = REPO_ROOT / "result/knn/1M"              # T6 labels, read-only
DEFAULT_OUT = REPO_ROOT / "result/approx_knn/1M"


def _fmt(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def out_dir(root: Path, model: str, pairing: str) -> Path:
    """result/approx_knn/1M/<model>/<pairing>/ -- model first, pairing second."""
    return root / model / pairing


def read_structure(index) -> dict:
    """
    What an index file actually IS, read from the file rather than its name,
    in the catalogue's own build-parameter keys.

    HNSW: M (level 0 holds 2M links, so M = nb_neighbors(0) // 2) and
    efConstruction. IVF families: nlist, PQ m and nbits, and whether a
    rotation sits in front ("rotation").

    OPQ's M is NOT recoverable from a file. In this FAISS build an OPQMatrix
    read back from disk is a plain d x d LinearTransform, so the file records
    that a rotation exists but not how it was trained. The check therefore
    verifies the rotation's presence and shape; opq_M itself is taken on trust
    from the catalogue.
    """
    out = {"d": int(index.d), "rotation": False}
    top = faiss.downcast_index(index)
    if isinstance(top, faiss.IndexIDMap):
        inner = faiss.downcast_index(top.index)
        if isinstance(inner, faiss.IndexHNSW):
            out["M"] = int(inner.hnsw.nb_neighbors(0)) // 2
            out["efConstruction"] = int(inner.hnsw.efConstruction)
            return out
    if isinstance(top, faiss.IndexPreTransform):
        vt = faiss.downcast_VectorTransform(top.chain.at(0))
        if top.chain.size() == 1 and isinstance(vt, faiss.LinearTransform) \
                and vt.d_in == vt.d_out == out["d"]:
            out["rotation"] = True
            if isinstance(vt, faiss.OPQMatrix):      # only if the type survived
                out["opq_M"] = int(vt.M)
        else:
            out["rotation"] = f"unexpected pre-transform chain ({type(vt).__name__})"
    ivf = faiss.downcast_index(faiss.extract_index_ivf(index))
    out["nlist"] = int(ivf.nlist)
    if isinstance(ivf, faiss.IndexIVFPQ):
        out["m"] = int(ivf.pq.M)
        out["nbits"] = int(ivf.pq.nbits)
    return out


def assert_matches_catalogue(index, cfg, path: Path, dim: int) -> dict:
    """
    Refuse an index whose structure is not the catalogue's for this config.

    Checking the search parameter alone was enough while a config's file never
    changed identity. It is not enough once tiers are derived by copying one
    file over another: an old IVF4096/PQ72 file re-stamped to the new nprobe
    would carry the right nprobe and be the wrong index, and every CF computed
    from it would be wrong with no error. So the build parameters are compared
    too, in both directions -- an OPQ file under an IVFPQ name fails as surely
    as a missing rotation does.
    """
    got = read_structure(index)
    want = {**cfg.build, "d": int(dim), "rotation": "opq_M" in cfg.build}
    if "opq_M" not in got:
        # Not recorded in the file (see read_structure); compare the rotation's
        # presence instead of its M.
        want.pop("opq_M", None)
    bad = {k: (got.get(k), want.get(k))
           for k in sorted(set(got) | set(want)) if got.get(k) != want.get(k)}
    if bad:
        detail = ", ".join(f"{k}: file={g} catalogue={w}" for k, (g, w) in bad.items())
        raise RuntimeError(
            f"{path}: index structure does not match catalogue entry "
            f"{cfg.name!r} ({detail}). The file is not the configuration its name "
            f"claims -- re-derive or rebuild it before searching.")
    return got


def write_json_atomic(path: Path, obj) -> None:
    """Write to a private temp file, then rename -- a reader never sees half."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def write_fragment(dest: Path, name: str, record: dict) -> None:
    """One config's record, in a file only the job running that config writes."""
    parts = dest / "meta_parts"
    parts.mkdir(exist_ok=True)
    write_json_atomic(parts / f"{name}.json", record)


def merged_records(dest: Path, start_records: dict) -> dict:
    """
    Every config record for one pairing, from every job that has written there.

    Several jobs may split one pairing's configs between them and finish in any
    order. Each writes its configs' fragments as it goes, so the full picture is
    the union of: this run's start-time records, then the meta currently on disk
    (another job may have rewritten it since this run started), then every
    fragment -- later sources winning, since a fragment is written at the moment
    its config finished.

    Two filters keep the result honest about what is on disk:
      * a config with no output file is dropped (its results were moved aside);
      * a record older than its own output file is NOT current -- the file was
        rewritten by a run that never recorded itself (e.g. an old-code job that
        stopped before its end-of-job meta write). Such records are returned
        separately so the caller can archive them, never discarded.

    Returns (current, superseded).
    """
    recs = dict(start_records)
    superseded = []

    def when(r):
        try:
            return datetime.fromisoformat(r["written"])
        except (KeyError, TypeError, ValueError):
            return datetime.min

    def take(r, source):
        # Two different records for one config (two runs): the newer one is
        # current and the other is archived -- whichever order the sources are
        # read in. Nothing is dropped.
        old = recs.get(r["config"])
        if old is None or old.get("written") == r.get("written"):
            recs[r["config"]] = r
            return
        newer, older = (r, old) if when(r) > when(old) else (old, r)
        superseded.append({**older, "superseded_reason":
                           f"a newer record for this config exists "
                           f"(written {newer.get('written')}); seen via {source}"})
        recs[r["config"]] = newer

    mp = dest / "approx_knn_meta.json"
    if mp.exists():
        try:
            for r in json.load(open(mp)).get("configs", []):
                take(r, "the meta on disk")
        except (json.JSONDecodeError, OSError):
            pass        # mid-write by an old-code job; start records cover it
    parts = dest / "meta_parts"
    if parts.is_dir():
        for f in sorted(parts.glob("*.json")):
            try:
                take(json.load(open(f)), f"meta_parts/{f.name}")
            except (json.JSONDecodeError, OSError, KeyError):
                pass
    current = {}
    for n, r in recs.items():
        files = [dest / f"approx_{ro}_{n}_ids.npy" for ro in ("B", "C")]
        files = [f for f in files if f.exists()]
        if not files:
            continue
        newest = max(f.stat().st_mtime for f in files)
        try:
            written = datetime.fromisoformat(r["written"]).timestamp()
        except (KeyError, ValueError):
            written = None
        # A record is written just after its last role file, so a file more
        # than a minute newer than the record was produced by some other run.
        if written is not None and newest - written > 60:
            superseded.append({**r, "superseded_reason":
                               f"output file rewritten at "
                               f"{datetime.fromtimestamp(newest).isoformat()} after "
                               f"this record ({r['written']})"})
        else:
            current[n] = r
    return current, superseded


def grade(D, I, a_ids, q_ids, bc_ids, k):
    """
    Everything that must hold before an approximate list is allowed to be written.

    Returns (stats, ) and raises on anything that would make T8's intersection
    meaningless. -1 is tolerated -- an approximate search may legitimately return
    fewer than k -- but it is counted, because a config quietly returning half a
    list looks exactly like a config with poor fidelity once CF is computed.
    """
    n = I.shape[0]
    if I.shape != (n, k) or D.shape != (n, k):
        raise RuntimeError(f"expected [{n}, {k}] results, got I{I.shape} D{D.shape}")

    valid = I >= 0
    n_short = int((~valid).sum())

    flat = I[valid]
    if flat.size:
        pos = np.minimum(np.searchsorted(a_ids, flat), len(a_ids) - 1)
        outside = int((a_ids[pos] != flat).sum())
        if outside:
            bad = flat[a_ids[pos] != flat][:5].tolist()
            raise RuntimeError(
                f"{outside:,} returned ids are not in A (e.g. {bad}). The index "
                f"was built over a different id space than the split defines."
            )

    # Implied by the above -- A and B/C are disjoint -- so this is really a check
    # on the split, kept because it is what actually inflates fidelity if wrong.
    leaked = np.intersect1d(np.unique(flat), bc_ids)
    if leaked.size:
        raise RuntimeError(
            f"{leaked.size:,} returned ids lie in B or C (e.g. {leaked[:5].tolist()}). "
            f"A query role leaked into the index."
        )

    self_hit = int((I == q_ids[:, None]).sum())
    if self_hit:
        raise RuntimeError(
            f"{self_hit:,} queries retrieved their own int_id: the query role is "
            f"inside the index, which inflates every CF computed from this."
        )

    # The same corpus row twice in one list would be counted twice by T8's
    # intersection and inflate CF above what the index actually delivered.
    srt = np.sort(I, axis=1)
    dup = int(((np.diff(srt, axis=1) == 0) & (srt[:, :-1] >= 0)).sum())
    if dup:
        raise RuntimeError(f"{dup:,} duplicate ids inside individual result lists")

    # Top-k is a prefix, not a bag. Only full rows are checked: FAISS pads the
    # tail of a short result with a sentinel distance that need not be ordered.
    full = valid.all(axis=1)
    if full.any():
        dd = np.diff(D[full], axis=1)
        if float(dd.min()) < -1e-3:
            raise RuntimeError(
                f"distances not non-decreasing (min delta {float(dd.min()):.6f})")

    d_at_1 = D[:, 0][valid[:, 0]]
    d_at_k = D[:, k - 1][valid[:, k - 1]]
    return {
        "n_queries": int(n), "k": int(k),
        "short_slots": n_short,
        "short_slot_frac": float(n_short) / float(n * k),
        "rows_with_short_result": int((~valid).any(axis=1).sum()),
        "self_retrievals": 0, "leaked_bc_ids": 0, "duplicate_ids": 0,
        "mean_dist_at_1": float(d_at_1.mean()) if d_at_1.size else None,
        "mean_dist_at_k": float(d_at_k.mean()) if d_at_k.size else None,
    }


def search_all(index, q, k, batch, label=""):
    """Batched search. Returns (D, I) of whatever the index reports, ids as ids."""
    n = q.shape[0]
    D = np.empty((n, k), dtype=np.float32)
    I = np.empty((n, k), dtype=np.int64)
    t0 = time.time()
    for s in range(0, n, batch):
        e = min(s + batch, n)
        d, i = index.search(np.ascontiguousarray(q[s:e]), k)
        D[s:e], I[s:e] = d, i
        rate = e / max(time.time() - t0, 1e-9)
        print(f"      {label} {e:,}/{n:,}  {rate:,.0f} q/s  "
              f"eta {_fmt((n - e) / max(rate, 1e-9))}   ", end="\r", flush=True)
    print()
    return D, I, time.time() - t0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="siglip2", choices=sorted(MODEL_DIMS))
    ap.add_argument("--pairing", nargs="+", default=["all"],
                    choices=sorted(PAIRINGS) + ["all"],
                    help="query->index pairings to search (default: all four)")
    ap.add_argument("--roles", nargs="+", default=["B", "C"], choices=["B", "C"])
    ap.add_argument("--configs", default="all",
                    help="comma-separated config names, or 'all'. The two "
                         "'high' IVF tiers are ~85%% of the wall time -- run them "
                         "in their own job if you want the rest early.")
    ap.add_argument("--k", type=int, default=100, help="CF is defined at k=100")
    ap.add_argument("--batch", type=int, default=4096, help="queries per search call")
    ap.add_argument("--pilot-dir", type=Path, default=DEFAULT_PILOT,
                    help="the SHARED pilot -- always siglip2/, see pilot_1M/README.md")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--index-dir", type=Path, default=None,
                    help="default result/indexes/<model>/<index modality>")
    ap.add_argument("--knn-dir", type=Path, default=DEFAULT_KNN,
                    help="T6 labels, read only to cross-check query order and to "
                         "report the exact-vs-approx speed ratio")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--threads", type=int, default=0, help="0 = leave FAISS default")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit-q", type=int, default=0,
                    help="smoke test: first N queries of each role")
    ap.add_argument("--allow-login-node", action="store_true",
                    help="run full size outside PBS. Don't.")
    ap.add_argument("--params-json", type=Path, default=None,
                    help="v2: chosen search values (final_params_chosen.json); configs "
                         "from the v2 build.json files, queries from the v2 embeddings")
    ap.add_argument("--index-root", type=Path, default=None,
                    help="v2: <root>/<model>/<index modality>/<config>.faiss")
    ap.add_argument("--query-cache", type=Path, default=None,
                    help="read the query matrices from this directory when present")
    ap.add_argument("--build-query-cache", action="store_true",
                    help="load the queries from the shards, write them to --query-cache, exit")
    args = ap.parse_args()
    if args.build_query_cache and args.query_cache is None:
        raise SystemExit("--build-query-cache needs --query-cache DIR")

    pairings = sorted(PAIRINGS) if "all" in args.pairing else list(dict.fromkeys(args.pairing))
    k = args.k

    smoke = bool(args.limit_q)
    if not smoke and not os.environ.get("PBS_JOBID") and not args.allow_login_node:
        raise SystemExit(
            "Refusing to run full size outside a PBS job: this loads the query "
            "matrices (a full pass over 2,875 shards per modality) and reads "
            "multi-GB index files. Submit a job, or pass --limit-q for a smoke test."
        )
    if args.threads:
        faiss.omp_set_num_threads(args.threads)

    v2 = args.params_json is not None
    if v2:
        if args.index_root is None or args.index_dir is not None:
            raise SystemExit("--params-json needs --index-root (and not --index-dir)")
        from v2_dirs import configs_v2, embed_dir                      # noqa: E402
        catalogue = configs_v2(args.model, args.params_json, args.index_root)
    else:
        catalogue = configs_for(args.model)
    by_name = {c.name: c for c in catalogue}
    names = [c.name for c in catalogue] if args.configs == "all" else args.configs.split(",")
    unknown = [n for n in names if n not in by_name]
    if unknown:
        raise SystemExit(f"unknown config(s): {unknown}\n\n{describe(args.model)}")

    pilot = args.pilot_dir
    role_ids = {r: np.load(pilot / f"{r}_ids.npy") for r in args.roles}
    a_ids = np.load(pilot / "A_ids.npy")
    bc_ids = np.sort(np.concatenate([role_ids[r] for r in args.roles]))
    if len(np.unique(bc_ids)) != len(bc_ids):
        raise RuntimeError("roles overlap at the pair level -- B and C must be disjoint")
    if np.intersect1d(bc_ids, a_ids).size:
        raise RuntimeError("query ids intersect A -- the split is not disjoint")

    print("=" * 78)
    print(f"T7 approximate-KNN -- model={args.model}")
    pretty = ", ".join(f"{p} ({PAIRINGS[p]['short']})" for p in pairings)
    print(f"  pairings {pretty}")
    print(f"  configs  {', '.join(names)}")
    print(f"  roles    {', '.join(f'{r}={len(role_ids[r]):,}' for r in args.roles)}")
    print(f"  k        {k}")
    print(f"  metric   {METRIC} (as T4 built and T6 labelled -- never re-set here)")
    print(f"  threads  {faiss.omp_get_max_threads()}")
    print(f"  out      {args.out_dir}/{args.model}/<pairing>/")
    print("=" * 78)

    # ---- query vectors: one read per query modality, shared by its pairings --
    all_q_ids = np.sort(np.concatenate([role_ids[r] for r in args.roles]))
    role_slice = {r: np.searchsorted(all_q_ids, role_ids[r]) for r in args.roles}

    q_needed = sorted({PAIRINGS[p]["query_modality"] for p in pairings})
    i_needed = sorted({PAIRINGS[p]["index_modality"] for p in pairings})

    t_all = time.time()
    included = load_included_shards(args.manifest)
    pair_to_int = load_pair_index_roles(pilot / "pair_index.csv", args.roles)
    if len(pair_to_int) != len(all_q_ids):
        raise RuntimeError(
            f"pair_index.csv has {len(pair_to_int):,} rows for {args.roles} but "
            f"the id arrays hold {len(all_q_ids):,}")

    qvecs = {}
    for qmod in q_needed:
        root = embed_dir(args.model, qmod) if v2 else None
        stem = f"{args.model}_{qmod}_{''.join(args.roles)}"
        cf = args.query_cache / f"{stem}.npy" if args.query_cache else None
        if cf is not None and cf.exists() and not args.build_query_cache:
            meta = json.load(open(cf.with_suffix(".json")))
            ids = np.load(args.query_cache / f"{stem}_ids.npy")
            if not np.array_equal(ids, all_q_ids) or meta["embedding_root"] != str(root):
                raise RuntimeError(f"{cf}: ids or embedding root differ from this run -- "
                                   f"rebuild it with --build-query-cache")
            t0 = time.time()
            qvecs[qmod] = np.load(cf, mmap_mode="r")
            if qvecs[qmod].shape != (len(all_q_ids), MODEL_DIMS[args.model]):
                raise RuntimeError(f"{cf}: shape {qvecs[qmod].shape}")
            print(f"\nqueries: {qmod} from the cache {cf} ({time.time() - t0:.1f}s)")
            continue
        print(f"\nloading queries: {qmod}")
        qvecs[qmod] = load_role_matrix(args.model, qmod, all_q_ids, pair_to_int,
                                       included, expect_dim=MODEL_DIMS[args.model],
                                       root=root)
        if args.build_query_cache:
            args.query_cache.mkdir(parents=True, exist_ok=True)
            tmp = args.query_cache / f".{stem}.{os.getpid()}.npy"
            np.save(tmp, np.ascontiguousarray(qvecs[qmod], dtype=np.float32))
            np.save(args.query_cache / f"{stem}_ids.npy", all_q_ids.astype(np.int64))
            write_json_atomic(cf.with_suffix(".json"), {
                "model": args.model, "modality": qmod, "roles": args.roles,
                "n": int(len(all_q_ids)), "dim": int(qvecs[qmod].shape[1]),
                "embedding_root": str(root), "split_dir": str(pilot),
                "manifest": str(args.manifest), "loader": "indexes/load_vectors.py load_role_matrix",
                "created": datetime.now().isoformat(), "pbs_jobid": os.environ.get("PBS_JOBID")})
            os.replace(tmp, cf)
            print(f"wrote {cf} {qvecs[qmod].shape}")
    if args.build_query_cache:
        print(f"query cache built in {_fmt(time.time() - t_all)}")
        return

    # ---- destinations, and the T6 cross-check where labels already exist -----
    dests, exact_meta = {}, {}
    for p in pairings:
        dest = out_dir(args.out_dir, args.model, p)
        dest.mkdir(parents=True, exist_ok=True)
        dests[p] = dest

        labels = args.knn_dir / p / args.model
        mpath = labels / "exact_knn_meta.json"
        if mpath.exists():
            exact_meta[p] = json.load(open(mpath))
            for r in args.roles:
                qf = labels / f"knn_{r}_qids.npy"
                if not qf.exists():
                    continue
                # Row i must mean the same query in both files: T8 zips by
                # position and never joins on the id.
                if not np.array_equal(np.load(qf), role_ids[r]):
                    raise RuntimeError(
                        f"{qf} disagrees with {pilot / (r + '_ids.npy')}. T6 and T7 "
                        f"would be row-misaligned and every CF in T8 would pair a "
                        f"query's exact list with another query's approximate one."
                    )
            print(f"  [{p}] T6 labels present, query order cross-checked")
        else:
            print(f"  [{p}] no T6 labels yet -- T7 does not need them; "
                  f"T8 will wait for them")

        # qids written once per pairing: the alignment is stated on disk rather
        # than inferred from the split's ordering.
        for r in args.roles:
            ids = role_ids[r][:args.limit_q] if args.limit_q else role_ids[r]
            # Several jobs may share this pairing and all write the same qids.
            # Leave an identical file alone; otherwise replace it atomically, so
            # no job ever reads a half-written one.
            qf = dest / f"approx_{r}_qids.npy"
            ids = ids.astype(np.int64)
            if not (qf.exists() and np.array_equal(np.load(qf), ids)):
                tmp = dest / f".approx_{r}_qids.{os.getpid()}.npy"
                np.save(tmp, ids)
                os.replace(tmp, qf)

    # ---- one index load per (index modality, config), shared by its pairings -
    # Prior records are kept even under --overwrite: --overwrite means "redo the
    # configs named in this run", never "forget every other config's record".
    # Dropping them made a partial re-run rewrite the meta without the configs
    # it did not touch, and T8 then refused the pairing.
    records = {p: {} for p in pairings}
    for p in pairings:
        mp = dests[p] / "approx_knn_meta.json"
        if mp.exists():
            records[p] = {r["config"]: r for r in json.load(open(mp)).get("configs", [])}

    for imod in i_needed:
        index_dir = (args.index_root / args.model / imod if v2 else
                     args.index_dir or (REPO_ROOT / "result/indexes" / args.model / imod))
        here = [p for p in pairings if PAIRINGS[p]["index_modality"] == imod]
        print(f"\n{'=' * 78}\nindex side: A / {imod}   ({index_dir})")

        for name in names:
            cfg = by_name[name]

            todo = []
            for p in here:
                need = [r for r in args.roles
                        if args.overwrite or
                        not (dests[p] / f"approx_{r}_{name}_ids.npy").exists()]
                if need:
                    todo.append((p, need))
            if not todo:
                print(f"\n[{name}] all pairings complete, skipping "
                      f"(--overwrite to redo)")
                continue

            path = index_dir / f"{name}.faiss"
            if not path.exists():
                raise SystemExit(f"missing index {path} -- run T4 first")

            t0 = time.time()
            index = faiss.read_index(str(path))
            load_s = time.time() - t0
            if v2:                       # see V2 EXCEPTION in the docstring
                (pn, pv), = cfg.search.items()
                if cfg.is_hnsw:
                    faiss.downcast_index(index.index).hnsw.efSearch = pv
                else:
                    faiss.extract_index_ivf(index).nprobe = pv
            pname, pval = read_search_param(index, cfg)
            if index.ntotal != len(a_ids):
                raise RuntimeError(
                    f"{name}: ntotal {index.ntotal:,} != |A| {len(a_ids):,} -- the "
                    f"index was built over a different split than this one")
            if pval != cfg.search[pname]:
                raise RuntimeError(
                    f"{name}: index carries {pname}={pval} but the catalogue says "
                    f"{cfg.search[pname]}. The file is stale -- rebuild it in T4. "
                    f"T7 must not set search parameters.")
            structure = assert_matches_catalogue(index, cfg, path, MODEL_DIMS[args.model])

            print(f"\n[{name}] {cfg.family}/{cfg.point}  {pname}={pval}  "
                  f"({path.stat().st_size / 2**30:.2f} GiB read in {_fmt(load_s)})")

            for p, need in todo:
                qmod = PAIRINGS[p]["query_modality"]
                print(f"    {p}  ({PAIRINGS[p]['short']})")
                per_role = {}
                for r in need:
                    q = qvecs[qmod][role_slice[r]]
                    q_ids = role_ids[r]
                    if args.limit_q:
                        q, q_ids = q[:args.limit_q], q_ids[:args.limit_q]

                    D, I, search_s = search_all(index, q, k, args.batch, label=f"{r}:")
                    stats = grade(D, I, a_ids, q_ids, bc_ids, k)

                    np.save(dests[p] / f"approx_{r}_{name}_ids.npy", I.astype(np.int64))
                    np.save(dests[p] / f"approx_{r}_{name}_dist.npy", D.astype(np.float32))

                    stats["search_seconds"] = round(search_s, 2)
                    stats["queries_per_second"] = round(len(q) / max(search_s, 1e-9), 1)
                    per_role[r] = stats
                    short = (f"  short slots {stats['short_slot_frac']:.3%}"
                             if stats["short_slots"] else "")
                    print(f"      {r}: {len(q):,} queries in {_fmt(search_s)} "
                          f"({stats['queries_per_second']:,.0f} q/s){short}")

                # The task's own cost question, answered with numbers rather than
                # assumed: how does this config compare to T6's exhaustive pass?
                speedup = None
                em = exact_meta.get(p)
                if em:
                    for r, st in per_role.items():
                        ex = em.get("roles", {}).get(r, {}).get("queries_per_second")
                        if ex:
                            speedup = round(st["queries_per_second"] / ex, 3)
                            verdict = "faster" if speedup > 1 else "SLOWER"
                            print(f"      vs T6 exact ({ex:,.0f} q/s on "
                                  f"{em['search']['device']}): {speedup:.2f}x {verdict}")
                            break

                prev = records[p].get(name, {})
                records[p][name] = {
                    "config": name, "family": cfg.family, "point": cfg.point,
                    "factory": cfg.factory or f"IndexHNSWFlat(M={cfg.build['M']})",
                    "build_params": cfg.build,
                    "search_param": {pname: pval},
                    "search_param_source": (f"set from {args.params_json}" if v2
                                            else "read from the index file"),
                    "index_file": str(path),
                    "index_ntotal": int(index.ntotal),
                    "index_structure_read_from_file": structure,
                    "index_load_seconds": round(load_s, 2),
                    "ref_cf100": cfg.ref_cf100,
                    "roles": {**prev.get("roles", {}), **per_role},
                    "speedup_vs_exact_knn": speedup,
                    "written": datetime.now().isoformat(),
                    "pbs_jobid": os.environ.get("PBS_JOBID"),
                    "host": os.uname().nodename,
                }
                # Written the moment the config finishes, to a file no other job
                # writes. The pairing's meta is assembled from these, so jobs
                # splitting one pairing's configs cannot erase each other.
                write_fragment(dests[p], name, records[p][name])

            del index

    # ---- meta, one file per pairing ----------------------------------------
    for p in pairings:
        qmod, imod = PAIRINGS[p]["query_modality"], PAIRINGS[p]["index_modality"]
        merged, stale = merged_records(dests[p], records[p])
        ordered = [merged[n] for n in [c.name for c in catalogue] if n in merged]
        # Stale records are archived inside the meta, never dropped.
        archived = []
        mp_old = dests[p] / "approx_knn_meta.json"
        if mp_old.exists():
            try:
                archived = json.load(open(mp_old)).get("superseded_records", [])
            except (json.JSONDecodeError, OSError):
                archived = []
        seen = {(a.get("config"), a.get("written")) for a in archived}
        for s in stale:                      # one copy of each archived record
            key = (s.get("config"), s.get("written"))
            if key not in seen:
                seen.add(key)
                archived.append(s)
        if stale:
            print(f"  [{p}] archived {len(stale)} stale record(s): "
                  f"{', '.join(s['config'] for s in stale)}")
        write_json_atomic(dests[p] / "approx_knn_meta.json", {
            "task": "T7 approximate-KNN",
            "created": datetime.now().isoformat(),
            "model": args.model,
            "pairing": p,
            "pairing_short": PAIRINGS[p]["short"],
            "query_modality": qmod,
            "index_modality": imod,
            "same_modality": PAIRINGS[p]["same_modality"],
            "k": k,
            "dim": MODEL_DIMS[args.model],
            "metric": METRIC,
            "metric_note": (
                "Squared L2, as FAISS returns it -- the same convention as T6's "
                "distances and T5's d1/d2, and the same metric T4 built under. "
                "CF@k intersects these lists with T6's; a metric mismatch yields "
                "a number rather than an error."
            ),
            "search_params_note": (
                f"v2: efSearch and nprobe are SET from {args.params_json} after the load "
                f"(the v2 files carry only the build's start value), then read back and "
                f"asserted. " if v2 else ""
            ) + (
                "efSearch and nprobe are read back from the serialised index and "
                "asserted against the catalogue -- never assigned here. Note that "
                "FAISS's HNSW search uses a candidate heap of max(efSearch, k), so "
                "at k=100 an efSearch below 100 does not bind; hnsw_low's declared "
                "16 is therefore an upper bound on its selectivity, not the value "
                "actually used. That is a property of the catalogue as built and is "
                "recorded here rather than corrected."
            ),
            "id_space": {
                "source": str(pilot / "pair_index.csv"),
                "scope": json.load(open(pilot / "pair_index_meta.json"))["scope"],
                "corpus_role": "A", "n_indexed": int(len(a_ids)),
                "id_min": int(a_ids[0]), "id_max": int(a_ids[-1]),
                "contract": (
                    "Labels are the canonical pilot int_ids as returned by FAISS -- "
                    "no gather, no renumbering. Every index was populated with "
                    "add_with_ids (HNSW via IndexIDMap2, the IVF families natively), "
                    "so A's ids stay sparse in 0..999,999 with B and C leaving the "
                    "gaps. -1 means the search returned fewer than k and is stored "
                    "as -1; T8 counts it as a miss."
                ),
            },
            "roles": {r: int(len(role_ids[r][:args.limit_q] if args.limit_q
                                 else role_ids[r])) for r in args.roles},
            "files": {
                f"approx_{{{','.join(args.roles)}}}_<config>_ids.npy":
                    f"int64 [n_queries, {k}] -- the deliverable, rank order, -1 padded",
                f"approx_{{{','.join(args.roles)}}}_<config>_dist.npy":
                    f"float32 [n_queries, {k}] -- SQUARED L2 as the index reports it "
                    f"(an ASYMMETRIC PQ estimate for the IVFPQ/O-IVFPQ families, not "
                    f"a true distance)",
                f"approx_{{{','.join(args.roles)}}}_qids.npy":
                    "int64 [n_queries] -- row i of the arrays above is the result "
                    "for this int_id; identical to the split's role ids and to T6's "
                    "knn_<role>_qids.npy, which is asserted when those exist",
            },
            "contract": (
                f"Row i of approx_<role>_<config>_ids.npy is config <config>'s "
                f"top-k for pilot int_id approx_<role>_qids.npy[i], over role A's "
                f"vectors in the {imod} modality. T8 intersects it with T6's row i "
                f"for the same pairing and role, BY POSITION. Never reorder or "
                f"recompact."
            ),
            "checks": {
                "all_ids_in_A": "PASS", "no_B_or_C_ids_returned": "PASS",
                "no_query_retrieved_itself": "PASS",
                "no_duplicate_id_within_a_list": "PASS",
                "distances_non_decreasing": "PASS",
                "ntotal_matches_split": "PASS",
                "search_params_match_catalogue": "PASS",
                "qids_match_T6_where_present": bool(p in exact_meta),
            },
            "depends_on": {
                "T4": str(args.index_dir or REPO_ROOT / "result/indexes" / args.model / imod),
                "T3": str(pilot),
                "T6": "not required -- read only to cross-check query order",
            },
            "faiss_version": faiss.__version__,
            "threads": faiss.omp_get_max_threads(),
            "smoke": {"limit_q": args.limit_q} if smoke else None,
            "configs": ordered,
            "superseded_records": archived,
            "assembled_from": "meta_parts/<config>.json fragments, then the "
                              "meta on disk, then this run's start-time records "
                              "-- restricted to configs whose output is on disk",
        })
        print(f"\nmeta -> {dests[p] / 'approx_knn_meta.json'}  "
              f"({len(ordered)} configs: {', '.join(r['config'] for r in ordered)})")

    print(f"\ntotal wall {_fmt(time.time() - t_all)}")
    if smoke:
        print("SMOKE TEST -- only the first "
              f"{args.limit_q:,} queries per role were searched.")


if __name__ == "__main__":
    main()
