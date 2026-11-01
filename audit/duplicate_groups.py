"""
Duplicate and near-duplicate groups over the whole pilot (A + B + C).

WHY THIS EXISTS
---------------
CF@100 grades an index's top-100 against an exact top-100. That comparison is
only meaningful if "the exact top-100" is a well-defined object. When the
corpus holds many copies of one item -- a scrape failure that returned a stock
placeholder instead of the caption's image -- every copy sits at distance zero
from every other, the true top-100 is an ARBITRARY 100 drawn from a much larger
tied set, and CF measures which arbitrary 100 each index happened to return.
That is not retrieval quality, and no index can be graded on it.

T5 saw the symptom from one side (cell 911: 11,740 members, tightness 0.99997)
and T6 saw it from the other (tie-tolerant verification, "the stored list is A
valid top-k, not THE top-k"). Neither enumerated the groups, and nothing
downstream excludes them. This enumerates them.

WHY THE WHOLE PILOT, NOT JUST A
-------------------------------
The damage is done by duplicates in A, because A is what the labels are
computed over. But a B or C row can itself be a copy, and whether it is
changes what its label means:

    a C row duplicating A rows   -> its top-100 is tied, its label is undefined
    a C row duplicating only C   -> harmless, C is never in an index

Those cannot be told apart from A alone, so all three roles are grouped
together and the per-role counts are reported separately.

WHY EMBEDDING TIES AND NOT BIT-IDENTICAL FILES
----------------------------------------------
Bit-identity of the source file is SUFFICIENT for a degenerate label, not
necessary. The chain is

    identical file => identical pixels => identical embedding => tied top-100

and every arrow is one-way. LAION fetched from many URLs and re-encoded on the
way in, so the same placeholder served from fifty sites is almost never
byte-identical, yet after decode, resize and embedding it lands on one vector.
Cell 911 is exactly this case: tightness 0.99997, not 1.0. Detecting on file
bytes would miss it, and it is the largest group in the corpus.

So detection happens at the level where the harm occurs -- the embedding --
and ``duplicate_catalogue.py`` then pulls each group's images and captions from
the LAION tars so the groups can be adjudicated on content. The epsilon here
decides GROUPING; it does not decide what counts as a scrape failure.

THE THRESHOLD IS REPORTED, NOT ASSUMED
--------------------------------------
Group counts are emitted at every epsilon in ``--eps-sweep`` so the value can
be read off a plateau rather than picked. The spread of a genuine tie sits at
the float32 noise floor (~1e-6 in squared L2 at d=1152) while the nearest
non-duplicate neighbour is four orders of magnitude further out, so the count
is expected to be flat across the sweep. If it is not, the default is wrong.

COST
----
One exhaustive self-search. At 1M x 1152 this is ~2 minutes on one A100
(scaled from T6's measured 9,050 q/s against 850,393). At 7M it is ~1.8 h on
one GPU or ~13 min sharded over eight. The shard-tree walk that assembles the
matrix dominates the small case.

    qsub -I -q standard -P mimr.spons -N mmir_dup \
         -l select=1:ncpus=128:ngpus=1 -l walltime=01:00:00

    cd mmir_optimizer/audit
    unset http_proxy https_proxy
    ../.venv/bin/python duplicate_groups.py --model siglip2 --modality image
    ../.venv/bin/python duplicate_groups.py --model siglip2 --modality text

Login-node-safe smoke test, reads no shard:

    ../.venv/bin/python duplicate_groups.py --synthetic 5000 --no-gpu --threads 4
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

import faiss

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "indexes"))

from load_vectors import (                                          # noqa: E402
    load_included_shards, load_role_matrix, modality_root,
)

DEFAULT_PILOT = REPO_ROOT / "result/pilot_1M/siglip2"
DEFAULT_MANIFEST = REPO_ROOT / "result/corpus_shard_manifest.json"
DEFAULT_OUT = REPO_ROOT / "result/duplicate_groups"

#: Squared-L2 radius defining "the same item". Unit-norm vectors, so this is
#: cosine >= 1 - EPS/2; 1e-5 is cosine >= 0.999995.
DEFAULT_EPS = 1e-5
DEFAULT_SWEEP = (1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2)


# ==========================================================================
# The pilot, all three roles at once
# ==========================================================================
def load_corpus(model: str, modality: str, manifest: Path, excluded: Path,
                verbose: bool = True, root: Path = None):
    """The whole 7M corpus: (vectors, pair_ids), in canonical shard order.

    Not load_role_matrix: that resolves rows through the pilot's int_id space,
    which only covers 1M pairs. At corpus scope there is no int_id space yet --
    T3 assigns one per split, and the 7M split does not exist. So rows are
    keyed by pair_id, read from each shard's .names.json, and the row order is
    the canonical one (workers ascending, shards by filename, records in
    stored order) that every other stage already uses.

    The 353 T1/T2 anomalies are dropped here rather than kept and masked: at
    pilot scope they were excluded at sampling time and never entered A/B/C,
    so excluding them keeps the two scopes comparable.
    """
    # modality_root, not a name match. The embedding tree contains
    # siglip2_part0..7 (the parallel generation) and
    # siglip2_so400m_patch16_512.stale_partial_20260904 (an abandoned run of
    # 716,105 of 8,003,932 images, build_log T2) alongside the canonical
    # directory. Any heuristic over directory names can pick one of those, and
    # T2's warning is that such a run "would have stratified on 12% of the
    # image corpus and reported nothing wrong". MODEL_DIRS is the single source
    # of truth that every other stage resolves through.
    # root overrides it for a tree outside result/ (--embed-root), e.g. the
    # lowercase SigLIP 2 text vectors in result_v2/.
    base = Path(root) if root else modality_root(model, modality)
    if not base.is_dir():
        raise FileNotFoundError(f"no embedding dir for {model}/{modality}: {base}")

    included = set(json.load(open(manifest))["included_shards"])
    drop = {ln.strip() for ln in open(excluded)} if Path(excluded).exists() else set()
    shards = sorted(base.glob("worker_*/shard_*.npy"))
    shards = [p for p in shards if f"{p.parent.name}/{p.stem}" in included]
    if len(shards) != len(included):
        raise RuntimeError(f"manifest lists {len(included)} shards, "
                           f"{len(shards)} found under {base}")

    names_per, total = [], 0
    for p in shards:
        nm = json.load(open(p.with_suffix(".names.json")))
        names_per.append(nm)
        total += len(nm)
    d = int(np.load(shards[0], mmap_mode="r").shape[1])
    if verbose:
        print(f"  {len(shards):,} shards, {total:,} rows x {d} "
              f"({total * d * 4 / 2**30:.1f} GiB before exclusions)")

    X = np.empty((total, d), dtype=np.float32)
    pids = np.empty(total, dtype=object)
    at = 0
    for si, (p, nm) in enumerate(zip(shards, names_per)):
        v = np.load(p, mmap_mode="r")
        if v.shape[0] != len(nm):
            raise RuntimeError(f"{p}: {v.shape[0]} rows vs {len(nm)} names")
        X[at:at + len(nm)] = v
        for j, name in enumerate(nm):
            pids[at + j] = name.rsplit(".", 1)[0]
        at += len(nm)
        if verbose and (si + 1) % 500 == 0:
            print(f"    {si + 1:,}/{len(shards):,} shards", flush=True)

    if drop:
        keep = np.array([p not in drop for p in pids])
        if verbose:
            print(f"  dropping {int((~keep).sum())} T1/T2 anomalies")
        X, pids = X[keep], pids[keep]
    return X, pids


def load_pilot_ids(pair_index_csv: Path):
    """(int_ids ascending, role per id, pair_id per id, pair_id -> int_id).

    ``load_vectors.load_pair_index`` filters to one role; this wants all three,
    so the CSV is read once here rather than three times there.
    """
    ids, roles, pairs = [], [], []
    with open(pair_index_csv) as f:
        for r in csv.DictReader(f):
            ids.append(int(r["int_id"]))
            roles.append(r["role"])
            pairs.append(r["pair_id"])
    order = np.argsort(np.asarray(ids, dtype=np.int64), kind="stable")
    ids = np.asarray(ids, dtype=np.int64)[order]
    roles = np.asarray(roles)[order]
    pairs = np.asarray(pairs)[order]
    if not np.all(np.diff(ids) > 0):
        raise RuntimeError(
            f"{pair_index_csv}: int_ids are not unique. The whole pipeline "
            f"joins positionally on this id space (T3); duplicates here mean "
            f"every downstream gather is ambiguous.")
    return ids, roles, pairs, {p: int(i) for p, i in zip(pairs, ids)}


# ==========================================================================
# Search
# ==========================================================================
def to_gpu(flat, ngpu: int):
    if ngpu <= 0:
        return flat, "cpu"
    if ngpu == 1:
        res = faiss.StandardGpuResources()
        return faiss.index_cpu_to_gpu(res, 0, flat), "gpu:1"
    co = faiss.GpuMultipleClonerOptions()
    co.shard = True                     # split the corpus, do not replicate it
    return faiss.index_cpu_to_all_gpus(flat, co=co, ngpu=ngpu), f"gpu:{ngpu}-sharded"


def self_knn(X: np.ndarray, k: int, ngpu: int, batch: int = 8192):
    """Exhaustive self-search. Returns (D, I), both [n, k], squared L2.

    The query set IS the corpus, so every row retrieves itself. Its self-hit is
    NOT assumed to be at rank 0: inside a tied group the ordering among equal
    distances is arbitrary and a sharded merge can reorder it further, so the
    caller drops the self id by VALUE rather than by position.
    """
    n, d = X.shape
    flat = faiss.IndexFlatL2(d)
    flat.add(X)
    index, where = to_gpu(flat, ngpu)
    print(f"  searching {n:,} x {n:,} at k={k} on {where} ...", flush=True)

    D = np.empty((n, k), dtype=np.float32)
    I = np.empty((n, k), dtype=np.int64)
    for s in range(0, n, batch):
        e = min(s + batch, n)
        D[s:e], I[s:e] = index.search(X[s:e], k)
        if (s // batch) % 20 == 0:
            print(f"    {e:,}/{n:,}", flush=True)
    return D, I


# ==========================================================================
# Grouping
# ==========================================================================
def components(D: np.ndarray, I: np.ndarray, eps: float, extra=None):
    """Connected components of the graph {(i, j) : d(i, j) < eps, i != j}.

    Single-linkage by construction. That is the right linkage here and not a
    convenience: the relation being recovered is "is the same item as", which
    is transitive, and a chain of exact ties really is one group.

    k neighbours rather than 2 because a blob of n mutually-identical rows must
    come back as ONE component. With k=2 the nearest-neighbour edges form
    arbitrary chains that can fragment; with k=100 each member draws ~99 edges
    into the blob and the induced subgraph connects with overwhelming margin.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n, k = I.shape
    rows = np.repeat(np.arange(n, dtype=np.int64), k)
    cols = I.ravel()
    keep = (D.ravel() < eps) & (cols != rows) & (cols >= 0)
    rows, cols = rows[keep], cols[keep]
    if extra is not None and len(extra[0]):
        rows = np.concatenate([rows, extra[0]])
        cols = np.concatenate([cols, extra[1]])

    g = coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)),
                   shape=(n, n))
    n_comp, label = connected_components(g, directed=False)

    size = np.bincount(label, minlength=n_comp)
    # A row with no edge is its own component of size 1 -- not a group.
    grouped = size[label] > 1
    return label, grouped, int(len(rows))


def k_saturated(D: np.ndarray, I: np.ndarray, eps: float) -> int:
    """Rows whose ENTIRE k-neighbourhood is within eps.

    These are the rows for which k was too small to see past their own group,
    so the k-NN graph alone cannot certify that the group is complete.
    ``close_groups`` resolves them exactly rather than leaving a caveat.
    """
    return int(((D < eps).sum(axis=1) >= D.shape[1]).sum())


def close_groups(X: np.ndarray, D: np.ndarray, I: np.ndarray, eps: float,
                 ngpu: int, chunk: int = 512, mode: str = "auto",
                 max_rows: int = 20000):
    """Extra edges for rows whose k-neighbourhood was entirely inside eps.

    Why this exists: a group of 11,740 mutually-identical rows searched at
    k=100 shows every member saturated, and single linkage over the resulting
    random k-NN graph connects it only WITH HIGH PROBABILITY. "Almost surely"
    is not a group size. This re-queries just the saturated rows, deep enough
    to see past the largest group found, and returns the edges that pass -- so
    the sizes in groups.csv are exact rather than probable.

    Cheap because it is restricted: at 1M the saturated set is ~1.5% of rows,
    so this is ~1.5% of one full pass.

    Two modes, because the exact one does not survive scale:

    range   CPU range_search: takes a RADIUS, so it has no k ceiling and
            returns the complete eps-ball. Exact, and the right choice at
            pilot scale. FAISS's GPU kernels cap k at 2048, so a k-based
            closure there would silently truncate at exactly the groups it
            exists to measure -- hence CPU.

    knn     GPU search capped at k=2048. Bounded, and necessary at corpus
            scale: the 13,796-member group becomes ~96,000 at 7M, and its
            exact eps-ball would be 96,000 x 96,000 edges -- over 100 GiB for
            one group. Not exact, but sufficient for CONNECTIVITY, which is
            all closure is for: every member already draws its own edges in
            the main pass, so a group cannot be missed, only in principle
            split, and 2,048 edges per node makes a split vanishingly
            unlikely.

    ``auto`` picks range below ``--closure-max-rows`` saturated rows and knn
    above it, and the choice is recorded in meta.json so the guarantee the
    group sizes carry is never ambiguous.
    """
    sat = np.nonzero((D < eps).sum(axis=1) >= D.shape[1])[0]
    if len(sat) == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64), "not needed"

    if mode == "auto":
        mode = "range" if len(sat) <= max_rows else "knn"

    rr, cc = [], []
    if mode == "range":
        index = faiss.IndexFlatL2(X.shape[1])
        index.add(X)
        print(f"  closing {len(sat):,} saturated rows by range_search "
              f"(radius {eps:g}, cpu) ...", flush=True)
        for s in range(0, len(sat), chunk):
            q = sat[s:s + chunk]
            lims, _, idx = index.range_search(X[q], float(eps))
            # lims comes back uint64; np.repeat refuses that cast, and letting
            # it through unsigned would wrap on any negative difference.
            counts = np.diff(lims.astype(np.int64))
            rows = np.repeat(q, counts)
            cols = idx.astype(np.int64)
            keep = cols != rows
            rr.append(rows[keep]); cc.append(cols[keep])
            if (s // chunk) % 10 == 0:
                print(f"    {min(s + chunk, len(sat)):,}/{len(sat):,}", flush=True)
    else:
        k2 = int(min(X.shape[0], 2048))
        flat = faiss.IndexFlatL2(X.shape[1]); flat.add(X)
        index, where = to_gpu(flat, ngpu)
        print(f"  closing {len(sat):,} saturated rows by capped knn "
              f"(k={k2:,} on {where}) -- connectivity only, sizes are not "
              f"certified exact", flush=True)
        for s in range(0, len(sat), 1024):
            q = sat[s:s + 1024]
            d2, i2 = index.search(X[q], k2)
            rows = np.repeat(q, k2); cols = i2.ravel()
            keep = (d2.ravel() < eps) & (cols != rows) & (cols >= 0)
            rr.append(rows[keep]); cc.append(cols[keep])
            if (s // 1024) % 20 == 0:
                print(f"    {min(s + 1024, len(sat)):,}/{len(sat):,}", flush=True)
    return np.concatenate(rr), np.concatenate(cc), mode


# ==========================================================================
# Run
# ==========================================================================
def run(args) -> dict:
    smoke = args.synthetic > 0
    if not smoke and not os.environ.get("PBS_JOBID") and not args.allow_login_node:
        sys.exit(
            "refusing to run on what looks like a login node: this assembles a "
            "multi-GB matrix over the shard tree and scans it exhaustively. "
            "Submit a job, or pass --allow-login-node if you are already inside "
            "one and PBS_JOBID is not set.")

    if args.threads:
        faiss.omp_set_num_threads(args.threads)

    if smoke:
        # Synthetic corpus with planted groups, so the detector has a known
        # answer to be wrong about. Reads no shard.
        rng = np.random.default_rng(0)
        n, d, n_groups, per = args.synthetic, 64, 10, 7
        X = rng.normal(size=(n, d)).astype(np.float32)
        for g in range(n_groups):                 # plant exact duplicates
            base = g * per
            X[base + 1: base + per] = X[base]
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        ids = np.arange(n, dtype=np.int64)
        roles = np.array(["A"] * n)
        pairs = np.array([f"synthetic/shard_00000/{i:09d}" for i in range(n)])
        planted = n_groups
        print(f"  SMOKE: {n:,} synthetic rows, {n_groups} planted groups of {per}")
    elif args.scope == "corpus":
        X, pairs = load_corpus(args.model, args.modality, args.manifest,
                               REPO_ROOT / "result/excluded_pairs.txt",
                               root=args.embed_root)
        # No int_id space exists at corpus scope (T3 assigns one per split and
        # the 7M split is not built), so rows are identified by position and
        # the pair_id is the durable key. No roles either -- A/B/C is a pilot
        # concept.
        ids = np.arange(len(pairs), dtype=np.int64)
        roles = np.full(len(pairs), "-")
        print(f"  corpus: {len(ids):,} pairs")
        planted = None
    else:
        ids, roles, pairs, pair_to_int = load_pilot_ids(args.pilot_dir / "pair_index.csv")
        included = load_included_shards(args.manifest)
        print(f"  pilot: {len(ids):,} rows  "
              f"(A {int((roles == 'A').sum()):,} / B {int((roles == 'B').sum()):,} "
              f"/ C {int((roles == 'C').sum()):,})")
        X = load_role_matrix(args.model, args.modality, ids, pair_to_int,
                             included, check_norms=True, verbose=True)
        planted = None

    D, I = self_knn(X, args.k, 0 if args.no_gpu else args.ngpu, args.batch)

    # ---- the sweep, so the epsilon is read off a plateau ------------------
    sweep = []
    for eps in args.eps_sweep:
        label, grouped, n_edges = components(D, I, eps)
        sizes = np.bincount(label)[label[grouped]] if grouped.any() else np.array([])
        sweep.append({
            "eps": float(eps),
            "n_rows_grouped": int(grouped.sum()),
            "n_groups": int(len(np.unique(label[grouped]))) if grouped.any() else 0,
            "largest_group": int(sizes.max()) if len(sizes) else 0,
            "n_edges": n_edges,
            "rows_k_saturated": k_saturated(D, I, eps),
        })
        print(f"    eps={eps:<9g} groups={sweep[-1]['n_groups']:>6} "
              f"rows={sweep[-1]['n_rows_grouped']:>7,} "
              f"largest={sweep[-1]['largest_group']:>7,} "
              f"k-saturated={sweep[-1]['rows_k_saturated']:>6,}")

    # ---- the chosen epsilon, with saturated rows resolved exactly ---------
    sat = k_saturated(D, I, args.eps)
    extra, k_close = None, 0
    if sat and not args.no_close:
        er, ec, k_close = close_groups(X, D, I, args.eps,
                                       0 if args.no_gpu else args.ngpu,
                                       mode=args.closure,
                                       max_rows=args.closure_max_rows)
        extra = (er, ec)
        before = components(D, I, args.eps)[1].sum()
        after = components(D, I, args.eps, extra)[1].sum()
        print(f"  closure added {after - before:,} rows "
              f"({'group sizes were complete already' if after == before else 'k=%d was too small' % args.k})")
    elif sat:
        print(f"\n  !! {sat:,} rows saturated at k={args.k} and --no-close was "
              f"passed: group sizes are LOWER BOUNDS, not measurements.")
    label, grouped, _ = components(D, I, args.eps, extra)

    # Renumber the surviving components densely, largest first, so group_id 0
    # is the biggest offender and the catalogue reads top-down.
    keep_labels = np.unique(label[grouped])
    order = sorted(keep_labels, key=lambda L: -int((label == L).sum()))
    remap = {int(L): g for g, L in enumerate(order)}
    row_group = np.full(len(label), -1, dtype=np.int32)
    for L, g in remap.items():
        row_group[label == L] = g

    # Scope is part of the path. A corpus run and a pilot run produce arrays of
    # the same name, dtype and meaning-per-row but over different populations;
    # sharing a directory means the second silently destroys the first.
    # Smoke output is quarantined for the same reason.
    root = (args.out_dir / "_smoke") if smoke else args.out_dir
    dest = root / args.model / args.scope / args.modality
    dest.mkdir(parents=True, exist_ok=True)
    is_pilot = (args.scope == "pilot") or smoke

    rows = []
    for g in range(len(order)):
        m = row_group == g
        r = roles[m]
        # Representative: lowest row, so the choice is deterministic and
        # re-running cannot silently change which record gets eyeballed.
        rep = int(np.argmax(m))
        row = [g, int(m.sum())]
        if is_pilot:            # roles exist only once T3 has split the pilot
            row += [int((r == "A").sum()), int((r == "B").sum()),
                    int((r == "C").sum()), int(ids[rep])]
        row.append(str(pairs[rep]))
        rows.append(row)
    rows.sort(key=lambda x: -x[1])

    header = ["group_id", "n_total"]
    if is_pilot:
        header += ["n_A", "n_B", "n_C", "rep_int_id"]
    header.append("rep_pair_id")
    with open(dest / "groups.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    np.save(dest / "row_group.npy", row_group)
    # The identifier, aligned row-for-row with row_group. pair_id is the key
    # that exists at every scope -- int_id is assigned by T3, per split, so at
    # corpus scope there is none and writing arange() would look like one.
    # Fixed-width ASCII rather than object dtype, so it loads without pickle.
    np.save(dest / "row_pair_id.npy",
            np.asarray([str(p) for p in pairs], dtype="S30"))
    if is_pilot:
        np.save(dest / "row_int_id.npy", ids)

    meta = {
        "created": datetime.now().isoformat(),
        "task": "audit -- duplicate group enumeration",
        "scope": args.scope,
        "model": args.model, "modality": args.modality,
        "embedding_root": (None if smoke else
                           str(args.embed_root or modality_root(args.model, args.modality))),
        "n_rows": int(len(ids)), "k": int(args.k),
        "eps_chosen": float(args.eps),
        "eps_units": "squared L2 on unit-norm vectors; eps = 2(1 - cos)",
        "eps_sweep": sweep,
        "rows_k_saturated_at_chosen_eps": sat,
        "closure": (k_close if k_close else "SKIPPED (--no-close)") if sat
                   else "not needed -- no row saturated k",
        "group_sizes_are": ("exact" if (not sat or k_close == "range")
                            else "connectivity-assured, sizes not certified exact"
                            if k_close == "knn" else
                            "LOWER BOUNDS -- closure was skipped"),
        "n_groups": len(order),
        "n_rows_grouped": int((row_group >= 0).sum()),
        "share_of_pilot_grouped": float((row_group >= 0).mean()),
        "linkage": "single, over the k-NN graph thresholded at eps",
        "smoke": bool(smoke),
        "files": {
            "groups.csv": "one row per group, largest first",
            "row_group.npy": "int32 [n_rows], group id or -1",
            "row_pair_id.npy": "S30 [n_rows], the pair_id of each row, aligned "
                               "row-for-row with row_group.npy -- the canonical key",
            "row_int_id.npy": "int64 [n_rows], pilot int_id; PILOT SCOPE ONLY, "
                              "T3 assigns int_ids per split so none exists at "
                              "corpus scope",
        },
        "not_an_exclusion_list": (
            "Grouping is geometric. Whether a group is a scrape failure or a "
            "legitimately tight cluster is decided on CONTENT by "
            "duplicate_catalogue.py and human adjudication, not here."),
    }
    if smoke and planted is not None:
        meta["smoke_planted_groups"] = planted
        meta["smoke_pass"] = bool(len(order) == planted)
        print(f"\n  SMOKE: planted {planted} groups, recovered {len(order)} -- "
              f"{'PASS' if len(order) == planted else 'FAIL'}")
    json.dump(meta, open(dest / "meta.json", "w"), indent=2)

    print(f"\n  {len(order):,} groups covering {int((row_group >= 0).sum()):,} rows "
          f"({(row_group >= 0).mean():.2%} of "
          f"{'the synthetic corpus' if smoke else 'the pilot'})")
    print(f"  wrote {dest}/groups.csv")
    return meta


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="siglip2")
    ap.add_argument("--modality", default="image", choices=["image", "text"])
    ap.add_argument("--k", type=int, default=100,
                    help="neighbours per row; must exceed the largest group's "
                         "size for that group to be fully connected in one pass")
    ap.add_argument("--eps", type=float, default=DEFAULT_EPS)
    ap.add_argument("--eps-sweep", type=float, nargs="+", default=list(DEFAULT_SWEEP))
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--ngpu", type=int, default=1)
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--scope", default="pilot", choices=["pilot", "corpus"],
                    help="pilot = the 1M sample; corpus = all 7M included shards")
    ap.add_argument("--closure", default="auto",
                    choices=["auto", "range", "knn"],
                    help="range is exact but unbounded; knn is capped and "
                         "assures connectivity only. auto switches at "
                         "--closure-max-rows")
    ap.add_argument("--closure-max-rows", type=int, default=20000)
    ap.add_argument("--no-close", action="store_true",
                    help="skip the closure pass; group sizes then become lower "
                         "bounds wherever a row saturated its k neighbours")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--pilot-dir", type=Path, default=DEFAULT_PILOT)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--embed-root", type=Path, default=None,
                    help="worker_*/shard_*.npy tree to read instead of the model's "
                         "result/embeddings_<modality>/ directory (corpus scope only)")
    ap.add_argument("--synthetic", type=int, default=0,
                    help="smoke test on N synthetic rows with planted groups; "
                         "reads no shard and writes nothing real")
    ap.add_argument("--allow-login-node", action="store_true")
    args = ap.parse_args()
    print(f"duplicate_groups  model={args.model}  modality={args.modality}")
    run(args)


if __name__ == "__main__":
    main()
