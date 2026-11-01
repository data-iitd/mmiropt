"""
T8: CF computation -- the per-query fidelity table.

    python cf_compute.py --model siglip2 --pairing all

    CF@k(q, A_j) = |exact top-k(q)  intersect  approx top-k(q, A_j)| / k

T6 gives the exact list, T7 gives the approximate one; this is the intersection,
per query, for every (config, pairing, role). Pure CPU set arithmetic -- cheap
per row, and the largest table the pilot produces: |B u C| x 9 configs x 4
pairings = 5,385,852 rows.

DEPENDS ON T6 AND T7, PER PAIRING
    A pairing is computed as soon as both of its inputs exist, and skipped with
    a message otherwise, so the text-index pairings can be picked up whenever
    their exact-KNN pass finishes. Each pairing writes its own part file;
    --merge concatenates whatever parts exist into the single flat table.

THE INTERSECTION IS BY POSITION, NEVER BY JOIN
    Row i of T6's labels and row i of T7's lists are the same query because both
    were written in the split's role order. That is not assumed: the qids files
    of both are required to be equal, to each other and to the split, before a
    single intersection is computed. A silent off-by-one here pairs one query's
    exact list with another's approximate one and produces a complete, plausible,
    entirely meaningless table.

-1 IS A MISS, NOT A MATCH
    FAISS pads a short approximate result with -1. It is filtered before the
    intersection: matching it against anything would credit a config for a
    neighbour it failed to return.

THE TIE FLOOR -- what CF differences are NOT meaningful
    ~2.3% of A has an exact duplicate vector elsewhere (LAION serves the same
    image at many URLs), so at the cross-modal pairings ~1% of top-100 slots are
    tie-ambiguous: which of two equidistant ids fills rank 100 is arbitrary and
    not the index's doing. A CF gap between two configs smaller than the tie
    floor recorded in T6's meta is tie shuffling. H1-H5 must not read it as
    signal. Carried into cf_summary.json per pairing so T10 cannot lose it.

V2 (--params-json, the second run in result_v2/)
    The config names, families and tiers come from the v2 catalogue
    (indexes/v2_dirs.py configs_v2) instead of index_configs.py; the paths are
    the existing options (--pilot-dir, --knn-dir, --approx-dir, --out-dir).
    The intersection itself is unchanged.

TOP-K BELOW THE STORED K (--topk; v3_k10, user 2026-10-30)
    --topk 10 grades fidelity@10 from the first 10 columns of the stored
    top-100 lists. Both lists are sorted by distance (FAISS returns its k best
    in order), so their first 10 columns are the exact and approximate top-10.
    The ANN search itself ran with k=100, so the candidate pool is that of a
    k=100 search; for HNSW (efSearch >= 100 > 10) and IVF-PQ (fixed nprobe) the
    best 10 of that pool are what a k=10 search returns. The tie floor is
    re-measured at rank --topk from the stored exact distances.

    python knn/cf_compute.py --model siglip2 --pairing all --merge \
        --params-json result_v2/param_check/7M_cleaned/final_params_chosen.json \
        --pilot-dir result_v2/7M_cleaned/siglip2 --knn-dir result_v2/knn/7M_cleaned \
        --approx-dir result_v2/approx_knn/7M_cleaned --out-dir result_v2/cf/7M_cleaned

THIS TABLE IS AN INTERMEDIATE
    T9 aggregates it into the map and T10 computes the GA/DO baselines from it;
    nothing reads it after that. It is kept, not deleted, because regenerating
    it means rerunning T7.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(REPO_ROOT / "indexes"))
from index_configs import MODEL_DIMS, configs_for, describe            # noqa: E402

from exact_knn import DEFAULT_PILOT, PAIRINGS                          # noqa: E402
from approx_knn import DEFAULT_KNN, DEFAULT_OUT as DEFAULT_APPROX      # noqa: E402

DEFAULT_CF = REPO_ROOT / "result/cf/1M"

HEADER = "query_id,role,configuration,modality_configuration,cf_score"


def _fmt(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def cf_per_query(exact: np.ndarray, approx: np.ndarray, k: int) -> np.ndarray:
    """
    |exact intersect approx| / k, per row, for two [n, k] id matrices.

    Vectorised over the whole block rather than looped per query: each id is
    offset by its row -- key = row * span + id -- so one global searchsorted
    answers all n rows at once, and a match can never cross row boundaries.
    -1 slots are dropped before keying, so padding matches nothing.
    """
    n = exact.shape[0]
    if approx.shape != exact.shape:
        raise RuntimeError(f"shape mismatch: exact {exact.shape} vs approx {approx.shape}")
    if int(exact.min()) < 0:
        raise RuntimeError(
            "T6's exact labels contain -1. They are ground truth and must be "
            "complete; keying by row would also let a sentinel collide with a "
            "real id in the previous row.")
    span = int(max(exact.max(), approx.max())) + 1

    rows = np.broadcast_to(np.arange(n, dtype=np.int64)[:, None], (n, k))
    e_keys = np.sort((rows * span + exact).ravel())

    valid = approx >= 0
    a_rows = rows[valid]
    a_keys = a_rows * span + approx[valid]

    pos = np.minimum(np.searchsorted(e_keys, a_keys), e_keys.size - 1)
    hit = e_keys[pos] == a_keys

    hits = np.bincount(a_rows[hit], minlength=n)
    if int(hits.max(initial=0)) > k:
        raise RuntimeError("more hits than k -- a duplicate id survived into a list")
    return hits.astype(np.float32) / float(k)


def load_aligned(labels: Path, approx: Path, role: str, split_ids: np.ndarray):
    """
    T6's and T7's qids for one role, checked against each other and the split
    before anything is intersected.
    """
    e_q = np.load(labels / f"knn_{role}_qids.npy")
    a_q = np.load(approx / f"approx_{role}_qids.npy")
    # A --limit-q smoke run of T7 covers a PREFIX of the role, so a shorter T7
    # side is allowed and CF is computed over that prefix. A longer one is not:
    # it would mean T7 searched queries T6 never labelled.
    if len(a_q) > len(e_q):
        raise RuntimeError(
            f"{role}: T7 holds {len(a_q):,} queries but T6 labelled {len(e_q):,}")
    if not np.array_equal(e_q[:len(a_q)], a_q):
        raise RuntimeError(
            f"{role}: T6 qids and T7 qids differ -- rows do not describe the same "
            f"queries, so every CF computed from them would pair one query's exact "
            f"list with another's approximate one.")
    if not np.array_equal(a_q, split_ids[:len(a_q)]):
        raise RuntimeError(
            f"{role}: qids are not the split's role ids in order "
            f"({labels}, {approx})")
    if len(a_q) < len(e_q):
        print(f"    {role}: PARTIAL -- T7 covers {len(a_q):,} of {len(e_q):,} "
              f"queries (a --limit-q run). CF is computed over that prefix only.")
    return a_q


def write_part(path: Path, qids, role, pairing, names, cf_by_config):
    """
    One CSV part per pairing: every role x config block for that pairing.

    Written by string join per block rather than row by row -- 1.3M rows per
    pairing is slow enough to notice through csv.writer, and the schema is four
    constant-width columns and a two-decimal score.
    """
    with open(path, "a") as f:
        for name in names:
            cf = cf_by_config[name]
            suffix = f",{role},{name},{pairing},"
            f.write("\n".join(
                f"{q}{suffix}{c:.2f}" for q, c in zip(qids.tolist(), cf.tolist())
            ))
            f.write("\n")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="siglip2", choices=sorted(MODEL_DIMS))
    ap.add_argument("--pairing", nargs="+", default=["all"],
                    choices=sorted(PAIRINGS) + ["all"])
    ap.add_argument("--roles", nargs="+", default=["B", "C"], choices=["B", "C"])
    ap.add_argument("--configs", default="all", help="comma-separated, or 'all'")
    ap.add_argument("--k", type=int, default=0,
                    help="0 = take k from T6's meta, which is the definition in use")
    ap.add_argument("--topk", type=int, default=0,
                    help="grade fidelity@topk from the first topk columns of the stored lists (0 = stored k)")
    ap.add_argument("--pilot-dir", type=Path, default=DEFAULT_PILOT)
    ap.add_argument("--knn-dir", type=Path, default=DEFAULT_KNN, help="T6 labels")
    ap.add_argument("--approx-dir", type=Path, default=DEFAULT_APPROX, help="T7 lists")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_CF)
    ap.add_argument("--merge", action="store_true",
                    help="after computing, concatenate the parts into cf_table.csv")
    ap.add_argument("--merge-only", action="store_true",
                    help="skip computation, just rebuild cf_table.csv from parts")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--allow-login-node", action="store_true")
    ap.add_argument("--params-json", type=Path, default=None,
                    help="v2: take the configs from the v2 catalogue (configs_v2)")
    ap.add_argument("--index-root", type=Path, default=None,
                    help="v2: the index tree whose build.json files configs_v2 reads")
    args = ap.parse_args()
    v2 = args.params_json is not None

    pairings = sorted(PAIRINGS) if "all" in args.pairing else list(dict.fromkeys(args.pairing))
    if not os.environ.get("PBS_JOBID") and not args.allow_login_node:
        raise SystemExit(
            "Refusing to run outside a PBS job. This is light -- set intersection "
            "over ~5.4M rows, minutes and a couple of GB -- but it still reads "
            "~9 GB of neighbour lists. Pass --allow-login-node if you have a "
            "reason to run it here.")

    dest = args.out_dir / args.model
    parts = dest / "parts"
    parts.mkdir(parents=True, exist_ok=True)

    if v2:
        from v2_dirs import INDEX_ROOT, configs_v2                     # noqa: E402
        catalogue = configs_v2(args.model, args.params_json, args.index_root or INDEX_ROOT)
    else:
        catalogue = configs_for(args.model)
    names_all = [c.name for c in catalogue]
    names = names_all if args.configs == "all" else args.configs.split(",")
    unknown = [n for n in names if n not in names_all]
    if unknown:
        raise SystemExit(f"unknown config(s): {unknown}\n\n{describe(args.model)}")
    family_of = {c.name: c.family for c in catalogue}
    point_of = {c.name: c.point for c in catalogue}
    ref_of = {c.name: (f"{list(c.search)[0]}={list(c.search.values())[0]}" if v2 else c.ref_cf100)
              for c in catalogue}

    split_ids = {r: np.load(args.pilot_dir / f"{r}_ids.npy") for r in args.roles}

    print("=" * 78)
    print(f"T8 CF computation -- model={args.model}")
    print(f"  pairings {', '.join(pairings)}")
    print(f"  configs  {', '.join(names)}")
    print(f"  roles    {', '.join(args.roles)}")
    print(f"  out      {dest}")
    print("=" * 78)

    summary_path = dest / "cf_summary.json"
    summary = json.load(open(summary_path)) if summary_path.exists() else {
        "task": "T8 CF computation", "model": args.model, "pairings": {},
    }

    t_all = time.time()
    if not args.merge_only:
        for p in pairings:
            labels = args.knn_dir / p / args.model
            approx = args.approx_dir / args.model / p
            part = parts / f"cf_{p}.csv"

            if not (labels / "exact_knn_meta.json").exists():
                print(f"\n[{p}] SKIPPED -- no T6 labels at {labels}. "
                      f"Rerun this pairing once exact-KNN finishes.")
                continue
            if not (approx / "approx_knn_meta.json").exists():
                print(f"\n[{p}] SKIPPED -- no T7 lists at {approx}. Run approx_knn.py.")
                continue
            if part.exists() and not args.overwrite:
                print(f"\n[{p}] {part.name} exists, skipping (--overwrite to redo)")
                continue

            emeta = json.load(open(labels / "exact_knn_meta.json"))
            ameta = json.load(open(approx / "approx_knn_meta.json"))
            k = args.k or int(emeta["k"])
            if int(ameta["k"]) != k:
                raise RuntimeError(
                    f"{p}: T6 stored k={emeta['k']} but T7 stored k={ameta['k']}. "
                    f"CF@k is only defined against lists of the same length.")
            if emeta["metric"] != ameta["metric"]:
                raise RuntimeError(
                    f"{p}: metric mismatch, T6 {emeta['metric']} vs T7 "
                    f"{ameta['metric']}. The intersection would still return a "
                    f"number, just a meaningless one.")
            # Which configs T7 has is decided by the output files on disk, not by
            # the meta's config list. Several jobs may split one pairing's
            # configs, and a meta written by one can momentarily lack another's
            # records -- the files cannot. The meta is still read for k and the
            # metric, which every writer agrees on.
            missing = [n for n in names
                       if not all((approx / f"approx_{r}_{n}_ids.npy").exists()
                                  for r in args.roles)]
            if missing:
                print(f"\n[{p}] SKIPPED -- T7 output files missing for {missing}")
                continue
            in_meta = {c["config"] for c in ameta.get("configs", [])}
            unlisted = [n for n in names if n not in in_meta]
            if unlisted:
                print(f"\n[{p}] note: outputs present but not yet listed in "
                      f"approx_knn_meta.json: {unlisted} -- proceeding on the files")

            k_stored = k
            if args.topk:
                if not 0 < args.topk <= k_stored:
                    raise SystemExit(f"--topk {args.topk} must lie in (0, {k_stored}]")
                k = args.topk
            print(f"\n[{p}] ({PAIRINGS[p]['short']})  k={k} (stored k={k_stored})")
            part.write_text(HEADER + "\n")
            rows_written = 0
            per_config = {}

            for role in args.roles:
                qids = load_aligned(labels, approx, role, split_ids[role])
                exact = np.load(labels / f"knn_{role}_ids.npy")
                if exact.shape[1] != k_stored:
                    raise RuntimeError(
                        f"{p}/{role}: exact labels have k={exact.shape[1]}, "
                        f"expected {k_stored}")
                exact = np.ascontiguousarray(exact[:len(qids), :k])

                cf_by_config = {}
                for name in names:
                    a = np.load(approx / f"approx_{role}_{name}_ids.npy")
                    if a.shape[1] != k_stored:
                        raise RuntimeError(f"{p}/{role}/{name}: approx lists have k={a.shape[1]}, expected {k_stored}")
                    a = np.ascontiguousarray(a[:len(qids), :k])
                    cf = cf_per_query(exact, a, k)
                    if float(cf.min()) < 0.0 or float(cf.max()) > 1.0:
                        raise RuntimeError(
                            f"{p}/{role}/{name}: CF outside [0,1] "
                            f"({cf.min()}, {cf.max()})")
                    cf_by_config[name] = cf
                    np.save(dest / f"cf_{p}_{role}_{name}.npy", cf)

                    st = per_config.setdefault(name, {})
                    st[role] = {
                        "n_queries": int(len(cf)),
                        "cf_mean": float(cf.mean()),
                        "cf_median": float(np.median(cf)),
                        "cf_p10": float(np.percentile(cf, 10)),
                        "cf_p90": float(np.percentile(cf, 90)),
                        "cf_std": float(cf.std()),
                        "frac_zero": float((cf == 0).mean()),
                        "frac_perfect": float((cf == 1).mean()),
                    }

                write_part(part, qids, role, p, names, cf_by_config)
                rows_written += len(qids) * len(names)
                print(f"    {role}: {len(qids):,} queries x {len(names)} configs "
                      f"= {len(qids) * len(names):,} rows")

            # ---- the sanity check the task requires, printed and recorded ----
            grade_role = args.roles[-1]
            print(f"\n    {'config':<14} {'family':<8} {'pt':<5} "
                  f"{'CF@' + str(k):>8} {'median':>8} {'zero':>7} {'perfect':>8}  ref")
            print("    " + "-" * 92)
            for name in names:
                st = per_config[name][grade_role]
                print(f"    {name:<14} {family_of[name]:<8} {point_of[name]:<5} "
                      f"{st['cf_mean']:>7.1%} {st['cf_median']:>8.1%} "
                      f"{st['frac_zero']:>6.1%} {st['frac_perfect']:>7.1%}  "
                      f"{ref_of[name]}")

            # A partial run may hold no (low, high) pair at all. That is not a
            # pass: an empty check reporting OK is exactly the green light a
            # subset run must not give.
            inversions, checked = [], []
            for fam in ("HNSW", "IVFPQ", "O-IVFPQ"):
                fam_names = [n for n in names if family_of[n] == fam]
                lo = [n for n in fam_names if point_of[n] == "low"]
                hi = [n for n in fam_names if point_of[n] == "high"]
                if lo and hi:
                    checked.append(fam)
                    a = per_config[lo[0]][grade_role]["cf_mean"]
                    b = per_config[hi[0]][grade_role]["cf_mean"]
                    if b <= a:
                        inversions.append(
                            f"{fam}: high ({hi[0]} {b:.1%}) does not beat low "
                            f"({lo[0]} {a:.1%})")
            if inversions:
                print("\n    ORDERING WARNING -- investigate before T9:")
                for line in inversions:
                    print(f"      {line}")
            elif checked:
                print(f"\n    ordering OK: every family's high tier beats its low "
                      f"tier (checked: {', '.join(checked)})")
            else:
                print("\n    ordering NOT CHECKED -- no family has both a low and a "
                      "high tier in this run")
            if not v2:
                print("    (ivfpq_mod sits below ivfpq_low by construction -- nlist=8 "
                      "with nprobe=1 caps it; that is the tier, not a bug)")

            tie = emeta.get("roles", {}).get(grade_role, {}).get("boundary_ties")
            if k != k_stored:                  # T6's tie floor is at rank k_stored; re-measure at rank k
                D = np.load(labels / f"knn_{grade_role}_dist.npy", mmap_mode="r")
                gap = np.asarray(D[:, k], dtype=np.float64) - np.asarray(D[:, k - 1], dtype=np.float64)
                tie = {"queries_with_boundary_tie": int((gap <= 1e-6).sum()),
                       "frac": float((gap <= 1e-6).mean()), "median_gap": float(np.median(gap)),
                       "rank": k, "note": f"re-measured at rank {k} from T6's stored distances"}
            if tie:
                print(f"    tie floor: {tie['frac']:.2%} of queries have a tied "
                      f"rank-{k} slot -- CF gaps below that are tie shuffling")

            summary["pairings"][p] = {
                "k": k,
                "k_stored": k_stored,
                "query_modality": PAIRINGS[p]["query_modality"],
                "index_modality": PAIRINGS[p]["index_modality"],
                "same_modality": PAIRINGS[p]["same_modality"],
                "rows": rows_written,
                "part_file": str(part),
                "labels_from": str(labels),
                "approx_from": str(approx),
                "tie_floor": tie,
                "ordering_families_checked": checked,
                "ordering_warnings": inversions,
                "configs": per_config,
                "catalogue": ("v2 configs_v2, " + str(args.params_json)) if v2 else "v1 index_configs",
                "computed": datetime.now().isoformat(),
            }
            json.dump(summary, open(summary_path, "w"), indent=2)
            print(f"    -> {part}")

    # ---- merge ------------------------------------------------------------
    if args.merge or args.merge_only:
        have = sorted(parts.glob("cf_*.csv"))
        if not have:
            raise SystemExit(f"no parts to merge under {parts}")
        table = dest / "cf_table.csv"
        total = 0
        with open(table, "w") as out:
            out.write(HEADER + "\n")
            for part in have:
                with open(part) as f:
                    next(f)                      # its own header
                    for line in f:
                        out.write(line)
                        total += 1
        json.dump({
            "task": "T8 CF computation -- flat table",
            "created": datetime.now().isoformat(),
            "model": args.model,
            "columns": HEADER.split(","),
            "rows": total,
            "parts_merged": [p.name for p in have],
            "complete": len(have) == len(PAIRINGS),
            "column_notes": {
                "query_id": "canonical pilot int_id -- the same space as T4's "
                            "add_with_ids, T6's labels and T7's lists",
                "role": "B (map construction) or C (held-out test). Not in the "
                        "task's minimal schema, but every downstream consumer "
                        "needs it: T9 builds the map from B only and T10 grades "
                        "on C only, and recovering it by joining against the "
                        "split is an avoidable chance to get it wrong.",
                "configuration": "index config name -- the T4 file stem",
                "modality_configuration": "query->index pairing, written in full",
                "cf_score": "CF@k in [0,1], exactly k/100 of a hundredth",
            },
            "downstream": (
                "T9 aggregates this per (landmark, config) from role B; T10 "
                "computes GA/DO baselines and the tau sweep from role C. Nothing "
                "reads it after that."
            ),
        }, open(dest / "cf_table_meta.json", "w"), indent=2)
        print(f"\nmerged {len(have)} part(s), {total:,} rows -> {table}")
        if len(have) != len(PAIRINGS):
            missing = sorted(set(PAIRINGS) - {p.stem[3:] for p in have})
            print(f"INCOMPLETE -- still missing: {', '.join(missing)}")

    print(f"\ntotal wall {_fmt(time.time() - t_all)}")


if __name__ == "__main__":
    main()
