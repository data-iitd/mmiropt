"""
Paper experiments, part 2 of 2: the map's online cost c_est -- the nearest-
landmark lookup -- measured the same way the configurations were.

    python paperexp/bench_lookup.py --model siglip2        (inside a PBS job)

Fills RQ5's lookup rows (mean, p99) and supplies the c_est that RQ4 adds to the
map's cost. The configurations' costs come from knn/bench_throughput.py; this
reuses its query sample, batch size and thread count so the two are comparable.

WHAT IS TIMED
  lookup    IndexFlatL2 over the K landmarks of the index modality, k = 1:
            Step 1 of the online stage, i = l(q). Reading the m-entry row of
            the map afterwards is a table index and is not timed.
  hnsw_low  the cheapest configuration, k = 100, timed the same way, so the
            lookup can be read against the search it precedes.

TWO WAYS, BOTH RECORDED
  batched   the whole 5,000-query sample in batches of 4,096 at --threads
            (64), median of repeated passes -- identical to bench_throughput.
            Gives an amortised seconds/query, comparable to the H5 cost basis.
            Done for the lookup at every K in --ks.
  single    one query per search call, each timed on its own with
            perf_counter, after a warm-up; mean, p50, p99 over the sample.
            Done at 1 thread and at --threads, for the lookup at K = 2500 and
            for hnsw_low. This is the only way to get a per-query p99.

CHECK
  The timed lookup must be the real one: its nearest landmark for the sample
  is compared with T5's stored assignment (queries/nn_C_*.npz) and the
  agreement rate is recorded. Anything but ~100% means the wrong landmarks or
  the wrong queries were loaded.

V2 (--params-json, the second run in result_v2/)
  Landmarks from --kmeans-root (result_v2/kmeans/7M_cleaned), queries from the
  v2 embedding trees, hnsw_low from --index-root with its CHOSEN efSearch set
  after the load (the v2 file carries only the build's start value). v2 has no
  stored query-to-landmark assignment yet, so the CHECK compares the timed
  lookup with an independent float64 numpy argmin over the same landmarks.
  Without --params-json the script is v1's.

Output: result/paper_experiments/7M_cleaned/<model>/lookup.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import faiss

REPO_ROOT = Path(__file__).resolve().parent.parent
for sub in ("indexes", "knn", "spatialmap"):
    sys.path.insert(0, str(REPO_ROOT / sub))

from index_configs import MODEL_DIMS, configs_for                      # noqa: E402
from load_vectors import load_selected                               # noqa: E402
from verify_indexes import read_search_param                         # noqa: E402
from approx_knn import assert_matches_catalogue, write_json_atomic   # noqa: E402
from map_config import PAIRINGS, layout, load_query_geometry         # noqa: E402
from v2_dirs import configs_v2, embed_dir                            # noqa: E402

SCALE = "7M_cleaned"
R = REPO_ROOT / "result"
BATCH = 4096          # bench_throughput's batch
K_MAIN = 2500


def cpu_model() -> str:
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


def batched(index, q: np.ndarray, k: int, warmup: int, min_window: float) -> dict:
    """bench_throughput.measure, with k as a parameter."""
    def one_pass() -> float:
        t0 = time.perf_counter()
        for s in range(0, q.shape[0], BATCH):
            index.search(np.ascontiguousarray(q[s:s + BATCH]), k)
        return time.perf_counter() - t0

    index.search(np.ascontiguousarray(q[:warmup]), k)
    passes = [one_pass()]
    if passes[0] < min_window:
        while sum(passes) < min_window or len(passes) < 3:
            passes.append(one_pass())
    med = statistics.median(passes)
    return {"n_passes": len(passes), "median_pass_seconds": round(med, 6),
            "pass_cv": (round(statistics.pstdev(passes) / statistics.mean(passes), 4)
                        if len(passes) > 1 else None),
            "s_per_query": med / q.shape[0], "us_per_query": round(med / q.shape[0] * 1e6, 3)}


def single(index, q: np.ndarray, k: int, warmup: int) -> dict:
    for i in range(warmup):
        index.search(q[i:i + 1], k)
    ts = np.empty(q.shape[0])
    for i in range(q.shape[0]):
        x = q[i:i + 1]
        t0 = time.perf_counter()
        index.search(x, k)
        ts[i] = time.perf_counter() - t0
    us = ts * 1e6
    return {"n": int(len(us)), "mean_us": float(us.mean()),
            "p50_us": float(np.percentile(us, 50)), "p99_us": float(np.percentile(us, 99)),
            "max_us": float(us.max())}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(MODEL_DIMS))
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--ks", default="1250,2500,5000,10000,20000")
    ap.add_argument("--n-queries", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--min-window", type=float, default=20.0)
    ap.add_argument("--pilot-dir", type=Path, default=R / SCALE / "siglip2",
                    help="bench_throughput's --pilot-dir: C_ids and pair_index.csv")
    ap.add_argument("--out-dir", type=Path, default=R / "paper_experiments" / SCALE)
    ap.add_argument("--allow-login-node", action="store_true")
    ap.add_argument("--params-json", type=Path, default=None,
                    help="v2: chosen search values (final_params_chosen.json)")
    ap.add_argument("--kmeans-root", type=Path, default=None, help="v2: T5 landmark tree")
    ap.add_argument("--index-root", type=Path, default=None, help="v2: index tree")
    args = ap.parse_args()
    v2 = args.params_json is not None
    if v2 and (args.kmeans_root is None or args.index_root is None):
        raise SystemExit("--params-json needs --kmeans-root and --index-root")
    if not os.environ.get("PBS_JOBID") and not args.allow_login_node:
        raise SystemExit("Refusing to run outside a PBS job: this times searches and "
                         "loads HNSW indexes of up to ~47 GB. Submit "
                         "paperexp/jobs/paperexp_lookup_7m.pbs.")

    ks = [int(x) for x in args.ks.split(",")]
    env = {"host": os.uname().nodename, "pbs_jobid": os.environ.get("PBS_JOBID"),
           "cpu_model": cpu_model(), "cpu_count": os.cpu_count(),
           "affinity_cpus": len(os.sched_getaffinity(0)),
           "faiss_version": faiss.__version__,
           "faiss_compile_options": faiss.get_compile_options(), "python": sys.executable,
           "loadavg_at_start": [round(x, 2) for x in os.getloadavg()],
           "started": datetime.now().isoformat()}

    # The benchmark's sample, drawn the same way from the same file.
    c_ids = np.load(args.pilot_dir / "C_ids.npy")
    rng = np.random.default_rng(args.seed)
    n = min(args.n_queries, len(c_ids))
    sample = np.sort(rng.choice(c_ids, size=n, replace=False))
    order = np.argsort(c_ids)
    pos = order[np.searchsorted(c_ids[order], sample)]        # row of each in C order
    assert np.array_equal(c_ids[pos], sample)

    print("=" * 78)
    print(f"lookup benchmark -- model={args.model}  host {env['host']}  "
          f"cpu {env['cpu_model']}  threads {args.threads}")
    print(f"  sample {n:,} Q_test queries (seed {args.seed}), K sweep {ks}")
    print("=" * 78, flush=True)

    qvecs = {}
    for qmod in ("image", "text"):
        t0 = time.time()
        qvecs[qmod] = np.ascontiguousarray(
            load_selected(args.model, qmod, sample, args.pilot_dir / "pair_index.csv",
                          root=embed_dir(args.model, qmod) if v2 else None),
            dtype=np.float32)
        print(f"loaded {n:,} {qmod} query vectors in {time.time() - t0:.1f}s", flush=True)

    a_ids = np.load(args.pilot_dir / "A_ids.npy", mmap_mode="r")
    catalogue = {c.name: c for c in (configs_v2(args.model, args.params_json, args.index_root)
                                     if v2 else configs_for(args.model))}
    out = {"run": "v2" if v2 else "v1", "params_json": str(args.params_json) if v2 else None,"task": "paper experiments part 2: landmark lookup cost c_est",
           "model": args.model, "scale": SCALE, "environment": env,
           "sample": {"role": "C", "n": int(n), "seed": args.seed, "warmup": args.warmup,
                      "batch": BATCH, "threads_batched": args.threads,
                      "same_draw_as": "knn/bench_throughput.py (same file, seed, n)"},
           "pairings": {}}
    dest = args.out_dir / args.model
    dest.mkdir(parents=True, exist_ok=True)

    for imod in ("image", "text"):
        here = [p for p in sorted(PAIRINGS) if PAIRINGS[p]["index_modality"] == imod]

        # ---- the lookup, at every K ------------------------------------------
        for k in ks:
            lay = layout(args.model, k, True, pilot=R / SCALE,
                         kmeans=args.kmeans_root if v2 else R / "kmeans" / SCALE,
                         cf=R / "cf" / SCALE, out=R / "spatial_map" / SCALE)
            cent = np.ascontiguousarray(np.load(lay.centroids(imod)), dtype=np.float32)
            index = faiss.IndexFlatL2(cent.shape[1])
            index.add(cent)
            for p in here:
                q = qvecs[PAIRINGS[p]["query_modality"]]
                faiss.omp_set_num_threads(args.threads)
                _, got = index.search(q, 1)
                if v2:          # no stored assignment yet: an independent argmin
                    c64, q64 = cent.astype(np.float64), q.astype(np.float64)
                    stored = np.argmin((c64 * c64).sum(1)[None, :] - 2 * q64 @ c64.T, axis=1)
                else:
                    stored = load_query_geometry(lay, p, "C")["assign"][pos]
                agree = float((got[:, 0] == stored).mean())
                e = out["pairings"].setdefault(p, {"lookup_batched": {}, "single": {}})
                e["lookup_batched"][str(k)] = {
                    **batched(index, q, 1, min(args.warmup, n), args.min_window),
                    ("assignment_agreement_with_numpy" if v2 else "assignment_agreement_with_T5"): agree,
                    "K": k, "dim": int(cent.shape[1])}
                if k == K_MAIN:
                    for th in sorted({1, args.threads}):
                        faiss.omp_set_num_threads(th)
                        e["single"].setdefault(str(th), {})["lookup"] = single(
                            index, q, 1, min(args.warmup, n))
                print(f"  lookup {imod} K={k:>6} {PAIRINGS[p]['short']:<5} "
                      f"batched {e['lookup_batched'][str(k)]['us_per_query']:>9.3f} us/q  "
                      f"agree {agree:.4%}", flush=True)
                write_json_atomic(dest / "lookup.json", out)
            del index

        # ---- the cheapest configuration, single-query ---------------------
        cfg = catalogue["hnsw_low"]
        path = (args.index_root if v2 else R / "indexes" / SCALE) / args.model / imod / "hnsw_low.faiss"
        t0 = time.time()
        index = faiss.read_index(str(path))
        if v2:              # the file holds the build's start value only
            faiss.downcast_index(index.index).hnsw.efSearch = cfg.search["efSearch"]
        if index.ntotal != len(a_ids):
            raise RuntimeError(f"{path}: ntotal {index.ntotal:,} != |A| {len(a_ids):,}")
        assert_matches_catalogue(index, cfg, path, MODEL_DIMS[args.model])
        pname, pval = read_search_param(index, cfg)
        if pval != cfg.search[pname]:
            raise RuntimeError(f"{path}: {pname}={pval}, catalogue says {cfg.search[pname]}")
        print(f"  hnsw_low {imod}: {pname}={pval}, loaded in {time.time() - t0:.0f}s", flush=True)
        for p in here:
            q = qvecs[PAIRINGS[p]["query_modality"]]
            for th in sorted({1, args.threads}):
                faiss.omp_set_num_threads(th)
                out["pairings"][p]["single"].setdefault(str(th), {})["hnsw_low"] = {
                    **single(index, q, 100, min(args.warmup, n)), pname: pval}
                s = out["pairings"][p]["single"][str(th)]
                print(f"    {PAIRINGS[p]['short']:<5} threads={th:<3} lookup mean "
                      f"{s['lookup']['mean_us']:8.1f} p99 {s['lookup']['p99_us']:8.1f} | "
                      f"hnsw_low mean {s['hnsw_low']['mean_us']:8.1f} "
                      f"p99 {s['hnsw_low']['p99_us']:8.1f} us", flush=True)
            write_json_atomic(dest / "lookup.json", out)
        del index

    env["finished"] = datetime.now().isoformat()
    env["loadavg_at_end"] = [round(x, 2) for x in os.getloadavg()]
    out["complete"] = True
    write_json_atomic(dest / "lookup.json", out)
    print(f"\nwrote {dest / 'lookup.json'}")


if __name__ == "__main__":
    main()
