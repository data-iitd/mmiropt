"""
Controlled search-throughput benchmark: the cost basis for H5, measured properly.

    python knn/bench_throughput.py --model siglip2 --scale-tag 7M_cleaned \
        --pilot-dir  result/7M_cleaned/siglip2 \
        --index-root result/indexes/7M_cleaned \
        --out-dir    result/throughput/7M_cleaned \
        --threads 64

WHY THIS EXISTS
    H5 orders configurations cheapest-first by T7's queries_per_second. Those
    numbers were taken while T7 did its real work: 15 of 16 cells had their nine
    configurations timed across a median of five different PBS jobs, on nodes
    shared with four to eight other jobs. A configuration could rank as cheap
    because it landed on a quiet node. This measures every configuration of one
    model in one job, on one node, on one fixed query sample, at one thread
    count, so that within a cell the only thing that differs is the index.

WHAT IS HELD FIXED
    - the queries: a seeded draw of --n-queries role-C pair-ids, identical for
      every configuration and pairing (and, since the split is shared, every
      model). Role C because H5 applies these costs to C's picks.
    - the index: each file is checked against the catalogue exactly as T7 does,
      and its search parameter is READ from the file, never set.
    - the search call: T7's batch size (4096) and k=100, CPU FAISS.
    - the threads: --threads, required, recorded.

WHAT IS TIMED
    Wall time of the batched search over the whole sample, measured with
    perf_counter and nothing printed inside the timed region. Index load time
    is recorded separately and excluded. A warm-up of --warmup queries runs
    first and is not timed.

REPEATS
    A pass that takes under --min-window seconds is too short to trust (thread
    spin-up, cache warm-up), so short passes repeat until the configuration has
    at least --min-window seconds of timed search and at least three passes;
    the median pass is reported. A pass already that long runs once: repeating
    a five-minute measurement triples the job for no gain in stability.

OUTPUT, WRITTEN AS IT GOES
    Each (pairing, configuration) is appended to records.jsonl the moment it is
    measured, and throughput.json -- including the cost table H5 needs,
    cost_s_per_query[pairing][config] -- is rewritten atomically after every
    record. A job killed part-way leaves every finished measurement usable.

    The load average is recorded before and after every configuration. On an
    exclusive node it stays near zero between configurations; if it does not,
    the node was not exclusive and the numbers should not be trusted.

V2 (--params-json, the second run in result_v2/)
    The configs come from the v2 build.json files and the chosen search values
    (indexes/v2_dirs.py configs_v2) instead of index_configs.py, and the query
    vectors from the v2 embedding trees. The v2 files carry only the build's
    starting value, so the chosen value is SET after the load, read back and
    recorded -- the only difference in what is timed. Without --params-json
    the configs, queries and timed passes are v1's.

    python knn/bench_throughput.py --model siglip2 --scale-tag 7M_cleaned \
        --params-json result_v2/param_check/7M_cleaned/final_params_chosen.json \
        --pilot-dir  result_v2/7M_cleaned/siglip2 \
        --index-root result_v2/indexes/7M_cleaned \
        --out-dir    result_v2/throughput/7M_cleaned --threads 64

TIMING SPLIT (both runs)
    After the timed passes, one more untimed pass per (pairing, config) reads
    FAISS's own counters: for IVF the time of the coarse quantizer and of the
    list scan (FAISS sums these over its parallel query slices, so the split is given as
    shares of quantizer + scan; OPQ rotation and call overhead are not in
    either counter) and the codes scanned; for HNSW the distances computed and
    the graph hops. Kept out of the timed passes, so it cannot move the cost.
"""

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
sys.path.insert(0, str(REPO_ROOT / "indexes"))

from index_configs import MODEL_DIMS, configs_for                      # noqa: E402
from load_vectors import load_selected                               # noqa: E402
from verify_indexes import read_search_param                         # noqa: E402

from exact_knn import PAIRINGS                                       # noqa: E402
from approx_knn import assert_matches_catalogue, write_json_atomic   # noqa: E402

BATCH = 4096      # T7's --batch default: identical search calls to the ones graded
K = 100                 # --k overrides (v3_k10: fidelity@10 is timed with k=10 searches)


def cpu_model() -> str:
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


def timed_search(index, q: np.ndarray) -> float:
    """Seconds for one batched pass over q. Nothing is printed in here."""
    t0 = time.perf_counter()
    for s in range(0, q.shape[0], BATCH):
        index.search(np.ascontiguousarray(q[s:s + BATCH]), K)
    return time.perf_counter() - t0


def measure(index, q: np.ndarray, warmup: int, min_window: float) -> dict:
    index.search(np.ascontiguousarray(q[:warmup]), K)          # warm-up, untimed
    passes = [timed_search(index, q)]
    if passes[0] < min_window:
        while sum(passes) < min_window or len(passes) < 3:
            passes.append(timed_search(index, q))
    med = statistics.median(passes)
    return {
        "passes_seconds": [round(p, 4) for p in passes],
        "n_passes": len(passes),
        "median_pass_seconds": round(med, 4),
        "pass_cv": (round(statistics.pstdev(passes) / statistics.mean(passes), 4)
                    if len(passes) > 1 else None),
        "s_per_query": med / q.shape[0],
        "us_per_query": round(med / q.shape[0] * 1e6, 2),
        "queries_per_second": round(q.shape[0] / med, 2),
    }


def split_pass(index, q: np.ndarray, is_hnsw: bool) -> dict:
    """One untimed pass with FAISS's counters on: where the time of a search goes."""
    if is_hnsw:
        st = faiss.cvar.hnsw_stats
        st.reset()
        t0 = time.perf_counter()
        for s in range(0, q.shape[0], BATCH):
            index.search(np.ascontiguousarray(q[s:s + BATCH]), K)
        wall = time.perf_counter() - t0
        n = q.shape[0]
        return {"pass_seconds": round(wall, 4), "distances_per_query": round(st.ndis / n, 1),
                "hops_per_query": round(st.nhops / n, 1)}
    st = faiss.cvar.indexIVF_stats
    st.reset()
    t0 = time.perf_counter()
    for s in range(0, q.shape[0], BATCH):
        index.search(np.ascontiguousarray(q[s:s + BATCH]), K)
    wall = time.perf_counter() - t0
    # FAISS adds these up over the query slices it runs in parallel, so they are
    # thread-seconds, not wall seconds: only their RATIO is compared to the pass.
    n, quant, scan = q.shape[0], st.quantization_time / 1e3, st.search_time / 1e3
    return {"pass_seconds": round(wall, 4),
            "quantizer_counter_s": round(quant, 4), "list_scan_counter_s": round(scan, 4),
            "quantizer_share": round(quant / max(quant + scan, 1e-12), 4),
            "list_scan_share": round(scan / max(quant + scan, 1e-12), 4),
            "codes_scanned_per_query": round(st.ndis / n, 1),
            "lists_probed_per_query": round(st.nlist / n, 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(MODEL_DIMS))
    ap.add_argument("--scale-tag", required=True,
                    help="e.g. 7M_cleaned; also the output subdirectory")
    ap.add_argument("--pilot-dir", type=Path, required=True,
                    help="the split directory holding A_ids/C_ids/pair_index.csv")
    ap.add_argument("--index-root", type=Path, required=True,
                    help="<root>/<model>/<image|text>/<config>.faiss")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="results go to <out-dir>/<model>/")
    ap.add_argument("--threads", type=int, required=True,
                    help="FAISS OpenMP threads, held fixed for every config")
    ap.add_argument("--pairing", nargs="+", default=["all"])
    ap.add_argument("--configs", default="all")
    ap.add_argument("--k", type=int, default=100, help="neighbours per search call")
    ap.add_argument("--n-queries", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--min-window", type=float, default=20.0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--allow-login-node", action="store_true")
    ap.add_argument("--params-json", type=Path, default=None,
                    help="v2: chosen search values (final_params_chosen.json); configs "
                         "from the v2 build.json files, queries from the v2 embeddings")
    args = ap.parse_args()
    global K
    K = args.k

    if not os.environ.get("PBS_JOBID") and not args.allow_login_node:
        raise SystemExit("Refusing to run outside a PBS job: this loads indexes of "
                         "up to ~47 GB and times searches on them. Submit a job "
                         "on an exclusive node.")

    faiss.omp_set_num_threads(args.threads)
    pairings = (sorted(PAIRINGS) if "all" in args.pairing
                else list(dict.fromkeys(args.pairing)))
    v2 = args.params_json is not None
    if v2:
        from v2_dirs import configs_v2, embed_dir                      # noqa: E402
        catalogue = {c.name: c for c in configs_v2(args.model, args.params_json, args.index_root)}
    else:
        catalogue = {c.name: c for c in configs_for(args.model)}
    names = (list(catalogue) if args.configs == "all"
             else args.configs.split(","))
    bad = [n for n in names if n not in catalogue]
    if bad:
        raise SystemExit(f"unknown config(s): {bad}")

    dest = args.out_dir / args.model
    dest.mkdir(parents=True, exist_ok=True)
    rec_path, sum_path = dest / "records.jsonl", dest / "throughput.json"
    if rec_path.exists() and rec_path.stat().st_size and not args.overwrite:
        raise SystemExit(f"{rec_path} already has measurements. Every number in "
                         f"one file must come from one job on one node, so this "
                         f"will not append to it -- pass --overwrite to start over.")
    rec_path.write_text("")

    env = {
        "host": os.uname().nodename,
        "pbs_jobid": os.environ.get("PBS_JOBID"),
        "cpu_model": cpu_model(),
        "cpu_count": os.cpu_count(),
        "affinity_cpus": len(os.sched_getaffinity(0)),
        "faiss_threads": faiss.omp_get_max_threads(),
        "faiss_version": faiss.__version__,
        "faiss_compile_options": faiss.get_compile_options(),
        "python": sys.executable,
        "loadavg_at_start": [round(x, 2) for x in os.getloadavg()],
        "started": datetime.now().isoformat(),
    }

    a_ids = np.load(args.pilot_dir / "A_ids.npy", mmap_mode="r")
    c_ids = np.load(args.pilot_dir / "C_ids.npy")
    rng = np.random.default_rng(args.seed)
    n = min(args.n_queries, len(c_ids))
    sample = np.sort(rng.choice(c_ids, size=n, replace=False))

    print("=" * 78)
    print(f"throughput benchmark -- model={args.model}  scale={args.scale_tag}")
    print(f"  host {env['host']}  job {env['pbs_jobid']}  cpu {env['cpu_model']}")
    print(f"  threads {env['faiss_threads']}  sample {n:,} C queries (seed {args.seed})"
          f"  warm-up {args.warmup}  min window {args.min_window:g}s")
    print(f"  load average at start {env['loadavg_at_start']}")
    print(f"  out {dest}")
    print("=" * 78)

    qvecs = {}
    for qmod in sorted({PAIRINGS[p]["query_modality"] for p in pairings}):
        t0 = time.time()
        qvecs[qmod] = load_selected(args.model, qmod, sample,
                                    args.pilot_dir / "pair_index.csv",
                                    root=embed_dir(args.model, qmod) if v2 else None)
        print(f"loaded {n:,} {qmod} query vectors in {time.time() - t0:.1f}s")

    records, cost = [], {}

    def publish():
        write_json_atomic(sum_path, {
            "task": "controlled search-throughput benchmark (H5 cost basis)",
            "model": args.model, "scale": args.scale_tag, "environment": env,
            "run": "v2" if v2 else "v1", "params_json": str(args.params_json) if v2 else None,
            "split_dir": str(args.pilot_dir), "index_root": str(args.index_root),
            "sample": {"role": "C", "n": int(n), "seed": args.seed,
                       "warmup": args.warmup, "batch": BATCH, "k": K},
            "repeat_rule": (f"passes shorter than {args.min_window:g}s repeat until "
                            f">= {args.min_window:g}s timed and >= 3 passes; median "
                            f"reported. Longer passes run once."),
            "complete": len(records) == len(pairings) * len(names),
            "n_records": len(records),
            "cost_s_per_query": cost,
            "records": records,
            "updated": datetime.now().isoformat(),
        })

    t_all = time.time()
    for imod in sorted({PAIRINGS[p]["index_modality"] for p in pairings}):
        here = [p for p in pairings if PAIRINGS[p]["index_modality"] == imod]
        for name in names:
            cfg = catalogue[name]
            path = args.index_root / args.model / imod / f"{name}.faiss"
            t0 = time.time()
            index = faiss.read_index(str(path))
            load_s = time.time() - t0
            if index.ntotal != len(a_ids):
                raise RuntimeError(f"{path}: ntotal {index.ntotal:,} != |A| "
                                   f"{len(a_ids):,} -- wrong split for this index")
            assert_matches_catalogue(index, cfg, path, MODEL_DIMS[args.model])
            if v2:                       # the file holds the build's start value only
                (pn, pv), = cfg.search.items()
                if cfg.is_hnsw:
                    faiss.downcast_index(index.index).hnsw.efSearch = pv
                else:
                    faiss.extract_index_ivf(index).nprobe = pv
            pname, pval = read_search_param(index, cfg)
            if pval != cfg.search[pname]:
                raise RuntimeError(f"{path}: {pname}={pval} in file, catalogue says "
                                   f"{cfg.search[pname]} -- stale file")
            print(f"\n[{imod}/{name}] {pname}={pval}  "
                  f"({path.stat().st_size / 2**30:.1f} GiB read in {load_s:.1f}s)")

            for p in here:
                q = qvecs[PAIRINGS[p]["query_modality"]]
                la0 = [round(x, 2) for x in os.getloadavg()]
                m = measure(index, q, min(args.warmup, n), args.min_window)
                la1 = [round(x, 2) for x in os.getloadavg()]
                split = split_pass(index, q, cfg.is_hnsw)
                rec = {"model": args.model, "scale": args.scale_tag,
                       "pairing": p, "pairing_short": PAIRINGS[p]["short"],
                       "config": name, "search_param": {pname: pval},
                       "index_file": str(path), "index_bytes": path.stat().st_size,
                       "index_load_seconds": round(load_s, 2),
                       "n_queries": int(n), "threads": env["faiss_threads"],
                       **m, "timing_split": split,
                       "search_param_source": str(args.params_json) if v2 else "read from the index file",
                       "loadavg_before": la0, "loadavg_after": la1,
                       "host": env["host"], "pbs_jobid": env["pbs_jobid"],
                       "measured_at": datetime.now().isoformat()}
                records.append(rec)
                cost.setdefault(p, {})[name] = rec["s_per_query"]
                with open(rec_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                publish()
                cv = f"cv {m['pass_cv']:.3f}" if m["pass_cv"] is not None else "single pass"
                print(f"    {PAIRINGS[p]['short']:<6} {m['us_per_query']:>11,.1f} us/q  "
                      f"{m['queries_per_second']:>9,.1f} q/s  "
                      f"{m['n_passes']} pass(es), {cv}  load {la0[0]}->{la1[0]}  "
                      + (f"dist/q {split['distances_per_query']:,.0f}" if cfg.is_hnsw else
                         f"quant {split['quantizer_share']:.0%} scan {split['list_scan_share']:.0%} "
                         f"codes/q {split['codes_scanned_per_query']:,.0f}"))
            del index

    env["finished"] = datetime.now().isoformat()
    env["loadavg_at_end"] = [round(x, 2) for x in os.getloadavg()]
    publish()
    print(f"\ndone: {len(records)} measurements in {time.time() - t_all:.0f}s -> {sum_path}")


if __name__ == "__main__":
    main()
