"""
Single-query landmark lookup (Step 1), exact scan vs HNSW, timed INTERLEAVED.

    python paperexp/lookup_interleaved.py --model jinav4        (inside a PBS job)

Why (user 2026-10-31): the HNSW sweep (split_single_query.py --hnsw-sweep) times the
exact scan first and the HNSW settings after it. On a shared node the load can change
between the two (Jina-v4 / Qwen3-VL i->i: exact scan ~1.9x slower than in the other
directions, median = mean, so a sustained slowdown, not outliers or a cold start), so
the in-job speed-up compares times taken under different conditions. Here both lookups
are timed query by query in alternation -- exact then HNSW for even queries, HNSW then
exact for odd ones -- so every pair sees the same node load, and the ratio is fair on a
shared node.

Same as the sweep: the 5,000-query Q_test sample (seed 0), v2 query vectors, the K
landmarks of the index modality, IndexFlatL2 (exact) vs IndexHNSWFlat (L2,
efConstruction 200) with the setting hnsw_choice.choose_uniform picks from the sweep
file, one thread, search(x, 1), untimed warm-up of 500 queries per lookup. Three full
interleaved passes; per pass the mean, p50 and p99 of each lookup; the reported values
are the medians over passes, and the speed-up is exact mean / HNSW mean (also the median
of the per-query ratios). Cell agreement on the sample is recorded as a check against
the sweep.

Output: <out-dir>/<model>/lookup_interleaved.json
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
for sub in ("paperexp", "indexes", "knn", "spatialmap"):
    sys.path.insert(0, str(REPO_ROOT / sub))

from index_configs import MODEL_DIMS                                  # noqa: E402
from load_vectors import load_selected                               # noqa: E402
from v2_dirs import embed_dir                                        # noqa: E402
from map_config import PAIRINGS                                      # noqa: E402
from selection_tables import R, SCALE, lay_at                        # noqa: E402
from bench_estimate import cpu_model                                 # noqa: E402
from hnsw_choice import choose_uniform, parse, sweep_name            # noqa: E402

EF_CONSTRUCTION = 200


def stats(us: np.ndarray) -> dict:
    return {"mean_us": float(us.mean()), "p50_us": float(np.percentile(us, 50)),
            "p99_us": float(np.percentile(us, 99))}


def interleaved_pass(flat, hnsw, q) -> tuple[np.ndarray, np.ndarray]:
    n = q.shape[0]
    tf, th = np.empty(n), np.empty(n)
    pc = time.perf_counter
    for i in range(n):
        x = q[i:i + 1]
        if i % 2 == 0:
            a = pc(); flat.search(x, 1); b = pc(); hnsw.search(x, 1); c = pc()
            tf[i], th[i] = b - a, c - b
        else:
            a = pc(); hnsw.search(x, 1); b = pc(); flat.search(x, 1); c = pc()
            th[i], tf[i] = b - a, c - b
    return tf * 1e6, th * 1e6


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(MODEL_DIMS))
    ap.add_argument("--k", type=int, default=10000)
    ap.add_argument("--sweep-dir", type=Path, required=True,
                    help="tree holding <model>/split_lookup_variants_K<k>.json")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--pilot-dir", type=Path, default=R / SCALE / "siglip2")
    ap.add_argument("--n-queries", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--allow-login-node", action="store_true")
    args = ap.parse_args()
    if not os.environ.get("PBS_JOBID") and not args.allow_login_node:
        raise SystemExit("Refusing to run outside a PBS job")

    sweep = json.loads((args.sweep_dir / args.model / sweep_name(args.k)).read_text())
    choice = choose_uniform(sweep)
    env = {"host": os.uname().nodename, "pbs_jobid": os.environ.get("PBS_JOBID"),
           "cpu_model": cpu_model(), "affinity_cpus": len(os.sched_getaffinity(0)),
           "faiss_version": faiss.__version__, "python": sys.executable,
           "loadavg_at_start": [round(x, 2) for x in os.getloadavg()],
           "started": datetime.now().isoformat()}
    c_ids = np.load(args.pilot_dir / "C_ids.npy")
    rng = np.random.default_rng(args.seed)
    sample = np.sort(rng.choice(c_ids, size=min(args.n_queries, len(c_ids)), replace=False))
    qvecs = {qm: np.ascontiguousarray(
        load_selected(args.model, qm, sample, args.pilot_dir / "pair_index.csv",
                      root=embed_dir(args.model, qm)), dtype=np.float32)
        for qm in ("image", "text")}

    out = {"task": "single-query landmark lookup, exact vs HNSW, interleaved timing",
           "model": args.model, "K": args.k, "environment": env, "choice_rule": "choose_uniform",
           "sample": {"role": "C", "n": int(len(sample)), "seed": args.seed, "warmup": args.warmup,
                      "passes": args.passes, "threads": 1},
           "pairings": {}}
    dest = args.out_dir / args.model
    dest.mkdir(parents=True, exist_ok=True)
    lay = lay_at(args.model, args.k)
    for p in sorted(PAIRINGS):
        spec = PAIRINGS[p]
        M, ef = parse(choice[p])
        cent = np.ascontiguousarray(np.load(lay.centroids(spec["index_modality"])), dtype=np.float32)
        flat = faiss.IndexFlatL2(cent.shape[1]); flat.add(cent)
        faiss.omp_set_num_threads(16)
        h = faiss.IndexHNSWFlat(cent.shape[1], M, faiss.METRIC_L2)
        h.hnsw.efConstruction = EF_CONSTRUCTION
        h.add(cent)
        h.hnsw.efSearch = ef
        q = qvecs[spec["query_modality"]]
        agree = float((flat.search(q, 1)[1][:, 0] == h.search(q, 1)[1][:, 0]).mean())
        faiss.omp_set_num_threads(1)
        for i in range(min(args.warmup, len(q))):
            flat.search(q[i:i + 1], 1); h.search(q[i:i + 1], 1)
        passes = []
        for _ in range(args.passes):
            la = os.getloadavg()[0]
            tf, th = interleaved_pass(flat, h, q)
            passes.append({"flat": stats(tf), "hnsw": stats(th),
                           "speedup_of_means": float(tf.mean() / th.mean()),
                           "median_per_query_ratio": float(np.median(tf / th)), "loadavg_1m": round(la, 2)})
        med = lambda f: float(statistics.median(f(x) for x in passes))
        r = {"setting": choice[p], "cell_agreement_sample": agree, "passes": passes,
             "flat_mean_us": med(lambda x: x["flat"]["mean_us"]), "flat_p99_us": med(lambda x: x["flat"]["p99_us"]),
             "hnsw_mean_us": med(lambda x: x["hnsw"]["mean_us"]), "hnsw_p99_us": med(lambda x: x["hnsw"]["p99_us"]),
             "speedup": med(lambda x: x["speedup_of_means"]),
             "speedup_median_per_query": med(lambda x: x["median_per_query_ratio"])}
        out["pairings"][p] = r
        print(f"  {spec['short']:<5} {choice[p]:<10} exact {r['flat_mean_us']:8.1f} us  HNSW {r['hnsw_mean_us']:7.1f} us"
              f"  speed-up {r['speedup']:.2f}x (per-query median {r['speedup_median_per_query']:.2f}x)"
              f"  passes {[round(x['speedup_of_means'], 2) for x in passes]}  cell agree {agree:.4f}", flush=True)
        (dest / "lookup_interleaved.json").write_text(json.dumps(out, indent=1))
    env["loadavg_at_end"] = [round(x, 2) for x in os.getloadavg()]
    env["finished"] = datetime.now().isoformat()
    out["complete"] = True
    (dest / "lookup_interleaved.json").write_text(json.dumps(out, indent=1))
    print(f"wrote {dest / 'lookup_interleaved.json'}")


if __name__ == "__main__":
    main()
