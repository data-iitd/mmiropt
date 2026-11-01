"""
BATCHED landmark lookup (Step 1), exact scan vs HNSW, timed INTERLEAVED pass by pass.

    python paperexp/lookup_batched_interleaved.py --model jinav4        (inside a PBS job)

Why (user 2026-10-31): the sweep's batched numbers (split_single_query.py --hnsw-sweep)
time the exact and the HNSW lookup at different moments on a shared node. Here, as in
lookup_interleaved.py for single queries, one batched pass of the exact lookup and one of
HNSW alternate until both have >= --min-window seconds and >= 3 passes; the median pass
per lookup is divided by the number of queries. Both see the same node load.

Exact lookup: the batched Step 1 of the paper's c_est (bench_estimate.lookup_fn, numpy
matmul argmin). HNSW: IndexHNSWFlat (L2, efConstruction 200) with the model's uniform
setting (hnsw_choice.choose_uniform). 5,000-query Q_test sample (seed 0), batches of
4,096, --threads (64).

Part 1: K = 10,000 (the paper's), all four directions.
Part 2 (cache check): the same HNSW setting at the other --ks, image->image only, so the
landmark set's size (K x d x 4 bytes) moves across the 32 MiB L3 of one 8-core group.
The batched gain (HNSW 1-query time / HNSW batched time per query) is recorded, with
the 1-query time measured in the same job (--single-n queries, one thread).

--no-interleave (user 2026-10-31): the interleaved passes slowed both lookups (exact
40-73 us/q against 21-32 exclusive; HNSW stuck near 40 us/q even at 7 MiB), probably
because numpy's OpenBLAS pool and FAISS's OpenMP pool spin-wait on the same cores
(not verified). With this flag each lookup is timed in one block of passes
(>= min-window s, >= 3 passes, median), the exact lookup first, as the sweep and the
c_est benchmark do.

Output: <out-dir>/<model>/lookup_batched_interleaved.json (lookup_batched_block.json
with --no-interleave)
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
from bench_estimate import cpu_model, lookup_fn, BATCH               # noqa: E402
from hnsw_choice import choose_uniform, parse, sweep_name            # noqa: E402

EF_CONSTRUCTION = 200


def one_pass(fn, q) -> float:
    t0 = time.perf_counter()
    for s in range(0, q.shape[0], BATCH):
        fn(np.ascontiguousarray(q[s:s + BATCH]))
    return time.perf_counter() - t0


def interleaved(fa, fb, q, min_window: float) -> dict:
    fa(q[:500]); fb(q[:500])                                     # warm-up, untimed
    pa, pb = [], []
    while sum(pa) < min_window or sum(pb) < min_window or len(pa) < 3:
        pa.append(one_pass(fa, q)); pb.append(one_pass(fb, q))
    n = q.shape[0]
    ma, mb = statistics.median(pa), statistics.median(pb)
    return {"exact_us_per_query": ma / n * 1e6, "hnsw_us_per_query": mb / n * 1e6,
            "n_passes": len(pa), "exact_passes_s": [round(x, 4) for x in pa],
            "hnsw_passes_s": [round(x, 4) for x in pb],
            "pass_ratio_exact_over_hnsw": [round(a / b, 3) for a, b in zip(pa, pb)]}


def block(fa, fb, q, min_window: float) -> dict:
    n = q.shape[0]
    res = {}
    for name, fn in (("exact", fa), ("hnsw", fb)):
        fn(q[:500])                                              # warm-up, untimed
        ps = [one_pass(fn, q)]
        while sum(ps) < min_window or len(ps) < 3:
            ps.append(one_pass(fn, q))
        res[f"{name}_us_per_query"] = statistics.median(ps) / n * 1e6
        res[f"{name}_passes_s"] = [round(x, 4) for x in ps]
        res[f"{name}_pass_cv"] = (statistics.pstdev(ps) / statistics.mean(ps)) if len(ps) > 1 else None
    res["pass_ratio_exact_over_hnsw"] = None
    return res


def single_us(index, q, n: int) -> float:
    faiss.omp_set_num_threads(1)
    for i in range(min(200, n)):
        index.search(q[i:i + 1], 1)
    t = np.empty(n)
    for i in range(n):
        a = time.perf_counter(); index.search(q[i:i + 1], 1); t[i] = time.perf_counter() - a
    return float(t.mean() * 1e6)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(MODEL_DIMS))
    ap.add_argument("--sweep-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--pilot-dir", type=Path, default=R / SCALE / "siglip2")
    ap.add_argument("--ks", default="2500,5000,20000")
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--n-queries", type=int, default=5000)
    ap.add_argument("--single-n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-window", type=float, default=20.0)
    ap.add_argument("--no-interleave", action="store_true", help="time each lookup in one block of passes")
    ap.add_argument("--allow-login-node", action="store_true")
    args = ap.parse_args()
    if not os.environ.get("PBS_JOBID") and not args.allow_login_node:
        raise SystemExit("Refusing to run outside a PBS job")

    choice = choose_uniform(json.loads((args.sweep_dir / args.model / sweep_name(10000)).read_text()))
    setting = set(choice.values()); assert len(setting) == 1, setting
    setting = setting.pop(); M, ef = parse(setting)
    env = {"host": os.uname().nodename, "pbs_jobid": os.environ.get("PBS_JOBID"), "cpu_model": cpu_model(),
           "affinity_cpus": len(os.sched_getaffinity(0)), "threads": args.threads,
           "faiss_version": faiss.__version__, "numpy_version": np.__version__,
           "loadavg_at_start": [round(x, 2) for x in os.getloadavg()], "started": datetime.now().isoformat()}
    c_ids = np.load(args.pilot_dir / "C_ids.npy")
    sample = np.sort(np.random.default_rng(args.seed).choice(c_ids, size=min(args.n_queries, len(c_ids)), replace=False))
    qv = {qm: np.ascontiguousarray(load_selected(args.model, qm, sample, args.pilot_dir / "pair_index.csv",
                                                 root=embed_dir(args.model, qm)), dtype=np.float32)
          for qm in ("image", "text")}
    out = {"task": "batched landmark lookup, exact vs HNSW, interleaved passes", "model": args.model,
           "setting": setting, "environment": env, "batch": BATCH, "min_window_s": args.min_window, "runs": []}
    dest = args.out_dir / args.model; dest.mkdir(parents=True, exist_ok=True)
    fname = "lookup_batched_block.json" if args.no_interleave else "lookup_batched_interleaved.json"
    out["mode"] = "block" if args.no_interleave else "interleaved"
    jobs = [(10000, p) for p in sorted(PAIRINGS)] + [(int(k), "image_query_image_index") for k in args.ks.split(",")]
    for K, p in jobs:
        spec = PAIRINGS[p]
        cent = np.ascontiguousarray(np.load(lay_at(args.model, K).centroids(spec["index_modality"])), dtype=np.float32)
        faiss.omp_set_num_threads(args.threads)
        h = faiss.IndexHNSWFlat(cent.shape[1], M, faiss.METRIC_L2)
        h.hnsw.efConstruction = EF_CONSTRUCTION; h.add(cent); h.hnsw.efSearch = ef
        q = qv[spec["query_modality"]]
        exact = lookup_fn(cent)
        hnsw = lambda x: h.search(x, 1)[1][:, 0]
        faiss.omp_set_num_threads(args.threads)
        r = (block if args.no_interleave else interleaved)(exact, hnsw, q, args.min_window)
        s1 = single_us(h, q, args.single_n)
        faiss.omp_set_num_threads(args.threads)
        r.update({"K": K, "pairing": p, "landmark_mib": cent.nbytes / 2**20, "hnsw_single_us": s1,
                  "batched_gain_hnsw": s1 / r["hnsw_us_per_query"],
                  "exact_over_hnsw_batched": r["exact_us_per_query"] / r["hnsw_us_per_query"],
                  "loadavg_1m": round(os.getloadavg()[0], 2)})
        out["runs"].append(r)
        print(f"  K={K:<6} {spec['short']:<5} {r['landmark_mib']:6.1f} MiB | exact {r['exact_us_per_query']:7.2f} us/q"
              f" | HNSW {r['hnsw_us_per_query']:7.2f} us/q (1 query {s1:6.1f} us, gain {r['batched_gain_hnsw']:5.1f}x)"
              f" | exact/HNSW {r['exact_over_hnsw_batched']:.2f} per-pass {r['pass_ratio_exact_over_hnsw']}"
              f" cv {r.get('exact_pass_cv')}/{r.get('hnsw_pass_cv')}", flush=True)
        (dest / fname).write_text(json.dumps(out, indent=1))
    env["loadavg_at_end"] = [round(x, 2) for x in os.getloadavg()]; env["finished"] = datetime.now().isoformat()
    out["complete"] = True
    (dest / fname).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
