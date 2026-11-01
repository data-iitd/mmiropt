"""
Paper experiments, part 3: c_est exactly as the paper defines it -- the per-query
cost of computing the estimates and choosing a configuration, i.e. Steps 1-3 of
the online stage (Section 3.3):

    Step 1  locate the query: nearest landmark i = l(q), a scan of the K landmarks
    Step 2  read row i of the map (m coverage values pi_hat(q, A_j; tau))
    Step 3  select: the cheapest A_j with pi_hat >= rho, else the highest pi_hat

    python paperexp/bench_estimate.py --model siglip2        (inside a PBS job)

Step 4 (running the chosen configuration) is the configuration's own cost and is
not part of c_est.

Same sample, batch and threads as knn/bench_throughput.py and bench_lookup.py
(5,000 Q_test queries, seed 0; batch 4,096; 64 threads for batched timing).
The map row read in Step 2 is the real served table: pi_hat at tau = 0.7 built
from Q exactly as paperexp/selection_tables.py builds it (q_min = 10 backoff,
configurations cheapest first by the controlled benchmark), at rho = 0.9.

TIMED
  single    one query per call, Steps 1-3 inside one perf_counter window, at
            1 thread and at --threads; mean / p50 / p99 over the sample. Step 1
            alone is timed in a separate pass of the same loop, so the share of
            Steps 2-3 is visible. K = 2500 only. Step 1 is FAISS IndexFlatL2
            search; for one query FAISS does not call BLAS.
  batched   Steps 1-3 for the whole sample in batches of 4,096, median of
            repeated passes; every K in --ks. Step 1 is numpy's matmul,
                i = argmin_i' ( |l_i'|^2 - 2 q . l_i' )  ==  argmin_i' |q - l_i'|^2
            using numpy's own OpenBLAS on --threads threads.

WHY NOT FAISS FOR THE BATCHED LOOKUP
  The first run of this script (2026-09-29) used FAISS IndexFlatL2 for the
  batched lookup too. The OpenBLAS bundled with the installed FAISS wheel
  (0.3.3) runs on one thread with generic kernels on this CPU (threadpoolctl:
  threads=1, arch=Barcelona), so the lookup ran at ~41 GFLOP/s and the batched
  c_est came out ~20x too high. paperexp/diag_lookup_blas.py established this
  (result/paper_experiments/7M_cleaned/_diagnostics/lookup_blas.jsonl): numpy's
  matmul picks the same landmark for 100% of queries at 800-1,400 GFLOP/s.
  That run's estimate.json was overwritten by this one.

CHECKS
  The configuration picked for each sample query must equal the one the
  coverage rule picks from T5's stored assignment of that query -- the pick
  selection_tables.py evaluated. The agreement rate is recorded.
  The batched lookup's rate (2*K*d per query over the measured time) is
  recorded; below MIN_GFLOPS it is flagged parallel_ok = false.

Output: result/paper_experiments/7M_cleaned/<model>/estimate.json
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import faiss
from threadpoolctl import threadpool_info

REPO_ROOT = Path(__file__).resolve().parent.parent
for sub in ("paperexp", "indexes", "knn", "spatialmap"):
    sys.path.insert(0, str(REPO_ROOT / sub))

from index_configs import MODEL_DIMS                                  # noqa: E402
from load_vectors import load_selected                               # noqa: E402
from v2_dirs import embed_dir                                        # noqa: E402
from approx_knn import write_json_atomic                             # noqa: E402
from map_config import PAIRINGS, load_query_geometry, resolve_configs  # noqa: E402
from selection_tables import (Q_MIN, R, SCALE, bench_costs, cell_coverage,  # noqa: E402
                              cell_tables, lay_at, load_overlap)

BATCH = 4096
from selection_tables import K_MAIN, KS                               # noqa: E402  (SFM_K_MAIN / SFM_KS)
TAU, RHO = 0.70, 0.90
MIN_GFLOPS = 150.0     # a batched lookup below this is not running in parallel


def lookup_fn(cent: np.ndarray):
    """Batched Step 1 with numpy's matmul (see WHY NOT FAISS above)."""
    cn = (cent.astype(np.float32) ** 2).sum(axis=1)
    cT = np.ascontiguousarray(cent.T)

    def f(x: np.ndarray) -> np.ndarray:
        return np.argmin(cn[None, :] - 2.0 * (x @ cT), axis=1)
    return f


def cpu_model() -> str:
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


def select_row(row: np.ndarray, rho: float) -> int:
    """Step 3 for one query: first (cheapest) column >= rho, else argmax."""
    ok = np.flatnonzero(row >= rho)
    return int(ok[0]) if ok.size else int(np.argmax(row))


def select_rows(rows: np.ndarray, rho: float) -> np.ndarray:
    ok = rows >= rho
    idx = np.argmax(ok, axis=1)
    none = ~ok.any(axis=1)
    if none.any():
        idx[none] = np.argmax(rows[none], axis=1)
    return idx


def served_table(model, pairing, order, n_b, k):
    """pi_hat [K, m] at TAU, cheapest-first columns, with the q_min backoff."""
    lay = lay_at(model, k)
    ab = load_query_geometry(lay, pairing, "B")["assign"].astype(np.int64)
    ov_b = load_overlap(lay_at(model, K_MAIN), pairing, "B", order, n_b)
    t = int(round(TAU * 100))
    n, _ = cell_tables(ab, ov_b, k)
    pi = cell_coverage(ab, ov_b, k, n, t)
    pi_bar = (ov_b >= t).mean(axis=0)
    thin = n < Q_MIN
    return np.ascontiguousarray(np.where(thin[:, None], pi_bar[None, :], pi))


def single(index, table, q, warmup, steps_2_3: bool) -> dict:
    def one(x):
        _, I = index.search(x, 1)
        if steps_2_3:
            select_row(table[I[0, 0]], RHO)
    for i in range(warmup):
        one(q[i:i + 1])
    ts = np.empty(q.shape[0])
    for i in range(q.shape[0]):
        x = q[i:i + 1]
        t0 = time.perf_counter()
        one(x)
        ts[i] = time.perf_counter() - t0
    us = ts * 1e6
    return {"n": int(len(us)), "mean_us": float(us.mean()),
            "p50_us": float(np.percentile(us, 50)), "p99_us": float(np.percentile(us, 99)),
            "max_us": float(us.max())}


def batched(lookup, table, q, warmup, min_window) -> dict:
    def one_pass():
        t0 = time.perf_counter()
        for s in range(0, q.shape[0], BATCH):
            select_rows(table[lookup(q[s:s + BATCH])], RHO)
        return time.perf_counter() - t0
    select_rows(table[lookup(q[:warmup])], RHO)
    passes = [one_pass()]
    if passes[0] < min_window:
        while sum(passes) < min_window or len(passes) < 3:
            passes.append(one_pass())
    med = statistics.median(passes)
    return {"n_passes": len(passes), "median_pass_seconds": round(med, 6),
            "s_per_query": med / q.shape[0], "us_per_query": round(med / q.shape[0] * 1e6, 3)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(MODEL_DIMS))
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--ks", default=",".join(str(k) for k in KS))
    ap.add_argument("--v2", action="store_true",
                    help="read query vectors from the v2 embedding trees (indexes/v2_dirs.py); "
                         "set SFM_RESULTS_ROOT=result_v2 so every other input is v2 too")
    ap.add_argument("--n-queries", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--min-window", type=float, default=20.0)
    ap.add_argument("--pilot-dir", type=Path, default=R / SCALE / "siglip2")
    ap.add_argument("--out-dir", type=Path, default=R / "paper_experiments" / SCALE)
    ap.add_argument("--cost-dir", type=Path, default=None,
                    help="controlled-throughput tree (<dir>/<model>/throughput.json); "
                         "default result/throughput/7M_cleaned")
    ap.add_argument("--allow-login-node", action="store_true")
    args = ap.parse_args()
    if not os.environ.get("PBS_JOBID") and not args.allow_login_node:
        raise SystemExit("Refusing to run outside a PBS job -- submit "
                         "paperexp/jobs/paperexp_estimate_7m.pbs")

    ks = [int(x) for x in args.ks.split(",")]
    env = {"host": os.uname().nodename, "pbs_jobid": os.environ.get("PBS_JOBID"),
           "cpu_model": cpu_model(), "cpu_count": os.cpu_count(),
           "affinity_cpus": len(os.sched_getaffinity(0)),
           "faiss_version": faiss.__version__,
           "faiss_compile_options": faiss.get_compile_options(), "python": sys.executable, "numpy_version": np.__version__,
           "threadpools": threadpool_info(),
           "loadavg_at_start": [round(x, 2) for x in os.getloadavg()],
           "started": datetime.now().isoformat()}

    c_ids = np.load(args.pilot_dir / "C_ids.npy")
    rng = np.random.default_rng(args.seed)
    n = min(args.n_queries, len(c_ids))
    sample = np.sort(rng.choice(c_ids, size=n, replace=False))
    order_c = np.argsort(c_ids)
    pos = order_c[np.searchsorted(c_ids[order_c], sample)]
    assert np.array_equal(c_ids[pos], sample)

    qvecs = {}
    for qmod in ("image", "text"):
        t0 = time.time()
        qvecs[qmod] = np.ascontiguousarray(
            load_selected(args.model, qmod, sample, args.pilot_dir / "pair_index.csv",
                          root=embed_dir(args.model, qmod) if args.v2 else None),
            dtype=np.float32)
        print(f"loaded {n:,} {qmod} query vectors in {time.time() - t0:.1f}s", flush=True)

    configs = resolve_configs(args.model, "all")
    costs = bench_costs(args.model, args.cost_dir)
    n_b = len(np.load(lay_at(args.model, K_MAIN).role_ids("B")))
    out = {"task": "paper experiments part 3: c_est = Steps 1-3 of the online stage",
           "model": args.model, "scale": SCALE, "environment": env,
           "rule": {"tau": TAU, "rho": RHO, "q_min": Q_MIN, "selection": "coverage rule"},
           "sample": {"role": "C", "n": int(n), "seed": args.seed, "warmup": args.warmup,
                      "batch": BATCH, "threads_batched": args.threads},
           "lookup": {"single": "FAISS IndexFlatL2 (no BLAS for one query)",
                      "batched": "numpy matmul argmin(|l|^2 - 2 q.l), numpy's OpenBLAS"},
           "min_gflops_flag": MIN_GFLOPS,
           "pairings": {}}
    dest = args.out_dir / args.model
    dest.mkdir(parents=True, exist_ok=True)

    for p in sorted(PAIRINGS):
        spec = PAIRINGS[p]
        order = sorted(configs, key=lambda c: (costs[p][c], configs.index(c)))
        q = qvecs[spec["query_modality"]]
        e = out["pairings"].setdefault(p, {"cost_order": order, "batched_steps_1_3": {},
                                           "single": {}})
        for k in ks:
            lay = lay_at(args.model, k)
            cent = np.ascontiguousarray(np.load(lay.centroids(spec["index_modality"])),
                                        dtype=np.float32)
            index = faiss.IndexFlatL2(cent.shape[1])
            index.add(cent)
            lookup = lookup_fn(cent)
            table = served_table(args.model, p, order, n_b, k)

            # the pick must be the one the analysis evaluated
            faiss.omp_set_num_threads(args.threads)
            got = select_rows(table[np.concatenate(
                [lookup(q[s:s + BATCH]) for s in range(0, n, BATCH)])], RHO)
            stored = load_query_geometry(lay, p, "C")["assign"].astype(np.int64)[pos]
            want = select_rows(table[stored], RHO)
            agree = float((got == want).mean())

            r = batched(lookup, table, q, min(args.warmup, n), args.min_window)
            gflops = 2.0 * k * cent.shape[1] / r["s_per_query"] / 1e9
            e["batched_steps_1_3"][str(k)] = {
                **r, "gflops_lookup_equiv": gflops, "parallel_ok": gflops >= MIN_GFLOPS,
                "pick_agreement_with_stored_assignment": agree, "K": k}
            if k == K_MAIN:
                for th in sorted({1, args.threads}):
                    faiss.omp_set_num_threads(th)
                    e["single"][str(th)] = {
                        "steps_1_3": single(index, table, q, min(args.warmup, n), True),
                        "step_1_only": single(index, table, q, min(args.warmup, n), False)}
                    s = e["single"][str(th)]
                    print(f"  {spec['short']:<5} K={k} threads={th:<3} steps 1-3 mean "
                          f"{s['steps_1_3']['mean_us']:8.1f} p99 {s['steps_1_3']['p99_us']:8.1f} | "
                          f"step 1 mean {s['step_1_only']['mean_us']:8.1f} us", flush=True)
            flag = "" if gflops >= MIN_GFLOPS else "   !! below MIN_GFLOPS -- not parallel"
            print(f"  {spec['short']:<5} K={k:>6} batched steps 1-3 "
                  f"{e['batched_steps_1_3'][str(k)]['us_per_query']:9.3f} us/q  "
                  f"{gflops:8.1f} GFLOP/s  pick agreement {agree:.4%}{flag}", flush=True)
            write_json_atomic(dest / "estimate.json", out)
            del index

    env["finished"] = datetime.now().isoformat()
    env["loadavg_at_end"] = [round(x, 2) for x in os.getloadavg()]
    out["complete"] = True
    write_json_atomic(dest / "estimate.json", out)
    print(f"\nwrote {dest / 'estimate.json'}")


if __name__ == "__main__":
    main()
