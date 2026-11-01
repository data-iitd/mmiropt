"""
c_est split by step, one query at a time and batched: how much of it is the
landmark search?

    python paperexp/split_single_query.py --model siglip2        (inside a PBS job)

SINGLE QUERY. For every query of the same 5,000-query Q_test sample as the
other benchmarks, at K = 2,500 and one thread (and at 64 threads), each step of
the online stage is timed in its own perf_counter window, in the same pass:

    step1_search   i = l(q): IndexFlatL2.search(q, 1) over the K landmarks
    step2_read     row = pi_hat[i, :]  -- the served coverage table (tau = 0.7)
    step3_select   the coverage rule at rho = 0.9 on that row
    total          step1 + step2 + step3 (one window around all three)

Two controls, timed in the same way:
    call_overhead  IndexFlatL2.search on a ONE-landmark index of the same
                   dimension: the fixed cost of a FAISS call (Python binding,
                   result allocation) with almost no scanning. step1_search minus
                   this is the scan itself.
    timer          an empty perf_counter window: the measurement floor.

Reported per direction: mean, p50, p99 (microseconds) of each part.

BATCHED. The same sample in batches of 4,096, with the batched lookup of
bench_estimate.py (numpy matmul, numpy's OpenBLAS on OMP_NUM_THREADS threads).
In every batch the three steps are timed in their own windows, summed over the
pass; after one untimed warm-up pass, passes repeat until >= 20 s and >= 3
passes, and the median pass is divided by the number of queries:

    step1_search   idx = argmin(|l|^2 - 2 q.l) for the batch
    step2_read     rows = pi_hat[idx, :]
    step3_select   the coverage rule on those rows
    total          the sum of the three

HNSW SWEEP (--hnsw-sweep). Can a graph index over the landmarks cut Step 1?
For each direction, the exact lookup above (FAISS flat, single query; numpy,
batched) is timed again on the same node as the reference, then a
full-precision IndexHNSWFlat over the same K landmarks (L2, efConstruction 200)
for every M x efSearch of --hnsw-m x --hnsw-ef, timed the same two ways. HNSW is
approximate, so each variant also records, on the same queries:
    cell_agreement   share of queries given the same landmark as the exact lookup
    pick_agreement   share given the same configuration by the coverage rule
The chosen variant is the fastest single-query Step 1 whose pick agreement is
>= 0.999 in every direction of the model.

Output: result/paper_experiments/7M_cleaned/<model>/split.json
        (--hnsw-sweep: split_lookup_variants.json)

V2 (--v2; user 2026-10-30, Maya's single-query point): query vectors from the v2
embedding trees (indexes/v2_dirs.embed_dir); run with SFM_RESULTS_ROOT=result_v2 and
SFM_K_MAIN=<K> so landmarks, labels and costs are v2's too.
"""
from __future__ import annotations

import argparse
import os
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
from approx_knn import write_json_atomic                             # noqa: E402
from map_config import PAIRINGS, resolve_configs                     # noqa: E402
from selection_tables import R, SCALE, bench_costs, lay_at           # noqa: E402
from bench_estimate import (BATCH, RHO, cpu_model, lookup_fn,         # noqa: E402
                            select_row, select_rows, served_table)

K = 2500          # default; --k changes it
MIN_WINDOW = 20.0
PICK_AGREE_MIN = 0.999


def run_batched(lookup, table, q, warmup, min_window: float = MIN_WINDOW) -> dict:
    """Per-query cost of each step when 4,096 queries are processed together."""
    pc = time.perf_counter
    n = q.shape[0]

    def one_pass() -> np.ndarray:
        t = np.zeros(3)
        for s in range(0, n, BATCH):
            x = q[s:s + BATCH]
            a = pc()
            idx = lookup(x)
            b = pc()
            rows = table[idx]
            c = pc()
            select_rows(rows, RHO)
            d = pc()
            t += (b - a, c - b, d - c)
        return t

    select_rows(table[lookup(q[:warmup])], RHO)
    one_pass()                                            # warm-up pass, untimed
    passes = [one_pass()]
    while sum(p.sum() for p in passes) < min_window or len(passes) < 3:
        passes.append(one_pass())
    P = np.array(passes)                                  # [n_passes, 3] seconds
    med = np.median(P, axis=0) / n * 1e6
    total = float(np.median(P.sum(axis=1)) / n * 1e6)
    return {"n_passes": len(passes), "step1_search_us": float(med[0]),
            "step2_read_us": float(med[1]), "step3_select_us": float(med[2]),
            "total_us": total, "step1_share_of_total": float(med[0]) / total}


def stats(us: np.ndarray) -> dict:
    return {"mean_us": float(us.mean()), "p50_us": float(np.percentile(us, 50)),
            "p99_us": float(np.percentile(us, 99))}


def run(index, index1, table, q, warmup) -> dict:
    n = q.shape[0]
    for i in range(warmup):
        x = q[i:i + 1]
        _, I = index.search(x, 1)
        select_row(table[I[0, 0]], RHO)
        index1.search(x, 1)
    t = np.empty((n, 6))
    pc = time.perf_counter
    for i in range(n):
        x = q[i:i + 1]
        a = pc()
        _, I = index.search(x, 1)
        b = pc()
        row = table[I[0, 0]]
        c = pc()
        select_row(row, RHO)
        d = pc()
        index1.search(x, 1)
        e = pc()
        f = pc()
        t[i] = (b - a, c - b, d - c, d - a, e - d, f - e)
    us = t * 1e6
    names = ("step1_search", "step2_read", "step3_select", "total", "call_overhead", "timer")
    out = {nm: stats(us[:, j]) for j, nm in enumerate(names)}
    out["scan_only_mean_us"] = out["step1_search"]["mean_us"] - out["call_overhead"]["mean_us"]
    out["step1_share_of_total"] = out["step1_search"]["mean_us"] / out["total"]["mean_us"]
    return out


def sweep_hnsw(index, index1, cent, table, q, warmup, args) -> dict:
    """Exact lookup vs full-precision HNSW over the landmarks: same queries, same node."""
    bt = int(args.threads.split(",")[-1])                  # threads for the batched runs
    faiss.omp_set_num_threads(bt)
    exact = index.search(q, 1)[1][:, 0]
    pick_exact = select_rows(table[exact], RHO)
    faiss.omp_set_num_threads(1)
    res = {"flat": {"single_1": run(index, index1, table, q, warmup)}}
    res["flat"]["batched"] = run_batched(lookup_fn(cent), table, q, warmup, args.min_window)
    f = res["flat"]
    print(f"      flat          step1 single {f['single_1']['step1_search']['mean_us']:8.1f} us | "
          f"batched {f['batched']['step1_search_us']:6.2f} us/q", flush=True)
    res["hnsw"] = {}
    for M in (int(x) for x in args.hnsw_m.split(",")):
        faiss.omp_set_num_threads(bt)
        h = faiss.IndexHNSWFlat(cent.shape[1], M, faiss.METRIC_L2)
        h.hnsw.efConstruction = args.ef_construction
        t0 = time.perf_counter()
        h.add(cent)
        build_s = time.perf_counter() - t0
        for ef in (int(x) for x in args.hnsw_ef.split(",")):
            h.hnsw.efSearch = ef
            faiss.omp_set_num_threads(bt)
            got = h.search(q, 1)[1][:, 0]
            faiss.omp_set_num_threads(1)
            r = run(h, index1, table, q, warmup)
            faiss.omp_set_num_threads(bt)
            rb = run_batched(lambda x: h.search(x, 1)[1][:, 0], table, q, warmup, args.min_window)
            v = {"M": M, "efSearch": ef, "efConstruction": args.ef_construction, "build_s": build_s,
                 "cell_agreement": float((got == exact).mean()),
                 "pick_agreement": float((select_rows(table[got], RHO) == pick_exact).mean()),
                 "single_1": r, "batched": rb}
            res["hnsw"][f"M{M}_ef{ef}"] = v
            print(f"      HNSW M={M:<2} ef={ef:<3} step1 single {r['step1_search']['mean_us']:8.1f} us "
                  f"(p99 {r['step1_search']['p99_us']:7.1f}) | batched {rb['step1_search_us']:6.2f} us/q | "
                  f"cell agree {v['cell_agreement']:.4f} pick agree {v['pick_agreement']:.4f}", flush=True)
    return res


def summarize_sweep(pairings: dict) -> dict:
    mean = lambda xs: float(np.mean(list(xs)))
    P = list(pairings.values())
    variants = {}
    for nm in P[0]["hnsw"]:
        V = [p["hnsw"][nm] for p in P]
        variants[nm] = {"min_pick_agreement": min(v["pick_agreement"] for v in V),
                        "min_cell_agreement": min(v["cell_agreement"] for v in V),
                        "step1_single_us": mean(v["single_1"]["step1_search"]["mean_us"] for v in V),
                        "step1_single_p99_us": mean(v["single_1"]["step1_search"]["p99_us"] for v in V),
                        "total_single_us": mean(v["single_1"]["total"]["mean_us"] for v in V),
                        "step1_batched_us": mean(v["batched"]["step1_search_us"] for v in V)}
    flat_s = mean(p["flat"]["single_1"]["step1_search"]["mean_us"] for p in P)
    flat_b = mean(p["flat"]["batched"]["step1_search_us"] for p in P)
    ok = [nm for nm, v in variants.items() if v["min_pick_agreement"] >= PICK_AGREE_MIN]
    best = min(ok, key=lambda nm: variants[nm]["step1_single_us"]) if ok else None
    return {"rule": f"fastest single-query Step 1 among variants whose pick agreement with the "
                    f"exact lookup is >= {PICK_AGREE_MIN} in every direction",
            "flat_step1_single_us": flat_s, "flat_step1_batched_us": flat_b,
            "flat_total_single_us": mean(p["flat"]["single_1"]["total"]["mean_us"] for p in P),
            "variants": variants, "best": best,
            "best_speedup_single": None if best is None else flat_s / variants[best]["step1_single_us"],
            "best_speedup_batched": None if best is None else flat_b / variants[best]["step1_batched_us"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(MODEL_DIMS))
    ap.add_argument("--threads", default="1,64")
    ap.add_argument("--n-queries", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--pilot-dir", type=Path, default=R / SCALE / "siglip2")
    ap.add_argument("--out-dir", type=Path, default=R / "paper_experiments" / SCALE)
    ap.add_argument("--cost-dir", type=Path, default=None,
                    help="controlled-throughput tree (<dir>/<model>/throughput.json); "
                         "default result/throughput/7M_cleaned")
    ap.add_argument("--k", type=int, default=K, help="landmarks per modality (a stored K)")
    ap.add_argument("--hnsw-sweep", action="store_true",
                    help="time full-precision HNSW over the landmarks against the exact lookup")
    ap.add_argument("--hnsw-m", default="8,16,32")
    ap.add_argument("--hnsw-ef", default="8,16,32,64,128")
    ap.add_argument("--ef-construction", type=int, default=200)
    ap.add_argument("--min-window", type=float, default=MIN_WINDOW,
                    help="batched timing: repeat passes until this many seconds")
    ap.add_argument("--v2", action="store_true", help="query vectors from the v2 embedding trees")
    ap.add_argument("--allow-login-node", action="store_true")
    args = ap.parse_args()
    k = args.k
    if k != K and not args.hnsw_sweep:
        raise SystemExit("--k other than 2500 is only for --hnsw-sweep")
    out_name = (("split_lookup_variants.json" if k == K else f"split_lookup_variants_K{k}.json")
                if args.hnsw_sweep else "split.json")
    if not os.environ.get("PBS_JOBID") and not args.allow_login_node:
        raise SystemExit("Refusing to run outside a PBS job -- submit "
                         "paperexp/jobs/paperexp_split_7m.pbs")

    env = {"host": os.uname().nodename, "pbs_jobid": os.environ.get("PBS_JOBID"),
           "cpu_model": cpu_model(), "faiss_version": faiss.__version__,
           "faiss_compile_options": faiss.get_compile_options(), "python": sys.executable,
           "loadavg_at_start": [round(x, 2) for x in os.getloadavg()],
           "started": datetime.now().isoformat()}
    c_ids = np.load(args.pilot_dir / "C_ids.npy")
    rng = np.random.default_rng(args.seed)
    sample = np.sort(rng.choice(c_ids, size=min(args.n_queries, len(c_ids)), replace=False))
    qvecs = {qm: np.ascontiguousarray(
        load_selected(args.model, qm, sample, args.pilot_dir / "pair_index.csv",
                      root=embed_dir(args.model, qm) if args.v2 else None), dtype=np.float32)
        for qm in ("image", "text")}

    configs = resolve_configs(args.model, "all")
    costs = bench_costs(args.model, args.cost_dir)
    n_b = len(np.load(lay_at(args.model, k).role_ids("B")))
    out = {"task": "c_est split by step, single query and batched", "model": args.model, "K": k,
           "environment": env, "rule": {"tau": 0.70, "rho": RHO},
           "sample": {"role": "C", "n": int(len(sample)), "seed": args.seed, "warmup": args.warmup,
                      "batch": BATCH},
           "lookup": {"single": "FAISS IndexFlatL2 (no BLAS for one query)",
                      "batched": "numpy matmul argmin(|l|^2 - 2 q.l), numpy's OpenBLAS"},
           "pairings": {}}
    dest = args.out_dir / args.model
    dest.mkdir(parents=True, exist_ok=True)

    for p in sorted(PAIRINGS):
        spec = PAIRINGS[p]
        order = sorted(configs, key=lambda c: (costs[p][c], configs.index(c)))
        cent = np.ascontiguousarray(np.load(lay_at(args.model, k).centroids(spec["index_modality"])),
                                    dtype=np.float32)
        index = faiss.IndexFlatL2(cent.shape[1]); index.add(cent)
        index1 = faiss.IndexFlatL2(cent.shape[1]); index1.add(cent[:1])
        table = served_table(args.model, p, order, n_b, k)
        q = qvecs[spec["query_modality"]]
        if args.hnsw_sweep:
            print(f"  {spec['short']}", flush=True)
            out["pairings"][p] = sweep_hnsw(index, index1, cent, table, q,
                                            min(args.warmup, len(q)), args)
            write_json_atomic(dest / out_name, out)
            continue
        out["pairings"][p] = {}
        for th in (int(x) for x in args.threads.split(",")):
            faiss.omp_set_num_threads(th)
            r = run(index, index1, table, q, min(args.warmup, len(q)))
            out["pairings"][p][str(th)] = r
            print(f"  {spec['short']:<5} threads={th:<3} step1 {r['step1_search']['mean_us']:8.1f} "
                  f"(scan {r['scan_only_mean_us']:8.1f} + call {r['call_overhead']['mean_us']:5.1f}) | "
                  f"step2 {r['step2_read']['mean_us']:6.2f} | step3 {r['step3_select']['mean_us']:6.2f} | "
                  f"total {r['total']['mean_us']:8.1f} us | step1 share {r['step1_share_of_total']:.1%} | "
                  f"timer {r['timer']['mean_us']:.3f}", flush=True)
        rb = run_batched(lookup_fn(cent), table, q, min(args.warmup, len(q)))
        rb["gflops_step1"] = 2.0 * K * cent.shape[1] / (rb["step1_search_us"] * 1e-6) / 1e9
        out["pairings"][p]["batched"] = rb
        print(f"  {spec['short']:<5} batched     step1 {rb['step1_search_us']:8.2f} "
              f"({rb['gflops_step1']:.0f} GFLOP/s) | step2 {rb['step2_read_us']:6.3f} | "
              f"step3 {rb['step3_select_us']:6.3f} | total {rb['total_us']:8.2f} us/q | "
              f"step1 share {rb['step1_share_of_total']:.1%}", flush=True)
        write_json_atomic(dest / out_name, out)

    if args.hnsw_sweep:
        out["task"] = "landmark lookup: exact flat vs full-precision HNSW over the landmarks"
        out["summary"] = sm = summarize_sweep(out["pairings"])
        b = sm["best"]
        print(f"  flat step1 single {sm['flat_step1_single_us']:.1f} us, batched "
              f"{sm['flat_step1_batched_us']:.2f} us/q | best {b}: " +
              ("none meets the agreement rule" if b is None else
               f"single {sm['variants'][b]['step1_single_us']:.1f} us "
               f"({sm['best_speedup_single']:.1f}x), batched {sm['variants'][b]['step1_batched_us']:.2f} "
               f"us/q ({sm['best_speedup_batched']:.2f}x), pick agreement "
               f"{sm['variants'][b]['min_pick_agreement']:.4f}"), flush=True)
    env["finished"] = datetime.now().isoformat()
    out["complete"] = True
    write_json_atomic(dest / out_name, out)
    print(f"wrote {dest / out_name}")


if __name__ == "__main__":
    main()
