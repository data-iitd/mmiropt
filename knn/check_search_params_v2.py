"""
v2, 5.8M: set each index's search parameter (efSearch / nprobe) on real scale.

    python knn/check_search_params_v2.py --stage sweep --model siglip2 --modality image --config ivfpq_low
    python knn/check_search_params_v2.py --stage propose

WHY (user, 2026-10-14)
    The 72 indexes in result_v2/indexes/7M_cleaned were built from the 50k tuning
    with STARTING search parameters only (CF falls with corpus size, and nlist
    256 -> 32,768 changes what an nprobe means). Before the controlled throughput
    benchmark and T7, each tier's search parameter is chosen again at 5.8M.

SWEEP (one index per job)
    5,000 seeded B queries (seed 0, the same for every index; B, never C -- C is
    held out), for BOTH directions the index serves (image index: i->i, t->i;
    text index: t->t, i->t). Each is graded against the v2 exact top-100
    (result_v2/knn/7M_cleaned, role B) by CF@100, the measure of the 50k tuning.
    Grid: HNSW efSearch 16..4,096 (extended 2026-10-14: at 5.8M OpenCLIP image M4
    still rose at ef 1,024, CF 0.60); IVF nprobe 8..8,192 (nlist 32,768; user, 2026-10-14: no more than a quarter
    of the lists, beyond that IVF approaches an exhaustive scan). A grid
    stops early once its mean CF reaches 0.995, gains less than 0.002 per step,
    or one value takes longer than --max-step-s. Search runs on the CPU (FAISS
    1.15.1 AVX2), batched; the time per value is recorded but is NOT a cost
    number (shared node) -- that is the controlled benchmark's job.

PROPOSE (the 50k tuning's rules, user decisions of 2026-10-13, plus two rules of 2026-10-14:
    the search value must not fall from low to mod to high within a family -- the
    tiers are cost levels -- and per model at least 2 of the 3 high configs reach
    CF >= 0.90 on both indexes)
    The builds are fixed; only the search value is chosen, per tier and per
    model, ONE value for the model's image AND text index. Soft bands: low
    [0.65, 0.75), mod [0.75, 0.85), high >= 0.85; the value with the most indexes
    inside the band, then the smallest distance to the target (0.70 / 0.80 /
    0.90, in 0.01 steps), then the smallest value.

Output: result_v2/param_check/7M_cleaned/<model>/<modality>/<config>.csv (+ .done.json)
        result_v2/param_check/7M_cleaned/final_params.json   (propose)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import faiss

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "indexes"))

from load_vectors import load_selected                                    # noqa: E402
from v2_dirs import SPLIT_DIR, SPLIT_SCOPE, V2, embed_dir                  # noqa: E402
import tune_catalogue_v2 as TC                                             # noqa: E402

MODELS = ("siglip2", "openclip", "qwen3vl", "jinav4")
CONFIGS = [f"{f}_{t}" for f in ("hnsw", "ivfpq", "oivfpq") for t in ("low", "mod", "high")]
DIRS = {"image": (("i->i", "image", "image_query_image_index"), ("t->i", "text", "text_query_image_index")),
        "text": (("t->t", "text", "text_query_text_index"), ("i->t", "image", "image_query_text_index"))}
GRID = {"HNSW": (16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096),
        "IVF": (8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)}   # user: max 8,192 (25% of nlist)
K = 100
INDEX_ROOT = V2 / "indexes" / "7M_cleaned"
KNN_ROOT = V2 / "knn" / "7M_cleaned"
OUT = V2 / "param_check" / "7M_cleaned"


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def loadavg() -> list:
    return [float(x) for x in open("/proc/loadavg").read().split()[:3]]


def sample_ids(n: int, seed: int) -> np.ndarray:
    return np.sort(np.random.default_rng(seed).choice(np.load(SPLIT_DIR / "B_ids.npy"), n, replace=False))


def set_param(index, family: str, v: int) -> None:
    if family == "HNSW":
        faiss.downcast_index(faiss.downcast_index(index).index).hnsw.efSearch = v
    else:
        faiss.extract_index_ivf(index).nprobe = v


def cf(I: np.ndarray, ex: np.ndarray) -> float:
    return float(np.mean([len(np.intersect1d(a[a >= 0], e, assume_unique=True)) for a, e in zip(I, ex)])) / K


def stage_sweep(args) -> None:
    dest = OUT / args.model / args.modality
    path, done_p = dest / f"{args.config}.csv", dest / f"{args.config}.done.json"
    if done_p.exists():
        log(f"{done_p} exists -- nothing to do"); return
    dest.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    meta = json.load(open(INDEX_ROOT / args.model / args.modality / f"{args.config}.build.json"))
    family = meta["family"]
    grid = GRID["HNSW" if family == "HNSW" else "IVF"]
    log(f"{args.model} {args.modality} {args.config} ({family}) load {loadavg()}")

    scope = json.load(open(SPLIT_DIR / "pair_index_meta.json"))["scope"]
    if scope != SPLIT_SCOPE:
        raise RuntimeError(f"{SPLIT_DIR}: scope {scope!r}, expected {SPLIT_SCOPE!r}")
    q_ids = sample_ids(args.n_queries, args.seed)
    Q, EX = {}, {}
    for label, qmod, pairing in DIRS[args.modality]:
        Q[label] = np.ascontiguousarray(load_selected(args.model, qmod, q_ids, SPLIT_DIR / "pair_index.csv",
                                                      root=embed_dir(args.model, qmod)), dtype=np.float32)
        kd = KNN_ROOT / pairing / args.model
        qids = np.load(kd / "knn_B_qids.npy")
        rows = np.searchsorted(qids, q_ids)
        if not np.array_equal(qids[rows], q_ids):
            raise RuntimeError(f"{kd}: sample ids missing from the exact k-NN")
        EX[label] = np.asarray(np.load(kd / "knn_B_ids.npy", mmap_mode="r")[rows, :K])
    log(f"queries + exact top-{K} loaded ({args.n_queries} B per direction)")

    t0 = time.time()
    index = faiss.read_index(str(INDEX_ROOT / args.model / args.modality / f"{args.config}.faiss"))
    load_index_s = time.time() - t0
    if index.ntotal != meta["ntotal"]:
        raise RuntimeError(f"ntotal {index.ntotal} != build.json {meta['ntotal']}")
    log(f"index loaded in {load_index_s:.0f} s ({index.ntotal:,} vectors)")

    cols = ["model", "modality", "config", "family", "value", "cf_same", "cf_cross", "cf_mean",
            "search_s", "us_per_query", "loadavg1"]
    rows_out, prev = [], None
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for v in grid:
            set_param(index, family, v)
            t0 = time.time()
            c = {}
            for label, _, _ in DIRS[args.modality]:
                _, I = index.search(Q[label], K)
                c[label] = cf(I, EX[label])
            s = time.time() - t0
            (la, ca), (lb, cb) = c.items()
            r = {"model": args.model, "modality": args.modality, "config": args.config, "family": family,
                 "value": v, "cf_same": ca, "cf_cross": cb, "cf_mean": (ca + cb) / 2, "search_s": round(s, 2),
                 "us_per_query": round(s / (2 * args.n_queries) * 1e6, 1), "loadavg1": loadavg()[0]}
            w.writerow(r); f.flush(); os.fsync(f.fileno())
            rows_out.append(r)
            log(f"  {'efSearch' if family == 'HNSW' else 'nprobe'} {v:>6}: {la} {ca:.4f}  {lb} {cb:.4f}  "
                f"mean {r['cf_mean']:.4f}  {s:7.1f} s")
            stop = (r["cf_mean"] >= 0.995 or (prev is not None and r["cf_mean"] - prev < 0.002)
                    or s > args.max_step_s)
            prev = r["cf_mean"]
            if stop:
                log(f"  grid stopped after {v} (mean CF {r['cf_mean']:.4f}, step {s:.0f} s)")
                break
    done = {"model": args.model, "modality": args.modality, "config": args.config, "family": family,
            "values": [r["value"] for r in rows_out], "n_queries": args.n_queries, "seed": args.seed,
            "role": "B", "split_scope": scope, "index_file": str(INDEX_ROOT / args.model / args.modality / f"{args.config}.faiss"),
            "load_index_s": round(load_index_s, 1), "created": datetime.now().isoformat(),
            "host": os.uname().nodename, "pbs_jobid": os.environ.get("PBS_JOBID"),
            "faiss_version": faiss.__version__, "faiss_compile_options": faiss.get_compile_options(),
            "python": sys.executable, "threads": args.threads, "wall_s": round(time.time() - t_all, 1)}
    done_p.write_text(json.dumps(done, indent=2))
    log(f"wrote {path.name} ({len(rows_out)} values) in {time.time() - t_all:.0f} s")


HIGH_MIN, HIGH_FAMILIES_MIN = 0.90, 2


def _rows(m: str, config: str) -> dict:
    """{index: {value: row}} for the image and text index of one model, or {} if a check is missing."""
    out = {}
    for x in ("image", "text"):
        if not (OUT / m / x / f"{config}.done.json").exists():
            return {}
        out[f"{m}/{x}"] = {int(r["value"]): {**r, "value": int(r["value"]), "cf_mean": float(r["cf_mean"]),
                                             "cf_same": float(r["cf_same"]), "cf_cross": float(r["cf_cross"])}
                           for r in csv.DictReader(open(OUT / m / x / f"{config}.csv"))}
    return out


def _family_triples(m: str, fam: str):
    """(low, mod, high) search values for one model and family, chosen TOP DOWN so the cost
    order holds without moving mod or high: high by the band rule (or the smallest value with
    CF >= HIGH_MIN on both indexes, when forced), then mod by the band rule among values <= high's,
    then low among values <= mod's. Returns (unforced, forced, rows); forced is None if no grid
    value reaches HIGH_MIN."""
    R = {t: _rows(m, f"{fam}_{t}") for t in ("low", "mod", "high")}
    if not all(R.values()):
        return None, None, R
    vals = {t: sorted(set.intersection(*(set(v) for v in R[t].values()))) for t in R}

    def sc(t, v):
        inside, dist = TC.score([R[t][ix][v]["cf_mean"] for ix in R[t]], t)
        return inside, dist

    def pick(t, allowed):
        """the band rule for one tier over the allowed values: most indexes inside, then the
        distance to the target, then the smallest value"""
        return max(allowed, key=lambda v: (sc(t, v)[0], -sc(t, v)[1], -v)) if allowed else None

    best = {}
    for forced in (False, True):                 # top down: high, then mod <= high, then low <= mod
        hv = [v for v in vals["high"] if not forced
              or min(R["high"][ix][v]["cf_mean"] for ix in R["high"]) >= HIGH_MIN]
        vh = pick("high", hv)
        vm = pick("mod", [v for v in vals["mod"] if vh is not None and v <= vh])
        vl = pick("low", [v for v in vals["low"] if vm is not None and v <= vm])
        if None in (vh, vm, vl):
            best[forced] = None
            continue
        ss = [sc("low", vl), sc("mod", vm), sc("high", vh)]
        ok = min(R["high"][ix][vh]["cf_mean"] for ix in R["high"]) >= HIGH_MIN
        best[forced] = ((sum(x[0] for x in ss), -sum(x[1] for x in ss), -(vl + vm + vh)), (vl, vm, vh), ok)
    return best[False], best[True], R


def stage_propose(args) -> None:
    """Search values per model: order rule (low <= mod <= high per family) and, per model, at
    least HIGH_FAMILIES_MIN of the 3 high configs with CF >= HIGH_MIN on both indexes (user,
    2026-10-14); within that the band rules of the 50k tuning."""
    fams = ("hnsw", "ivfpq", "oivfpq")
    final, notes = {c: {} for c in CONFIGS}, {}
    for m in MODELS:
        res = {f: _family_triples(m, f) for f in fams}
        if any(res[f][0] is None for f in fams):
            notes[m] = "missing checks"; continue
        best = None
        for mask in range(8):                                   # which families must reach HIGH_MIN
            force = {f: bool(mask >> i & 1) for i, f in enumerate(fams)}
            picks = {}
            for f in fams:
                pick = res[f][1] if force[f] else res[f][0]
                if pick is None:
                    break
                picks[f] = pick
            else:
                n_high = sum(p[2] for p in picks.values())
                if n_high < HIGH_FAMILIES_MIN:
                    continue
                key = tuple(map(sum, zip(*(p[0] for p in picks.values()))))
                if best is None or key > best[0]:
                    best = (key, picks, force)
        if best is None:                                        # rule impossible: say so, keep unforced
            best = (None, {f: res[f][0] for f in fams}, {f: False for f in fams})
            notes[m] = f"fewer than {HIGH_FAMILIES_MIN} high configs can reach {HIGH_MIN} within the grid"
        for f, (key, values, high_ok) in best[1].items():
            R = res[f][2]
            for t, v in zip(("low", "mod", "high"), values):
                rows = {ix: R[t][ix][v] for ix in R[t]}
                inside = sum(TC.in_band(r["cf_mean"], t) for r in rows.values())
                final[f"{f}_{t}"][m] = {"value": v, "param": "efSearch" if f == "hnsw" else "nprobe",
                                        "in_band": f"{inside}/2", "band": TC.BANDS[t],
                                        "high_forced_to_0.90": bool(best[2][f]) if t == "high" else None,
                                        "per_index": {ix: {"same": r["cf_same"], "cross": r["cf_cross"],
                                                           "mean": r["cf_mean"]} for ix, r in rows.items()}}
    out = {"task": "v2 5.8M search-parameter choice", "created": datetime.now().isoformat(),
           "rule": ("per model and family: search value low <= mod <= high (cost order); per model at least "
                    f"{HIGH_FAMILIES_MIN} of the 3 high configs with CF >= {HIGH_MIN} on both indexes (smallest "
                    "such value); within that the most indexes inside the soft band, then the distance to the "
                    "target in 0.01 steps, then the smallest values"),
           "bands": TC.BANDS, "targets": TC.TARGETS, "high_min": HIGH_MIN, "high_families_min": HIGH_FAMILIES_MIN,
           "n_queries": args.n_queries, "seed": args.seed, "role": "B", "notes": notes, "params": final}
    p = OUT / "final_params.json"
    if p.exists() and not (OUT / "final_params.before_order.json").exists():
        os.replace(p, OUT / "final_params.before_order.json")
    p.write_text(json.dumps(out, indent=2))
    log(f"wrote {p}; notes {notes}")
    for config, per in final.items():
        log(f"  {config:<12} " + " | ".join(
            f"{m}: {p['param'][0:2]}{p['value']} {p['per_index'][m + '/image']['mean']:.3f}/"
            f"{p['per_index'][m + '/text']['mean']:.3f} in {p['in_band']}" for m, p in per.items()))


def stage_propose_shared(args) -> None:
    """Variant (user, 2026-10-14): ONE search value per tier for ALL models (the 8 indexes), with
    the same rules -- band rule over the 8 indexes, top-down order (high, mod <= high, low <= mod),
    and per model at least HIGH_FAMILIES_MIN of the 3 high configs with CF >= HIGH_MIN on both
    indexes. Only values measured for all 8 indexes are candidates. Writes final_params_shared.json
    next to the per-model final_params.json, which it does not touch."""
    from itertools import product
    fams = ("hnsw", "ivfpq", "oivfpq")
    R = {f: {t: {} for t in ("low", "mod", "high")} for f in fams}
    for f in fams:
        for t in R[f]:
            for m in MODELS:
                R[f][t].update(_rows(m, f"{f}_{t}"))
    vals = {f: {t: sorted(set.intersection(*(set(v) for v in R[f][t].values()))) for t in R[f]} for f in fams}

    def sc(f, t, v):
        return TC.score([R[f][t][ix][v]["cf_mean"] for ix in R[f][t]], t)

    def pick(f, t, allowed):
        return max(allowed, key=lambda v: (sc(f, t, v)[0], -sc(f, t, v)[1], -v)) if allowed else None

    def high_ok(f, v, m):
        return min(R[f]["high"][f"{m}/{x}"][v]["cf_mean"] for x in ("image", "text")) >= HIGH_MIN

    best = None
    for hv in product(*(vals[f]["high"] for f in fams)):
        hv = dict(zip(fams, hv))
        if any(sum(high_ok(f, hv[f], m) for f in fams) < HIGH_FAMILIES_MIN for m in MODELS):
            continue
        key, chosen = (0, 0, 0), {}
        for f in fams:
            vm = pick(f, "mod", [v for v in vals[f]["mod"] if v <= hv[f]])
            vl = pick(f, "low", [v for v in vals[f]["low"] if vm is not None and v <= vm])
            if None in (vm, vl):
                break
            ss = [sc(f, "low", vl), sc(f, "mod", vm), sc(f, "high", hv[f])]
            key = (key[0] + sum(x[0] for x in ss), key[1] - sum(x[1] for x in ss), key[2] - (vl + vm + hv[f]))
            chosen[f] = (vl, vm, hv[f])
        else:
            if best is None or key > best[0]:
                best = (key, chosen)
    final = {}
    if best is None:
        log("no shared setting meets the 0.90 rule for every model within the grids")
    else:
        for f, values in best[1].items():
            for t, v in zip(("low", "mod", "high"), values):
                rows = {ix: R[f][t][ix][v] for ix in R[f][t]}
                final[f"{f}_{t}"] = {"value": v, "param": "efSearch" if f == "hnsw" else "nprobe",
                                     "in_band": f"{sum(TC.in_band(r['cf_mean'], t) for r in rows.values())}/8",
                                     "per_index": {ix: {"same": r["cf_same"], "cross": r["cf_cross"],
                                                        "mean": r["cf_mean"]} for ix, r in rows.items()}}
    out = {"task": "v2 5.8M search-parameter choice -- SHARED across models (variant)",
           "created": datetime.now().isoformat(), "rule": stage_propose_shared.__doc__.split("Writes")[0].strip(),
           "candidate_values": vals, "params": final}
    (OUT / "final_params_shared.json").write_text(json.dumps(out, indent=2))
    log(f"wrote {OUT}/final_params_shared.json")
    for c, p in final.items():
        log(f"  {c:<12} {p['param'][:2]}{p['value']:<5} in {p['in_band']}  " + " ".join(
            f"{ix}:{v['mean']:.3f}" for ix, v in p["per_index"].items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("sweep", "propose", "propose-shared"))
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--modality", choices=("image", "text"))
    ap.add_argument("--config", choices=CONFIGS)
    ap.add_argument("--n-queries", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-step-s", type=float, default=1800.0)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", 32)))
    args = ap.parse_args()
    if not os.environ.get("PBS_JOBID"):
        raise SystemExit("Refusing to run outside a PBS job")
    faiss.omp_set_num_threads(args.threads)
    if args.stage == "sweep":
        if not (args.model and args.modality and args.config):
            ap.error("--stage sweep needs --model, --modality and --config")
        stage_sweep(args)
    elif args.stage == "propose":
        stage_propose(args)
    else:
        stage_propose_shared(args)


if __name__ == "__main__":
    main()
