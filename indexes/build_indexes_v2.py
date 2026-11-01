"""
v2 index build at 5.8M: ONE configuration of the v2 catalogue per call.

    python indexes/build_indexes_v2.py --model siglip2 --modality image --config ivfpq_low

THE CATALOGUE
    The 9 configs (hnsw / ivfpq / oivfpq x low / mod / high) come from the 50k
    tuning proposal (result_v2/index_tuning/50k/proposal.json, CPU builds; user,
    2026-10-13): HNSW M and efConstruction shared by all models, IVFPQ d/m per
    model, O-IVFPQ exactly the IVFPQ d/m. The IVF partition is scaled from the
    50k sample to the corpus: nlist 256 -> 32,768. Search parameters are stored
    as STARTING values only (efSearch as tuned; nprobe = 50k nprobe x 128, capped
    at nlist) -- the 5.8M check sets the final ones.

BUILD (user, 2026-10-13: CPU build, GPU only where it accelerates; no CAGRA)
    FAISS 1.15.1 AVX2 CPU code for everything (.venv-faisscuvs: its CPU builds
    gave the same CF as faiss-cpu 1.15.1 to 4 decimals at 50k), except the IVF
    coarse k-means (nlist 32,768 over 5.8M vectors), which runs on the GPU via
    IndexIVF.clustering_index -- as v1's build_indexes.py. OPQ, PQ training, the
    add and all of HNSW are CPU. O-IVFPQ through index_factory with OPQ niter 10,
    exactly as the 50k sweep built it.

WHAT IS INDEXED
    v2 role A only (5.8M), with add_with_ids on the canonical v2 int_ids; B and C
    are never added. HNSW is wrapped in IndexIDMap2 (as v1). The add runs in
    chunks of 500,000 so the log shows the add rate as it grows.

Output: result_v2/indexes/7M_cleaned/<model>/<modality>/<config>.faiss
        + <config>.build.json (params, phase times, environment, node load)
"""
from __future__ import annotations

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
sys.path.insert(0, str(REPO_ROOT / "indexes"))

from index_configs import MODEL_DIMS                                       # noqa: E402
from load_vectors import load_included_shards, load_pair_index, load_role_matrix  # noqa: E402
from v2_dirs import MANIFEST, SPLIT_DIR, SPLIT_SCOPE, V2, embed_dir         # noqa: E402

CONFIGS = [f"{f}_{t}" for f in ("hnsw", "ivfpq", "oivfpq") for t in ("low", "mod", "high")]
NLIST, NLIST_50K, OPQ_NITER, CHUNK = 32_768, 256, 10, 500_000


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def loadavg() -> list:
    return [float(x) for x in open("/proc/loadavg").read().split()[:3]]


def config_from_proposal(proposal: dict, model: str, name: str) -> dict:
    fam, tier = name.split("_")
    if fam == "hnsw":
        p = proposal["HNSW"]["tiers"][tier]
        return {"family": "HNSW", "tier": tier, "M": p["M"], "efConstruction": p["efConstruction"],
                "efSearch_start": p["efSearch"][model], "factory": f"IDMap2,HNSW{p['M']},Flat"}
    key = "IVFPQ" if fam == "ivfpq" else "O-IVFPQ"
    p = proposal[key]["models"][model]["tiers"][tier]
    m = p["m"]
    factory = f"IVF{NLIST},PQ{m}" if key == "IVFPQ" else f"OPQ{m},IVF{NLIST},PQ{m}"
    return {"family": key, "tier": tier, "dsub": p["dsub"], "m": m, "nlist": NLIST,
            "factory": factory, "nprobe_50k": p["nprobe"],
            "nprobe_start": min(p["nprobe"] * NLIST // NLIST_50K, NLIST)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=("siglip2", "openclip", "qwen3vl", "jinav4"))
    ap.add_argument("--modality", required=True, choices=("image", "text"))
    ap.add_argument("--config", required=True, choices=CONFIGS)
    ap.add_argument("--proposal", type=Path, default=V2 / "index_tuning" / "50k" / "proposal.json")
    ap.add_argument("--out-dir", type=Path, default=V2 / "indexes" / "7M_cleaned")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", 32)))
    ap.add_argument("--no-gpu", action="store_true", help="coarse k-means on the CPU (smoke tests)")
    ap.add_argument("--limit-a", type=int, default=None, help="SMOKE TEST: index only the first N of A")
    ap.add_argument("--nlist", type=int, default=NLIST, help="SMOKE TEST only")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if not os.environ.get("PBS_JOBID"):
        raise SystemExit("Refusing to run outside a PBS job -- submit indexes/jobs/build_indexes_v2.pbs")
    faiss.omp_set_num_threads(args.threads)
    t_all = time.time()

    proposal = json.load(open(args.proposal))
    cfg = config_from_proposal(proposal, args.model, args.config)
    if args.nlist != NLIST and cfg["family"] != "HNSW":
        cfg["factory"] = cfg["factory"].replace(f"IVF{NLIST}", f"IVF{args.nlist}")
        cfg["nlist"] = args.nlist
    dest = args.out_dir / args.model / args.modality
    path, meta_p = dest / f"{args.config}.faiss", dest / f"{args.config}.build.json"
    if path.exists() and not args.overwrite:
        raise SystemExit(f"{path} exists -- pass --overwrite to rebuild")
    dest.mkdir(parents=True, exist_ok=True)
    log(f"{args.model} {args.modality} {args.config}: {cfg}  load {loadavg()}")

    # ---- vectors of v2 role A ------------------------------------------------
    scope = json.load(open(SPLIT_DIR / "pair_index_meta.json"))["scope"]
    if scope != SPLIT_SCOPE:
        raise RuntimeError(f"{SPLIT_DIR}: scope {scope!r}, expected {SPLIT_SCOPE!r}")
    t0 = time.time()
    a_ids = np.load(SPLIT_DIR / "A_ids.npy").astype(np.int64)
    pair_to_int = load_pair_index(SPLIT_DIR / "pair_index.csv", role="A")
    if len(pair_to_int) != len(a_ids):
        raise RuntimeError(f"pair_index.csv has {len(pair_to_int):,} A rows, A_ids.npy {len(a_ids):,}")
    X = load_role_matrix(args.model, args.modality, a_ids, pair_to_int, load_included_shards(MANIFEST),
                         expect_dim=MODEL_DIMS[args.model], root=embed_dir(args.model, args.modality))
    if args.limit_a:
        X, a_ids = np.ascontiguousarray(X[:args.limit_a]), a_ids[:args.limit_a].copy()
        log(f"SMOKE TEST: A cut to {len(a_ids):,}")
    X = np.ascontiguousarray(X, dtype=np.float32)
    n, d = X.shape
    load_s = time.time() - t0
    log(f"loaded A {n:,} x {d} in {load_s:.0f} s")

    # ---- build -----------------------------------------------------------------
    keep, phases = [], {}
    if cfg["family"] == "HNSW":
        inner = faiss.IndexHNSWFlat(d, cfg["M"], faiss.METRIC_L2)
        inner.hnsw.efConstruction = cfg["efConstruction"]
        index = faiss.IndexIDMap2(inner)          # own_fields False: `inner` stays referenced
        keep.append(inner)
        phases["train_s"] = 0.0
    else:
        index = faiss.index_factory(d, cfg["factory"], faiss.METRIC_L2)
        ivf = faiss.extract_index_ivf(index)
        ivf.verbose = True                         # FAISS prints the k-means / PQ training steps
        if cfg["family"] == "O-IVFPQ":
            opq = faiss.downcast_VectorTransform(index.chain.at(0))
            opq.niter = OPQ_NITER
            opq.verbose = True
        if not args.no_gpu:
            res = faiss.StandardGpuResources()
            clus = faiss.index_cpu_to_gpu(res, 0, faiss.IndexFlatL2(ivf.d))
            ivf.clustering_index = clus
            keep += [res, clus]
            log(f"coarse k-means on GPU (d {ivf.d}, nlist {ivf.nlist})")
        t0 = time.time()
        index.train(X)
        phases["train_s"] = time.time() - t0
        log(f"train {phases['train_s']:.0f} s  load {loadavg()}")

    t0, add_marks = time.time(), []
    for s in range(0, n, CHUNK):
        e = min(s + CHUNK, n)
        index.add_with_ids(X[s:e], a_ids[s:e])
        add_marks.append([e, round(time.time() - t0, 1)])
        log(f"  added {e:,}/{n:,}  {add_marks[-1][1]:.0f} s  ({(e - s) / (time.time() - t0 - (add_marks[-2][1] if len(add_marks) > 1 else 0)):,.0f} vec/s this chunk)")
    phases["add_s"] = time.time() - t0
    if index.ntotal != n:
        raise RuntimeError(f"ntotal {index.ntotal:,} != |A| {n:,}")

    if cfg["family"] == "HNSW":
        inner.hnsw.efSearch = cfg["efSearch_start"]
    else:
        faiss.extract_index_ivf(index).nprobe = min(cfg["nprobe_start"], cfg["nlist"])
    t0 = time.time()
    faiss.write_index(index, str(path) + ".tmp")
    os.replace(str(path) + ".tmp", path)
    phases["write_s"] = time.time() - t0

    meta = {
        "task": "v2 index build at 5.8M", "model": args.model, "modality": args.modality,
        "config": args.config, **cfg, "ntotal": int(index.ntotal), "dim": d,
        "file": path.name, "size_bytes": path.stat().st_size,
        "ids": "add_with_ids with the canonical v2 int_ids of role A (B and C never added)",
        "split_dir": str(SPLIT_DIR), "split_scope": scope, "embedding_root": str(embed_dir(args.model, args.modality)),
        "proposal": str(args.proposal), "load_s": round(load_s, 1),
        "phases_s": {k: round(v, 1) for k, v in phases.items()}, "add_marks_n_s": add_marks,
        "coarse_kmeans": "cpu" if (args.no_gpu or cfg["family"] == "HNSW") else "gpu (clustering_index)",
        "smoke_limit_a": args.limit_a, "created": datetime.now().isoformat(),
        "host": os.uname().nodename, "pbs_jobid": os.environ.get("PBS_JOBID"),
        "faiss_version": faiss.__version__, "faiss_compile_options": faiss.get_compile_options(),
        "python": sys.executable, "threads": args.threads, "loadavg_end": loadavg(),
        "wall_s": round(time.time() - t_all, 1),
    }
    meta_p.write_text(json.dumps(meta, indent=2))
    log(f"wrote {path} ({path.stat().st_size / 2**30:.2f} GiB) and {meta_p.name}; phases {meta['phases_s']}")


if __name__ == "__main__":
    main()
