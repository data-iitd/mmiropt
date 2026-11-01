"""
v2 index catalogue, step 1: a 50k-scale sweep that chooses the tiers.

Three stages, each inside a PBS job (indexes/jobs/tune_catalogue_v2_*.pbs):

    python indexes/tune_catalogue_v2.py --stage sample  --model siglip2
    python indexes/tune_catalogue_v2.py --stage sweep   --model siglip2 \\
                                        --index-modality image --family O-IVFPQ
    python indexes/tune_catalogue_v2.py --stage propose          (all 4 models)

THE GOAL (user, 2026-10-13)
    Per family -- HNSW, IVFPQ, O-IVFPQ -- a low tier near 0.70 CF@100, a mod tier
    near 0.80 and a high tier at 0.90 or more, where CF is the MEAN over the two
    directions an index serves:
        image index   i->i and t->i
        text index    t->t and i->t
    The tiers must differ in the BUILD, not only in a search parameter: HNSW by
    graph degree M (with efConstruction), the PQ families by code length d/m.
    IVFPQ and O-IVFPQ may use different d/m.
    SHARED BUILD, PER-MODEL SEARCH (user, 2026-10-13): a tier has ONE config
    for the image and the text index of a model -- never one per modality. The
    BUILD is shared across models so they stay comparable: HNSW M and
    efConstruction; PQ d/m where the model's d allows it (d = 2048 has no d/m
    3). The SEARCH parameter (efSearch, nprobe) is per model, so every model
    can reach the same band: at one config the CLIP-type models (SigLIP 2,
    OpenCLIP) score far lower in the cross-modal directions (50k, IVFPQ d/m 1
    nprobe 32: SigLIP 2 t->t 0.83, i->t 0.11; Qwen3-VL 0.90, 0.90).
    SOFT BANDS (user, 2026-10-13): low 0.65-0.75, mod 0.75-0.85, high >= 0.85;
    the targets 0.70 / 0.80 / 0.90 only break ties. Other config parameters
    may vary to fill a band (user); here the search parameter, and nbits per
    PQ code if a band stays empty.

WHY 50k (user: tune at a very small scale, then build once at full scale)
    A seeded random 50,000 of v2 A and 500 of v2 B, in each modality, from the
    result_v2 trees (indexes/v2_dirs.py). The exact top-100 of every query is
    computed here by brute force, so no k-NN job is needed. HNSW and IVFPQ
    settings build in seconds to a few minutes; an O-IVFPQ setting takes 10-15
    min (OPQ training: OpenCLIP d/m 2 817 s, d/m 3 706 s on 16 cores, job
    107387). The IVF partition is scaled with the sample: nlist 256 here (~195
    vectors per list) stands for 32,768 at 5.8M (~177).

THE STAGES (job 107387 ran all of a model in one job, wrote nothing before the
end, and was stopped before its 1-hour limit killed it)
    sample    reads the sample of one model once (both modalities; ~10-30 min
              of targeted reads) and stores it, with the exact top-100 of all
              four directions, in <model>/sample/. sample_meta.json is written
              last and marks the sample complete.
    sweep     one index modality x one family per job (24 jobs for 4 models).
              It reads the stored sample in seconds. The rows of a build are
              appended to sweep_<modality>_<family>.csv as soon as the build is
              graded, and a restart skips the builds that are already in the
              file, so a job that hits its time limit loses at most one build.
              sweep_<modality>_<family>.done.json marks the file complete.
    propose   reads all 24 complete sweep files (the HNSW tiers are shared by
              every model) and writes proposal.json.

THE CAVEAT -- CF FALLS AS THE CORPUS GROWS
    At 5.8M the neighbours are denser, so a setting gives a lower CF than here.
    The chosen tiers are checked once at full scale after the real build (5,000
    B queries); if a tier misses, only its search parameter is moved.

WHAT IS SWEPT
    HNSW      M in {4, 8, 16, 32} x efConstruction in {40, 80, 160}, built once
              each; efSearch in {16 .. 512} at search time.
    IVFPQ     IVF256,PQ<m> for every dims-per-slice d/m in {1,2,3,4,6,8,12,16,
              24,32} that divides d (d = 2048 allows only 1,2,4,8,16,32);
              nprobe in {1 .. 256}.
    O-IVFPQ   OPQ<m>,IVF256,PQ<m> (rotation matched to the code, as v1), the same
              d/m list without 1 (an M = d rotation is the slowest to train and
              v1 never used it); OPQ niter 10, as v1.
    Factory strings and defaults are the ones build_indexes.py uses, so a chosen
    setting is the index that will be built.
    Every d/m is swept on both indexes (no early stop): a shared tier needs the
    CF of the other index at the same d/m. The PQ families go from the shortest
    code (d/m 32) to the longest, so a job that hits its limit has the cheap
    builds saved first.

THE PROPOSAL (written to proposal.json; the user decides)
    The tiers take strictly larger builds, low < mod < high: a longer PQ code,
    a larger HNSW M. Over all such triples, the one with the most sharing
    indexes inside their bands wins; then the cheapest builds; then the
    smallest summed distance to the targets.
    PQ family: per model one d/m per tier for its image AND text index, and a
    reference d/m triple shared by all models: a model leaves the reference
    only where its d forbids that d/m (d = 2048 has no d/m 3; user: Jina v4
    keeps the same 4/2/1 as Qwen3-VL). nprobe per tier per model. A long code at a low nprobe
    can fill a band that no plateau reaches (at d = 2048 the mod band lies
    between d/m 8 and d/m 4).
    HNSW: one (M, efConstruction) per tier for all 8 indexes (4 models x 2
    modalities); per M, the efConstruction with the most indexes inside the
    band, then the smallest. efSearch per tier per model.
    Distances to the targets count in steps of 0.01 (about the sampling error of
    a CF from 500 queries), so a larger search parameter must gain a real 0.01.

Output: result_v2/index_tuning/50k/
        <model>/sample/                                the sample, its ids, the exact top-100
        <model>/sweep_<image|text>_<family>.csv        one row per (build, search parameter)
        <model>/sweep_<image|text>_<family>.done.json  complete marker, run info
        proposal.json                                  the shared tiers, all models
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import time
from collections import Counter
from itertools import combinations
from datetime import datetime
from pathlib import Path

import numpy as np

import faiss

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "indexes"))

from index_configs import MODEL_DIMS                                 # noqa: E402
from load_vectors import load_selected                              # noqa: E402
from v2_dirs import SPLIT_DIR, SPLIT_SCOPE, V2, embed_dir           # noqa: E402

K = 100
N_A, N_B = 50_000, 500
NLIST = 256
HNSW_M = (4, 8, 16, 32)
HNSW_EFC = (40, 80, 160)
HNSW_EFS = (16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512)
DSUB = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
NPROBE = (1, 2, 4, 8, 16, 32, 64, 128, 256)
OPQ_NITER = 10
TIERS = ("low", "mod", "high")
TARGETS = {"low": 0.70, "mod": 0.80, "high": 0.90}
BANDS = {"low": (0.65, 0.75), "mod": (0.75, 0.85), "high": (0.85, 1.0)}   # soft (user)
DIRECTIONS = {"image": (("i->i", "image"), ("t->i", "text")),     # index modality ->
              "text": (("t->t", "text"), ("i->t", "image"))}      #   (label, query modality)
MODELS = ("siglip2", "openclip", "qwen3vl", "jinav4")
FAMILIES = ("HNSW", "IVFPQ", "O-IVFPQ")
LABEL_FILE = {"i->i": "i2i", "t->i": "t2i", "t->t": "t2t", "i->t": "i2t"}
COLS_HNSW = ["model", "index_modality", "family", "M", "efConstruction", "efSearch",
             "build_s", "search_us_per_query", "cf_same", "cf_cross", "cf_mean"]
COLS_PQ = ["model", "index_modality", "family", "dsub", "m", "nlist", "factory", "nprobe",
           "build_s", "search_us_per_query", "cf_same", "cf_cross", "cf_mean"]
INT_COLS = {"M", "efConstruction", "efSearch", "dsub", "m", "nlist", "nprobe"}
FLOAT_COLS = {"build_s", "search_us_per_query", "cf_same", "cf_cross", "cf_mean"}


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def exact_topk(Q: np.ndarray, X: np.ndarray, k: int = K) -> np.ndarray:
    """Exact top-k by inner product; the vectors are unit-norm, so this is the
    L2 ranking every index below searches under."""
    S = Q @ X.T
    part = np.argpartition(-S, k, axis=1)[:, :k]
    return part


def cf_at_k(approx: np.ndarray, exact: np.ndarray, k: int = K) -> float:
    """Mean |approx top-k ∩ exact top-k| / k; -1 (a short result) matches nothing."""
    hits = [len(np.intersect1d(a[a >= 0], e, assume_unique=False)) for a, e in zip(approx, exact)]
    return float(np.mean(hits)) / k


def timed_search(index, Q: np.ndarray) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    _, I = index.search(Q, K)
    return I, (time.perf_counter() - t0) / len(Q) * 1e6


def run_info(args) -> dict:
    return {"host": os.uname().nodename, "pbs_jobid": os.environ.get("PBS_JOBID"),
            "faiss_version": faiss.__version__, "faiss_compile_options": faiss.get_compile_options(),
            "python": sys.executable, "threads": args.threads,
            "backend": getattr(args, "backend", "cpu")}


def grids() -> dict:
    return {"HNSW_M": HNSW_M, "HNSW_efConstruction": HNSW_EFC, "HNSW_efSearch": HNSW_EFS,
            "dsub": DSUB, "nprobe": NPROBE, "nlist": NLIST, "opq_niter": OPQ_NITER}


# ---- the sweep files ---------------------------------------------------------

def sweep_path(dest: Path, imod: str, family: str) -> Path:
    return dest / f"sweep_{imod}_{family}.csv"


def done_path(dest: Path, imod: str, family: str) -> Path:
    return dest / f"sweep_{imod}_{family}.done.json"


def columns(family: str) -> list:
    return COLS_HNSW if family == "HNSW" else COLS_PQ


def build_key(row: dict):
    return (row["M"], row["efConstruction"]) if row["family"] == "HNSW" else row["dsub"]


def n_search(family: str) -> int:
    return len(HNSW_EFS) if family == "HNSW" else len(NPROBE)


def write_rows(path: Path, cols: list, rows: list, mode: str = "a") -> None:
    header = mode == "w" or not path.exists()
    with open(path, mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if header:
            w.writeheader()
        w.writerows(rows)
        f.flush()
        os.fsync(f.fileno())


def read_rows(path: Path, model: str, imod: str, family: str) -> tuple[list, int]:
    """(readable rows, number of unreadable rows) of a sweep file. A last line
    without its newline was cut short by a kill during the write ("0.5312" can
    become a valid-looking "0.53"), so it counts as unreadable."""
    if not path.exists():
        return [], 0
    cols, rows, bad = columns(family), [], 0
    text = path.read_text()
    if text and not text.endswith("\n"):
        text = text[: text.rfind("\n") + 1]
        bad += 1
    for r in csv.DictReader(io.StringIO(text, newline="")):
        try:
            rows.append({c: int(r[c]) if c in INT_COLS else float(r[c]) if c in FLOAT_COLS
                         else r[c] for c in cols})
        except (TypeError, ValueError, KeyError):
            bad += 1
    other = {(r["model"], r["index_modality"], r["family"]) for r in rows} - {(model, imod, family)}
    if other:
        raise RuntimeError(f"{path}: rows of {sorted(other)}, expected {(model, imod, family)}")
    return rows, bad


def resume_rows(path: Path, model: str, imod: str, family: str) -> list:
    """The complete builds already in path. A build with missing rows is dropped,
    and the file is rewritten without it, so that build runs again."""
    rows, bad = read_rows(path, model, imod, family)
    count = Counter(build_key(r) for r in rows)
    keep = [r for r in rows if count[build_key(r)] == n_search(family)]
    if bad or len(keep) != len(rows):
        write_rows(path, columns(family), keep, mode="w")
        log(f"  {path.name}: dropped {bad + len(rows) - len(keep)} rows of an unfinished build")
    return keep


# ---- stage 1: the sample -----------------------------------------------------

def stage_sample(args, dest: Path) -> None:
    sdir = dest / "sample"
    if (sdir / "sample_meta.json").exists():
        log(f"{sdir} is complete -- nothing to do")
        return
    t_all = time.time()
    scope = json.load(open(SPLIT_DIR / "pair_index_meta.json"))["scope"]
    if scope != SPLIT_SCOPE:
        raise RuntimeError(f"{SPLIT_DIR}: scope {scope!r}, expected {SPLIT_SCOPE!r}")
    rng = np.random.default_rng(args.seed)
    a_ids = np.sort(rng.choice(np.load(SPLIT_DIR / "A_ids.npy"), N_A, replace=False))
    b_ids = np.sort(rng.choice(np.load(SPLIT_DIR / "B_ids.npy"), N_B, replace=False))
    pidx = SPLIT_DIR / "pair_index.csv"
    dim = MODEL_DIMS[args.model]

    vec, roots = {}, {}
    for mod in ("image", "text"):
        root = embed_dir(args.model, mod)
        roots[mod] = str(root)
        for role, ids in (("A", a_ids), ("B", b_ids)):
            v = np.ascontiguousarray(load_selected(args.model, mod, ids, pidx, root=root),
                                     dtype=np.float32)
            if v.shape != (len(ids), dim):
                raise RuntimeError(f"{role}/{mod}: shape {v.shape}, expected {(len(ids), dim)}")
            n = np.linalg.norm(v, axis=1)
            if np.abs(n - 1).max() > 1e-3:
                raise RuntimeError(f"{role}/{mod}: vectors are not unit-norm (max |n-1| "
                                   f"{np.abs(n - 1).max():.2e})")
            vec[(role, mod)] = v
        log(f"loaded {N_A:,} A + {N_B} B {mod} vectors from {root}")

    exact = {label: exact_topk(vec[("B", qmod)], vec[("A", imod)])
             for imod, dirs in DIRECTIONS.items() for label, qmod in dirs}
    log(f"exact top-{K} for {', '.join(exact)}")

    sdir.mkdir(parents=True, exist_ok=True)
    np.save(sdir / "a_ids.npy", a_ids.astype(np.int64))
    np.save(sdir / "b_ids.npy", b_ids.astype(np.int64))
    for (role, mod), v in vec.items():
        np.save(sdir / f"{role}_{mod}.npy", v)
    for label, e in exact.items():
        np.save(sdir / f"exact_{LABEL_FILE[label]}.npy", e.astype(np.int64))
    meta = {
        "task": "v2 index catalogue -- 50k tuning sample",
        "model": args.model, "created": datetime.now().isoformat(), **run_info(args),
        "n_A": N_A, "n_B": N_B, "seed": args.seed, "k": K, "dim": dim,
        "split_dir": str(SPLIT_DIR), "split_scope": scope, "embedding_roots": roots,
        "ids": "a_ids/b_ids: sorted int_ids of the split; row i of A_<mod>/B_<mod> is id i",
        "exact": ("exact_<dir>.npy: per B query, the unordered top-100 by inner product "
                  "(argpartition) as row numbers of the A sample"),
        "directions": {label: {"queries": f"B_{qmod}", "corpus": f"A_{imod}"}
                       for imod, dirs in DIRECTIONS.items() for label, qmod in dirs},
        "files": sorted(p.name for p in sdir.iterdir()),
        "wall_s": round(time.time() - t_all, 1),
    }
    (sdir / "sample_meta.json").write_text(json.dumps(meta, indent=2))
    log(f"wrote {sdir} in {time.time() - t_all:.0f}s")


def load_sample(args, dest: Path) -> tuple[dict, dict, dict]:
    sdir = dest / "sample"
    meta_p = sdir / "sample_meta.json"
    if not meta_p.exists():
        raise SystemExit(f"{meta_p} missing -- run --stage sample first")
    meta = json.load(open(meta_p))
    want = {"model": args.model, "n_A": N_A, "n_B": N_B, "seed": args.seed, "k": K,
            "split_scope": SPLIT_SCOPE}
    bad = {k: (meta.get(k), v) for k, v in want.items() if meta.get(k) != v}
    if bad:
        raise RuntimeError(f"{meta_p} does not match this run (stored, wanted): {bad}")
    vec = {(r, m): np.load(sdir / f"{r}_{m}.npy") for r in "AB" for m in ("image", "text")}
    exact = {label: np.load(sdir / f"exact_{f}.npy") for label, f in LABEL_FILE.items()}
    return meta, vec, exact


# ---- stage 2: one family on one index modality ---------------------------------

_GPU_RES = None


def build_pq_cuvs(family: str, d: int, m: int, X: np.ndarray):
    """IVF<NLIST>,PQ<m> (8 bits) trained and filled on the GPU with cuVS
    (use_cuvs, interleaved layout), then copied to a CPU IndexIVFPQ, so the search
    is the same CPU search as the cpu backend. O-IVFPQ: the OPQ rotation trains on
    the CPU (FAISS has no GPU OPQ; niter as the cpu backend) and the IVF-PQ part is
    built on the rotated vectors. Measured 2026-10-13: cuVS IVF-PQ gives 0.002-0.041
    lower CF than the CPU build at the same code (memory faiss-cuvs-env)."""
    global _GPU_RES
    if _GPU_RES is None:
        if faiss.get_num_gpus() == 0:
            raise SystemExit("--backend cuvs needs a GPU (request ngpus=1)")
        _GPU_RES = faiss.StandardGpuResources()
    opq, Xr = None, X
    if family == "O-IVFPQ":
        opq = faiss.OPQMatrix(d, m)
        opq.niter = OPQ_NITER
        opq.train(X)
        Xr = np.ascontiguousarray(opq.apply(X))
    cfg = faiss.GpuIndexIVFPQConfig()
    cfg.device = 0
    cfg.use_cuvs = True
    cfg.interleavedLayout = True
    g = faiss.GpuIndexIVFPQ(_GPU_RES, d, NLIST, m, 8, faiss.METRIC_L2, cfg)
    g.train(Xr)
    g.add(Xr)
    cpu = faiss.index_gpu_to_cpu(g)
    del g
    if opq is None:
        return cpu, [cpu]
    return faiss.IndexPreTransform(opq, cpu), [opq, cpu]     # keep both alive


def stage_sweep(args, dest: Path) -> None:
    imod, family = args.index_modality, args.family
    path, done_p = sweep_path(dest, imod, family), done_path(dest, imod, family)
    extra = sorted(set(args.dsub or []), reverse=True)     # --dsub: add these codes only
    if done_p.exists():
        if not extra:
            log(f"{done_p.name} exists -- this sweep is complete")
            return
        if set(extra) <= {build_key(r) for r in read_rows(path, args.model, imod, family)[0]}:
            log(f"{path.name} already holds d/m {extra} -- nothing to add")
            return
    t_all = time.time()
    meta, vec, exact = load_sample(args, dest)
    X = vec[("A", imod)]
    d = X.shape[1]
    dirs = DIRECTIONS[imod]
    Qs = {m: vec[("B", m)] for m in ("image", "text")}
    cols = columns(family)
    rows = resume_rows(path, args.model, imod, family)
    have = {build_key(r) for r in rows}
    log(f"{args.model} {imod} index, {family}: A {X.shape[0]:,} x {d}, {N_B} B queries "
        f"per direction ({', '.join(l for l, _ in dirs)})"
        + (f"; resuming, {len(have)} builds already in {path.name}" if have else ""))

    def grade(index, build: dict, search: dict, build_s: float) -> dict:
        cfs, us = {}, []
        for label, qmod in dirs:
            I, t = timed_search(index, Qs[qmod])
            cfs[label] = cf_at_k(I, exact[label])
            us.append(t)
        return {"model": args.model, "index_modality": imod, "family": family, **build, **search,
                "build_s": round(build_s, 3), "search_us_per_query": round(float(np.mean(us)), 1),
                "cf_same": cfs[dirs[0][0]], "cf_cross": cfs[dirs[1][0]],
                "cf_mean": (cfs[dirs[0][0]] + cfs[dirs[1][0]]) / 2}

    if family == "HNSW":
        for M in HNSW_M:
            for efc in HNSW_EFC:
                if (M, efc) in have:
                    continue
                h = faiss.IndexHNSWFlat(d, M, faiss.METRIC_L2)
                h.hnsw.efConstruction = efc
                t0 = time.perf_counter()
                h.add(X)
                b = time.perf_counter() - t0
                new = []
                for ef in HNSW_EFS:
                    h.hnsw.efSearch = ef
                    new.append(grade(h, {"M": M, "efConstruction": efc}, {"efSearch": ef}, b))
                write_rows(path, cols, new)
                rows += new
                log(f"  {imod} HNSW M={M:<2} efC={efc:<3} built {b:6.1f}s  "
                    f"mean CF {new[0]['cf_mean']:.3f} (ef {HNSW_EFS[0]}) .. "
                    f"{new[-1]['cf_mean']:.3f} (ef {HNSW_EFS[-1]})")
                del h
    else:
        # shortest code first, so the cheap builds are saved first; --dsub (user, 2026-10-13: O-IVFPQ d/m 1, so O-IVFPQ can
        # use exactly the IVFPQ codes) overrides the grid and its O-IVFPQ d/m 1 skip
        for dsub in (extra or sorted(DSUB, reverse=True)):
            if d % dsub or (not extra and family == "O-IVFPQ" and dsub == 1) or dsub in have:
                continue
            m = d // dsub
            spec = f"IVF{NLIST},PQ{m}" if family == "IVFPQ" else f"OPQ{m},IVF{NLIST},PQ{m}"
            t0 = time.perf_counter()
            if args.backend == "cuvs":
                index, keep = build_pq_cuvs(family, d, m, X)
            else:
                index = faiss.index_factory(d, spec, faiss.METRIC_L2)
                if family == "O-IVFPQ":
                    faiss.downcast_VectorTransform(
                        faiss.downcast_index(index).chain.at(0)).niter = OPQ_NITER
                index.train(X)
                index.add(X)
            b = time.perf_counter() - t0
            ivf = faiss.extract_index_ivf(index)
            new = []
            for npb in NPROBE:
                ivf.nprobe = npb
                new.append(grade(index, {"dsub": dsub, "m": m, "nlist": NLIST, "factory": spec},
                                 {"nprobe": npb}, b))
            write_rows(path, cols, new)
            rows += new
            log(f"  {imod} {family:<7} d/m={dsub:<2} m={m:<4} built {b:6.1f}s  "
                f"mean CF {new[0]['cf_mean']:.3f} (nprobe 1) .. "
                f"{max(r['cf_mean'] for r in new):.3f} (best)")
            del index

    done = {
        "task": "v2 index catalogue -- 50k sweep", "model": args.model,
        "index_modality": imod, "family": family, "complete": True,
        "created": datetime.now().isoformat(), **run_info(args),
        "rows": len(rows), "builds": len({build_key(r) for r in rows}),
        "added_dsub_this_run": extra or None,
        "grids": grids(), "targets_mean_of_two_directions": TARGETS,
        "sample_created": meta["created"], "sample_pbs_jobid": meta["pbs_jobid"],
        "wall_s_this_run": round(time.time() - t_all, 1),
    }
    done_p.write_text(json.dumps(done, indent=2))
    log(f"wrote {path.name} ({len(rows)} rows) and {done_p.name} in {time.time() - t_all:.0f}s")


# ---- stage 3: the proposal ---------------------------------------------------
# User, 2026-10-13: the BUILD is shared and the SEARCH parameter is per model.
#   HNSW   M and efConstruction: one per tier for all 8 indexes (4 models x
#          image/text); efSearch: one per tier per model (its image + text index).
#   PQ     d/m: one per tier for the image and text index of a model, the SAME
#          for every model -- a model takes another d/m only where its d
#          forbids the common one (d = 2048 has no d/m 3); nprobe: one per tier
#          per model.
# The bands are soft (user): a config is scored by how many of its indexes fall
# inside the tier's band, then by cost, then by the distance to the target in
# steps of 0.01 (about the sampling error of a CF from 500 queries, so a larger
# search parameter must gain a real 0.01 to win).

STEP = 0.01


def _cf(r: dict) -> dict:
    return {"same": r["cf_same"], "cross": r["cf_cross"], "mean": r["cf_mean"]}


def _joint(rows_by_index: dict, key) -> dict:
    """key -> {index: row} over the keys that every index has."""
    look = {ix: {key(r): r for r in rs} for ix, rs in rows_by_index.items()}
    return {k: {ix: look[ix][k] for ix in look}
            for k in set.intersection(*(set(v) for v in look.values()))}


def in_band(cf: float, tier: str) -> bool:
    lo, hi = BANDS[tier]
    return lo <= cf <= hi if tier == "high" else lo <= cf < hi


def score(cfs, tier: str) -> tuple[int, int]:
    """(indexes inside the band, summed distance to the target in STEPs)."""
    cfs = list(cfs)
    return (sum(in_band(c, tier) for c in cfs),
            sum(int(abs(c - TARGETS[tier]) / STEP + 0.5) for c in cfs))


def add(*ts) -> tuple:
    return tuple(map(sum, zip(*ts)))


def summary(rows: dict, tier: str) -> dict:
    cf = [r["cf_mean"] for r in rows.values()]
    return {"cf_avg": float(np.mean(cf)), "cf_min": min(cf), "cf_max": max(cf),
            "in_band": f"{sum(in_band(c, tier) for c in cf)}/{len(cf)}", "band": BANDS[tier],
            "per_index": {ix: _cf(r) for ix, r in rows.items()}}


def best_search(J: dict, build: tuple, grid: tuple, ixs: list, tier: str) -> tuple:
    """(inside, distance, value): the search value of `grid` for build + (value,)
    with the most of ixs inside the band, then the smallest distance, then the
    smallest value."""
    cand = []
    for v in grid:
        if build + (v,) in J:
            inside, dist = score((J[build + (v,)][ix]["cf_mean"] for ix in ixs), tier)
            cand.append((inside, -dist, -v))
    inside, ndist, nv = max(cand)
    return inside, -ndist, -nv


def propose_pq(rows_by_model: dict) -> dict:
    """rows_by_model: model -> {index: rows} of one PQ family. Per model, a code
    (d/m) per tier for its image and text index, strictly longer low < mod < high,
    and a nprobe per tier. A reference d/m triple R is shared by all models: a
    model uses R's d/m wherever its d allows it, and only where its d forbids one
    (d = 2048 has no d/m 3) takes another d/m that keeps the order (user,
    2026-10-13: Jina v4 the same 4/2/1 as Qwen3-VL). Under each R every model
    takes its best such triple -- the most indexes inside the bands, then the
    shortest codes, then the smallest distance -- and the R with the best total
    wins."""
    models = list(rows_by_model)
    Js = {m: _joint(rbi, lambda r: (r["dsub"], r["nprobe"])) for m, rbi in rows_by_model.items()}
    codes = {m: sorted({k[0] for k in J}, reverse=True) for m, J in Js.items()}  # shortest first
    short = {m: c for m, c in codes.items() if len(c) < 3}
    if short:
        raise RuntimeError(f"need 3 codes per model for 3 tiers, have {short}")
    # best nprobe per (model, code, tier), for both indexes of the model
    S = {(m, c, t): best_search(Js[m], (c,), NPROBE, list(rows_by_model[m]), t)
         for m in models for c in codes[m] for t in TIERS}
    trips = {m: [tuple(codes[m][i] for i in ijk)
                 for ijk in combinations(range(len(codes[m])), 3)] for m in models}
    rank = {m: {c: i for i, c in enumerate(codes[m])} for m in models}

    def model_best(m, R):
        fixed = [c if c in rank[m] else None for c in R]          # None: d forbids it
        cands = [tr for tr in trips[m] if all(f is None or f == c for f, c in zip(fixed, tr))]
        if not cands:
            return None

        def key(tr):
            ss = [S[(m, c, t)] for c, t in zip(tr, TIERS)]
            return sum(s[0] for s in ss), -sum(rank[m][c] for c in tr), -sum(s[1] for s in ss)
        tr = max(cands, key=key)
        return key(tr), tr

    all_codes = sorted({c for m in models for c in codes[m]}, reverse=True)
    best = None
    for R in combinations(all_codes, 3):            # (low, mod, high): codes get longer
        per = {m: model_best(m, R) for m in models}
        if any(v is None for v in per.values()):
            continue
        total = add(*(k for k, _ in per.values()))
        if best is None or total > best[0]:
            best = (total, R, per)
    if best is None:
        raise RuntimeError("no reference d/m triple fits every model")
    _, R, per = best

    out = {"reference_dsub": dict(zip(TIERS, R)), "models": {}}
    for m, (_, tr) in per.items():
        tiers = {}
        for c, t in zip(tr, TIERS):
            npb = S[(m, c, t)][2]
            rows = Js[m][(c, npb)]
            any_row = next(iter(rows.values()))
            tiers[t] = {"dsub": c, "m": any_row["m"], "factory": any_row["factory"], "nprobe": npb,
                        "dsub_as_reference": c == R[TIERS.index(t)], **summary(rows, t)}
        plateaus = [{"dsub": c, **{ix: max(Js[m][(c, n)][ix]["cf_mean"] for n in NPROBE
                                           if (c, n) in Js[m]) for ix in rows_by_model[m]}}
                    for c in codes[m]]
        out["models"][m] = {"tiers": tiers, "plateaus": plateaus}
    return out


def propose_hnsw(rows_by_index: dict) -> dict:
    """One (M, efConstruction) per tier for every index of every model, M strictly
    larger low < mod < high; efSearch per tier per model (its image and text
    index). Per M: the efConstruction with the most indexes inside the band, then
    the smallest efConstruction, then the smallest distance; then the M triple
    with the best total, then the smallest M."""
    J = _joint(rows_by_index, lambda r: (r["M"], r["efConstruction"], r["efSearch"]))
    models = sorted({ix.split("/")[0] for ix in rows_by_index}, key=MODELS.index)
    Ms = sorted({k[0] for k in J})
    if len(Ms) < 3:
        raise RuntimeError(f"need 3 values of M for 3 tiers, have {Ms}")

    def best(M, tier):
        cand = []
        for efc in HNSW_EFC:
            per = {m: best_search(J, (M, efc), HNSW_EFS, [f"{m}/image", f"{m}/text"], tier)
                   for m in models}
            inside, dist = sum(p[0] for p in per.values()), sum(p[1] for p in per.values())
            cand.append(((inside, -efc, -dist), efc, {m: p[2] for m, p in per.items()}, dist))
        (inside, _, _), efc, efs, dist = max(cand, key=lambda c: c[0])
        rows = {ix: J[(M, efc, efs[ix.split("/")[0]])][ix] for ix in rows_by_index}
        return inside, efc, dist, {"M": M, "efConstruction": efc, "efSearch": efs,
                                   **summary(rows, tier)}

    B = {(M, t): best(M, t) for M in Ms for t in TIERS}

    def key(ijk):
        picks = [B[(Ms[i], t)] for i, t in zip(ijk, TIERS)]
        return (sum(p[0] for p in picks), -sum(ijk), -sum(p[1] for p in picks),
                -sum(p[2] for p in picks))
    trip = max(combinations(range(len(Ms)), 3), key=key)
    return {"tiers": {t: B[(Ms[i], t)][3] for i, t in zip(trip, TIERS)}}


def follow_ivfpq(ivfpq: dict, rows_by_model: dict) -> dict:
    """O-IVFPQ takes EXACTLY the IVFPQ tier parameters, d/m and nprobe (user,
    2026-10-13); this reports its CF there."""
    out = {"rule": "same d/m and nprobe as the IVFPQ tier", "models": {}}
    for m, rbi in rows_by_model.items():
        J = _joint(rbi, lambda r: (r["dsub"], r["nprobe"]))
        tiers = {}
        for t in TIERS:
            q = ivfpq["models"][m]["tiers"][t]
            key = (q["dsub"], q["nprobe"])
            if key not in J:
                raise SystemExit(f"O-IVFPQ {m}: no rows for d/m {key[0]}, nprobe {key[1]} -- "
                                 f"sweep it with --dsub {key[0]}")
            any_row = next(iter(J[key].values()))
            tiers[t] = {"dsub": key[0], "m": any_row["m"], "factory": any_row["factory"],
                        "nprobe": key[1], **summary(J[key], t)}
        out["models"][m] = {"tiers": tiers}
    return out


def stage_propose(args, root: Path) -> None:
    rows, sweeps, samples, missing = {}, {}, {}, []
    for model in MODELS:
        dest = root / model
        samples[model] = {k: v for k, v in json.load(open(dest / "sample" / "sample_meta.json")).items()
                          if k in ("n_A", "n_B", "seed", "split_scope", "embedding_roots",
                                   "created", "pbs_jobid")}
        for imod in ("image", "text"):
            for fam in FAMILIES:
                dp = done_path(dest, imod, fam)
                if not dp.exists():
                    if fam == "O-IVFPQ":                    # optional: it follows IVFPQ
                        missing.append(f"{model}/{imod}")
                        continue
                    raise SystemExit(f"{dp} missing -- the {model} {imod} {fam} sweep is not complete")
                done = json.load(open(dp))
                rs, bad = read_rows(sweep_path(dest, imod, fam), model, imod, fam)
                if bad or len(rs) != done["rows"]:
                    raise RuntimeError(f"{sweep_path(dest, imod, fam)}: {len(rs)} readable rows "
                                       f"({bad} unreadable), {dp.name} says {done['rows']}")
                rows[(model, imod, fam)] = rs
                sweeps[f"{model}/{imod}/{fam}"] = {k: done.get(k) for k in
                                                   ("rows", "builds", "pbs_jobid", "host", "backend",
                                                    "faiss_version", "threads", "created")}
    ivfpq = propose_pq({m: {f"{m}/{x}": rows[(m, x, "IVFPQ")] for x in ("image", "text")}
                        for m in MODELS})
    oivfpq = (follow_ivfpq(ivfpq, {m: {f"{m}/{x}": rows[(m, x, "O-IVFPQ")] for x in ("image", "text")}
                                   for m in MODELS})
              if not missing else {"rule": "same d/m and nprobe as the IVFPQ tier",
                                   "not_swept": missing,
                                   "codes_to_sweep": {m: sorted({ivfpq["models"][m]["tiers"][t]["dsub"]
                                                                 for t in TIERS}, reverse=True)
                                                      for m in MODELS}})
    proposal = {
        "task": "v2 index catalogue -- 50k sweep and shared tier proposal",
        "created": datetime.now().isoformat(), **run_info(args), "root": str(root),
        "rules": {
            "shared": "build shared, search per model (user, 2026-10-13): HNSW M and efC one "
                      "per tier for all 8 indexes, efSearch per tier per model; IVFPQ d/m one per "
                      "tier for the image and text index of a model, the reference d/m "
                      "(reference_dsub) wherever the model's d allows it, nprobe per tier per "
                      "model; O-IVFPQ exactly the IVFPQ d/m and nprobe",
            "cf": "CF@100 of an index = mean of its two directions (image: i->i, t->i; "
                  "text: t->t, i->t)",
            "bands": "soft (user): low [0.65, 0.75), mod [0.75, 0.85), high >= 0.85; targets "
                     "0.70 / 0.80 / 0.90 break ties",
            "score": "tiers take strictly larger builds low < mod < high (PQ: longer code; "
                     "HNSW: larger M). Most indexes inside their band first, then the "
                     "cheapest builds, then the "
                     f"smallest distance to the targets in steps of {STEP}, then the smallest "
                     "search parameter",
            "nprobe_at_5.8M": f"nlist {NLIST} here stands for 32,768: start nprobe at "
                              f"{32768 // NLIST} x the value here, then check at full scale",
        },
        "bands": BANDS, "targets": TARGETS, "distance_step": STEP, "nlist_at_50k": NLIST,
        "grids": grids(), "samples": samples, "sweeps": sweeps,
        "caveat": "CF falls with corpus size; check the tiers once at 5.8M after the build",
        "HNSW": propose_hnsw({f"{m}/{x}": rows[(m, x, "HNSW")]
                              for m in MODELS for x in ("image", "text")}),
        "IVFPQ": ivfpq, "O-IVFPQ": oivfpq,
    }
    (root / "proposal.json").write_text(json.dumps(proposal, indent=2))
    log(f"wrote {root}/proposal.json from {sum(map(len, rows.values())):,} sweep rows")

    def rng(p):
        return f"[{p['cf_min']:.2f}..{p['cf_max']:.2f}] in {p['in_band']}"
    H = proposal["HNSW"]["tiers"]
    for t in TIERS:
        log(f"  HNSW {t:<4} M{H[t]['M']}/efC{H[t]['efConstruction']}  ef "
            + "/".join(str(H[t]["efSearch"][m]) for m in MODELS) + f"  {rng(H[t])}")
    log("  IVFPQ reference d/m " + " | ".join(f"{t}: {ivfpq['reference_dsub'][t]}" for t in TIERS))
    for fam, P in (("IVFPQ", ivfpq), ("O-IVFPQ", oivfpq)):
        if "models" not in P:
            log(f"  {fam}: not swept yet ({len(P['not_swept'])} indexes); codes to sweep "
                f"{P['codes_to_sweep']}")
            continue
        for m in MODELS:
            T = P["models"][m]["tiers"]
            log(f"    {fam:<7} {m:<9}  " + " | ".join(
                f"{t}: d/m {T[t]['dsub']} np{T[t]['nprobe']} {rng(T[t])}" for t in TIERS))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("sample", "sweep", "propose"))
    ap.add_argument("--model", choices=MODELS, help="--stage sample / sweep")
    ap.add_argument("--index-modality", choices=("image", "text"), help="--stage sweep only")
    ap.add_argument("--family", choices=FAMILIES, help="--stage sweep only")
    ap.add_argument("--dsub", type=int, nargs="+", help="--stage sweep, PQ families: build only "
                    "these d/m and add their rows to the sweep file (also to a complete one)")
    ap.add_argument("--backend", choices=("cpu", "cuvs"), default="cpu",
                    help="PQ families: cpu = FAISS CPU build (the 50k/ tuning); cuvs = GPU IVF-PQ "
                         "with cuVS, copied to a CPU index (needs .venv-faisscuvs and a GPU)")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", 16)))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=V2 / "index_tuning" / "50k")
    ap.add_argument("--allow-login-node", action="store_true")
    args = ap.parse_args()
    if args.stage != "propose" and not args.model:
        ap.error(f"--stage {args.stage} needs --model")
    if args.stage == "sweep" and not (args.index_modality and args.family):
        ap.error("--stage sweep needs --index-modality and --family")
    if not os.environ.get("PBS_JOBID") and not args.allow_login_node:
        raise SystemExit("Refusing to run outside a PBS job -- submit "
                         "indexes/jobs/tune_catalogue_v2_<stage>.pbs")
    faiss.omp_set_num_threads(args.threads)
    if args.stage == "propose":
        stage_propose(args, args.out_dir)
        return
    dest = args.out_dir / args.model
    dest.mkdir(parents=True, exist_ok=True)
    {"sample": stage_sample, "sweep": stage_sweep}[args.stage](args, dest)


if __name__ == "__main__":
    main()
