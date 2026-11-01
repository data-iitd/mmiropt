"""
Cells of every construction (B) and test (C) query under the HNSW landmark lookup.

    python paperexp/hnsw_assign.py --model siglip2 --out-dir result/paper_experiments/7M_cleaned_avx2

The stored cell assignments come from an exact scan of the landmarks. When the
online stage finds the landmark with HNSW instead (split_single_query.py
--hnsw-sweep), a small share of queries lands in a neighbouring cell. To measure
what that does to plan selection, this gives every query the cell the HNSW lookup
gives it, at every K of the sweep, with the setting paperexp/hnsw_choice.py picks
from that K's sweep file (<out-dir>/<model>/split_lookup_variants[_K<k>].json).
Where the rule falls back to the exact scan, the stored assignment is copied.

The graph is rebuilt here with the sweep's parameters (L2, efConstruction 200,
64 threads); multi-threaded insertion is not deterministic, so it is the same
setting, not the byte-identical graph the timings came from.

V2 (--v2, user 2026-10-30): query vectors from the v2 embedding trees; run with
SFM_RESULTS_ROOT=result_v2, SFM_K_MAIN=SFM_KS=<K>.

NO TIMING. Output, per K and direction (row order = role_ids, as stored):
    <out-dir>/<model>/hnsw_assign/K<k>/<pairing>_{B,C}.npy     int32 landmark ids
    <out-dir>/<model>/hnsw_assign.json                          settings, agreement
Consumers: selection_tables.py --assign-dir (Q_test cells) and
matched_violation.py --assign-dir (Q_cal and Q_test cells).
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
for sub in ("paperexp", "indexes", "knn", "spatialmap"):
    sys.path.insert(0, str(REPO_ROOT / sub))

from index_configs import MODEL_DIMS                                  # noqa: E402
from load_vectors import load_included_shards, load_role_matrix      # noqa: E402
from exact_knn import DEFAULT_MANIFEST, load_pair_index_roles        # noqa: E402
from map_config import PAIRINGS, load_query_geometry                 # noqa: E402
from selection_tables import KS, R, SCALE, lay_at                    # noqa: E402
from hnsw_choice import choose, choose_uniform, parse, sweep_name    # noqa: E402
from v2_dirs import embed_dir                                        # noqa: E402

EF_CONSTRUCTION = 200
CHUNK = 65536


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(MODEL_DIMS))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--pilot-dir", type=Path, default=R / SCALE / "siglip2")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", 16)))
    ap.add_argument("--v2", action="store_true", help="query vectors from the v2 embedding trees")
    ap.add_argument("--uniform", action="store_true",
                    help="one setting for all directions (hnsw_choice.choose_uniform, user 2026-10-31)")
    ap.add_argument("--allow-login-node", action="store_true")
    args = ap.parse_args()
    if not os.environ.get("PBS_JOBID") and not args.allow_login_node:
        raise SystemExit("Refusing to run outside a PBS job -- submit "
                         "paperexp/jobs/paperexp_hnsw_assign_7m.pbs")
    faiss.omp_set_num_threads(args.threads)
    t0 = time.time()
    dest = args.out_dir / args.model

    sweeps = {}
    for k in KS:
        p = dest / sweep_name(k)
        d = json.loads(p.read_text()) if p.exists() else None
        if not d or not d.get("complete"):
            raise SystemExit(f"{p} missing or incomplete -- run the HNSW sweep at K={k} first")
        sweeps[k] = d

    lay = lay_at(args.model, 2500)
    ids = {r: np.load(lay.role_ids(r)) for r in ("B", "C")}
    all_q = np.sort(np.concatenate([ids["B"], ids["C"]]))
    if len(np.unique(all_q)) != len(all_q):
        raise RuntimeError("B and C share ids")
    rows = {r: np.searchsorted(all_q, ids[r]) for r in ("B", "C")}
    pair_to_int = load_pair_index_roles(args.pilot_dir / "pair_index.csv", ["B", "C"])
    included = load_included_shards(args.manifest)
    qvecs = {}
    for qm in sorted({s["query_modality"] for s in PAIRINGS.values()}):
        log(f"loading {len(all_q):,} {qm} query vectors (B + C)")
        qvecs[qm] = np.ascontiguousarray(load_role_matrix(
            args.model, qm, all_q, pair_to_int, included,
            expect_dim=MODEL_DIMS[args.model],
            root=embed_dir(args.model, qm) if args.v2 else None), dtype=np.float32)

    summary = {"task": "cells of B and C under the HNSW landmark lookup", "model": args.model,
               "created": datetime.now().isoformat(), "host": os.uname().nodename,
               "pbs_jobid": os.environ.get("PBS_JOBID"), "faiss_version": faiss.__version__,
               "faiss_compile_options": faiss.get_compile_options(), "threads": args.threads,
               "ef_construction": EF_CONSTRUCTION,
               "choice_rule": "choose_uniform (one setting, <= 2 directions below agreement)" if args.uniform
                              else "choose (per direction, exact-scan fallback)",
               "method": __doc__.split("NO TIMING")[0].strip(),
               "by_K": {}}
    for k in KS:
        choice = choose_uniform(sweeps[k]) if args.uniform else choose(sweeps[k])
        lk = lay_at(args.model, k)
        kdir = dest / "hnsw_assign" / f"K{k}"
        kdir.mkdir(parents=True, exist_ok=True)
        summary["by_K"][str(k)] = {}
        for p in sorted(PAIRINGS):
            spec = PAIRINGS[p]
            stored = {r: load_query_geometry(lk, p, r)["assign"].astype(np.int64) for r in ("B", "C")}
            setting = parse(choice[p])
            e = {"lookup": choice[p]}
            if setting is None:
                got = stored
            else:
                M, ef = setting
                cent = np.ascontiguousarray(np.load(lk.centroids(spec["index_modality"])),
                                            dtype=np.float32)
                h = faiss.IndexHNSWFlat(cent.shape[1], M, faiss.METRIC_L2)
                h.hnsw.efConstruction = EF_CONSTRUCTION
                h.add(cent)
                h.hnsw.efSearch = ef
                q = qvecs[spec["query_modality"]]
                t1 = time.time()
                allg = np.concatenate([h.search(q[s:s + CHUNK], 1)[1][:, 0]
                                       for s in range(0, len(q), CHUNK)])
                e["search_seconds"] = time.time() - t1
                got = {r: allg[rows[r]].astype(np.int64) for r in ("B", "C")}
            for r in ("B", "C"):
                if len(got[r]) != len(stored[r]):
                    raise RuntimeError(f"K={k} {p} {r}: {len(got[r])} rows, stored {len(stored[r])}")
                np.save(kdir / f"{p}_{r}.npy", got[r].astype(np.int32))
                e[f"cell_agreement_{r}"] = float((got[r] == stored[r]).mean())
            summary["by_K"][str(k)][p] = e
            log(f"  K={k:<6} {spec['short']:<5} {choice[p]:<10} cell agreement "
                f"B {e['cell_agreement_B']:.4f}  C {e['cell_agreement_C']:.4f}")
    (dest / "hnsw_assign.json").write_text(json.dumps(summary, indent=1))
    log(f"wrote {dest / 'hnsw_assign'} and hnsw_assign.json in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
