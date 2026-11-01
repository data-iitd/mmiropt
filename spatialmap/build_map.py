"""
T9, part 2 of 2 -- the served spatial fidelity map.

    python build_map.py --model siglip2 --spherical --histogram

Groups T8's per-query CF@k over T5's landmark cells and stores, for every
(landmark, configuration, modality_configuration) cell built from role B:

    n  cf_mean  cf_std  cf_sem  cf_min  cf_max  reliable  d1_mean  gap_mean  corpus_count

That table IS the map. At query time T10 finds a query's nearest landmark
(O(K), index-agnostic) and reads the row off instead of running exact-KNN.

THE BACKOFF RULE
----------------
A cell holding fewer than RELIABILITY_FLOOR (10) B queries gets reliable=0. Its
mean is still written -- discarding it would lose information H4 needs when it
sweeps K into the thin regime -- but T10 must serve the GA number for that
configuration in its place. GA is therefore not merely a baseline to beat in H5;
it is a live component of the served system, so it is computed here and stored
beside the table in ga_fallback.json rather than being left for T10 to re-derive
from data it might filter differently.

WHY B ONLY
----------
Cells are built from B and from B alone. C is the held-out grader: a single C
query entering a cell mean would make T10's evaluation partly a test on training
data, and the effect would be invisible in every metric. C's geometry is emitted
by assign_queries.py so H1/H2 can use it, but no CF of C's is ever aggregated
here. The --include-c flag exists only for the leakage demonstration in T10 and
writes to a separate, clearly-named file.

ALIGNMENT
---------
T8 writes cf_{pairing}_{role}_{config}.npy in role order -- row i is the CF of
{role}_ids[i]. T5's assignment is in the same order. So the join is positional,
and the guard is that both have the length of {role}_ids.npy. A partial T7 run
(--limit-q) produces a shorter CF vector covering a PREFIX of the role; that is
accepted, and the prefix length is recorded, but a longer one is an error.
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from map_config import (                                               # noqa: E402
    DEFAULT_CF, DEFAULT_KMEANS, DEFAULT_OUT, DEFAULT_PILOT, MAP_COLUMNS,
    PAIRINGS, PAPER_FLOORS, REFERENCE_MODEL, RELIABILITY_FLOOR,
    RELIABILITY_FLOORS, config_family, layout, load_query_geometry,
    load_role_ids, map_columns, resolve_configs, resolve_pairings, write_json,
    CF_HIST_BINS, cf_histogram,
)


# --------------------------------------------------------------------------
# CF loading
# --------------------------------------------------------------------------
def load_cf(lay, pairing: str, role: str, config: str,
            n_expected: int) -> Optional[np.ndarray]:
    """
    One config's CF vector for one (pairing, role), or None if T8 has not
    produced it. Length is validated against the role, never assumed.
    """
    p = lay.cf_vector(pairing, role, config)
    if not p.exists():
        return None
    cf = np.load(p).astype(np.float32)
    if len(cf) > n_expected:
        raise RuntimeError(
            f"{p.name}: {len(cf):,} CF values but role {role} holds "
            f"{n_expected:,} queries. T8 scored queries T3 never allocated.")
    if cf.size and (cf.min() < 0.0 or cf.max() > 1.0):
        raise RuntimeError(f"{p.name}: CF outside [0,1] "
                           f"([{cf.min():.3f}, {cf.max():.3f}])")
    return cf


def load_cf_from_table(lay, pairings, configs, roles) -> Dict:
    """
    Fallback reader for T8's merged cf_table.csv, used when the per-config .npy
    files are absent. Streams the file once -- it is ~1.3M rows per pairing --
    and indexes by query_id rather than by position, since a CSV carries no
    ordering guarantee.
    """
    table = lay.cf_table()
    if not table.exists():
        return {}
    want_p, want_c, want_r = set(pairings), set(configs), set(roles)
    out: Dict = {}
    with open(table) as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            p, c, r = (row["modality_configuration"], row["configuration"],
                       row["role"])
            if p not in want_p or c not in want_c or r not in want_r:
                continue
            out.setdefault((p, r, c), {})[int(row["query_id"])] = float(row["cf_score"])
    return out


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------
def aggregate(assign: np.ndarray, cf: np.ndarray, K: int):
    """
    Per-cell n, mean, std (population), min, max -- in four bincounts rather
    than a Python loop over K cells x 9 configs x 4 pairings.

    std is the population standard deviation of the cell's B queries, i.e. the
    spread of true CF inside the cell. It is what P1 (smoothness) predicts
    should be small; H1 reads it directly.
    """
    n = np.bincount(assign, minlength=K).astype(np.int64)
    s1 = np.bincount(assign, weights=cf.astype(np.float64), minlength=K)
    s2 = np.bincount(assign, weights=(cf.astype(np.float64) ** 2), minlength=K)

    nz = n > 0
    mean = np.zeros(K); std = np.zeros(K)
    mean[nz] = s1[nz] / n[nz]
    # max(.,0) guards the float cancellation that makes E[x^2]-E[x]^2 go a few
    # ulps negative when every value in a cell is identical (common at CF=0).
    var = np.zeros(K)
    var[nz] = np.maximum(s2[nz] / n[nz] - mean[nz] ** 2, 0.0)
    std = np.sqrt(var)

    # Seeded at +/-inf, not nan: np.minimum PROPAGATES nan, so a nan seed would
    # leave every cell nan however many values were folded into it.
    cmin = np.full(K, np.inf); cmax = np.full(K, -np.inf)
    np.minimum.at(cmin, assign, cf.astype(np.float64))
    np.maximum.at(cmax, assign, cf.astype(np.float64))
    cmin[~nz] = np.nan
    cmax[~nz] = np.nan

    sem = np.zeros(K)
    many = n > 1
    sem[many] = std[many] / np.sqrt(n[many])
    return n, mean, std, sem, cmin, cmax


def cell_geometry(assign: np.ndarray, d1: np.ndarray, gap: np.ndarray, K: int):
    """Mean distance-to-landmark and mean boundary gap of each cell's B queries."""
    n = np.bincount(assign, minlength=K).astype(np.int64)
    nz = n > 0
    dm = np.zeros(K); gm = np.zeros(K)
    dm[nz] = np.bincount(assign, weights=d1.astype(np.float64), minlength=K)[nz] / n[nz]
    gm[nz] = np.bincount(assign, weights=gap.astype(np.float64), minlength=K)[nz] / n[nz]
    return dm, gm


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="siglip2")
    ap.add_argument("--k", type=int, default=1024)
    ap.add_argument("--spherical", action="store_true")
    ap.add_argument("--pairing", nargs="+", default=["all"])
    ap.add_argument("--configs", default="all", help="comma-separated, or 'all'")
    ap.add_argument("--floor", type=int, default=RELIABILITY_FLOOR,
                    help=f"the PRIMARY floor, used for the console summary and "
                         f"as T10's default policy (default {RELIABILITY_FLOOR}, "
                         f"from the T9 dispatch)")
    ap.add_argument("--floors", type=int, nargs="+", default=list(RELIABILITY_FLOORS),
                    help=f"every floor to emit a reliable_<floor> column for. "
                         f"Default {list(RELIABILITY_FLOORS)}: the dispatch's "
                         f"{RELIABILITY_FLOOR} plus the paper's S6.4 range "
                         f"{list(PAPER_FLOORS)}, so T10 can switch policy "
                         f"without a rebuild.")
    ap.add_argument("--cf-histogram", action="store_true",
                    help="also store the map's 20-bin CF histogram per (cell, config) "
                         "(draft Step 4: mean + histogram; cf_histogram.npz)")
    ap.add_argument("--histogram", action="store_true",
                    help="also write the queries-per-cell histogram (a deliverable)")
    ap.add_argument("--include-c", action="store_true",
                    help="ALSO build a C-contaminated map, for T10's leakage demo "
                         "only. Never the served table.")
    ap.add_argument("--pilot-dir", type=Path, default=DEFAULT_PILOT)
    ap.add_argument("--kmeans-dir", type=Path, default=DEFAULT_KMEANS)
    ap.add_argument("--cf-dir", type=Path, default=DEFAULT_CF)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    lay = layout(args.model, args.k, args.spherical, pilot=args.pilot_dir,
                 kmeans=args.kmeans_dir, cf=args.cf_dir, out=args.out_dir)
    pairings = resolve_pairings(args.pairing)
    configs = resolve_configs(args.model, args.configs)

    # The primary floor must be among the emitted ones, or the console summary
    # and verify_map would describe a column the table does not carry.
    floors = sorted(set(args.floors) | {args.floor})
    columns = map_columns(floors)

    print(f"T9 spatial map  model={lay.model}  tiling={lay.variant}")
    print(f"  floors: {floors}  (primary {args.floor})")
    if lay.uses_shared_split:
        print(f"  split:  shared {REFERENCE_MODEL} A/B/C "
              f"(the 1M pilot's shared-split control)")
    print(f"  CF from: {lay.cf_dir()}")
    print(f"  dest:    {lay.dest}\n")

    if not lay.cf_dir().exists():
        sys.exit(f"no CF labels at {lay.cf_dir()} -- T8 has not run for this model.\n"
                 f"  run: knn/cf_compute.py --model {lay.model} --pairing all")

    b_ids = load_role_ids(lay, "B")

    # The CSV fallback is parsed lazily, on the first config whose .npy is
    # missing, and never when they are all present. cf_table.csv is the merged
    # per-query table -- ~200 MB and millions of rows at 1M pilot scale -- so
    # loading it eagerly costs a full parse to answer a question the .npy files
    # have usually already answered.
    _fallback: Dict = {}

    def fallback_table() -> Dict:
        if not _fallback:
            t = load_cf_from_table(lay, pairings, configs, ["B"])
            _fallback["t"] = t
            print(f"  a per-config .npy was missing; parsed cf_table.csv "
                  f"({len(t)} config blocks)\n" if t else
                  "  a per-config .npy was missing and no cf_table.csv exists\n")
        return _fallback.get("t") or {}

    meta = {
        "created": datetime.now().isoformat(),
        "task": "T9 spatial map construction",
        "model": lay.model, "K": lay.k, "spherical": lay.spherical,
        "reliability_floor": args.floor,
        "reliability_floors": floors,
        "floor_provenance": {
            "10": "T9 dispatch: flag cells under ~10 B queries, serve GA instead",
            "30_50": "paper S6.4: size B to clear 30-50 in sparse cells; cells "
                     "below are an H4 finding about landmark resolution, "
                     "'not silently averaged away'",
            "note": "The two instructions differ in kind, not just in value -- "
                    "one prescribes a repair (GA backoff), the other a report. "
                    "T9 emits a flag per floor and leaves the policy to T10.",
        },
        "built_from_role": "B",
        "cf_source": str(lay.cf_dir()),
        "tiling_source": str(lay.kmeans / lay.model),
        "split_source": str(lay.role_ids("B").parent),
        "uses_shared_split": lay.uses_shared_split,
        "columns": columns,
        "pairings": {},
        "missing": [],
    }
    ga_all: Dict[str, Dict[str, float]] = {}

    for pairing in pairings:
        spec = PAIRINGS[pairing]
        pdir = lay.pairing_dir(pairing)
        pdir.mkdir(parents=True, exist_ok=True)

        geom = load_query_geometry(lay, pairing, "B")
        assign, d1, gap = geom["assign"], geom["d1"], geom["gap"]
        if len(assign) != len(b_ids):
            raise RuntimeError(
                f"{pairing}: {len(assign):,} assignments vs {len(b_ids):,} B ids")

        K = int(np.load(lay.centroids(spec["index_modality"])).shape[0])
        if K != lay.k:
            raise RuntimeError(f"{lay.centroids(spec['index_modality'])} holds {K} "
                               f"centroids but --k is {lay.k}")
        corpus = np.load(lay.cell_counts(spec["index_modality"])).astype(np.int64)

        rows: List[list] = []
        per_config = {}
        ga: Dict[str, float] = {}
        present: List[str] = []
        cf_hists: Dict[str, np.ndarray] = {}

        for cfg in configs:
            cf = load_cf(lay, pairing, "B", cfg, len(b_ids))
            if cf is None:
                by_id = fallback_table().get((pairing, "B", cfg))
                if by_id:
                    cf = np.array([by_id.get(int(q), np.nan) for q in b_ids],
                                  dtype=np.float32)
                    if np.isnan(cf).any():
                        cf = None            # partial coverage: treat as absent
            if cf is None:
                meta["missing"].append(f"{pairing}/B/{cfg}")
                continue
            present.append(cfg)

            # A partial T7 run covers a prefix of B. Slice the assignment to
            # match rather than aligning a short CF against a full assignment,
            # which would pair each score with the wrong query.
            m = len(cf)
            a_m, d_m, g_m = assign[:m], d1[:m], gap[:m]

            n, mean, std, sem, cmin, cmax = aggregate(a_m, cf, K)
            if args.cf_histogram:
                cf_hists[cfg] = cf_histogram(a_m, cf, K)
                if not np.array_equal(cf_hists[cfg].sum(axis=1), n):
                    raise RuntimeError(f"{pairing}/{cfg}: histogram counts != cell n")
            dm, gm = cell_geometry(a_m, d_m, g_m, K)
            rel_by_floor = {f: (n >= f).astype(np.int8) for f in floors}
            reliable = rel_by_floor[args.floor]

            # GA: the single number this config predicts for every query --
            # H5's first baseline, and the backoff value for unreliable cells.
            ga[cfg] = float(cf.mean())

            for c in np.nonzero(n)[0]:
                rows.append([
                    int(c), cfg, pairing, int(n[c]),
                    round(float(mean[c]), 6), round(float(std[c]), 6),
                    round(float(sem[c]), 6), round(float(cmin[c]), 4),
                    round(float(cmax[c]), 4),
                    round(float(dm[c]), 6), round(float(gm[c]), 6),
                    int(corpus[c]),
                ] + [int(rel_by_floor[f][c]) for f in floors])

            occupied = int((n > 0).sum())
            rel = int(reliable.sum())
            per_config[cfg] = {
                "family": config_family(lay.model, cfg),
                "n_b_queries": int(m),
                "partial": bool(m < len(b_ids)),
                "occupied_cells": occupied,
                "reliable_cells": rel,
                "unreliable_cells": occupied - rel,
                "b_covered_by_reliable": float(n[reliable == 1].sum() / max(n.sum(), 1)),
                # What each candidate policy would cost, so H4 can read the
                # resolution trade-off straight off the map.
                "by_floor": {
                    str(f): {
                        "reliable_cells": int(rel_by_floor[f].sum()),
                        "b_covered": float(
                            n[rel_by_floor[f] == 1].sum() / max(n.sum(), 1)),
                    } for f in floors
                },
                "ga_mean": ga[cfg],
                "cell_mean_of_means": float(mean[n > 0].mean()),
                "mean_within_cell_std": float(std[n > 0].mean()),
                "global_std": float(cf.std()),
            }

        if not present:
            print(f"  {pairing:24s} no CF on disk for any config -- skipped")
            continue

        # ---- the served table ----
        out_csv = pdir / "map_table.csv"
        if out_csv.exists() and not args.overwrite:
            sys.exit(f"{out_csv} exists; pass --overwrite")
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(columns)
            w.writerows(rows)

        # Same content as a dense array, for T10's vectorised lookups: a
        # [n_configs, K] block per statistic, so predicting a batch of C
        # queries is map[cfg_idx, assign] with no dict or CSV in the loop.
        dense = {stat: np.full((len(present), K), np.nan, dtype=np.float32)
                 for stat in ("cf_mean", "cf_std")}
        dense["n"] = np.zeros((len(present), K), dtype=np.int64)
        for f in floors:
            dense[f"reliable_{f}"] = np.zeros((len(present), K), dtype=np.int8)
        # Column positions by name, not by hardcoded index: the reliable_* block
        # moved once already when the floors became configurable.
        col = {name: i for i, name in enumerate(columns)}
        cidx = {c: i for i, c in enumerate(present)}
        for r in rows:
            i, c = cidx[r[col["configuration"]]], r[col["landmark"]]
            dense["n"][i, c] = r[col["n"]]
            dense["cf_mean"][i, c] = r[col["cf_mean"]]
            dense["cf_std"][i, c] = r[col["cf_std"]]
            for f in floors:
                dense[f"reliable_{f}"][i, c] = r[col[f"reliable_{f}"]]
        np.savez(pdir / "cell_stats.npz",
                 configs=np.array(present), landmark=np.arange(K),
                 corpus_count=corpus, floors=np.array(floors),
                 primary_floor=np.array(args.floor), **dense)

        if args.cf_histogram:
            # The map as the draft defines it: per (cell, config), the mean
            # (cell_stats.npz) and this histogram, from which pi(tau) is read.
            np.savez(pdir / "cf_histogram.npz", configs=np.array(present),
                     bins=np.array(CF_HIST_BINS), bin_width=np.array(0.05),
                     counts=np.stack([cf_hists[c] for c in present]).astype(np.int32))

        write_json(pdir / "ga_fallback.json", {
            "note": "served for any cell with reliable=0, and H5's GA baseline",
            "pairing": pairing, "role": "B", "floor": args.floor, "ga": ga,
        })
        ga_all[pairing] = ga

        if args.histogram:
            write_histogram(pdir, assign, K, args.floor, floors, pairing)

        meta["pairings"][pairing] = {
            "query_modality": spec["query_modality"],
            "index_modality": spec["index_modality"],
            "same_modality": spec["same_modality"],
            "K": K, "rows": len(rows), "configs": present,
            "per_config": per_config,
        }

        cnt = np.bincount(assign, minlength=K)
        occ = int((cnt > 0).sum())
        by_floor = "  ".join(f"@{f}:{int((cnt >= f).sum())}" for f in floors)
        print(f"  {pairing:24s} {len(present)} cfg  {len(rows):>6,} rows  "
              f"cells {occ:>4}/{K} occupied   reliable {by_floor}")

    write_json(lay.dest / "map_meta.json", meta)
    write_json(lay.dest / "ga_fallback.json",
               {"note": "GA per (pairing, config)", "ga": ga_all})

    if meta["missing"]:
        print(f"\n  {len(meta['missing'])} (pairing, config) CF vectors absent -- "
              f"T7/T8 incomplete. Listed in map_meta.json.")
    print(f"\nwrote {lay.dest}")


def write_histogram(pdir: Path, assign: np.ndarray, K: int, floor: int,
                    floors, pairing: str) -> None:
    """
    Queries-per-cell histogram -- the second deliverable.

    Cell occupancy is heavy-tailed under the cross-modal pairings (the modality
    gap piles queries into a few cells), so the bins are the decision
    boundaries, not equal widths: what matters is how much of B sits below each
    candidate reliability floor, not the shape of the tail.

    Also reports the bottom-decile analysis the paper asks for by name (S6.4:
    "cells in the bottom decile falling below the floor are treated as an H4
    finding about landmark resolution, not silently averaged away"). Deciles
    are taken over OCCUPIED cells: with 286 empty cells in the image->text
    tiling, a decile over all K would be entirely empty cells and would say
    nothing about resolution.
    """
    counts = np.bincount(assign, minlength=K).astype(np.int64)
    edges = sorted({0, 1, 5, *floors, 100, 250, 500, 1000, 5000}) + [np.inf]
    total = int(counts.sum())
    with open(pdir / "cell_histogram.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bin_low", "bin_high", "n_cells", "frac_cells",
                    "n_queries", "frac_queries", "below_primary_floor"])
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (counts >= lo) & (counts < hi)
            nc, nq = int(sel.sum()), int(counts[sel].sum())
            w.writerow([int(lo), "inf" if hi == np.inf else int(hi), nc,
                        round(nc / K, 6), nq, round(nq / max(total, 1), 6),
                        int(hi <= floor)])

    occupied = counts[counts > 0]
    # The paper's unit of concern: the sparsest tenth of the cells that exist.
    decile_cut = float(np.percentile(occupied, 10)) if occupied.size else 0.0
    bottom = occupied[occupied <= decile_cut]

    per_floor = {}
    for f in sorted(floors):
        below = counts < f
        per_floor[str(f)] = {
            "cells_below_floor": int(below.sum()),
            "occupied_cells_below_floor": int(((counts > 0) & below).sum()),
            "b_queries_below_floor": int(counts[below].sum()),
            "frac_b_below_floor": float(counts[below].sum() / max(total, 1)),
            # The H4 finding, stated as the paper frames it.
            "bottom_decile_below_floor": int((bottom < f).sum()),
            "bottom_decile_frac_below_floor": (
                float((bottom < f).mean()) if bottom.size else 0.0),
        }

    write_json(pdir / "cell_histogram_meta.json", {
        "pairing": pairing, "K": K, "primary_floor": floor,
        "floors": sorted(floors), "n_b_queries": total,
        "occupied_cells": int((counts > 0).sum()),
        "empty_cells": int((counts == 0).sum()),
        "by_floor": per_floor,
        "bottom_decile": {
            "note": "deciles over OCCUPIED cells only; empty cells are a "
                    "separate tiling failure, not a resolution finding",
            "cutoff_count": decile_cut,
            "n_cells": int(bottom.size),
            "max_count": int(bottom.max()) if bottom.size else 0,
        },
        "deciles_of_occupied": {
            str(q): float(np.percentile(occupied, q)) if occupied.size else 0.0
            for q in range(10, 100, 10)
        },
        "quantiles": {q: float(np.percentile(counts, q))
                      for q in (1, 5, 25, 50, 75, 95, 99)},
        "max_cell": int(counts.max()),
        # Largest cell over the mean occupied cell: 1.0 is a perfectly even
        # tiling, and the cross-modal pairings run an order of magnitude above
        # it because the modality gap piles queries into a few cells.
        "imbalance_factor": float(counts.max() / max(occupied.mean(), 1e-9))
        if occupied.size else 0.0,
    })


if __name__ == "__main__":
    main()
