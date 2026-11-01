"""
T9 verification -- checks the built map is what T10 is entitled to assume.

    python verify_map.py --model siglip2 --spherical

Every check here targets a failure that produces NO error at build time and a
plausible-looking wrong number downstream. In rough order of how quietly they
would corrupt T10:

  1. B/C DISJOINTNESS. If any C query id appears in B, the map was calibrated
     on part of its own test set and every H1-H5 metric is optimistic. This is
     the one that cannot be recovered from after the fact.

  2. CELL COUNTS RECONCILE. Sum of n over a config's cells must equal the
     number of B queries scored for it. A mismatch means queries were dropped
     or double-counted by the join -- the row-alignment invariant failing
     silently, which is exactly how a whole-array off-by-one hides.

  3. MEANS RECONCILE. The count-weighted mean of the cell means must equal the
     GA number for that config, to float tolerance. This catches the map having
     been grouped from a DIFFERENT vector than the one GA was taken from -- a
     wrong config or wrong pairing wired into the join.

     It does NOT catch a permutation. Both sides are computed from the same
     array, and a sum is invariant to reordering, so a CF vector shifted by one
     row against its assignment reconciles perfectly. Check 3b exists because
     of that: it was found by deliberately shifting a vector by one row and
     watching every other check pass.

  3b. VARIANCE RATIO and EXPLAINED VARIANCE (diagnostic, not pass/fail). Mean
     within-cell std over between-cell std, reported alongside

         R^2 = between_var / (between_var + within_var) = 1 / (1 + ratio^2)

     which is the interpretable half: the share of CF variance the cell means
     account for, i.e. how much the map buys over GA (which is R^2 = 0 by
     construction, predicting one number everywhere).

     A misaligned join destroys the grouping and sends the ratio far up -- a
     deliberate one-row shift measured 10.07 (R^2 = 0.01) against 0.23 aligned
     on synthetic data. But real CF sits in a much less extreme band: the
     SigLIP 2 map runs 0.30-2.18 (R^2 0.92-0.17) across the 28 blocks, so a
     ratio near 1 is an ordinary, useful map (R^2 = 0.50), NOT a warning.

     The threshold is therefore set at 4 (R^2 < 0.06), which is the
     misalignment regime rather than the weak-signal one. An earlier version
     warned above 1 and fired on 15 of 28 real blocks, which is noise.

     It stays a diagnostic either way, because a genuine failure of P1 and a
     misalignment both raise it, and telling them apart is H1's job. The hard
     guard is check 7, which pins provenance at the source instead of inferring
     it from the values.

  4. CONFIG KEYING IS LOAD-BEARING (P2 / H3). If per-config cell means were
     identical across configs, keying the map by (landmark, config) would be
     pointless and a landmark-only map would do. Reports the spread so H3 has
     a prior rather than starting from zero.

  5. GEOMETRY SANITY. d2 >= d1 everywhere, no negative gaps, assignments in
     [0, K).

  6. FLOOR ACCOUNTING. How much of B, and of C at serve time, lands in a cell
     that backs off to GA.

  7. PROVENANCE. T6's and T7's stored query-id vectors must equal the role's
     ids in order. This is the anchor the whole positional join rests on: T8's
     CF .npy files carry no ids of their own, so if the qids they were derived
     from already disagreed with the split, every row of the map would be built
     from the wrong query with nothing in the values to reveal it.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from map_config import (                                               # noqa: E402
    DEFAULT_CF, DEFAULT_KMEANS, DEFAULT_OUT, DEFAULT_PILOT, REFERENCE_MODEL,
    REPO_ROOT, layout, load_query_geometry, load_role_ids, resolve_pairings,
    write_json,
)

TOL = 1e-4

#: Variance-ratio above which the grouping looks broken rather than merely
#: weak. R^2 < 0.06 at this point. Set from measured data: real SigLIP 2 blocks
#: run 0.30-2.18, a deliberately misaligned one measured 10.07.
MISALIGNMENT_RATIO = 4.0


def check_provenance(lay, pairing: str, b_ids: np.ndarray, check, results_root=None) -> None:
    """
    Pin the positional join at its source.

    T8's cf_*.npy carry no query ids, so their row i means "B_ids[i]" only
    because T6 and T7 were themselves in role order. Those two DO store their
    qids. If either disagrees with the split, the map is built from the wrong
    queries and nothing in the CF values can show it.

    A short qids vector is a --limit-q partial run and is checked as a prefix.
    Missing files are reported, not failed: T7/T8 are still in flight.
    """
    # From REPO_ROOT, not from lay.cf: --cf-dir is routinely pointed at a
    # scratch copy, and deriving result/ from it would silently look for T6/T7
    # beside the scratch CF and report them "not yet run".
    # Note the two layouts genuinely differ: T6 is <pairing>/<model>, T7 is
    # <model>/<pairing>.
    # --results-root picks the run's tree (v2: result_v2); the default is v1's.
    root = Path(results_root) if results_root else REPO_ROOT / "result"
    scale = lay.scale
    sources = {
        "T6 exact": root / "knn" / scale / pairing / lay.model / "knn_B_qids.npy",
        "T7 approx": root / "approx_knn" / scale / lay.model / pairing
                     / "approx_B_qids.npy",
    }
    for label, path in sources.items():
        if not path.exists():
            # Distinguish "that stage has not run" from "this whole scale tree
            # is not where I looked". The second silently disabled the check
            # for every 7M pairing when the tag was hardcoded to 1M.
            stage_root = path.parents[2] if label == "T6 exact" else path.parents[2]
            why = ("not yet run" if stage_root.exists()
                   else f"no such tree {stage_root} -- is --cf-dir's scale tag "
                        f"({scale!r}) right?")
            print(f"    [ -- ] {label} qids absent ({path.name}) -- {why}")
            continue
        q = np.load(path)
        ok = len(q) <= len(b_ids) and np.array_equal(q, b_ids[:len(q)])
        check(f"{pairing} {label} qids are B in order", ok,
              f"{len(q):,} of {len(b_ids):,}"
              + ("" if len(q) == len(b_ids) else " (partial run)")
              if ok else "qids do NOT match the split's B order")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="siglip2")
    ap.add_argument("--k", type=int, default=1024)
    ap.add_argument("--spherical", action="store_true")
    ap.add_argument("--pairing", nargs="+", default=["all"])
    ap.add_argument("--pilot-dir", type=Path, default=DEFAULT_PILOT)
    ap.add_argument("--kmeans-dir", type=Path, default=DEFAULT_KMEANS)
    ap.add_argument("--cf-dir", type=Path, default=DEFAULT_CF)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if any check fails")
    ap.add_argument("--results-root", type=Path, default=None,
                    help="tree holding knn/ and approx_knn/ for the provenance check "
                         "(default: result/, the v1 run; v2 passes result_v2)")
    args = ap.parse_args()

    lay = layout(args.model, args.k, args.spherical, pilot=args.pilot_dir,
                 kmeans=args.kmeans_dir, cf=args.cf_dir, out=args.out_dir)
    meta_path = lay.dest / "map_meta.json"
    if not meta_path.exists():
        sys.exit(f"no map at {meta_path} -- run build_map.py first")
    meta = json.load(open(meta_path))
    floor = int(meta["reliability_floor"])
    floors = [int(f) for f in meta.get("reliability_floors", [floor])]
    pairings = [p for p in resolve_pairings(args.pairing) if p in meta["pairings"]]

    report = {"created": datetime.now().isoformat(), "model": lay.model,
              "K": lay.k, "spherical": lay.spherical, "floor": floor,
              "checks": {}, "pairings": {}}
    failures = []

    def check(name, ok, detail=""):
        report["checks"][name] = {"pass": bool(ok), "detail": detail}
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
              f"{'  -- ' + detail if detail else ''}")
        if not ok:
            failures.append(name)

    print(f"T9 verification  model={lay.model}  tiling={lay.variant}\n")

    # ---- 1. the leakage check -------------------------------------------
    b_ids, c_ids = load_role_ids(lay, "B"), load_role_ids(lay, "C")
    overlap = np.intersect1d(b_ids, c_ids)
    check("B and C are disjoint", overlap.size == 0,
          f"{overlap.size:,} shared ids" if overlap.size else
          f"|B|={len(b_ids):,} |C|={len(c_ids):,}")
    check("B ids are unique", len(np.unique(b_ids)) == len(b_ids))

    # The 1M pilot draws one split in the reference model's space and reuses it
    # everywhere. Reusing it is correct; reusing it SILENTLY is not, because the
    # same fallback would be wrong at 7M, where each model gets its own split.
    report["uses_shared_split"] = lay.uses_shared_split
    report["split_source"] = str(lay.role_ids("B").parent)
    if lay.uses_shared_split:
        print(f"    [INFO] {lay.model} reads the shared {REFERENCE_MODEL} A/B/C "
              f"split (1M pilot control). Its own length is checked against "
              f"this model's T5 geometry below.")

    # ---- per pairing -----------------------------------------------------
    for pairing in pairings:
        print(f"\n  {pairing}")
        pdir = lay.pairing_dir(pairing)
        pm = meta["pairings"][pairing]
        K = int(pm["K"])

        ga = json.load(open(pdir / "ga_fallback.json"))["ga"]
        rows = list(csv.DictReader(open(pdir / "map_table.csv")))

        by_cfg = defaultdict(list)
        for r in rows:
            by_cfg[r["configuration"]].append(r)

        geom_b = load_query_geometry(lay, pairing, "B")
        geom_c = load_query_geometry(lay, pairing, "C")

        # The shared split's guard: T5 assigned this model's OWN B/C vectors, so
        # if the split being read describes a different set of pairs the two
        # lengths disagree. With the fallback in play this is the check standing
        # between "reused by design" and "reused by accident".
        check(f"{pairing} split matches this model's T5 geometry",
              len(geom_b["assign"]) == len(b_ids)
              and len(geom_c["assign"]) == len(c_ids),
              f"B {len(geom_b['assign']):,}/{len(b_ids):,}, "
              f"C {len(geom_c['assign']):,}/{len(c_ids):,}")

        # ---- 7. provenance ------------------------------------------------
        check_provenance(lay, pairing, b_ids, check, args.results_root)

        # ---- 5. geometry -------------------------------------------------
        for role, g in (("B", geom_b), ("C", geom_c)):
            bad_gap = int((g["gap"] < -1e-5).sum())
            bad_asn = int(((g["assign"] < 0) | (g["assign"] >= K)).sum())
            check(f"{pairing}/{role} geometry well-formed",
                  bad_gap == 0 and bad_asn == 0,
                  f"{bad_gap} negative gaps, {bad_asn} out-of-range assignments")

        pres = {}
        for cfg, crows in sorted(by_cfg.items()):
            n = np.array([int(r["n"]) for r in crows], dtype=np.int64)
            mu = np.array([float(r["cf_mean"]) for r in crows])
            rel = np.array([int(r[f"reliable_{floor}"]) for r in crows], dtype=bool)

            expected_n = int(pm["per_config"][cfg]["n_b_queries"])

            # ---- 2. counts reconcile -------------------------------------
            ok_n = int(n.sum()) == expected_n
            # ---- 3. means reconcile --------------------------------------
            recon = float((n * mu).sum() / max(n.sum(), 1))
            ok_mu = abs(recon - float(ga[cfg])) < TOL
            # ---- 6. floor accounting -------------------------------------
            backed = float(n[~rel].sum() / max(n.sum(), 1))

            if not ok_n:
                check(f"{pairing}/{cfg} counts reconcile", False,
                      f"cells sum to {n.sum():,}, expected {expected_n:,}")
            if not ok_mu:
                check(f"{pairing}/{cfg} means reconcile", False,
                      f"weighted cell mean {recon:.6f} vs GA {ga[cfg]:.6f}")

            # ---- 3b. variance ratio (diagnostic) --------------------------
            sd = np.array([float(r["cf_std"]) for r in crows])
            grand = float((n * mu).sum() / max(n.sum(), 1))
            within = float((n * sd).sum() / max(n.sum(), 1))
            between = float(np.sqrt((n * (mu - grand) ** 2).sum() / max(n.sum(), 1)))
            ratio = within / between if between > 1e-9 else float("inf")
            # Share of CF variance the cell means account for. GA is 0 here by
            # construction, so this is directly "what the map buys over GA" --
            # on B, in sample; C will be lower and T10 measures it properly.
            r2 = 1.0 / (1.0 + ratio * ratio) if ratio != float("inf") else 0.0

            pres[cfg] = {
                "cells": len(crows), "reliable": int(rel.sum()),
                "within_cell_std": within, "between_cell_std": between,
                "variance_ratio": ratio,
                "explained_variance_B": r2,
                "ga": float(ga[cfg]), "reconstructed_ga": recon,
                "counts_reconcile": ok_n, "means_reconcile": ok_mu,
                "frac_b_backed_off": backed,
                "cell_mean_min": float(mu.min()), "cell_mean_max": float(mu.max()),
                "cell_mean_spread": float(mu.max() - mu.min()),
            }

        # Reported per pairing rather than per config: a single config above 1
        # is a finding about that index, all of them above 1 is a smell about
        # the join.
        ratios = {c: v["variance_ratio"] for c, v in pres.items()}
        r2s = {c: v["explained_variance_B"] for c, v in pres.items()}
        worst = max(ratios.values()) if ratios else 0.0
        print("    R^2 on B (share of CF variance the map explains; GA = 0): "
              + "  ".join(f"{c}={r2s[c]:.2f}" for c in sorted(r2s)))
        if worst > MISALIGNMENT_RATIO:
            n_bad = sum(r > MISALIGNMENT_RATIO for r in ratios.values())
            print(f"    NOTE: variance ratio > {MISALIGNMENT_RATIO} for {n_bad} "
                  f"config(s) (R^2 < {1 / (1 + MISALIGNMENT_RATIO ** 2):.2f}). "
                  f"The cell means explain almost nothing. Either P1 fails "
                  f"badly here, or the CF vector is misaligned to the "
                  f"assignment -- check 7 distinguishes them.")

        if all(v["counts_reconcile"] for v in pres.values()):
            check(f"{pairing} all counts reconcile", True,
                  f"{len(pres)} configs")
        if all(v["means_reconcile"] for v in pres.values()):
            check(f"{pairing} all means reconcile", True,
                  f"max |delta| < {TOL}")

        # ---- 4. is per-config keying load-bearing? ------------------------
        # Per cell, the spread of the cell's mean ACROSS configs. If P2 holds,
        # this is large: the same neighbourhood is easy for one index family
        # and hard for another. If it were ~0, H3 would already be answered no.
        common = sorted(by_cfg)
        if len(common) > 1:
            mat = np.full((len(common), K), np.nan)
            for i, cfg in enumerate(common):
                for r in by_cfg[cfg]:
                    mat[i, int(r["landmark"])] = float(r["cf_mean"])
            # Cells empty under every config are all-NaN columns; drop them
            # rather than letting nanmax warn and return NaN for each.
            live = ~np.isnan(mat).all(axis=0)
            per_cell_spread = (np.nanmax(mat[:, live], axis=0)
                               - np.nanmin(mat[:, live], axis=0))
            spread = float(per_cell_spread.mean()) if live.any() else 0.0
            check(f"{pairing} per-config keying carries signal", spread > 0.01,
                  f"mean across-config spread within a cell = {spread:.4f}")
            report.setdefault("h3_prior", {})[pairing] = spread

        # ---- serve-time floor cost on the real test set -------------------
        # A cell clearing a higher floor must clear every lower one. This is
        # arithmetically forced, so a violation means the columns were written
        # out of order or misaligned -- cheap to check, and it would otherwise
        # only show up as T10 reading the wrong policy's flags.
        mono_ok = True
        for r in rows:
            flags = [int(r[f"reliable_{f}"]) for f in sorted(floors)]
            if any(a < b for a, b in zip(flags, flags[1:])):
                mono_ok = False
                break
        check(f"{pairing} reliable_* flags are monotone in the floor", mono_ok,
              f"floors {sorted(floors)}")

        # What each candidate policy costs at serve time, on the real test set.
        cnt_b = np.bincount(geom_b["assign"], minlength=K)
        serve = {}
        for f in sorted(floors):
            unrel = cnt_b < f
            serve[str(f)] = {
                "cells_below_floor": int(unrel.sum()),
                "frac_C_served_by_GA": float(unrel[geom_c["assign"]].mean()),
                "frac_B_below_floor": float(
                    cnt_b[unrel].sum() / max(cnt_b.sum(), 1)),
            }
        report["pairings"][pairing] = {"configs": pres, "by_floor": serve}
        for f in sorted(floors):
            s = serve[str(f)]
            print(f"    floor {f:>3}: {s['cells_below_floor']:>4}/{K} cells below, "
                  f"{s['frac_C_served_by_GA']:6.2%} of C backs off to GA")

    report["failures"] = failures
    report["verdict"] = "PASS" if not failures else "FAIL"
    write_json(lay.dest / "verify_report.json", report)
    print(f"\n  verdict: {report['verdict']}"
          f"{'  (' + ', '.join(failures) + ')' if failures else ''}")
    print(f"  wrote {lay.dest / 'verify_report.json'}")

    if failures and args.strict:
        sys.exit(1)


if __name__ == "__main__":
    main()
