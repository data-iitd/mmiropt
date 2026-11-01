"""
Run the whole of T10 for one model, in dependency order, and assemble the report.

    python hypothesis_testing/run_all.py --model siglip2 --spherical

Order matters only in that ``report.py`` reads what the others wrote. H0, H1,
H2, H3, H4 and H5 are independent of each other.

WHAT EACH STEP COSTS
--------------------
Every step reads a few MB -- T9's map, T5's assignments, T8's CF vectors -- and
is arithmetic over ~50,000 held-out queries per pairing. None of it touches the
corpus, an index, or a GPU.

H4's landmark ladder -- a fresh k-means over A at each K -- is NOT run here.
It belongs to T5 and needs the 3.65 GiB corpus read, so run

    landmarks/build_landmarks.py --model <m> --modality both --spherical --k ...
    landmarks/occupancy_check.py --model <m> --spherical --k ...

first. H4 evaluates whichever K have a real tiling on disk and names the rest
as absent; it never substitutes anything for a missing K.

``--skip`` drops steps by name, which is what to use when re-running one
hypothesis after changing its threshold rather than recomputing the lot.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import h0_predictability                                               # noqa: E402
import h1_error_vs_distance                                            # noqa: E402
import h2_boundary                                                     # noqa: E402
import h3_keying                                                       # noqa: E402
import h4_resolution                                                   # noqa: E402
import h5_baselines                                                    # noqa: E402
import map_figures                                                     # noqa: E402
import import_t9_artifacts                                             # noqa: E402
import report as report_mod                                            # noqa: E402
from ht_config import add_common_args, ht_layout, write_json           # noqa: E402

STEPS = ("import", "H0", "H1", "maps", "H2", "H3", "H4", "H5", "report")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--sweep", nargs="+", type=int,
                    default=[64, 128, 256, 512, 1024, 2048, 4096])
    ap.add_argument("--skip", nargs="+", default=[],
                    help=f"steps to skip; any of {STEPS}")
    ap.add_argument("--keep-going", action="store_true",
                    help="continue after a step raises, and record the "
                         "traceback rather than stopping")
    args = ap.parse_args()
    args.check_1024 = True

    plan = [s for s in STEPS if s not in args.skip]
    print("=" * 78)
    print(f"T10 hypothesis testing   model={args.model}   "
          f"tiling=K{args.k:05d}{'_spherical' if args.spherical else ''}")
    print(f"steps: {' -> '.join(plan)}")
    print("=" * 78)

    runners = {
        "import": import_t9_artifacts.run,
        "H0": h0_predictability.run,
        "H1": h1_error_vs_distance.run,
        "maps": map_figures.run,
        "H2": h2_boundary.run,
        "H3": h3_keying.run,
        "H4": h4_resolution.run,
        "H5": h5_baselines.run,
        "report": report_mod.run,
    }

    log = {"created": datetime.now().isoformat(), "model": args.model,
           "steps": {}}
    failed = []
    for step in plan:
        print(f"\n{'-' * 78}\n[{step}]\n{'-' * 78}")
        t0 = datetime.now()
        try:
            runners[step](args)
            log["steps"][step] = {
                "status": "ok",
                "seconds": (datetime.now() - t0).total_seconds()}
        except Exception:                                  # noqa: BLE001
            tb = traceback.format_exc()
            log["steps"][step] = {"status": "failed", "traceback": tb}
            failed.append(step)
            print(tb)
            if not args.keep_going:
                print(f"\nstopped at [{step}]. Re-run with --keep-going to "
                      f"continue past failures.")
                break

    ht = ht_layout(args.model, args.out_dir)
    ht.dest.mkdir(parents=True, exist_ok=True)
    log["failed"] = failed
    write_json(ht.dest / "run_log.json", log)

    print("\n" + "=" * 78)
    if failed:
        print(f"FAILED steps: {', '.join(failed)}  (see run_log.json)")
    print(f"artifacts under {ht.dest}")
    print("=" * 78)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
