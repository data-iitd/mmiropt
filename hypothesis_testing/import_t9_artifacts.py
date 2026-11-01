"""
Copy the hypothesis-relevant artifacts T9 produced first into T10's tree.

T9 ran ahead of its remit: while building the map it also produced an H5
preview, an ICC analysis with per-configuration agreement heatmaps, and three
figures that belong to H1 and H5. Those are hypothesis results and the request
is that they live with the rest of them.

COPY, NOT MOVE
--------------
T9's ``result/spatial_map/1M/<model>/K01024_spherical/`` stays byte-identical.
Its README documents those files as part of its own output and its verifier
reads from the same tree; relocating them would leave both describing files
that are no longer there, and the upstream artifacts of a finished task are not
this task's to rearrange. The copies land under ``imported_from_t9/`` with a
manifest recording where each came from and whether T10 has since superseded
it.

WHAT IS SUPERSEDED, AND WHY THE COPY IS STILL WORTH KEEPING
-----------------------------------------------------------
``h5_preview.json``   -> superseded by H5. The preview has MAE, RMSE, Spearman
                         and threshold accuracy but no calibration curves and no
                         realized regret, and it scores only the config-keyed
                         map. Kept because it is the number that existed when
                         the map was built, and a later disagreement between the
                         two is worth being able to see.
``figures/h1_*.png``  -> superseded by H1's own binned profiles with trends.
``figures/baselines`` -> superseded by H5's skill ladder.
``figures/calibration``-> superseded by H5's calibration figure.
``icc_analysis/``     -> feeds H0 (does location carry signal) and H3 (do
                         configurations agree about geography). H0 and H3
                         recompute both on held-out C; T9's ran on B. Both are
                         informative and they are NOT interchangeable -- the B
                         version is in-sample and reads high by construction.
"""
from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

from ht_config import add_common_args, resolve_from_args, write_json

#: (relative path under T9's variant dir, note recorded in the manifest)
WANTED = [
    ("h5_preview.json",
     "T9's early H5; superseded by H5/h5_report.json, which adds calibration, "
     "realized regret, and the shared/family keying rungs"),
    ("verify_report.json",
     "T9's structural checks -- the provenance check is what licenses T10's "
     "positional joins"),
    ("map_meta.json",
     "what the map was built from; the provenance record for every T10 number"),
    ("ga_fallback.json", "GA per (pairing, config), as T9 computed it"),
    ("icc_analysis", "ICC and per-configuration agreement, computed on B; H0 "
                     "and H3 recompute both on held-out C"),
    ("figures/h1_error_vs_distance.png", "superseded by H1/figures/"),
    ("figures/map_image_query_image_index.png",
     "THE MAP coloured by cell mean CF -- where the index degrades. T10's "
     "map_error_/map_bias_ figures are its companions: where the map is wrong, "
     "and in which direction"),
    ("figures/heatmap_image_query_image_index.png",
     "T9's per-config cell profile and spatially-binned CF for this pairing"),
    ("figures/map_image_query_text_index.png",
     "THE MAP coloured by cell mean CF -- where the index degrades. T10's "
     "map_error_/map_bias_ figures are its companions: where the map is wrong, "
     "and in which direction"),
    ("figures/heatmap_image_query_text_index.png",
     "T9's per-config cell profile and spatially-binned CF for this pairing"),
    ("figures/map_text_query_image_index.png",
     "THE MAP coloured by cell mean CF -- where the index degrades. T10's "
     "map_error_/map_bias_ figures are its companions: where the map is wrong, "
     "and in which direction"),
    ("figures/heatmap_text_query_image_index.png",
     "T9's per-config cell profile and spatially-binned CF for this pairing"),
    ("figures/map_text_query_text_index.png",
     "THE MAP coloured by cell mean CF -- where the index degrades. T10's "
     "map_error_/map_bias_ figures are its companions: where the map is wrong, "
     "and in which direction"),
    ("figures/heatmap_text_query_text_index.png",
     "T9's per-config cell profile and spatially-binned CF for this pairing"),
    ("figures/baselines.png", "superseded by H5/figures/h5_skill_ladder.png"),
    ("figures/calibration.png", "superseded by H5/figures/h5_calibration.png"),
]


def run(args) -> dict:
    lay, ht, floor, pairings, configs, meta = resolve_from_args(args)
    src_root = lay.dest
    dest = ht.imported
    dest.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created": datetime.now().isoformat(),
        "policy": "copy; T9's tree is left byte-identical",
        "source_root": str(src_root),
        "model": lay.model, "tiling": lay.variant,
        "items": [],
    }
    for rel, note in WANTED:
        src = src_root / rel
        if not src.exists():
            manifest["items"].append({"path": rel, "status": "absent", "note": note})
            print(f"    absent  {rel}")
            continue
        dst = dest / Path(rel).name
        if dst.exists():
            shutil.rmtree(dst) if dst.is_dir() else dst.unlink()
        if src.is_dir():
            shutil.copytree(src, dst)
            n = sum(1 for _ in dst.rglob("*") if _.is_file())
            print(f"    copied  {rel}/  ({n} files)")
        else:
            shutil.copy2(src, dst)
            print(f"    copied  {rel}")
        manifest["items"].append({
            "path": rel, "status": "copied", "copied_to": str(dst.relative_to(dest)),
            "note": note})

    write_json(dest / "IMPORT_MANIFEST.json", manifest)
    print(f"\n  wrote {dest / 'IMPORT_MANIFEST.json'}")
    return manifest


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    args = ap.parse_args()
    print(f"import T9 hypothesis artifacts  model={args.model}  (copy, not move)")
    run(args)


if __name__ == "__main__":
    main()
