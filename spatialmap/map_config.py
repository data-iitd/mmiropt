"""
T9 shared vocabulary: how a (model, K, variant, pairing) resolves to files.

Nothing here computes. It is the single place that knows

  * which landmark set a pairing is keyed to,
  * where T5 put the tiling and the per-query assignments,
  * where T8 put the CF labels,
  * where T9 writes the served map,

so build_map.py, assign_queries.py and verify_map.py cannot drift apart on any
of it.

THE PAIRING RULE, stated once
-----------------------------
A pairing is written ``{q}_query_{i}_index``. The map is keyed to the landmarks
of the INDEX modality ``i``, never the query modality, because the tiling exists
to describe where the *index* degrades. The queries scored against it are in
modality ``q``. So

    text_query_image_index  ->  landmarks: image/K.../   queries: nn_{role}_textq.npz

T5's file naming already encodes exactly this: ``{lmod}/K.../queries/nn_{role}_{qmod}q.npz``
is "the {qmod} queries of {role}, assigned to the {lmod} landmarks". The four
files per role are the four pairings.

WHAT IS SQUARED AND WHAT IS NOT
-------------------------------
T5 stores ``d1``/``d2`` as SQUARED L2 (they come straight out of FAISS, which
does not take the root). Everything in T9 that reports a distance takes the root
first, and the column names say so (``d1``, ``d2``, ``gap``). H1/H2 in T10 want
ordinary distances; a squared axis would bend every error-vs-distance curve it
is asked to fit.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(REPO_ROOT / "indexes"))
sys.path.insert(0, str(REPO_ROOT / "knn"))
from index_configs import MODEL_DIMS, configs_for                      # noqa: E402

MODALITIES = ("image", "text")
ROLES = ("B", "C")

#: The model whose embedding space the 1M pilot was stratified in. Per the build
#: plan (S5, "Pilot subsample rule"), the A/B/C split is drawn ONCE in this
#: space and the same pairs fill the same role for every model -- the
#: shared-split control. So only this model has a pilot directory; the others
#: share it by design, not by omission, and role_ids() falls back to it.
#:
#: This holds for the 1M pilot only. At 7M the split is over the whole corpus
#: and each model's own directory is expected to exist, at which point the
#: fallback stops firing on its own.
REFERENCE_MODEL = "siglip2"

#: pairing name -> the two modalities it couples.
PAIRINGS: Dict[str, Dict[str, object]] = {
    f"{q}_query_{i}_index": {
        "query_modality": q,
        "index_modality": i,
        "short": f"{q[0]}->{i[0]}",
        "same_modality": q == i,
    }
    for i in MODALITIES for q in MODALITIES
}

DEFAULT_PILOT = REPO_ROOT / "result/pilot_1M"
DEFAULT_KMEANS = REPO_ROOT / "result/kmeans/1M"
DEFAULT_CF = REPO_ROOT / "result/cf/1M"
DEFAULT_OUT = REPO_ROOT / "result/spatial_map/1M"

# --------------------------------------------------------------------------
# The reliability floor, and why there is more than one
# --------------------------------------------------------------------------
# A cell built from too few B queries has a noisy mean. Two sources disagree on
# where "too few" is, so the map carries BOTH rather than silently picking one:
#
#   * The T9 dispatch says "fewer than about 10 B queries ... falls back to the
#     GA baseline's number for that config".
#   * The paper (MMIROptimizer draft, S6.4) sizes B so per-cell counts clear
#     "a floor of roughly 30-50 queries in the sparse cells -- not merely on
#     average", and says cells below it are "treated as an H4 finding about
#     landmark resolution, not silently averaged away". T5's occupancy_check
#     independently used 30, siding with the paper.
#
# These are not the same instruction. The dispatch asks for a repair (serve GA
# instead); the paper asks for a report (this is what landmark resolution costs
# you). Choosing 10 alone would hide how much of the tiling the paper would
# consider unusable -- at K=1024 the text->image pairing has 760 cells clearing
# 10 but only 454 clearing 30.
#
# So every floor gets its own column. T10 picks the policy; T9 does not decide
# it, and no rebuild is needed to change it.
RELIABILITY_FLOOR = 10                      #: the dispatch's floor; the default
PAPER_FLOORS = (30, 50)                     #: the draft's S6.4 range
RELIABILITY_FLOORS = (RELIABILITY_FLOOR,) + PAPER_FLOORS

#: Base column order of the served table. Frozen -- T10 reads it positionally
#: in places, and a reordering here would silently transpose two float columns.
#: The reliable_* columns are appended by map_columns(), one per floor.
MAP_COLUMNS = [
    "landmark", "configuration", "modality_configuration",
    "n", "cf_mean", "cf_std", "cf_sem", "cf_min", "cf_max",
    "d1_mean", "gap_mean", "corpus_count",
]


def map_columns(floors=RELIABILITY_FLOORS) -> List[str]:
    """Served-table header for a given set of floors: base columns, then one
    reliable_<floor> flag per floor, in ascending floor order."""
    return MAP_COLUMNS + [f"reliable_{f}" for f in sorted(floors)]


@dataclass(frozen=True)
class Layout:
    """Every path one (model, K, variant) run needs."""
    model: str
    k: int
    spherical: bool
    pilot: Path
    kmeans: Path
    cf: Path
    out: Path

    # ---- naming -------------------------------------------------------
    @property
    def variant(self) -> str:
        return f"K{self.k:05d}{'_spherical' if self.spherical else ''}"

    @property
    def dest(self) -> Path:
        """result/spatial_map/<scale>/<model>/K01024_spherical/"""
        return self.out / self.model / self.variant

    @property
    def scale(self) -> str:
        """
        The artifact-tree tag: '1M', '7M_cleaned'.

        Taken from the CF directory's own name so that pointing --cf-dir at
        another tree moves every sibling lookup with it. Sibling stages
        (result/knn/<scale>/, result/approx_knn/<scale>/) are addressed through
        this rather than through a hardcoded literal, which is how the
        provenance check silently reported 'not yet run' for the whole 7M tree
        before this existed -- passing verification without ever testing the
        alignment anchor it exists to test.
        """
        return self.cf.name

    # ---- upstream -----------------------------------------------------
    def role_ids(self, role: str) -> Path:
        """This model's split if it has one, else the shared reference split."""
        own = self.pilot / self.model / f"{role}_ids.npy"
        return own if own.exists() else (
            self.pilot / REFERENCE_MODEL / f"{role}_ids.npy")

    @property
    def uses_shared_split(self) -> bool:
        """True when this model is reading REFERENCE_MODEL's A/B/C split.

        Surfaced rather than hidden: it is correct at 1M and wrong at 7M, so
        every artifact records which split it was built against.
        """
        return not (self.pilot / self.model / "B_ids.npy").exists()

    def tiling(self, landmark_modality: str) -> Path:
        return self.kmeans / self.model / landmark_modality / self.variant

    def centroids(self, landmark_modality: str) -> Path:
        return self.tiling(landmark_modality) / "centroids.npy"

    def cell_counts(self, landmark_modality: str) -> Path:
        """T5's corpus density per cell -- the DO baseline's input, carried
        into the served table so H5 does not have to re-open T5."""
        return self.tiling(landmark_modality) / "cell_counts_A.npy"

    def query_nn(self, landmark_modality: str, role: str, query_modality: str) -> Path:
        """T5's precomputed (assign, d1, d2) for one role x query modality."""
        return (self.tiling(landmark_modality) / "queries"
                / f"nn_{role}_{query_modality}q.npz")

    def cf_dir(self) -> Path:
        return self.cf / self.model

    def cf_vector(self, pairing: str, role: str, config: str) -> Path:
        """T8 writes one .npy per (pairing, role, config), aligned to role order."""
        return self.cf_dir() / f"cf_{pairing}_{role}_{config}.npy"

    def cf_table(self) -> Path:
        return self.cf_dir() / "cf_table.csv"

    # ---- T9 output ----------------------------------------------------
    def pairing_dir(self, pairing: str) -> Path:
        return self.dest / pairing

    def geometry(self, role: str, pairing: str) -> Path:
        """Per-query landmark geometry for H1/H2 -- assign, d1, d2, gap."""
        return self.dest / "geometry" / f"geom_{role}_{pairing}.npz"


def layout(model: str, k: int, spherical: bool, *,
           pilot: Optional[Path] = None, kmeans: Optional[Path] = None,
           cf: Optional[Path] = None, out: Optional[Path] = None) -> Layout:
    if model not in MODEL_DIMS:
        raise KeyError(f"unknown model {model!r}; known: {sorted(MODEL_DIMS)}")
    return Layout(
        model=model, k=int(k), spherical=bool(spherical),
        pilot=Path(pilot) if pilot else DEFAULT_PILOT,
        kmeans=Path(kmeans) if kmeans else DEFAULT_KMEANS,
        cf=Path(cf) if cf else DEFAULT_CF,
        out=Path(out) if out else DEFAULT_OUT,
    )


def resolve_pairings(names: List[str]) -> List[str]:
    """'all' -> every pairing; otherwise validate what was asked for."""
    if not names or names == ["all"]:
        return sorted(PAIRINGS)
    bad = [n for n in names if n not in PAIRINGS]
    if bad:
        raise SystemExit(f"unknown pairing(s) {bad}; known: {sorted(PAIRINGS)}")
    return list(dict.fromkeys(names))


# --------------------------------------------------------------------------
# The map's CF histogram (draft main.tex, Step 4, l.303-311 and Output l.323)
# --------------------------------------------------------------------------
#: "B = 20 bins of width 0.05 over the fidelity values 0, 0.01, ..., 1, the last
#: bin closed": bin b holds CF in [0.05 b, 0.05 (b+1)) for b < 19, and [0.95, 1].
CF_HIST_BINS = 20
CF_STEP = 0.01          # CF@100 takes the values j/100


def cf_histogram(assign: np.ndarray, cf: np.ndarray, K: int) -> np.ndarray:
    """[K, 20] int64 counts of each cell's CF values per bin."""
    h = np.rint(np.asarray(cf, dtype=np.float64) / CF_STEP).astype(np.int64)
    if h.min() < 0 or h.max() > 100:
        raise ValueError(f"CF outside [0, 1]: {h.min() * CF_STEP}, {h.max() * CF_STEP}")
    b = np.minimum(h // 5, CF_HIST_BINS - 1)
    flat = np.bincount(np.asarray(assign, dtype=np.int64) * CF_HIST_BINS + b,
                       minlength=K * CF_HIST_BINS)
    return flat.reshape(K, CF_HIST_BINS)


def coverage_from_histogram(hist: np.ndarray, tau: float) -> np.ndarray:
    """pi(tau) per cell read off the histogram, as the draft defines it: whole
    bins at or above tau count fully; for a tau INSIDE a bin, the bin's count
    is divided evenly among the fidelity values it holds. Exact for a tau that
    is a bin edge (0.7, 0.8, 0.9). Empty cells get 0."""
    t = int(round(tau / CF_STEP))
    first_full = -(-t // 5)                       # ceil(t / 5): bins entirely >= tau
    above = hist[:, first_full:].sum(axis=1).astype(np.float64)
    if t % 5:
        b = t // 5
        lo, hi = 5 * b, (5 * b + 4 if b < CF_HIST_BINS - 1 else 100)
        n_vals = hi - lo + 1
        above += hist[:, b] * (hi - t + 1) / n_vals
    n = hist.sum(axis=1)
    out = np.zeros(hist.shape[0])
    nz = n > 0
    out[nz] = above[nz] / n[nz]
    return out


def resolve_configs(model: str, spec: str) -> List[str]:
    """'all' -> the model's whole index catalogue, in catalogue order."""
    known = [c.name for c in configs_for(model)]
    if spec in ("all", "", None):
        return known
    want = [s.strip() for s in spec.split(",") if s.strip()]
    bad = [w for w in want if w not in known]
    if bad:
        raise SystemExit(f"unknown config(s) {bad}; known: {known}")
    return [c for c in known if c in want]      # catalogue order, not user order


def config_family(model: str, name: str) -> str:
    for c in configs_for(model):
        if c.name == name:
            return c.family
    return "?"


def load_query_geometry(lay: Layout, pairing: str, role: str) -> Dict[str, np.ndarray]:
    """
    T5's assignment for one (pairing, role), with distances de-squared.

    Returns assign (int32), d1, d2, gap (float32, ordinary L2). Rows are in
    ``{role}_ids.npy`` order -- the invariant every downstream join relies on.
    """
    spec = PAIRINGS[pairing]
    path = lay.query_nn(spec["index_modality"], role, spec["query_modality"])
    if not path.exists():
        raise FileNotFoundError(
            f"missing T5 assignment {path}\n"
            f"  run: landmarks/occupancy_check.py --model {lay.model} "
            f"--k {lay.k}{' --spherical' if lay.spherical else ''}")
    z = np.load(path)
    # sqrt on a clipped floor: FAISS can return a tiny negative for a point
    # sitting exactly on its centroid, and sqrt of that is nan, which would
    # then poison every mean it touches.
    d1 = np.sqrt(np.maximum(z["d1"].astype(np.float64), 0.0)).astype(np.float32)
    d2 = np.sqrt(np.maximum(z["d2"].astype(np.float64), 0.0)).astype(np.float32)
    return {
        "assign": z["assign"].astype(np.int32),
        "d1": d1,
        "d2": d2,
        # Boundary proximity, H2's splitting variable. Small gap = the query sits
        # near the wall between two cells, so its cell's mean describes it worst.
        "gap": (d2 - d1).astype(np.float32),
    }


def load_role_ids(lay: Layout, role: str) -> np.ndarray:
    p = lay.role_ids(role)
    if not p.exists():
        raise FileNotFoundError(
            f"missing {p} -- T3 has not been run for {lay.model}, and the "
            f"shared {REFERENCE_MODEL} split is not there either")
    return np.load(p)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
