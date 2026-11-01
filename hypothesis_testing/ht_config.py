"""
T10 shared vocabulary: how a (model, K, variant, pairing) resolves to the
labels, geometry, map and costs the H0-H5 tests need.

Nothing here computes a hypothesis. It is the single place that knows

  * where T9 put the served map and the per-query landmark geometry,
  * where T8 put the CF labels for the held-out role C,
  * where T7 recorded each configuration's MEASURED query throughput,
  * where T10 writes its own artifacts,

so h0..h5 cannot drift apart on any of it.

WHAT IS REUSED AND WHAT IS NEW
------------------------------
Paths, the pairing rule and the de-squaring of T5's distances all come from
``spatialmap/map_config.py`` -- imported, not re-implemented, because T9's map
was built under those definitions and grading it under different ones would be
a silent mismatch. What is new here is the *cost model* (T7's measured
queries-per-second, which H5's regret metric needs and T9 never had a use for)
and the output layout below.

THE ONE INVARIANT EVERY TEST RESTS ON
-------------------------------------
Row i of ``cf_<pairing>_C_<config>.npy`` and row i of ``geom_C_<pairing>.npz``
are the same query, in ``C_ids.npy`` order. T8 and T5 both write in role order
and neither stores ids alongside the values, so the join is positional. T9's
``verify_map.py`` provenance check pins this at the source; ``load_block``
re-asserts the lengths agree before anything is computed from them.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "spatialmap"))
sys.path.insert(0, str(REPO_ROOT / "indexes"))

from map_config import (                                               # noqa: E402
    DEFAULT_CF, DEFAULT_KMEANS, DEFAULT_OUT, DEFAULT_PILOT, PAIRINGS,
    Layout, config_family, layout, load_query_geometry, load_role_ids,
    resolve_configs, resolve_pairings, write_json,
)

#: Fidelity floors swept by H5. The paper (S5.7) declines to fix one, because
#: the cost model that would imply it is deferred to the parent work.
#: Swept 0.70 -> 0.95 in steps of 0.05. Written as literals rather than
#: np.arange because each tau is stringified into a dict key downstream, and
#: arange's float drift would produce keys like '0.7500000000000001'.
TAUS = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95)

#: T7's approximate-KNN metadata, which carries the measured throughput.
DEFAULT_APPROX = REPO_ROOT / "result/approx_knn/1M"

#: Where load_costs() looks, for the current run. Set by resolve_from_args from
#: --approx-dir; the module default is the pilot's tree.
#:
#: This is a module global rather than a parameter threaded through every call
#: site because the cost lookup happens inside load_block(), which every h*.py
#: calls without knowing anything about argv. The failure it prevents is silent:
#: at 7M the pilot's throughput numbers exist and load cleanly, so a run that
#: forgot to redirect would produce a complete, plausible, wrong cost model for
#: H5's regret rather than an error.
_APPROX_DIR = DEFAULT_APPROX

#: When set (by --cost-dir), load_costs() reads the controlled throughput
#: benchmark (knn/bench_throughput.py) instead of T7's metadata. T7's timings
#: were taken while T7 did its real work, on shared nodes, with a cell's nine
#: configurations spread across several jobs; the benchmark times them in one
#: job on one exclusive node on one fixed query sample. Same global pattern as
#: _APPROX_DIR, for the same reason: load_block() cannot see argv.
_COST_DIR: Optional[Path] = None

#: T10's own output root. One subdirectory per model, then one per hypothesis.
DEFAULT_HT_OUT = REPO_ROOT / "result/hypothesis_testing"

HYPOTHESES = ("H0", "H1", "H2", "H3", "H4", "H5")


# ==========================================================================
# Output layout
# ==========================================================================
@dataclass(frozen=True)
class HTLayout:
    """Where one model's hypothesis-test artifacts live.

    Deliberately keyed by model and not by tiling: H4 sweeps K, so K cannot be
    part of the path or the sweep would scatter across sibling directories.
    Each test records the tiling it used in its own meta.
    """
    model: str
    root: Path

    @property
    def dest(self) -> Path:
        """result/hypothesis_testing/<model>/"""
        return self.root / self.model

    def h(self, name: str) -> Path:
        if name not in HYPOTHESES:
            raise KeyError(f"unknown hypothesis {name!r}; known: {HYPOTHESES}")
        return self.dest / name

    def figures(self, name: str) -> Path:
        return self.h(name) / "figures"

    @property
    def imported(self) -> Path:
        """Copies of the hypothesis-relevant artifacts T9 produced first."""
        return self.dest / "imported_from_t9"

    def ensure(self, name: str) -> Path:
        d = self.h(name)
        (d / "figures").mkdir(parents=True, exist_ok=True)
        return d


def ht_layout(model: str, out: Optional[Path] = None) -> HTLayout:
    return HTLayout(model=model, root=Path(out) if out else DEFAULT_HT_OUT)


# ==========================================================================
# The cost model -- measured, not assumed
# ==========================================================================
def cost_basis_note() -> str:
    """What the current run's cost ordering is built from, for the reports."""
    if _COST_DIR is not None:
        return (f"controlled throughput benchmark ({_COST_DIR}/<model>/"
                f"throughput.json): all configurations of a model timed in one "
                f"job on one exclusive node, on one fixed role-C query sample, "
                f"at one thread count; 'cheapest' means lowest measured search "
                f"time per query")
    return (f"T7's measured queries_per_second on role C ({_APPROX_DIR}); "
            f"'cheapest' means lowest measured search latency per query")


def load_bench_costs(model: str, pairing: str, bench: Path) -> Dict[str, float]:
    """Seconds per query per configuration, from the controlled benchmark.

    Refuses a benchmark that did not finish: a partial file would leave some
    configurations with no cost, and realized_regret would silently drop them
    from the cheapest-first ordering.
    """
    path = bench / model / "throughput.json"
    if not path.exists():
        raise FileNotFoundError(f"missing controlled-throughput results {path}")
    doc = json.load(open(path))
    if not doc.get("complete"):
        raise RuntimeError(f"{path} is incomplete ({doc.get('n_records')} records) "
                           f"-- finish the benchmark before costing H5 with it")
    table = doc.get("cost_s_per_query", {}).get(pairing)
    if not table:
        raise KeyError(f"{path} has no costs for pairing {pairing}")
    return {c: float(s) for c, s in table.items()}


def load_costs(model: str, pairing: str, *,
               approx: Path = DEFAULT_APPROX, role: str = "C",
               bench: Optional[Path] = None) -> Dict[str, float]:
    """Seconds per query for each configuration, from T7's measured throughput.

    H5's regret metric asks "did the map pick the CHEAPEST configuration that
    clears tau", so it needs an ordering over configurations by cost. The build
    plan defers the full cost model (latency + storage + energy) to the parent
    work, but latency is the axis actually measured on this hardware: T7 wrote
    ``queries_per_second`` per (pairing, config, role) while running the very
    searches whose CF is being graded. Using that is a measurement; assigning a
    rank by hand would be a guess.

    Returns seconds/query. Missing entries are omitted rather than defaulted,
    so a caller can see what it does not have.

    With ``bench`` set, the controlled benchmark is used instead (see
    load_bench_costs); T7's metadata is then not read at all.
    """
    if bench is not None:
        return load_bench_costs(model, pairing, bench)
    meta_path = approx / model / pairing / "approx_knn_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"missing T7 metadata {meta_path} -- the cost basis for H5's regret")
    meta = json.load(open(meta_path))
    out: Dict[str, float] = {}
    for entry in meta.get("configs", []):
        qps = entry.get("roles", {}).get(role, {}).get("queries_per_second")
        if qps:
            out[entry["config"]] = 1.0 / float(qps)
    return out


# ==========================================================================
# One (pairing) block: every array the tests read
# ==========================================================================
@dataclass
class Block:
    """Everything H0-H5 need for one modality configuration, loaded once.

    ``cell_mean``/``cell_n`` are [n_configs, K] in ``configs`` order, straight
    from T9's ``cell_stats.npz``. ``cf_b``/``cf_c`` are the true labels in role
    order. ``geom_b``/``geom_c`` carry assign/d1/d2/gap in the same row order.
    """
    model: str
    pairing: str
    K: int
    floor: int
    configs: List[str]
    families: Dict[str, str]
    cell_mean: np.ndarray
    cell_n: np.ndarray
    cf_b: Dict[str, np.ndarray]
    cf_c: Dict[str, np.ndarray]
    geom_b: Dict[str, np.ndarray]
    geom_c: Dict[str, np.ndarray]
    ga: Dict[str, float]
    costs: Dict[str, float] = field(default_factory=dict)

    @property
    def n_b(self) -> int:
        return len(self.geom_b["assign"])

    @property
    def n_c(self) -> int:
        return len(self.geom_c["assign"])

    @property
    def index_modality(self) -> str:
        return str(PAIRINGS[self.pairing]["index_modality"])

    @property
    def same_modality(self) -> bool:
        return bool(PAIRINGS[self.pairing]["same_modality"])

    def cfg_row(self, config: str) -> int:
        return self.configs.index(config)

    def ratio(self, role: str) -> np.ndarray:
        """d1/d2 in (0, 1] -- 1.0 means the query sits exactly on a cell wall.

        H2 splits on this rather than on the raw gap ``d2 - d1`` because the gap
        is measured in the units of whatever distances the pairing happens to
        produce, and the cross-modal pairings sit an order of magnitude further
        from their landmarks than the same-modal ones (T9's README: d1 ~ 1.35
        against ~ 0.5). A raw-gap threshold would therefore call almost every
        cross-modal query a boundary query for a reason that has nothing to do
        with boundaries. The ratio is scale-free, so one threshold means the
        same thing in all four pairings.
        """
        g = self.geom_b if role == "B" else self.geom_c
        d2 = np.maximum(g["d2"].astype(np.float64), 1e-12)
        return np.clip(g["d1"].astype(np.float64) / d2, 0.0, 1.0)


def load_block(lay: Layout, pairing: str, configs: List[str], floor: int,
               *, costs: bool = True) -> Block:
    """Load one pairing's labels, geometry, map and costs, asserting alignment."""
    stats = np.load(lay.pairing_dir(pairing) / "cell_stats.npz", allow_pickle=False)
    stored = [str(c) for c in stats["configs"].tolist()]
    keep = [c for c in configs if c in stored]
    rows = [stored.index(c) for c in keep]

    geom_b = load_query_geometry(lay, pairing, "B")
    geom_c = load_query_geometry(lay, pairing, "C")

    cf_b, cf_c, live = {}, {}, []
    for c in keep:
        fb, fc = lay.cf_vector(pairing, "B", c), lay.cf_vector(pairing, "C", c)
        if not (fb.exists() and fc.exists()):
            continue                      # T8 has not produced this one yet
        b, cc = np.load(fb).astype(np.float64), np.load(fc).astype(np.float64)
        if len(b) != len(geom_b["assign"]) or len(cc) != len(geom_c["assign"]):
            raise ValueError(
                f"{pairing}/{c}: CF has {len(b)}/{len(cc)} rows but geometry has "
                f"{len(geom_b['assign'])}/{len(geom_c['assign'])}. The join here is "
                f"POSITIONAL -- a length mismatch means the arrays are not the same "
                f"queries, and every number downstream would be silently wrong.")
        cf_b[c], cf_c[c], live = b, cc, live + [c]

    rows = [stored.index(c) for c in live]
    return Block(
        model=lay.model, pairing=pairing, K=int(stats["cf_mean"].shape[1]),
        floor=int(floor), configs=live,
        families={c: config_family(lay.model, c) for c in live},
        cell_mean=stats["cf_mean"][rows].astype(np.float64),
        cell_n=stats["n"][rows].astype(np.int64),
        cf_b=cf_b, cf_c=cf_c, geom_b=geom_b, geom_c=geom_c,
        # GA is recomputed from B rather than read from ga_fallback.json so it
        # is guaranteed to match the very label vector being scored against.
        ga={c: float(cf_b[c].mean()) for c in live},
        costs=(load_costs(lay.model, pairing, approx=_APPROX_DIR, bench=_COST_DIR)
               if costs else {}),
    )


# ==========================================================================
# Small shared helpers
# ==========================================================================
def open_map_meta(lay: Layout) -> dict:
    p = lay.dest / "map_meta.json"
    if not p.exists():
        raise FileNotFoundError(
            f"missing {p} -- T9 has not built the map for "
            f"{lay.model} at {lay.variant}")
    return json.load(open(p))


def write_csv(path: Path, header: List[str], rows) -> None:
    """CSV without pandas, which this repo does not depend on."""
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def add_common_args(ap, *, model: str = "siglip2") -> None:
    """The argument set every h*.py shares, so their invocations stay uniform."""
    ap.add_argument("--model", default=model)
    ap.add_argument("--k", type=int, default=1024)
    ap.add_argument("--spherical", action="store_true",
                    help="select the K*_spherical tiling; T5 established this "
                         "as the variant to build on -- pass it")
    ap.add_argument("--pairing", nargs="+", default=["all"])
    ap.add_argument("--configs", default="all")
    ap.add_argument("--floor", type=int, default=None,
                    help="SM backoff floor; defaults to the map's primary floor")
    ap.add_argument("--pilot-dir", type=Path, default=DEFAULT_PILOT)
    ap.add_argument("--kmeans-dir", type=Path, default=DEFAULT_KMEANS)
    ap.add_argument("--cf-dir", type=Path, default=DEFAULT_CF)
    ap.add_argument("--map-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_HT_OUT)
    ap.add_argument("--approx-dir", type=Path, default=DEFAULT_APPROX,
                    help="T7 metadata root -- the measured-throughput cost "
                         "basis for H5's regret. MUST be redirected alongside "
                         "--cf-dir etc. when running a scale other than the "
                         "pilot, or the cost model is silently read from 1M.")
    ap.add_argument("--own-cell-floor", action="store_true",
                    help="v2: SM_shared/SM_family back off on the cell's own B "
                         "count n_i < floor (draft q_min rule), not the pooled count")
    ap.add_argument("--boot-r", type=int, default=0,
                    help="v2: cell-resampled bootstrap with this many resamples (draft "
                         "l.478; user: 1000). H1 then passes a block iff the 95%% CI of "
                         "both Spearman correlations lies above 0. 0 = off (v1 p-value rule)")
    ap.add_argument("--do-protocol", action="store_true",
                    help="v2: DO = the protocol's density score (mean distance to "
                         "the 100 nearest corpus vectors) + offset; adds DO_shared")
    ap.add_argument("--cost-dir", type=Path, default=None,
                    help="controlled throughput benchmark root "
                         "(<dir>/<model>/throughput.json, from "
                         "knn/bench_throughput.py). When given, H5's cost "
                         "ordering comes from it instead of T7's timings.")


#: v2 (--own-cell-floor): every keying rung backs off when the cell's OWN B count
#: n_i is below the floor -- the draft's q_min rule (main.tex l.313/435). Off
#: (v1): SM_shared / SM_family compare the floor with the count POOLED over their
#: configs (9 n_i / 3 n_i), i.e. they serve cells of 2-9 / 4-9 B queries.
OWN_CELL_FLOOR = False
#: v2 (--do-protocol): DO is the protocol's one density score per landmark (mean
#: distance to its 100 nearest corpus vectors) + a fitted offset on GA, in two
#: forms -- "DO" (one slope per config) and "DO_shared" (one slope for all).
#: Off (v1): DO = three cell features + per-config OLS.
DO_PROTOCOL = False


def resolve_from_args(args):
    """(map Layout, HTLayout, floor, pairings, configs) from parsed args."""
    global _APPROX_DIR, _COST_DIR, OWN_CELL_FLOOR, DO_PROTOCOL
    OWN_CELL_FLOOR = bool(getattr(args, "own_cell_floor", False))
    DO_PROTOCOL = bool(getattr(args, "do_protocol", False))
    _APPROX_DIR = Path(getattr(args, "approx_dir", DEFAULT_APPROX))
    cd = getattr(args, "cost_dir", None)
    _COST_DIR = Path(cd) if cd is not None else None
    lay = layout(args.model, args.k, args.spherical, pilot=args.pilot_dir,
                 kmeans=args.kmeans_dir, cf=args.cf_dir, out=args.map_dir)
    meta = open_map_meta(lay)
    floor = args.floor if args.floor is not None else int(meta["reliability_floor"])
    pairings = [p for p in resolve_pairings(args.pairing) if p in meta["pairings"]]
    configs = resolve_configs(args.model, args.configs)
    return lay, ht_layout(args.model, args.out_dir), floor, pairings, configs, meta
