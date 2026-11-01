"""
T4 index catalogue: the 9 ANN configurations, per model, as data.

Four models, three dimensions:

    openclip   768 = 2^8 * 3
    siglip2   1152 = 2^7 * 3^2
    qwen3vl   2048 = 2^11
    jinav4    2048 = 2^11

Table 1 of the build plan is written at d=768, and its PQ sizes do not survive
the move. PQ slices a vector into m contiguous sub-vectors, so FAISS asserts
d % m == 0, and OPQ's M must likewise divide the post-transform dimension.
2048 is a pure power of two, so NO m divisible by 3 exists there at all:
2048/48 = 42.67, 2048/96 = 21.33, 2048/192 = 10.67. Table 1's numbers are
unbuildable for Qwen3-VL and Jina v4, not merely inconvenient.

WHAT SCALES WITH DIMENSION, AND WHAT DOES NOT
    Only the PQ/OPQ slice counts. Two other parameter groups are deliberately
    held identical across all four models:

    nlist / nprobe  are functions of the CORPUS, not the embedding. Every model
                    indexes the same |A| = 850,393 pilot pairs (the shared
                    split), so the IVF partition is sized identically. Varying
                    it per model would confound "different geometry" with
                    "different partition granularity".
    HNSW M / efC /  are graph parameters with no dimensional term. HNSW stores
    efSearch        full vectors, so its FILES differ in size with d, but the
                    graph it builds does not.

THE SCALING INVARIANT: dims-per-slice
    Table 1's values are kept exactly, and only rescaled for dimension. The
    invariant is dims-per-slice -- d/m for the encoder, d/M for the rotation --
    because that is what fixes the compression ratio: one byte replaces d/m
    float32s, so compression is 4*(d/m). Holding m constant instead would make
    the same nominal row mean different things per space: PQ48 is 64x
    compression at 768 but 171x at 2048.

    So Table 1's m=48 (16 dims/slice at its native 768) becomes m=72 at 1152
    and m=128 at 2048 -- the same configuration, not a coarser one.

    At d=2048 four entries cannot honour their target, because 2048 is a pure
    power of two and dims-per-slice must divide it: PQ 3 -> 4, OPQ 12 -> 8,
    OPQ 3 -> 4. These are listed in SNAPPED_2048 and asserted, so no other
    entry can quietly drift. Recorded, not corrected: CF is measured fresh per
    model in T8, and the study never compares absolute CF across models --
    multiple spaces exist to test whether H1-H5 REPLICATE, so what each
    catalogue must do is span the CF axis within its own model.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

# The four models of the spatial-fidelity-map plan, in priority order. This is
# NOT the seven-backbone roster of the earlier embedding run; ColPali (M5) is
# gated and has no single-vector landmark definition, so it is absent here.
MODEL_DIMS = {
    "siglip2": 1152,   # M1  contrastive dual-encoder; the pilot's reference space
    "qwen3vl": 2048,   # M2  LLM-backed unified retriever
    "openclip": 768,   # M3  contrastive dual-encoder, LAION-native
    "jinav4": 2048,    # M4  gap-reducing unified model
}

MODEL_DIRS = {
    "siglip2": "siglip2_so400m_patch16_512",
    "qwen3vl": "qwen3_vl_embedding_2b",
    "openclip": "openclip_vit_l14_laion2b",
    "jinav4": "jina_embeddings_v4",
}

# ---------------------------------------------------------------------------
# Per-tier parameters
# ---------------------------------------------------------------------------

# Corpus-sized, identical for every model (see module docstring). Table 1
# verbatim; oivfpq_low takes ivfpq_low's partition (see TIER NOTES).
# All three tiers of a family share ONE partition and ONE code, so low and mod
# are the high index with a smaller nprobe re-stamped in -- no separate build.
# Measured on gen 2: at fine code rates nprobe is a real fidelity dial (siglip2
# i->t went 0.343 -> 0.713 for 12.5% -> 25% scanned), whereas at gen 3's 16
# dims/slice a 64x nprobe increase moved CF not at all. Small nlist is what must
# be avoided: gen 2's IVF8 scanned 12.5% and still collapsed to 0.008 i->t.
NLIST = {
    "ivfpq_low": 32768, "ivfpq_mod": 32768, "ivfpq_high": 32768,
    "oivfpq_low": 32768, "oivfpq_mod": 32768, "oivfpq_high": 32768,
}
NPROBE = {
    "ivfpq_low": 512, "ivfpq_mod": 1024, "ivfpq_high": 4096,
    # oivfpq_high is Table 1's one exception: it specifies nprobe=32768 == nlist,
    # an exhaustive scan of every cell. Built at 8192 (25%) instead -- see
    # DEVIATIONS["oivfpq_high nprobe"]. Measured on SigLIP 2, probing depth is
    # already saturated well below this, and the exhaustive setting was ~86% of
    # T7's entire wall time.
    "oivfpq_low": 512, "oivfpq_mod": 2048, "oivfpq_high": 8192,
}

# Dims-per-slice each tier targets, read off Table 1 at its native d=768.
# These are documentation AND the assertion configs_for() checks PQ_M/OPQ_M
# against, so the tables cannot silently drift from the rule that generated
# them. SNAPPED_2048 lists the entries that provably cannot honour the target.
TARGET_PQ_DSUB = {
    "ivfpq_low": 8, "ivfpq_mod": 8, "ivfpq_high": 8,
    "oivfpq_low": 3, "oivfpq_mod": 3, "oivfpq_high": 3,
}
TARGET_OPQ_DSUB = {"oivfpq_low": 3, "oivfpq_mod": 3, "oivfpq_high": 3}

# m = d / TARGET_PQ_DSUB, and M = d / TARGET_OPQ_DSUB, written out per
# dimension rather than computed so the exact number built is auditable.
# configs_for() asserts these agree with the targets except where SNAPPED_2048
# says otherwise, so neither table can drift from the rule silently.
#
#   d=768   Table 1's own dimension -- every value is Table 1 verbatim
#   d=1152  every target is reachable exactly (1152 = 2^7 * 3^2)
#   d=2048  a pure power of two, so dims-per-slice must itself be a power of
#           two: PQ 3 -> 4, OPQ 12 -> 8, OPQ 3 -> 4. See SNAPPED_2048.
PQ_M = {
    768:  {"ivfpq_low": 96,  "ivfpq_mod": 96,  "ivfpq_high": 96,
           "oivfpq_low": 256, "oivfpq_mod": 256, "oivfpq_high": 256},
    1152: {"ivfpq_low": 144, "ivfpq_mod": 144, "ivfpq_high": 144,
           "oivfpq_low": 384, "oivfpq_mod": 384, "oivfpq_high": 384},
    2048: {"ivfpq_low": 256, "ivfpq_mod": 256, "ivfpq_high": 256,
           "oivfpq_low": 512, "oivfpq_mod": 512, "oivfpq_high": 512},
}

# OPQ's block count. A SEPARATE table from PQ_M because Table 1 deliberately
# sets them differently on its O-IVFPQ MOD row: OPQ64 with PQ48, so the rotation
# balances 12-dim blocks while the encoder cuts 16-dim slices. FAISS warns about
# that mismatch; it is Table 1's own choice and is reproduced faithfully.
#
# low and high both set M == m instead:
#   high  is Table 1's (OPQ256 with an assumed PQ256 -- see DEVIATIONS)
#   low   is the added row, matched deliberately so that ivfpq_low and
#         oivfpq_low differ ONLY in whether a rotation is applied, and that
#         rotation is optimally configured rather than mismatched. That makes
#         the pair a clean test of what OPQ buys, which is what H3 needs.
#         It also removes a 2048 snap: d/M = 16 is a power of two, so M=128
#         is exact there, whereas the mod row's 12 is not.
OPQ_M = {
    768:  {"oivfpq_low": 256, "oivfpq_mod": 256, "oivfpq_high": 256},
    1152: {"oivfpq_low": 384, "oivfpq_mod": 384, "oivfpq_high": 384},
    2048: {"oivfpq_low": 512, "oivfpq_mod": 512, "oivfpq_high": 512},
}

# Entries that cannot honour their target because the dimension forbids it.
# Keyed (d, tier, which) -> (target dims-per-slice, built dims-per-slice).
SNAPPED_2048 = {
    (2048, "oivfpq_low", "pq"):   (3, 4),    # 2048/3  = 682.67
    (2048, "oivfpq_mod", "pq"):   (3, 4),
    (2048, "oivfpq_high", "pq"):  (3, 4),
    (2048, "oivfpq_low", "opq"):  (3, 4),
    (2048, "oivfpq_mod", "opq"):  (3, 4),
    (2048, "oivfpq_high", "opq"): (3, 4),
}

# TIER NOTES
#
# oivfpq_low is the one row NOT in Table 1, which gives O-IVFPQ only a mod and
# a high. It is built from values Table 1 already contains rather than invented
# ones: ivfpq_low's partition and code (IVF4096, PQ48, nprobe=8) plus a rotation
# matched to that code (OPQ M = PQ m). That makes it a genuine low point --
# 8/4096 = 0.2% of cells scanned -- and makes ivfpq_low and oivfpq_low differ in
# EXACTLY ONE THING, whether a rotation is applied. A controlled pair by
# construction, which is what H3 needs to attribute fidelity to the rotation
# rather than to code length or probe depth. The rotation is matched rather than
# mismatched so the pair measures what a correctly-configured OPQ buys; the mod
# row keeps Table 1's OPQ64/PQ48 mismatch.
#
# Table 1's nprobe/nlist fractions, for reading the tiers: ivfpq_low and
# oivfpq_low scan 0.2% of cells, ivfpq_mod / ivfpq_high / oivfpq_mod scan
# 12.5%, and oivfpq_high scans 100% (nprobe == nlist), so oivfpq_high is the
# only configuration whose loss is pure quantisation -- its measured CF is
# therefore the PQ ceiling at that code length.

# HNSW: no dimensional term, identical everywhere. Table 1 verbatim.
HNSW_PARAMS = {
    # efSearch 16 -> 128: see DEVIATIONS["hnsw_low efSearch"]. Search-time only;
    # the M=8 graph is unchanged, so the file is re-stamped, not rebuilt.
    "hnsw_low": ({"M": 8, "efConstruction": 40}, {"efSearch": 128}, "~50-56%"),
    "hnsw_mod": ({"M": 32, "efConstruction": 200}, {"efSearch": 128}, "~66-92%"),
    "hnsw_high": ({"M": 48, "efConstruction": 400}, {"efSearch": 2048}, "~91%"),
}

# Orientation only. The IVFPQ figures are the CIKM LAION-7M text->image
# reference values Table 1 carries; the O-IVFPQ figures are the predicted
# text->image values for the respecified family. Actual CF is measured fresh
# in T8 and will differ by scale, modality and direction.
REF_CF100 = {
    "ivfpq_low": "~28%", "ivfpq_mod": "~60%", "ivfpq_high": "~53%",
    "oivfpq_low": "n/a (row not in Table 1)",
    "oivfpq_mod": "~74-78%", "oivfpq_high": "~84%",
}

TIER_ORDER = [
    "hnsw_low", "hnsw_mod", "hnsw_high",
    "ivfpq_low", "ivfpq_mod", "ivfpq_high",
    "oivfpq_low", "oivfpq_mod", "oivfpq_high",
]


@dataclass(frozen=True)
class IndexConfig:
    name: str                      # stable id; the filename stem and the T8 join key
    family: str                    # HNSW | IVFPQ | O-IVFPQ  -- the H3 grouping
    point: str                     # low | mod | high -- position on the CF axis
    factory: Optional[str]         # FAISS factory string; None => explicit ctor
    build: Dict[str, int] = field(default_factory=dict)
    search: Dict[str, int] = field(default_factory=dict)
    trains_on_gpu: bool = False    # IVF coarse quantiser goes to the GPU
    ref_cf100: str = ""
    note: str = ""

    @property
    def is_hnsw(self) -> bool:
        return self.family == "HNSW"


def configs_for(model: str) -> List[IndexConfig]:
    """The 9 configurations for one model, sized to that model's dimension."""
    if model not in MODEL_DIMS:
        raise KeyError(f"unknown model {model!r}; known: {sorted(MODEL_DIMS)}")
    d = MODEL_DIMS[model]
    if d not in PQ_M:
        raise KeyError(f"no PQ sizing table for d={d}")
    ms = PQ_M[d]

    out: List[IndexConfig] = []

    for name, (build, search, ref) in HNSW_PARAMS.items():
        out.append(IndexConfig(
            name=name, family="HNSW", point=name.split("_")[1], factory=None,
            build=dict(build), search=dict(search), trains_on_gpu=False,
            ref_cf100=ref,
        ))

    def _check(name, value, kind, targets):
        """Divisibility, plus agreement with the dims-per-slice rule."""
        assert d % value == 0, \
            f"{model}/{name}: {kind} {value} does not divide d={d}"
        got, want = d // value, targets[name]
        snapped = SNAPPED_2048.get((d, name, kind.lower()))
        if got != want:
            assert snapped == (want, got), (
                f"{model}/{name}: {kind} {value} gives {got} dims per slice, "
                f"target is {want}, and this is not a recorded snap. Either fix "
                f"the table or add it to SNAPPED_2048."
            )
        return got

    for name in ["ivfpq_low", "ivfpq_mod", "ivfpq_high"]:
        m, nlist = ms[name], NLIST[name]
        dsub = _check(name, m, "PQ", TARGET_PQ_DSUB)
        out.append(IndexConfig(
            name=name, family="IVFPQ", point=name.split("_")[1],
            factory=f"IVF{nlist},PQ{m}",
            build={"nlist": nlist, "m": m, "nbits": 8},
            search={"nprobe": NPROBE[name]},
            trains_on_gpu=True, ref_cf100=REF_CF100[name],
            note=f"{m} B/vec, {dsub} dims/slice ({4 * dsub}x compression); "
                 f"{NPROBE[name]}/{nlist} cells scanned "
                 f"({100 * NPROBE[name] / nlist:.3g}%).",
        ))

    for name in ["oivfpq_low", "oivfpq_mod", "oivfpq_high"]:
        m, nlist, M = ms[name], NLIST[name], OPQ_M[d][name]
        dsub = _check(name, m, "PQ", TARGET_PQ_DSUB)
        odsub = _check(name, M, "OPQ", TARGET_OPQ_DSUB)
        # OPQ M and PQ m are independent parameters and Table 1 sets them
        # differently (OPQ64 with PQ48). No d_out is given, so the rotation is
        # square and discards no dimension.
        out.append(IndexConfig(
            name=name, family="O-IVFPQ", point=name.split("_")[1],
            factory=f"OPQ{M},IVF{nlist},PQ{m}",
            build={"opq_M": M, "nlist": nlist, "m": m, "nbits": 8},
            search={"nprobe": NPROBE[name]},
            trains_on_gpu=True, ref_cf100=REF_CF100[name],
            note=f"{m} B/vec, {dsub} dims/slice ({4 * dsub}x compression); "
                 f"OPQ M={M} balances {odsub} dims/block"
                 f"{'' if M == m else ' (!= PQ m, FAISS will warn -- Table 1s own choice)'}; "
                 f"{NPROBE[name]}/{nlist} cells scanned "
                 f"({100 * NPROBE[name] / nlist:.3g}%).",
        ))

    order = {n: i for i, n in enumerate(TIER_ORDER)}
    return sorted(out, key=lambda c: order[c.name])


FAMILIES = ["HNSW", "IVFPQ", "O-IVFPQ"]

# The two index modalities. Four query->index pairings are evaluated in T7, but
# only two indexes exist per model: a pairing is a choice of which vector set
# queries an already-built index, not a separate build.
INDEX_MODALITIES = ["image", "text"]

# Rank-identical to cosine on these vectors, which are L2-normalised at
# generation. For unit vectors ||x-y||^2 = 2 - 2<x,y>, so L2-argmin and
# IP-argmax return the same ordering, including across modalities. L2 is chosen
# over METRIC_INNER_PRODUCT because PQ/OPQ minimise an L2 reconstruction
# objective and IVF k-means is unambiguous under it.
#
# T6 (exact-KNN) MUST use this same metric. CF@k intersects T6's top-k with
# T7's; if the two disagree on metric the intersection still returns a number,
# just a meaningless one.
METRIC = "L2"


DEVIATIONS = {
    "hnsw_low efSearch": {
        "table_1_says": "M=8, efConstruction=40, efSearch=16.",
        "built_as": "M=8, efConstruction=40, efSearch=128. Graph unchanged.",
        "why": (
            "At efSearch=16 the tier cleared tau=0.7 for 0.0-10% of queries at "
            "7M, and for exactly 0% in four cross-modal cells (OpenCLIP and "
            "SigLIP 2, t->i and i->t) -- no CF variance for the map to predict. "
            "Measured by knn/sweep_search_params.py on a fixed 5,000-query B "
            "sample, efSearch 16..256 against T6 labels."
        ),
        "evidence": (
            "efSearch=128 is the lowest grid value that lifts every hnsw_low cell "
            "off 0% at tau=0.7 (minimum 0.6%, SigLIP 2 i->t), while hnsw_mod at "
            "the same efSearch stays 7.8-60.7 points ahead in every cell -- the "
            "tiers still separate, now on graph degree M alone. At 256 the "
            "closest gap to hnsw_mod shrinks to 3.0 points."
        ),
        "residual": (
            "At tau>=0.8, one to two SigLIP 2 i->t cells stay under 0.5% at any "
            "efSearch tested: an M=8 graph against the widest modality gap in the "
            "roster. A graph-degree limit, not a search-depth one."
        ),
    },
    "oivfpq_high nprobe": {
        "table_1_says": "nprobe=32768, equal to nlist -- every cell scanned.",
        "built_as": "nprobe=8192, a quarter of the cells.",
        "why": (
            "At nprobe==nlist the IVF partition stops doing anything: the search "
            "is an exhaustive PQ scan of all 850,393 codes, and cost is |A| x m "
            "lookups regardless of nlist. Measured on SigLIP 2, that made this "
            "single configuration ~86% of T7's entire wall time across the whole "
            "catalogue."
        ),
        "evidence_it_is_safe": (
            "SigLIP 2's ivfpq_high (nprobe 4096/32768 = 12.5%, m=384) and "
            "oivfpq_high (nprobe 16384 = 50%, m=384) differ only in probing "
            "depth and the OPQ rotation. Their CF@100 on role C: i->i 92.0% vs "
            "92.0%, t->t 92.5% vs 92.5%, i->t 71.3% vs 74.5%, t->i 39.3% vs "
            "40.6%. Two are identical and the other two sit at or near the tie "
            "floor, so recall is already saturated by 12.5% and quadrupling the "
            "depth buys under a point. The binding constraint at this tier is "
            "quantisation error, not cell coverage."
        ),
        "consequence": (
            "The tier still spans the top of the CF axis -- it holds the longest "
            "codes in the catalogue (m = d/3) and probes 2x deeper than "
            "ivfpq_high, so the two remain distinct configurations. What it no "
            "longer is, is exhaustive; it must not be described as a full scan."
        ),
        "note_siglip2": (
            "SigLIP 2's indexes were built before this decision and carry "
            "nprobe=16384. Its T7/T8 results are therefore at 50% rather than "
            "25% -- one of several parameters on which that model's build "
            "predates the current catalogue."
        ),
    },
    "O-IVFPQ family": {
        "table_1_says": (
            "two rows only -- mod: OPQ64, IVF128, PQ48, nprobe=16; "
            "high: OPQ256, IVF32768, nprobe=32768."
        ),
        "built_as": (
            "three rows, per model, at dsub 6/4/3 (2048: 8/4/2) -- see PQ_M."
        ),
        "why": (
            "Table 1's values are the CIKM study's, at d=768. Its high row "
            "states no PQ code size at all, and its OPQ M=256 does not divide "
            "1152 or 2048. The family is respecified by dims-per-slice so it "
            "is constructible and comparable in every space."
        ),
        "note": (
            "At d=768 the replacement reproduces Table 1's OPQ256 exactly, "
            "since dsub=3 there IS m=256. The deviation is one of spelling "
            "for OpenCLIP and of substance only where d != 768."
        ),
    },
    "d=2048 snapping": {
        "affects": "qwen3vl, jinav4",
        "why": (
            "2048 = 2^11 has only power-of-two divisors, so the dsub targets "
            "24/12/6/3 are unreachable and snap to 32/16/8/2. No m divisible "
            "by 3 exists at all, which is why Table 1's 48/96/192 cannot be "
            "used for these two models under any reading."
        ),
        "consequence": (
            "Their catalogues span 128x..8x compression against 96x..12x for "
            "the other two. Recorded, not corrected: the study compares "
            "whether H1-H5 replicate across spaces, never absolute CF between "
            "models, so each catalogue need only span the axis within its own."
        ),
        "cost": (
            "oivfpq_high at 2048 is m=1024 -- 1024 bytes/vector (870 MB over "
            "|A|) and 1024 sub-quantiser codebooks to train. The heaviest "
            "index in the plan. Drop to m=512 (dsub 4) if that proves "
            "impractical, accepting that mod and high then coincide."
        ),
    },
}


def describe(model: str) -> str:
    d = MODEL_DIMS[model]
    lines = [f"model={model}  d={d}",
             f"{'name':<14} {'family':<8} {'pt':<5} {'build':<44} {'search':<16} "
             f"{'dsub':<5} {'B/vec':<7} ref"]
    lines.append("-" * 116)
    for c in configs_for(model):
        b = ", ".join(f"{k}={v}" for k, v in c.build.items())
        s = ", ".join(f"{k}={v}" for k, v in c.search.items())
        dsub = str(d // c.build["m"]) if "m" in c.build else "-"
        bpv = str(c.build["m"]) if "m" in c.build else f"{d * 4}"
        lines.append(f"{c.name:<14} {c.family:<8} {c.point:<5} {b:<44} {s:<16} "
                     f"{dsub:<5} {bpv:<7} {c.ref_cf100}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    models = sys.argv[1:] or list(MODEL_DIMS)
    for m in models:
        print(describe(m))
        print()
    for name, dev in DEVIATIONS.items():
        print(f"DEVIATION -- {name}")
        for k, v in dev.items():
            print(f"  {k}: {v}")
        print()
