"""
Where the second run (result_v2/) keeps its inputs -- one place, so v2 stages
cannot drift apart on it.

result_v2/ is the results tree built on the corrected embeddings: lowercase
SigLIP 2 text and OpenCLIP images with the checkpoint's own normalization.
Every v2 stage reads its vectors and its split from here, never from result/
(v1). Vectors that did not change are COPIED from v1 into the matching place
(embeddingsgenerator/jobs/copy_v1_embeddings_v2.pbs); COPIED_FROM_V1 says which.

    from v2_dirs import embed_dir, SPLIT_DIR
    load_selected(model, modality, ids, SPLIT_DIR / "pair_index.csv",
                  root=embed_dir(model, modality))
"""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
V1 = REPO_ROOT / "result"
V2 = REPO_ROOT / "result_v2"

#: (model, modality) -> the worker_*/shard_*.npy tree v2 reads.
EMBED_DIRS = {
    ("siglip2", "image"): V2 / "embeddings_image/siglip2_so400m_patch16_512",
    ("siglip2", "text"): V2 / "embeddings_text/siglip2_so400m_patch16_512_lowercase",
    ("openclip", "image"): V2 / "embeddings_image/openclip_vit_l14_laion2b_norm05",
    ("openclip", "text"): V2 / "embeddings_text/openclip_vit_l14_laion2b",
    ("qwen3vl", "image"): V2 / "embeddings_image/qwen3_vl_embedding_2b",
    ("qwen3vl", "text"): V2 / "embeddings_text/qwen3_vl_embedding_2b",
    ("jinav4", "image"): V2 / "embeddings_image/jina_embeddings_v4",
    ("jinav4", "text"): V2 / "embeddings_text/jina_embeddings_v4",
}

#: The trees v2 takes unchanged from v1: v2 path -> v1 path. The other two
#: (SigLIP 2 text, OpenCLIP image) were embedded again for v2.
COPIED_FROM_V1 = {
    EMBED_DIRS[("siglip2", "image")]: V1 / "embeddings_image/siglip2_so400m_patch16_512",
    EMBED_DIRS[("openclip", "text")]: V1 / "embeddings_text/openclip_vit_l14_laion2b",
    EMBED_DIRS[("qwen3vl", "image")]: V1 / "embeddings_image/qwen3_vl_embedding_2b",
    EMBED_DIRS[("qwen3vl", "text")]: V1 / "embeddings_text/qwen3_vl_embedding_2b",
    EMBED_DIRS[("jinav4", "image")]: V1 / "embeddings_image/jina_embeddings_v4",
    EMBED_DIRS[("jinav4", "text")]: V1 / "embeddings_text/jina_embeddings_v4",
}

#: The v2 A/B/C split: A_ids.npy, B_ids.npy, C_ids.npy, pair_index.csv. Its ids
#: are scoped "result_v2/7M_cleaned/siglip2" -- never mix them with v1 ids.
SPLIT_DIR = V2 / "7M_cleaned/siglip2"
SPLIT_SCOPE = "result_v2/7M_cleaned/siglip2"

#: The 7M shard cut, unchanged from v1 (the user kept the v1 corpus procedure).
MANIFEST = V1 / "corpus_shard_manifest.json"


#: The 72 v2 indexes (indexes/build_indexes_v2.py): <root>/<model>/<modality>/<config>.faiss
#: + <config>.build.json.
INDEX_ROOT = V2 / "indexes/7M_cleaned"

#: The search values the user chose at 5.8M (2026-10-14) -- what the throughput
#: benchmark and T7 search with. The files themselves carry only the build's
#: starting values.
PARAMS_CHOSEN = V2 / "param_check/7M_cleaned/final_params_chosen.json"

CONFIG_ORDER = [f"{f}_{t}" for f in ("hnsw", "ivfpq", "oivfpq") for t in ("low", "mod", "high")]


def embed_dir(model: str, modality: str) -> Path:
    try:
        return EMBED_DIRS[(model, modality)]
    except KeyError:
        raise KeyError(f"no v2 embedding tree for {model}/{modality}; "
                       f"known: {sorted(EMBED_DIRS)}") from None


def configs_v2(model: str, params: Path = PARAMS_CHOSEN, index_root: Path = INDEX_ROOT):
    """
    The 9 v2 configurations of one model as index_configs.IndexConfig objects,
    so the v1 search scripts (bench_throughput.py, approx_knn.py) can take them
    in place of the v1 catalogue.

    build  from the image index's build.json, in read_structure's keys; the text
           index must carry the same build (one config per tier for both
           modalities, user 2026-10-13) or this refuses.
    search the chosen value from `params`. The caller SETS it on the loaded
           index -- unlike v1, the file's own value is only the build's start.
    """
    from index_configs import IndexConfig                       # noqa: E402
    chosen = json.loads(Path(params).read_text())["params"]
    out = []
    for name in CONFIG_ORDER:
        b = {x: json.loads((index_root / model / x / f"{name}.build.json").read_text())
             for x in ("image", "text")}
        keys = ("family", "M", "efConstruction", "nlist", "m", "dsub", "factory")
        diff = {k: (b["image"].get(k), b["text"].get(k)) for k in keys
                if b["image"].get(k) != b["text"].get(k)}
        if diff:
            raise RuntimeError(f"{model}/{name}: image and text builds differ {diff}")
        bj, p = b["image"], chosen[name][model]
        if bj["family"] == "HNSW":
            build = {"M": bj["M"], "efConstruction": bj["efConstruction"]}
            factory = None
        else:
            build = {"nlist": bj["nlist"], "m": bj["m"], "nbits": 8}
            if bj["family"] == "O-IVFPQ":
                build["opq_M"] = bj["m"]
            factory = bj["factory"]
        out.append(IndexConfig(name=name, family=bj["family"], point=name.split("_")[1],
                               factory=factory, build=build, search={p["param"]: int(p["value"])},
                               note=f"v2: build {index_root / model}/<modality>/{name}.build.json, "
                                    f"search value from {params}"))
    return out
