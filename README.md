# Fidelity Synopses — pipeline code

Code that produced the results of *Fidelity Synopses: Pre-Execution Plan Selection for
Approximate Vector Search*: embeddings, data split, index catalog, exact and approximate
search, fidelity labels, landmarks, the synopsis, hypothesis tests, and the plan-selection
experiments, including the smaller catalogs, the k = 10 study, and the HNSW index over the
landmarks.

Not included: results, embeddings, indexes, model weights, the LAION data, and the scripts
that draw the paper's figures and tables.

## Layout

| Directory | Contents |
|---|---|
| `models/` | download and check the model checkpoints |
| `embeddingsgenerator/` | image and caption embeddings per model; anomaly scans; merge of sharded runs |
| `audit/` | duplicate groups and exclusion list; checks behind the embedding corrections |
| `stratification/` | 7M shard cut; stratified 85/10/5 split into corpus, Q, Q_test |
| `indexes/` | catalog tuning on a 50k sample; index builds; embedding and split paths (`v2_dirs.py`) |
| `knn/` | full-scale search-value check; exact k-NN; throughput benchmark; approximate k-NN (T7); fidelity labels (T8); single-index catalog setup |
| `landmarks/` | spherical k-means landmarks (T5); query-to-cell assignment; landmark density |
| `spatialmap/` | the synopsis (T9): cell statistics, verification, baselines |
| `hypothesis_testing/` | tests H0–H5 (T10) |
| `paperexp/` | lookup cost, plan selection, budget rule, trade-off curves, extras, landmark-HNSW study |
| `paperexp/analysis/` | same-configuration rates of the HNSW lookup; random-blend comparison |
| `configs/` | inputs chosen by hand: 7M shard manifest, final search values, single-index search values |
| `envs/` | package versions of the two Python environments |

## Environments

Two environments, both Python 3.11. They cannot be merged: `torch 2.13.0` pins the CUDA 13.0
runtime libraries, and `faiss-gpu-cuvs 1.15.1` needs version 13.2 or later of the same libraries.

| Variable | Package list | Used for |
|---|---|---|
| `$PY` | `envs/requirements.txt` | everything except the files in the next row: anomaly scans, duplicates, split, index tuning and builds, every search, landmarks, synopsis, tests, plan selection (FAISS 1.15.1 with GPU, cuVS and AVX2 CPU) |
| `$PYE` | `envs/requirements-embed.txt` | model download and check, `embeddingsgenerator/gen_*.py`, the four `audit/*_check.py` scripts (torch 2.13.0, transformers 5.16.1, open-clip 3.3.0) |

Differences from the environments that made the results:

- Index tuning and the search-value check ran with `faiss-cpu` 1.15.1, and exact k-NN ran
  with `faiss-gpu-cu12` 1.14.1. Both now use `faiss-gpu-cuvs` 1.15.1.
- `audit/jina_prefix_check.py` ran with `transformers` 4.57.1, `torch` 2.8.0,
  `huggingface_hub` 0.36.2, `tokenizers` 0.22.2 and `flash_attn` 2.8.3.post1. It was not run
  with the versions in `requirements-embed.txt`.

## Before you run

- Run every command from the repository root, with these variables set:

  ```bash
  R=$(pwd)
  PY=$R/.venv/bin/python          # envs/requirements.txt
  PYE=$R/.venv-embed/bin/python   # envs/requirements-embed.txt
  RES=$R/result_v2
  MODELS="siglip2 openclip qwen3vl jinav4"
  CONFIGS="hnsw_low hnsw_mod hnsw_high ivfpq_low ivfpq_mod ivfpq_high oivfpq_low oivfpq_mod oivfpq_high"
  ```

- Set `OMP_NUM_THREADS` (and `OPENBLAS_NUM_THREADS`) to the `--threads` value of each command.
- Many scripts stop when `PBS_JOBID` is not set, so that they do not run on a login node.
  Run them on a compute node. Some accept `--allow-login-node`.
- All outputs go to `result_v2/`. A few inputs come from the earlier embedding run in
  `result/`: `result/corpus_shard_manifest.json` (copy of `configs/corpus_shard_manifest.json`),
  `result/excluded_pairs.txt` (anomaly list from stage 1), and the embedding trees that are
  reused unchanged (`indexes/v2_dirs.py`, `COPIED_FROM_V1`).
- All paths are relative to the repository. The scripts that read the LAION tar files
  without a `--data-root` / `--laion` argument expect them at `../laion/worker1_20/tar1`
  (next to the repository).
- `<IMG>` and `<TXT>` below are the embedding trees of a model, from
  `indexes/v2_dirs.py` (`embed_dir(model, "image")`, `embed_dir(model, "text")`).
- Timing runs (stages 4, 7 and 11) used one exclusive node, except where a step says
  "shared node".

## Pipeline

### 1. Models and embeddings

```bash
$PYE models/download_models.py
$PYE models/download_new_models.py
$PYE models/verify_models.py
```

Embed every LAION shard with `embeddingsgenerator/gen_<model>.py` (images) and
`embeddingsgenerator/gen_<model>_text.py` (captions). Each generator takes `--data-root`
(the LAION tar root) and `--output-dir`. Then scan for degenerate items:

```bash
$PY embeddingsgenerator/scan_anomalies.py --workers 48
$PY embeddingsgenerator/scan_caption_anomalies.py --workers 48
$PY embeddingsgenerator/build_anomaly_union.py           # -> result/excluded_pairs.txt
```

Checks behind the embedding corrections (SigLIP2 caption case, OpenCLIP image
normalization, Qwen3-VL caption length, Jina-v4 query prefix):

```bash
$PYE audit/siglip2_lowercase_check.py --out result/audit/siglip2_lowercase_check --device cuda:0
$PYE audit/openclip_imagenorm_check.py --device cuda:0 --num-workers 24
$PYE audit/qwen3vl_truncation_check.py --device cuda:0 --procs 30
OUT=result/audit/jina_prefix_check
$PYE audit/jina_prefix_check.py sample  --out $OUT --seed 0 --min-b 20000
for i in 0 1 2 3; do                                      # one GPU per part
  $PYE audit/jina_prefix_check.py encode --out $OUT --part $i --parts 4 --device cuda:$i &
done; wait
$PYE audit/jina_prefix_check.py analyze --out $OUT --parts 4 --device cuda:0
```

Embedding corrections. SigLIP2 captions in lower case:

```bash
$PYE embeddingsgenerator/gen_siglip2_text.py --data-root ../laion/worker1_20/tar1 --tokenizer siglip2 \
    --output-dir $RES/embeddings_text/siglip2_so400m_patch16_512_lowercase \
    --batch-size 512 --num-workers 12 --torch-threads 1 --shards-per-loader 16 --device cuda:0
```

OpenCLIP images with the checkpoint's mean/std (0.5), in 8 parts, then merged:

```bash
E=$RES/embeddings_image
for I in 0 1 2 3 4 5 6 7; do                              # one GPU job per part
  $PYE embeddingsgenerator/gen_openclip.py --data-root ../laion/worker1_20/tar1 --image-norm checkpoint \
      --shard-list $E/openclip_shard_list.txt --shard-stride 8 --shard-index $I \
      --output-dir $E/openclip_norm05_part$I --max-images 100000000 \
      --batch-size 256 --num-workers 14 --torch-threads 1 --shards-per-loader 16 --device cuda:0
done
$PY embeddingsgenerator/merge_sharded_parts.py --model openclip --modality image --embed-root $E \
    --parts $E/openclip_norm05_part{0..7} --dest $E/openclip_vit_l14_laion2b_norm05
```

Copy the six unchanged embedding trees from `result/` into `result_v2/` (`COPIED_FROM_V1` in
`indexes/v2_dirs.py`, source → destination). After each step, check that the new tree has the
same shards and the same names per shard as the other modality, unit-norm rows, and
8,003,932 rows in total.

### 2. Duplicates and split

The image duplicate groups are unchanged from the earlier run (copy
`result/duplicate_groups/siglip2/corpus/image` into `$RES/duplicate_groups/siglip2/corpus/`).
The text groups are new:

```bash
$PY audit/duplicate_groups.py --model siglip2 --modality text --scope corpus \
    --embed-root $RES/embeddings_text/siglip2_so400m_patch16_512_lowercase \
    --out-dir $RES/duplicate_groups --ngpu 1 --threads 16
```

The minimum group size per modality is the 95th percentile of the group sizes + 1
(`n_total` in `groups.csv`): 8 for images, 6 for text.

```bash
$PY audit/build_exclusion_list.py --model siglip2 --scope corpus \
    --groups-dir $RES/duplicate_groups --out-dir $RES/duplicate_exclusions \
    --anomalies result/excluded_pairs.txt --image-min-size 8 --text-min-size 6
```

Stratified split (corpus A, Q = B, Q_test = C) on the 7M shard cut:

```bash
STRAT=$RES/stratification/7M_cleaned/siglip2; SPLIT=$RES/7M_cleaned/siglip2
$PY stratification/run_stratification.py --model siglip2 \
    --image-root result/embeddings_image/siglip2_so400m_patch16_512 \
    --text-root $RES/embeddings_text/siglip2_so400m_patch16_512_lowercase \
    --excluded-pairs $RES/duplicate_exclusions/exclusions_combined.txt \
    --corpus-manifest result/corpus_shard_manifest.json \
    --out-dir $STRAT --k 25 --n-epochs 3 --seed 0
$PY stratification/split_abc.py --image-labels $STRAT/image_stratum_labels.csv \
    --text-labels $STRAT/text_stratum_labels.csv \
    --out $SPLIT/abc_split.csv --meta $SPLIT/abc_split_meta.json --seed 0
$PY stratification/assign_ids.py --split $SPLIT/abc_split.csv --out $SPLIT/pair_index.csv \
    --meta $SPLIT/pair_index_meta.json --scope result_v2/7M_cleaned/siglip2
```

The 7M shard cut (`configs/corpus_shard_manifest.json`) comes from
`stratification/select_corpus.py`. `audit/duplicate_catalogue.py` and `audit/byte_identity.py`
describe the duplicate groups.

### 3. Index catalog

Tuning on a 50k sample (exact top-100 first, then a sweep per family, then the proposal of
the low / mod / high settings):

```bash
for M in $MODELS; do $PY indexes/tune_catalogue_v2.py --stage sample --model $M --threads 4; done
for F in HNSW IVFPQ O-IVFPQ; do for M in $MODELS; do for X in image text; do
  $PY indexes/tune_catalogue_v2.py --stage sweep --model $M --index-modality $X --family $F --threads 16
done; done; done
$PY indexes/tune_catalogue_v2.py --stage propose --threads 1
```

Builds (72 indexes: 4 models × 2 modalities × 9 configurations):

```bash
for M in $MODELS; do for X in image text; do for C in $CONFIGS; do
  $PY indexes/build_indexes_v2.py --model $M --modality $X --config $C --threads 32
done; done; done
```

Search values at full scale:

```bash
for C in $CONFIGS; do for M in $MODELS; do for X in image text; do
  $PY knn/check_search_params_v2.py --stage sweep --model $M --modality $X --config $C --threads 32
done; done; done
$PY knn/check_search_params_v2.py --stage propose --threads 1
```

The search values used from here on were chosen by hand from the proposal:
`configs/final_params_chosen.json`, placed at `$RES/param_check/7M_cleaned/final_params_chosen.json`.

### 4. Exact search, costs, approximate search, fidelity labels

```bash
P=$RES/param_check/7M_cleaned/final_params_chosen.json
PILOT=$RES/7M_cleaned/siglip2
MAN=result/corpus_shard_manifest.json
```

Exact top-100 for Q and Q_test (2 GPUs), per model and index modality:

```bash
for M in $MODELS; do
  $PY knn/exact_knn.py --model $M --pairing image_query_image_index text_query_image_index --roles B C \
      --pilot-dir $PILOT --manifest $MAN --out-dir $RES/knn/7M_cleaned \
      --image-root <IMG> --text-root <TXT> --gpu --ngpu 2 --batch 8192
  $PY knn/exact_knn.py --model $M --pairing text_query_text_index image_query_text_index --roles B C \
      --pilot-dir $PILOT --manifest $MAN --out-dir $RES/knn/7M_cleaned \
      --image-root <IMG> --text-root <TXT> --gpu --ngpu 2 --batch 8192
done
```

Cost per configuration (throughput benchmark, exclusive node):

```bash
for M in $MODELS; do
  $PY knn/bench_throughput.py --model $M --scale-tag 7M_cleaned --params-json $P \
      --pilot-dir $PILOT --index-root $RES/indexes/7M_cleaned \
      --out-dir $RES/throughput/7M_cleaned --threads 64 --n-queries 5000
done
```

Approximate top-100 of every configuration (T7) and fidelity labels (T8). Both run from
`knn/`. The original runs split T7 into one job per (model, direction, group of
configurations); `--pairing` and `--configs` select the part.

```bash
cd knn
for M in $MODELS; do
  $PY approx_knn.py --model $M --pairing all --configs all --threads 64 --params-json $P \
      --index-root $RES/indexes/7M_cleaned --pilot-dir $PILOT --manifest $R/$MAN \
      --knn-dir $RES/knn/7M_cleaned --out-dir $RES/approx_knn/7M_cleaned \
      --query-cache $RES/approx_knn/7M_cleaned/_queries
done
for M in $MODELS; do
  $PY cf_compute.py --model $M --pairing all --merge --params-json $P \
      --index-root $RES/indexes/7M_cleaned --pilot-dir $PILOT --knn-dir $RES/knn/7M_cleaned \
      --approx-dir $RES/approx_knn/7M_cleaned --out-dir $RES/cf/7M_cleaned
done
cd $R
```

`--query-cache` is read when it holds the query matrices; `--build-query-cache` writes them.

### 5. Landmarks and synopsis

Spherical k-means landmarks (T5), per model and modality:

```bash
for M in $MODELS; do for X in image text; do
  for KS in "500 1250 2500 5000 10000" "20000 25000" "40000"; do
    $PY landmarks/build_landmarks.py --model $M --modality $X --k $KS --spherical --gpu --threads 16 \
        --pilot-dir $PILOT --manifest $MAN --out-dir $RES/kmeans/7M_cleaned --seed 0 \
        --image-root <IMG> --text-root <TXT>
  done
done; done
```

Query-to-cell assignment and landmark density:

```bash
KALL="500 1250 2500 5000 10000 20000 25000 40000"
for M in $MODELS; do
  $PY landmarks/occupancy_check.py --model $M --k $KALL --spherical --gpu --threads 16 \
      --pilot-dir $PILOT --manifest $MAN --out-dir $RES/kmeans/7M_cleaned \
      --image-root <IMG> --text-root <TXT>
  for X in image text; do
    $PY landmarks/landmark_density.py --model $M --modality $X --k-list $KALL --nn 100 \
        --kmeans-dir $RES/kmeans/7M_cleaned
  done
done
```

Synopsis (T9) at K = 10,000:

```bash
COMMON="--k 10000 --spherical --pilot-dir $RES/7M_cleaned --kmeans-dir $RES/kmeans/7M_cleaned \
        --cf-dir $RES/cf/7M_cleaned --out-dir $RES/spatial_map/7M_cleaned"
for M in $MODELS; do
  $PY spatialmap/assign_queries.py    --model $M $COMMON
  $PY spatialmap/build_map.py         --model $M $COMMON --histogram --cf-histogram
  $PY spatialmap/verify_map.py        --model $M $COMMON --strict --results-root $RES
  $PY spatialmap/compare_baselines.py --model $M $COMMON --do-protocol
  $PY spatialmap/plot_map.py          --model $M $COMMON
done
```

### 6. Hypothesis tests (T10)

```bash
for M in $MODELS; do
  $PY hypothesis_testing/run_all.py --model $M --spherical --k 10000 \
      --pilot-dir $RES/7M_cleaned --kmeans-dir $RES/kmeans/7M_cleaned --cf-dir $RES/cf/7M_cleaned \
      --map-dir $RES/spatial_map/7M_cleaned --approx-dir $RES/approx_knn/7M_cleaned \
      --cost-dir $RES/throughput/7M_cleaned --out-dir $RES/hypothesis_testing/7M_cleaned/K10000 \
      --own-cell-floor --do-protocol --boot-r 1000 --sweep $KALL --keep-going
done
```

### 7. Plan selection

```bash
export SFM_RESULTS_ROOT=result_v2 SFM_K_MAIN=10000 SFM_TILING_TAG=7M_cleaned
export SFM_HT_DIR=$RES/hypothesis_testing/7M_cleaned/K10000
KLIST=500,1250,2500,5000,10000,20000,25000,40000
PE=$RES/paper_experiments; OUT=$PE/7M_cleaned/K10000; COSTS=$RES/throughput/7M_cleaned
```

Per model `M`, in this order:

```bash
# lookup cost c_est (exclusive node)
SFM_KS=$KLIST $PY paperexp/bench_estimate.py --model $M --threads 64 --v2 --cost-dir $COSTS --out-dir $OUT
# single-query lookup cost (exclusive node)
$PY paperexp/bench_lookup.py --model $M --threads 64 --ks $KLIST --params-json $P \
    --kmeans-root $RES/kmeans/7M_cleaned --index-root $RES/indexes/7M_cleaned \
    --pilot-dir $PILOT --out-dir $PE/7M_cleaned
# selection rules, K sweep; budget rule at the static plan's rate
SFM_KS=$KLIST $PY paperexp/selection_tables.py --model $M --v2-stats --boot-r 1000 --cost-dir $COSTS --out-dir $OUT
SFM_KS=$KLIST $PY paperexp/matched_violation.py --model $M --v2-protocol --cost-dir $COSTS --out-dir $OUT
# budget rule at the requirement (1 - rho = 10%)
SFM_KS=10000 $PY paperexp/matched_violation.py --model $M --v2-protocol --configs all \
    --viol-cap 0.10 --out-name matched_cap10.json --cost-dir $COSTS --out-dir $OUT
# trade-off curves, calibration
SFM_KS=10000 $PY paperexp/matched_violation.py --model $M --v2-protocol --curves --cost-dir $COSTS --out-dir $OUT
SFM_KS=10000 $PY paperexp/extras_v2.py --model $M --out-dir $OUT --cost-dir $COSTS
```

Random-blend comparison (after all four models): `$PY paperexp/analysis/blend_check.py`.

### 8. Landmark seeds

Same stages with landmark seeds 1 and 2, at K = 2,500 and 10,000:

```bash
for S in 1 2; do
  KT=7M_cleaned_seed$S
  for M in $MODELS; do
    for X in image text; do
      $PY landmarks/build_landmarks.py --model $M --modality $X --k 2500 10000 --spherical --gpu \
          --threads 16 --pilot-dir $PILOT --manifest $MAN --out-dir $RES/kmeans/$KT --seed $S \
          --image-root <IMG> --text-root <TXT>
    done
    $PY landmarks/occupancy_check.py --model $M --k 2500 10000 --spherical --gpu --threads 16 \
        --pilot-dir $PILOT --manifest $MAN --out-dir $RES/kmeans/$KT --image-root <IMG> --text-root <TXT>
  done
done
```

Then, for each seed, stage 5's T9 loop with `--kmeans-dir $RES/kmeans/$KT --out-dir
$RES/spatial_map/$KT`, and stage 6 with `--kmeans-dir $RES/kmeans/$KT --map-dir
$RES/spatial_map/$KT --out-dir $RES/hypothesis_testing/$KT/K10000`. Plan selection reuses the
seed-0 lookup cost:

```bash
O=$PE/$KT/K10000; mkdir -p $O/$M; cp $OUT/$M/estimate.json $O/$M/
SFM_TILING_TAG=$KT SFM_KS=10000 $PY paperexp/selection_tables.py --model $M --v2-stats --boot-r 1000 \
    --cost-dir $COSTS --out-dir $O
SFM_TILING_TAG=$KT SFM_KS=10000 $PY paperexp/matched_violation.py --model $M --v2-protocol \
    --cost-dir $COSTS --out-dir $O
```

### 9. Smaller catalogs

Single-index catalog: one HNSW index (the `hnsw_high` build) at three efSearch values.

```bash
$PY knn/setup_single_index_v2.py          # index links and single_index_params.json
SP=$RES/param_check/7M_cleaned/single_index_params.json
cd knn
for M in $MODELS; do
  for C in hnsw_low hnsw_mod hnsw_high; do
    $PY approx_knn.py --model $M --pairing all --configs $C --threads 64 --params-json $SP \
        --index-root $RES/indexes/7M_cleaned_single --pilot-dir $PILOT --manifest $R/$MAN \
        --knn-dir $RES/knn/7M_cleaned --out-dir $RES/approx_knn/7M_cleaned_single \
        --query-cache $RES/approx_knn/7M_cleaned/_queries
  done
  $PY cf_compute.py --model $M --pairing all --merge --configs hnsw_low,hnsw_mod,hnsw_high \
      --params-json $SP --index-root $RES/indexes/7M_cleaned_single --pilot-dir $PILOT \
      --knn-dir $RES/knn/7M_cleaned --approx-dir $RES/approx_knn/7M_cleaned_single \
      --out-dir $RES/cf/7M_cleaned_single
done
cd $R
for M in $MODELS; do                       # exclusive node
  $PY knn/bench_throughput.py --model $M --scale-tag 7M_cleaned --params-json $SP --pilot-dir $PILOT \
      --index-root $RES/indexes/7M_cleaned_single --out-dir $RES/throughput/7M_cleaned_single \
      --configs hnsw_low,hnsw_mod,hnsw_high --threads 64 --n-queries 5000
  $PY knn/bench_throughput.py --model $M --scale-tag 7M_cleaned --params-json $P --pilot-dir $PILOT \
      --index-root $RES/indexes/7M_cleaned --out-dir $RES/throughput/7M_cleaned_single_ref \
      --configs hnsw_high --threads 64 --n-queries 5000
done
$PY knn/setup_single_index_v2.py --combine   # high setting = the main hnsw_high (single_b)
$PY knn/setup_single_index_v2.py --rescale   # costs on the main benchmark's scale (single_c)
$PY knn/setup_single_index_v2.py --union     # nine configurations + two single-index settings
```

Budget rule per catalog, per model `M` (copy the main `estimate.json` into each tree first):

| Catalog | `SFM_CF_TAG` | `COSTS` | `--configs` | extra arguments |
|---|---|---|---|---|
| `hnswonly` | not set | `throughput/7M_cleaned` | `hnsw_low,hnsw_mod,hnsw_high` | |
| `single_c` | `7M_cleaned_single_b` | `throughput/7M_cleaned_single_c` | `hnsw_low,hnsw_mod,hnsw_high` | |
| `union` | `7M_cleaned_union` | `throughput/7M_cleaned_union` | `all` | `--extra-configs hsi_low,hsi_mod` |

```bash
export SFM_KS=10000
O=$PE/7M_cleaned_<catalog>/K10000; mkdir -p $O/$M; cp $OUT/$M/estimate.json $O/$M/
$PY paperexp/matched_violation.py --model $M --v2-protocol --configs <configs> \
    --cost-dir $RES/<COSTS> --out-dir $O [extra arguments]
$PY paperexp/matched_violation.py --model $M --v2-protocol --configs <configs> \
    --viol-cap 0.10 --out-name matched_cap10.json --cost-dir $RES/<COSTS> --out-dir $O [extra arguments]
```

### 10. Result depth k = 10

Fidelity@10 is taken from the stored top-100 lists; no new approximate search.

```bash
ln -s 7M_cleaned $RES/knn/7M_cleaned_k10            # verify_map reads these trees by tag
ln -s 7M_cleaned $RES/approx_knn/7M_cleaned_k10
cd knn
for M in $MODELS; do
  $PY cf_compute.py --model $M --pairing all --merge --topk 10 --params-json $P \
      --index-root $RES/indexes/7M_cleaned --pilot-dir $PILOT --knn-dir $RES/knn/7M_cleaned \
      --approx-dir $RES/approx_knn/7M_cleaned --out-dir $RES/cf/7M_cleaned_k10
done
cd $R
for M in $MODELS; do                       # exclusive node
  $PY knn/bench_throughput.py --model $M --scale-tag 7M_cleaned_k10 --k 10 --params-json $P \
      --pilot-dir $PILOT --index-root $RES/indexes/7M_cleaned \
      --out-dir $RES/throughput/7M_cleaned_k10 --threads 64 --n-queries 5000
done
```

Then stage 5's T9 loop with `--cf-dir $RES/cf/7M_cleaned_k10 --out-dir
$RES/spatial_map/7M_cleaned_k10` (landmarks unchanged), and stage 6 with `--cf-dir
$RES/cf/7M_cleaned_k10 --map-dir $RES/spatial_map/7M_cleaned_k10 --cost-dir
$RES/throughput/7M_cleaned_k10 --out-dir $RES/hypothesis_testing/7M_cleaned_k10/K10000`.
Plan selection at k = 10, per model:

```bash
export SFM_CF_TAG=7M_cleaned_k10 SFM_MAP_TAG=7M_cleaned_k10 SFM_FID_K=10 SFM_KS=$KLIST
export SFM_HT_DIR=$RES/hypothesis_testing/7M_cleaned_k10/K10000
O=$PE/7M_cleaned_k10/K10000; C10=$RES/throughput/7M_cleaned_k10
mkdir -p $O/$M; cp $OUT/$M/estimate.json $O/$M/
$PY paperexp/selection_tables.py  --model $M --v2-stats --boot-r 1000 --cost-dir $C10 --out-dir $O
$PY paperexp/matched_violation.py --model $M --v2-protocol --cost-dir $C10 --out-dir $O
$PY paperexp/matched_violation.py --model $M --v2-protocol --configs all --viol-cap 0.10 \
    --out-name matched_cap10.json --cost-dir $C10 --out-dir $O
$PY paperexp/matched_violation.py --model $M --v2-protocol --curves --cost-dir $C10 --out-dir $O
$PY paperexp/extras_v2.py --model $M --out-dir $O --cost-dir $C10
```

The paper reports only the estimation accuracy at k = 10.

### 11. HNSW index over the landmarks

All at K = 10,000, with `SFM_KS=10000` and the stage 7 variables. Per model `M`:

```bash
LH=$PE/7M_cleaned_lmhnsw/K10000; LU=$PE/7M_cleaned_lmhnsw_uniform/K10000
# M x efSearch sweep, cell agreement with the exact scan
$PY paperexp/split_single_query.py --model $M --v2 --hnsw-sweep --min-window 10 --k 10000 \
    --hnsw-m 8,16,32 --hnsw-ef 8,16,32,64,128,256 --pilot-dir $PILOT --cost-dir $COSTS --out-dir $LH
# one setting per model (--uniform); HNSW cells for the tuning fold and Q_test
mkdir -p $LU/$M; cp $LH/$M/split_lookup_variants_K10000.json $OUT/$M/estimate.json $LU/$M/
$PY paperexp/hnsw_assign.py --model $M --v2 --uniform --out-dir $LU --threads 16 --pilot-dir $PILOT
A=$LU/$M/hnsw_assign
$PY paperexp/selection_tables.py  --model $M --v2-stats --boot-r 1000 --assign-dir $A --cost-dir $COSTS --out-dir $LU
$PY paperexp/matched_violation.py --model $M --v2-protocol --assign-dir $A --cost-dir $COSTS --out-dir $LU
$PY paperexp/matched_violation.py --model $M --v2-protocol --configs all --assign-dir $A \
    --viol-cap 0.10 --out-name matched_cap10.json --cost-dir $COSTS --out-dir $LU
# single-query timing, exact scan and HNSW alternately per query (shared node)
$PY paperexp/lookup_interleaved.py --model $M --k 10000 --sweep-dir $LH --out-dir $LU --pilot-dir $PILOT
# batched timing, each lookup in its own block of passes
$PY paperexp/lookup_batched_interleaved.py --model $M --sweep-dir $LH --out-dir $LU \
    --pilot-dir $PILOT --no-interleave
```

The same steps without `--uniform` and with `--out-dir $LH` give the per-direction setting
(tree `7M_cleaned_lmhnsw`). Same-configuration rates of both trees, after all four models:
`$PY paperexp/analysis/same_plan.py`.

## Outputs

Everything is written under `result_v2/`:

| Path | Stage |
|---|---|
| `embeddings_image/`, `embeddings_text/` | 1 |
| `duplicate_groups/`, `duplicate_exclusions/`, `stratification/`, `7M_cleaned/` (split) | 2 |
| `index_tuning/`, `indexes/7M_cleaned/`, `param_check/` | 3 |
| `knn/`, `throughput/`, `approx_knn/`, `cf/` | 4 |
| `kmeans/`, `spatial_map/` | 5 |
| `hypothesis_testing/` | 6 |
| `paper_experiments/7M_cleaned/` | 7 |
| `*_seed1`, `*_seed2` trees | 8 |
| `*_single*`, `*_hnswonly`, `*_union` trees | 9 |
| `*_k10` trees | 10 |
| `paper_experiments/7M_cleaned_lmhnsw*` | 11 |
