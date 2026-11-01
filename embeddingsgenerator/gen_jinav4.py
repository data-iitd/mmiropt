#!/usr/bin/env python3
"""
jina-embeddings-v4 IMAGE embedding generation from TAR shards.

Weights load from the local models/ directory - this never touches the network,
which matters because Pragya compute nodes have no proxy route.

ABOUT THIS MODEL
    jinaai/jina-embeddings-v4, 3.75B params, 2048-d single-vector. A Qwen2.5-VL
    backbone with TASK-SPECIFIC LoRA ADAPTERS: config.task_names is
    ["retrieval", "text-matching", "code"], and every call must name one.

    peft is REQUIRED. The 0.34 GiB adapters/adapter_model.safetensors is applied
    per task_label; without peft the adapter cannot load and you would silently
    get base Qwen2.5-VL embeddings labelled as jina-v4 - the same class of
    silent-wrong-model failure as loading a checkpoint with strict=False and
    ignoring the result.

    The repo also declares:
      single_vector_pool_strategy  mean
      multi_vector_projector_dim   128     (multi-vector mode, NOT used here)
      matryoshka_dims              [128, 256, 512, 1024, 2048]

    return_multivector stays False, so this writes ONE 2048-d vector per image
    and fits the same [N, dim] layout as every other generator. --truncate-dim
    can shorten it to any matryoshka size, but the default keeps the full 2048.

    Loading needs trust_remote_code=True. Unlike jina-clip-v2, whose auto_map
    points at a SEPARATE repo, v4's modeling code ships inside the snapshot
    (modeling_jina_embeddings_v4.py, qwen2_5_vl.py, custom_lora_module.py), so
    it works offline with HF_HUB_OFFLINE=1.

NATIVE RESOLUTION
    Qwen2.5-VL keeps native resolution, so images cannot be stacked into one
    tensor - the DataLoader hands back a LIST and the model's own processor sees
    each PIL image. The LAION shards cap at 256px, so nothing is upsampled.

OUTPUT - one file pair per shard, mirroring the dataset layout
    result/embeddings_image/jina_embeddings_v4/
        worker_0/
            shard_00000.npy            float32 [N, 2048], L2-normalised
            shard_00000.names.json     N names, same row order as the .npy
            ...
        progress.json / failed_images.json / anomalous_images.json
        run_summary.json

    Names are fully qualified - worker_0/shard_00000/000000000.jpg - because
    workers reuse the same id space and a bare filename would collide.

RUN THIS IN A PBS JOB, not on the login node.
"""

import os
import sys
import io
import re
import gc
import json
import time
import queue
import tarfile
import argparse
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Set

# Never reach for the network: compute nodes have no proxy. v4's remote code is
# inside the snapshot, so trust_remote_code works offline.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
from transformers import AutoModel

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_KEY = "jina_embeddings_v4"
DEFAULT_MODEL_DIR = REPO_ROOT / "models/jina_embeddings_v4"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "result/embeddings_image" / MODEL_KEY
DEFAULT_MAX_IMAGES = 8_000_000
DEFAULT_TASK = "retrieval"


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def l2norm(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-12)


def is_image_file(name: str) -> bool:
    return Path(name).suffix.lower() in IMAGE_EXTENSIONS


def parse_worker_index(name: str) -> Optional[int]:
    m = re.match(r"^worker_(\d+)$", name)
    return int(m.group(1)) if m else None


def discover_workers(data_root: Path, min_worker: int, max_worker: int) -> List[Tuple[int, str, Path]]:
    workers = []
    for p in data_root.iterdir():
        if not p.is_dir():
            continue
        idx = parse_worker_index(p.name)
        if idx is None:
            continue
        if min_worker <= idx <= max_worker:
            workers.append((idx, p.name, p))
    workers.sort(key=lambda x: x[0])
    return workers


def discover_tars(worker_path: Path) -> List[Path]:
    tars = [p for p in worker_path.iterdir() if p.is_file() and p.suffix.lower() == ".tar"]
    tars.sort()
    return tars


# An image is "anomalous" if it is too degenerate for an embedding to mean
# anything: a sliver, or a single pixel. RECORDED, not dropped - the vectors are
# still written, so nothing silently disappears and row alignment with the text
# side is preserved.
MIN_DIM = 16
MAX_ASPECT = 20.0


# Qwen's image processor rejects an aspect ratio above 200:1 outright:
#     ValueError: absolute aspect ratio must be smaller than 200, got 256.0
# raised inside smart_resize, which kills the whole process. Exactly two images
# in this corpus trip it - worker_1/shard_00031/000319880.jpg at 256x1 and
# worker_2/shard_00251/002517061.jpg at 1x256 - both already catalogued among
# the known-degenerate images.
#
# Dropping them is not an option: every later row would shift and silently pair
# with the wrong caption. So the short side is padded with black just far enough
# to clear the limit (256x1 -> 256x2, i.e. 128:1) and the row is embedded.
#
# The guard is a no-op for anything at or below 180:1, so every other image in
# the corpus reaches the processor byte-identical and its embedding is
# unchanged. The reported width/height stay the ORIGINAL values, so the anomaly
# catalogue still records the true 256x1.
MAX_PROC_ASPECT = 180


def pad_extreme_aspect(arr: torch.Tensor) -> torch.Tensor:
    """Pad the short side only if the aspect ratio would break smart_resize."""
    ah, aw = arr.shape[0], arr.shape[1]
    if min(ah, aw) <= 0 or max(ah, aw) / min(ah, aw) <= MAX_PROC_ASPECT:
        return arr
    need = -(-max(ah, aw) // MAX_PROC_ASPECT)      # ceil division
    if ah < aw:
        pad = torch.zeros((need - ah, aw, 3), dtype=torch.uint8)
        return torch.cat([arr, pad], dim=0)
    pad = torch.zeros((ah, need - aw, 3), dtype=torch.uint8)
    return torch.cat([arr, pad], dim=1)


def classify_anomaly(w: int, h: int) -> Optional[str]:
    if w <= 0 or h <= 0:
        return "zero_dimension"
    if min(w, h) < MIN_DIM:
        return f"tiny_side_{min(w, h)}px"
    if max(w, h) / min(w, h) > MAX_ASPECT:
        return f"extreme_aspect_{max(w, h) / min(w, h):.0f}to1"
    return None


class JsonLog:
    """Append-only JSON sidecar (failed / anomalous images)."""

    def __init__(self, path: Path, label: str):
        self.path = path
        self.label = label
        self.lock = threading.Lock()
        self.entries: List[Dict] = []
        self.dirty = False
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.entries = json.load(f)
                log(f"Loaded {len(self.entries)} previous {label} entries")
            except Exception:
                self.entries = []

    def add(self, rows: List[Dict]) -> None:
        if not rows:
            return
        with self.lock:
            self.entries.extend(rows)
            self.dirty = True

    def count(self) -> int:
        with self.lock:
            return len(self.entries)

    def save(self) -> None:
        with self.lock:
            if not self.dirty:
                return
            tmp = self.path.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.entries, f, indent=2)
            os.replace(tmp, self.path)
            self.dirty = False


class ProgressManager:
    """
    Resume state. Shards are atomic - a shard is either fully written or not
    written at all - so this is just a completed set plus counters.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.progress_path = output_dir / "progress.json"
        self.lock = threading.Lock()
        self.started_at = time.time()
        self.completed_tars: Set[str] = set()
        self.total_images = 0
        self.total_failed = 0
        self.load()

    def load(self) -> None:
        if not self.progress_path.exists():
            return
        try:
            with open(self.progress_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.completed_tars = set(data.get("completed_tars", []))
            self.total_images = int(data.get("total_images", 0))
            self.total_failed = int(data.get("total_failed", 0))
            log(f"Resume: {len(self.completed_tars)} shards done, "
                f"{self.total_images:,} embeddings")
        except Exception as e:
            log(f"WARNING: could not load progress.json: {e}")

    def save(self) -> None:
        with self.lock:
            data = {
                "completed_tars": sorted(self.completed_tars),
                "total_images": self.total_images,
                "total_failed": self.total_failed,
                "elapsed_seconds": time.time() - self.started_at,
                "updated_at": datetime.now().isoformat(),
            }
            tmp = self.progress_path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, self.progress_path)

    def is_completed(self, tar_key: str) -> bool:
        return tar_key in self.completed_tars

    def mark(self, tar_key: str, added: int, failed: int) -> None:
        with self.lock:
            self.completed_tars.add(tar_key)
            self.total_images += added
            self.total_failed += failed


def write_shard(output_dir: Path, worker_name: str, tar_name: str,
                names: List[str], mat: np.ndarray) -> Path:
    """
    Write one shard's vectors + names atomically.

    The .npy is renamed into place BEFORE the names file, so a reader driven by
    the names sidecar never sees names whose vectors are not yet on disk.
    """
    if mat.shape[0] != len(names):
        raise RuntimeError(f"vector/name mismatch: {mat.shape[0]} vs {len(names)}")

    d = output_dir / worker_name
    d.mkdir(parents=True, exist_ok=True)
    stem = Path(tar_name).stem

    npy_final = d / f"{stem}.npy"
    json_final = d / f"{stem}.names.json"
    npy_tmp = d / f".{stem}.npy.tmp.npy"
    json_tmp = d / f".{stem}.names.json.tmp"

    try:
        np.save(npy_tmp, mat)
        with open(json_tmp, "w", encoding="utf-8") as f:
            json.dump(names, f)
        os.replace(npy_tmp, npy_final)
        os.replace(json_tmp, json_final)
    except Exception:
        for p in (npy_tmp, json_tmp):
            try:
                p.unlink()
            except Exception:
                pass
        raise
    return npy_final


class TarPrefetcher:
    """
    Loads image bytes from shards in a background thread.

    Queue item: (worker_name, tar_name, tar_key, items, error_msg)
    where items is List[(member_name, raw_bytes)] sorted by filename.
    """

    def __init__(self, queue_size: int = 2):
        self.queue = queue.Queue(maxsize=queue_size)
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.shard_list: List[Tuple[str, Path]] = []

    def start(self, shard_list: List[Tuple[str, Path]]) -> None:
        self.shard_list = shard_list
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        for worker_name, tar_path in self.shard_list:
            if self.stop_event.is_set():
                break

            tar_key = f"{worker_name}/{tar_path.name}"
            items: List[Tuple[str, bytes]] = []
            err = None

            try:
                with tarfile.open(tar_path, "r") as tf:
                    for member in tf:
                        if not member.isfile() or not is_image_file(member.name):
                            continue
                        f = tf.extractfile(member)
                        if f is None:
                            continue
                        try:
                            items.append((member.name, f.read()))
                        except Exception:
                            continue
            except Exception as e:
                err = str(e)

            # Deterministic row order, and the SAME order every other generator
            # uses. Ids are zero-padded to 9 digits so lexicographic == numeric.
            items.sort(key=lambda t: Path(t[0]).name)
            self.queue.put((worker_name, tar_path.name, tar_key, items, err))

        self.queue.put(None)

    def get_next(self):
        return self.queue.get()

    def stop(self) -> None:
        self.stop_event.set()


class MultiShardDataset(Dataset):
    """
    Decodes SEVERAL shards' images within one DataLoader lifetime.

    No preprocessing here: jina-v4 keeps native resolution and its encode_image
    takes PIL objects, so workers do the expensive part - the JPEG decode - and
    hand back a uint8 HWC array, which travels over shared memory far more
    cheaply than a pickled PIL object.

    Grouping shards amortises DataLoader fork/join (~34 ms per worker per
    shard), and output stays strictly shard-aligned: each item carries its flat
    index and the caller maps it back to (group, position) before writing.
    """

    def __init__(self, groups):
        self.groups = groups
        self.index: List[Tuple[int, int]] = []
        for gi, g in enumerate(groups):
            for ii in range(len(g["items"])):
                self.index.append((gi, ii))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, k: int):
        gi, ii = self.index[k]
        _, raw = self.groups[gi]["items"][ii]
        try:
            with Image.open(io.BytesIO(raw)) as im:
                w, h = im.size
                arr = torch.from_numpy(np.asarray(im.convert("RGB"), dtype=np.uint8).copy())
            return pad_extreme_aspect(arr), k, True, w, h
        except Exception:
            return torch.zeros(1, 1, 3, dtype=torch.uint8), k, False, 0, 0


def collate_variable(batch):
    """Keep images as a LIST - native resolution means they cannot be stacked."""
    arrs = [b[0] for b in batch]
    ks = torch.tensor([b[1] for b in batch], dtype=torch.long)
    oks = torch.tensor([b[2] for b in batch], dtype=torch.bool)
    ws = torch.tensor([b[3] for b in batch], dtype=torch.long)
    hs = torch.tensor([b[4] for b in batch], dtype=torch.long)
    return arrs, ks, oks, ws, hs


def group_image_name(g: Dict, ii: int) -> str:
    """Fully-qualified name: worker_0/shard_00000/000000000.jpg"""
    return (f"{g['worker_name']}/{Path(g['tar_name']).stem}/"
            f"{Path(g['items'][ii][0]).name}")


def _silence_jina_tqdm(model) -> None:
    """
    Stop jina's encode_* from printing a tqdm bar for every batch.

    encode_image/encode_text wrap their internal loader in
        for batch in tqdm(dataloader, desc=desc, disable=self.verbosity == 0)
    and _encode_batch calls them once per DataLoader batch, so a full bar is
    printed per batch and the shard progress lines drown in it.

    Setting .verbosity on what from_pretrained returns does NOT work:
    jina's loader returns a PeftModel WRAPPING the jina model, so the attribute
    lands on the wrapper and the inner model - the one that reads it - never
    sees it. So two belts:
      1. set verbosity on the wrapper and on the model it wraps
      2. neutralise tqdm inside jina's own module, which holds no matter how
         the model is wrapped
    Logging only; the encoding path is untouched.
    """
    for m in (model,
              getattr(model, "model", None),
              getattr(model, "base_model", None)):
        if m is not None:
            try:
                m.verbosity = 0
            except Exception:
                pass
    for name, mod in list(sys.modules.items()):
        if "modeling_jina_embeddings_v4" in name and hasattr(mod, "tqdm"):
            mod.tqdm = lambda it, *a, **k: it


class JinaV4ImageEmbedder:
    def __init__(self, device: str, batch_size: int, model_dir: Path,
                 num_workers: int = 8, task: str = DEFAULT_TASK,
                 truncate_dim: Optional[int] = None):
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.task = task
        self.truncate_dim = truncate_dim

        if str(device).startswith("cuda"):
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        log(f"Loading jina-embeddings-v4 from {model_dir}")
        # trust_remote_code: the modeling code ships inside this snapshot, so
        # this resolves offline. peft must be installed for the task adapter.
        self.model = AutoModel.from_pretrained(
            str(model_dir), trust_remote_code=True, dtype=torch.bfloat16
        ).to(device).eval()
        _silence_jina_tqdm(self.model)

        tasks = list(getattr(self.model.config, "task_names", []) or [])
        if tasks and self.task not in tasks:
            raise ValueError(f"--task {self.task!r} not in model task_names {tasks}")
        dims = list(getattr(self.model.config, "matryoshka_dims", []) or [])
        if self.truncate_dim is not None and dims and self.truncate_dim not in dims:
            raise ValueError(f"--truncate-dim {self.truncate_dim} not in {dims}")

        self.dim = int(self.truncate_dim or max(dims) if dims else 2048)
        log(f"Model loaded (task {self.task!r}, tasks available {tasks}, "
            f"embedding dim {self.dim})")

    def _encode_batch(self, images: List[Image.Image]) -> torch.Tensor:
        """
        Encode one batch of PIL images, halving it on OOM.

        encode_image does its own preprocessing and batching, so batch_size is
        pinned to the whole list here and the outer DataLoader batch is what
        actually controls GPU memory.
        """
        try:
            vecs = self.model.encode_image(
                images=images,
                task=self.task,
                batch_size=len(images),
                return_multivector=False,   # one vector per image, [N, dim]
                return_numpy=False,
                truncate_dim=self.truncate_dim,
            )
            if isinstance(vecs, list):
                vecs = torch.stack([v if isinstance(v, torch.Tensor)
                                    else torch.as_tensor(v) for v in vecs])
            return l2norm(vecs.float().cpu())
        except torch.cuda.OutOfMemoryError:
            if str(self.device).startswith("cuda"):
                torch.cuda.empty_cache()
            if len(images) == 1:
                raise
            half = len(images) // 2
            log(f"  OOM: splitting batch of {len(images)} -> {half}")
            return torch.cat([self._encode_batch(images[:half]),
                              self._encode_batch(images[half:])], dim=0)

    @torch.inference_mode()
    def encode_groups(self, groups: List[Dict]):
        """
        Encode several shards under ONE DataLoader, returning a per-shard
        (names, vectors[float32, L2-normalised], failed, anomalous) tuple in the
        same order as `groups`.

        For every shard, len(items) == len(names) + len(failed) still holds.
        """
        empty = [([], np.zeros((0, self.dim), dtype=np.float32), [], []) for _ in groups]
        if not groups or all(not g["items"] for g in groups):
            return empty

        ds = MultiShardDataset(groups)
        names_by: List[List[str]] = [[] for _ in groups]
        anom_by: List[List[Dict]] = [[] for _ in groups]
        vecs_by: List[List[np.ndarray]] = [[] for _ in groups]
        failed_by: List[List[Tuple[str, str]]] = [[] for _ in groups]

        loader = DataLoader(
            ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,           # row order must stay deterministic
            drop_last=False,
            collate_fn=collate_variable,
            prefetch_factor=4 if self.num_workers > 0 else None,
        )

        for arrs, ks, oks, ws, hs in loader:
            for k in ks[~oks].tolist():
                gi, ii = ds.index[k]
                failed_by[gi].append((group_image_name(groups[gi], ii), "decode"))

            for k, w, h in zip(ks.tolist(), ws.tolist(), hs.tolist()):
                reason = classify_anomaly(w, h)
                if reason:
                    gi, ii = ds.index[k]
                    anom_by[gi].append({"image_name": group_image_name(groups[gi], ii),
                                        "width": w, "height": h, "reason": reason})

            good = [i for i, ok in enumerate(oks.tolist()) if ok]
            if not good:
                continue

            images = [Image.fromarray(arrs[i].numpy()) for i in good]
            feats = self._encode_batch(images).numpy().astype(np.float32)
            for row, i in enumerate(good):
                gi, ii = ds.index[int(ks[i])]
                names_by[gi].append(group_image_name(groups[gi], ii))
                vecs_by[gi].append(feats[row])
            del images, feats

        out = []
        for gi in range(len(groups)):
            v = vecs_by[gi]
            mat = (np.stack(v).astype(np.float32) if v
                   else np.zeros((0, self.dim), dtype=np.float32))
            out.append((names_by[gi], mat, failed_by[gi], anom_by[gi]))
        return out


def build_pending_shards(
    data_root: Path,
    min_worker: int,
    max_worker: int,
    progress: ProgressManager,
    shard_stride: int = 1,
    shard_index: int = 0,
) -> List[Tuple[str, Path]]:
    """
    Shards still to do, optionally taking only every Nth one.

    shard_stride/shard_index let several processes split ONE worker across
    several GPUs: same range, stride N, indices 0..N-1 gives disjoint partitions
    that cover everything.
    """
    pending: List[Tuple[str, Path]] = []
    seen = 0
    for _, worker_name, worker_path in discover_workers(data_root, min_worker, max_worker):
        for tar_path in discover_tars(worker_path):
            mine = (seen % shard_stride) == shard_index
            seen += 1
            if not mine:
                continue
            tar_key = f"{worker_name}/{tar_path.name}"
            if not progress.is_completed(tar_key):
                pending.append((worker_name, tar_path))
    return pending


def run_pipeline(
    data_root: Path,
    output_dir: Path,
    model_dir: Path,
    min_worker: int,
    max_worker: int,
    max_images: int,
    batch_size: int,
    device: str,
    prefetch_shards: int,
    num_workers: int,
    shards_per_loader: int,
    shard_stride: int,
    shard_index: int,
    task: str,
    truncate_dim: Optional[int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    progress = ProgressManager(output_dir)
    if progress.total_images >= max_images:
        log(f"Target already reached: {progress.total_images:,}/{max_images:,}")
        return

    failures = JsonLog(output_dir / "failed_images.json", "failed")
    anomalies = JsonLog(output_dir / "anomalous_images.json", "anomalous")
    pending = build_pending_shards(data_root, min_worker, max_worker, progress,
                                   shard_stride, shard_index)
    log(f"Pending shards: {len(pending)}")
    if not pending:
        log("Nothing pending. Exiting.")
        return

    embedder = JinaV4ImageEmbedder(device, batch_size, model_dir, num_workers,
                                   task, truncate_dim)
    prefetcher = TarPrefetcher(queue_size=max(1, prefetch_shards))
    prefetcher.start(pending)

    start_t = time.time()
    processed = 0
    run_images = 0
    hit_limit = False
    stream_done = False

    while not stream_done:
        if progress.total_images >= max_images:
            hit_limit = True
            log(f"Soft limit reached ({progress.total_images:,} >= {max_images:,})")
            break

        groups: List[Dict] = []
        projected = progress.total_images
        while len(groups) < shards_per_loader:
            item = prefetcher.get_next()
            if item is None:
                stream_done = True
                break
            worker_name, tar_name, tar_key, items, err = item
            if err:
                log(f"ERROR loading {tar_key}: {err}")
                continue
            groups.append({"worker_name": worker_name, "tar_name": tar_name,
                           "tar_key": tar_key, "items": items})
            projected += len(items)
            if projected >= max_images:
                break

        if not groups:
            break

        t0 = time.time()
        results = embedder.encode_groups(groups)
        span = time.time() - t0
        batch_imgs = sum(len(r[0]) for r in results)

        for g, (names, vecs, failed, anom) in zip(groups, results):
            if len(names) + len(failed) != len(g["items"]):
                log(f"  WARNING {g['tar_key']}: accounting mismatch (found "
                    f"{len(g['items'])}, embedded {len(names)}, failed {len(failed)})")
            if failed:
                log(f"  {len(failed)} image(s) failed in {g['tar_key']} -> failed_images.json")
                failures.add([{"image_name": n, "reason": r} for n, r in failed])
            if anom:
                log(f"  {len(anom)} anomalous image(s) in {g['tar_key']} -> anomalous_images.json")
                anomalies.add(anom)

            write_shard(output_dir, g["worker_name"], g["tar_name"], names, vecs)
            progress.mark(g["tar_key"], len(names), len(failed))
            processed += 1
            run_images += len(names)

        progress.save()
        failures.save()
        anomalies.save()

        elapsed = time.time() - start_t
        rate = run_images / elapsed if elapsed > 0 else 0.0
        log(f"[{processed}/{len(pending)}] {len(groups)} shards "
            f"({groups[0]['tar_key']} .. {groups[-1]['tar_key'].split('/')[-1]}): "
            f"{batch_imgs:,} in {span:.1f}s ({batch_imgs/span if span else 0:.0f}/s) | "
            f"total: {progress.total_images:,}/{max_images:,} | run avg: {rate:.0f}/s")

        del groups, results
        gc.collect()

    prefetcher.stop()
    failures.save()
    anomalies.save()

    total_time = time.time() - start_t
    summary = {
        "finished_at": datetime.now().isoformat(),
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "worker_range": [min_worker, max_worker],
        "target_images": max_images,
        "total_images": progress.total_images,
        "total_failed": progress.total_failed,
        "total_anomalous": anomalies.count(),
        "overshoot": max(0, progress.total_images - max_images),
        "stopped_on_limit": hit_limit,
        "completed_shards": len(progress.completed_tars),
        "processed_shards_this_run": processed,
        "duration": str(timedelta(seconds=int(total_time))),
        "device": device,
        "modality": "image",
        "model": "jinaai/jina-embeddings-v4",
        "model_dir": str(model_dir),
        "task_label": task,
        "return_multivector": False,
        "truncate_dim": truncate_dim,
        "image_resolution": "native (Qwen2.5-VL, no resize)",
        "embedding_dim": embedder.dim,
        "dtype": "float32",
        "normalised": True,
        "decode_workers": num_workers,
        "shards_per_loader": shards_per_loader,
        "shard_stride": shard_stride,
        "shard_index": shard_index,
        "limit_semantics": "soft - the shard in progress always finishes",
        "output_format": "<worker>/<shard>.npy + <worker>/<shard>.names.json",
    }
    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    log("=" * 70)
    log(f"DONE | total embeddings: {progress.total_images:,}")
    log(f"Shards written      : {len(progress.completed_tars):,}")
    log(f"Failed images       : {progress.total_failed:,}")
    log(f"Anomalous images    : {anomalies.count():,}  (recorded, still embedded)")
    if hit_limit:
        log(f"Overshoot past limit: {max(0, progress.total_images - max_images):,}")
    log("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="jina-embeddings-v4 image embeddings from TAR shards."
    )
    parser.add_argument("--data-root", type=str, required=True,
                        help="Path containing worker_* folders with .tar shards")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model-dir", type=str, default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--min-worker", type=int, default=0)
    parser.add_argument("--max-worker", type=int, default=20)
    parser.add_argument("--max-images", type=int, default=DEFAULT_MAX_IMAGES,
                        help="Soft limit: the shard in progress still finishes")
    parser.add_argument("--task", type=str, default=DEFAULT_TASK,
                        help="LoRA adapter to apply: retrieval / text-matching / "
                             "code. MUST match the text run.")
    parser.add_argument("--truncate-dim", type=int, default=None,
                        help="Matryoshka size (128/256/512/1024/2048). Default "
                             "keeps the full 2048.")
    # 3.75B params but images stay at native ~256px, so the sequence is short.
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--prefetch-shards", type=int, default=2)
    parser.add_argument("--shard-stride", type=int, default=1,
                        help="Split shards across N processes (e.g. one per GPU).")
    parser.add_argument("--shard-index", type=int, default=0,
                        help="Which slice this process takes, 0..stride-1.")
    parser.add_argument("--shards-per-loader", type=int, default=8,
                        help="Shards shared per DataLoader.")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Processes decoding JPEGs in parallel.")
    parser.add_argument("--torch-threads", type=int, default=2,
                        help="Intra-op threads PER process. Left at the default "
                             "PyTorch picks the core count (128 here), and several "
                             "processes then spin-wait against each other.")
    parser.add_argument("--device", type=str,
                        default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.set_num_threads(max(1, args.torch_threads))

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"data-root not found: {data_root}")
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"model-dir not found: {model_dir}")

    log("=" * 70)
    log("jina-embeddings-v4 IMAGE embedding generation")
    log(f"Data root       : {data_root}")
    log(f"Output dir      : {args.output_dir}")
    log(f"Workers         : worker_{args.min_worker}..worker_{args.max_worker}")
    log(f"Target images   : {args.max_images:,} (soft)")
    log(f"Task adapter    : {args.task}   truncate_dim: {args.truncate_dim or 'full'}")
    log(f"Batch size      : {args.batch_size}")
    log(f"Decode workers  : {args.num_workers}   torch threads: {args.torch_threads}")
    log(f"Device          : {args.device}")
    log("=" * 70)

    run_pipeline(
        data_root=data_root,
        output_dir=Path(args.output_dir),
        model_dir=model_dir,
        min_worker=args.min_worker,
        max_worker=args.max_worker,
        max_images=args.max_images,
        batch_size=args.batch_size,
        device=args.device,
        prefetch_shards=args.prefetch_shards,
        num_workers=args.num_workers,
        shards_per_loader=args.shards_per_loader,
        shard_stride=args.shard_stride,
        shard_index=args.shard_index,
        task=args.task,
        truncate_dim=args.truncate_dim,
    )


if __name__ == "__main__":
    main()
