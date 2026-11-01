#!/usr/bin/env python3
"""
OpenCLIP ViT-L-14 / laion2b_s32b_b82k IMAGE embedding generation from TAR shards.

Weights load from the local models/ directory - this never touches the network,
which matters because Pragya compute nodes have no proxy route.

NORMALIZATION (--image-norm)
    checkpoint  mean = std = (0.5, 0.5, 0.5), what this checkpoint was trained
                with (its open_clip_config.json preprocess_cfg). The default.
    openai      the OpenAI CLIP mean/std. create_model_and_transforms with a
                FILE path as `pretrained` never reads the tag table or the
                config, so the Normalize step keeps open_clip's OpenAI default.
                The first run (2026-09-02) got this; the option exists only to
                reproduce it.

    Only the Normalize step differs; resize, crop and everything else are the
    transform open_clip builds either way. Measured on 20,205 B queries
    (audit/openclip_imagenorm_check.py, 2026-10-11): mean cosine 0.929 between
    the two, about a third of each image top-100 list changes, image->text
    recall@1 83.0% (openai) vs 86.8% (checkpoint). The choice is recorded in
    progress.json and run_summary.json, and a run refuses an output directory
    whose vectors were made with the other one.

SHARD LIST (--shard-list)
    A file of tar keys (worker_N/shard_NNNNN.tar), one per line. Only those
    shards are embedded -- applied before --shard-stride, so N strided parts
    split exactly that list. Used to cover the same 3,236 shards as the first
    run without the soft limit.

OUTPUT - one file pair per shard, mirroring the dataset layout
    result_v2/embeddings_image/openclip_vit_l14_laion2b_norm05/   --image-norm checkpoint
    result/embeddings_image/openclip_vit_l14_laion2b/             --image-norm openai
        worker_0/
            shard_00000.npy            float32 [N, 768], L2-normalised
            shard_00000.names.json     N names, same row order as the .npy
            shard_00001.npy
            shard_00001.names.json
            ...
        worker_1/
            ...
        progress.json                  resume state
        failed_images.json             every image that did NOT make it in
        run_summary.json

    Row i of shard_XXXXX.npy corresponds to entry i of shard_XXXXX.names.json.
    Names are fully qualified:
        worker_0/shard_00000/000000000.jpg

    The qualification is NOT cosmetic. Workers reuse the same id space: the id
    001000002 exists in worker_0 AND worker_3 and they are different images
    (different URLs, different captions). A bare filename would collide.

STOPPING (soft limit)
    Shards are consumed in order from the very first one, every image in each,
    nothing sampled or skipped. Before starting a shard the running total is
    checked against --max-images (default 8,000,000); if it is already reached
    the run stops. The shard in progress is always finished, so the final total
    slightly EXCEEDS the limit and every shard on disk is complete.

ON "MISSING" FILE NUMBERS
    Do not expect contiguous numbering. The filename number is the row index of
    the URL in LAION's parquet, assigned before any download: shard k owns rows
    [k*10000, k*10000+9999]. Only URLs that still resolved survive - roughly
    13-17% in early shards and ~40% in later ones. A shard holding 1317 images
    spanning ids 0..9990 is COMPLETE, not damaged; the holes are URLs that were
    dead by download time. What is verified instead, per shard: every jpg found
    is either embedded or recorded in failed_images.json, so nothing is ever
    dropped silently.

RUN THIS IN A PBS JOB, not on the login node.
"""

import os
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

# Never reach for the network: compute nodes have no proxy.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
import open_clip

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_KEY = "openclip_vit_l14_laion2b"
ARCH = "ViT-L-14"
DEFAULT_WEIGHTS = REPO_ROOT / "models/openclip_vit_l14_laion2b_s32b_b82k/open_clip_pytorch_model.safetensors"
DEFAULT_MAX_IMAGES = 8_000_000

# One output directory per normalization, so the two kinds of vector never
# share a tree. "openai" is the first run's directory and must not be written
# again; "checkpoint" lives in result_v2/, the second run's results tree.
NORMS = ("checkpoint", "openai")
DEFAULT_NORM = "checkpoint"
OUTPUT_DIRS = {
    "checkpoint": REPO_ROOT / "result_v2/embeddings_image" / f"{MODEL_KEY}_norm05",
    "openai": REPO_ROOT / "result/embeddings_image" / MODEL_KEY,
}
DEFAULT_OUTPUT_DIR = OUTPUT_DIRS[DEFAULT_NORM]


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


# An image is "anomalous" if it is too degenerate for a 224/336px embedding to
# mean anything: a sliver, or a single pixel. LAION contains these because the
# source URL served a tracking pixel or an error placeholder instead of the
# picture. They are RECORDED, not dropped - the vectors are still written, so
# nothing silently disappears from the output.
MIN_DIM = 16          # any side below this
MAX_ASPECT = 20.0     # long:short ratio above this


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
    Resume state. Shards are atomic now - a shard is either fully written or
    not written at all - so this is just a completed set plus counters.
    """

    def __init__(self, output_dir: Path, image_norm: str):
        self.output_dir = output_dir
        self.progress_path = output_dir / "progress.json"
        self.lock = threading.Lock()
        self.started_at = time.time()
        self.completed_tars: Set[str] = set()
        self.total_images = 0
        self.total_failed = 0
        self.image_norm = image_norm
        # What made the shards already on disk. The first run's progress.json
        # predates this field, and that run used the OpenAI mean/std.
        self.recorded_norm: Optional[str] = None
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
            self.recorded_norm = data.get("image_norm", "openai")
            log(
                f"Resume: {len(self.completed_tars)} shards done, "
                f"{self.total_images:,} embeddings"
            )
        except Exception as e:
            log(f"WARNING: could not load progress.json: {e}")

    def save(self) -> None:
        with self.lock:
            data = {
                "completed_tars": sorted(self.completed_tars),
                "total_images": self.total_images,
                "total_failed": self.total_failed,
                "image_norm": self.image_norm,
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
    # np.save appends ".npy" unless the path already ends in it, so the temp
    # name must end in .npy or the rename target would be wrong.
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

            # Deterministic row order. Tar archive order is arbitrary (a shard
            # may list 000005093.json before 000003010.json). Ids are uniformly
            # zero-padded to 9 digits, so a lexicographic sort is also numeric.
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

    Building a DataLoader per shard cost ~34 ms per worker in fork/join - at 48
    workers that was ~1.6 s of a 7.5 s shard, over 20% of wall time, and it grew
    with worker count, which is why adding workers scaled so badly. Grouping
    shards amortises that spawn across all of them.

    Output stays strictly shard-aligned: every item carries its flat index, and
    the caller maps it back through `index` to (group, position) before writing,
    so each shard still gets its own .npy in the original row order.

    A failed decode yields a zero tensor with ok=False rather than raising, so
    the batch keeps its shape and index->name alignment never shifts.
    """

    def __init__(self, groups, transform, blank):
        self.groups = groups
        self.transform = transform
        self.blank = blank
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
                # Source dimensions travel with the row so the main process can
                # flag degenerate images without re-reading or decoding them.
                w, h = im.size
                return self.transform(im.convert("RGB")), k, True, w, h
        except Exception:
            return self.blank, k, False, 0, 0


def group_image_name(g: Dict, ii: int) -> str:
    """Fully-qualified name: worker_0/shard_00000/000000000.jpg"""
    return (f"{g['worker_name']}/{Path(g['tar_name']).stem}/"
            f"{Path(g['items'][ii][0]).name}")


class OpenClipEmbedder:
    # image_norm defaults to "openai" HERE, unlike the CLI, so that callers
    # written before the option existed (audit/openclip_imagenorm_check.py) keep
    # getting the first run's transform and still reproduce its vectors.
    def __init__(self, device: str, batch_size: int, use_amp: bool, weights: Path,
                 num_workers: int = 8, image_norm: str = "openai"):
        self.device = device
        self.batch_size = batch_size
        self.use_amp = use_amp
        self.num_workers = num_workers

        if str(device).startswith("cuda"):
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        log(f"Loading OpenCLIP {ARCH} from {weights}")
        self.model, _, self.transform = open_clip.create_model_and_transforms(
            ARCH, pretrained=str(weights), device=device
        )
        self.model.eval()
        self.norm_info = self._set_norm(image_norm, Path(weights))
        self.dim = int(self.model.text_projection.shape[1])
        # Shape of one preprocessed image, for the decode-failure placeholder.
        px = self.model.visual.image_size
        px = px if isinstance(px, (tuple, list)) else (px, px)
        self.blank = torch.zeros(3, px[0], px[1])
        log(f"Model loaded (embedding dim {self.dim}, input {px[0]}x{px[1]})")

    def _set_norm(self, name: str, weights: Path) -> Dict:
        """
        Set the Normalize step of the eval transform, and nothing else.

        "checkpoint" takes mean/std from the checkpoint's open_clip_config.json;
        "openai" leaves what create_model_and_transforms built. Exactly one
        Normalize must exist, or the transform is not the one this was written for.
        """
        from torchvision.transforms import Normalize
        if name not in NORMS:
            raise ValueError(f"image_norm must be one of {NORMS}, got {name!r}")
        norms = [t for t in getattr(self.transform, "transforms", [])
                 if isinstance(t, Normalize)]
        if len(norms) != 1:
            raise RuntimeError(f"expected one Normalize in the transform, got "
                               f"{len(norms)}: {self.transform}")
        if name == "checkpoint":
            cfg_path = weights.parent / "open_clip_config.json"
            cfg = json.load(open(cfg_path, encoding="utf-8"))["preprocess_cfg"]
            norms[0].mean = tuple(float(x) for x in cfg["mean"])
            norms[0].std = tuple(float(x) for x in cfg["std"])
        info = {"image_norm": name,
                "image_mean": [float(x) for x in norms[0].mean],
                "image_std": [float(x) for x in norms[0].std],
                "transform": repr(self.transform)}
        log(f"Image transform ({name}): {info['transform']}")
        return info

    def _autocast(self):
        if self.use_amp and str(self.device).startswith("cuda"):
            return torch.autocast("cuda", dtype=torch.float16)
        return torch.autocast("cpu", enabled=False)

    def _encode_batch(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode one already-decoded batch, halving it on OOM.

        Splitting the tensor is better than shrinking a global batch size and
        retrying the decode: the images are already in memory, so no work is
        thrown away and nothing is dropped.
        """
        try:
            xb = x.to(self.device, non_blocking=True)
            with self._autocast():
                feats = self.model.encode_image(xb)
            out = l2norm(feats.float()).cpu()
            del xb, feats
            return out
        except torch.cuda.OutOfMemoryError:
            if str(self.device).startswith("cuda"):
                torch.cuda.empty_cache()
            if x.shape[0] == 1:
                raise
            half = x.shape[0] // 2
            log(f"  OOM: splitting batch of {x.shape[0]} -> {half}")
            return torch.cat([self._encode_batch(x[:half]),
                              self._encode_batch(x[half:])], dim=0)

    @torch.inference_mode()
    def encode_groups(self, groups: List[Dict]) -> List[Tuple[List[str], np.ndarray, List[Tuple[str, str]]]]:
        """
        Encode several shards under ONE DataLoader, returning a per-shard
        (names, vectors[float32, L2-normalised], failed) triple in the same
        order as `groups`.

        For every shard, len(items) == len(names) + len(failed) still holds.
        """
        empty = [([], np.zeros((0, self.dim), dtype=np.float32), [], []) for _ in groups]
        if not groups or all(not g["items"] for g in groups):
            return empty

        ds = MultiShardDataset(groups, self.transform, self.blank)
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
            pin_memory=str(self.device).startswith("cuda"),
            prefetch_factor=4 if self.num_workers > 0 else None,
        )

        for x, ks, oks, ws, hs in loader:
            oks = oks.bool()
            for k in ks[~oks].tolist():
                gi, ii = ds.index[k]
                failed_by[gi].append((group_image_name(groups[gi], ii), "decode"))

            for k, w, h in zip(ks.tolist(), ws.tolist(), hs.tolist()):
                reason = classify_anomaly(w, h)
                if reason:
                    gi, ii = ds.index[k]
                    anom_by[gi].append({"image_name": group_image_name(groups[gi], ii),
                                        "width": w, "height": h, "reason": reason})

            if not bool(oks.any()):
                continue

            feats = self._encode_batch(x[oks]).numpy().astype(np.float32)
            for row, k in enumerate(ks[oks].tolist()):
                gi, ii = ds.index[k]
                names_by[gi].append(group_image_name(groups[gi], ii))
                vecs_by[gi].append(feats[row])
            del feats

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
    shard_list: Optional[Set[str]] = None,
) -> List[Tuple[str, Path]]:
    """
    Shards still to do, optionally only those in shard_list, then every Nth one.

    shard_stride/shard_index let several processes split ONE worker across
    several GPUs: run with the same range and stride N, indices 0..N-1, and the
    partitions are disjoint and cover everything. The counter advances over all
    shards in range regardless of completion, so a resumed run lands on exactly
    the same partition as the original.
    """
    pending: List[Tuple[str, Path]] = []
    seen = 0
    for _, worker_name, worker_path in discover_workers(data_root, min_worker, max_worker):
        for tar_path in discover_tars(worker_path):
            tar_key = f"{worker_name}/{tar_path.name}"
            # The list filters BEFORE the stride, so the parts split the list.
            if shard_list is not None and tar_key not in shard_list:
                continue
            mine = (seen % shard_stride) == shard_index
            seen += 1
            if not mine:
                continue
            if not progress.is_completed(tar_key):
                pending.append((worker_name, tar_path))
    return pending


def run_pipeline(
    data_root: Path,
    output_dir: Path,
    weights: Path,
    min_worker: int,
    max_worker: int,
    max_images: int,
    batch_size: int,
    device: str,
    prefetch_shards: int,
    use_amp: bool,
    num_workers: int,
    shards_per_loader: int,
    shard_stride: int,
    shard_index: int,
    image_norm: str,
    shard_list: Optional[Set[str]] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    progress = ProgressManager(output_dir, image_norm)
    # A resumed run appends shards to what is there, so the two kinds of vector
    # would end up in one tree. Shards without a progress.json are refused too.
    on_disk = any(output_dir.glob("worker_*/shard_*.npy"))
    if on_disk and progress.recorded_norm != image_norm:
        raise SystemExit(
            f"{output_dir} already holds shards made with image_norm "
            f"{progress.recorded_norm!r}; this run uses {image_norm!r}. Write to "
            f"another directory (default for {image_norm!r}: {OUTPUT_DIRS[image_norm]}).")
    if progress.total_images >= max_images:
        log(f"Target already reached: {progress.total_images:,}/{max_images:,}")
        return

    failures = JsonLog(output_dir / "failed_images.json", "failed")
    anomalies = JsonLog(output_dir / "anomalous_images.json", "anomalous")
    pending = build_pending_shards(data_root, min_worker, max_worker, progress,
                                   shard_stride, shard_index, shard_list)
    log(f"Pending shards: {len(pending)}")
    if not pending:
        log("Nothing pending. Exiting.")
        return

    embedder = OpenClipEmbedder(device, batch_size, use_amp, weights, num_workers,
                                image_norm=image_norm)
    prefetcher = TarPrefetcher(queue_size=max(1, prefetch_shards))
    prefetcher.start(pending)

    start_t = time.time()
    processed = 0
    run_images = 0          # this run only - progress.total_images includes resumed work
    hit_limit = False
    stream_done = False

    while not stream_done:
        if progress.total_images >= max_images:
            hit_limit = True
            log(f"Soft limit reached ({progress.total_images:,} >= {max_images:,})")
            break

        # Collect a group of shards to share one DataLoader. Accumulation stops
        # early once the projected total would pass the limit, so the overshoot
        # stays at most one shard - same guarantee as before the grouping.
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
                log(
                    f"  WARNING {g['tar_key']}: accounting mismatch (found "
                    f"{len(g['items'])}, embedded {len(names)}, failed {len(failed)})"
                )
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
        log(
            f"[{processed}/{len(pending)}] {len(groups)} shards "
            f"({groups[0]['tar_key']} .. {groups[-1]['tar_key'].split('/')[-1]}): "
            f"{batch_imgs:,} in {span:.1f}s ({batch_imgs/span if span else 0:.0f}/s) | "
            f"total: {progress.total_images:,}/{max_images:,} | run avg: {rate:.0f}/s"
        )

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
        "model": f"{ARCH}/laion2b_s32b_b82k",
        "weights": str(weights),
        "embedding_dim": embedder.dim,
        "dtype": "float32",
        "normalised": True,
        **embedder.norm_info,
        "shard_list_size": None if shard_list is None else len(shard_list),
        "batch_size": batch_size,
        "amp": "float16 autocast" if (use_amp and str(device).startswith("cuda")) else "off",
        "torch_version": torch.__version__,
        "open_clip_version": getattr(open_clip, "__version__", None),
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
        description="OpenCLIP ViT-L-14 laion2b_s32b_b82k image embeddings from TAR shards."
    )
    parser.add_argument("--data-root", type=str, required=True,
                        help="Path containing worker_* folders with .tar shards")
    parser.add_argument("--image-norm", choices=NORMS, default=DEFAULT_NORM,
                        help="checkpoint: mean = std = 0.5, as this checkpoint was "
                             "trained (default). openai: the first run's OpenAI "
                             "mean/std, only to reproduce result/embeddings_image/"
                             + MODEL_KEY)
    parser.add_argument("--output-dir", type=str, default=None,
                        help="default: the normalization's own directory, "
                             + ", ".join(f"{k} -> {v}" for k, v in OUTPUT_DIRS.items()))
    parser.add_argument("--shard-list", type=str, default=None,
                        help="file of tar keys (worker_N/shard_NNNNN.tar), one per "
                             "line; only these shards are embedded")
    parser.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--min-worker", type=int, default=0)
    parser.add_argument("--max-worker", type=int, default=20)
    parser.add_argument("--max-images", type=int, default=DEFAULT_MAX_IMAGES,
                        help="Soft limit: the shard in progress still finishes")
    parser.add_argument("--batch-size", type=int, default=1536)
    parser.add_argument("--prefetch-shards", type=int, default=2)
    parser.add_argument("--shard-stride", type=int, default=1,
                        help="Split shards across N processes (e.g. one per GPU).")
    parser.add_argument("--shard-index", type=int, default=0,
                        help="Which slice this process takes, 0..stride-1.")
    parser.add_argument("--shards-per-loader", type=int, default=8,
                        help="Shards shared per DataLoader. Amortises the ~34ms "
                             "per-worker fork/join cost across several shards.")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Processes decoding JPEGs in parallel. This is the "
                             "throughput knob - decode, not the GPU, is the limit.")
    parser.add_argument("--torch-threads", type=int, default=2,
                        help="Intra-op threads PER process. Left at the default "
                             "PyTorch picks the core count (128 here), and several "
                             "processes then spin-wait against each other.")
    parser.add_argument("--device", type=str,
                        default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    args = parser.parse_args()

    torch.set_num_threads(max(1, args.torch_threads))

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"data-root not found: {data_root}")
    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"weights not found: {weights}")

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIRS[args.image_norm]
    shard_list = None
    if args.shard_list:
        shard_list = {ln.strip() for ln in open(args.shard_list) if ln.strip()}

    log("=" * 70)
    log(f"OpenCLIP {ARCH} laion2b_s32b_b82k IMAGE embedding generation")
    log(f"Data root       : {data_root}")
    log(f"Output dir      : {output_dir}")
    log(f"Image norm      : {args.image_norm}")
    log(f"Shard list      : {args.shard_list or 'none'}"
        + (f" ({len(shard_list):,} shards)" if shard_list else ""))
    log(f"Workers         : worker_{args.min_worker}..worker_{args.max_worker}")
    log(f"Target images   : {args.max_images:,} (soft)")
    log(f"Batch size      : {args.batch_size}")
    log(f"Decode workers  : {args.num_workers}   torch threads: {args.torch_threads}")
    log(f"Shards/loader   : {args.shards_per_loader}")
    log(f"Device          : {args.device}   AMP: {not args.no_amp}")
    log("=" * 70)

    run_pipeline(
        data_root=data_root,
        output_dir=output_dir,
        weights=weights,
        min_worker=args.min_worker,
        max_worker=args.max_worker,
        max_images=args.max_images,
        batch_size=args.batch_size,
        device=args.device,
        prefetch_shards=args.prefetch_shards,
        use_amp=(not args.no_amp),
        num_workers=args.num_workers,
        shards_per_loader=args.shards_per_loader,
        shard_stride=args.shard_stride,
        shard_index=args.shard_index,
        image_norm=args.image_norm,
        shard_list=shard_list,
    )


if __name__ == "__main__":
    main()
