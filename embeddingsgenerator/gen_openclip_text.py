#!/usr/bin/env python3
"""
OpenCLIP ViT-L-14 / laion2b_s32b_b82k TEXT embedding generation from TAR shards.

Weights load from the local models/ directory - this never touches the network,
which matters because Pragya compute nodes have no proxy route.

WHERE THE TEXT COMES FROM
    Every image in a shard has a sibling metadata file written by img2dataset:

        shard_00000/000000000.jpg     the image
        shard_00000/000000000.json    {"url", "caption", "width", "height", "sim"}

    This script reads the .json members and embeds the "caption" field. The
    image counterpart (gen_openclip.py) reads the .jpg members of the same
    shards, so the two runs cover exactly the same LAION pairs.

OUTPUT - one file pair per shard, mirroring the dataset layout
    result/embeddings_text/openclip_vit_l14_laion2b/
        worker_0/
            shard_00000.npy            float32 [N, 768], L2-normalised
            shard_00000.names.json     N names, same row order as the .npy
            ...
        worker_1/
            ...
        progress.json                  resume state
        failed_captions.json           every caption that did NOT make it in
        anomalous_captions.json        degenerate captions (embedded, flagged)
        run_summary.json

    Row i of shard_XXXXX.npy corresponds to entry i of shard_XXXXX.names.json.
    Names are fully qualified:
        worker_0/shard_00000/000000000.json

    The qualification is NOT cosmetic. Workers reuse the same id space: the id
    001000002 exists in worker_0 AND worker_3 and they are different records
    (different URLs, different captions). A bare filename would collide.

ROW ALIGNMENT WITH THE IMAGE EMBEDDINGS
    Both scripts sort a shard's members by filename before embedding, and the
    ids are zero-padded to 9 digits, so the two orders are the same sequence of
    ids. Row i of the text .npy is therefore the caption of row i of the image
    .npy - provided neither side dropped a record. Nothing is dropped silently:
    per shard, found == embedded + failed is asserted, and any failure lands in
    failed_captions.json. Damaged records - a null caption, a zero-byte .json -
    do NOT fail: they are embedded as the empty string and flagged in
    anomalous_captions.json, so the row still exists. Only the tokeniser raising
    could drop one. To confirm alignment on disk, compare name stems:

        [n.rsplit(".", 1)[0] for n in text_names] == \\
        [n.rsplit(".", 1)[0] for n in image_names]

STOPPING (soft limit)
    Shards are consumed in order from the very first one, every caption in each,
    nothing sampled or skipped. Before starting a shard the running total is
    checked against --max-texts (default 8,000,000); if it is already reached
    the run stops. The shard in progress is always finished, so the final total
    slightly EXCEEDS the limit and every shard on disk is complete.

ON "MISSING" FILE NUMBERS
    Do not expect contiguous numbering. The filename number is the row index of
    the URL in LAION's parquet, assigned before any download: shard k owns rows
    [k*10000, k*10000+9999]. Only URLs that still resolved survive - roughly
    13-17% in early shards and ~40% in later ones. A shard holding 1317 records
    spanning ids 0..9990 is COMPLETE, not damaged; the holes are URLs that were
    dead by download time.

NOTE ON THE API
    open_clip takes tokens, not strings: get_tokenizer(ARCH) returns a
    SimpleTokenizer that pads every caption to the model's context length, and
    encode_text returns the projected embedding directly - there is no
    attention mask and no output wrapper.

    The tokeniser reads open_clip's own bundled BPE vocabulary, so it needs no
    network either. The vocabulary is OpenAI's; laion2b models reuse it.

    There is no --max-text-len here, unlike the two transformers-based
    generators. open_clip adds the positional embedding un-sliced
    (x + self.positional_embedding), so the sequence must be exactly
    model.context_length - 77 - and a shorter window would not broadcast.

RUN THIS IN A PBS JOB, not on the login node.
"""

import os
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
import open_clip

CAPTION_EXTENSION = ".json"
CAPTION_KEY = "caption"

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_KEY = "openclip_vit_l14_laion2b"
ARCH = "ViT-L-14"
DEFAULT_WEIGHTS = REPO_ROOT / "models/openclip_vit_l14_laion2b_s32b_b82k/open_clip_pytorch_model.safetensors"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "result/embeddings_text" / MODEL_KEY
DEFAULT_MAX_TEXTS = 8_000_000


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def l2norm(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-12)


def is_caption_file(name: str) -> bool:
    return Path(name).suffix.lower() == CAPTION_EXTENSION


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


# A caption is "anomalous" only if there is nothing to encode: absent
# ("caption": null), empty, or a stray character or two. LAION is full of these
# because alt text is often missing entirely.
#
# A LONG caption is deliberately NOT an anomaly. Every model here has a fixed
# context window and clipping the tail is normal operation for the whole family,
# not damage - and flagging it buried the ~440 records that genuinely carry no
# text under ~76,000 that carry plenty.
#
# Anomalies are RECORDED, not dropped - the vectors are still written, so
# nothing silently disappears from the output and row alignment with the image
# embeddings is preserved. A null caption is embedded as the empty string; the
# image side embedded that pair, so the row has to exist here too.
MIN_CHARS = 3


def classify_anomaly(text) -> Optional[str]:
    if text is None:
        return "missing_caption"
    if not isinstance(text, str):
        return f"non_string_caption_{type(text).__name__}"
    s = text.strip()
    if not s:
        return "empty_caption"
    if len(s) < MIN_CHARS:
        return f"tiny_caption_{len(s)}chars"
    return None


class JsonLog:
    """Append-only JSON sidecar (failed / anomalous captions)."""

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

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.progress_path = output_dir / "progress.json"
        self.lock = threading.Lock()
        self.started_at = time.time()
        self.completed_tars: Set[str] = set()
        self.total_texts = 0
        self.total_failed = 0
        self.load()

    def load(self) -> None:
        if not self.progress_path.exists():
            return
        try:
            with open(self.progress_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.completed_tars = set(data.get("completed_tars", []))
            self.total_texts = int(data.get("total_texts", 0))
            self.total_failed = int(data.get("total_failed", 0))
            log(
                f"Resume: {len(self.completed_tars)} shards done, "
                f"{self.total_texts:,} embeddings"
            )
        except Exception as e:
            log(f"WARNING: could not load progress.json: {e}")

    def save(self) -> None:
        with self.lock:
            data = {
                "completed_tars": sorted(self.completed_tars),
                "total_texts": self.total_texts,
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
            self.total_texts += added
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
    Loads caption bytes from shards in a background thread.

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
                        if not member.isfile() or not is_caption_file(member.name):
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

            # Deterministic row order, and the SAME order the image run used.
            # Tar archive order is arbitrary (a shard may list 000005093.json
            # before 000003010.json). Ids are uniformly zero-padded to 9 digits,
            # so a lexicographic sort is also numeric.
            items.sort(key=lambda t: Path(t[0]).name)
            self.queue.put((worker_name, tar_path.name, tar_key, items, err))

        self.queue.put(None)

    def get_next(self):
        return self.queue.get()

    def stop(self) -> None:
        self.stop_event.set()


class OpenClipTextTransform:
    """
    Adapts open_clip's tokeniser to the single-caption call a Dataset makes.

    It only accepts a list, and returns [1, context_length], so the wrapper is
    a list-wrap and an unwrap. Every row comes out padded to the same width,
    which is what the model requires and what keeps a caption's vector
    independent of whichever captions shared its batch.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, text: str):
        return self.tokenizer([text])[0]


class MultiShardDataset(Dataset):
    """
    Parses and tokenises SEVERAL shards' captions within one DataLoader lifetime.

    Building a DataLoader per shard cost ~34 ms per worker in fork/join - at 48
    workers that was ~1.6 s of a 7.5 s shard, over 20% of wall time, and it grew
    with worker count, which is why adding workers scaled so badly. Grouping
    shards amortises that spawn across all of them.

    Output stays strictly shard-aligned: every item carries its flat index, and
    the caller maps it back through `index` to (group, position) before writing,
    so each shard still gets its own .npy in the original row order.

    A caption that cannot be read yields a zero token row with ok=False rather
    than raising, so the batch keeps its shape and index->name alignment never
    shifts.
    """

    def __init__(self, groups, transform, blank_ids):
        self.groups = groups
        self.transform = transform
        self.blank_ids = blank_ids
        self.index: List[Tuple[int, int]] = []
        for gi, g in enumerate(groups):
            for ii in range(len(g["items"])):
                self.index.append((gi, ii))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, k: int):
        gi, ii = self.index[k]
        _, raw = self.groups[gi]["items"][ii]
        # A record can be unreadable (img2dataset sometimes writes a zero-byte
        # .json) or carry "caption": null. Neither may drop the row: the image
        # side embedded this pair, so a missing row would shift every later row
        # of the shard out of alignment. Both are embedded as the empty string
        # and flagged, exactly as a 1x1 image is on the image side.
        try:
            meta = json.loads(raw.decode("utf-8"))
            caption = meta.get(CAPTION_KEY) if isinstance(meta, dict) else None
            reason = classify_anomaly(caption) or ""
        except Exception:
            caption, reason = None, "unparseable_json"
        text = caption if isinstance(caption, str) else ""
        try:
            ids = self.transform(text)
        except Exception:
            # The tokeniser raising is the ONLY thing that can still lose a row,
            # and it would be a bug rather than bad data - hence still a failure.
            return self.blank_ids, k, False, reason, 0
        return ids, k, True, reason, len(text.strip())


def group_caption_name(g: Dict, ii: int) -> str:
    """Fully-qualified name: worker_0/shard_00000/000000000.json"""
    return (f"{g['worker_name']}/{Path(g['tar_name']).stem}/"
            f"{Path(g['items'][ii][0]).name}")


class OpenClipTextEmbedder:
    def __init__(self, device: str, batch_size: int, use_amp: bool, weights: Path,
                 num_workers: int = 8):
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
        self.model, _, _ = open_clip.create_model_and_transforms(
            ARCH, pretrained=str(weights), device=device
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(ARCH)
        self.dim = int(self.model.text_projection.shape[1])
        self.max_text_len = int(self.model.context_length)
        log(f"Model loaded (context {self.max_text_len} tokens, embedding dim {self.dim})")

        self.transform = OpenClipTextTransform(self.tokenizer)
        # Shape of one tokenised caption, for the parse-failure placeholder.
        self.blank_ids = torch.zeros(self.max_text_len, dtype=torch.long)

    def _autocast(self):
        if self.use_amp and str(self.device).startswith("cuda"):
            return torch.autocast("cuda", dtype=torch.float16)
        return torch.autocast("cpu", enabled=False)

    def _encode_batch(self, ids: torch.Tensor) -> torch.Tensor:
        """
        Encode one already-tokenised batch, halving it on OOM.

        Splitting the tensor is better than shrinking a global batch size and
        retrying the tokenise: the captions are already in memory, so no work is
        thrown away and nothing is dropped.
        """
        try:
            i = ids.to(self.device, non_blocking=True)
            with self._autocast():
                feats = self.model.encode_text(i)
            out = l2norm(feats.float()).cpu()
            del i, feats
            return out
        except torch.cuda.OutOfMemoryError:
            if str(self.device).startswith("cuda"):
                torch.cuda.empty_cache()
            if ids.shape[0] == 1:
                raise
            half = ids.shape[0] // 2
            log(f"  OOM: splitting batch of {ids.shape[0]} -> {half}")
            return torch.cat([self._encode_batch(ids[:half]),
                              self._encode_batch(ids[half:])], dim=0)

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

        ds = MultiShardDataset(groups, self.transform, self.blank_ids)
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

        for ids, ks, oks, reasons, nchars in loader:
            oks = oks.bool()
            for k in ks[~oks].tolist():
                gi, ii = ds.index[k]
                failed_by[gi].append((group_caption_name(groups[gi], ii), "parse"))

            for k, reason, n in zip(ks.tolist(), reasons, nchars.tolist()):
                if reason:
                    gi, ii = ds.index[k]
                    anom_by[gi].append({"caption_name": group_caption_name(groups[gi], ii),
                                        "chars": n, "reason": reason})

            if not bool(oks.any()):
                continue

            feats = self._encode_batch(ids[oks]).numpy().astype(np.float32)
            for row, k in enumerate(ks[oks].tolist()):
                gi, ii = ds.index[k]
                names_by[gi].append(group_caption_name(groups[gi], ii))
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
) -> List[Tuple[str, Path]]:
    """
    Shards still to do, optionally taking only every Nth one.

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
    weights: Path,
    min_worker: int,
    max_worker: int,
    max_texts: int,
    batch_size: int,
    device: str,
    prefetch_shards: int,
    use_amp: bool,
    num_workers: int,
    shards_per_loader: int,
    shard_stride: int,
    shard_index: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    progress = ProgressManager(output_dir)
    if progress.total_texts >= max_texts:
        log(f"Target already reached: {progress.total_texts:,}/{max_texts:,}")
        return

    failures = JsonLog(output_dir / "failed_captions.json", "failed")
    anomalies = JsonLog(output_dir / "anomalous_captions.json", "anomalous")
    pending = build_pending_shards(data_root, min_worker, max_worker, progress,
                                   shard_stride, shard_index)
    log(f"Pending shards: {len(pending)}")
    if not pending:
        log("Nothing pending. Exiting.")
        return

    embedder = OpenClipTextEmbedder(device, batch_size, use_amp, weights, num_workers)
    prefetcher = TarPrefetcher(queue_size=max(1, prefetch_shards))
    prefetcher.start(pending)

    start_t = time.time()
    processed = 0
    run_texts = 0           # this run only - progress.total_texts includes resumed work
    hit_limit = False
    stream_done = False

    while not stream_done:
        if progress.total_texts >= max_texts:
            hit_limit = True
            log(f"Soft limit reached ({progress.total_texts:,} >= {max_texts:,})")
            break

        # Collect a group of shards to share one DataLoader. Accumulation stops
        # early once the projected total would pass the limit, so the overshoot
        # stays at most one shard - same guarantee as before the grouping.
        groups: List[Dict] = []
        projected = progress.total_texts
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
            if projected >= max_texts:
                break

        if not groups:
            break

        t0 = time.time()
        results = embedder.encode_groups(groups)
        span = time.time() - t0
        batch_texts = sum(len(r[0]) for r in results)

        for g, (names, vecs, failed, anom) in zip(groups, results):
            if len(names) + len(failed) != len(g["items"]):
                log(
                    f"  WARNING {g['tar_key']}: accounting mismatch (found "
                    f"{len(g['items'])}, embedded {len(names)}, failed {len(failed)})"
                )
            if failed:
                log(f"  {len(failed)} caption(s) failed in {g['tar_key']} -> failed_captions.json")
                failures.add([{"caption_name": n, "reason": r} for n, r in failed])
            if anom:
                log(f"  {len(anom)} anomalous caption(s) in {g['tar_key']} -> anomalous_captions.json")
                anomalies.add(anom)

            write_shard(output_dir, g["worker_name"], g["tar_name"], names, vecs)
            progress.mark(g["tar_key"], len(names), len(failed))
            processed += 1
            run_texts += len(names)

        progress.save()
        failures.save()
        anomalies.save()

        elapsed = time.time() - start_t
        rate = run_texts / elapsed if elapsed > 0 else 0.0
        log(
            f"[{processed}/{len(pending)}] {len(groups)} shards "
            f"({groups[0]['tar_key']} .. {groups[-1]['tar_key'].split('/')[-1]}): "
            f"{batch_texts:,} in {span:.1f}s ({batch_texts/span if span else 0:.0f}/s) | "
            f"total: {progress.total_texts:,}/{max_texts:,} | run avg: {rate:.0f}/s"
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
        "target_texts": max_texts,
        "total_texts": progress.total_texts,
        "total_failed": progress.total_failed,
        "total_anomalous": anomalies.count(),
        "overshoot": max(0, progress.total_texts - max_texts),
        "stopped_on_limit": hit_limit,
        "completed_shards": len(progress.completed_tars),
        "processed_shards_this_run": processed,
        "duration": str(timedelta(seconds=int(total_time))),
        "device": device,
        "modality": "text",
        "model": f"{ARCH}/laion2b_s32b_b82k",
        "weights": str(weights),
        "caption_source": f"<id>.json / \"{CAPTION_KEY}\"",
        "max_text_len": embedder.max_text_len,
        "embedding_dim": embedder.dim,
        "dtype": "float32",
        "normalised": True,
        "tokenize_workers": num_workers,
        "shards_per_loader": shards_per_loader,
        "shard_stride": shard_stride,
        "shard_index": shard_index,
        "limit_semantics": "soft - the shard in progress always finishes",
        "output_format": "<worker>/<shard>.npy + <worker>/<shard>.names.json",
    }
    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    log("=" * 70)
    log(f"DONE | total embeddings: {progress.total_texts:,}")
    log(f"Shards written      : {len(progress.completed_tars):,}")
    log(f"Failed captions     : {progress.total_failed:,}")
    log(f"Anomalous captions  : {anomalies.count():,}  (recorded, still embedded)")
    if hit_limit:
        log(f"Overshoot past limit: {max(0, progress.total_texts - max_texts):,}")
    log("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="OpenCLIP ViT-L-14 laion2b_s32b_b82k text embeddings from TAR shards."
    )
    parser.add_argument("--data-root", type=str, required=True,
                        help="Path containing worker_* folders with .tar shards")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--min-worker", type=int, default=0)
    parser.add_argument("--max-worker", type=int, default=20)
    parser.add_argument("--max-texts", type=int, default=DEFAULT_MAX_TEXTS,
                        help="Soft limit: the shard in progress still finishes")
    # A 77-token caption is far cheaper than a 224px image, so the text tower is
    # the bottleneck only at much larger batches than the image run used.
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--prefetch-shards", type=int, default=2)
    parser.add_argument("--shard-stride", type=int, default=1,
                        help="Split shards across N processes (e.g. one per GPU).")
    parser.add_argument("--shard-index", type=int, default=0,
                        help="Which slice this process takes, 0..stride-1.")
    parser.add_argument("--shards-per-loader", type=int, default=8,
                        help="Shards shared per DataLoader. Amortises the ~34ms "
                             "per-worker fork/join cost across several shards.")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Processes tokenising captions in parallel. Text is "
                             "far cheaper than JPEG decode, so this matters less "
                             "here than in the image run.")
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

    log("=" * 70)
    log(f"OpenCLIP {ARCH} laion2b_s32b_b82k TEXT embedding generation")
    log(f"Data root       : {data_root}")
    log(f"Output dir      : {args.output_dir}")
    log(f"Workers         : worker_{args.min_worker}..worker_{args.max_worker}")
    log(f"Target captions : {args.max_texts:,} (soft)")
    log(f"Batch size      : {args.batch_size}")
    log(f"Tokenise workers: {args.num_workers}   torch threads: {args.torch_threads}")
    log(f"Shards/loader   : {args.shards_per_loader}")
    log(f"Device          : {args.device}   AMP: {not args.no_amp}")
    log("=" * 70)

    run_pipeline(
        data_root=data_root,
        output_dir=Path(args.output_dir),
        weights=weights,
        min_worker=args.min_worker,
        max_worker=args.max_worker,
        max_texts=args.max_texts,
        batch_size=args.batch_size,
        device=args.device,
        prefetch_shards=args.prefetch_shards,
        use_amp=(not args.no_amp),
        num_workers=args.num_workers,
        shards_per_loader=args.shards_per_loader,
        shard_stride=args.shard_stride,
        shard_index=args.shard_index,
    )


if __name__ == "__main__":
    main()
