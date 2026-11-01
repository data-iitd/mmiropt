#!/usr/bin/env python3
"""
SigLIP2 so400m-patch16-512 TEXT embedding generation from TAR shards.

Embeds the "caption" string of every <id>.json member. Weights load from the
local models/ directory - this never touches the network, which matters because
Pragya compute nodes have no proxy route.

ABOUT THIS MODEL
    google/siglip2-so400m-patch16-512. No projection head: the embedding IS the
    pooled hidden state, so the vector is 1152-d, matching the tower width.
    get_text_features returns the text model's BaseModelOutputWithPooling and
    the embedding is its .pooler_output.

CONTEXT LENGTH - 64 TOKENS, THE SHORTEST IN THE PROJECT
    SigLIP's text_config declares no max_position_embeddings and the tokenizer
    reports the "unset" sentinel, so the limit is read off the WEIGHTS: the
    position embedding table has exactly 64 rows. That is shorter than the 77 of
    the CLIP-family models and far shorter than FLAVA's 512.

    SigLIP is trained with fixed-width padding and no attention mask, so every
    caption is padded to exactly 64 - the model has no notion of a shorter
    sequence. Captions longer than 64 tokens are truncated, which is normal
    operation for this checkpoint, not damage.

    This widens the context spread already documented for FLAVA: ranking models
    on long-caption queries compares 64 vs 77 vs 512 tokens of input, not just
    backbones.

TOKENIZER - LOWERCASE, AS SIGLIP 2 WAS TRAINED (--tokenizer)
    siglip2  transformers' Siglip2Tokenizer: the Gemma tokenizer with a
             Lowercase() normalizer in front ("SigLIP2 training default:
             lowercase normalization"). The default.
    gemma    what AutoTokenizer loads from this checkpoint. Its
             tokenizer_config.json names GemmaTokenizer, whose normalizer only
             replaces spaces, so captions keep their case. The first run
             (2026-09-04) used this; the option exists only to reproduce it.

    The two give different vectors for the 93% of captions with an uppercase
    letter. Measured on 20,205 B queries (audit/siglip2_lowercase_check.py,
    2026-10-11): text->image recall@1 is 23.4% with gemma and 89.0% with
    lowercase text. The tokenizer is recorded in progress.json and
    run_summary.json, and a run refuses an output directory whose vectors were
    made with the other one.

OUTPUT - one file pair per shard, mirroring the dataset layout
    result_v2/embeddings_text/siglip2_so400m_patch16_512_lowercase/   --tokenizer siglip2
    result/embeddings_text/siglip2_so400m_patch16_512/                --tokenizer gemma
        worker_0/
            shard_00000.npy            float32 [N, 1152], L2-normalised
            shard_00000.names.json     N names, same row order as the .npy
            ...
        progress.json                  resume state
        failed_captions.json           every caption that did NOT make it in
        anomalous_captions.json        empty/missing captions (embedded, flagged)
        run_summary.json

    Row i of shard_XXXXX.npy corresponds to entry i of shard_XXXXX.names.json,
    and to row i of the matching IMAGE shard - both sides sort a shard's members
    by filename and ids are zero-padded to 9 digits.

    Names are fully qualified:
        worker_0/shard_00000/000000000.json

STOPPING (soft limit)
    Shards are consumed in order from the very first one, every caption in each,
    nothing sampled or skipped. Checked BETWEEN shards, so the shard in progress
    always finishes and the total slightly exceeds --max-texts.

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
import transformers
from torch.utils.data import Dataset, DataLoader
from transformers import SiglipModel, AutoTokenizer

CAPTION_EXTENSION = ".json"
CAPTION_KEY = "caption"

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_KEY = "siglip2_so400m_patch16_512"
DEFAULT_MODEL_DIR = REPO_ROOT / "models/siglip2_so400m_patch16_512"
DEFAULT_MAX_TEXTS = 8_000_000

# One output directory per tokenizer, so the two kinds of vector never share a
# tree. "gemma" is the first run's directory and must not be written again;
# "siglip2" lives in result_v2/, the results tree built on the lowercase vectors.
TOKENIZERS = ("siglip2", "gemma")
DEFAULT_TOKENIZER = "siglip2"
OUTPUT_DIRS = {
    "siglip2": REPO_ROOT / "result_v2/embeddings_text" / f"{MODEL_KEY}_lowercase",
    "gemma": REPO_ROOT / "result/embeddings_text" / MODEL_KEY,
}
DEFAULT_OUTPUT_DIR = OUTPUT_DIRS[DEFAULT_TOKENIZER]


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
# not damage - and at 64 tokens SigLIP2 clips more captions than any other model
# in the project, which would drown the records that genuinely carry no text.
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
    Resume state. Shards are atomic - a shard is either fully written or not
    written at all - so this is just a completed set plus counters.
    """

    def __init__(self, output_dir: Path, tokenizer: str):
        self.output_dir = output_dir
        self.progress_path = output_dir / "progress.json"
        self.lock = threading.Lock()
        self.started_at = time.time()
        self.completed_tars: Set[str] = set()
        self.total_texts = 0
        self.total_failed = 0
        self.tokenizer = tokenizer
        # What made the shards already on disk. The first run's progress.json
        # predates this field, and that run used the gemma tokenizer.
        self.recorded_tokenizer: Optional[str] = None
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
            self.recorded_tokenizer = data.get("tokenizer", "gemma")
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
                "tokenizer": self.tokenizer,
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


class SiglipTextTransform:
    """
    Adapts the tokeniser to the single-caption call a Dataset makes.

    Padding is to a FIXED width, not to the batch longest. SigLIP is trained
    without an attention mask - the text tower has exactly `max_len` position
    rows and consumes all of them - so a caption's vector must not depend on
    whichever captions happened to share its batch.
    """

    def __init__(self, tokenizer, max_len: int):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, text: str):
        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return enc["input_ids"][0]


class MultiShardDataset(Dataset):
    """
    Parses and tokenises SEVERAL shards' captions within one DataLoader lifetime.

    Building a DataLoader per shard cost ~34 ms per worker in fork/join - at 48
    workers that was ~1.6 s of a 7.5 s shard, over 20% of wall time, and it grew
    with worker count. Grouping shards amortises that spawn across all of them.

    Output stays strictly shard-aligned: every item carries its flat index, and
    the caller maps it back through `index` to (group, position) before writing.

    A caption that cannot be read yields a zero token row with ok=False rather
    than raising, so the batch keeps its shape and alignment never shifts.
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


class Siglip2TextEmbedder:
    # tokenizer defaults to "gemma" HERE, unlike the CLI, so that callers written
    # before the option existed (audit/siglip2_lowercase_check.py) keep getting the
    # first run's tokenizer and still reproduce its vectors.
    def __init__(self, device: str, batch_size: int, use_amp: bool, model_dir: Path,
                 num_workers: int = 8, tokenizer: str = "gemma"):
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

        log(f"Loading SigLIP2 so400m-patch16-512 from {model_dir}")
        self.model = SiglipModel.from_pretrained(str(model_dir)).to(device).eval()
        if tokenizer == "siglip2":
            # Imported here: only transformers >= 5 has it, and the gemma path
            # must keep working without it.
            from transformers import Siglip2Tokenizer
            self.tokenizer = Siglip2Tokenizer.from_pretrained(str(model_dir))
        elif tokenizer == "gemma":
            self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        else:
            raise ValueError(f"tokenizer must be one of {TOKENIZERS}, got {tokenizer!r}")
        self.tokenizer_name = tokenizer
        self.tokenizer_info = self._check_tokenizer(tokenizer)
        # No projection head: the pooled hidden state IS the embedding.
        self.dim = int(self.model.config.text_config.hidden_size)
        # The context length is read off the WEIGHTS. text_config declares no
        # max_position_embeddings and the tokenizer reports the "unset"
        # sentinel, so the position table is the only honest source.
        self.max_text_len = int(
            self.model.text_model.embeddings.position_embedding.num_embeddings
        )
        log(f"Model loaded (context {self.max_text_len} tokens, embedding dim {self.dim})")

        self.transform = SiglipTextTransform(self.tokenizer, self.max_text_len)
        self.blank_ids = torch.zeros(self.max_text_len, dtype=torch.long)

    def _check_tokenizer(self, name: str) -> Dict:
        """
        Fail before any shard is read if the tokenizer is not what its name says.

        Two properties decide the vectors:
          lowercase    siglip2 must map "A Photo" and "a photo" to the same ids,
                       gemma must not.
          padding side SigLIP's text tower pools the LAST position and takes no
                       attention mask, so left padding would pool a different
                       token. Both tokenizers must pad on the right, as the first
                       run did.
        """
        tok = self.tokenizer
        lowercases = tok("A Photo")["input_ids"] == tok("a photo")["input_ids"]
        if lowercases != (name == "siglip2"):
            raise RuntimeError(f"tokenizer {name!r} ({type(tok).__name__}) "
                               f"lowercases={lowercases}: not the expected behaviour")
        if tok.padding_side != "right":
            raise RuntimeError(f"tokenizer {name!r} pads on the {tok.padding_side}; "
                               f"SigLIP pools the last position, so it must be right")
        backend = getattr(tok, "backend_tokenizer", None)
        info = {"tokenizer": name, "tokenizer_class": type(tok).__name__,
                "normalizer": str(backend.normalizer) if backend is not None else None,
                "lowercases": lowercases, "padding_side": tok.padding_side}
        log(f"Tokenizer: {info}")
        return info

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
                # SigLIP returns the text model's BaseModelOutputWithPooling.
                # No attention_mask: the model is trained on fixed-width padded
                # input and does not take one here.
                feats = self.model.get_text_features(input_ids=i).pooler_output
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
    model_dir: Path,
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
    tokenizer: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    progress = ProgressManager(output_dir, tokenizer)
    # A resumed run appends shards to what is there, so the two kinds of vector
    # would end up in one tree. Shards without a progress.json are refused too:
    # nothing says what made them.
    on_disk = any(output_dir.glob("worker_*/shard_*.npy"))
    if on_disk and progress.recorded_tokenizer != tokenizer:
        raise SystemExit(
            f"{output_dir} already holds shards made with tokenizer "
            f"{progress.recorded_tokenizer!r}; this run uses {tokenizer!r}. "
            f"Write to the directory for {tokenizer!r} ({OUTPUT_DIRS[tokenizer]}).")
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

    embedder = Siglip2TextEmbedder(device, batch_size, use_amp, model_dir, num_workers,
                                   tokenizer=tokenizer)
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
        "model": "google/siglip2-so400m-patch16-512",
        "model_dir": str(model_dir),
        "caption_source": f"<id>.json / \"{CAPTION_KEY}\"",
        "max_text_len": embedder.max_text_len,
        **embedder.tokenizer_info,
        "embedding_dim": embedder.dim,
        "dtype": "float32",
        "normalised": True,
        "batch_size": batch_size,
        "amp": "float16 autocast" if (use_amp and str(device).startswith("cuda")) else "off",
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
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
        description="SigLIP2 so400m-patch16-512 text embeddings from TAR shards."
    )
    parser.add_argument("--data-root", type=str, required=True,
                        help="Path containing worker_* folders with .tar shards")
    parser.add_argument("--tokenizer", choices=TOKENIZERS, default=DEFAULT_TOKENIZER,
                        help="siglip2: lowercase, as SigLIP 2 was trained (default). "
                             "gemma: the first run's cased tokenizer, only to "
                             "reproduce result/embeddings_text/" + MODEL_KEY)
    parser.add_argument("--output-dir", type=str, default=None,
                        help="default: the tokenizer's own directory, "
                             + ", ".join(f"{k} -> {v}" for k, v in OUTPUT_DIRS.items()))
    parser.add_argument("--model-dir", type=str, default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--min-worker", type=int, default=0)
    parser.add_argument("--max-worker", type=int, default=20)
    parser.add_argument("--max-texts", type=int, default=DEFAULT_MAX_TEXTS,
                        help="Soft limit: the shard in progress still finishes")
    # 64 tokens is the shortest context in the project, so the text tower runs
    # cheaply and tolerates a large batch.
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
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"model-dir not found: {model_dir}")
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIRS[args.tokenizer]

    log("=" * 70)
    log("SigLIP2 so400m-patch16-512 TEXT embedding generation")
    log(f"Data root       : {data_root}")
    log(f"Output dir      : {output_dir}")
    log(f"Tokenizer       : {args.tokenizer}")
    log(f"Workers         : worker_{args.min_worker}..worker_{args.max_worker}")
    log(f"Target captions : {args.max_texts:,} (soft)")
    log(f"Batch size      : {args.batch_size}")
    log(f"Tokenise workers: {args.num_workers}   torch threads: {args.torch_threads}")
    log(f"Shards/loader   : {args.shards_per_loader}")
    log(f"Device          : {args.device}   AMP: {not args.no_amp}")
    log("=" * 70)

    run_pipeline(
        data_root=data_root,
        output_dir=output_dir,
        model_dir=model_dir,
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
        tokenizer=args.tokenizer,
    )


if __name__ == "__main__":
    main()
