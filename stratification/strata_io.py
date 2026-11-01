"""
Shared I/O and allocation helpers for T2 (sampling) and T3 (A/B/C split).

Both stages follow the same two-pass shape, which is what lets them run over
the full 7M corpus without holding pair-ids in RAM:

    pass 1  count pairs per joint stratum
    ---     decide each stratum's allocation, and build a SHUFFLED sequence
            of per-row outcomes for it
    pass 2  stream the pairs again, handing out that stratum's outcomes in order

The shuffle matters. Pairs arrive in (worker, shard) order, so handing out
roles sequentially would put every early shard in A and every late shard in C
-- role would correlate with crawl order. Shuffling the outcome sequence
instead of the pair-ids costs one byte per pair (~7 MB at full corpus) rather
than ~1 GB of strings.
"""

import csv
import json
import random
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

Stratum = Tuple[int, int]


# --------------------------------------------------------------- input

def iter_from_label_csvs(image_csv: Path, text_csv: Path) -> Iterator[Tuple[str, int, int]]:
    """
    Stream (pair_id, image_stratum, text_stratum) from the two per-modality
    label files written by run_stratification.py. They are row-aligned by
    construction (identical pair_id column, verified), so a zip is a valid join.
    """
    with open(image_csv) as fi, open(text_csv) as ft:
        next(fi); next(ft)
        for li, lt in zip(fi, ft):
            pid_i, si = li.rstrip("\n").rsplit(",", 1)
            pid_t, st = lt.rstrip("\n").rsplit(",", 1)
            if pid_i != pid_t:
                raise ValueError(f"label files out of alignment at {pid_i!r} vs {pid_t!r}")
            yield pid_i, int(si), int(st)


def iter_from_combined_csv(path: Path) -> Iterator[Tuple[str, int, int]]:
    """Stream from a combined file (pair_id, image_stratum, text_stratum, ...)."""
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            yield row["pair_id"], int(row["image_stratum"]), int(row["text_stratum"])


def open_pairs(args) -> Iterator[Tuple[str, int, int]]:
    """Pick the input form from parsed args: --input, else the two label CSVs."""
    if getattr(args, "input", None):
        return iter_from_combined_csv(args.input)
    return iter_from_label_csvs(args.image_labels, args.text_labels)


def count_strata(pairs: Iterator[Tuple[str, int, int]]) -> Dict[Stratum, int]:
    counts: Dict[Stratum, int] = {}
    for _, i, t in pairs:
        k = (i, t)
        counts[k] = counts.get(k, 0) + 1
    return counts


# ---------------------------------------------------------- allocation

def largest_remainder(n: int, ratios: Tuple[float, ...], rng: random.Random) -> List[int]:
    """
    Split n into len(ratios) parts. Integer floors first, then each leftover
    unit is handed to one part at random, weighted by the ratios -- so the
    allocation is unbiased in expectation rather than always rounding the
    same way.
    """
    base = [int(n * r) for r in ratios]
    idx = list(range(len(ratios)))
    for _ in range(n - sum(base)):
        base[rng.choices(idx, weights=ratios)[0]] += 1
    return base


def shuffled_sequence(outcomes: Dict[str, int], rng: random.Random) -> List[str]:
    """A shuffled flat list, e.g. {'A':4,'B':1} -> ['A','B','A','A','A']."""
    seq: List[str] = []
    for label, n in outcomes.items():
        seq.extend([label] * n)
    rng.shuffle(seq)
    return seq


class Dealer:
    """Hands out a stratum's pre-shuffled outcomes, one per row, in pass 2."""

    def __init__(self, sequences: Dict[Stratum, List[str]]):
        self._seq = sequences
        self._pos: Dict[Stratum, int] = {k: 0 for k in sequences}

    def take(self, stratum: Stratum) -> str:
        p = self._pos[stratum]
        self._pos[stratum] = p + 1
        return self._seq[stratum][p]

    def assert_exhausted(self) -> None:
        for k, seq in self._seq.items():
            if self._pos[k] != len(seq):
                raise RuntimeError(f"stratum {k}: dealt {self._pos[k]} of {len(seq)}")


def write_meta(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))
