"""
Assemble one role's vectors, in canonical int_id order, for one modality.

T4 needs A's 850,393 vectors as a single contiguous matrix, row i of which is
the vector for A_ids[i]. Everything about how that matrix is built is designed
so a mistake raises rather than silently mis-pairs.

THE ONE INVARIANT
    out[i] is the vector for A_ids[i].
Every FAISS index is then populated with add_with_ids(out, A_ids), so the IDs
FAISS returns at search time are the canonical pilot int_ids -- the same space
T6's exact-KNN labels use, which is what makes the T8 intersection meaningful.

HOW THE MAPPING IS ESTABLISHED
    Not by relying on shard iteration order. Each shard's pair-ids are looked up
    in the pair_index table to get their int_ids, and each vector is written to
    row searchsorted(A_ids, int_id) of the output. That is correct no matter
    what order shards arrive in, and it is checked: a `filled` mask must be all
    True at the end, so a missing shard is an assertion, not a short matrix.

DEGENERATE PAIRS
    Not handled here, and deliberately so. The 353 excluded pairs were dropped
    at T2 labelling, so they never received a stratum, never entered the 1M
    sample, and hold no int_id. They are still physically present in the .npy
    files at their original row positions -- we simply never ask for those rows.
    The gap discipline is inherited, not re-implemented.
"""

import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

# Single source of truth -- index_configs owns the model roster and its dims.
from index_configs import MODEL_DIRS, MODEL_DIMS  # noqa: E402

_SHARD_KEY = re.compile(r"(worker_\d+)/(shard_\d+)\.npy$")


def modality_root(model: str, modality: str) -> Path:
    """result/embeddings_{image,text}/<model dir>"""
    if modality not in ("image", "text"):
        raise ValueError(f"modality must be 'image' or 'text', got {modality!r}")
    return REPO_ROOT / f"result/embeddings_{modality}" / MODEL_DIRS.get(model, model)


def load_included_shards(manifest: Path) -> Set[str]:
    """
    The canonical 7M cut. Read it -- never re-glob the embedding tree, or the
    corpus definition drifts between stages (stratification/README.md).
    """
    return set(json.load(open(manifest))["included_shards"])


def load_pair_index(pair_index_csv: Path, role: str) -> Dict[str, int]:
    """pair_id -> int_id, restricted to one role."""
    out: Dict[str, int] = {}
    with open(pair_index_csv) as f:
        for r in csv.DictReader(f):
            if r["role"] == role:
                out[r["pair_id"]] = int(r["int_id"])
    return out


def _pair_id(name: str) -> str:
    """'worker_0/shard_00130/001306322.jpg' -> 'worker_0/shard_00130/001306322'"""
    return name.rsplit(".", 1)[0]


def _shard_key(path: Path) -> str:
    m = _SHARD_KEY.search(str(path))
    if not m:
        raise ValueError(f"unparseable shard path: {path}")
    return f"{m.group(1)}/{m.group(2)}"


def load_role_matrix(
    model: str,
    modality: str,
    role_ids: np.ndarray,
    pair_to_int: Dict[str, int],
    included_shards: Set[str],
    expect_dim: int = None,
    check_norms: bool = True,
    verbose: bool = True,
    root: Path = None,
) -> np.ndarray:
    """
    Returns float32 [len(role_ids), d] with out[i] the vector for role_ids[i].

    root: the worker_*/shard_*.npy tree to read; default the model's
    result/embeddings_<modality>/ directory (a v2 stage passes its result_v2 tree).

    role_ids must be sorted ascending and unique (assign_ids.py emits them that
    way); both are asserted, since searchsorted silently misplaces rows
    otherwise.
    """
    assert role_ids.ndim == 1, "role_ids must be 1-D"
    assert np.all(np.diff(role_ids) > 0), \
        "role_ids must be strictly ascending -- searchsorted mis-maps otherwise"

    root = Path(root) if root else modality_root(model, modality)
    shard_paths = sorted(root.glob("worker_*/shard_*.npy"))
    shard_paths = [p for p in shard_paths if _shard_key(p) in included_shards]
    if len(shard_paths) != len(included_shards):
        raise RuntimeError(
            f"{modality}: manifest lists {len(included_shards)} shards but "
            f"{len(shard_paths)} are on disk under {root}"
        )

    # Peek the dim from the first shard's header rather than hardcoding it.
    d = int(np.load(shard_paths[0], mmap_mode="r").shape[1])
    if expect_dim is not None and d != expect_dim:
        raise RuntimeError(f"{modality}: expected dim {expect_dim}, shards are {d}")

    n = len(role_ids)
    out = np.empty((n, d), dtype=np.float32)
    filled = np.zeros(n, dtype=bool)
    if verbose:
        print(f"  {modality}: {len(shard_paths):,} shards -> [{n:,} x {d}] "
              f"({out.nbytes / 2**30:.2f} GiB)")

    for si, npy_path in enumerate(shard_paths):
        names = json.load(open(npy_path.with_suffix("").with_suffix(".names.json")))
        rows, iids = [], []
        for row, name in enumerate(names):
            iid = pair_to_int.get(_pair_id(name))
            if iid is not None:                # else: other role, or excluded
                rows.append(row)
                iids.append(iid)
        if not rows:
            continue

        # One batched searchsorted per shard rather than one call per row --
        # the per-row form costs ~7M Python-level numpy calls over the corpus.
        iids = np.asarray(iids, dtype=np.int64)
        dests = np.searchsorted(role_ids, iids)
        keep = (dests < n) & (role_ids[np.minimum(dests, n - 1)] == iids)
        if not keep.any():
            continue
        rows = np.asarray(rows)[keep]
        dests = dests[keep]

        vecs = np.load(npy_path, mmap_mode="r")
        if vecs.shape[0] != len(names):
            raise RuntimeError(f"{npy_path}: {vecs.shape[0]} rows vs {len(names)} names")
        out[dests] = vecs[rows]
        filled[dests] = True

        if verbose and (si + 1) % 500 == 0:
            print(f"    {si + 1:,}/{len(shard_paths):,} shards, {int(filled.sum()):,}/{n:,} rows")

    missing = int((~filled).sum())
    if missing:
        first = role_ids[~filled][:5].tolist()
        raise RuntimeError(
            f"{modality}: {missing:,} of {n:,} rows never filled (e.g. int_ids {first}). "
            f"A vector is missing from the embedding tree, or the manifest and "
            f"pair_index disagree about the corpus."
        )

    if check_norms:
        # The whole metric argument (L2 == cosine ranking) rests on unit norm.
        sample = out[np.linspace(0, n - 1, min(n, 4096)).astype(np.int64)]
        norms = np.linalg.norm(sample, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            raise RuntimeError(
                f"{modality}: vectors are not unit-norm "
                f"(min {norms.min():.6f}, max {norms.max():.6f}). "
                f"index_configs.METRIC='L2' assumes they are."
            )
        if verbose:
            print(f"    norms ok: [{norms.min():.6f}, {norms.max():.6f}]")

    return out


def load_selected(
    model: str,
    modality: str,
    wanted_ids: np.ndarray,
    pair_index_csv: Path,
    verbose: bool = True,
    root: Path = None,
) -> np.ndarray:
    """
    Vectors for a specific, small set of int_ids -- out[i] for wanted_ids[i].

    root: the worker_*/shard_*.npy tree to read; default the model's
    result/embeddings_<modality>/ directory. A v2 stage passes its result_v2
    tree (indexes/v2_dirs.py).

    load_role_matrix streams every shard in the corpus because it wants every
    row of a role. Verification wants a few hundred rows, so paying a full
    ~30 GB pass (twice: once for A's probes, once for B's queries) is most of
    its runtime for none of its work.

    A pair_id encodes its own location ("worker_3/shard_00412/002345678"), so
    the shards holding a given set of ids are known before any file is opened.
    This reads only those, and only the rows inside them.

    wanted_ids must be sorted ascending and unique.
    """
    wanted = np.asarray(wanted_ids, dtype=np.int64)
    assert wanted.ndim == 1 and np.all(np.diff(wanted) > 0), \
        "wanted_ids must be strictly ascending"
    want = set(wanted.tolist())

    # int_id -> pair_id, for these ids only.
    id_to_pair: Dict[int, str] = {}
    with open(pair_index_csv) as f:
        for r in csv.DictReader(f):
            iid = int(r["int_id"])
            if iid in want:
                id_to_pair[iid] = r["pair_id"]
    missing = want - set(id_to_pair)
    if missing:
        raise RuntimeError(f"{len(missing)} requested ids absent from "
                           f"{pair_index_csv}, e.g. {sorted(missing)[:5]}")

    # Group by the shard each pair lives in, straight from the pair_id.
    by_shard: Dict[str, List[Tuple[str, int]]] = {}
    for iid, pid in id_to_pair.items():
        worker, shard, name = pid.rsplit("/", 2)
        by_shard.setdefault(f"{worker}/{shard}", []).append((name, iid))

    root = Path(root) if root else modality_root(model, modality)
    d = int(np.load(next(root.glob("worker_*/shard_*.npy")), mmap_mode="r").shape[1])
    out = np.empty((len(wanted), d), dtype=np.float32)
    filled = np.zeros(len(wanted), dtype=bool)
    if verbose:
        print(f"  {modality}: {len(wanted):,} vectors from {len(by_shard):,} shards "
              f"(of ~2,875 -- targeted read, not a full pass)")

    for shard_key, entries in by_shard.items():
        npy_path = root / f"{shard_key}.npy"
        names = json.load(open(npy_path.with_suffix("").with_suffix(".names.json")))
        row_of = {_pair_id(n): i for i, n in enumerate(names)}
        rows, dests = [], []
        for name, iid in entries:
            row = row_of.get(f"{shard_key}/{name}")
            if row is None:
                raise RuntimeError(f"{shard_key}: pair {name} not in names.json")
            rows.append(row)
            dests.append(int(np.searchsorted(wanted, iid)))
        vecs = np.load(npy_path, mmap_mode="r")
        out[np.asarray(dests)] = vecs[np.asarray(rows)]
        filled[np.asarray(dests)] = True

    if not filled.all():
        raise RuntimeError(f"{int((~filled).sum())} of {len(wanted)} rows never filled")
    return out


def load_A(model: str, modality: str, pilot_dir: Path, manifest: Path,
           expect_dim: int = None, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper: returns (vectors, A_ids) ready for add_with_ids."""
    a_ids = np.load(pilot_dir / "A_ids.npy")
    pair_to_int = load_pair_index(pilot_dir / "pair_index.csv", role="A")
    if len(pair_to_int) != len(a_ids):
        raise RuntimeError(
            f"pair_index.csv has {len(pair_to_int):,} role-A rows but "
            f"A_ids.npy has {len(a_ids):,}"
        )
    included = load_included_shards(manifest)
    vecs = load_role_matrix(model, modality, a_ids, pair_to_int, included,
                            expect_dim=expect_dim, verbose=verbose)
    return vecs, a_ids
