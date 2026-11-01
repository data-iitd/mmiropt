"""
The predictors under test, and the three keying rungs H3 compares.

Every one of them is FITTED ON B and SCORED ON C. B calibrated the map; C is
the grader that no construction step has touched. Mixing them would make T10
partly a test on training data and would do so invisibly -- the numbers would
simply come out better.

THE LADDER, from least to most information
------------------------------------------
GA          The configuration's mean CF over B. One number per configuration,
            identical for every query. Knows nothing about location. This is
            the bar: if SM does not clear it, H0 is unsupported and no amount
            of the rest matters (S5.6).

DO          Landmark-keyed but LABEL-FREE in shape: per-landmark corpus
            geometry, mapped into CF units by a fit on B. Guards the cheap
            rival explanation -- "dense regions are easy, sparse regions are
            hard" -- which, if it were the whole story, would mean no CF
            labelling pass was ever needed.

SM_shared   Location-keyed, one column of K numbers shared by EVERY
            configuration, re-levelled to each configuration's own mean. This
            is the "some zones are just hard" story.

SM_shared_scaled
            The same single shared column, but expressed as a UNIT-SCALE shape
            and given a per-configuration scale as well as a per-configuration
            level -- two fitted numbers per configuration instead of one. See
            "why the scaled rung exists" below; it is the rung that decides
            whether P2 is about geography or only about magnitude.

SM_family   One column per index FAMILY (HNSW / IVFPQ / O-IVFPQ), re-levelled
            per configuration. P2 is stated about families -- graph
            connectivity versus quantisation distortion -- not about individual
            parameter settings, so this is the rung P2 actually predicts.

SM_config   The proposed map: one column per (landmark, configuration).

WHY THE SCALED RUNG EXISTS
--------------------------
Added after the first full run, which produced a result the original ladder
could not interpret. SM_config beat SM_shared everywhere (by +0.10 to +0.69
skill), which reads as support for P2 -- except that the per-cell deviation
columns turned out to correlate about as strongly ACROSS families as within
them (text->image: 0.673 within, 0.659 across). Configurations agree about
WHICH cells are hard. P2 says they should not.

The tell was ivfpq_mod under image->text: true CF standard deviation 0.019,
and SM_shared scored skill -4.72 on it. A shared deviation column carrying the
absolute magnitudes of configurations with ten times that spread does not
mis-locate ivfpq_mod's hard cells -- it overwhelms them. That is a SCALE
failure, not a geography failure, and the original SM_shared confounds the two:
it can be beaten either because the shared column points at the wrong cells or
merely because it points with the wrong amplitude.

This rung separates them. The shared column is standardised per configuration
before pooling, so it carries shape only, and each configuration then fits its
own level and amplitude on B. If SM_shared_scaled recovers most of SM_config's
skill, configurations share one geography and differ only in how hard it bites
them: P2 is false as stated, the served map collapses from K x m to K + 2m, and
this ablation is the headline rather than a supporting check. If a gap to
SM_config survives, the residual is genuine per-configuration geography and P2
stands in modified form.

Two numbers per configuration, not one, is still a "shared" map in the sense
that matters: the K-sized object -- the only part that grows with corpus size
-- is shared.

WHY THE SHARED AND FAMILY RUNGS RE-LEVEL RATHER THAN POOL
---------------------------------------------------------
Naively sharing a column would mean averaging hnsw_low's cell means (CF ~ 0.43)
with hnsw_mod's (CF ~ 0.92) and serving the result to both. That variant loses
to GA on level alone, before geography enters, and beating it would prove
nothing. So the shared column carries each cell's DEVIATION from its
configuration's global mean, and prediction adds that deviation back onto the
configuration's own GA. The rungs then differ only in how finely the geography
is keyed, which is exactly the question H3 asks.

BACKOFF
-------
A cell whose B population is under the floor has a mean built from an anecdote.
SM falls back to GA there. T9 deliberately declined to pick the floor (its
sources disagree: the dispatch says ~10, the paper says 30-50 and prescribes
reporting rather than repair), so it is a parameter here and every report
records which value produced it.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "spatialmap"))

from ht_config import Block                                            # noqa: E402
import ht_config as _htc                                               # noqa: E402
# Imported, not re-implemented: T9's H5 preview used exactly these, and DO must
# mean the same thing in both places or the two reports cannot be compared.
from compare_baselines import (density_features, fit_do,              # noqa: E402
                               density_score, fit_do_offsets)

#: The keyed variants, coarsest keying first. H3 walks this ladder.
KEYINGS = ("SM_shared", "SM_shared_scaled", "SM_family", "SM_config")

#: Numbers stored per configuration, beyond the K-sized column(s) each rung
#: needs. H3 reports it because the storage argument is half of P2's stake.
PER_CONFIG_PARAMS = {"SM_shared": 1, "SM_shared_scaled": 2,
                     "SM_family": 1, "SM_config": 0}


# ==========================================================================
# GA
# ==========================================================================
def predict_ga(block: Block, config: str, n: int) -> np.ndarray:
    return np.full(n, block.ga[config], dtype=np.float64)


# ==========================================================================
# DO
# ==========================================================================
def do_features(lay, block: Block) -> np.ndarray:
    """Per-landmark corpus geometry, standardised, with no CF label touched.

    S5.1 defines the density score as "mean distance to its k nearest corpus
    points". No upstream artifact holds that quantity, so this uses the three
    corpus-only measures T5 did store -- log cell population, mean member
    distance to the centroid, and cell tightness -- fitted by full OLS rather
    than the paper's "simple fitted offset". That is MORE generous to DO than
    the specification, deliberately: SM beating DO must not be an artifact of
    handicapping DO. The deviation is recorded in every report that uses it.
    """
    return density_features(lay, block.index_modality, block.K)


def predict_do(X_cells: np.ndarray, block: Block, config: str
               ) -> Tuple[np.ndarray, np.ndarray]:
    """Fit on B, predict on both B and C. Returns (pred_b, pred_c)."""
    coef = fit_do(X_cells, block.geom_b["assign"], block.cf_b[config])

    def apply(assign: np.ndarray) -> np.ndarray:
        A = np.column_stack([np.ones(len(assign)), X_cells[assign]])
        return np.clip(A @ coef, 0.0, 1.0)

    return apply(block.geom_b["assign"]), apply(block.geom_c["assign"])


# ==========================================================================
# SM, at three keying grains
# ==========================================================================
def _deviation_column(block: Block, config_rows: List[int], floor: int
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Population-weighted mean deviation-from-GA per cell, over a set of
    configurations, plus the pooled B count backing each cell.

    Weighted by n, not a plain mean over configurations: a cell holding 400 B
    queries for one configuration and 3 for another should not have the two
    weighted equally when they are pooled into a single column.
    """
    K = block.cell_mean.shape[1]
    num = np.zeros(K, dtype=np.float64)
    den = np.zeros(K, dtype=np.float64)
    for r in config_rows:
        cfg = block.configs[r]
        n = block.cell_n[r].astype(np.float64)
        dev = block.cell_mean[r] - block.ga[cfg]
        occupied = n > 0
        num[occupied] += n[occupied] * dev[occupied]
        den[occupied] += n[occupied]
    col = np.zeros(K, dtype=np.float64)
    nz = den > 0
    col[nz] = num[nz] / den[nz]
    return col, den


def _shared_shape_column(block: Block, config_rows: List[int], floor: int
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """A SHAPE-ONLY shared column: which cells are hard, with the magnitude
    divided out before pooling.

    Each configuration's deviation column is standardised to zero mean and unit
    standard deviation over its own well-populated cells FIRST, then the
    standardised columns are pooled. Without that step the pooled column is
    dominated by whichever configurations happen to have the widest CF spread,
    and applying it to a narrow-spread configuration overwhelms that
    configuration's entire range -- measured, not hypothetical: image->text's
    ivfpq_mod has a CF standard deviation of 0.019 and scored skill -4.72 under
    the unstandardised shared column.

    Both the per-configuration mean and standard deviation are taken WEIGHTED by
    cell population, so a cell holding 400 B queries counts for more than one
    holding 11 when the shape is being defined.
    """
    K = block.cell_mean.shape[1]
    num = np.zeros(K, dtype=np.float64)
    den = np.zeros(K, dtype=np.float64)
    for r in config_rows:
        cfg = block.configs[r]
        n = block.cell_n[r].astype(np.float64)
        ok = n >= floor
        if ok.sum() < 2:
            continue
        dev = block.cell_mean[r] - block.ga[cfg]
        w = n[ok]
        mu = float(np.average(dev[ok], weights=w))
        sd = float(np.sqrt(np.average((dev[ok] - mu) ** 2, weights=w)))
        if sd < 1e-9:                      # a configuration flat across space
            continue                       # contributes no shape information
        num[ok] += w * (dev[ok] - mu) / sd
        den[ok] += w
    shape = np.zeros(K, dtype=np.float64)
    nz = den > 0
    shape[nz] = num[nz] / den[nz]
    return shape, den


def _fit_shared_scale(block: Block, config: str, shape: np.ndarray,
                      support: np.ndarray, floor: int) -> np.ndarray:
    """Per-configuration level and amplitude for the shared shape, fitted on B.

    Two numbers per configuration: ``pred = a + s * shape[cell]``. The intercept
    alone reproduces GA, so this rung can only beat GA by finding a real
    amplitude, and it can never be handicapped by the shape being expressed in
    the wrong units.

    Fitted only over B queries whose cell is well-populated enough to have
    contributed to the shape; queries in thin cells back off to GA at serve
    time, so including them here would fit the amplitude to points the rung
    does not actually serve.
    """
    assign = block.geom_b["assign"]
    usable = support[assign] >= floor
    y = block.cf_b[config]
    if usable.sum() < 10 or np.ptp(shape[assign[usable]]) == 0:
        return np.array([block.ga[config], 0.0])
    A = np.column_stack([np.ones(int(usable.sum())), shape[assign[usable]]])
    coef, *_ = np.linalg.lstsq(A, y[usable], rcond=None)
    return coef


def predict_sm(block: Block, config: str, assign: np.ndarray, floor: int,
               keying: str = "SM_config") -> Tuple[np.ndarray, np.ndarray]:
    """Serve the map for one configuration at one keying grain.

    Returns (prediction, backed_off_mask). ``backed_off`` marks the queries
    served by GA instead of a cell mean -- either because their cell holds
    fewer than ``floor`` B queries or because it holds none at all. Every
    report carries the rate, because a map that backs off everywhere IS GA and
    a headline MAE would not say so.
    """
    r = block.cfg_row(config)
    ga = block.ga[config]

    if keying == "SM_config":
        cell_pred = block.cell_mean[r].copy()
        support = block.cell_n[r].astype(np.float64)
    elif keying == "SM_family":
        fam = block.families[config]
        rows = [i for i, c in enumerate(block.configs)
                if block.families[c] == fam]
        col, support = _deviation_column(block, rows, floor)
        cell_pred = ga + col
    elif keying == "SM_shared":
        col, support = _deviation_column(block, list(range(len(block.configs))), floor)
        cell_pred = ga + col
    elif keying == "SM_shared_scaled":
        shape, support = _shared_shape_column(
            block, list(range(len(block.configs))), floor)
        a, s = _fit_shared_scale(block, config, shape, support, floor)
        cell_pred = a + s * shape
    else:
        raise ValueError(f"unknown keying {keying!r}; known: {KEYINGS}")

    if _htc.OWN_CELL_FLOOR and keying in ("SM_family", "SM_shared"):
        # the draft's q_min rule: the cell's own B count, the same test as SM_config
        support = block.cell_n[r].astype(np.float64)
    thin = support < floor
    cell_pred = np.clip(cell_pred, 0.0, 1.0)
    cell_pred[thin] = ga

    pred = cell_pred[assign]
    return pred, thin[assign]


def sm_on_c(block: Block, config: str, floor: int, keying: str = "SM_config"):
    return predict_sm(block, config, block.geom_c["assign"], floor, keying)


def sm_on_b(block: Block, config: str, floor: int, keying: str = "SM_config"):
    return predict_sm(block, config, block.geom_b["assign"], floor, keying)


# ==========================================================================
# Everything at once, for one pairing
# ==========================================================================
def all_predictions_on_c(lay, block: Block, floor: int
                         ) -> Dict[str, Dict[str, np.ndarray]]:
    """{predictor: {config: prediction on C}} plus the true labels under
    'Oracle' and the backoff masks under 'backoff_<keying>'.

    Assembled in one place so H3 and H5 cannot disagree about what "SM" is.
    """
    if _htc.DO_PROTOCOL:
        z_cells, beta, beta_sh = fit_do_offsets(
            density_score(lay, block.index_modality, block.K),
            block.geom_b["assign"], block.cf_b, block.ga)
        X = None
    else:
        X = do_features(lay, block)
    out: Dict[str, Dict[str, np.ndarray]] = {
        "GA": {}, "DO": {}, "Oracle": {},
        **({"DO_shared": {}} if _htc.DO_PROTOCOL else {}),
        **{k: {} for k in KEYINGS},
        **{f"backoff_{k}": {} for k in KEYINGS},
    }
    n_c = block.n_c
    for cfg in block.configs:
        out["GA"][cfg] = predict_ga(block, cfg, n_c)
        if _htc.DO_PROTOCOL:
            zc = z_cells[block.geom_c["assign"]]
            out["DO"][cfg] = np.clip(block.ga[cfg] + beta[cfg] * zc, 0.0, 1.0)
            out["DO_shared"][cfg] = np.clip(block.ga[cfg] + beta_sh * zc, 0.0, 1.0)
        else:
            out["DO"][cfg] = predict_do(X, block, cfg)[1]
        out["Oracle"][cfg] = block.cf_c[cfg]
        for keying in KEYINGS:
            pred, backed = sm_on_c(block, cfg, floor, keying)
            out[keying][cfg] = pred
            out[f"backoff_{keying}"][cfg] = backed
    return out
