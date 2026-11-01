"""
The scoring vocabulary shared by H0-H5.

Three kinds of number live here, and they answer different questions:

REGRESSION  How far off is the predicted CF?  MAE, RMSE, bias, Spearman.
            The paper's S5.7 headline metrics.

CALIBRATION Is the predicted CF *believable* as a number, or only as a
            ranking?  A predictor can rank perfectly and still say 0.4 where
            the truth is 0.9. Spearman cannot see that; a fidelity-floor policy
            would be destroyed by it. Hence binned predicted-vs-observed curves
            and the expected calibration error that summarises them.

DECISION    Does the map make the RIGHT CALL at a fidelity floor tau?  This is
            what the map is for (S1: "among the configurations predicted to be
            good enough, use the cheapest"), and it is not implied by a good
            MAE: error far from tau is free, error at tau flips the decision.
            Hence threshold-decision accuracy and realized CF regret.

A NOTE ON SKILL
---------------
``skill = 1 - MSE(pred)/MSE(GA)`` is reported everywhere alongside MAE because
MAE alone is not comparable across configurations: hnsw_high's CF sits near
0.99 with almost no spread, so a MAE of 0.02 there is unimpressive, while the
same 0.02 on hnsw_low (CF ~ 0.43, wide) would be remarkable. Skill divides that
scale out -- 0 means "no better than one global number", 1 means perfect.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np
from scipy.stats import spearmanr


#: CF is overlap/100 stored as float32, so a query at exactly 0.70 is held as
#: 0.699999988 and fails ">= 0.70" against a float64 tau (same at 0.90, 0.95).
#: The tolerance is far below the 0.01 CF step, so it only rescues those ties.
CF_EPS = 1e-6


def clears(x, tau: float) -> np.ndarray:
    """CF >= tau, counting values stored exactly on tau as clearing it."""
    return np.asarray(x) >= tau - CF_EPS


# ==========================================================================
# Regression
# ==========================================================================
def regression_metrics(pred: np.ndarray, true: np.ndarray,
                       ga: float | None = None) -> Dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    err = pred - true
    out = {
        "n": int(len(true)),
        "MAE": float(np.abs(err).mean()),
        "RMSE": float(np.sqrt((err ** 2).mean())),
        "bias": float(err.mean()),
    }
    # Spearman is undefined against a constant predictor, which GA always is.
    # Reported as nan rather than 0: "no ordering information" is not the same
    # claim as "an ordering that happens to be uncorrelated".
    out["spearman"] = (float(spearmanr(pred, true).statistic)
                       if np.ptp(pred) > 0 else float("nan"))
    if ga is not None:
        mse_ga = float(((ga - true) ** 2).mean())
        out["skill"] = float(1.0 - out["RMSE"] ** 2 / mse_ga) if mse_ga > 0 else 0.0
    return out


# ==========================================================================
# Calibration
# ==========================================================================
def calibration_curve(pred: np.ndarray, true: np.ndarray, bins: int = 10
                      ) -> Dict[str, list]:
    """Equal-COUNT bins of the prediction, each reporting mean predicted and
    mean observed CF.

    Equal-count rather than equal-width because SM's predictions pile up hard
    near the config's own mean; equal-width bins would leave most of the range
    holding a handful of queries and produce a curve made mostly of noise.

    A perfectly calibrated predictor puts every point on the diagonal. ``ece``
    is the count-weighted mean vertical distance from it -- the number a
    fidelity-floor policy actually feels.
    """
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    n = len(pred)
    if n == 0:
        return {"bin": [], "n": [], "pred_mean": [], "true_mean": [],
                "true_std": [], "ece": float("nan")}

    order = np.argsort(pred, kind="stable")
    edges = np.linspace(0, n, bins + 1).astype(int)
    rows = {"bin": [], "n": [], "pred_mean": [], "true_mean": [], "true_std": []}
    ece = 0.0
    for b in range(bins):
        idx = order[edges[b]:edges[b + 1]]
        if len(idx) == 0:
            continue
        p, t = pred[idx].mean(), true[idx].mean()
        rows["bin"].append(b)
        rows["n"].append(int(len(idx)))
        rows["pred_mean"].append(float(p))
        rows["true_mean"].append(float(t))
        rows["true_std"].append(float(true[idx].std()))
        ece += len(idx) / n * abs(p - t)
    rows["ece"] = float(ece)
    return rows


# ==========================================================================
# Decision: does this configuration clear tau?
# ==========================================================================
def threshold_metrics(pred: np.ndarray, true: np.ndarray, tau: float
                      ) -> Dict[str, float]:
    """Per-(query, configuration) classification of "clears tau".

    Reported with precision and recall as well as accuracy because the classes
    are wildly unbalanced at the ends of the catalogue: hnsw_high clears 0.7 for
    virtually every query, so predicting "yes" always scores 99% accuracy and
    says nothing. Precision/recall of the positive class, and the base rate
    itself, are what make that visible.
    """
    p, t = clears(pred, tau), clears(true, tau)
    tp = float(np.sum(p & t)); fp = float(np.sum(p & ~t))
    fn = float(np.sum(~p & t)); tn = float(np.sum(~p & ~t))
    n = max(len(t), 1)
    return {
        "tau": float(tau),
        "accuracy": float((tp + tn) / n),
        "precision": float(tp / (tp + fp)) if tp + fp else float("nan"),
        "recall": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "base_rate": float(t.mean()),
        "predicted_rate": float(p.mean()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def realized_regret(pred: Dict[str, np.ndarray], true: Dict[str, np.ndarray],
                    costs: Dict[str, float], tau: float,
                    configs: Sequence[str] | None = None) -> Dict[str, object]:
    """The metric the map exists to serve: how much true fidelity is lost by
    following the map's cheapest-clearing pick instead of the oracle's.

    THE POLICY, STATED ONCE
    -----------------------
    Order the catalogue by measured cost, cheapest first. For each query:

        pick = the first configuration whose CF is >= tau
        if none clears tau, escalate: take the configuration with the highest CF

    The map runs that policy on its predictions; the oracle runs the identical
    policy on the true CF. Regret is ``true[oracle pick] - true[map pick]``.

    Why the escalation clause matters: without it, a query no configuration can
    serve would have no defined pick, and those queries are exactly the sparse,
    hard ones the map is being stress-tested on. Dropping them would quietly
    remove the hardest cases from the metric. Taking the best available
    configuration is also what a real system would do before falling back to
    exact search.

    Regret is SIGNED and can be negative: the map can accidentally choose a
    configuration whose true CF beats the oracle's cheapest clearing one, since
    the oracle stops at the first configuration over the floor rather than
    maximising. ``mean_regret`` keeps the sign; ``mean_shortfall`` is the
    positive part, which is the loss a deployment actually suffers.

    Cost is reported alongside, because a predictor can buy low regret by
    always escalating to the most expensive configuration -- which is precisely
    the trivial answer S1 says no predictor is needed for.
    """
    names = list(configs) if configs is not None else list(true)
    names = [c for c in names if c in pred and c in true and c in costs]
    if not names:
        return {"tau": float(tau), "n_configs": 0,
                "note": "no configuration has both a prediction and a measured cost"}

    order = sorted(names, key=lambda c: costs[c])          # cheapest first
    P = np.column_stack([pred[c] for c in order])          # [n_queries, n_cfg]
    T = np.column_stack([true[c] for c in order])
    C = np.array([costs[c] for c in order], dtype=np.float64)

    def choose(scores: np.ndarray) -> np.ndarray:
        ok = clears(scores, tau)
        first = np.argmax(ok, axis=1)                      # 0 when none clears
        none = ~ok.any(axis=1)
        first[none] = np.argmax(scores[none], axis=1)      # escalate
        return first

    map_pick, oracle_pick = choose(P), choose(T)
    rows = np.arange(len(map_pick))
    got, best = T[rows, map_pick], T[rows, oracle_pick]
    regret = best - got
    disagree = map_pick != oracle_pick

    return {
        "tau": float(tau),
        "n_queries": int(len(rows)),
        "n_configs": len(order),
        "cost_order": order,
        "disagreement_rate": float(disagree.mean()),
        "mean_regret": float(regret.mean()),
        "mean_shortfall": float(np.maximum(regret, 0.0).mean()),
        "mean_regret_when_disagreeing": (float(regret[disagree].mean())
                                         if disagree.any() else 0.0),
        "p95_shortfall": float(np.percentile(np.maximum(regret, 0.0), 95)),
        "mean_cf_map_pick": float(got.mean()),
        "mean_cf_oracle_pick": float(best.mean()),
        "mean_cost_map_pick": float(C[map_pick].mean()),
        "mean_cost_oracle_pick": float(C[oracle_pick].mean()),
        "cost_ratio_map_over_oracle": (float(C[map_pick].mean() / C[oracle_pick].mean())
                                       if C[oracle_pick].mean() > 0 else float("nan")),
        "frac_escalated_map": float((~clears(P, tau).any(axis=1)).mean()),
        "frac_escalated_oracle": float((~clears(T, tau).any(axis=1)).mean()),
    }


# ==========================================================================
# Binned diagnostics (H1's error-vs-distance curves)
# ==========================================================================
def binned_profile(x: np.ndarray, y: np.ndarray, bins: int = 10
                   ) -> Dict[str, list]:
    """Mean y per equal-count bin of x, with the standard error of each bin.

    Equal-count again: d1 and the boundary ratio both have long thin tails, and
    equal-width bins would put the entire interesting range in one bucket.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)
    out = {"bin": [], "n": [], "x_mean": [], "y_mean": [], "y_sem": []}
    if n == 0:
        return out
    order = np.argsort(x, kind="stable")
    edges = np.linspace(0, n, bins + 1).astype(int)
    for b in range(bins):
        idx = order[edges[b]:edges[b + 1]]
        if len(idx) == 0:
            continue
        out["bin"].append(b)
        out["n"].append(int(len(idx)))
        out["x_mean"].append(float(x[idx].mean()))
        out["y_mean"].append(float(y[idx].mean()))
        out["y_sem"].append(float(y[idx].std(ddof=1) / np.sqrt(len(idx)))
                            if len(idx) > 1 else float("nan"))
    return out


def trend(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Spearman and a standardised OLS slope of y on x.

    Both, not one: Spearman answers "is the relationship monotone at all",
    which is what P1 predicts, while the slope in standardised units answers
    "how much error does a one-sigma move in x cost", which is what says
    whether the effect is worth anything. A relationship can be significantly
    monotone and numerically irrelevant.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 3 or np.ptp(x) == 0:
        return {"spearman": float("nan"), "p_value": float("nan"),
                "slope_per_sd": float("nan"), "n": int(len(x))}
    r = spearmanr(x, y)
    xs = (x - x.mean()) / (x.std() if x.std() > 0 else 1.0)
    slope = float(np.polyfit(xs, y, 1)[0])
    return {"spearman": float(r.statistic), "p_value": float(r.pvalue),
            "slope_per_sd": slope, "n": int(len(x))}


def fmt(v: float, nd: int = 4) -> str:
    return "  nan" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def mean_ignoring_nan(values: Iterable[float]) -> float:
    arr = np.array([v for v in values], dtype=np.float64)
    return float(np.nanmean(arr)) if len(arr) else float("nan")


# ==========================================================================
# Statistics of the draft (main.tex "Statistical Analysis", l.476-486)
# ==========================================================================
#: Bootstrap resamples (user, 2026-10-19: the draft's \mr{number} -> 1,000).
BOOT_R = 1000
BOOT_SEED = 0


class CellBootstrap:
    """Resample CELLS, not queries (draft l.478): queries in one cell share
    their estimate, so they are not independent. Each draw picks n_cells cells
    with replacement and takes ALL their queries along, with multiplicity.

        cb = CellBootstrap(cell_of_query)
        for idx in cb.draws():          # index arrays into the query axis
            stat(idx)
    """

    def __init__(self, cells: np.ndarray, R: int = BOOT_R, seed: int = BOOT_SEED):
        cells = np.asarray(cells, dtype=np.int64)
        self.order = np.argsort(cells, kind="stable")
        uniq, self.starts, self.counts = np.unique(cells[self.order], return_index=True,
                                                   return_counts=True)
        self.n_cells = len(uniq)
        self.R, self.seed = R, seed

    def draws(self):
        rng = np.random.default_rng(self.seed)
        for _ in range(self.R):
            pick = rng.integers(0, self.n_cells, self.n_cells)
            cnt = self.counts[pick]
            total = int(cnt.sum())
            ends = np.cumsum(cnt)
            within = np.arange(total) - np.repeat(ends - cnt, cnt)
            yield self.order[np.repeat(self.starts[pick], cnt) + within]


def boot_ci(values: Sequence[float], estimate: float, level: float = 0.95) -> Dict[str, float]:
    """Percentile interval of bootstrap replicates around the full-sample estimate."""
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=np.float64)
    a = (1.0 - level) / 2.0
    if len(v) == 0:
        return {"estimate": float(estimate), "lo": float("nan"), "hi": float("nan"), "R": 0}
    return {"estimate": float(estimate), "lo": float(np.quantile(v, a)),
            "hi": float(np.quantile(v, 1.0 - a)), "R": int(len(v))}


def benjamini_hochberg(p: Sequence[float]) -> np.ndarray:
    """BH-adjusted p-values (q-values); the draft tests at q < 0.05 (l.486).
    NaN p-values stay NaN and are not counted in the family."""
    p = np.asarray(p, dtype=np.float64)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        from scipy.stats import false_discovery_control
        out[ok] = false_discovery_control(p[ok], method="bh")
    return out


def wilcoxon_paired(a: np.ndarray, b: np.ndarray, zero_method: str = "wilcox") -> Dict[str, float]:
    """Wilcoxon signed-rank on paired per-query values (draft l.482-483):
    'wilcox' for squared errors (RQ2, RQ3), 'pratt' for shortfall (RQ4)."""
    from scipy.stats import wilcoxon
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    if not np.any(d != 0):
        return {"statistic": float("nan"), "p_value": 1.0, "n": int(len(d)),
                "median_diff": 0.0, "zero_method": zero_method}
    r = wilcoxon(d, zero_method=zero_method, alternative="two-sided", method="approx")
    return {"statistic": float(r.statistic), "p_value": float(r.pvalue), "n": int(len(d)),
            "median_diff": float(np.median(d)), "zero_method": zero_method}


def mcnemar(miss_a: np.ndarray, miss_b: np.ndarray) -> Dict[str, float]:
    """McNemar's test on paired met/missed outcomes (draft l.484): exact binomial
    on the discordant pairs."""
    from scipy.stats import binomtest
    a, b = np.asarray(miss_a, bool), np.asarray(miss_b, bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    n = n01 + n10
    p = 1.0 if n == 0 else float(binomtest(n10, n, 0.5).pvalue)
    return {"a_missed_b_met": n10, "a_met_b_missed": n01, "p_value": p}


def spearman_fast(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman's rho as the Pearson correlation of average ranks (ties averaged)."""
    from scipy.stats import rankdata
    rx, ry = rankdata(x), rankdata(y)
    rx -= rx.mean(); ry -= ry.mean()
    den = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


class CellWeights:
    """The same cell bootstrap as CellBootstrap, as WEIGHTS: draw r gives each cell
    the number of times it was picked, and every per-query statistic that is a
    ratio of sums becomes a weighted sum over cells -- fast enough for 1,000 draws
    of every (pairing, configuration, predictor). Same seed => the same draws as
    CellBootstrap, and the same draws for every statistic of one block, so joint
    quantities (a mean over configurations, a difference of two predictors) are
    resampled consistently."""

    def __init__(self, cells: np.ndarray, R: int = BOOT_R, seed: int = BOOT_SEED):
        uniq, self.inv = np.unique(np.asarray(cells, dtype=np.int64), return_inverse=True)
        self.n_cells = len(uniq)
        rng = np.random.default_rng(seed)
        self.W = np.stack([np.bincount(rng.integers(0, self.n_cells, self.n_cells),
                                       minlength=self.n_cells) for _ in range(R)]).astype(np.float64)

    def sums(self, values: np.ndarray) -> np.ndarray:
        """Per-cell sums of a per-query array."""
        return np.bincount(self.inv, weights=np.asarray(values, dtype=np.float64),
                           minlength=self.n_cells)

    def totals(self, values: np.ndarray) -> np.ndarray:
        """[R] resampled totals of a per-query array."""
        return self.W @ self.sums(values)

    def query_weights(self, r: int) -> np.ndarray:
        """Per-query multiplicities of draw r."""
        return self.W[r][self.inv]


def skill_replicates(cw: "CellWeights", pred: np.ndarray, base: np.ndarray,
                     true: np.ndarray) -> np.ndarray:
    """[R] replicates of 1 - MSE(pred)/MSE(base) under the cell bootstrap."""
    num = cw.totals((pred - true) ** 2)
    den = cw.totals((base - true) ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, 1.0 - num / den, np.nan)


def resampled_matched_optimum(cw: "CellWeights", cstar: np.ndarray, none: np.ndarray,
                              static_miss: np.ndarray, charge: float) -> np.ndarray:
    """[R] mean cost of the v2 matched optimum on each cell-bootstrap draw:
    unservable queries and the extra misses are charged `charge` (the static
    configuration's cost); the miss budget is the static plan's misses IN THE DRAW.
    `cstar` already holds `charge` for unservable queries."""
    saving = np.where(none, 0.0, cstar - charge)
    order = np.argsort(-saving, kind="stable")
    s_sorted = saving[order]
    pos = s_sorted > 0
    out = np.empty(cw.W.shape[0])
    for r in range(cw.W.shape[0]):
        w = cw.query_weights(r)
        n = w.sum()
        extra = float((w * static_miss).sum() - (w * none).sum())
        total = float((w * cstar).sum())
        if extra > 0:
            ws = w[order][pos]
            cum = np.cumsum(ws)
            take = np.clip(extra - (cum - ws), 0.0, ws)
            total -= float((take * s_sorted[pos]).sum())
        out[r] = total / n
    return out
