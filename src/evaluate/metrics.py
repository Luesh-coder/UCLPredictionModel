"""Scoring rules for three-way football forecasts.

Accuracy is a poor metric here: draws are ~25% of matches and almost never the
argmax, so a model can score well on accuracy while being useless. Use RPS and
log loss. RPS is the standard in the football-forecasting literature because it
is *ordinal* — predicting a home win when the away side wins is punished more
than predicting a draw.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import OUTCOMES

EPS = 1e-15


def _as_matrix(proba: pd.DataFrame | np.ndarray) -> np.ndarray:
    if isinstance(proba, pd.DataFrame):
        proba = proba[OUTCOMES].to_numpy()
    return np.asarray(proba, dtype=float)


def _as_onehot(y_true: pd.Series | np.ndarray) -> np.ndarray:
    y = pd.Series(np.asarray(y_true).ravel()).astype(str)
    onehot = np.column_stack([(y == c).to_numpy(dtype=float) for c in OUTCOMES])
    # An unrecognised label would otherwise become an all-zero row and quietly
    # drag every metric toward zero instead of failing.
    unknown = set(y[onehot.sum(axis=1) == 0])
    if unknown:
        raise ValueError(f"outcomes {sorted(unknown)} are not in {OUTCOMES}")
    return onehot


def ranked_probability_score(y_true, proba) -> float:
    """Mean RPS. Lower is better; 0 is perfect, and ~0.21 is a good football model.

    RPS compares cumulative distributions over the *ordered* outcome scale
    H < D < A, so the penalty grows with how far the truth sits from the mass.
    """
    p = _as_matrix(proba)
    o = _as_onehot(y_true)
    cum_p = np.cumsum(p, axis=1)
    cum_o = np.cumsum(o, axis=1)
    # The final cumulative pair is always (1, 1), so only the first k-1 count.
    # Dividing by k-1 is what puts RPS on the [0, 1] scale the literature quotes.
    squared = np.sum((cum_p[:, :-1] - cum_o[:, :-1]) ** 2, axis=1)
    return float(np.mean(squared) / (len(OUTCOMES) - 1))


def multiclass_log_loss(y_true, proba) -> float:
    """Cross-entropy. Punishes confident mistakes hardest.

    Computed directly rather than via `sklearn.metrics.log_loss`, which sorts
    `labels` lexicographically and assumes the probability columns follow that
    order — with H/D/A columns it silently scores the wrong class.
    """
    p = np.clip(_as_matrix(proba), EPS, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    o = _as_onehot(y_true)
    return float(-np.mean(np.sum(o * np.log(p), axis=1)))


def multiclass_brier(y_true, proba) -> float:
    """Mean squared error between the probability vector and the one-hot truth."""
    p = _as_matrix(proba)
    o = _as_onehot(y_true)
    return float(np.mean(np.sum((p - o) ** 2, axis=1)))


def accuracy(y_true, proba) -> float:
    """Top-1 accuracy. Reported for intuition only — do not optimise it."""
    p = _as_matrix(proba)
    pred = np.array(OUTCOMES)[p.argmax(axis=1)]
    return float((pred == np.asarray(y_true).ravel().astype(str)).mean())


def evaluate(y_true, proba) -> dict[str, float]:
    """All headline metrics for one set of predictions."""
    return {
        "rps": ranked_probability_score(y_true, proba),
        "log_loss": multiclass_log_loss(y_true, proba),
        "brier": multiclass_brier(y_true, proba),
        "accuracy": accuracy(y_true, proba),
        "n": int(len(np.asarray(y_true).ravel())),
    }


def calibration_table(y_true, proba, outcome: str = "H", bins: int = 10) -> pd.DataFrame:
    """Predicted vs. observed frequency for one outcome, bucketed by confidence.

    A well-calibrated model has `mean_predicted` ≈ `observed` in every row. This
    is the fastest way to see over-confidence, which aggregate metrics hide.
    """
    p = _as_matrix(proba)[:, OUTCOMES.index(outcome)]
    hit = (np.asarray(y_true).ravel().astype(str) == outcome).astype(float)

    df = pd.DataFrame({"p": p, "hit": hit})
    df["bin"] = pd.cut(df["p"], np.linspace(0, 1, bins + 1), include_lowest=True)
    out = (
        df.groupby("bin", observed=True)
        .agg(n=("hit", "size"), mean_predicted=("p", "mean"), observed=("hit", "mean"))
        .reset_index()
    )
    out["gap"] = out["observed"] - out["mean_predicted"]
    return out
