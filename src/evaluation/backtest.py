"""Walk-forward (expanding-window) backtesting.

A random train/test split leaks the future into the past and will flatter every
model here — the Elo and rolling-form features are built from match history, so
a shuffled split lets the model learn from matches that had not been played yet.
The only honest evaluation is chronological: train on everything up to a cutoff,
predict the next slice, roll forward.

Both model families are driven through one small adapter interface so that a
Dixon-Coles fit and a LightGBM fit can be compared on identical folds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from src.config import OUTCOMES
from src.evaluation.metrics import evaluate
from src.features.build_features import FEATURE_COLUMNS
from src.models.classifiers import predict_proba_frame

log = logging.getLogger(__name__)


class Predictor(Protocol):
    """What a backtestable model must do: fit on a slice, score another slice."""

    def fit(self, train: pd.DataFrame) -> Predictor: ...

    def predict_proba(self, test: pd.DataFrame) -> pd.DataFrame: ...


@dataclass
class SklearnAdapter:
    """Wraps any sklearn-style estimator so it consumes the feature frame."""

    factory: object
    features: list[str] | None = None

    def __post_init__(self):
        self.features = self.features or FEATURE_COLUMNS

    def fit(self, train: pd.DataFrame):
        self.model_ = self.factory() if callable(self.factory) else self.factory
        played = train[train["result"].notna()]
        self.model_.fit(played[self.features].astype(float), played["result"].astype(str))
        return self

    def predict_proba(self, test: pd.DataFrame) -> pd.DataFrame:
        return predict_proba_frame(self.model_, test[self.features].astype(float))


@dataclass
class DixonColesAdapter:
    """Wraps `DixonColesModel`, which needs raw scorelines rather than features."""

    xi: float = 0.0018

    def fit(self, train: pd.DataFrame):
        from src.models.dixon_coles import DixonColesModel

        # Anchor the time decay at the end of the training window, so "recent"
        # means recent relative to the fold, not to today.
        self.model_ = DixonColesModel(xi=self.xi).fit(train, reference_date=train["date"].max())
        return self

    def predict_proba(self, test: pd.DataFrame) -> pd.DataFrame:
        return self.model_.predict_proba(test)


def walk_forward(
    df: pd.DataFrame,
    predictor: Predictor,
    *,
    min_train_seasons: int = 3,
    group_by: str = "season",
) -> pd.DataFrame:
    """Refit once per fold on all earlier data and predict that fold.

    Returns one row per predicted match with the true result and the H/D/A
    probabilities, ready to hand to `src.evaluation.metrics`.
    """
    played = df[df["result"].notna()].sort_values("date", kind="stable").reset_index(drop=True)
    folds = sorted(played[group_by].unique())

    if len(folds) <= min_train_seasons:
        raise ValueError(
            f"need more than {min_train_seasons} folds to backtest, got {len(folds)}"
        )

    predictions = []
    for fold in folds[min_train_seasons:]:
        train = played[played[group_by] < fold]
        test = played[played[group_by] == fold]
        if train.empty or test.empty:
            continue

        proba = predictor.fit(train).predict_proba(test)
        proba = proba.set_axis(test.index)

        block = test[["match_id", "date", "season", "stage", "home", "away", "result"]].copy()
        block[OUTCOMES] = proba[OUTCOMES]
        block["fold"] = fold
        predictions.append(block)

        log.info("fold %s: trained on %d, predicted %d", fold, len(train), len(test))

    return pd.concat(predictions, ignore_index=True)


def score_predictions(predictions: pd.DataFrame) -> dict[str, float]:
    return evaluate(predictions["result"], predictions[OUTCOMES])


def score_by_fold(predictions: pd.DataFrame) -> pd.DataFrame:
    """Per-fold metrics — useful for spotting a model that only works pre-2020."""
    rows = []
    for fold, block in predictions.groupby("fold"):
        rows.append({"fold": fold, **evaluate(block["result"], block[OUTCOMES])})
    return pd.DataFrame(rows)


def compare(
    df: pd.DataFrame,
    predictors: dict[str, Predictor],
    *,
    min_train_seasons: int = 3,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Backtest several models on identical folds and return a leaderboard."""
    rows, all_preds = [], {}
    for name, predictor in predictors.items():
        log.info("Backtesting %s", name)
        preds = walk_forward(df, predictor, min_train_seasons=min_train_seasons)
        all_preds[name] = preds
        rows.append({"model": name, **score_predictions(preds)})

    leaderboard = pd.DataFrame(rows).sort_values("rps").reset_index(drop=True)
    return leaderboard, all_preds
