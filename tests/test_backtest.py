"""Walk-forward backtesting: chronology, fold coverage and the adapter contract.

The point of this module is that no fold may ever see its own future. These
tests assert that directly rather than trusting the loop to be written right.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import OUTCOMES
from src.evaluation.backtest import (
    DixonColesAdapter,
    SklearnAdapter,
    compare,
    score_by_fold,
    score_predictions,
    walk_forward,
)
from src.features.build_features import build_features
from src.models.classifiers import PriorBaseline


@pytest.fixture
def features(synthetic_matches):
    return build_features(synthetic_matches)


def test_walk_forward_skips_the_first_training_seasons(features):
    preds = walk_forward(features, SklearnAdapter(PriorBaseline), min_train_seasons=3)
    seasons = sorted(features["season"].unique())
    assert sorted(preds["fold"].unique()) == seasons[3:]


def test_walk_forward_predicts_every_match_in_its_folds_exactly_once(features):
    preds = walk_forward(features, SklearnAdapter(PriorBaseline), min_train_seasons=3)
    seasons = sorted(features["season"].unique())[3:]
    expected = features[features["season"].isin(seasons)]
    assert len(preds) == len(expected)
    assert preds["match_id"].is_unique


def test_walk_forward_never_trains_on_the_fold_it_predicts(features):
    """Records the training window each fold saw and checks it stays in the past."""
    seen = []

    class Spy(SklearnAdapter):
        def fit(self, train):
            seen.append((train["season"].max(), train["date"].max()))
            return super().fit(train)

    preds = walk_forward(features, Spy(PriorBaseline), min_train_seasons=3)

    folds = sorted(preds["fold"].unique())
    for (max_train_season, max_train_date), fold in zip(seen, folds, strict=True):
        assert max_train_season < fold
        assert max_train_date < preds.loc[preds["fold"] == fold, "date"].min()


def test_walk_forward_output_is_a_valid_probability_table(features):
    preds = walk_forward(features, SklearnAdapter(PriorBaseline), min_train_seasons=3)
    assert set(OUTCOMES).issubset(preds.columns)
    assert preds[OUTCOMES].to_numpy().min() >= 0.0
    assert preds[OUTCOMES].sum(axis=1).to_numpy() == pytest.approx(np.ones(len(preds)))
    assert preds["result"].notna().all()


def test_walk_forward_refuses_too_few_folds(features):
    one_season = features[features["season"] == features["season"].min()]
    with pytest.raises(ValueError, match="need more than"):
        walk_forward(one_season, SklearnAdapter(PriorBaseline), min_train_seasons=3)


def test_dixon_coles_adapter_runs_the_same_folds(features):
    preds = walk_forward(features, DixonColesAdapter(xi=0.0), min_train_seasons=4)
    assert preds[OUTCOMES].sum(axis=1).to_numpy() == pytest.approx(np.ones(len(preds)))
    assert preds["fold"].nunique() == 1


def test_dixon_coles_adapter_anchors_decay_inside_the_training_window(features):
    """The reference date must be the fold's own end, not today."""
    adapter = DixonColesAdapter(xi=0.0018)
    train = features[features["season"] < 2021]
    adapter.fit(train)
    assert adapter.model_.reference_date_ == train["date"].max()


def test_score_predictions_and_score_by_fold_agree_on_coverage(features):
    preds = walk_forward(features, SklearnAdapter(PriorBaseline), min_train_seasons=3)
    overall = score_predictions(preds)
    per_fold = score_by_fold(preds)

    assert overall["n"] == len(preds)
    assert per_fold["n"].sum() == len(preds)
    assert set(per_fold["fold"]) == set(preds["fold"])
    assert 0.0 < overall["rps"] < 1.0


def test_compare_ranks_by_rps(features):
    leaderboard, all_preds = compare(
        features,
        {"prior": SklearnAdapter(PriorBaseline), "dixon_coles": DixonColesAdapter(xi=0.0)},
        min_train_seasons=3,
    )
    assert set(all_preds) == {"prior", "dixon_coles"}
    assert leaderboard["rps"].is_monotonic_increasing
    # Every model must be scored on exactly the same matches.
    assert len(all_preds["prior"]) == len(all_preds["dixon_coles"])


def test_dixon_coles_beats_the_prior_out_of_sample(features):
    """The synthetic data has real structure, so a fitted model should exploit it."""
    leaderboard, _ = compare(
        features,
        {"prior": SklearnAdapter(PriorBaseline), "dixon_coles": DixonColesAdapter(xi=0.0)},
        min_train_seasons=3,
    )
    scores = leaderboard.set_index("model")["rps"]
    assert scores["dixon_coles"] < scores["prior"]


def test_predictions_carry_the_columns_needed_to_inspect_a_bad_call(features):
    preds = walk_forward(features, SklearnAdapter(PriorBaseline), min_train_seasons=3)
    for col in ("match_id", "date", "season", "stage", "home", "away", "result", "fold"):
        assert col in preds.columns
    assert isinstance(preds["date"].iloc[0], pd.Timestamp)
