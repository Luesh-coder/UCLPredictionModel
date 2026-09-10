"""Model contracts: valid probability vectors, correct column order, sane fits.

The synthetic fixture has a known strength gradient (ENG Team 00 strongest), so
each model gets one substantive check that it recovered that structure on top
of the mechanical output-shape checks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import OUTCOMES
from src.features.build_features import build_features, xy
from src.models.classifiers import (
    EloBaseline,
    PriorBaseline,
    make_logistic,
    predict_proba_frame,
)
from src.models.dixon_coles import DixonColesModel, tau, time_decay_weights


@pytest.fixture
def features(synthetic_matches):
    return build_features(synthetic_matches)


# --- Dixon-Coles ------------------------------------------------------------


def test_time_decay_weights_favour_recent_matches():
    dates = pd.Series(pd.to_datetime(["2020-01-01", "2021-01-01", "2022-01-01"]))
    w = time_decay_weights(dates, pd.Timestamp("2022-01-01"), xi=0.0018)
    assert w[2] == pytest.approx(1.0)
    assert w[0] < w[1] < w[2]


def test_time_decay_with_xi_zero_weights_everything_equally():
    dates = pd.Series(pd.to_datetime(["2015-01-01", "2022-01-01"]))
    w = time_decay_weights(dates, pd.Timestamp("2022-01-01"), xi=0.0)
    assert w == pytest.approx(np.ones(2))


def test_tau_only_touches_low_scores():
    lam = np.full(3, 1.4)
    mu = np.full(3, 1.1)
    h = np.array([0, 1, 3])
    a = np.array([0, 1, 2])
    out = tau(h, a, lam, mu, rho=-0.05)
    assert out[0] != 1.0
    assert out[1] != 1.0
    # 3-2 is outside the correction's range and must be left alone.
    assert out[2] == 1.0


def test_dixon_coles_recovers_the_strength_ordering(synthetic_matches):
    model = DixonColesModel(xi=0.0).fit(synthetic_matches)
    ratings = model.team_ratings().set_index("team")["attack"]
    assert ratings["ENG Team 00"] > ratings["ENG Team 11"]
    # Attack is identified only up to a constant, which `_unpack` pins at zero.
    assert ratings.mean() == pytest.approx(0.0, abs=1e-8)


def test_dixon_coles_finds_a_home_advantage(synthetic_matches):
    """The fixture generates home goals with a +0.25 boost."""
    model = DixonColesModel(xi=0.0).fit(synthetic_matches)
    _, _, home_adv, _ = model._unpack(model.params_)
    assert home_adv > 0.0


def test_dixon_coles_score_matrix_is_a_distribution(synthetic_matches):
    model = DixonColesModel(xi=0.0).fit(synthetic_matches)
    m = model.score_matrix("ENG Team 00", "ENG Team 11")
    assert m.sum() == pytest.approx(1.0)
    assert (m >= 0).all()


def test_dixon_coles_probabilities_favour_the_stronger_side(synthetic_matches):
    model = DixonColesModel(xi=0.0).fit(synthetic_matches)
    fixtures = pd.DataFrame({"home": ["ENG Team 00"], "away": ["ENG Team 11"]})
    proba = model.predict_proba(fixtures)
    assert list(proba.columns) == OUTCOMES
    assert proba.sum(axis=1).iloc[0] == pytest.approx(1.0)
    assert proba["H"].iloc[0] > proba["A"].iloc[0]


def test_dixon_coles_handles_a_team_it_never_saw(synthetic_matches):
    """New clubs enter the competition every season; that must not raise."""
    model = DixonColesModel(xi=0.0).fit(synthetic_matches)
    fixtures = pd.DataFrame({"home": ["ENG Team 00"], "away": ["Newcomer FC"]})
    proba = model.predict_proba(fixtures)
    assert proba.sum(axis=1).iloc[0] == pytest.approx(1.0)


def test_dixon_coles_rejects_an_empty_training_set():
    empty = pd.DataFrame({"home": [], "away": [], "home_goals": [], "away_goals": [], "date": []})
    with pytest.raises(ValueError):
        DixonColesModel().fit(empty)


# --- Classifiers ------------------------------------------------------------


def test_prior_baseline_reproduces_the_base_rates(features):
    X, y = xy(features)
    model = PriorBaseline().fit(X, y)
    proba = predict_proba_frame(model, X)
    observed = y.value_counts(normalize=True)
    for outcome in OUTCOMES:
        assert proba[outcome].iloc[0] == pytest.approx(observed.get(outcome, 0.0))
    # Every row is identical by construction.
    assert proba.nunique().max() == 1


def test_elo_baseline_is_monotone_in_elo_difference(features):
    X, y = xy(features)
    model = EloBaseline().fit(X, y)
    grid = pd.DataFrame({c: [0.0] * 3 for c in X.columns})
    grid["elo_diff"] = [-300.0, 0.0, 300.0]
    proba = predict_proba_frame(model, grid)
    assert proba["H"].is_monotonic_increasing
    assert proba["A"].is_monotonic_decreasing


@pytest.mark.parametrize("factory", [PriorBaseline, EloBaseline, make_logistic])
def test_every_model_emits_valid_probability_vectors(features, factory):
    X, y = xy(features)
    model = factory()
    model.fit(X, y)
    proba = predict_proba_frame(model, X)

    assert list(proba.columns) == OUTCOMES
    assert len(proba) == len(X)
    assert proba.to_numpy().min() >= 0.0
    assert proba.sum(axis=1).to_numpy() == pytest.approx(np.ones(len(X)))


def test_predict_proba_frame_reorders_sklearn_alphabetical_classes(features):
    """sklearn sorts classes to A/D/H; the frame must come back as H/D/A."""
    X, y = xy(features)
    model = make_logistic().fit(X, y)
    assert list(model.steps[-1][1].classes_) == ["A", "D", "H"]

    frame = predict_proba_frame(model, X)
    raw = model.predict_proba(X)
    assert list(frame.columns) == OUTCOMES
    assert frame["H"].to_numpy() == pytest.approx(raw[:, 2])
    assert frame["A"].to_numpy() == pytest.approx(raw[:, 0])


def test_logistic_beats_the_prior_baseline_in_sample(features):
    """A model with real features that cannot beat the base rates is broken."""
    from src.evaluate.metrics import ranked_probability_score

    X, y = xy(features)
    prior = predict_proba_frame(PriorBaseline().fit(X, y), X)
    logistic = predict_proba_frame(make_logistic().fit(X, y), X)
    assert ranked_probability_score(y, logistic) < ranked_probability_score(y, prior)
