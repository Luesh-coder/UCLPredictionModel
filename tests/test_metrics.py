"""Metric properties, checked against hand-computed values."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (
    accuracy,
    calibration_table,
    evaluate,
    multiclass_brier,
    multiclass_log_loss,
    ranked_probability_score,
)


def test_rps_is_zero_for_perfect_forecasts():
    y = ["H", "D", "A"]
    p = pd.DataFrame([[1, 0, 0], [0, 1, 0], [0, 0, 1]], columns=["H", "D", "A"], dtype=float)
    assert ranked_probability_score(y, p) == pytest.approx(0.0)


def test_rps_worst_case_is_one():
    """Predicting a certain home win when the away side wins is the max penalty."""
    p = pd.DataFrame([[1.0, 0.0, 0.0]], columns=["H", "D", "A"])
    assert ranked_probability_score(["A"], p) == pytest.approx(1.0)


def test_rps_is_ordinal():
    """A wrong-but-adjacent forecast must score better than a wrong-and-distant one."""
    p_draw = pd.DataFrame([[0.0, 1.0, 0.0]], columns=["H", "D", "A"])
    p_home = pd.DataFrame([[1.0, 0.0, 0.0]], columns=["H", "D", "A"])
    assert ranked_probability_score(["A"], p_draw) < ranked_probability_score(["A"], p_home)


def test_rps_known_value():
    # cum p = (0.5, 0.75); truth H -> cum o = (1, 1).
    # ((0.5)^2 + (0.25)^2) / (3 - 1) = 0.3125 / 2 = 0.15625
    p = pd.DataFrame([[0.5, 0.25, 0.25]], columns=["H", "D", "A"])
    assert ranked_probability_score(["H"], p) == pytest.approx(0.15625)


def test_brier_known_value():
    p = pd.DataFrame([[0.5, 0.25, 0.25]], columns=["H", "D", "A"])
    # (0.5-1)^2 + 0.25^2 + 0.25^2 = 0.375
    assert multiclass_brier(["H"], p) == pytest.approx(0.375)


def test_log_loss_matches_manual():
    p = pd.DataFrame([[0.6, 0.2, 0.2]], columns=["H", "D", "A"])
    assert multiclass_log_loss(["H"], p) == pytest.approx(-np.log(0.6))


def test_uniform_forecast_scores_are_finite_and_ordered():
    y = ["H", "D", "A"] * 10
    p = pd.DataFrame(np.full((30, 3), 1 / 3), columns=["H", "D", "A"])
    m = evaluate(y, p)
    assert m["n"] == 30
    assert m["log_loss"] == pytest.approx(np.log(3))
    assert 0 < m["rps"] < 1


def test_accuracy_picks_argmax():
    p = pd.DataFrame([[0.6, 0.2, 0.2], [0.1, 0.1, 0.8]], columns=["H", "D", "A"])
    assert accuracy(["H", "A"], p) == pytest.approx(1.0)
    assert accuracy(["A", "A"], p) == pytest.approx(0.5)


def test_column_order_is_respected_not_positional():
    """A frame with shuffled columns must still be read by name."""
    ordered = pd.DataFrame([[0.5, 0.25, 0.25]], columns=["H", "D", "A"])
    shuffled = ordered[["A", "D", "H"]]
    assert ranked_probability_score(["H"], shuffled) == pytest.approx(
        ranked_probability_score(["H"], ordered)
    )


def test_calibration_table_recovers_a_known_rate():
    """80 of 100 matches with p=0.8 actually being home wins should show no gap."""
    n = 100
    p = pd.DataFrame({"H": [0.8] * n, "D": [0.1] * n, "A": [0.1] * n})
    y = ["H"] * 80 + ["A"] * 20
    table = calibration_table(y, p, outcome="H", bins=10)
    row = table[table["n"] > 0].iloc[0]
    assert row["mean_predicted"] == pytest.approx(0.8)
    assert row["observed"] == pytest.approx(0.8)
    assert row["gap"] == pytest.approx(0.0)
