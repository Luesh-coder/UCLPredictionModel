"""Feature engineering, with leakage as the thing actually under test.

Every feature must be computable before kickoff. A leak here would not raise —
it would just produce a backtest score that never survives contact with a real
fixture list, so these tests check the shift/ordering discipline directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.build_features import (
    FEATURE_COLUMNS,
    add_rolling_form,
    build_features,
    to_long,
    xy,
)
from src.features.elo import EloConfig, EloModel, add_elo_features


def test_elo_is_zero_sum_within_a_match():
    model = EloModel(EloConfig(regress_to_mean=0.0))
    model.update("A", "B", 2, 0)
    assert model.get("A") + model.get("B") == 1500.0 * 2


def test_elo_rewards_the_winner_and_more_for_a_bigger_margin():
    narrow, wide = EloModel(), EloModel()
    narrow.update("A", "B", 1, 0)
    wide.update("A", "B", 5, 0)
    assert narrow.get("A") > 1500.0
    assert wide.get("A") > narrow.get("A")


def test_elo_home_advantage_shifts_the_expectation():
    model = EloModel()
    assert model.expected("A", "B") > 0.5
    assert model.expected("A", "B", neutral=True) == 0.5


def test_elo_regression_pulls_toward_the_mean_without_crossing_it():
    model = EloModel(EloConfig(regress_to_mean=0.25))
    model.ratings = {"A": 1700.0, "B": 1300.0}
    model.regress()
    assert model.ratings["A"] == 1650.0
    assert model.ratings["B"] == 1350.0


def test_elo_columns_are_pre_match(synthetic_matches):
    """The first match of the dataset must see both teams at the initial rating.

    If Elo were computed after the fact, this row would already be updated.
    """
    out = add_elo_features(synthetic_matches)
    first = out.iloc[0]
    assert first["elo_home"] == 1500.0
    assert first["elo_away"] == 1500.0


def test_elo_separates_strong_from_weak_teams(synthetic_matches):
    """Team 00 is built to be the strongest, Team 11 the weakest."""
    out = add_elo_features(synthetic_matches)
    final = EloModel()
    for row in out.itertuples(index=False):
        final.update(row.home, row.away, int(row.home_goals), int(row.away_goals))
    assert final.get("ENG Team 00") > final.get("ENG Team 11")


def test_rolling_form_excludes_the_current_match(synthetic_matches):
    """A team's first match has no history, and later rows must lag by one."""
    long = add_rolling_form(to_long(synthetic_matches), window=5)
    team = long[long["team"] == "ENG Team 00"].sort_values("date").reset_index(drop=True)

    assert pd.isna(team.loc[0, "form_gf"])
    # The second row's form is exactly the first match's goals — not an average
    # that includes the second match itself.
    assert team.loc[1, "form_gf"] == float(team.loc[0, "gf"])
    assert team.loc[2, "form_gf"] == float(team.loc[:1, "gf"].mean())


def test_rolling_form_respects_the_window(synthetic_matches):
    long = add_rolling_form(to_long(synthetic_matches), window=3)
    team = long[long["team"] == "ENG Team 00"].sort_values("date").reset_index(drop=True)
    assert team.loc[5, "form_gf"] == float(team.loc[2:4, "gf"].mean())


def test_experience_counts_previous_appearances(synthetic_matches):
    long = add_rolling_form(to_long(synthetic_matches), window=5)
    team = long[long["team"] == "ENG Team 00"].sort_values("date").reset_index(drop=True)
    assert team["experience"].tolist()[:4] == [0.0, 1.0, 2.0, 3.0]


def test_build_features_emits_every_declared_column(synthetic_matches):
    feats = build_features(synthetic_matches)
    assert set(FEATURE_COLUMNS).issubset(feats.columns)
    assert len(feats) == len(synthetic_matches)


def test_build_features_leaves_no_nan_in_the_model_matrix(synthetic_matches):
    """Rolling features are NaN for a team's debut; build_features must fill them."""
    feats = build_features(synthetic_matches)
    assert not feats[FEATURE_COLUMNS].isna().any().any()


def test_elo_pools_are_independent_per_league(synthetic_matches):
    """Two leagues that never meet must not influence each other's ratings.

    The fixture's leagues are disjoint, so rating them together would let one
    league's scoring rate drift the other's numbers with no match ever
    connecting them. Rating each league alone is the only defensible reading
    until Stage 2's Champions League fixtures join the two pools.
    """
    both = add_elo_features(synthetic_matches, pool="league")
    only_eng = add_elo_features(
        synthetic_matches[synthetic_matches["league"] == "ENG-Premier League"], pool="league"
    )

    merged = both.merge(only_eng[["match_id", "elo_home"]], on="match_id", suffixes=("", "_alone"))
    assert np.allclose(merged["elo_home"], merged["elo_home_alone"])


def test_pooling_protects_against_misaligned_season_boundaries():
    """A global pool regresses ratings on *another* league's season boundary.

    Leagues do not change season on the same day, so once rows from several
    leagues are interleaved by date the season label flips back and forth. A
    single global pass reads each flip as a new season and applies the
    between-season regression to the mean, repeatedly, mid-campaign. Pooling by
    league is what stops one league's calendar from corrupting another's
    ratings, and this is the case that proves `pool` is load-bearing.
    """
    rows = []
    for i in range(12):
        # League A crosses a season boundary halfway through; league B does not.
        rows.append(
            {
                "match_id": f"a{i}",
                "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=2 * i),
                "season": 2019 if i < 6 else 2020,
                "league": "A",
                "stage": "league",
                "home": "A Home",
                "away": "A Away",
                "home_goals": 2,
                "away_goals": 0,
            }
        )
        rows.append(
            {
                "match_id": f"b{i}",
                "date": pd.Timestamp("2020-01-02") + pd.Timedelta(days=2 * i),
                "season": 2019,
                "league": "B",
                "stage": "league",
                "home": "B Home",
                "away": "B Away",
                "home_goals": 2,
                "away_goals": 0,
            }
        )
    df = pd.DataFrame(rows)

    per_league = add_elo_features(df, pool="league")
    global_pool = add_elo_features(df, pool=None)

    b_rows = per_league["league"] == "B"
    b_pooled = per_league.loc[b_rows].set_index("match_id")["elo_home"]
    b_global = global_pool.set_index("match_id").loc[b_pooled.index, "elo_home"]

    # League B never changes season, so its ratings must rise monotonically as
    # "B Home" keeps winning. Under a global pool they are dragged back toward
    # 1500 every time league A's season label flips.
    assert b_pooled.is_monotonic_increasing
    assert not np.allclose(b_pooled, b_global)
    assert b_pooled.iloc[-1] > b_global.iloc[-1]


def test_features_do_not_correlate_perfectly_with_the_target(synthetic_matches):
    """A perfect correlation is the signature of a leaked result column."""
    feats = build_features(synthetic_matches)
    home_win = (feats["result"] == "H").astype(float)
    for col in FEATURE_COLUMNS:
        values = feats[col].astype(float)
        if values.nunique() <= 1:
            continue
        assert abs(np.corrcoef(values, home_win)[0, 1]) < 0.9, col


def test_xy_drops_unplayed_fixtures(synthetic_matches):
    matches = synthetic_matches.copy()
    matches.loc[matches.index[-5:], "result"] = pd.NA
    feats = build_features(matches)
    X, y = xy(feats)
    assert len(X) == len(y) == len(matches) - 5
    assert set(y.unique()) <= {"H", "D", "A"}
