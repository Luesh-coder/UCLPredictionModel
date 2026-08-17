"""Parsing of FBref's raw fields into the canonical match table.

These are the functions that touch scraped strings, which is where the format
surprises live: en dashes, penalty shootouts, country prefixes and three
different ways of spelling a season.
"""

from __future__ import annotations

import pandas as pd

from src.data.build_dataset import _result_from_goals, _split_score, normalise_team
from src.data.ingest import season_to_start_year


def test_split_score_handles_en_dash():
    """FBref uses an en dash, not a hyphen."""
    out = _split_score(pd.Series(["2–1", "0–0"]))
    assert out["home_goals"].tolist() == [2, 0]
    assert out["away_goals"].tolist() == [1, 0]


def test_split_score_strips_penalty_shootouts():
    """A tie won on penalties is still a draw on the night."""
    out = _split_score(pd.Series(["1–1 (4–3)"]))
    assert out["home_goals"].tolist() == [1]
    assert out["away_goals"].tolist() == [1]


def test_split_score_returns_na_for_unplayed_fixtures():
    out = _split_score(pd.Series(["", None, "postponed"]))
    assert out["home_goals"].isna().all()
    assert out["away_goals"].isna().all()


def test_season_labels_map_to_start_year():
    seasons = pd.Series(["1617", "2425", "2024", "2024-2025", "2024-25", "24"])
    assert season_to_start_year(seasons).tolist() == [2016, 2024, 2024, 2024, 2024, 2024]


def test_unparseable_season_label_becomes_na_instead_of_raising():
    """One odd label must not take down an entire backfill."""
    out = season_to_start_year(pd.Series(["2425", "unknown", None]))
    assert out.tolist()[0] == 2024
    assert out.isna().tolist() == [False, True, True]


def test_normalise_team_collapses_aliases_and_country_prefixes():
    assert normalise_team("Paris S-G") == "Paris Saint-Germain"
    assert normalise_team("eng Liverpool") == "Liverpool"
    assert normalise_team("Manchester  Utd") == "Manchester United"
    # A club with no alias passes through untouched.
    assert normalise_team("Ajax") == "Ajax"


def test_normalise_team_passes_through_non_strings():
    assert pd.isna(normalise_team(pd.NA))


def test_result_is_na_until_the_match_is_played():
    home = pd.Series([2, 1, 0, pd.NA], dtype="Int64")
    away = pd.Series([1, 1, 2, pd.NA], dtype="Int64")
    res = _result_from_goals(home, away)
    assert res.tolist()[:3] == ["H", "D", "A"]
    assert pd.isna(res.iloc[3])
