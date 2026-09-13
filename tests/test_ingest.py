"""The ingest layer: parsing hazards, odds de-vigging and team canonicalisation.

These cover the genuinely new code in Stage 1. All three hazards exercised by
the `football_data_csv` fixture were observed in the real files, not invented:
the site switched to UTF-8-with-BOM partway through its history, mixes two date
formats, and leaves a blank line at the end of most CSVs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ingest.build_dataset import CANONICAL_COLUMNS, build
from src.ingest.football_data import (
    _decode,
    _harmonise_numeric,
    _parse_dates,
    _read_csv,
    season_code,
)
from src.ingest.odds import add_market_probabilities
from src.ingest.teams import canonical_team, normalise, unmapped

BOM = "﻿"

# --- football-data parsing --------------------------------------------------


def test_season_code_builds_the_url_fragment():
    assert season_code(2024) == "2425"
    assert season_code(1999) == "9900"
    assert season_code(2009) == "0910"


def test_decode_strips_the_byte_order_mark(football_data_csv):
    """Read as plain UTF-8 the first column is named BOM + "Div", not "Div"."""
    decoded = _decode(football_data_csv.encode("utf-8"))
    assert decoded.startswith("Div,")
    assert BOM not in decoded


def test_decode_falls_back_to_latin1_on_pre_utf8_files():
    """Older files contain bytes that are not valid UTF-8 and must not raise."""
    payload = "E0,Malaga\xe9\n".encode("latin-1")
    with pytest.raises(UnicodeDecodeError):
        payload.decode("utf-8")
    assert _decode(payload).startswith("E0,")


def test_read_csv_handles_bom_and_blank_trailing_line(football_data_csv):
    df = _read_csv(_decode(football_data_csv.encode("utf-8")), "fixture")
    assert list(df.columns)[:3] == ["Div", "Date", "HomeTeam"]
    assert len(df) == 3  # the blank fourth line is not a match


def test_read_csv_rejects_a_file_without_the_expected_schema():
    with pytest.raises(ValueError, match="HomeTeam"):
        _read_csv("Something,Else\n1,2\n", "fixture")


def test_parse_dates_reads_both_formats_day_first():
    """Both "17/08/2019" and "18/08/19" occur, sometimes within one file."""
    parsed = _parse_dates(pd.Series(["17/08/2019", "18/08/19"]))
    assert parsed.tolist() == [pd.Timestamp("2019-08-17"), pd.Timestamp("2019-08-18")]


def test_parse_dates_is_day_first_not_month_first():
    """13/01 is unambiguous; 01/02 is not, and must read as 1 February."""
    parsed = _parse_dates(pd.Series(["13/01/2020", "01/02/2020"]))
    assert parsed[0] == pd.Timestamp("2020-01-13")
    assert parsed[1] == pd.Timestamp("2020-02-01")


def test_parse_dates_coerces_junk_rather_than_raising():
    parsed = _parse_dates(pd.Series(["not a date", "17/08/2019"]))
    assert pd.isna(parsed[0])
    assert parsed[1] == pd.Timestamp("2019-08-17")


# --- odds -------------------------------------------------------------------


def _fixture_frame(csv: str) -> pd.DataFrame:
    return _read_csv(_decode(csv.encode("utf-8")), "fixture")


def test_market_probabilities_sum_to_one(football_data_csv):
    out = add_market_probabilities(_fixture_frame(football_data_csv))
    assert out[["odds_H", "odds_D", "odds_A"]].sum(axis=1).to_numpy() == pytest.approx(1.0)


def test_market_prefers_pinnacle_when_present(football_data_csv):
    out = add_market_probabilities(_fixture_frame(football_data_csv))
    assert (out["odds_source"] == "pinnacle").all()


def test_market_falls_back_when_the_preferred_book_is_absent(football_data_csv):
    """Early seasons predate Pinnacle; those rows must still get a price."""
    raw = _fixture_frame(football_data_csv).drop(columns=["PSH", "PSD", "PSA"])
    out = add_market_probabilities(raw)
    assert (out["odds_source"] == "bet365").all()
    assert out["odds_H"].notna().all()


def test_overround_is_recorded_and_exceeds_one(football_data_csv):
    """The bookmaker margin. At or below 1.0 means we misread a column."""
    out = add_market_probabilities(_fixture_frame(football_data_csv))
    assert (out["overround"] > 1.0).all()
    assert (out["overround"] < 1.5).all()


def test_market_leaves_unpriced_rows_as_nan():
    df = pd.DataFrame({"HomeTeam": ["A"], "AwayTeam": ["B"], "B365H": [np.nan]})
    out = add_market_probabilities(df)
    assert out["odds_source"].isna().all()
    assert out["odds_H"].isna().all()


def test_market_ignores_placeholder_odds_of_one():
    """Odds of 1.00 pay nothing; treating them as a price implies p = 1.0."""
    df = pd.DataFrame(
        {
            "HomeTeam": ["A"],
            "AwayTeam": ["B"],
            "PSH": [1.0],
            "PSD": [1.0],
            "PSA": [1.0],
            "B365H": [2.0],
            "B365D": [3.5],
            "B365A": [4.0],
        }
    )
    out = add_market_probabilities(df)
    assert out["odds_source"].iloc[0] == "bet365"


def test_market_probabilities_rank_the_favourite_highest(football_data_csv):
    """A shorter price must map to a higher probability, not a lower one."""
    out = add_market_probabilities(_fixture_frame(football_data_csv))
    shortest = out["PSH"].idxmin()
    assert out.loc[shortest, "odds_H"] == out["odds_H"].max()


# --- team canonicalisation --------------------------------------------------


def test_normalise_folds_accents_and_case():
    assert normalise("Atlético Madrid") == normalise("Atletico Madrid") == "atletico madrid"
    assert normalise("M'gladbach") == "m gladbach"


def test_aliases_resolve_across_sources():
    """Three sources spell this club three ways; all must land in one place."""
    assert (
        canonical_team("Man United")
        == canonical_team("Manchester Utd")
        == canonical_team("ManUnited")
        == "Manchester United"
    )


def test_normalise_keeps_distinct_clubs_distinct():
    """Paris FC and Paris SG are different clubs and share a Ligue 1 table.

    This is why `normalise` does not strip "FC": doing so would reduce the first
    to "paris", one careless alias away from being merged into the second.
    """
    assert normalise("Paris FC") != normalise("Paris SG")
    assert canonical_team("Paris FC") != canonical_team("Paris SG")


def test_unknown_clubs_pass_through_unchanged():
    """A promoted club with no alias entry must not vanish or raise."""
    assert canonical_team("Newly Promoted FC") == "Newly Promoted FC"


def test_known_divergent_spellings_are_all_mapped():
    """Guards the alias file against regressions as it is edited."""
    divergent = ["Man United", "Ath Bilbao", "M'gladbach", "Paris SG", "Inter", "Ein Frankfurt"]
    assert unmapped(divergent) == []


def test_documentation_keys_are_not_treated_as_clubs():
    """The alias file carries a leading "_comment"; it must not become an alias."""
    assert canonical_team("comment") == "comment"


# --- canonical table --------------------------------------------------------


def _raw_frame(csv: str) -> pd.DataFrame:
    df = _fixture_frame(csv)
    return df.assign(
        division="E0",
        league="ENG-Premier League",
        season=2019,
        date=_parse_dates(df["Date"]),
    )


def test_build_emits_exactly_the_canonical_schema(football_data_csv):
    out = build(_raw_frame(football_data_csv))
    assert list(out.columns) == CANONICAL_COLUMNS
    assert len(out) == 3


def test_build_canonicalises_team_names(football_data_csv):
    out = build(_raw_frame(football_data_csv))
    assert "Manchester United" in out["away"].tolist()
    assert "Man United" not in out["away"].tolist()


def test_build_ids_are_stable_across_row_order(football_data_csv):
    """Ids key the feature cache, so they must not depend on row order."""
    first = build(_raw_frame(football_data_csv))
    shuffled = _raw_frame(football_data_csv).iloc[::-1].reset_index(drop=True)
    second = build(shuffled)
    assert first["match_id"].tolist() == second["match_id"].tolist()
    assert first["match_id"].is_unique


def test_build_keeps_unplayed_fixtures_with_a_null_result(football_data_csv):
    """A postponed match still shapes rest-day features, so the row must survive."""
    raw = _raw_frame(football_data_csv)
    raw.loc[0, ["FTHG", "FTAG"]] = np.nan
    out = build(raw)
    assert len(out) == 3
    assert out["result"].isna().sum() == 1


def test_build_odds_stay_attached_to_the_right_match(football_data_csv):
    """The odds ride along through filtering and sorting.

    This is the failure this test exists for: splicing odds in after the rows
    were filtered and re-sorted would attach every price to the wrong match and
    nothing downstream would raise — the market baseline would simply score
    badly for no visible reason.
    """
    out = build(_raw_frame(football_data_csv))
    favourite = out.loc[out["odds_H"].idxmax()]
    assert favourite["home"] == "Manchester City"
    assert favourite["away"] == "Everton"


def test_build_drops_rows_with_an_unrecognised_result(football_data_csv):
    raw = _raw_frame(football_data_csv)
    raw.loc[1, "FTR"] = "X"
    out = build(raw)
    assert len(out) == 2
    assert set(out["result"].dropna()) <= {"H", "D", "A"}


# --- mixed-dtype harmonisation ----------------------------------------------


def test_harmonise_coerces_a_column_one_bad_cell_turned_into_text():
    """All four poison values were observed in real files, not invented.

    One junk cell makes pandas read that column as text for the whole file while
    the other 150+ files read it as float, so the concatenation is an object
    column and `to_parquet` refuses it.
    """
    clean = [f"{v:.2f}" for v in np.linspace(1.2, 9.5, 49)]
    df = pd.DataFrame(
        {
            "PSCH": ["#REF!", *clean],
            "B365CH": ["#", *clean],
            "1XBH": ["1xBet", *clean],
            "BbAH": ["8 ", *clean],
        }
    )
    out = _harmonise_numeric(df.copy())

    assert all(out[c].dtype.kind == "f" for c in df.columns)
    assert out["PSCH"].tolist()[1] == pytest.approx(1.2)
    # Only the poisoned cell is lost; the other 49 survive.
    assert all(out[c].isna().sum() == 1 for c in df.columns)


def test_harmonise_leaves_a_column_that_is_mostly_damaged_alone():
    """The 90% share is a guard: a column that is half junk is not a number
    column with a bad cell, it is something this parser does not understand,
    and quietly turning most of it into NaN would hide that."""
    df = pd.DataFrame({"Mystery": ["1.5", "n/k", "2.0", "void", "3.0", "tbc"]})
    out = _harmonise_numeric(df.copy())
    assert out["Mystery"].tolist() == df["Mystery"].tolist()


def test_harmonise_leaves_genuine_text_columns_alone():
    """Team names must survive: coercing them would erase the whole column."""
    df = pd.DataFrame(
        {"HomeTeam": ["Arsenal", "Celtic", "Ajax"], "FTR": ["H", "D", "A"]}
    )
    out = _harmonise_numeric(df.copy())
    assert out["HomeTeam"].tolist() == ["Arsenal", "Celtic", "Ajax"]
    assert out["FTR"].tolist() == ["H", "D", "A"]


def test_harmonise_keeps_the_odds_columns_readable_by_the_odds_layer():
    """The point of the coercion: `PSCH` left as text fails `_valid` silently.

    A text Pinnacle column does not raise — it simply never matches, and every
    affected match falls through to a less sharp bookmaker.
    """
    raw = pd.DataFrame(
        {"PSCH": ["2.00", "#REF!"], "PSCD": [3.5, 3.5], "PSCA": [4.0, 4.0]}
    )
    priced = add_market_probabilities(_harmonise_numeric(raw.copy()))
    assert priced.loc[0, "odds_source"] == "pinnacle_closing"
    assert pd.isna(priced.loc[1, "odds_source"])  # the damaged row, honestly unpriced
