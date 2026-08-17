"""Turn the raw FBref schedule into one canonical, tidy match table.

The output of `build_match_table` is the single source of truth every feature
and model in this project reads. Its contract:

    season      int    calendar year the season started (2024 = 2024/25)
    date        datetime64[ns]
    stage       str    "league phase" / "group stage" / "round of 16" / ...
    home, away  str    team names, normalised
    home_goals, away_goals  Int64 (nullable — NA for unplayed fixtures)
    result      str    "H" / "D" / "A" (NA for unplayed fixtures)

Rows are sorted by date, which every downstream rolling/backtest routine
depends on.
"""

from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR
from src.data.ingest import fetch_schedule, season_to_start_year

log = logging.getLogger(__name__)

MATCH_TABLE_PATH = PROCESSED_DIR / "matches.parquet"

# FBref is not consistent about club naming across seasons; collapse the variants
# so a team's history is one continuous series rather than two broken ones.
TEAM_ALIASES = {
    "Manchester Utd": "Manchester United",
    "Manchester City": "Manchester City",
    "Newcastle Utd": "Newcastle United",
    "Paris S-G": "Paris Saint-Germain",
    "Paris SG": "Paris Saint-Germain",
    "Atletico Madrid": "Atlético Madrid",
    "Atlético": "Atlético Madrid",
    "Betis": "Real Betis",
    "Dortmund": "Borussia Dortmund",
    "Gladbach": "Borussia Mönchengladbach",
    "M'Gladbach": "Borussia Mönchengladbach",
    "Leverkusen": "Bayer Leverkusen",
    "Bayern Munich": "Bayern München",
    "Eint Frankfurt": "Eintracht Frankfurt",
    "RB Leipzig": "RB Leipzig",
    "Leipzig": "RB Leipzig",
    "Inter": "Internazionale",
    "Sporting CP": "Sporting CP",
    "Shakhtar": "Shakhtar Donetsk",
    "Zenit": "Zenit St. Petersburg",
    "Red Star": "Crvena Zvezda",
    "Salzburg": "RB Salzburg",
    "Young Boys": "BSC Young Boys",
    "Club Brugge": "Club Brugge",
    "Copenhagen": "FC Copenhagen",
    "Dynamo Kyiv": "Dynamo Kyiv",
    "Olympiacos": "Olympiacos",
    "Slavia Prague": "Slavia Prague",
}


def normalise_team(name: object) -> object:
    if not isinstance(name, str):
        return name
    name = re.sub(r"\s+", " ", name).strip()
    # FBref prefixes some names with a country code, e.g. "eng Liverpool".
    name = re.sub(r"^[a-z]{2,3} ", "", name)
    return TEAM_ALIASES.get(name, name)


def _pick(df: pd.DataFrame, *candidates: str) -> pd.Series:
    """Return the first column present in `df` out of `candidates`."""
    for c in candidates:
        if c in df.columns:
            return df[c]
    raise KeyError(f"none of {candidates} found in columns {list(df.columns)}")


def _split_score(score: pd.Series) -> pd.DataFrame:
    """Parse FBref's "2–1" score strings into integer goal columns.

    FBref uses an en dash, and appends penalty shootout scores in parentheses
    for knockout ties — "1–1 (4–3)". Shootouts are stripped: for prediction
    purposes the match itself was a draw.
    """
    text = score.astype("string").str.replace(r"\(.*\)", "", regex=True).str.strip()
    parts = text.str.extract(r"^(\d+)\s*[–\-−:]\s*(\d+)$")
    return pd.DataFrame(
        {
            "home_goals": pd.to_numeric(parts[0], errors="coerce").astype("Int64"),
            "away_goals": pd.to_numeric(parts[1], errors="coerce").astype("Int64"),
        },
        index=score.index,
    )


def _result_from_goals(home: pd.Series, away: pd.Series) -> pd.Series:
    res = pd.Series(pd.NA, index=home.index, dtype="string")
    played = home.notna() & away.notna()
    res[played & (home > away)] = "H"
    res[played & (home == away)] = "D"
    res[played & (home < away)] = "A"
    return res


def build_match_table(
    seasons: list[int] | None = None,
    *,
    refresh: bool = False,
    save: bool = True,
) -> pd.DataFrame:
    """Build (and optionally persist) the canonical match table."""
    raw = fetch_schedule(seasons, refresh=refresh)

    df = pd.DataFrame(index=raw.index)
    # `season_start` is stamped on by the ingest loop; fall back to parsing
    # FBref's own "2425"-style label if we are reading an older cache.
    if "season_start" in raw.columns:
        df["season"] = raw["season_start"].astype("Int64")
    else:
        df["season"] = season_to_start_year(_pick(raw, "season"))
    df["date"] = pd.to_datetime(_pick(raw, "date"), errors="coerce")
    df["stage"] = _pick(raw, "stage", "round", "week").astype("string").str.lower().str.strip()
    df["home"] = _pick(raw, "home_team", "home").map(normalise_team)
    df["away"] = _pick(raw, "away_team", "away").map(normalise_team)
    for col in ("game_id", "venue", "attendance", "referee"):
        if col in raw.columns:
            df[col] = raw[col]

    if "home_goals" in raw.columns and "away_goals" in raw.columns:
        df["home_goals"] = pd.to_numeric(raw["home_goals"], errors="coerce").astype("Int64")
        df["away_goals"] = pd.to_numeric(raw["away_goals"], errors="coerce").astype("Int64")
    else:
        df[["home_goals", "away_goals"]] = _split_score(_pick(raw, "score"))

    df["result"] = _result_from_goals(df["home_goals"], df["away_goals"])

    # A fixture with no date is unusable: rolling features and the walk-forward
    # backtest are both ordered by date.
    before = len(df)
    df = df.dropna(subset=["date", "home", "away"])
    if len(df) < before:
        log.warning("Dropped %d rows with no date or team names.", before - len(df))

    df = df[df["home"] != df["away"]]
    df = df.sort_values("date", kind="stable").reset_index(drop=True)
    df["match_id"] = np.arange(len(df))

    if save:
        df.to_parquet(MATCH_TABLE_PATH, index=False)
        log.info("Wrote %d matches to %s", len(df), MATCH_TABLE_PATH)
    return df


def load_match_table(*, played_only: bool = True) -> pd.DataFrame:
    """Load the persisted match table, building it first if it is missing."""
    if MATCH_TABLE_PATH.exists():
        df = pd.read_parquet(MATCH_TABLE_PATH)
    else:
        df = build_match_table()
    if played_only:
        df = df[df["result"].notna()].reset_index(drop=True)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    table = build_match_table()
    print(table.shape)
    print(table.head(10).to_string())
    print(table["result"].value_counts(dropna=False))
