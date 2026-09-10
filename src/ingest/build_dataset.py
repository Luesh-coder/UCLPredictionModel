"""Fold raw source columns into the one match table everything downstream reads.

football-data.co.uk ships 130+ columns whose names and presence drift by season.
Nothing outside this module should have to know that. The canonical table is
narrow, stable and identically shaped whether a row came from a 2012 CSV with 77
columns or a 2025 one with 136 — and, in Stage 2, whether it came from FBref
instead.
"""

from __future__ import annotations

import hashlib
import logging

import pandas as pd

from src.config import OUTCOMES, PROCESSED_DIR
from src.ingest.odds import add_market_probabilities
from src.ingest.teams import add_canonical_teams

log = logging.getLogger(__name__)

MATCHES_PATH = PROCESSED_DIR / "matches.parquet"

CANONICAL_COLUMNS = [
    "match_id",
    "date",
    "season",
    "league",
    "division",
    "stage",
    "home",
    "away",
    "home_goals",
    "away_goals",
    "result",
    "odds_H",
    "odds_D",
    "odds_A",
    "odds_source",
    "overround",
]


def _match_id(row: pd.Series) -> str:
    """Stable id derived from the match's identity, not its position.

    A row-number id would change the moment a season is re-fetched or a division
    is added, silently invalidating every cached feature keyed on it. Hashing the
    identifying fields means the same match always gets the same id.
    """
    key = f"{row['season']}|{row['division']}|{row['date']:%Y-%m-%d}|{row['home']}|{row['away']}"
    return hashlib.blake2s(key.encode("utf-8"), digest_size=8).hexdigest()


def build(raw: pd.DataFrame) -> pd.DataFrame:
    """Return the canonical match table from a raw football-data frame."""
    df = raw.copy()

    required = ["HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "date", "season", "division"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"raw frame is missing {missing}")

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df["date"]),
            "season": df["season"].astype(int),
            "league": df["league"].astype(str),
            "division": df["division"].astype(str),
            # Domestic league matches have no stage. The column exists so that
            # Stage 2's UCL rows ("group stage", "final", ...) share this schema.
            "stage": "league",
            "home": df["HomeTeam"].astype(str),
            "away": df["AwayTeam"].astype(str),
            "home_goals": pd.to_numeric(df["FTHG"], errors="coerce"),
            "away_goals": pd.to_numeric(df["FTAG"], errors="coerce"),
            "result": df["FTR"].astype(str).str.strip().str.upper(),
        },
        index=df.index,
    )

    # Carry the raw bookmaker columns along from the start, on the *same* index.
    # Splicing them in after the filtering below would align a full-length odds
    # frame against a shortened match frame and quietly attach the wrong prices.
    out = pd.concat([out, df[_odds_columns(df)]], axis=1)

    # A postponed or abandoned match has a row but no score. Keeping it with a
    # NaN result is deliberate: the feature layer needs the fixture to exist to
    # compute rest days, while the model layer filters on `result.notna()`.
    unplayed = out["home_goals"].isna() | out["away_goals"].isna()
    out.loc[unplayed, "result"] = pd.NA

    bad = out["result"].notna() & ~out["result"].isin(OUTCOMES)
    if bad.any():
        log.warning(
            "Dropping %d rows with unrecognised results: %s",
            int(bad.sum()),
            sorted(out.loc[bad, "result"].dropna().unique())[:5],
        )
        out = out[~bad]

    out = add_canonical_teams(out)
    out = out[out["date"].notna() & (out["home"] != "") & (out["away"] != "")]

    # Sorting before assigning ids keeps the on-disk table chronological, which
    # every downstream sort then gets for free.
    out = out.sort_values(["date", "league", "home"], kind="stable").reset_index(drop=True)
    out.insert(0, "match_id", out.apply(_match_id, axis=1))

    dupes = out["match_id"].duplicated()
    if dupes.any():
        log.warning("Dropping %d duplicate matches", int(dupes.sum()))
        out = out[~dupes].reset_index(drop=True)

    out = add_market_probabilities(out)
    return out[CANONICAL_COLUMNS].reset_index(drop=True)


def _odds_columns(df: pd.DataFrame) -> list[str]:
    """The raw bookmaker columns the odds layer knows how to read."""
    from src.ingest.odds import ODDS_PREFERENCES

    wanted = [c for _, cols in ODDS_PREFERENCES for c in cols]
    return [c for c in wanted if c in df.columns]


def save(matches: pd.DataFrame) -> pd.DataFrame:
    matches.to_parquet(MATCHES_PATH, index=False)
    log.info("Wrote %d matches to %s", len(matches), MATCHES_PATH)
    return matches


def load() -> pd.DataFrame:
    if not MATCHES_PATH.exists():
        raise FileNotFoundError(
            f"{MATCHES_PATH} not found — run `python pipeline.py dataset` first"
        )
    return pd.read_parquet(MATCHES_PATH)
