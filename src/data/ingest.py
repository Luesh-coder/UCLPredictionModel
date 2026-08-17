"""Pull raw Champions League match data from FBref via `soccerdata`.

FBref is behind Cloudflare and soccerdata rate-limits itself, so a full backfill
takes minutes. Every fetch is therefore cached to `data/raw` as parquet (on top
of soccerdata's own HTML cache in `data/cache`). Pass ``refresh=True`` to
re-scrape.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import DEFAULT_SEASONS, RAW_DIR
from src.data.soccerdata_setup import club_elo, fbref

log = logging.getLogger(__name__)

SCHEDULE_PATH = RAW_DIR / "fbref_schedule.parquet"
ELO_PATH = RAW_DIR / "clubelo_snapshots.parquet"


def season_to_start_year(season: pd.Series) -> pd.Series:
    """Map any season label soccerdata emits to the calendar year it started.

    Handles "1617"/"2425" (a two-year pair), a bare "2016", "2024-25" and
    "2024-2025". Anything unrecognised becomes NA rather than raising, so one
    odd label cannot take down a whole backfill.

    Every conversion is applied to the matching subset only. Running `.astype`
    over the full column instead would blow up on the first row that does not
    fit the pattern being tested, whatever the mask says.
    """
    s = season.astype("string").str.strip()
    out = pd.Series(pd.NA, index=s.index, dtype="Int64")

    four = s.str.fullmatch(r"\d{4}").fillna(False)
    if four.any():
        head = s[four].str[:2].astype("Int64")
        tail = s[four].str[2:].astype("Int64")
        # "2425" is a season pair; a bare "2024" is already the start year.
        is_pair = head + 1 == tail
        out[four] = np.where(is_pair, 2000 + head, s[four].astype("Int64"))

    long_form = s.str.fullmatch(r"\d{4}[-/]\d{2,4}").fillna(False)
    if long_form.any():
        out[long_form] = s[long_form].str[:4].astype("Int64")

    two_digit = s.str.fullmatch(r"\d{2}").fillna(False)
    if two_digit.any():
        out[two_digit] = 2000 + s[two_digit].astype("Int64")

    return out


def _stamp_season(df: pd.DataFrame) -> pd.DataFrame:
    """Add `season_start` (the calendar year the season began) from FBref's label."""
    df["season_start"] = season_to_start_year(df["season"])
    return df


def fetch_schedule(
    seasons: list[int] | None = None,
    *,
    refresh: bool = False,
) -> pd.DataFrame:
    """Return the raw FBref schedule (one row per match) for `seasons`.

    All missing seasons go through a *single* reader. soccerdata reuses one
    browser session across the whole list; constructing a reader per season
    instead launches a fresh Chrome each time, and those sessions die under the
    load. If the batch fails, fall back to one season at a time so a single bad
    season cannot cost the entire backfill.
    """
    seasons = list(seasons or DEFAULT_SEASONS)

    cached = pd.DataFrame()
    if SCHEDULE_PATH.exists() and not refresh:
        cached = pd.read_parquet(SCHEDULE_PATH)
        have = set(cached["season_start"].unique())
        missing = [s for s in seasons if s not in have]
        if not missing:
            log.info("All %d seasons already cached in %s", len(seasons), SCHEDULE_PATH)
            return cached[cached["season_start"].isin(seasons)].reset_index(drop=True)
        log.info("Cached seasons %s; scraping %s", sorted(have), missing)
        seasons = missing

    frames = [cached] if not cached.empty else []
    try:
        log.info("Scraping seasons %s in one session ...", seasons)
        batch = fbref(seasons=seasons).read_schedule().reset_index()
        frames.append(_stamp_season(batch))
    except Exception:
        log.exception("Batch scrape failed; retrying season by season.")
        for season in seasons:
            log.info("Scraping season %s ...", season)
            try:
                df = fbref(seasons=season).read_schedule().reset_index()
            except Exception:
                log.exception("Season %s failed; continuing with the rest.", season)
                continue
            frames.append(_stamp_season(df))

    if not frames:
        raise RuntimeError("no schedule data could be fetched")

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset="game_id", keep="last").reset_index(drop=True)
    out.to_parquet(SCHEDULE_PATH, index=False)
    log.info(
        "Wrote %d rows covering %d seasons to %s",
        len(out),
        out["season_start"].nunique(),
        SCHEDULE_PATH,
    )
    return out


def fetch_club_elo(dates: list[str], *, refresh: bool = False) -> pd.DataFrame:
    """Fetch clubelo.net rating snapshots for the given dates (YYYY-MM-DD).

    ClubElo is a plain HTTP CSV endpoint — fast, and no browser needed. One
    request returns every club's rating on that date, so a handful of snapshots
    per season is enough to interpolate a strength signal that (unlike the Elo
    in `src.features.elo`) also reflects domestic form.
    """
    if ELO_PATH.exists() and not refresh:
        cached = pd.read_parquet(ELO_PATH)
        missing = [d for d in dates if d not in set(cached["snapshot_date"].astype(str))]
        if not missing:
            return cached
    else:
        cached = pd.DataFrame()
        missing = list(dates)

    reader = club_elo()
    frames = [cached] if not cached.empty else []
    for date in missing:
        log.info("ClubElo snapshot %s", date)
        try:
            snap = reader.read_by_date(date).reset_index()
        except Exception:
            log.exception("ClubElo snapshot %s failed; skipping.", date)
            continue
        snap["snapshot_date"] = pd.Timestamp(date)
        frames.append(snap)

    if not frames:
        raise RuntimeError("no ClubElo data could be fetched")

    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(ELO_PATH, index=False)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    df = fetch_schedule()
    print(df.shape)
    print(df.groupby("season_start").size())
