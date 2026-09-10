"""FBref Champions League scraping — Stage 2, wired but not yet on the path.

FBref is the only source in this project that actually contains Champions League
results, which makes it unavoidable for Stage 2 and useless for Stage 1. It is
also the most awkward: a direct `requests.get` returns **HTTP 403** (verified
2026-09-10) because the site sits behind Cloudflare, so every fetch goes through
`soccerdata`'s browser backend and a full backfill takes minutes, not seconds.

Two wrinkles this module exists to absorb:

- `soccerdata` does not ship the Champions League as a built-in league, so it
  has to be registered in the league dict before any reader will accept it.
- One reader instance is reused across the whole season list. `soccerdata`
  keeps a single browser session per reader; constructing one reader per season
  launches a fresh Chrome each time and those sessions die under the load.

Nothing in Stage 1 imports this. It is here so Stage 2 starts from working code
rather than a blank file.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from src.config import CACHE_DIR, RAW_DIR, UCL_LEAGUE

log = logging.getLogger(__name__)

SCHEDULE_PATH = RAW_DIR / "fbref_ucl_schedule.parquet"

# soccerdata reads this file at import time to learn about non-built-in leagues.
LEAGUE_DICT_PATH = Path.home() / "soccerdata" / "config" / "league_dict.json"


def register_ucl() -> None:
    """Teach `soccerdata` about the Champions League.

    Idempotent: re-registering an already-known league rewrites the same entry.
    """
    LEAGUE_DICT_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if LEAGUE_DICT_PATH.exists():
        existing = json.loads(LEAGUE_DICT_PATH.read_text(encoding="utf-8"))

    if UCL_LEAGUE not in existing:
        existing[UCL_LEAGUE] = {
            "FBref": "Champions League",
            "season_start": "Aug",
            "season_end": "May",
        }
        LEAGUE_DICT_PATH.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        log.info("Registered %s in %s", UCL_LEAGUE, LEAGUE_DICT_PATH)


def fetch_schedule(seasons: list[int], *, refresh: bool = False) -> pd.DataFrame:
    """Scrape the UCL schedule for `seasons` (season start years).

    Falls back to one season at a time if the batch scrape fails, so a single
    bad season cannot cost an entire backfill.
    """
    import soccerdata as sd

    register_ucl()

    if SCHEDULE_PATH.exists() and not refresh:
        cached = pd.read_parquet(SCHEDULE_PATH)
        missing = [s for s in seasons if s not in set(cached["season"].unique())]
        if not missing:
            return cached[cached["season"].isin(seasons)].reset_index(drop=True)
        log.info("Cached %s; scraping %s", sorted(set(cached["season"])), missing)
        seasons = missing
        frames = [cached]
    else:
        frames = []

    def _reader(s):
        return sd.FBref(leagues=UCL_LEAGUE, seasons=s, data_dir=CACHE_DIR / "FBref")

    try:
        log.info("Scraping UCL seasons %s in one session ...", seasons)
        frames.append(_reader(seasons).read_schedule().reset_index())
    except Exception:
        log.exception("Batch scrape failed; retrying season by season.")
        for season in seasons:
            try:
                frames.append(_reader(season).read_schedule().reset_index())
            except Exception:
                log.exception("Season %s failed; continuing.", season)

    if not frames:
        raise RuntimeError("no FBref schedule data could be fetched")

    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(SCHEDULE_PATH, index=False)
    return out
