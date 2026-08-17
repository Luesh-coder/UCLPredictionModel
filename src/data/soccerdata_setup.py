"""Configure `soccerdata` before it is imported anywhere else.

`soccerdata` ships with only eight leagues built in, and the Champions League is
not one of them. It does read extra competitions from a `league_dict.json` under
`$SOCCERDATA_DIR/config/`, so this module writes that file (pointing at the
in-project cache) and hands back a ready-to-use reader.

Import order matters: `soccerdata._config` reads `SOCCERDATA_DIR` and the league
dict at import time. Always call `ensure_soccerdata_config()` *before*
`import soccerdata`, which is why every reader in this package is constructed
through the helpers here.
"""

from __future__ import annotations

import json
import logging
import os

from src.config import CACHE_DIR, UCL_LEAGUE

log = logging.getLogger(__name__)

# The competition name exactly as it appears on FBref's /en/comps/ index page.
# soccerdata matches on this string, so a typo yields a silently empty frame.
LEAGUE_DICT = {
    UCL_LEAGUE: {
        "FBref": "UEFA Champions League",
        "season_start": "Aug",
        "season_end": "Jun",
    },
    "UEFA-Europa League": {
        "FBref": "UEFA Europa League",
        "season_start": "Aug",
        "season_end": "Jun",
    },
}

CONFIG_PATH = CACHE_DIR / "config" / "league_dict.json"


def ensure_soccerdata_config() -> None:
    """Point soccerdata at the project cache and register the UEFA competitions."""
    os.environ["SOCCERDATA_DIR"] = str(CACHE_DIR)
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    existing = {}
    if CONFIG_PATH.exists():
        try:
            existing = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log.warning("Ignoring malformed %s", CONFIG_PATH)

    merged = {**existing, **LEAGUE_DICT}
    if merged != existing:
        CONFIG_PATH.write_text(json.dumps(merged, indent=2), encoding="utf-8")
        log.info("Wrote league config to %s", CONFIG_PATH)


def fbref(seasons, leagues: str | list[str] = UCL_LEAGUE, **kwargs):
    """Return a configured `soccerdata.FBref` reader.

    FBref sits behind Cloudflare, so soccerdata drives a real Chrome via
    seleniumbase. The first call downloads a matching chromedriver, and each
    request is deliberately rate-limited — a full backfill takes minutes, not
    seconds. Results are cached under `data/cache`, so pay that cost once.
    """
    ensure_soccerdata_config()
    import soccerdata as sd

    return sd.FBref(leagues=leagues, seasons=seasons, **kwargs)


def club_elo(**kwargs):
    """Return a `soccerdata.ClubElo` reader (clubelo.net, plain HTTP, fast)."""
    ensure_soccerdata_config()
    import soccerdata as sd

    return sd.ClubElo(**kwargs)
