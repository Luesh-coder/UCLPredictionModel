"""ClubElo rating snapshots — an optional cross-check, never a dependency.

clubelo.net publishes a plain-CSV endpoint returning every club's rating on a
given date. That is genuinely useful here: unlike `src.features.elo`, which sees
only the matches in our dataset, ClubElo's ratings span every competition a club
plays, so they carry information our per-league pools structurally cannot.

**As of 2026-09-10 the API is returning HTTP 502 on every endpoint, and
`/Fixtures` responds with the literal text "Fixtures API deactivated".** This
module is therefore written to fail soft: every entry point returns an empty
frame on failure rather than raising, and nothing downstream may require its
output. If the service comes back, it starts working again with no code change.
"""

from __future__ import annotations

import io
import logging

import pandas as pd
import requests

from src.config import RAW_DIR

log = logging.getLogger(__name__)

API_URL = "http://api.clubelo.com/{path}"
SNAPSHOT_PATH = RAW_DIR / "clubelo_snapshots.parquet"
TIMEOUT = 20


def _get(path: str) -> pd.DataFrame:
    """Fetch one ClubElo endpoint, returning an empty frame on any failure."""
    url = API_URL.format(path=path)
    try:
        response = requests.get(url, timeout=TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as exc:
        log.warning("ClubElo %s unavailable (%s)", path, exc)
        return pd.DataFrame()

    text = response.text
    # The service answers some deactivated endpoints with a 200 and a prose
    # message rather than an error status — "/Fixtures" currently replies
    # "Fixtures API deactivated". `pd.read_csv` would parse that into a
    # one-column frame rather than failing, so check for a real CSV header.
    first_line = text.splitlines()[0] if text.strip() else ""
    if "," not in first_line:
        log.warning("ClubElo %s returned no CSV: %r", path, text[:80])
        return pd.DataFrame()

    try:
        return pd.read_csv(io.StringIO(text))
    except (pd.errors.ParserError, ValueError) as exc:
        log.warning("ClubElo %s returned unparseable CSV (%s)", path, exc)
        return pd.DataFrame()


def fetch_snapshots(dates: list[str], *, refresh: bool = False) -> pd.DataFrame:
    """Ratings for every club on each of `dates` (YYYY-MM-DD).

    Returns an empty frame if the service is unreachable — check `.empty` at the
    call site rather than assuming columns exist.
    """
    cached = pd.DataFrame()
    if SNAPSHOT_PATH.exists() and not refresh:
        cached = pd.read_parquet(SNAPSHOT_PATH)
        have = set(cached["snapshot_date"].astype(str))
        dates = [d for d in dates if d not in have]
        if not dates:
            return cached

    frames = [cached] if not cached.empty else []
    for date in dates:
        snap = _get(date)
        if snap.empty:
            continue
        frames.append(snap.assign(snapshot_date=pd.Timestamp(date)))

    if not frames:
        log.warning("No ClubElo snapshots retrieved; continuing without them.")
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(SNAPSHOT_PATH, index=False)
    log.info("Wrote %d ClubElo rows to %s", len(out), SNAPSHOT_PATH)
    return out


def is_available() -> bool:
    """Cheap liveness probe, so a caller can skip the feature entirely."""
    return not _get("2020-01-01").empty
