"""Download match results and bookmaker odds from football-data.co.uk.

The site publishes one static CSV per (season, division) at a predictable URL.
Unlike FBref there is no Cloudflare in the way, so plain `requests` is enough —
no browser, no rate limiting, a full 14-season backfill in under a minute.

We fetch the URL directly rather than going through `soccerdata.MatchHistory`,
which wraps the same files: `MatchHistory.available_leagues()` exposes only the
big five, while the underlying site publishes a couple of dozen more divisions. Going
direct keeps those reachable when the model later wants Eredivisie or Primeira
Liga sides, which reach the Champions League every year.

Every response is cached to `data/raw/football_data/` so re-runs are offline.
"""

from __future__ import annotations

import io
import logging

import pandas as pd
import requests

from src.config import DIVISION_NAMES, FOOTBALL_DATA_URL, RAW_DIR

log = logging.getLogger(__name__)

CACHE_DIR = RAW_DIR / "football_data"
COMBINED_PATH = RAW_DIR / "football_data.parquet"

TIMEOUT = 30


def season_code(season_start: int) -> str:
    """2024 -> "2425", the four-digit form football-data uses in its URLs."""
    return f"{season_start % 100:02d}{(season_start + 1) % 100:02d}"


def _decode(payload: bytes) -> str:
    """Decode a CSV, tolerating both encodings the site has used over the years.

    Recent files are UTF-8 with a byte-order mark. Read as plain UTF-8, those
    three leading bytes stay glued to the first column name, so it arrives as
    something that is not "Div" and every lookup of "Div" fails. Older files
    predate the BOM and are Latin-1, where UTF-8 decoding raises outright.
    """
    try:
        return payload.decode("utf-8-sig")
    except UnicodeDecodeError:
        log.debug("UTF-8 decode failed; falling back to latin-1")
        return payload.decode("latin-1")


def _parse_dates(raw: pd.Series) -> pd.Series:
    """Parse football-data's day-first dates, which changed format mid-history.

    Both "07/08/2015" and "07/08/15" occur, sometimes within one season. Trying
    the four-digit form first and filling the failures from the two-digit form
    is more predictable than `format="mixed"`, which infers per row and can read
    an unambiguous day-first date as month-first.
    """
    text = raw.astype("string").str.strip()
    parsed = pd.to_datetime(text, format="%d/%m/%Y", errors="coerce")
    short = parsed.isna() & text.notna()
    if short.any():
        two_digit = pd.to_datetime(text.where(short), format="%d/%m/%y", errors="coerce")
        parsed = parsed.fillna(two_digit)
    return parsed


def _read_csv(text: str, source: str) -> pd.DataFrame:
    """Parse one division-season CSV, reporting rows the parser had to drop.

    Older files carry trailing blank lines and the occasional row with more
    fields than the header. Skipping them is right, but skipping them *silently*
    would hide a real truncation, so the count is logged.
    """
    df = pd.read_csv(io.StringIO(text), on_bad_lines="skip", low_memory=False)

    # A data line is one that actually starts with a division code; this ignores
    # the header and the blank tail, giving a count to compare the parse against.
    data_lines = sum(
        1 for line in text.splitlines()[1:] if line.strip() and not line.startswith(",")
    )
    if data_lines > len(df):
        log.warning("%s: %d data lines but %d parsed rows", source, data_lines, len(df))

    if "HomeTeam" not in df.columns:
        raise ValueError(f"{source}: no HomeTeam column; columns were {list(df.columns)[:10]}")

    # Rows without a home team are the blank tail, not matches.
    return df[df["HomeTeam"].notna()].reset_index(drop=True)


def fetch_division_season(
    division: str, season_start: int, *, refresh: bool = False
) -> pd.DataFrame:
    """Return one division-season, from cache unless `refresh`."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    code = season_code(season_start)
    path = CACHE_DIR / f"{code}_{division}.csv"

    if path.exists() and not refresh:
        text = _decode(path.read_bytes())
    else:
        url = FOOTBALL_DATA_URL.format(season=code, div=division)
        log.info("GET %s", url)
        response = requests.get(url, timeout=TIMEOUT)
        response.raise_for_status()
        path.write_bytes(response.content)
        text = _decode(response.content)

    df = _read_csv(text, f"{division} {season_start}")
    # Concatenated in one shot rather than assigned column by column: these
    # frames carry 130+ odds columns, and inserting into a frame that wide four
    # times over makes pandas rebuild its block manager on every insert.
    meta = pd.DataFrame(
        {
            "division": division,
            "league": DIVISION_NAMES.get(division, division),
            "season": season_start,
            "date": _parse_dates(df["Date"]),
        },
        index=df.index,
    )
    return pd.concat([df, meta], axis=1)


# Columns that are numeric in intent. Anything else — team names, referees,
# result letters — must never be coerced, so the decision is made from the data
# rather than from a hand-maintained list of the 199 columns the site publishes.
_NUMERIC_SHARE = 0.9


def _harmonise_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce odds/statistic columns that one bad cell turned into text.

    A single junk cell makes pandas read that column as text *for the whole
    file* — observed in the wild as ``#REF!`` (a broken Excel reference), a
    stray ``#``, a header row leaked into the data (``1xBet``) and a digit with
    a trailing non-breaking space. The other 150+ files parse the same column as
    float, so concatenating yields a mixed-type object column, and `to_parquet`
    fails on it.

    Coercion is decided per column by what the values look like: a column whose
    non-null entries are at least 90% numeric is one, and the remainder is
    damage. Team names and result letters are 0% numeric and are left alone.

    This matters beyond the write. ``PSCH`` and ``B365CH`` are in
    `odds.ODDS_PREFERENCES`; left as text they would fail `odds._valid` and
    those matches would quietly fall through to a less sharp bookmaker.
    """
    for col in df.columns:
        if df[col].dtype.kind not in {"O", "U", "T"}:
            continue
        values = df[col]
        present = values.notna()
        if not present.any():
            continue
        coerced = pd.to_numeric(values, errors="coerce")
        parsed = coerced.notna()
        if parsed.sum() < _NUMERIC_SHARE * present.sum():
            continue  # genuinely a text column

        lost = present & ~parsed
        if lost.any():
            log.warning(
                "%s: %d non-numeric cell(s) coerced to NaN: %s",
                col,
                int(lost.sum()),
                sorted({repr(v) for v in values[lost].unique()})[:5],
            )
        df[col] = coerced
    return df


def fetch(
    divisions: list[str],
    seasons: list[int],
    *,
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch every (division, season) pair and concatenate.

    A single missing file is logged and skipped rather than raised: football-data
    does not publish every division for every season, and one gap must not cost
    the whole backfill.
    """
    frames = []
    for division in divisions:
        for season in seasons:
            try:
                frames.append(fetch_division_season(division, season, refresh=refresh))
            except requests.HTTPError as exc:
                log.warning("%s %s unavailable (%s); skipping", division, season, exc)
            except ValueError as exc:
                log.warning("%s %s unparseable (%s); skipping", division, season, exc)

    if not frames:
        raise RuntimeError("no football-data CSVs could be fetched")

    # Column sets differ across seasons as bookmakers come and go, so the union
    # is intentional; missing odds columns become NaN and the odds layer falls
    # back to whichever bookmaker that season actually has.
    out = _harmonise_numeric(pd.concat(frames, ignore_index=True, sort=False))
    out.to_parquet(COMBINED_PATH, index=False)
    log.info(
        "Wrote %d matches across %d divisions and %d seasons to %s",
        len(out),
        out["division"].nunique(),
        out["season"].nunique(),
        COMBINED_PATH,
    )
    return out
