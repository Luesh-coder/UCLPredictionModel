"""Team name canonicalisation.

The same club is spelled differently by every source: football-data.co.uk says
``Man United``, FBref says ``Manchester Utd``, ClubElo says ``ManUnited``. Any
cross-source join is silently wrong until those collapse to one name, and the
failure mode is quiet — a bad join drops rows rather than raising.

Two layers do the work:

1. `normalise` strips differences that are purely cosmetic — case, accents and
   punctuation. That alone reconciles ``Atlético Madrid`` with ``Atletico
   Madrid`` and ``M'gladbach`` with ``Mgladbach``.
2. `ALIASES` handles everything else, because the strings genuinely differ:
   ``Man United`` -> ``Manchester United``.

`normalise` deliberately does *not* strip club-name noise words like "FC" or
"AC", tempting as it looks. Ligue 1 currently contains both ``Paris FC`` and
``Paris SG``; stripping "FC" collapses the first to ``paris``, one step from
being confused with the second. Explicit aliases are verbose but they cannot
merge two clubs behind your back.

Canonical names are the full, unabbreviated forms rather than football-data's
abbreviations, so that Stage 2's FBref names — which are already close to the
full forms — join cleanly against the same table.
"""

from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import pandas as pd

ALIASES_PATH = Path(__file__).with_name("team_aliases.json")


def normalise(name: str) -> str:
    """Reduce a club name to a matching key.

    Lowercases, strips accents and punctuation, collapses whitespace. The result
    is a lookup key, never something to display.
    """
    if not isinstance(name, str):
        return ""
    # NFKD splits "é" into "e" + a combining accent; dropping the combining
    # marks leaves plain ASCII without a per-character translation table.
    decomposed = unicodedata.normalize("NFKD", name)
    ascii_only = "".join(c for c in decomposed if not unicodedata.combining(c))
    cleaned = re.sub(r"[^a-z0-9\s]", " ", ascii_only.lower())
    return " ".join(cleaned.split())


@lru_cache(maxsize=1)
def _alias_index() -> dict[str, str]:
    """Alias file keyed by normalised form, so lookups survive spelling drift."""
    if not ALIASES_PATH.exists():
        return {}
    raw = json.loads(ALIASES_PATH.read_text(encoding="utf-8"))
    # Keys starting with "_" are documentation, not clubs.
    return {
        normalise(source): canonical
        for source, canonical in raw.items()
        if not source.startswith("_")
    }


def canonical_team(name: str) -> str:
    """Return the canonical club name for any source's spelling.

    Unknown names pass through with their original spelling rather than raising:
    a promoted club appearing for the first time is normal, and refusing to
    ingest a season over one unrecognised name would be worse than the
    inconsistency. Use `unmapped` to audit what fell through.
    """
    if not isinstance(name, str) or not name.strip():
        return ""
    return _alias_index().get(normalise(name), name.strip())


def add_canonical_teams(
    df: pd.DataFrame, columns: tuple[str, ...] = ("home", "away")
) -> pd.DataFrame:
    """Canonicalise the named team columns, returning a copy."""
    out = df.copy()
    for col in columns:
        out[col] = out[col].map(canonical_team)
    return out


def unmapped(names: pd.Series | list[str]) -> list[str]:
    """Names with no alias entry, i.e. those that pass through unchanged.

    Most are already canonical and need no entry. This exists so a test can
    assert the known-divergent spellings are covered, and so a human can eyeball
    what a new season introduced.
    """
    index = _alias_index()
    unique = pd.Series(list(names), dtype="object").dropna().unique()
    return sorted({n for n in unique if normalise(n) not in index})
