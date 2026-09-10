"""Turn bookmaker 1X2 odds into a probability forecast.

This is what football-data.co.uk buys us over a results-only source. Bookmaker
odds are the single strongest publicly available forecast of a football match,
so the de-vigged market probabilities become the benchmark every model in this
project is measured against — not a feature, a *target to approach*.

Two steps:

1. Pick a bookmaker. Column availability drifts across 14 seasons as firms come
   and go, so we try several in order of sharpness and record which one answered.
2. Remove the overround. Raw odds encode the bookmaker's margin, so 1/odds sums
   to roughly 1.02-1.07 rather than 1.0 and is not a probability distribution.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# In preference order. Pinnacle first: it is a low-margin, high-limit book whose
# prices are the closest thing to a market consensus. "Avg"/"Max" are
# football-data's own aggregates across all books it collected that season, and
# are the best fallback for early seasons where Pinnacle is absent. Bet365 is
# last but present in essentially every file.
ODDS_PREFERENCES: list[tuple[str, tuple[str, str, str]]] = [
    ("pinnacle", ("PSH", "PSD", "PSA")),
    ("pinnacle_closing", ("PSCH", "PSCD", "PSCA")),
    ("market_average", ("AvgH", "AvgD", "AvgA")),
    ("market_average_closing", ("AvgCH", "AvgCD", "AvgCA")),
    ("bet365", ("B365H", "B365D", "B365A")),
    ("bet365_closing", ("B365CH", "B365CD", "B365CA")),
    ("william_hill", ("WHH", "WHD", "WHA")),
]

OUTPUT_COLUMNS = ["odds_H", "odds_D", "odds_A", "odds_source", "overround"]


def _valid(df: pd.DataFrame, cols: tuple[str, str, str]) -> pd.Series:
    """Rows where all three odds exist and are genuine payouts (> 1.0).

    An odds of exactly 1.0 pays nothing and is a placeholder, not a price;
    letting one through produces an implied probability of 1.0 for that outcome
    and drives the de-vig to nonsense.
    """
    if not all(c in df.columns for c in cols):
        return pd.Series(False, index=df.index)
    block = df[list(cols)].apply(pd.to_numeric, errors="coerce")
    return block.notna().all(axis=1) & (block > 1.0).all(axis=1)


def add_market_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    """Attach de-vigged market probabilities and their provenance.

    Each row is served by the most preferred bookmaker that actually priced it,
    so a season missing Pinnacle silently falls back to the market average
    rather than dropping out of the benchmark entirely. `odds_source` records
    the choice, because a benchmark you cannot attribute is not a benchmark.

    Rows no bookmaker priced keep NaN probabilities. They are still valid
    matches for training — only the market baseline has nothing to say about
    them, and the evaluation layer excludes them from that comparison alone.
    """
    out = df.copy()
    n = len(out)

    implied = np.full((n, 3), np.nan)
    source = pd.Series(pd.NA, index=out.index, dtype="object")

    for name, cols in ODDS_PREFERENCES:
        # Only fill rows still unclaimed by a more preferred bookmaker.
        usable = _valid(out, cols) & source.isna()
        if not usable.any():
            continue
        block = out.loc[usable, list(cols)].apply(pd.to_numeric, errors="coerce")
        implied[usable.to_numpy(), :] = 1.0 / block.to_numpy(dtype=float)
        source[usable] = name
        log.debug("%s priced %d rows", name, int(usable.sum()))

    # The overround is the bookmaker's margin: how much more than 100% the raw
    # implied probabilities sum to. Kept as a column because a sudden jump is
    # the clearest sign a season's odds columns are being misread.
    overround = implied.sum(axis=1)
    normalised = implied / overround[:, None]

    market = pd.DataFrame(
        {
            "odds_H": normalised[:, 0],
            "odds_D": normalised[:, 1],
            "odds_A": normalised[:, 2],
            "odds_source": source.to_numpy(),
            "overround": overround,
        },
        index=out.index,
    )
    priced = market["odds_source"].notna().sum()
    log.info("Market probabilities on %d/%d matches (%.1f%%)", priced, n, 100 * priced / max(n, 1))
    return pd.concat([out, market], axis=1)
