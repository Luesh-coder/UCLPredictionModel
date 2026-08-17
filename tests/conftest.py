"""Shared fixtures.

Tests must never hit the network: FBref scraping is slow, rate-limited and
non-deterministic. Everything here runs against a synthetic match table with a
known structure, so a failure points at our code rather than at a website.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_matches() -> pd.DataFrame:
    """Five seasons of a 12-team competition with a deliberate strength gradient.

    Team 0 is the strongest and team 11 the weakest, so any working model should
    rank them in roughly that order — a cheap end-to-end sanity check.

    Each season is a double round robin (132 matches) spaced so the whole
    campaign lands inside one September-to-June window. Seasons must not
    overlap in calendar time, or a chronological backtest could legitimately
    train on dates later than the fold it predicts and the leakage tests would
    be testing the fixture rather than the code.
    """
    rng = np.random.default_rng(0)
    teams = [f"Team {i:02d}" for i in range(12)]
    # Attack strength decreasing with index; defence mirrors it.
    attack = {t: 0.5 - 0.09 * i for i, t in enumerate(teams)}

    rows = []
    for season in range(2018, 2023):
        start = pd.Timestamp(f"{season}-09-15")
        match_no = 0
        for home in teams:
            for away in teams:
                if home == away:
                    continue
                lam = np.exp(attack[home] - attack[away] * 0.5 + 0.25)
                mu = np.exp(attack[away] - attack[home] * 0.5)
                rows.append(
                    {
                        "season": season,
                        "date": start + pd.to_timedelta(2 * match_no, unit="D"),
                        "stage": "group stage" if match_no < 80 else "round of 16",
                        "home": home,
                        "away": away,
                        "home_goals": int(rng.poisson(lam)),
                        "away_goals": int(rng.poisson(mu)),
                    }
                )
                match_no += 1

    df = pd.DataFrame(rows).sort_values("date", kind="stable").reset_index(drop=True)
    df["home_goals"] = df["home_goals"].astype("Int64")
    df["away_goals"] = df["away_goals"].astype("Int64")
    df["result"] = np.select(
        [df["home_goals"] > df["away_goals"], df["home_goals"] == df["away_goals"]],
        ["H", "D"],
        default="A",
    )
    df["match_id"] = np.arange(len(df))
    return df
