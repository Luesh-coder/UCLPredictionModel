"""Shared fixtures.

Tests must never hit the network. The live sources are slow, occasionally down
(ClubElo is returning 502s as of this writing) and always changing, so a suite
that depended on them would fail for reasons that have nothing to do with this
code. Everything here runs against a synthetic match table with a known
structure, so a failure points at our code rather than at a website.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_matches() -> pd.DataFrame:
    """Five seasons of two 12-team leagues with a deliberate strength gradient.

    Team 00 is the strongest and team 11 the weakest within each league, so any
    working model should rank them in roughly that order — a cheap end-to-end
    sanity check.

    Two leagues, with **no fixtures between them**, mirror the real Stage 1
    structure: the domestic match graph is disconnected, and code that assumes
    otherwise (a single global Elo pool, say) should be caught here rather than
    in production.

    Each season is a double round robin spaced so the whole campaign lands
    inside one September-to-June window. Seasons must not overlap in calendar
    time, or a chronological backtest could legitimately train on dates later
    than the fold it predicts, and the leakage tests would be testing the
    fixture rather than the code.
    """
    rng = np.random.default_rng(0)
    rows = []

    for league_no, league in enumerate(["ENG-Premier League", "GER-Bundesliga"]):
        teams = [f"{league[:3]} Team {i:02d}" for i in range(12)]
        attack = {t: 0.5 - 0.09 * i for i, t in enumerate(teams)}

        for season in range(2018, 2023):
            start = pd.Timestamp(f"{season}-09-15")
            match_no = 0
            for home in teams:
                for away in teams:
                    if home == away:
                        continue
                    lam = np.exp(attack[home] - attack[away] * 0.5 + 0.25)
                    mu = np.exp(attack[away] - attack[home] * 0.5)
                    home_goals = int(rng.poisson(lam))
                    away_goals = int(rng.poisson(mu))

                    # A believable bookmaker: the true Poisson win probability
                    # nudged off-target and loaded with a 5% margin, so the
                    # market baseline is strong but not clairvoyant.
                    p_home = 1.0 / (1.0 + np.exp(-(lam - mu)))
                    p_draw = 0.26
                    probs = np.array([p_home * (1 - p_draw), p_draw, (1 - p_home) * (1 - p_draw)])
                    probs = probs * (1.05 / probs.sum())

                    rows.append(
                        {
                            "season": season,
                            "league": league,
                            "division": f"L{league_no}",
                            "date": start + pd.to_timedelta(2 * match_no, unit="D"),
                            "stage": "league",
                            "home": home,
                            "away": away,
                            "home_goals": home_goals,
                            "away_goals": away_goals,
                            "B365H": round(1 / probs[0], 2),
                            "B365D": round(1 / probs[1], 2),
                            "B365A": round(1 / probs[2], 2),
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
    df["match_id"] = [f"m{i:05d}" for i in range(len(df))]
    return df


@pytest.fixture
def synthetic_features(synthetic_matches) -> pd.DataFrame:
    """`synthetic_matches` with market probabilities and modelling features."""
    from src.features.build_features import build_features
    from src.ingest.odds import add_market_probabilities

    return build_features(add_market_probabilities(synthetic_matches))


@pytest.fixture
def football_data_csv() -> str:
    """A football-data.co.uk CSV exercising the three real parsing hazards.

    Namely: a UTF-8 BOM on the header, both `dd/mm/yy` and `dd/mm/yyyy` dates in
    one file, and a blank trailing line. Written as a literal so the suite never
    needs the network.
    """
    return (
        "﻿Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,B365H,B365D,B365A,PSH,PSD,PSA\n"
        "E0,17/08/2019,Arsenal,Man United,2,1,H,2.10,3.40,3.60,2.12,3.45,3.70\n"
        "E0,18/08/19,Liverpool,Chelsea,3,0,H,1.80,3.80,4.50,1.82,3.90,4.60\n"
        "E0,19/08/2019,Man City,Everton,1,1,D,1.25,6.00,12.0,1.26,6.10,12.5\n"
        "\n"
    )
