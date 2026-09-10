"""Match-level feature engineering.

Every feature here is computed from information available *strictly before*
kickoff. The rolling helpers all shift by one match per team, and the Elo column
carries the pre-match rating. That discipline is what makes the walk-forward
backtest in `src.evaluate.backtest` meaningful — and it is easy to lose, because
a leaked feature makes the metrics look better, not worse.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.elo import EloConfig, add_elo_features

FEATURE_COLUMNS = [
    "elo_diff",
    "elo_prob_home",
    "form_gf_diff",
    "form_ga_diff",
    "form_pts_diff",
    "rest_diff",
    "experience_diff",
]

# For experiments only. The bookmaker's own probabilities are the benchmark this
# project is measured against, so feeding them in as inputs makes "beat the
# market" circular: the model inherits the market's skill and reports it as its
# own. Use this set to ask a different question — what a model adds *on top of*
# the market — never to fill the leaderboard.
FEATURE_COLUMNS_WITH_ODDS = [*FEATURE_COLUMNS, "odds_H", "odds_D", "odds_A"]

# A team's first match in the dataset has no rest history. Zero is the neutral
# value for every *_diff column: it says "no difference between these sides",
# which is the honest prior when we know nothing about either.
_NEUTRAL_FILL = 0.0


def to_long(matches: pd.DataFrame) -> pd.DataFrame:
    """Explode one row per match into two rows per match, one per team.

    Rolling per-team statistics are far easier to express in this shape, and
    much harder to get wrong: a team's history is a single contiguous group
    rather than a mix of home and away columns.
    """
    home = pd.DataFrame(
        {
            "match_id": matches["match_id"],
            "date": matches["date"],
            "season": matches["season"],
            "league": matches["league"],
            "team": matches["home"],
            "opponent": matches["away"],
            "is_home": 1,
            "gf": matches["home_goals"],
            "ga": matches["away_goals"],
        }
    )
    away = pd.DataFrame(
        {
            "match_id": matches["match_id"],
            "date": matches["date"],
            "season": matches["season"],
            "league": matches["league"],
            "team": matches["away"],
            "opponent": matches["home"],
            "is_home": 0,
            "gf": matches["away_goals"],
            "ga": matches["home_goals"],
        }
    )
    long = pd.concat([home, away], ignore_index=True)
    long["points"] = np.select(
        [long["gf"] > long["ga"], long["gf"] == long["ga"]], [3.0, 1.0], default=0.0
    )
    long.loc[long["gf"].isna() | long["ga"].isna(), "points"] = np.nan
    return long.sort_values(["team", "date"], kind="stable").reset_index(drop=True)


def add_rolling_form(long: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Rolling per-team means over the team's previous `window` matches.

    `shift(1)` before `rolling` is the whole point of this function. Without it
    a team's average includes the very match we are trying to predict, the
    model learns to read the answer off its own input, and every metric in the
    project becomes a lie that looks like a triumph.
    """
    g = long.groupby("team", sort=False)
    for col, name in (("gf", "form_gf"), ("ga", "form_ga"), ("points", "form_pts")):
        long[name] = (
            g[col]
            .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
            .astype(float)
        )

    # Days since this team's previous match — a crude fatigue/congestion proxy.
    long["rest_days"] = g["date"].transform(lambda s: s.diff().dt.days).astype(float)

    # Matches this team has behind it in the dataset. `cumcount` is already
    # exclusive of the current row, so it needs no shift.
    long["experience"] = g.cumcount().astype(float)
    return long


def _wide_from_long(long: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Pivot per-team columns back to one row per match, home/away suffixed."""
    home = long[long["is_home"] == 1].set_index("match_id")[columns]
    away = long[long["is_home"] == 0].set_index("match_id")[columns]
    wide = home.add_suffix("_home").join(away.add_suffix("_away"), how="outer")
    return wide.reset_index()


def build_features(
    matches: pd.DataFrame,
    *,
    form_window: int = 5,
    elo_config: EloConfig | None = None,
    elo_pool: str = "league",
) -> pd.DataFrame:
    """Return the match table plus every modelling feature and the target.

    The result keeps all original columns, so it doubles as the table you eyeball
    when a prediction looks wrong.
    """
    df = add_elo_features(matches, elo_config, pool=elo_pool)

    long = add_rolling_form(to_long(df), window=form_window)
    per_team = ["form_gf", "form_ga", "form_pts", "rest_days", "experience"]
    wide = _wide_from_long(long, per_team)
    df = df.merge(wide, on="match_id", how="left")

    # Models see differences rather than raw home/away pairs: a difference is
    # invariant to competition-wide drift in scoring rates, and halves the width
    # of the matrix without discarding anything the pair encoded.
    diffs = pd.DataFrame(
        {
            "form_gf_diff": df["form_gf_home"] - df["form_gf_away"],
            "form_ga_diff": df["form_ga_home"] - df["form_ga_away"],
            "form_pts_diff": df["form_pts_home"] - df["form_pts_away"],
            "rest_diff": df["rest_days_home"] - df["rest_days_away"],
            "experience_diff": df["experience_home"] - df["experience_away"],
        },
        index=df.index,
    )
    df = pd.concat([df, diffs.fillna(_NEUTRAL_FILL)], axis=1)

    return df


def xy(df: pd.DataFrame, features: list[str] | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Split a feature frame into the model matrix X and the 1X2 target y."""
    features = features or FEATURE_COLUMNS
    played = df[df["result"].notna()]
    return played[features].astype(float), played["result"].astype(str)
