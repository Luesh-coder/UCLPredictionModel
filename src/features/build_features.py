"""Match-level feature engineering.

Every feature here is computed from information available *strictly before*
kickoff. The rolling helpers all shift by one match per team, and the Elo
column carries the pre-match rating. That discipline is what makes the
walk-forward backtest in `src.evaluation.backtest` meaningful.
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
    "ucl_exp_diff",
    "is_knockout",
    "is_neutral",
]


def to_long(matches: pd.DataFrame) -> pd.DataFrame:
    """Explode one row per match into two rows per match (one per team).

    Rolling per-team statistics are far easier to express in this shape.
    """
    home = pd.DataFrame(
        {
            "match_id": matches["match_id"],
            "date": matches["date"],
            "season": matches["season"],
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
    """Rolling per-team means over the previous `window` UCL matches.

    `shift(1)` before rolling is the whole point: without it a team's average
    would include the match we are trying to predict.
    """
    g = long.groupby("team", sort=False)
    for col, name in (("gf", "form_gf"), ("ga", "form_ga"), ("points", "form_pts")):
        long[name] = (
            g[col]
            .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
            .astype(float)
        )

    # Days since the team's last European match — a crude fatigue/rust proxy.
    long["rest_days"] = g["date"].transform(lambda s: s.diff().dt.days).astype(float)

    # How many UCL matches this team has behind it. Debutants behave differently
    # from clubs that play the competition every year.
    long["ucl_exp"] = g.cumcount().astype(float)
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
) -> pd.DataFrame:
    """Return the match table plus every modelling feature and the target.

    The result keeps all original columns, so it doubles as the table you eyeball
    when a prediction looks wrong.
    """
    df = add_elo_features(matches, elo_config)

    long = add_rolling_form(to_long(df), window=form_window)
    per_team = ["form_gf", "form_ga", "form_pts", "rest_days", "ucl_exp"]
    wide = _wide_from_long(long, per_team)
    df = df.merge(wide, on="match_id", how="left")

    # Models see differences rather than raw home/away pairs: a difference is
    # invariant to competition-wide drift in scoring, and halves the width.
    df["form_gf_diff"] = df["form_gf_home"] - df["form_gf_away"]
    df["form_ga_diff"] = df["form_ga_home"] - df["form_ga_away"]
    df["form_pts_diff"] = df["form_pts_home"] - df["form_pts_away"]
    df["rest_diff"] = df["rest_days_home"] - df["rest_days_away"]
    df["ucl_exp_diff"] = df["ucl_exp_home"] - df["ucl_exp_away"]

    stage = df["stage"].fillna("").astype(str)
    group_like = stage.str.contains("group|league phase", regex=True)
    df["is_knockout"] = (~group_like).astype(int)
    df["is_neutral"] = stage.str.contains("final").astype(int)

    # A team's first-ever UCL match has no rest history; 14 days is roughly a
    # normal midweek-to-midweek gap and keeps the column free of NaN.
    df["rest_diff"] = df["rest_diff"].fillna(0.0)
    for col in ("form_gf_diff", "form_ga_diff", "form_pts_diff", "ucl_exp_diff"):
        df[col] = df[col].fillna(0.0)

    return df


def xy(df: pd.DataFrame, features: list[str] | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Split a feature frame into the model matrix X and the 1X2 target y."""
    features = features or FEATURE_COLUMNS
    played = df[df["result"].notna()]
    return played[features].astype(float), played["result"].astype(str)
