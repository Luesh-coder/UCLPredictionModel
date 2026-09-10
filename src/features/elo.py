"""Elo ratings computed strictly forward in time.

The rating attached to a match is always the rating *before* that match is
played, so the column can be fed to a model without leaking the result it is
supposed to predict.

Ratings are maintained **per league**, not in one global pool. Stage 1 data is
domestic only, so no Premier League side ever plays a Bundesliga side and the
two leagues form disconnected components of the match graph. Rating them
together would produce numbers that look comparable across leagues and are not:
their relative level would be an artefact of nothing more than each pool's
arbitrary starting point. Stage 2 adds Champions League results, which are the
edges that actually connect the components, and that is when a single global
pool starts to mean something.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class EloConfig:
    k: float = 20.0
    """Base update size. Larger = ratings react faster to recent results."""

    home_advantage: float = 60.0
    """Rating points added to the home side when forming the expectation."""

    goal_diff_scaling: bool = True
    """Scale the update by margin of victory (FIFA/World-Football-Elo style)."""

    initial: float = 1500.0

    regress_to_mean: float = 0.20
    """Fraction pulled back toward `initial` between seasons. Squad turnover and
    promotion/relegation both argue against carrying a rating across a summer
    untouched."""

    neutral_stages: tuple[str, ...] = ("final",)
    """Stages played at a neutral venue, where no home advantage applies. Never
    fires on domestic data, where every stage is "league"; it is here for the
    Champions League rows Stage 2 introduces."""


@dataclass
class EloModel:
    config: EloConfig = field(default_factory=EloConfig)
    ratings: dict[str, float] = field(default_factory=dict)

    def get(self, team: str) -> float:
        return self.ratings.get(team, self.config.initial)

    def expected(self, home: str, away: str, *, neutral: bool = False) -> float:
        """P(home wins) under the logistic Elo curve, draws split evenly."""
        adv = 0.0 if neutral else self.config.home_advantage
        diff = self.get(home) + adv - self.get(away)
        return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))

    def update(
        self,
        home: str,
        away: str,
        home_goals: int,
        away_goals: int,
        *,
        neutral: bool = False,
    ) -> None:
        exp_home = self.expected(home, away, neutral=neutral)
        if home_goals > away_goals:
            score = 1.0
        elif home_goals == away_goals:
            score = 0.5
        else:
            score = 0.0

        k = self.config.k
        if self.config.goal_diff_scaling:
            margin = abs(home_goals - away_goals)
            # log1p keeps the multiplier from exploding on a 7-0; a 1-goal win
            # gets 1.0, a 4-goal win about 1.6.
            k *= 1.0 + np.log1p(max(margin - 1, 0))

        delta = k * (score - exp_home)
        self.ratings[home] = self.get(home) + delta
        self.ratings[away] = self.get(away) - delta

    def regress(self) -> None:
        """Pull every rating partway back to the mean (call between seasons)."""
        r = self.config.regress_to_mean
        base = self.config.initial
        self.ratings = {t: base + (v - base) * (1 - r) for t, v in self.ratings.items()}


def _rate_one_pool(df: pd.DataFrame, config: EloConfig) -> pd.DataFrame:
    """Run the forward pass over one connected pool of teams (one league)."""
    model = EloModel(config)
    df = df.sort_values("date", kind="stable")

    elo_home = np.empty(len(df))
    elo_away = np.empty(len(df))
    prob_home = np.empty(len(df))

    prev_season = None
    for i, row in enumerate(df.itertuples(index=False)):
        if prev_season is not None and row.season != prev_season:
            model.regress()
        prev_season = row.season

        stage = str(getattr(row, "stage", "") or "")
        neutral = any(s in stage for s in config.neutral_stages)

        # Read before update: this is the pre-match rating, the only one a model
        # is allowed to see.
        elo_home[i] = model.get(row.home)
        elo_away[i] = model.get(row.away)
        prob_home[i] = model.expected(row.home, row.away, neutral=neutral)

        hg, ag = row.home_goals, row.away_goals
        if pd.notna(hg) and pd.notna(ag):
            model.update(row.home, row.away, int(hg), int(ag), neutral=neutral)

    return pd.DataFrame(
        {
            "elo_home": elo_home,
            "elo_away": elo_away,
            "elo_diff": elo_home - elo_away,
            "elo_prob_home": prob_home,
        },
        index=df.index,
    )


def add_elo_features(
    matches: pd.DataFrame,
    config: EloConfig | None = None,
    *,
    pool: str = "league",
) -> pd.DataFrame:
    """Attach pre-match Elo columns to the match table.

    Adds `elo_home`, `elo_away`, `elo_diff` and `elo_prob_home`. Unplayed
    fixtures receive pre-match ratings but do not update anything.

    `pool` names the column whose groups are rated independently. Pass
    `pool=None` to rate every match in one pool, which is correct only once the
    competitions in the frame actually share opponents.
    """
    config = config or EloConfig()
    df = matches.sort_values("date", kind="stable").reset_index(drop=True)

    if pool is None:
        rated = _rate_one_pool(df, config)
    else:
        rated = pd.concat(
            [_rate_one_pool(block, config) for _, block in df.groupby(pool, sort=False)]
        )

    return pd.concat([df, rated.reindex(df.index)], axis=1)
