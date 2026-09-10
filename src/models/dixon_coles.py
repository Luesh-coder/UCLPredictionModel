"""Dixon-Coles bivariate Poisson model for match scorelines.

Each team gets an attack and a defence parameter; home and away goals are
Poisson with

    lambda = exp(attack_home + defence_away + home_adv)
    mu     = exp(attack_away + defence_home)

plus the Dixon & Coles (1997) low-score correction `tau`, which repairs the
independence assumption where it fails worst — 0-0, 1-0, 0-1 and 1-1 — and an
exponential time-decay weight so old matches count for less.

Reference: Dixon, M.J. & Coles, S.G. (1997), "Modelling Association Football
Scores and Inefficiencies in the Football Betting Market", JRSS-C 46(2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from src.config import OUTCOMES

MAX_GOALS = 10


def tau(
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    lam: np.ndarray,
    mu: np.ndarray,
    rho: float,
) -> np.ndarray:
    """Dixon-Coles dependence correction, applied only to scores below 2-2."""
    out = np.ones_like(lam, dtype=float)
    h, a = home_goals, away_goals
    out = np.where((h == 0) & (a == 0), 1.0 - lam * mu * rho, out)
    out = np.where((h == 0) & (a == 1), 1.0 + lam * rho, out)
    out = np.where((h == 1) & (a == 0), 1.0 + mu * rho, out)
    out = np.where((h == 1) & (a == 1), 1.0 - rho, out)
    return out


def time_decay_weights(dates: pd.Series, reference: pd.Timestamp, xi: float) -> np.ndarray:
    """exp(-xi * age_in_days). xi=0 weights all history equally.

    xi = 0.0018 halves a match's weight after roughly a year, which is close to
    the value Dixon & Coles found optimal for domestic league data.
    """
    age_days = (reference - pd.to_datetime(dates)).dt.total_seconds().to_numpy() / 86400.0
    age_days = np.clip(age_days, 0.0, None)
    return np.exp(-xi * age_days)


@dataclass
class DixonColesModel:
    """Fit on a match table, then predict 1X2 probabilities or scorelines."""

    xi: float = 0.0018
    max_goals: int = MAX_GOALS

    teams_: list[str] | None = None
    params_: np.ndarray | None = None
    reference_date_: pd.Timestamp | None = None

    # --- internals ---------------------------------------------------------

    def _unpack(self, params: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
        n = len(self.teams_)
        attack = params[:n]
        defence = params[n : 2 * n]
        home_adv, rho = params[2 * n], params[2 * n + 1]
        # Attack is only identified up to an additive constant (it trades off
        # against defence and the intercept), so pin its mean at zero.
        attack = attack - attack.mean()
        return attack, defence, home_adv, rho

    def _neg_log_likelihood(
        self,
        params: np.ndarray,
        hi: np.ndarray,
        ai: np.ndarray,
        hg: np.ndarray,
        ag: np.ndarray,
        w: np.ndarray,
    ) -> float:
        attack, defence, home_adv, rho = self._unpack(params)
        lam = np.exp(attack[hi] + defence[ai] + home_adv)
        mu = np.exp(attack[ai] + defence[hi])

        adj = tau(hg, ag, lam, mu, rho)
        # tau can go non-positive for extreme rho; clip rather than let the
        # optimiser wander into NaN territory.
        adj = np.clip(adj, 1e-10, None)

        ll = w * (np.log(adj) + poisson.logpmf(hg, lam) + poisson.logpmf(ag, mu))
        return -float(ll.sum())

    # --- API ---------------------------------------------------------------

    def fit(self, matches: pd.DataFrame, reference_date: pd.Timestamp | None = None):
        """Fit on played matches only. `reference_date` anchors the time decay."""
        df = matches.dropna(subset=["home_goals", "away_goals"]).copy()
        if df.empty:
            raise ValueError("no played matches to fit on")

        self.teams_ = sorted(set(df["home"]) | set(df["away"]))
        index = {t: i for i, t in enumerate(self.teams_)}
        n = len(self.teams_)

        hi = df["home"].map(index).to_numpy()
        ai = df["away"].map(index).to_numpy()
        hg = df["home_goals"].astype(int).to_numpy()
        ag = df["away_goals"].astype(int).to_numpy()

        self.reference_date_ = pd.Timestamp(reference_date or df["date"].max())
        w = time_decay_weights(df["date"], self.reference_date_, self.xi)

        x0 = np.concatenate([np.zeros(n), np.zeros(n), [0.25], [-0.05]])
        bounds = [(-3, 3)] * n + [(-3, 3)] * n + [(-1, 1), (-0.4, 0.4)]

        result = minimize(
            self._neg_log_likelihood,
            x0,
            args=(hi, ai, hg, ag, w),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500},
        )
        self.params_ = result.x
        self.result_ = result
        return self

    def _rates(self, home: str, away: str) -> tuple[float, float]:
        attack, defence, home_adv, _ = self._unpack(self.params_)
        index = {t: i for i, t in enumerate(self.teams_)}
        # An unseen team gets the league-average attack and defence rather than
        # an error — new entrants show up in the UCL every season.
        h, a = index.get(home), index.get(away)
        att_h = attack[h] if h is not None else 0.0
        def_h = defence[h] if h is not None else defence.mean()
        att_a = attack[a] if a is not None else 0.0
        def_a = defence[a] if a is not None else defence.mean()
        return float(np.exp(att_h + def_a + home_adv)), float(np.exp(att_a + def_h))

    def score_matrix(self, home: str, away: str) -> np.ndarray:
        """P(home scores i, away scores j) for i, j in 0..max_goals."""
        lam, mu = self._rates(home, away)
        _, _, _, rho = self._unpack(self.params_)

        goals = np.arange(self.max_goals + 1)
        matrix = np.outer(poisson.pmf(goals, lam), poisson.pmf(goals, mu))

        matrix[0, 0] *= 1.0 - lam * mu * rho
        matrix[0, 1] *= 1.0 + lam * rho
        matrix[1, 0] *= 1.0 + mu * rho
        matrix[1, 1] *= 1.0 - rho

        matrix = np.clip(matrix, 0.0, None)
        return matrix / matrix.sum()

    def predict_proba(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        """1X2 probabilities for a frame with `home` and `away` columns."""
        rows = []
        for row in fixtures.itertuples(index=False):
            m = self.score_matrix(row.home, row.away)
            rows.append(
                {
                    "H": float(np.tril(m, -1).sum()),
                    "D": float(np.trace(m)),
                    "A": float(np.triu(m, 1).sum()),
                }
            )
        return pd.DataFrame(rows, columns=OUTCOMES, index=fixtures.index)

    def team_ratings(self) -> pd.DataFrame:
        """Fitted attack/defence strengths, strongest attack first."""
        attack, defence, _, _ = self._unpack(self.params_)
        return (
            pd.DataFrame({"team": self.teams_, "attack": attack, "defence": defence})
            .sort_values("attack", ascending=False)
            .reset_index(drop=True)
        )
