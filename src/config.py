"""Central paths and project-wide constants.

Everything else imports from here so that no module hard-codes a path.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = Path(os.getenv("SOCCERDATA_DIR", DATA_DIR / "cache"))

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

for _d in (RAW_DIR, PROCESSED_DIR, CACHE_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# soccerdata reads this env var at import time to decide where to cache scrapes.
os.environ.setdefault("SOCCERDATA_DIR", str(CACHE_DIR))

# MLflow 3 put the filesystem backend into maintenance mode and now raises
# rather than writing to a bare "./mlruns" directory, so default to the
# SQLite backend it recommends. Still a local file, still no server to run.
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", f"sqlite:///{MLFLOW_DB.as_posix()}"
)
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "ucl-match-outcome")

# --- Data sources -----------------------------------------------------------

# football-data.co.uk publishes one static CSV per (season, division). `season`
# is the four-digit pair of two-digit years, e.g. 2425 for 2024/25.
FOOTBALL_DATA_URL = "https://www.football-data.co.uk/mmz4281/{season}/{div}.csv"

# football-data's division codes, mapped to the league names used in the
# canonical match table.
#
# The big five are the bulk of the data. The six below them are here for Stage
# 2: they are where the rest of the Champions League field plays its domestic
# football, and a club with no domestic history would enter the global Elo pool
# at the default 1500 and stay near it — its rating moving on ~8 European
# matches a season — which would then contaminate every side it faced.
DIVISION_NAMES = {
    "E0": "ENG-Premier League",
    "SP1": "ESP-La Liga",
    "I1": "ITA-Serie A",
    "D1": "GER-Bundesliga",
    "F1": "FRA-Ligue 1",
    "N1": "NED-Eredivisie",
    "P1": "POR-Primeira Liga",
    "B1": "BEL-Pro League",
    "T1": "TUR-Super Lig",
    "SC0": "SCO-Premiership",
    "G1": "GRE-Super League",
}
DEFAULT_DIVISIONS = list(DIVISION_NAMES)

# Seasons as the calendar year the season started: 2024 means 2024/25.
# The floor is 2012 because Pinnacle (PS*) odds, the sharpest column set and the
# one the market baseline prefers, only start appearing around then.
DEFAULT_SEASONS = list(range(2012, 2026))

# soccerdata's league id for the Champions League (Stage 2).
UCL_LEAGUE = "UEFA-Champions League"

# --- Domain constants -------------------------------------------------------

# Ordered class labels for the 1X2 target. Order matters: RPS is computed on the
# cumulative distribution, so it must run home -> draw -> away.
OUTCOMES = ["H", "D", "A"]

RANDOM_SEED = 42
