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
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = Path(os.getenv("SOCCERDATA_DIR", DATA_DIR / "cache"))

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

for _d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, CACHE_DIR, FIGURES_DIR, MLRUNS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# soccerdata reads this env var at import time to decide where to cache scrapes.
os.environ.setdefault("SOCCERDATA_DIR", str(CACHE_DIR))

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", MLRUNS_DIR.as_uri())
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "ucl-match-outcome")

# --- Domain constants -------------------------------------------------------

# soccerdata's league id for the Champions League.
UCL_LEAGUE = "UEFA-Champions League"

# Seasons expressed the way soccerdata wants them: the calendar year the season
# started. 2024 means the 2024/25 campaign (the first "league phase" season).
DEFAULT_SEASONS = list(range(2016, 2025))

# Ordered class labels for the 1X2 target. Order matters: RPS is computed on the
# cumulative distribution, so it must run home -> draw -> away.
OUTCOMES = ["H", "D", "A"]

RANDOM_SEED = 42
