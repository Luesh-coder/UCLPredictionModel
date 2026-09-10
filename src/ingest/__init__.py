"""Data acquisition: raw sources in, one canonical match table out."""

from src.ingest.build_dataset import build, load, save
from src.ingest.football_data import fetch
from src.ingest.odds import add_market_probabilities
from src.ingest.teams import canonical_team, normalise, unmapped

__all__ = [
    "add_market_probabilities",
    "build",
    "canonical_team",
    "fetch",
    "load",
    "normalise",
    "save",
    "unmapped",
]
