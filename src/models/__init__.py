"""Models: the baselines worth beating, and the candidates that must beat them."""

from src.models.classifiers import (
    MODEL_REGISTRY,
    EloBaseline,
    MarketBaseline,
    PriorBaseline,
    make_lightgbm,
    make_logistic,
    predict_proba_frame,
)
from src.models.dixon_coles import DixonColesModel

__all__ = [
    "MODEL_REGISTRY",
    "DixonColesModel",
    "EloBaseline",
    "MarketBaseline",
    "PriorBaseline",
    "make_lightgbm",
    "make_logistic",
    "predict_proba_frame",
]
