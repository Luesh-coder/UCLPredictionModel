"""Discriminative 1X2 classifiers and the baselines worth beating.

Every model here follows the sklearn API (`fit` / `predict_proba`) and emits
columns in `src.config.OUTCOMES` order, so `src.evaluation` can score any of
them without special-casing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import OUTCOMES, RANDOM_SEED
from src.features.build_features import FEATURE_COLUMNS


class PriorBaseline(BaseEstimator, ClassifierMixin):
    """Predicts the training set's base rates for every match.

    This is the floor. A model that cannot beat the historical H/D/A split has
    learned nothing, and it is remarkably easy to build one that doesn't.
    """

    def fit(self, X, y):
        y = pd.Series(y)
        counts = y.value_counts(normalize=True)
        self.prior_ = np.array([counts.get(c, 0.0) for c in OUTCOMES])
        self.classes_ = np.array(OUTCOMES)
        return self

    def predict_proba(self, X):
        return np.tile(self.prior_, (len(X), 1))


class EloBaseline(BaseEstimator, ClassifierMixin):
    """Turns the pre-match Elo win expectation into a calibrated 1X2 vector.

    Elo emits P(home) with draws split evenly, which is not a three-way
    distribution. A multinomial logit on the single Elo feature learns the draw
    rate and the home/away split from data, which is both simpler and better
    calibrated than the usual hand-tuned draw constant.
    """

    def __init__(self, feature: str = "elo_diff"):
        self.feature = feature

    def fit(self, X, y):
        X = pd.DataFrame(X)
        self.model_ = LogisticRegression(max_iter=1000, C=1.0)
        self.model_.fit(X[[self.feature]], y)
        # `predict_proba` below already returns OUTCOMES order, so advertise
        # that — not sklearn's alphabetical order, which would make callers
        # reorder a second time and silently swap home for away.
        self.classes_ = np.array(OUTCOMES)
        return self

    def predict_proba(self, X):
        X = pd.DataFrame(X)
        proba = self.model_.predict_proba(X[[self.feature]])
        return _reorder(proba, self.model_.classes_)


def _reorder(proba: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Reindex an sklearn probability matrix into OUTCOMES order."""
    index = {c: i for i, c in enumerate(classes)}
    return proba[:, [index[c] for c in OUTCOMES]]


def make_logistic(features: list[str] | None = None) -> Pipeline:
    """Multinomial logistic regression: scaled, imputed, mildly regularised."""
    features = features or FEATURE_COLUMNS
    return Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        (
                            "num",
                            Pipeline(
                                [
                                    ("impute", SimpleImputer(strategy="median")),
                                    ("scale", StandardScaler()),
                                ]
                            ),
                            features,
                        )
                    ],
                    remainder="drop",
                ),
            ),
            ("clf", LogisticRegression(max_iter=2000, C=0.5, random_state=RANDOM_SEED)),
        ]
    )


def make_lightgbm(features: list[str] | None = None, *, calibrate: bool = True, **params):
    """Gradient-boosted trees for the 1X2 target.

    Defaults are deliberately small. Roughly 1,500 UCL matches is a *tiny*
    dataset by GBDT standards, and an unconstrained LightGBM will memorise it
    and produce badly over-confident probabilities — which log loss punishes
    hard. `calibrate=True` wraps the model in isotonic calibration fitted on
    held-out folds, which usually buys more than any hyperparameter tweak.
    """
    from lightgbm import LGBMClassifier

    features = features or FEATURE_COLUMNS
    defaults = dict(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=7,
        max_depth=3,
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=RANDOM_SEED,
        verbose=-1,
    )
    defaults.update(params)

    clf = LGBMClassifier(**defaults)
    if calibrate:
        # `cv` splits chronologically-ordered data, so keep folds modest to
        # leave each one enough matches to calibrate on.
        clf = CalibratedClassifierCV(clf, method="isotonic", cv=3)

    return Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [("num", SimpleImputer(strategy="median"), features)],
                    remainder="drop",
                ),
            ),
            ("clf", clf),
        ]
    )


def predict_proba_frame(model, X: pd.DataFrame) -> pd.DataFrame:
    """Call `predict_proba` and return a frame with H/D/A columns in order."""
    proba = model.predict_proba(X)
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "steps"):
        classes = model.steps[-1][1].classes_
    if classes is not None and list(classes) != OUTCOMES:
        proba = _reorder(proba, np.asarray(classes))
    return pd.DataFrame(proba, columns=OUTCOMES, index=X.index)


MODEL_REGISTRY = {
    "prior": PriorBaseline,
    "elo": EloBaseline,
    "logistic": make_logistic,
    "lightgbm": make_lightgbm,
}
