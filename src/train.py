"""End-to-end experiment runner: data -> features -> walk-forward -> MLflow.

    python -m src.train                      # backtest every model
    python -m src.train --models elo lightgbm
    python -m src.train --refresh            # re-scrape first
    python -m src.train --no-mlflow          # just print the leaderboard

Each model becomes one MLflow run under the `ucl-match-outcome` experiment, with
its per-fold metrics and its raw predictions logged as artifacts. Browse them
with `mlflow ui --backend-store-uri ./mlruns`.
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from src.config import (
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    PROCESSED_DIR,
    RANDOM_SEED,
)
from src.data.build_dataset import build_match_table, load_match_table
from src.evaluation.backtest import DixonColesAdapter, SklearnAdapter, score_by_fold, walk_forward
from src.evaluation.metrics import calibration_table, evaluate
from src.features.build_features import FEATURE_COLUMNS, build_features
from src.models.classifiers import EloBaseline, PriorBaseline, make_lightgbm, make_logistic

log = logging.getLogger(__name__)

FEATURES_PATH = PROCESSED_DIR / "features.parquet"


def build_predictors(names: list[str]) -> dict:
    """Instantiate the requested backtest adapters by name."""
    available = {
        "prior": lambda: SklearnAdapter(PriorBaseline),
        "elo": lambda: SklearnAdapter(EloBaseline, features=FEATURE_COLUMNS),
        "logistic": lambda: SklearnAdapter(make_logistic),
        "lightgbm": lambda: SklearnAdapter(make_lightgbm),
        "dixon_coles": lambda: DixonColesAdapter(xi=0.0018),
    }
    unknown = set(names) - set(available)
    if unknown:
        raise SystemExit(f"unknown model(s): {sorted(unknown)}; pick from {sorted(available)}")
    return {name: available[name]() for name in names}


def prepare(*, refresh: bool = False) -> pd.DataFrame:
    """Load or rebuild the match table, then attach every feature."""
    matches = build_match_table(refresh=True) if refresh else load_match_table(played_only=False)
    features = build_features(matches)
    features.to_parquet(FEATURES_PATH, index=False)
    log.info("Features for %d matches -> %s", len(features), FEATURES_PATH)
    return features


def run(
    models: list[str],
    *,
    refresh: bool = False,
    min_train_seasons: int = 3,
    use_mlflow: bool = True,
) -> pd.DataFrame:
    features = prepare(refresh=refresh)
    predictors = build_predictors(models)

    if use_mlflow:
        import mlflow

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)

    rows = []
    for name, predictor in predictors.items():
        log.info("=== %s ===", name)
        preds = walk_forward(features, predictor, min_train_seasons=min_train_seasons)
        metrics = evaluate(preds["result"], preds[["H", "D", "A"]])
        rows.append({"model": name, **metrics})

        if use_mlflow:
            import mlflow

            with mlflow.start_run(run_name=name):
                mlflow.log_params(
                    {
                        "model": name,
                        "features": ",".join(FEATURE_COLUMNS),
                        "min_train_seasons": min_train_seasons,
                        "n_matches": len(features),
                        "seed": RANDOM_SEED,
                    }
                )
                mlflow.log_metrics({k: v for k, v in metrics.items() if k != "n"})
                mlflow.log_metric("n_predictions", metrics["n"])

                for _, fold_row in score_by_fold(preds).iterrows():
                    mlflow.log_metric("rps_by_season", fold_row["rps"], step=int(fold_row["fold"]))

                mlflow.log_table(preds, artifact_file="predictions.json")
                mlflow.log_table(
                    calibration_table(preds["result"], preds[["H", "D", "A"]]),
                    artifact_file="calibration_home.json",
                )

    leaderboard = pd.DataFrame(rows).sort_values("rps").reset_index(drop=True)
    return leaderboard


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["prior", "elo", "logistic", "lightgbm", "dixon_coles"],
        help="which models to backtest",
    )
    parser.add_argument("--refresh", action="store_true", help="re-scrape FBref before running")
    parser.add_argument("--min-train-seasons", type=int, default=3)
    parser.add_argument("--no-mlflow", action="store_true", help="skip MLflow logging")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    leaderboard = run(
        args.models,
        refresh=args.refresh,
        min_train_seasons=args.min_train_seasons,
        use_mlflow=not args.no_mlflow,
    )
    print("\nWalk-forward leaderboard (lower RPS is better)\n")
    print(leaderboard.to_string(index=False))


if __name__ == "__main__":
    main()
