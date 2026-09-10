"""Command-line entry point for the UCL predictor pipeline.

Each stage reads and writes parquet under `data/`, so any stage can be re-run on
its own without repeating the one before it — in particular, nothing after
`ingest` touches the network.

    python pipeline.py ingest      # football-data.co.uk CSVs -> data/raw
    python pipeline.py dataset     # raw -> canonical match table
    python pipeline.py features    # match table -> Elo, form, rest, experience
    python pipeline.py backtest    # walk-forward comparison of every baseline
    python pipeline.py train       # fit on all history, log to MLflow
"""

from __future__ import annotations

import logging

import pandas as pd
import typer

from src.config import DEFAULT_DIVISIONS, DEFAULT_SEASONS, PROCESSED_DIR

app = typer.Typer(add_completion=False, help=__doc__)

FEATURES_PATH = PROCESSED_DIR / "features.parquet"
PREDICTIONS_PATH = PROCESSED_DIR / "backtest_predictions.parquet"
LEADERBOARD_PATH = PROCESSED_DIR / "leaderboard.csv"


def _setup_logging(verbose: bool = True) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )


def _parse_seasons(spec: str | None) -> list[int]:
    """Accept "2012-2025", "2018,2019,2020" or None for the configured default."""
    if not spec:
        return DEFAULT_SEASONS
    if "-" in spec:
        start, end = spec.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(s) for s in spec.split(",") if s.strip()]


def _parse_divisions(spec: str | None) -> list[str]:
    if not spec:
        return DEFAULT_DIVISIONS
    return [d.strip().upper() for d in spec.split(",") if d.strip()]


@app.command()
def ingest(
    seasons: str = typer.Option(None, help='Season range "2012-2025" or list "2018,2019".'),
    divisions: str = typer.Option(None, help='Division codes, e.g. "E0,SP1".'),
    refresh: bool = typer.Option(False, help="Re-download instead of using the cache."),
) -> None:
    """Download match results and bookmaker odds from football-data.co.uk."""
    _setup_logging()
    from src.ingest.football_data import fetch

    raw = fetch(_parse_divisions(divisions), _parse_seasons(seasons), refresh=refresh)
    typer.echo(f"Fetched {len(raw):,} raw rows across {raw['division'].nunique()} divisions.")


@app.command()
def dataset() -> None:
    """Fold the raw CSVs into the canonical match table."""
    _setup_logging()
    from src.ingest import build_dataset
    from src.ingest.football_data import COMBINED_PATH

    if not COMBINED_PATH.exists():
        raise typer.BadParameter(f"{COMBINED_PATH} not found — run `ingest` first")

    matches = build_dataset.save(build_dataset.build(pd.read_parquet(COMBINED_PATH)))
    shares = matches["result"].value_counts(normalize=True)
    typer.echo(
        f"{len(matches):,} matches | "
        f"H {shares.get('H', 0):.1%} D {shares.get('D', 0):.1%} A {shares.get('A', 0):.1%}"
    )


@app.command()
def features(
    form_window: int = typer.Option(5, help="Matches in the rolling form window."),
) -> None:
    """Build the modelling features from the canonical match table."""
    _setup_logging()
    from src.features.build_features import FEATURE_COLUMNS, build_features
    from src.ingest import build_dataset

    df = build_features(build_dataset.load(), form_window=form_window)
    df.to_parquet(FEATURES_PATH, index=False)
    typer.echo(f"Wrote {len(df):,} rows x {len(FEATURE_COLUMNS)} features to {FEATURES_PATH}")


def _load_features() -> pd.DataFrame:
    if not FEATURES_PATH.exists():
        raise typer.BadParameter(f"{FEATURES_PATH} not found — run `features` first")
    return pd.read_parquet(FEATURES_PATH)


def _predictors(fast: bool) -> dict:
    """The models to compare. `fast` drops the two slowest for a quick loop."""
    from src.evaluate.backtest import DixonColesAdapter, MarketAdapter, SklearnAdapter
    from src.models.classifiers import (
        EloBaseline,
        PriorBaseline,
        make_lightgbm,
        make_logistic,
    )

    models = {
        "prior": SklearnAdapter(PriorBaseline),
        "market": MarketAdapter(),
        "elo": SklearnAdapter(EloBaseline),
        "logistic": SklearnAdapter(make_logistic),
    }
    if not fast:
        models["dixon_coles"] = DixonColesAdapter()
        models["lightgbm"] = SklearnAdapter(make_lightgbm)
    return models


@app.command()
def backtest(
    min_train_seasons: int = typer.Option(3, help="Seasons held back before the first fold."),
    fast: bool = typer.Option(False, help="Skip Dixon-Coles and LightGBM."),
    pool: str = typer.Option("league", help='Fit separately per pool; "none" for one global fit.'),
) -> None:
    """Walk-forward backtest every baseline and print the leaderboard."""
    _setup_logging()
    from src.evaluate.backtest import compare

    df = _load_features()
    leaderboard, predictions = compare(
        df,
        _predictors(fast),
        min_train_seasons=min_train_seasons,
        pool=None if pool.lower() == "none" else pool,
    )

    leaderboard.to_csv(LEADERBOARD_PATH, index=False)
    pd.concat(
        [p.assign(model=name) for name, p in predictions.items()], ignore_index=True
    ).to_parquet(PREDICTIONS_PATH, index=False)

    typer.echo("\n" + leaderboard.to_string(index=False))
    typer.echo(f"\nLeaderboard -> {LEADERBOARD_PATH}\nPredictions -> {PREDICTIONS_PATH}")

    market = leaderboard[leaderboard["model"] == "market"]
    if not market.empty:
        beat = leaderboard[leaderboard["rps"] < market["rps"].iloc[0]]["model"].tolist()
        beat = [m for m in beat if m != "market"]
        if beat:
            typer.secho(
                f"\n{', '.join(beat)} beat the market on RPS. Closing odds are very hard to "
                "beat — check the feature layer for leakage before believing it.",
                fg=typer.colors.YELLOW,
            )


@app.command()
def train(
    model: str = typer.Option("lightgbm", help="One of: prior, elo, logistic, lightgbm"),
) -> None:
    """Fit one model on all history and log it to MLflow."""
    _setup_logging()
    import mlflow

    from src.config import MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI
    from src.evaluate.metrics import evaluate
    from src.features.build_features import FEATURE_COLUMNS, xy
    from src.models.classifiers import MODEL_REGISTRY, predict_proba_frame

    # "market" is in the registry as a benchmark, but it has nothing to fit and
    # reads odds columns rather than features. Backtest it instead.
    trainable = [m for m in MODEL_REGISTRY if m != "market"]
    if model == "market":
        raise typer.BadParameter(
            "the market baseline has nothing to train — see `pipeline.py backtest`"
        )
    if model not in trainable:
        raise typer.BadParameter(f"unknown model {model!r}; choose from {trainable}")

    df = _load_features()
    X, y = xy(df)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=model):
        estimator = MODEL_REGISTRY[model]()
        estimator.fit(X, y)

        # In-sample scores only — they say the fit converged, nothing about
        # skill. `backtest` is the honest number.
        metrics = evaluate(y, predict_proba_frame(estimator, X))
        mlflow.log_params({"model": model, "n_features": len(FEATURE_COLUMNS), "n_train": len(X)})
        mlflow.log_metrics({f"insample_{k}": v for k, v in metrics.items()})
        typer.echo(f"{model}: in-sample {metrics}")


if __name__ == "__main__":
    app()
