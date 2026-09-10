# UCL Predictor

Match-outcome modelling for the Champions League, built in stages.

**Stage 1 (this repo, working today)** is the data pipeline and the baselines:
football-data.co.uk results and bookmaker odds for the big five domestic leagues,
a canonical match table, leakage-free features, and a walk-forward backtest that
scores every model against the market's own implied probabilities.

Stage 1 is deliberately *not* a Champions League model. football-data.co.uk
contains no UCL matches, and a UCL season is only ~125–190 matches — far too few
to fit on alone. What the domestic data buys is 25,000 matches with a bookmaker
benchmark attached, which is enough to build and honestly evaluate the machinery
before pointing it at the competition that matters.

## Setup

```bash
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.lock.txt   # exact tested set
```

## Running

Each stage reads and writes parquet under `data/`, so any stage can be re-run on
its own. Only `ingest` touches the network.

```bash
python pipeline.py ingest      # football-data.co.uk CSVs -> data/raw   (~1 min)
python pipeline.py dataset     # raw -> canonical match table
python pipeline.py features    # match table -> Elo, form, rest, experience
python pipeline.py backtest    # walk-forward comparison    (add --fast to skip the slow two)
python pipeline.py train       # fit on all history, log to MLflow
```

## Results

14 seasons (2012/13–2025/26), five leagues, 25,241 matches, of which 19,763 are
scored (the first three seasons per league are held back as the initial training
window). Every model sees identical folds.

| model | RPS | log loss | Brier | accuracy |
|---|---|---|---|---|
| **market** (Pinnacle, de-vigged) | **0.1948** | 0.9665 | 0.5742 | 53.8% |
| elo | 0.2009 | 0.9869 | 0.5878 | 52.5% |
| logistic | 0.2015 | 0.9962 | 0.5898 | 52.5% |
| dixon_coles | 0.2035 | 0.9941 | 0.5931 | 51.9% |
| lightgbm | 0.2045 | 1.0148 | 0.5974 | 51.8% |
| prior (base rates) | 0.2299 | 1.0711 | 0.6479 | 44.1% |

Read this the right way round. **The market winning is the expected and correct
outcome.** Closing odds aggregate the money of everyone with an opinion,
including people with team news this pipeline never sees. Approaching them is the
Stage 1 success condition; the gap between `prior` and `elo` is the skill the
pipeline actually adds, and the gap between `elo` and `market` is what is left to
find.

If a model ever beats the market by a wide margin, suspect a leaked feature
before believing it — `pipeline.py backtest` prints a warning when this happens.

Ignore accuracy. Draws are ~25% of matches and almost never the argmax, so a
model can score well on accuracy while being useless. RPS is the metric; it is
ordinal, so predicting a home win when the away side wins is punished more than
predicting a draw.

## Layout

```
pipeline.py           Typer CLI, one subcommand per stage
src/config.py         paths, division codes, seasons, outcome ordering
src/ingest/
  football_data.py    the CSV fetcher and its three parsing hazards
  odds.py             bookmaker preference chain + overround removal
  teams.py            name canonicalisation across sources
  team_aliases.json   the alias map
  build_dataset.py    the canonical match table every downstream module reads
  clubelo.py          optional cross-check; fails soft (see below)
  fbref.py            Stage 2, wired but not on the path
src/features/
  elo.py              forward-only ratings, pooled per league
  build_features.py   rolling form, rest, experience — all shifted by one match
src/models/
  dixon_coles.py      time-decayed bivariate Poisson with the low-score correction
  classifiers.py      prior / market / Elo baselines, logistic, LightGBM
src/evaluate/
  metrics.py          RPS, log loss, Brier, calibration tables
  backtest.py         expanding-window walk-forward behind one adapter interface
```

## Source status

Verified live on 2026-09-10. These change without notice — re-check before
relying on any of them.

| source | state | role |
|---|---|---|
| football-data.co.uk | Live. Plain static CSVs, no blocking. | Stage 1 foundation |
| ClubElo | **Down.** `api.clubelo.com` returns 502 on every endpoint; `/Fixtures` replies "Fixtures API deactivated". | optional; `src/ingest/clubelo.py` degrades to an empty frame |
| FBref | 403 on direct requests (Cloudflare); needs `soccerdata`'s browser backend. | Stage 2 — the only listed source with UCL results |
| Understat | Site up, but the old inline-`JSON.parse` scrape is dead. `soccerdata` ≥1.9 targets the new `/getLeagueData` JSON API. | Stage 3 (xG) |
| Transfermarkt | No API; Kaggle dumps. | Stage 3 (squad value, injuries) |
| StatsBomb | `statsbombpy` not installed; free data has thin UCL coverage. | learning only |
| API-Football | Needs a key; free tier ~100 req/day. | Stage 4 (live fixtures) |

## Two constraints worth knowing before extending this

**Elo is pooled per league.** Domestic leagues form a disconnected graph — no
Premier League side ever plays a Bundesliga side in this dataset — so ratings are
comparable *within* a league and meaningless *across* leagues. `add_elo_features`
takes `pool="league"` for that reason. Champions League fixtures are the edges
that would connect the components, which is exactly what Stage 2 adds.

**Odds are a benchmark, not a feature.** `FEATURE_COLUMNS` deliberately excludes
them. Feeding the market's probabilities in as inputs makes "beat the market"
circular: the model inherits the market's skill and reports it as its own.
`FEATURE_COLUMNS_WITH_ODDS` exists for the different question of what a model
adds *on top of* the market.

## Tests

```bash
python -m pytest        # 82 tests, no network access
python -m ruff check .
```

The suite runs entirely against synthetic fixtures. The live sources are slow,
sometimes down and always changing; a suite that depended on them would fail for
reasons unrelated to this code.
