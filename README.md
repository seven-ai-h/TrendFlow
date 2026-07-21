# AlphaSignal

AlphaSignal is an end-to-end machine-learning pipeline that predicts the
**next-day return** of a small universe of technology equities and major
cryptocurrencies from **news-headline sentiment** combined with price and
market-context features. It benchmarks three model families on the same task
and presents the results, the live signals, and the full data lineage in a
Streamlit dashboard.

The emphasis of this project is the *engineering*: data collection, feature
engineering, leakage-free evaluation, and model comparison, wired together into
a reproducible pipeline. It is a research/portfolio system, not a trading
product — see [Honest limitations](#honest-limitations).

## Contents

- [What it does](#what-it-does)
- [Architecture](#architecture)
- [Design decisions](#design-decisions)
- [Repository layout](#repository-layout)
- [Getting started](#getting-started)
- [Configuration](#configuration)
- [Data model](#data-model)
- [Modeling and evaluation](#modeling-and-evaluation)
- [Honest limitations](#honest-limitations)
- [Roadmap](#roadmap)
- [License](#license)

## What it does

1. Collects recent headlines per ticker (Yahoo Finance news, plus Reddit and RSS
   feeds) and daily OHLCV price bars (Yahoo Finance via `yfinance`).
2. Scores every headline for sentiment (VADER plus a finance-specific lexicon).
3. Aggregates headlines and prices into one feature vector per `(ticker, day)` —
   sentiment level, momentum and dispersion, cross-sectional rank, buzz volume,
   price momentum, volatility, and a whole-market context term.
4. Trains and compares three models — Linear Regression, Random Forest, and an
   LSTM — on a strict temporal train/test split, predicting next-day return.
5. Serves a four-page dashboard: model comparison and a strategy backtest, live
   BUY / HOLD / SELL signals with plain-language reasons, sentiment breakdowns,
   and a "how it works" page that traces a single prediction end to end.

## Architecture

```
 Data sources                Processing                 Serving
 ------------                ----------                 -------
 yfinance (prices)  ─┐
 yfinance (news)     ├─►  sentiment scoring  ─►  feature engineering  ─►  model training
 Reddit / RSS       ─┘    (analysis/            (analysis/                (analysis/
 NewsAPI (optional)        sentiment.py)          market_features.py)      model_lab.py)
                                                        │                        │
                                 SQLite (SQLAlchemy) ◄──┘                        ▼
                                 database/models.py                     Streamlit dashboard
                                                                        (dashboard.py)
```

The collector writes to SQLite; every other stage reads from it. Training is
pure and cached, so the dashboard re-aggregates results for any asset selection
without retraining.

## Design decisions

A few choices that are worth calling out, because they are the difference
between a notebook and a system:

- **Leakage-free evaluation.** The train/test split is temporal — models are
  trained on the earlier period and tested on the later one. The label is the
  *next* day's return, computed strictly after the feature window. No future
  information reaches the features.
- **A linear baseline is a first-class model, not an afterthought.** On a noisy,
  low-signal target, a simple model is often competitive with or better than a
  tuned ensemble or an LSTM. The leaderboard reports this honestly rather than
  cherry-picking the neural network.
- **Configuration is data, not code.** The tracked universe (tickers, display
  names, and the keywords that map a headline to a ticker) lives in `config.py`
  and can be overridden entirely with a `tickers.json` file or the
  `TRENDFLOW_TICKERS` environment variable — no source changes required.
- **Graceful degradation.** The LSTM is skipped with a clear message if PyTorch
  is unavailable; the AI-adjacent features no-op without API keys; the collector
  tolerates individual source failures without aborting the run.
- **Explainability by construction.** Every live signal exposes the specific,
  numeric evidence behind it (sentiment, headline counts, price momentum) and
  the model's per-feature contribution, so a prediction is never a bare number.
- **Reproducible offline demo.** `seed_data.py` generates a synthetic dataset in
  which prices are partially driven by lagged sentiment, so the full pipeline is
  runnable and the models are learnable without any network access.

## Repository layout

```
config.py                  Single source of truth for the tracked universe
dashboard.py               Streamlit app (four pages)
seed_data.py               Generate the synthetic demo dataset
test_hn_api.py             Live collection pipeline (headlines + prices)
scheduler.py               Run the collector on a fixed interval

analysis/
  sentiment.py             VADER + finance-lexicon headline scoring
  market_features.py       Feature engineering; keyword -> ticker mapping
  model_lab.py             Training, evaluation, backtest, live predictions

data_collection/
  market_collector.py      yfinance prices and per-ticker news
  reddit_collector.py      Reddit JSON API
  rss_collector.py         RSS feeds (feedparser with an stdlib fallback)
  news_collector.py        NewsAPI (optional)
  devto_collector.py       Dev.to API
  github_collector.py      GitHub trending

database/
  models.py                SQLAlchemy models
  db_setup.py              Engine and session helpers
```

The repository also contains modules from an earlier keyword-detection
iteration; the pipeline above is the current, active system.

## Getting started

Requires Python 3.10+.

```bash
git clone https://github.com/seven-ai-h/TrendFlow.git
cd TrendFlow

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Run the demo (no network or keys required)

```bash
python seed_data.py               # build the synthetic dataset (trendflow.db)
streamlit run dashboard.py        # open http://localhost:8501
```

### Run against live data

```bash
python test_hn_api.py             # collect real headlines + prices once
streamlit run dashboard.py
```

To keep the database growing on its own, run the scheduler (collects
immediately, then every 24 hours; interval configurable):

```bash
python scheduler.py
TRENDFLOW_INTERVAL_HOURS=48 python scheduler.py
```

The first dashboard load trains the models (a few seconds, including the LSTM)
and caches the result; subsequent interactions are instant.

## Configuration

All keys are optional. Copy `.env.example` to `.env` and fill in what you have:

| Variable            | Purpose                                             |
| ------------------- | --------------------------------------------------- |
| `ANTHROPIC_API_KEY` | Optional LLM features                               |
| `NEWS_API_KEY`      | Optional NewsAPI headline source                    |
| `GITHUB_TOKEN`      | Raises the GitHub API rate limit                    |
| `TRENDFLOW_TICKERS` | Path to a JSON file overriding the tracked universe |

To change the universe without touching code, drop a `tickers.json` in the
project root:

```json
{
  "NVDA": { "name": "NVIDIA", "keywords": ["nvidia", "gpu", "cuda"] },
  "COIN": { "name": "Coinbase", "keywords": ["coinbase", "exchange"] }
}
```

## Data model

SQLite via SQLAlchemy (`database/models.py`):

- **Story** — a headline: title, score, comments, URL, platform, sentiment,
  timestamp.
- **MarketData** — a daily price bar: ticker, date, OHLC, volume, return.
- **PipelineRun** — one collection run: status, counts, sources, duration.
- **Keyword**, **Article** — supporting tables from the collectors.

## Modeling and evaluation

- **Task:** regression on next-day percentage return, per `(ticker, day)`. A
  BUY / HOLD / SELL label is derived by thresholding the predicted return.
- **Features (16):** sentiment level, engagement-weighted sentiment, sentiment
  momentum and volatility, bullish ratio, cross-sectional sentiment rank, buzz
  volume and velocity, previous return, 3- and 5-day price momentum, volatility,
  volume ratio, a whole-market return term, and day of week.
- **Models:** Linear Regression and Random Forest on the tabular features; an
  LSTM (PyTorch) over the trailing five-day feature sequence.
- **Metrics:** MAE, RMSE, R^2, directional accuracy, and d-prime (a
  signal-detection measure of how cleanly up-days are separated from down-days).
- **Backtest:** a long-or-flat strategy is evaluated over the held-out period and
  reported with total return, annualised Sharpe, maximum drawdown, and win rate,
  benchmarked against buy-and-hold.

## Honest limitations

- **The demo runs on synthetic data.** The seeded prices are deliberately driven
  by sentiment so the pipeline is learnable offline. Metrics on this data are
  optimistic and the backtest statistics are flattering; they demonstrate that
  the system computes the right things, not that the strategy is profitable.
- **Real markets are close to efficient.** On live data, expect directional
  accuracy in the low-to-mid 50s and risk-adjusted returns far below the demo.
  Treat any single prediction as a weak prior.
- **This is not financial advice** and not a production trading system.

## Roadmap

- Walk-forward (rolling-origin) evaluation in place of a single split.
- SHAP-based attribution alongside the linear contributions.
- Model and prediction persistence so training survives restarts.
- Containerised deployment.

## License

MIT.
