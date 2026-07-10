"""
Seed the database for the sentiment -> market demo.

Generates, over ~50 trading days for each tracked ticker:
  * sentiment-tagged news/social STORIES (headlines that mention the ticker)
  * a synthetic PRICE series whose next-day return is partially driven by that
    day's LAGGED sentiment + buzz — so the ML task is genuinely learnable while
    still noisy (like the real thing).

On a machine with network access, run `python test_hn_api.py` instead to pull
real headlines + real prices via yfinance.
"""
import math
import random
from datetime import datetime, timedelta

from database.db_setup import db_connection, getSession
from database.models import Story, Article, MarketData, PipelineRun, Keyword
from analysis.sentiment import score_sentiment
from config import TRACKED_TICKERS, TICKER_NAMES, TICKER_KEYWORDS, seed_start_price

random.seed(11)

DAYS = 130
TICKERS = TRACKED_TICKERS

# Headline templates by tone. {name}/{kw} filled per ticker.
BULLISH = [
    "{name} soars to record high on strong {kw} demand",
    "{name} beats earnings expectations, {kw} revenue surges",
    "Analysts upgrade {name} as {kw} growth accelerates",
    "{name} announces breakthrough in {kw}, stock rallies",
    "{name} {kw} milestone sends shares higher",
    "Bullish outlook: {name} dominates {kw} market",
]
BEARISH = [
    "{name} plunges after disappointing {kw} guidance",
    "{name} faces lawsuit over {kw}, shares slump",
    "Analysts downgrade {name} amid {kw} slowdown",
    "{name} misses estimates as {kw} demand cools",
    "Regulators probe {name} {kw} practices, stock falls",
    "{name} announces layoffs, {kw} unit hit hardest",
]
NEUTRAL = [
    "{name} to report quarterly earnings next week",
    "{name} unveils new {kw} roadmap at conference",
    "What {name}'s {kw} strategy means for the industry",
    "{name} {kw} update rolls out to users",
    "{name} hires new executive to lead {kw} division",
]

def _display_keyword(ticker: str, name: str) -> str:
    """Pick a natural filler keyword for headline templates from the config
    keywords — skip the company name and the bare ticker symbol so headlines
    read like 'strong AWS demand' rather than 'strong amzn demand'."""
    name_low = name.lower()
    symbol = ticker.split("-")[0].lower()  # 'BTC-USD' -> 'btc'
    for kw in TICKER_KEYWORDS.get(ticker, []):
        if kw in name_low or name_low in kw or kw == symbol or len(kw) <= 2:
            continue
        return kw
    return "tech"


def seed():
    print("Initializing database…")
    db_connection()
    session = getSession()

    for tbl in (Story, Article, MarketData, PipelineRun, Keyword):
        session.query(tbl).delete()
    session.commit()
    print("Cleared existing data")

    now = datetime.utcnow()
    base_day = now - timedelta(days=DAYS)

    total_stories = 0
    total_prices = 0

    for ticker in TICKERS:
        name = TICKER_NAMES[ticker]
        kw = _display_keyword(ticker, name)
        price = seed_start_price(ticker)
        # each ticker has its own daily sentiment "mood" random walk
        mood = 0.0
        daily_sentiment = {}

        # ── Generate stories + record each day's aggregate sentiment ──────────
        for d in range(DAYS):
            day_ts = base_day + timedelta(days=d)
            mood = 0.7 * mood + 0.3 * random.uniform(-1, 1)  # autocorrelated mood
            # number of stories scales with |mood| (big news days are busier)
            n_stories = random.randint(1, 3) + int(abs(mood) * 3)
            day_scores = []
            for _ in range(n_stories):
                r = random.random()
                if mood > 0.2:
                    pool = BULLISH if r < 0.6 else (NEUTRAL if r < 0.85 else BEARISH)
                elif mood < -0.2:
                    pool = BEARISH if r < 0.6 else (NEUTRAL if r < 0.85 else BULLISH)
                else:
                    pool = NEUTRAL if r < 0.5 else (BULLISH if r < 0.75 else BEARISH)
                title = random.choice(pool).format(name=name, kw=kw)
                sent = score_sentiment(title)
                day_scores.append(sent)
                platform = random.choice(['hackernews', 'reddit', 'news', 'rss', 'devto'])
                session.add(Story(
                    title=title,
                    score=random.randint(5, 900),
                    num_comments=random.randint(0, 400),
                    url=f"https://example.com/{ticker}/{d}/{_}",
                    platform=platform,
                    sentiment=sent,
                    timestamp=day_ts - timedelta(hours=random.randint(0, 20)),
                ))
                total_stories += 1
            daily_sentiment[d] = sum(day_scores) / len(day_scores) if day_scores else 0.0

        # ── Generate prices: next-day return driven by lagged sentiment ───────
        prev_close = price
        for d in range(DAYS):
            day_ts = base_day + timedelta(days=d)
            # yesterday's sentiment + 2-day momentum drive today's return
            # (the learnable signal — strong enough to beat 50%, noisy enough
            #  to stay realistic so no model hits ceiling)
            signal = daily_sentiment.get(d - 1, 0.0)
            signal2 = daily_sentiment.get(d - 2, 0.0)
            drift = signal * 4.2 + signal2 * 1.5   # sentiment effect (%)
            noise = random.gauss(0, 0.7)           # market noise (%)
            ret = drift + noise
            close = prev_close * (1 + ret / 100.0)
            high = close * (1 + abs(random.gauss(0, 0.6)) / 100)
            low = close * (1 - abs(random.gauss(0, 0.6)) / 100)
            openp = prev_close * (1 + random.gauss(0, 0.3) / 100)
            vol = abs(random.gauss(1, 0.4)) * (1 + abs(signal)) * 1e6
            session.add(MarketData(
                ticker=ticker, date=day_ts, open=openp, close=close,
                high=high, low=low, volume=vol, return_pct=ret))
            total_prices += 1
            prev_close = close

    session.commit()
    print(f"Seeded {total_stories} sentiment-tagged stories")
    print(f"Seeded {total_prices} price bars across {len(TICKERS)} tickers")

    # ── Pipeline runs ─────────────────────────────────────────────────────────
    for i in range(6):
        start = now - timedelta(hours=i * 5)
        session.add(PipelineRun(
            started_at=start,
            finished_at=start + timedelta(seconds=random.randint(9, 28)),
            status='success',
            stories_collected=random.randint(60, 200),
            keywords_extracted=random.randint(90, 220),
            sources_run='hackernews,reddit,news,rss,market'))
    session.commit()
    print("Seeded 6 pipeline runs")
    print("\nDone! Restart / refresh the dashboard.")


if __name__ == "__main__":
    seed()
