"""
Feature engineering for the sentiment -> market task.

Joins the daily SOCIAL signal (how much a ticker is being talked about, and how
positively) with the daily PRICE signal (momentum, volume) to predict the real,
observable target: does the asset close UP tomorrow?

This is where keyword/mention volume is "folded in" — it becomes the `buzz`
feature per ticker, not a prediction target of its own.
"""
import re
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from database.models import Story, MarketData
from config import TICKER_KEYWORDS as TICKER_MAP, TICKER_NAMES  # single source of truth

FEATURE_COLS = [
    'buzz', 'buzz_velocity', 'avg_sentiment', 'sentiment_std', 'bullish_ratio',
    'weighted_sentiment', 'sentiment_momentum', 'sentiment_volatility',
    'cross_sec_rank', 'prev_return', 'momentum_3d', 'volume_ratio',
    'volatility_3d', 'market_return', 'day_of_week',
]

# Human-readable labels for the dashboard's "why" / how-it-works views
FEATURE_LABELS = {
    'buzz': 'headline volume',
    'buzz_velocity': 'buzz vs recent avg',
    'avg_sentiment': 'avg sentiment',
    'sentiment_std': 'sentiment spread',
    'bullish_ratio': 'share of bullish headlines',
    'weighted_sentiment': 'engagement-weighted sentiment',
    'sentiment_momentum': 'sentiment momentum (accelerating?)',
    'sentiment_volatility': 'sentiment volatility',
    'cross_sec_rank': 'sentiment rank vs other assets',
    'prev_return': 'yesterday’s return',
    'momentum_3d': '3-day price momentum',
    'volume_ratio': 'volume vs 5-day avg',
    'volatility_3d': '3-day volatility',
    'market_return': 'whole-market move today',
    'day_of_week': 'day of week',
}


def tickers_in_text(text: str) -> list:
    """Return the tickers whose keywords appear in the text."""
    low = text.lower()
    hits = []
    for ticker, terms in TICKER_MAP.items():
        for term in terms:
            # word-boundary match so 'eth' doesn't hit 'method'
            if re.search(rf'\b{re.escape(term)}\b', low):
                hits.append(ticker)
                break
    return hits


def _daily_social(session, days_back: int) -> pd.DataFrame:
    """Aggregate stories into per-(ticker, date) social features."""
    cutoff = datetime.utcnow() - timedelta(days=days_back)
    stories = session.query(Story).filter(Story.timestamp >= cutoff).all()

    recs = []
    for s in stories:
        for ticker in tickers_in_text(s.title or ''):
            recs.append({
                'ticker': ticker,
                'date': s.timestamp.date(),
                'sentiment': s.sentiment if s.sentiment is not None else 0.0,
                'score': s.score or 0,
            })
    if not recs:
        return pd.DataFrame()

    df = pd.DataFrame(recs)
    agg = df.groupby(['ticker', 'date']).agg(
        buzz=('sentiment', 'size'),
        avg_sentiment=('sentiment', 'mean'),
        sentiment_std=('sentiment', 'std'),
        bullish_ratio=('sentiment', lambda x: (x > 0.15).mean()),
        weighted_sentiment=('sentiment', 'mean'),  # refined below with score weights
        total_score=('score', 'sum'),
    ).reset_index()

    # engagement-weighted sentiment
    wsum = df.groupby(['ticker', 'date']).apply(
        lambda g: np.average(g['sentiment'], weights=(g['score'] + 1))
    ).reset_index(name='weighted_sentiment_w')
    agg = agg.merge(wsum, on=['ticker', 'date'], how='left')
    agg['weighted_sentiment'] = agg['weighted_sentiment_w'].fillna(agg['avg_sentiment'])
    agg.drop(columns=['weighted_sentiment_w'], inplace=True)
    agg['sentiment_std'] = agg['sentiment_std'].fillna(0.0)
    return agg


def _daily_prices(session, days_back: int) -> pd.DataFrame:
    cutoff = datetime.utcnow() - timedelta(days=days_back + 5)
    rows = session.query(MarketData).filter(MarketData.date >= cutoff).all()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([{
        'ticker': r.ticker,
        'date': r.date.date() if hasattr(r.date, 'date') else r.date,
        'close': r.close,
        'volume': r.volume,
        'return_pct': r.return_pct,
    } for r in rows])


def build_market_dataset(session, days_back: int = 60) -> pd.DataFrame:
    """
    Build the supervised training frame: one row per (ticker, date) with social
    + price features and a next-day-up label. Requires both social and price data.
    """
    social = _daily_social(session, days_back)
    prices = _daily_prices(session, days_back)
    if social.empty or prices.empty:
        return pd.DataFrame()

    social['date'] = pd.to_datetime(social['date'])
    prices['date'] = pd.to_datetime(prices['date'])

    # ── Cross-asset context (computed across ALL tickers per day) ─────────────
    # market_return = equal-weight universe return that day (a market-context
    # feature: "is the whole market up today?"). Known at close of day d → no leak.
    market_ret = prices.groupby('date')['return_pct'].mean()
    # cross_sec_rank = percentile rank of this asset's sentiment vs all others today
    social['cross_sec_rank'] = social.groupby('date')['avg_sentiment'].rank(pct=True)
    rank_lookup = social.set_index(['ticker', 'date'])['cross_sec_rank'].to_dict()

    samples = []
    for ticker, pgrp in prices.groupby('ticker'):
        pgrp = pgrp.sort_values('date').reset_index(drop=True)
        if len(pgrp) < 6:
            continue
        pgrp['momentum_3d'] = pgrp['close'].pct_change(3).fillna(0) * 100
        pgrp['volatility_3d'] = pgrp['return_pct'].rolling(3, min_periods=1).std().fillna(0)
        vol_ma = pgrp['volume'].rolling(5, min_periods=1).mean()
        pgrp['volume_ratio'] = (pgrp['volume'] / vol_ma).fillna(1.0)
        pgrp['prev_return'] = pgrp['return_pct'].shift(1).fillna(0)
        pgrp['next_return'] = pgrp['return_pct'].shift(-1)

        sgrp = social[social['ticker'] == ticker].sort_values('date').set_index('date')
        if not sgrp.empty:
            buzz_ma = sgrp['buzz'].rolling(3, min_periods=1).mean()
            # sentiment momentum: today vs the mean of the prior up-to-3 days
            sent_prev_ma = sgrp['avg_sentiment'].shift(1).rolling(3, min_periods=1).mean()
            sent_mom = (sgrp['avg_sentiment'] - sent_prev_ma).fillna(0)
            sent_vol = sgrp['avg_sentiment'].rolling(3, min_periods=1).std().fillna(0)

        for _, prow in pgrp.iterrows():
            if pd.isna(prow['next_return']):
                continue
            d = prow['date']
            if not sgrp.empty and d in sgrp.index:
                srow = sgrp.loc[d]
                if isinstance(srow, pd.DataFrame):
                    srow = srow.iloc[0]
                buzz = float(srow['buzz'])
                base_buzz = float(buzz_ma.loc[d]) if d in buzz_ma.index else buzz
                buzz_vel = buzz / base_buzz if base_buzz > 0 else 1.0
                avg_sent = float(srow['avg_sentiment'])
                sent_std = float(srow['sentiment_std'])
                bull = float(srow['bullish_ratio'])
                wsent = float(srow['weighted_sentiment'])
                s_mom = float(sent_mom.loc[d]) if d in sent_mom.index else 0.0
                s_vol = float(sent_vol.loc[d]) if d in sent_vol.index else 0.0
                x_rank = float(rank_lookup.get((ticker, d), 0.5))
            else:
                buzz = buzz_vel = avg_sent = sent_std = bull = wsent = 0.0
                s_mom = s_vol = 0.0
                x_rank = 0.5  # neutral rank when no coverage

            samples.append({
                'ticker': ticker,
                'date': d,
                'buzz': buzz,
                'buzz_velocity': buzz_vel,
                'avg_sentiment': avg_sent,
                'sentiment_std': sent_std,
                'bullish_ratio': bull,
                'weighted_sentiment': wsent,
                'sentiment_momentum': s_mom,
                'sentiment_volatility': s_vol,
                'cross_sec_rank': x_rank,
                'prev_return': prow['prev_return'],
                'momentum_3d': prow['momentum_3d'],
                'volume_ratio': prow['volume_ratio'],
                'volatility_3d': prow['volatility_3d'],
                'market_return': float(market_ret.get(d, 0.0)),
                'day_of_week': pd.Timestamp(d).dayofweek,
                'next_return': prow['next_return'],
                'will_rise': 1 if prow['next_return'] > 0 else 0,
            })

    return pd.DataFrame(samples)


def build_live_snapshot(session, days_back: int = 30) -> pd.DataFrame:
    """Latest feature row per ticker — for live 'what does each model say' predictions."""
    ds = build_market_dataset(session, days_back=days_back)
    if ds.empty:
        return ds
    latest = ds.sort_values('date').groupby('ticker').tail(1).reset_index(drop=True)
    latest['name'] = latest['ticker'].map(TICKER_NAMES).fillna(latest['ticker'])
    return latest
