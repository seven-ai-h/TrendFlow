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

# ── Keyword -> ticker mapping ────────────────────────────────────────────────
# A story mentioning any of these terms contributes to that ticker's daily buzz.
TICKER_MAP = {
    'NVDA': ['nvidia', 'nvda', 'gpu', 'cuda', 'jensen huang', 'geforce', 'h100', 'blackwell'],
    'MSFT': ['microsoft', 'msft', 'azure', 'copilot', 'openai', 'gpt', 'chatgpt', 'windows'],
    'GOOGL': ['google', 'googl', 'alphabet', 'gemini', 'deepmind', 'android', 'chrome', 'waymo'],
    'AAPL': ['apple', 'aapl', 'iphone', 'ios', 'macos', 'ipad', 'vision pro', 'mac'],
    'META': ['meta', 'facebook', 'instagram', 'llama', 'metaverse', 'whatsapp', 'zuckerberg'],
    'AMZN': ['amazon', 'amzn', 'aws', 'bezos', 'prime', 'alexa'],
    'TSLA': ['tesla', 'tsla', 'musk', 'electric vehicle', 'autopilot', 'cybertruck', 'ev'],
    'AMD': ['amd', 'ryzen', 'radeon', 'epyc', 'lisa su'],
    'BTC-USD': ['bitcoin', 'btc', 'crypto', 'satoshi', 'halving'],
    'ETH-USD': ['ethereum', 'eth', 'ether', 'vitalik', 'smart contract'],
}

TICKER_NAMES = {
    'NVDA': 'NVIDIA', 'MSFT': 'Microsoft', 'GOOGL': 'Alphabet', 'AAPL': 'Apple',
    'META': 'Meta', 'AMZN': 'Amazon', 'TSLA': 'Tesla', 'AMD': 'AMD',
    'BTC-USD': 'Bitcoin', 'ETH-USD': 'Ethereum',
}

FEATURE_COLS = [
    'buzz', 'buzz_velocity', 'avg_sentiment', 'sentiment_std', 'bullish_ratio',
    'weighted_sentiment', 'prev_return', 'momentum_3d', 'volume_ratio',
    'volatility_3d', 'day_of_week',
]


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
        buzz_series = sgrp['buzz'] if not sgrp.empty else pd.Series(dtype=float)
        buzz_ma = buzz_series.rolling(3, min_periods=1).mean() if not sgrp.empty else buzz_series

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
            else:
                buzz = buzz_vel = avg_sent = sent_std = bull = wsent = 0.0

            samples.append({
                'ticker': ticker,
                'date': d,
                'buzz': buzz,
                'buzz_velocity': buzz_vel,
                'avg_sentiment': avg_sent,
                'sentiment_std': sent_std,
                'bullish_ratio': bull,
                'weighted_sentiment': wsent,
                'prev_return': prow['prev_return'],
                'momentum_3d': prow['momentum_3d'],
                'volume_ratio': prow['volume_ratio'],
                'volatility_3d': prow['volatility_3d'],
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
