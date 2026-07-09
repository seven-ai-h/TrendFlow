"""
Market price collector — the real-world target signal.

Pulls daily OHLCV for the tickers TrendFlow tracks using yfinance (free,
no API key). Network-restricted environments (like the remote demo box)
can't reach Yahoo Finance; in that case this returns [] gracefully and the
dashboard falls back to the synthetic prices written by seed_data.py.
"""
from datetime import datetime, timedelta

from config import TRACKED_TICKERS, TICKER_NAMES  # single source of truth


def collect_market_data(tickers=None, days_back: int = 60) -> list:
    """
    Fetch daily bars for each ticker. Returns a list of dicts:
    {ticker, date, open, close, high, low, volume, return_pct}
    """
    tickers = tickers or TRACKED_TICKERS
    try:
        import yfinance as yf
    except ImportError:
        print("  yfinance not installed — skipping live market fetch "
              "(pip install yfinance). Using seeded prices.")
        return []

    end = datetime.utcnow()
    start = end - timedelta(days=days_back)
    rows = []

    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start.strftime('%Y-%m-%d'),
                             end=end.strftime('%Y-%m-%d'), progress=False,
                             auto_adjust=True)
            if df is None or df.empty:
                print(f"  {ticker}: no data returned")
                continue
            df = df.reset_index()
            prev_close = None
            for _, r in df.iterrows():
                # yfinance may return multiindex columns; normalise access
                def _v(col):
                    val = r[col]
                    return float(val.iloc[0]) if hasattr(val, 'iloc') else float(val)

                close = _v('Close')
                ret = ((close - prev_close) / prev_close * 100) if prev_close else 0.0
                rows.append({
                    'ticker': ticker,
                    'date': r['Date'].to_pydatetime() if hasattr(r['Date'], 'to_pydatetime') else r['Date'],
                    'open': _v('Open'),
                    'close': close,
                    'high': _v('High'),
                    'low': _v('Low'),
                    'volume': _v('Volume'),
                    'return_pct': ret,
                })
                prev_close = close
            print(f"  {ticker}: {len(df)} daily bars")
        except Exception as e:
            print(f"  {ticker} fetch error: {e}")

    print(f"  Market: {len(rows)} total price rows")
    return rows


def _news_item_fields(item: dict):
    """Normalise a yfinance news item across API versions. Returns (title, url, ts)."""
    # Newer yfinance nests everything under 'content'
    content = item.get('content', item)
    title = content.get('title') or item.get('title', '')

    # URL can live in several places depending on version
    url = ''
    cu = content.get('canonicalUrl') or content.get('clickThroughUrl')
    if isinstance(cu, dict):
        url = cu.get('url', '')
    url = url or content.get('link') or item.get('link', '')

    # timestamp: epoch seconds (old) or ISO string (new)
    ts = datetime.utcnow()
    if item.get('providerPublishTime'):
        try:
            ts = datetime.utcfromtimestamp(int(item['providerPublishTime']))
        except Exception:
            pass
    elif content.get('pubDate'):
        try:
            ts = datetime.fromisoformat(content['pubDate'].replace('Z', '+00:00')).replace(tzinfo=None)
        except Exception:
            pass
    return title.strip(), url, ts


def collect_ticker_news(tickers=None, max_per_ticker: int = 15) -> list:
    """
    Fetch real recent headlines PER ticker via yfinance — already tagged to the
    asset, so no keyword-matching guesswork and no API key. Returns Story dicts.
    """
    tickers = tickers or TRACKED_TICKERS
    try:
        import yfinance as yf
    except ImportError:
        print("  yfinance not installed — skipping ticker news.")
        return []

    results = []
    for ticker in tickers:
        try:
            items = yf.Ticker(ticker).news or []
            got = 0
            for item in items[:max_per_ticker]:
                title, url, ts = _news_item_fields(item)
                if not title:
                    continue
                results.append({
                    'title': title,
                    'url': url,
                    'score': 0,
                    'num_comments': 0,
                    'platform': 'finance',
                    '_ticker': ticker,       # already known — no matching needed
                    '_timestamp': ts,
                })
                got += 1
            print(f"  {ticker} news: {got} headlines")
        except Exception as e:
            print(f"  {ticker} news error: {e}")

    print(f"  Ticker news: {len(results)} total headlines")
    return results
