"""
Market price collector — the real-world target signal.

Pulls daily OHLCV for the tickers TrendFlow tracks using yfinance (free,
no API key). Network-restricted environments (like the remote demo box)
can't reach Yahoo Finance; in that case this returns [] gracefully and the
dashboard falls back to the synthetic prices written by seed_data.py.
"""
from datetime import datetime, timedelta

# The universe TrendFlow tracks — tech equities + major crypto.
TRACKED_TICKERS = [
    'NVDA', 'MSFT', 'GOOGL', 'AAPL', 'META', 'AMZN', 'TSLA', 'AMD',
    'BTC-USD', 'ETH-USD',
]


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
