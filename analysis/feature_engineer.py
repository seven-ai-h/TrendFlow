import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from database.models import Keyword

PLATFORM_WEIGHTS = {
    'hackernews': 1.5,
    'reddit': 1.3,
    'github': 1.4,
    'devto': 1.1,
    'news': 1.2,
    'rss': 1.0,
}


def build_feature_matrix(session, hours_back: int = 72) -> pd.DataFrame:
    """
    Build a rich per-entity feature matrix from the Keyword table.

    Features per entity (latest snapshot):
      raw_count           – total mentions in the window
      platform_diversity  – number of distinct platforms covering it
      cross_source_score  – platform-weighted mention count
      ema_3h / ema_6h / ema_24h – exponential moving averages over hourly bins
      velocity_1h         – last-1h count / mean-of-last-24h (trending spike signal)
      acceleration        – Δ velocity (velocity_now − velocity_1h_ago)
      day_of_week         – 0=Mon … 6=Sun for time-of-week seasonality
    """
    cutoff = datetime.utcnow() - timedelta(hours=hours_back)
    rows = session.query(Keyword).filter(Keyword.timestamp >= cutoff).all()

    if not rows:
        return pd.DataFrame()

    records = [
        {
            'keyword': kw.keyword,
            'platform': kw.platform,
            'count': kw.count,
            'hour': kw.timestamp.replace(minute=0, second=0, microsecond=0),
        }
        for kw in rows
    ]
    df = pd.DataFrame(records)

    # ── Hourly aggregates ────────────────────────────────────────────────────
    hourly = (
        df.groupby(['keyword', 'hour'])['count']
        .sum()
        .reset_index()
        .sort_values(['keyword', 'hour'])
    )

    plat_diversity = (
        df.groupby(['keyword', 'hour'])['platform']
        .nunique()
        .reset_index()
        .rename(columns={'platform': 'platform_diversity'})
    )
    hourly = hourly.merge(plat_diversity, on=['keyword', 'hour'], how='left')

    df['weighted_count'] = df.apply(
        lambda r: r['count'] * PLATFORM_WEIGHTS.get(r['platform'], 1.0), axis=1
    )
    cross = (
        df.groupby(['keyword', 'hour'])['weighted_count']
        .sum()
        .reset_index()
        .rename(columns={'weighted_count': 'cross_source_score'})
    )
    hourly = hourly.merge(cross, on=['keyword', 'hour'], how='left')

    # ── EMA per keyword ──────────────────────────────────────────────────────
    ema_frames = []
    for kw, grp in hourly.groupby('keyword'):
        grp = grp.sort_values('hour').copy()
        grp['ema_3h'] = grp['count'].ewm(span=3, adjust=False).mean()
        grp['ema_6h'] = grp['count'].ewm(span=6, adjust=False).mean()
        grp['ema_24h'] = grp['count'].ewm(span=24, adjust=False).mean()
        ema_frames.append(grp)

    hourly = pd.concat(ema_frames, ignore_index=True)

    # ── Velocity and acceleration ────────────────────────────────────────────
    now_floor = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    t_1h = now_floor - timedelta(hours=1)
    t_2h = now_floor - timedelta(hours=2)
    t_24h = now_floor - timedelta(hours=24)

    def _velocity(grp, start, end_excl, baseline_start, baseline_end_excl):
        recent = grp[(grp['hour'] >= start) & (grp['hour'] < end_excl)]['count'].sum()
        baseline_vals = grp[
            (grp['hour'] >= baseline_start) & (grp['hour'] < baseline_end_excl)
        ]['count']
        baseline = baseline_vals.mean() if len(baseline_vals) > 0 else 0
        if baseline == 0:
            return float(recent) if recent > 0 else 0.0
        return recent / baseline

    vel_now = (
        hourly.groupby('keyword')
        .apply(lambda g: _velocity(g, t_1h, now_floor, t_24h, t_1h))
        .reset_index()
    )
    vel_now.columns = ['keyword', 'velocity_1h']

    vel_prev = (
        hourly.groupby('keyword')
        .apply(lambda g: _velocity(g, t_2h, t_1h, t_24h, t_2h))
        .reset_index()
    )
    vel_prev.columns = ['keyword', 'velocity_prev']

    # ── Collapse to one row per keyword (latest window) ──────────────────────
    latest = hourly.sort_values('hour').groupby('keyword').last().reset_index()
    latest = latest.merge(vel_now, on='keyword', how='left')
    latest = latest.merge(vel_prev, on='keyword', how='left')
    latest['acceleration'] = latest['velocity_1h'] - latest['velocity_prev']
    latest['day_of_week'] = datetime.utcnow().weekday()

    num_cols = ['ema_3h', 'ema_6h', 'ema_24h', 'velocity_1h', 'velocity_prev',
                'acceleration', 'platform_diversity', 'cross_source_score']
    for col in num_cols:
        if col in latest.columns:
            latest[col] = latest[col].fillna(0)

    latest.drop(columns=['velocity_prev'], inplace=True)
    return latest


FEATURE_COLS = [
    'count', 'platform_diversity', 'cross_source_score',
    'ema_3h', 'ema_6h', 'ema_24h',
    'velocity_1h', 'acceleration', 'day_of_week',
]
