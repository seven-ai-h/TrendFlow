"""
Headline sentiment scoring — the complementary signal that makes the
social-buzz data actually predictive of market moves.

Uses VADER (vaderSentiment package) which ships its own lexicon, so it works
fully offline — no model download, no network. VADER is tuned for short,
informal, headline-style text, which is exactly what we collect.
"""

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _analyzer = SentimentIntensityAnalyzer()
    _AVAILABLE = True
except Exception:  # pragma: no cover - fallback if package missing
    _analyzer = None
    _AVAILABLE = False

# Domain lexicon boosts — finance/tech words VADER doesn't weight well.
_BOOST = {
    'surge': 2.5, 'soar': 3.0, 'rally': 2.0, 'breakthrough': 2.5, 'record': 1.5,
    'beats': 2.0, 'outperform': 2.0, 'bullish': 3.0, 'boom': 2.5, 'milestone': 1.5,
    'launch': 1.0, 'growth': 1.5, 'profit': 2.0, 'upgrade': 1.5,
    'crash': -3.0, 'plunge': -3.0, 'slump': -2.5, 'bearish': -3.0, 'layoffs': -2.5,
    'lawsuit': -2.0, 'ban': -2.0, 'recall': -2.0, 'breach': -2.5, 'outage': -2.0,
    'downgrade': -2.0, 'miss': -1.5, 'fraud': -3.0, 'bankruptcy': -3.0, 'delay': -1.0,
}


def score_sentiment(text: str) -> float:
    """
    Return a compound sentiment score in [-1, 1].
    Positive = optimistic/bullish headline, negative = pessimistic/bearish.
    """
    if not text:
        return 0.0

    if _AVAILABLE:
        base = _analyzer.polarity_scores(text)['compound']
    else:
        base = 0.0

    # Blend in the domain lexicon so finance-specific tone is captured.
    lower = text.lower()
    boost = sum(w for term, w in _BOOST.items() if term in lower)
    boost = max(-5.0, min(5.0, boost)) / 5.0  # normalise to [-1, 1]

    combined = 0.7 * base + 0.3 * boost
    return max(-1.0, min(1.0, combined))


def sentiment_label(score: float) -> str:
    if score >= 0.35:
        return 'bullish'
    if score <= -0.35:
        return 'bearish'
    return 'neutral'


def is_available() -> bool:
    return _AVAILABLE
