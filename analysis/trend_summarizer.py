import os
from datetime import datetime

_client = None


def _get_client():
    global _client
    if _client is None:
        try:
            from anthropic import Anthropic
            _client = Anthropic()
        except Exception:
            return None
    return _client


def summarize_market_signals(live_df, leaderboard: list, sentiment_note: str = "") -> str:
    """
    Generate an analyst-style briefing on the current sentiment -> market signals.
    """
    client = _get_client()
    if client is None:
        return "Claude API unavailable. Set ANTHROPIC_API_KEY in your .env to enable AI briefings."
    if live_df is None or len(live_df) == 0:
        return "No live signals yet. Run the collector or seed_data.py first."

    signal_lines = "\n".join(
        f"- {r['name']} ({r['ticker']}): consensus {r['Consensus']:.0f}% up, "
        f"signal {r['Signal']}, avg headline sentiment {r['avg_sentiment']:+.2f}, "
        f"{int(r['buzz'])} stories"
        for _, r in live_df.iterrows()
    )
    best = leaderboard[0] if leaderboard else {}
    model_note = (f"Best model: {best.get('Model')} "
                  f"(F1 {best.get('F1', 0):.2f}, ROC-AUC {best.get('ROC-AUC', 0):.2f}, "
                  f"accuracy {best.get('Accuracy', 0):.2f})." if best else "")

    prompt = f"""You are a quantitative market analyst. Below are today's model-generated
signals from TrendFlow, which predicts next-day price direction for tech equities and
crypto from social-media sentiment + price momentum.

{model_note}

SIGNALS (consensus = average predicted probability the asset rises tomorrow):
{signal_lines}

{sentiment_note}

Write a sharp 3-paragraph briefing:
1. The strongest BUY conviction(s) and what sentiment/buzz is driving them.
2. The bearish / SELL signals and the caution flags.
3. An honest reliability caveat — these models run at ~55-65% accuracy on a noisy
   signal; state plainly how a trader should (and shouldn't) use this.

Tone: crisp, data-driven, no hype. Cite specific tickers and numbers."""

    try:
        response = client.messages.create(
            model="claude-opus-4-8",
            max_tokens=1100,
            thinking={"type": "adaptive"},
            messages=[{"role": "user", "content": prompt}],
        )
        return "\n\n".join(b.text for b in response.content if b.type == "text")
    except Exception as e:
        return f"Briefing generation failed: {e}"


def summarize_trends(feature_df, platform_breakdown: dict) -> str:
    """
    Generate a human-readable trend narrative using Claude.
    Returns the summary string, or an error message if unavailable.
    """
    client = _get_client()
    if client is None:
        return "Claude API unavailable. Set ANTHROPIC_API_KEY in your .env to enable AI summaries."

    if feature_df is None or feature_df.empty:
        return "No trend data available yet. Run the data collector first."

    top = feature_df.nlargest(12, 'velocity_1h')[
        ['keyword', 'velocity_1h', 'platform_diversity', 'cross_source_score', 'acceleration']
    ]

    entity_lines = "\n".join(
        f"- {r['keyword']}: velocity={r['velocity_1h']:.2f}x, "
        f"platforms={int(r.get('platform_diversity', 1))}, "
        f"weighted_score={r['cross_source_score']:.1f}, "
        f"acceleration={r['acceleration']:+.2f}"
        for _, r in top.iterrows()
    )

    platform_lines = "\n".join(
        f"- {platform}: {count} keyword occurrences"
        for platform, count in platform_breakdown.items()
    )

    prompt = f"""You are a senior tech industry trend analyst. Based on real-time cross-platform data from Hacker News, Reddit, GitHub Trending, Dev.to, and RSS news feeds, write a concise trend report.

DATA SNAPSHOT — {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}

Top entities ranked by velocity (how fast mentions are growing vs. 24h baseline):
{entity_lines}

Platform coverage:
{platform_lines}

Metrics guide:
- velocity: recent_1h_count / mean_24h_count (>2x = trending, >10x = viral)
- platforms: number of distinct sources covering this entity (cross-source signal strength)
- weighted_score: platform-weighted mention sum (HN/GitHub weighted higher)
- acceleration: velocity change vs 1h ago (positive = momentum building)

Write a 3–4 paragraph executive summary:
1. The dominant trend this hour and why it's surging (cite specific velocity/score numbers)
2. Two or three strong secondary trends gaining cross-platform momentum
3. Acceleration signals — topics that aren't the biggest yet but are picking up speed fastest
4. A "Watch List" of 3–5 emerging terms with early acceleration worth monitoring

Tone: analytical, sharp, data-driven. Technical audience. No filler."""

    try:
        response = client.messages.create(
            model="claude-opus-4-8",
            max_tokens=1200,
            thinking={"type": "adaptive"},
            messages=[{"role": "user", "content": prompt}],
        )
        text_blocks = [b.text for b in response.content if b.type == "text"]
        return "\n\n".join(text_blocks)
    except Exception as e:
        return f"Summary generation failed: {e}"
