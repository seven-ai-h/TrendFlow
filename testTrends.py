from database.db_setup import getSession
from analysis.trend_detector import detect_trending_keywords

session = getSession()
trends = detect_trending_keywords(session, velocity_threshold=2.0)

print(f"Found {len(trends)} trending keywords:\n")
for trend in trends[:10]:
    print(f"{trend['keyword']}: {trend['velocity']:.2f}x velocity")
    print(f"  Recent: {trend['recent_count']}, Baseline: {trend['baseline_count']}")