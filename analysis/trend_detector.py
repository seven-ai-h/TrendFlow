from datetime import datetime, timedelta
from database.models import Keyword

def detect_trending_keywords(session, velocity_threshold=2.0):
    # Get recent keywords (last hour)
    recent_time = datetime.utcnow() - timedelta(hours=1)
    recent_keywords = session.query(Keyword).filter(
        Keyword.timestamp >= recent_time
    ).all()
    
    # Get baseline keywords (same period, 7 days ago)
    baseline_start = datetime.utcnow() - timedelta(days=7, hours=1)
    baseline_end = datetime.utcnow() - timedelta(days=7)
    baseline_keywords = session.query(Keyword).filter(
        Keyword.timestamp >= baseline_start,
        Keyword.timestamp <= baseline_end
    ).all()
    
    # Convert to dictionaries {keyword: count}
    recent_dict = {kw.keyword: kw.count for kw in recent_keywords}
    baseline_dict = {kw.keyword: kw.count for kw in baseline_keywords}
    
    # Calculate velocity for each keyword
    trending = []
    for keyword, recent_count in recent_dict.items():
        baseline_count = baseline_dict.get(keyword, 0)
        
        # Avoid division by zero
        if baseline_count == 0:
            if recent_count > 0:
                velocity = float('inf')  # New keyword, infinite growth!
            else:
                velocity = 0
        else:
            velocity = (recent_count - baseline_count) / baseline_count
        
        # If velocity exceeds threshold, it's trending
        if velocity >= velocity_threshold:
            trending.append({
                'keyword': keyword,
                'velocity': velocity,
                'recent_count': recent_count,
                'baseline_count': baseline_count
            })
    
    # Sort by velocity (highest first)
    trending.sort(key=lambda x: x['velocity'], reverse=True)
    
    return trending