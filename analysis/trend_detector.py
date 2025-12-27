from datetime import datetime, timedelta
from keyword_extractor import extract_keywords
from database.models import Keyword

def detect_trending_keywords(session, velocity_threshold=2.0):
    baseline_start = datetime.utcnow() - timedelta(days=7, hours=1)
    baseline_end = datetime.utcnow() - timedelta(days=7)
    baseline_keywords = session.query(Keyword).filter(
        Keyword.timestamp >= baseline_start,
        Keyword.timestamp <= baseline_end
    ).all()
    baseline_count = 
    velocity = (recent_count - baseline_count) / baseline_count
    
