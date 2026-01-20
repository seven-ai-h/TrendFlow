import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from database.db_setup import getSession
from database.models import Keyword
from collections import defaultdict

def prepare_training_data(session, days_back=14):
    """
    Prepare historical data for training
    Returns features and labels
    """
    cutoff = datetime.utcnow() - timedelta(days=days_back)
    keywords = session.query(Keyword).filter(Keyword.timestamp >= cutoff).all()
    
    # Group by keyword and day
    keyword_daily = defaultdict(lambda: defaultdict(int))
    
    for kw in keywords:
        day = kw.timestamp.date()
        keyword_daily[kw.keyword][day] += kw.count
    
    # Create features for each keyword
    training_data = []
    
    for keyword, daily_counts in keyword_daily.items():
        sorted_days = sorted(daily_counts.keys())
        
        if len(sorted_days) < 3:  # Need at least 3 days of data
            continue
        
        # For each day (except last 2), create features
        for i in range(len(sorted_days) - 2):
            current_day = sorted_days[i]
            next_day = sorted_days[i + 1]
            day_after = sorted_days[i + 2] if i + 2 < len(sorted_days) else None
            
            # Features: counts from current and previous days
            current_count = daily_counts[current_day]
            prev_count = daily_counts.get(sorted_days[i-1], 0) if i > 0 else 0
            
            # Calculate velocity
            velocity = (current_count - prev_count) / prev_count if prev_count > 0 else 0
            
            # Label: will it trend tomorrow? (increase by 50%+)
            future_count = daily_counts.get(next_day, 0)
            will_trend = 1 if future_count > current_count * 1.5 else 0
            
            training_data.append({
                'keyword': keyword,
                'current_count': current_count,
                'prev_count': prev_count,
                'velocity': velocity,
                'day_of_week': current_day.weekday(),
                'will_trend': will_trend
            })
    
    return pd.DataFrame(training_data)

def train_prediction_model(session):
    """
    Train a Random Forest model to predict trending keywords
    """
    df = prepare_training_data(session)
    
    if len(df) < 10:
        return None, "Not enough historical data. Need at least 10 data points."
    
    # Features
    X = df[['current_count', 'prev_count', 'velocity', 'day_of_week']]
    y = df['will_trend']
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Accuracy
    accuracy = model.score(X_test, y_test)
    
    return model, accuracy

def predict_trending_keywords(session, model, top_n=10):
    """
    Predict which keywords will trend in next 24-48 hours
    """
    # Get recent keywords (last 3 days)
    cutoff = datetime.utcnow() - timedelta(days=3)
    recent_keywords = session.query(Keyword).filter(Keyword.timestamp >= cutoff).all()
    
    # Group by keyword
    keyword_data = defaultdict(lambda: {'counts': [], 'dates': []})
    
    for kw in recent_keywords:
        keyword_data[kw.keyword]['counts'].append(kw.count)
        keyword_data[kw.keyword]['dates'].append(kw.timestamp.date())
    
    # Prepare features for prediction
    predictions = []
    
    for keyword, data in keyword_data.items():
        if len(data['counts']) < 2:
            continue
        
        current_count = data['counts'][-1]
        prev_count = data['counts'][-2] if len(data['counts']) > 1 else 0
        velocity = (current_count - prev_count) / prev_count if prev_count > 0 else 0
        day_of_week = datetime.utcnow().weekday()
        
        features = [[current_count, prev_count, velocity, day_of_week]]
        
        # Predict
        will_trend = model.predict(features)[0]
        confidence = model.predict_proba(features)[0][1]  # Probability of trending
        
        if will_trend == 1:
            predictions.append({
                'keyword': keyword,
                'confidence': confidence * 100,
                'current_count': current_count,
                'velocity': velocity
            })
    
    # Sort by confidence
    predictions.sort(key=lambda x: x['confidence'], reverse=True)
    
    return predictions[:top_n]

# For testing
if __name__ == "__main__":
    session = getSession()
    
    print("Training prediction model...")
    model, result = train_prediction_model(session)
    
    if model:
        print(f"Model trained! Accuracy: {result:.2%}")
        
        print("\nPredicting trending keywords...")
        predictions = predict_trending_keywords(session, model)
        
        print("\nKeywords predicted to trend in next 24-48 hours:")
        for pred in predictions:
            print(f"• {pred['keyword']}: {pred['confidence']:.1f}% confidence")
    else:
        print(f"Error: {result}")