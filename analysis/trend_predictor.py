import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

from database.models import Keyword
from analysis.feature_engineer import build_feature_matrix, FEATURE_COLS

MODEL_PATH = "trendflow_model.joblib"
SCALER_PATH = "trendflow_scaler.joblib"


def _label_feature_matrix(df: pd.DataFrame, velocity_thresh: float = 2.0) -> pd.DataFrame:
    """
    Assign training labels: 'will_trend' = 1 when velocity > threshold
    and acceleration is positive (momentum building).
    """
    df = df.copy()
    df['will_trend'] = (
        (df['velocity_1h'] >= velocity_thresh) & (df['acceleration'] >= 0)
    ).astype(int)
    return df


def train_prediction_model(session):
    """
    Train a Gradient Boosting classifier on the rich feature matrix.
    Returns (model, cv_accuracy) or (None, error_message).
    """
    df = build_feature_matrix(session, hours_back=14 * 24)

    if df.empty or len(df) < 15:
        return None, "Not enough data. Need at least 15 keyword records (run the collector a few times)."

    df = _label_feature_matrix(df)

    # Drop rows missing any feature
    valid = df.dropna(subset=FEATURE_COLS + ['will_trend'])
    if len(valid) < 10:
        return None, f"Only {len(valid)} complete rows after dropping NaN — need 10+."

    X = valid[FEATURE_COLS].values
    y = valid['will_trend'].values

    # Normalise features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )

    # Cross-validated accuracy (3-fold to tolerate small datasets)
    n_splits = min(3, max(2, len(valid) // 5))
    cv_scores = cross_val_score(model, X_scaled, y, cv=n_splits, scoring='accuracy')
    cv_accuracy = cv_scores.mean()

    model.fit(X_scaled, y)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    return model, cv_accuracy


def load_or_train_model(session):
    """Load a cached model if <6 hours old, otherwise retrain."""
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        mtime = datetime.fromtimestamp(os.path.getmtime(MODEL_PATH))
        if datetime.now() - mtime < timedelta(hours=6):
            model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            return model, scaler, None  # None = cached (no new accuracy)
    model, result = train_prediction_model(session)
    if model is None:
        return None, None, result
    scaler = joblib.load(SCALER_PATH)
    return model, scaler, result


def predict_trending_keywords(session, model, scaler, top_n: int = 10) -> list:
    """
    Run the trained model on the current feature snapshot.
    Returns a ranked list of dicts with keyword, confidence, and key features.
    """
    df = build_feature_matrix(session, hours_back=48)
    if df.empty:
        return []

    valid = df.dropna(subset=FEATURE_COLS)
    if valid.empty:
        return []

    X = valid[FEATURE_COLS].values
    X_scaled = scaler.transform(X)

    probas = model.predict_proba(X_scaled)
    # Column 1 = P(will_trend=1)
    valid = valid.copy()
    valid['confidence'] = probas[:, 1] * 100

    results = []
    for _, row in valid.iterrows():
        results.append({
            'keyword': row['keyword'],
            'confidence': row['confidence'],
            'velocity_1h': row['velocity_1h'],
            'acceleration': row['acceleration'],
            'platform_diversity': int(row.get('platform_diversity', 1)),
            'cross_source_score': row.get('cross_source_score', 0),
            'current_count': row.get('count', 0),
        })

    results.sort(key=lambda x: x['confidence'], reverse=True)
    return results[:top_n]


def get_feature_importances(model) -> pd.DataFrame:
    """Return a DataFrame of feature name → importance for visualization."""
    importances = model.feature_importances_
    return (
        pd.DataFrame({'feature': FEATURE_COLS, 'importance': importances})
        .sort_values('importance', ascending=False)
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    from database.db_setup import getSession
    session = getSession()
    print("Training model…")
    m, s, result = load_or_train_model(session)
    if m:
        acc = f"{result:.1%}" if isinstance(result, float) else "loaded from cache"
        print(f"Model ready — CV accuracy: {acc}")
        preds = predict_trending_keywords(session, m, s)
        print("\nTop predicted trending keywords:")
        for p in preds:
            print(f"  {p['keyword']}: {p['confidence']:.1f}% confidence "
                  f"(vel={p['velocity_1h']:.2f}x, accel={p['acceleration']:+.2f})")
    else:
        print(f"Could not train: {result}")
