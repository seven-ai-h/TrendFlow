"""
Model Lab — trains several ML classifiers on the SAME trend-prediction task
and returns a side-by-side comparison (metrics, ROC curves, confusion matrices,
feature importances, and each model's live predictions lined up).

The supervised task:  given a keyword's behaviour up to day T, will its mention
count JUMP (> 1.4x) on day T+1?  This is a real, learnable time-series signal —
different model families capture it with different strengths, which is exactly
what the dashboard visualises.
"""
import time
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
)

from database.models import Keyword

warnings.filterwarnings("ignore")

PLATFORM_WEIGHTS = {
    'hackernews': 1.5, 'reddit': 1.3, 'github': 1.4,
    'devto': 1.1, 'news': 1.2, 'rss': 1.0,
}

FEATURE_COLS = [
    'count', 'prev_count', 'velocity', 'acceleration',
    'ema_3', 'ema_7', 'roll_mean_3', 'roll_std_3',
    'platform_diversity', 'cross_source_score', 'day_of_week',
]

JUMP_THRESHOLD = 1.3  # next-day count must exceed 1.3x today's to be a positive


# ── Model registry ───────────────────────────────────────────────────────────
def _model_registry():
    return {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, C=1.0, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=6, random_state=42, n_jobs=-1,
            class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.08, random_state=42),
        'SVM (RBF)': SVC(kernel='rbf', probability=True, C=2.0, gamma='scale',
                         random_state=42, class_weight='balanced'),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=7),
        'Neural Net (MLP)': MLPClassifier(
            hidden_layer_sizes=(32, 16), max_iter=800, alpha=1e-3, random_state=42),
    }


MODEL_EXPLANATIONS = {
    'Logistic Regression': {
        'family': 'Linear',
        'how': 'Fits a weighted linear boundary and squashes it through a sigmoid to output probabilities.',
        'strengths': 'Fast, interpretable coefficients, strong baseline, hard to overfit.',
        'weaknesses': 'Can only draw straight decision boundaries — misses non-linear interactions.',
        'best_for': 'A transparent baseline you can read the weights off of.',
    },
    'Random Forest': {
        'family': 'Bagging ensemble',
        'how': 'Averages hundreds of decorrelated decision trees, each grown on a bootstrap sample.',
        'strengths': 'Handles non-linearity, robust to outliers, gives feature importances, rarely overfits.',
        'weaknesses': 'Larger memory footprint, less interpretable than a single tree.',
        'best_for': 'A strong, low-maintenance default on tabular data.',
    },
    'Gradient Boosting': {
        'family': 'Boosting ensemble',
        'how': 'Builds trees sequentially, each one correcting the residual errors of the last.',
        'strengths': 'Often the top performer on tabular data; captures subtle interactions.',
        'weaknesses': 'Sensitive to hyper-parameters, slower to train, can overfit noisy data.',
        'best_for': 'Squeezing out maximum accuracy when you can tune it.',
    },
    'SVM (RBF)': {
        'family': 'Kernel method',
        'how': 'Projects data into a high-dimensional space and finds the maximum-margin separator.',
        'strengths': 'Effective in high dimensions, flexible non-linear boundaries via the RBF kernel.',
        'weaknesses': 'Scales poorly to large datasets, needs feature scaling, slower probability estimates.',
        'best_for': 'Clean, medium-sized datasets with complex boundaries.',
    },
    'K-Nearest Neighbors': {
        'family': 'Instance-based',
        'how': 'Classifies a point by majority vote of its k closest neighbours in feature space.',
        'strengths': 'No training phase, naturally non-linear, easy to reason about.',
        'weaknesses': 'Slow at prediction time, sensitive to scaling and irrelevant features.',
        'best_for': 'A sanity-check baseline and locally-structured data.',
    },
    'Neural Net (MLP)': {
        'family': 'Neural network',
        'how': 'Stacked layers of weighted sums + non-linear activations, trained by backprop.',
        'strengths': 'Universal approximator — learns arbitrary non-linear functions given enough data.',
        'weaknesses': 'Data-hungry, opaque, many knobs, can overfit small tabular sets.',
        'best_for': 'Large datasets with rich non-linear structure.',
    },
}


# ── Dataset construction ─────────────────────────────────────────────────────
def build_training_dataset(session, days_back: int = 21) -> pd.DataFrame:
    """
    Collapse the Keyword table into a per-(keyword, day) time series and engineer
    forward-looking features + a next-day 'jump' label.
    """
    cutoff = datetime.utcnow() - timedelta(days=days_back)
    rows = session.query(Keyword).filter(Keyword.timestamp >= cutoff).all()
    if not rows:
        return pd.DataFrame()

    recs = [{
        'keyword': r.keyword,
        'platform': r.platform,
        'count': r.count,
        'date': r.timestamp.date(),
        'weighted': r.count * PLATFORM_WEIGHTS.get(r.platform, 1.0),
    } for r in rows]
    df = pd.DataFrame(recs)

    daily = df.groupby(['keyword', 'date']).agg(
        count=('count', 'sum'),
        platform_diversity=('platform', 'nunique'),
        cross_source_score=('weighted', 'sum'),
    ).reset_index()

    samples = []
    for kw, grp in daily.groupby('keyword'):
        grp = grp.sort_values('date').reset_index(drop=True)
        if len(grp) < 4:
            continue
        grp['ema_3'] = grp['count'].ewm(span=3, adjust=False).mean()
        grp['ema_7'] = grp['count'].ewm(span=7, adjust=False).mean()
        grp['roll_mean_3'] = grp['count'].rolling(3, min_periods=1).mean()
        grp['roll_std_3'] = grp['count'].rolling(3, min_periods=1).std().fillna(0)
        grp['prev_count'] = grp['count'].shift(1).fillna(0)
        grp['velocity'] = np.where(
            grp['prev_count'] > 0,
            (grp['count'] - grp['prev_count']) / grp['prev_count'], 0.0)
        grp['prev_velocity'] = grp['velocity'].shift(1).fillna(0)
        grp['acceleration'] = grp['velocity'] - grp['prev_velocity']
        grp['next_count'] = grp['count'].shift(-1)

        for _, row in grp.iterrows():
            if pd.isna(row['next_count']):
                continue
            label = 1 if row['next_count'] > row['count'] * JUMP_THRESHOLD else 0
            samples.append({
                'keyword': kw,
                'count': row['count'],
                'prev_count': row['prev_count'],
                'velocity': row['velocity'],
                'acceleration': row['acceleration'],
                'ema_3': row['ema_3'],
                'ema_7': row['ema_7'],
                'roll_mean_3': row['roll_mean_3'],
                'roll_std_3': row['roll_std_3'],
                'platform_diversity': row['platform_diversity'],
                'cross_source_score': row['cross_source_score'],
                'day_of_week': pd.Timestamp(row['date']).dayofweek,
                'will_jump': label,
            })

    return pd.DataFrame(samples)


def build_live_snapshot(session, days_back: int = 10) -> pd.DataFrame:
    """One feature row per keyword at its most recent day — for live predictions."""
    ds = build_training_dataset(session, days_back=days_back)
    if ds.empty:
        return ds
    # keep the last sample per keyword (most recent day with a full feature set)
    return ds.groupby('keyword').tail(1).reset_index(drop=True)


# ── Training + comparison ────────────────────────────────────────────────────
def train_all_models(session, days_back: int = 21) -> dict:
    """
    Train every model on the same split and return a rich comparison payload.
    Returns dict with keys: ok, error, n_samples, class_balance, leaderboard,
    roc, confusion, feature_importance, feature_cols, models, scaler, live.
    """
    ds = build_training_dataset(session, days_back=days_back)
    if ds.empty or len(ds) < 25:
        return {'ok': False,
                'error': f"Need ~25+ training rows, have {0 if ds.empty else len(ds)}. "
                         f"Seed or collect more data."}

    if ds['will_jump'].nunique() < 2:
        return {'ok': False,
                'error': "Only one class present in labels — not enough variation to train. "
                         "Re-seed with more varied trajectories."}

    X = ds[FEATURE_COLS].fillna(0).values
    y = ds['will_jump'].values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    strat = y if np.bincount(y).min() >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        Xs, y, test_size=0.25, random_state=42, stratify=strat)

    leaderboard, roc_data, conf_data, importances = [], {}, {}, {}
    fitted = {}

    for name, model in _model_registry().items():
        t0 = time.time()
        model.fit(X_tr, y_tr)
        train_time = time.time() - t0

        y_pred = model.predict(X_te)
        try:
            y_proba = model.predict_proba(X_te)[:, 1]
        except Exception:
            y_proba = y_pred.astype(float)

        # Cross-validated accuracy for stability
        n_splits = min(5, max(2, np.bincount(y).min()))
        try:
            cv = cross_val_score(model, Xs, y, cv=n_splits, scoring='accuracy')
            cv_mean, cv_std = float(cv.mean()), float(cv.std())
        except Exception:
            cv_mean, cv_std = float('nan'), float('nan')

        try:
            auc = roc_auc_score(y_te, y_proba)
        except Exception:
            auc = float('nan')

        leaderboard.append({
            'Model': name,
            'Accuracy': accuracy_score(y_te, y_pred),
            'Precision': precision_score(y_te, y_pred, zero_division=0),
            'Recall': recall_score(y_te, y_pred, zero_division=0),
            'F1': f1_score(y_te, y_pred, zero_division=0),
            'ROC-AUC': auc,
            'CV Acc': cv_mean,
            'CV Std': cv_std,
            'Train (ms)': train_time * 1000,
        })

        try:
            fpr, tpr, _ = roc_curve(y_te, y_proba)
            roc_data[name] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'auc': auc}
        except Exception:
            pass

        conf_data[name] = confusion_matrix(y_te, y_pred, labels=[0, 1]).tolist()

        if hasattr(model, 'feature_importances_'):
            importances[name] = dict(zip(FEATURE_COLS, model.feature_importances_))
        elif hasattr(model, 'coef_'):
            importances[name] = dict(zip(FEATURE_COLS, np.abs(model.coef_[0])))

        fitted[name] = model

    leaderboard.sort(key=lambda r: (r['F1'], r['ROC-AUC'] if not np.isnan(r['ROC-AUC']) else 0),
                     reverse=True)

    # Live predictions — every model's verdict on current keywords, lined up
    live = _live_predictions(session, fitted, scaler, days_back=10)

    return {
        'ok': True,
        'error': None,
        'n_samples': len(ds),
        'class_balance': {'no_jump': int((y == 0).sum()), 'jump': int((y == 1).sum())},
        'leaderboard': leaderboard,
        'roc': roc_data,
        'confusion': conf_data,
        'feature_importance': importances,
        'feature_cols': FEATURE_COLS,
        'models': fitted,
        'scaler': scaler,
        'live': live,
    }


def _live_predictions(session, fitted: dict, scaler, days_back: int = 10) -> pd.DataFrame:
    snap = build_live_snapshot(session, days_back=days_back)
    if snap.empty:
        return pd.DataFrame()

    Xs = scaler.transform(snap[FEATURE_COLS].fillna(0).values)
    out = snap[['keyword', 'count', 'velocity', 'cross_source_score']].copy()

    prob_cols = []
    for name, model in fitted.items():
        try:
            proba = model.predict_proba(Xs)[:, 1]
        except Exception:
            proba = model.predict(Xs).astype(float)
        col = f"{name}"
        out[col] = (proba * 100).round(1)
        prob_cols.append(col)

    out['Consensus'] = out[prob_cols].mean(axis=1).round(1)
    out['Agreement'] = out[prob_cols].apply(
        lambda r: f"{int((r > 50).sum())}/{len(prob_cols)}", axis=1)
    out = out.sort_values('Consensus', ascending=False).reset_index(drop=True)
    return out


if __name__ == "__main__":
    from database.db_setup import getSession
    result = train_all_models(getSession())
    if not result['ok']:
        print("ERROR:", result['error'])
    else:
        print(f"Trained on {result['n_samples']} samples, balance={result['class_balance']}\n")
        lb = pd.DataFrame(result['leaderboard'])
        print(lb[['Model', 'Accuracy', 'F1', 'ROC-AUC', 'CV Acc', 'Train (ms)']].to_string(index=False))
        print("\nLive predictions (top 5):")
        print(result['live'].head().to_string(index=False))
