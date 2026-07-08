"""
Model Lab — the sentiment -> market comparison.

Trains six classifiers, organised into THREE algorithm families, on one shared
task: *given today's social sentiment/buzz for a ticker plus its price context,
will the asset close UP tomorrow?*  Every model sees the identical feature matrix
and test split, so the leaderboard is a fair fight. Grouping by family keeps the
dashboard tidy (three tidy columns instead of one long list).
"""
import time
import warnings
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
)

from analysis.market_features import (
    build_market_dataset, build_live_snapshot, FEATURE_COLS, TICKER_NAMES,
)

warnings.filterwarnings("ignore")


# ── Three model families (validated data-viz dark palette accents) ───────────
def _model_categories():
    return {
        'Linear & Probabilistic': {
            'color': '#3987e5',
            'blurb': 'Fast, transparent baselines that draw straight or '
                     'probabilistic decision boundaries.',
            'models': {
                'Logistic Regression': LogisticRegression(
                    max_iter=1000, C=1.0, class_weight='balanced'),
                'Naive Bayes': GaussianNB(),
            },
        },
        'Tree Ensembles': {
            'color': '#199e70',
            'blurb': 'Combine many decision trees to capture non-linear '
                     'feature interactions — strong tabular defaults.',
            'models': {
                'Random Forest': RandomForestClassifier(
                    n_estimators=200, max_depth=6, random_state=42,
                    n_jobs=-1, class_weight='balanced'),
                'Gradient Boosting': GradientBoostingClassifier(
                    n_estimators=150, max_depth=3, learning_rate=0.08,
                    random_state=42),
            },
        },
        'Kernel & Neural': {
            'color': '#9085e9',
            'blurb': 'Flexible non-linear learners that bend the decision '
                     'boundary through kernels or hidden layers.',
            'models': {
                'SVM (RBF)': SVC(kernel='rbf', probability=True, C=2.0,
                                 gamma='scale', random_state=42,
                                 class_weight='balanced'),
                'Neural Net (MLP)': MLPClassifier(
                    hidden_layer_sizes=(32, 16), max_iter=800,
                    alpha=1e-3, random_state=42),
            },
        },
    }


# per-model accent colours (2 per family, drawn from the family hue neighbourhood)
MODEL_COLORS = {
    'Logistic Regression': '#3987e5',
    'Naive Bayes': '#6da7ec',
    'Random Forest': '#199e70',
    'Gradient Boosting': '#1bb37f',
    'SVM (RBF)': '#9085e9',
    'Neural Net (MLP)': '#c3b6f5',
}

MODEL_CATEGORY = {
    'Logistic Regression': 'Linear & Probabilistic',
    'Naive Bayes': 'Linear & Probabilistic',
    'Random Forest': 'Tree Ensembles',
    'Gradient Boosting': 'Tree Ensembles',
    'SVM (RBF)': 'Kernel & Neural',
    'Neural Net (MLP)': 'Kernel & Neural',
}

MODEL_EXPLANATIONS = {
    'Logistic Regression': {
        'how': 'Fits a weighted linear boundary and squashes it through a sigmoid to output probabilities.',
        'strengths': 'Fast, interpretable coefficients, strong baseline, hard to overfit.',
        'weaknesses': 'Only straight decision boundaries — misses non-linear interactions.',
        'best_for': 'A transparent baseline whose weights you can read directly.',
    },
    'Naive Bayes': {
        'how': 'Applies Bayes’ theorem assuming features are conditionally independent given the class.',
        'strengths': 'Extremely fast, needs little data, surprisingly strong on noisy signals.',
        'weaknesses': 'The independence assumption is usually false; probabilities can be poorly calibrated.',
        'best_for': 'A quick probabilistic yardstick and high-dimensional sparse data.',
    },
    'Random Forest': {
        'how': 'Averages hundreds of decorrelated decision trees grown on bootstrap samples.',
        'strengths': 'Handles non-linearity, robust to outliers, exposes feature importances.',
        'weaknesses': 'Larger memory footprint, less interpretable than a single tree.',
        'best_for': 'A strong, low-maintenance default on tabular data.',
    },
    'Gradient Boosting': {
        'how': 'Builds trees sequentially, each correcting the residual errors of the last.',
        'strengths': 'Often the top performer on tabular data; captures subtle interactions.',
        'weaknesses': 'Sensitive to hyper-parameters, slower to train, can overfit noise.',
        'best_for': 'Squeezing out maximum accuracy when you can tune it.',
    },
    'SVM (RBF)': {
        'how': 'Projects data into a high-dimensional space and finds the maximum-margin separator.',
        'strengths': 'Effective in high dimensions, flexible non-linear boundaries via the RBF kernel.',
        'weaknesses': 'Scales poorly to large data, needs feature scaling, slower probabilities.',
        'best_for': 'Clean, medium-sized datasets with complex boundaries.',
    },
    'Neural Net (MLP)': {
        'how': 'Stacked layers of weighted sums + non-linear activations, trained by backprop.',
        'strengths': 'Universal approximator — learns arbitrary non-linear functions given data.',
        'weaknesses': 'Data-hungry, opaque, many knobs, can overfit small tabular sets.',
        'best_for': 'Large datasets with rich non-linear structure.',
    },
}


def train_all_models(session, days_back: int = 60) -> dict:
    ds = build_market_dataset(session, days_back=days_back)
    if ds.empty or len(ds) < 40:
        return {'ok': False,
                'error': f"Need ~40+ training rows, have {0 if ds.empty else len(ds)}. "
                         f"Run seed_data.py (demo) or test_hn_api.py (live)."}
    if ds['will_rise'].nunique() < 2:
        return {'ok': False, 'error': "Only one class present — not learnable."}

    X = ds[FEATURE_COLS].fillna(0).values
    y = ds['will_rise'].values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    strat = y if np.bincount(y).min() >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        Xs, y, test_size=0.25, random_state=42, stratify=strat)

    categories = _model_categories()
    leaderboard, roc_data, conf_data, importances, fitted = [], {}, {}, {}, {}

    for cat_name, cat in categories.items():
        for name, model in cat['models'].items():
            t0 = time.time()
            model.fit(X_tr, y_tr)
            train_ms = (time.time() - t0) * 1000

            y_pred = model.predict(X_te)
            try:
                y_proba = model.predict_proba(X_te)[:, 1]
            except Exception:
                y_proba = y_pred.astype(float)

            n_splits = min(5, max(2, int(np.bincount(y).min())))
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
                'Model': name, 'Category': cat_name,
                'Accuracy': accuracy_score(y_te, y_pred),
                'Precision': precision_score(y_te, y_pred, zero_division=0),
                'Recall': recall_score(y_te, y_pred, zero_division=0),
                'F1': f1_score(y_te, y_pred, zero_division=0),
                'ROC-AUC': auc, 'CV Acc': cv_mean, 'CV Std': cv_std,
                'Train (ms)': train_ms,
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

    leaderboard.sort(
        key=lambda r: (r['F1'], r['ROC-AUC'] if r['ROC-AUC'] == r['ROC-AUC'] else 0),
        reverse=True)

    live = _live_predictions(session, fitted, scaler, days_back=min(days_back, 45))

    return {
        'ok': True, 'error': None,
        'n_samples': len(ds),
        'n_tickers': ds['ticker'].nunique(),
        'class_balance': {'down': int((y == 0).sum()), 'up': int((y == 1).sum())},
        'leaderboard': leaderboard,
        'categories': {k: {'color': v['color'], 'blurb': v['blurb'],
                           'models': list(v['models'].keys())}
                       for k, v in categories.items()},
        'roc': roc_data, 'confusion': conf_data,
        'feature_importance': importances, 'feature_cols': FEATURE_COLS,
        'models': fitted, 'scaler': scaler, 'live': live,
    }


def _live_predictions(session, fitted, scaler, days_back=45) -> pd.DataFrame:
    snap = build_live_snapshot(session, days_back=days_back)
    if snap.empty:
        return pd.DataFrame()
    Xs = scaler.transform(snap[FEATURE_COLS].fillna(0).values)
    out = snap[['ticker', 'name', 'avg_sentiment', 'buzz']].copy()
    prob_cols = []
    for name, model in fitted.items():
        try:
            proba = model.predict_proba(Xs)[:, 1]
        except Exception:
            proba = model.predict(Xs).astype(float)
        out[name] = (proba * 100).round(1)
        prob_cols.append(name)
    out['Consensus'] = out[prob_cols].mean(axis=1).round(1)
    out['Signal'] = out['Consensus'].apply(
        lambda v: 'BUY' if v >= 55 else ('SELL' if v <= 45 else 'HOLD'))
    out = out.sort_values('Consensus', ascending=False).reset_index(drop=True)
    return out


if __name__ == "__main__":
    from database.db_setup import getSession
    r = train_all_models(getSession())
    if not r['ok']:
        print("ERROR:", r['error'])
    else:
        print(f"Trained on {r['n_samples']} samples, {r['n_tickers']} tickers, "
              f"balance={r['class_balance']}\n")
        lb = pd.DataFrame(r['leaderboard'])
        print(lb[['Model', 'Category', 'Accuracy', 'F1', 'ROC-AUC', 'CV Acc']].to_string(index=False))
        print("\nLive signals:")
        print(r['live'][['ticker', 'name', 'avg_sentiment', 'Consensus', 'Signal']].to_string(index=False))
