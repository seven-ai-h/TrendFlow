"""
Model Lab — regression edition.

One task: predict each asset's NEXT-DAY RETURN (%) from its social sentiment,
buzz and price context. Three models, each a clear step up in sophistication:

  1. Linear Regression    — the transparent baseline (a weighted sum of features)
  2. Random Forest        — non-linear ensemble of decision trees
  3. LSTM (PyTorch)       — a recurrent net that reads the SEQUENCE of recent days

We judge them on honest, domain-meaningful measures instead of a wall of charts:
  * MAE / RMSE         — how far off the predicted return is, on average
  * Directional acc.   — how often we get the up/down direction right
  * d-prime (d')       — signal-detection separation of up-days from down-days
  * Strategy Index     — grow $100 by following the model vs. buy-and-hold
"""
import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from analysis.market_features import build_market_dataset, FEATURE_COLS, TICKER_NAMES

warnings.filterwarnings("ignore")

SEQ_LEN = 5          # LSTM looks back one trading week
MODEL_COLORS = {
    "Linear Regression": "#3987e5",   # blue  — baseline
    "Random Forest": "#199e70",       # green — ensemble
    "LSTM": "#9085e9",                # violet — deep learning
}

MODEL_EXPLAINERS = {
    "Linear Regression": {
        "tag": "Baseline · linear",
        "how": "Fits one weight per feature and sums them. The simplest honest yardstick.",
        "read": "If the fancy models can't beat this, the signal is mostly linear (or mostly noise).",
    },
    "Random Forest": {
        "tag": "Ensemble · non-linear",
        "how": "Averages hundreds of decision trees, each seeing a random slice of the data.",
        "read": "Captures non-linear interactions the linear model can't — but can overfit noise.",
    },
    "LSTM": {
        "tag": "Deep learning · sequential",
        "how": "A recurrent neural net that reads the last 5 days in order, keeping a memory of momentum.",
        "read": "The only model that sees sentiment *building over time* rather than one snapshot.",
    },
}


# ── d-prime (signal detection) ───────────────────────────────────────────────
def _d_prime(y_true, y_pred) -> float:
    """
    Treat 'predict a positive return' as a detector for 'day actually rose'.
    d' = z(hit rate) - z(false-alarm rate). Higher = cleaner separation.
    """
    actual_up = np.array(y_true) > 0
    pred_up = np.array(y_pred) > 0
    if actual_up.sum() == 0 or (~actual_up).sum() == 0:
        return float("nan")
    hit = pred_up[actual_up].mean()
    fa = pred_up[~actual_up].mean()
    hit = min(max(hit, 0.01), 0.99)
    fa = min(max(fa, 0.01), 0.99)
    return float(norm.ppf(hit) - norm.ppf(fa))


def _directional_accuracy(y_true, y_pred) -> float:
    return float((np.sign(y_true) == np.sign(y_pred)).mean())


# ── Build aligned tabular + sequence samples ─────────────────────────────────
def _build_samples(session, days_back):
    ds = build_market_dataset(session, days_back=days_back)
    if ds.empty:
        return None
    ds = ds.sort_values(["ticker", "date"]).reset_index(drop=True)

    X_tab, X_seq, y, dates, tickers = [], [], [], [], []
    for ticker, grp in ds.groupby("ticker"):
        grp = grp.sort_values("date").reset_index(drop=True)
        feats = grp[FEATURE_COLS].fillna(0).values
        targets = grp["next_return"].values
        dts = grp["date"].values
        for i in range(SEQ_LEN - 1, len(grp)):
            if np.isnan(targets[i]):
                continue
            X_tab.append(feats[i])
            X_seq.append(feats[i - SEQ_LEN + 1: i + 1])
            y.append(targets[i])
            dates.append(dts[i])
            tickers.append(ticker)
    if len(y) < 40:
        return None
    return (np.array(X_tab), np.array(X_seq), np.array(y),
            np.array(dates), np.array(tickers))


# ── LSTM (PyTorch) ───────────────────────────────────────────────────────────
def _train_lstm(X_seq_tr, y_tr, X_seq_te, input_size):
    try:
        import torch
        import torch.nn as nn
    except Exception:
        return None  # torch unavailable -> caller skips LSTM

    torch.manual_seed(42)
    np.random.seed(42)

    class LSTMReg(nn.Module):
        def __init__(self, n_feat, hidden=48):
            super().__init__()
            self.lstm = nn.LSTM(n_feat, hidden, num_layers=1, batch_first=True)
            self.drop = nn.Dropout(0.2)
            self.head = nn.Sequential(nn.Linear(hidden, 24), nn.ReLU(), nn.Linear(24, 1))

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(self.drop(out[:, -1, :])).squeeze(-1)

    # standardise the target so MSE training is well-conditioned, then invert
    y_mean, y_std = float(y_tr.mean()), float(y_tr.std() + 1e-8)
    y_tr_n = (y_tr - y_mean) / y_std

    model = LSTMReg(input_size)
    opt = torch.optim.Adam(model.parameters(), lr=0.008, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=80, gamma=0.5)
    lossf = nn.SmoothL1Loss()  # robust to outlier returns

    Xtr = torch.tensor(X_seq_tr, dtype=torch.float32)
    ytr = torch.tensor(y_tr_n, dtype=torch.float32)

    model.train()
    for _ in range(200):
        opt.zero_grad()
        loss = lossf(model(Xtr), ytr)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

    model.eval()
    with torch.no_grad():
        pred_n = model(torch.tensor(X_seq_te, dtype=torch.float32)).numpy()
    return pred_n * y_std + y_mean  # back to return units


# ── Main entry ───────────────────────────────────────────────────────────────
def train_all_models(session, days_back: int = 60) -> dict:
    samples = _build_samples(session, days_back)
    if samples is None:
        return {"ok": False, "error": "Not enough aligned history. Seed or collect more data."}

    X_tab, X_seq, y, dates, tickers = samples

    # temporal split — train on the earlier period, test on the later (no leakage)
    order = np.argsort(dates)
    X_tab, X_seq, y, dates, tickers = (X_tab[order], X_seq[order], y[order],
                                       dates[order], tickers[order])
    split = int(len(y) * 0.75)
    tr, te = slice(0, split), slice(split, None)

    scaler = StandardScaler().fit(X_tab[tr])
    Xtab_tr, Xtab_te = scaler.transform(X_tab[tr]), scaler.transform(X_tab[te])
    n_feat = X_seq.shape[2]
    flat = X_seq.reshape(-1, n_feat)
    seq_scaled = scaler.transform(flat).reshape(X_seq.shape)
    Xseq_tr, Xseq_te = seq_scaled[tr], seq_scaled[te]
    y_tr, y_te = y[tr], y[te]
    dates_te, tickers_te = dates[te], tickers[te]

    preds = {}

    lr = LinearRegression().fit(Xtab_tr, y_tr)
    preds["Linear Regression"] = lr.predict(Xtab_te)

    rf = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
    rf.fit(Xtab_tr, y_tr)
    preds["Random Forest"] = rf.predict(Xtab_te)

    lstm_pred = _train_lstm(Xseq_tr, y_tr, Xseq_te, n_feat)
    torch_ok = lstm_pred is not None
    if torch_ok:
        preds["LSTM"] = lstm_pred

    # ── Metrics (global — computed on the full test set) ──────────────────────
    leaderboard = []
    for name, yp in preds.items():
        leaderboard.append({
            "Model": name,
            "MAE": mean_absolute_error(y_te, yp),
            "RMSE": float(np.sqrt(mean_squared_error(y_te, yp))),
            "R2": r2_score(y_te, yp),
            "Dir. Acc": _directional_accuracy(y_te, yp),
            "d-prime": _d_prime(y_te, yp),
        })

    # ── Per-ticker test records (so the dashboard can filter by asset) ────────
    model_names = list(preds.keys())
    test_df = pd.DataFrame({"date": pd.to_datetime(dates_te), "ticker": tickers_te,
                            "actual": y_te})
    for name, yp in preds.items():
        test_df[f"pred::{name}"] = np.asarray(yp)
    test_records = test_df.assign(
        date=test_df["date"].dt.strftime("%Y-%m-%d")).to_dict("records")

    pred_vs_actual = compute_pred_vs_actual(test_records, model_names)
    strat_dates, strategy, buyhold = compute_strategy(test_records, model_names)

    # rank by directional accuracy then d-prime (what a trader cares about)
    leaderboard.sort(key=lambda r: (r["Dir. Acc"],
                                    r["d-prime"] if r["d-prime"] == r["d-prime"] else -9),
                     reverse=True)

    # feature importance from RF (single, clear chart)
    importance = dict(zip(FEATURE_COLS, rf.feature_importances_))

    live = _live_predictions(session, {"Linear Regression": lr, "Random Forest": rf},
                             scaler, days_back)

    return {
        "ok": True, "error": None,
        "n_samples": len(y), "n_test": int(len(y_te)),
        "n_tickers": len(set(tickers)),
        "torch_ok": torch_ok,
        "leaderboard": leaderboard,
        "model_names": model_names,
        "pred_vs_actual": pred_vs_actual,
        "strategy": strategy, "strategy_dates": strat_dates,
        "buyhold": list(buyhold),
        "test_records": test_records,
        "importance": importance,
        "feature_cols": FEATURE_COLS,
        "live": live,
    }


# ── Re-aggregation helpers (used for the whole set AND asset-filtered subsets) ─
def compute_pred_vs_actual(records, model_names) -> dict:
    df = pd.DataFrame(records)
    out = {}
    if df.empty:
        return out
    for name in model_names:
        col = f"pred::{name}"
        if col in df.columns:
            out[name] = {"actual": df["actual"].tolist(), "pred": df[col].tolist()}
    return out


def compute_leaderboard(records, model_names) -> list:
    """Recompute the metrics table from a (possibly asset-filtered) record set."""
    df = pd.DataFrame(records)
    board = []
    if df.empty:
        return board
    y = df["actual"].values
    for name in model_names:
        col = f"pred::{name}"
        if col not in df.columns:
            continue
        yp = df[col].values
        board.append({
            "Model": name,
            "MAE": mean_absolute_error(y, yp),
            "RMSE": float(np.sqrt(mean_squared_error(y, yp))),
            "R2": r2_score(y, yp) if len(y) > 2 else float("nan"),
            "Dir. Acc": _directional_accuracy(y, yp),
            "d-prime": _d_prime(y, yp),
        })
    board.sort(key=lambda r: (r["Dir. Acc"],
                              r["d-prime"] if r["d-prime"] == r["d-prime"] else -9),
               reverse=True)
    return board


def compute_strategy(records, model_names):
    """Long-or-flat by predicted sign, averaged across tickers per day, indexed
    to 100. Works on the full test set or any asset-filtered subset."""
    df = pd.DataFrame(records)
    if df.empty:
        return [], {}, []
    df = df.sort_values("date")
    strategy, dates, buyhold = {}, None, None
    for name in model_names:
        col = f"pred::{name}"
        if col not in df.columns:
            continue
        d = df[["date", "actual", col]].copy()
        d["sret"] = np.where(d[col] > 0, d["actual"], 0.0)
        daily = d.groupby("date").agg(s=("sret", "mean"),
                                      b=("actual", "mean")).reset_index().sort_values("date")
        strategy[name] = (100 * np.cumprod(1 + daily["s"].values / 100)).tolist()
        if dates is None:
            dates = daily["date"].tolist()
            buyhold = (100 * np.cumprod(1 + daily["b"].values / 100)).tolist()
    return dates, strategy, buyhold


def _live_predictions(session, tab_models, scaler, days_back) -> pd.DataFrame:
    ds = build_market_dataset(session, days_back=days_back)
    if ds.empty:
        return pd.DataFrame()
    latest = ds.sort_values("date").groupby("ticker").tail(1).reset_index(drop=True)
    Xs = scaler.transform(latest[FEATURE_COLS].fillna(0).values)
    rf = tab_models["Random Forest"]
    lr = tab_models["Linear Regression"]
    pred = (rf.predict(Xs) + lr.predict(Xs)) / 2  # ensemble of the two tabular models
    out = pd.DataFrame({
        "ticker": latest["ticker"],
        "name": latest["ticker"].map(TICKER_NAMES).fillna(latest["ticker"]),
        "avg_sentiment": latest["avg_sentiment"].round(2),
        "pred_return": np.round(pred, 2),
    })
    out["Signal"] = out["pred_return"].apply(
        lambda v: "BUY" if v > 0.15 else ("SELL" if v < -0.15 else "HOLD"))
    return out.sort_values("pred_return", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    from database.db_setup import getSession
    r = train_all_models(getSession())
    if not r["ok"]:
        print("ERROR:", r["error"])
    else:
        print(f"{r['n_samples']} samples, {r['n_test']} test, torch={r['torch_ok']}\n")
        print(pd.DataFrame(r["leaderboard"]).to_string(index=False))
        print("\nFinal strategy index:")
        for m, idx in r["strategy"].items():
            print(f"  {m:18s} {idx[-1]:.1f}  (buy&hold {r['buyhold'][-1]:.1f})")
        print("\nLive:")
        print(r["live"].to_string(index=False))
