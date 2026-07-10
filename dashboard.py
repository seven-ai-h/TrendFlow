import os
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from database.db_setup import getSession
from database.models import Story, MarketData
from analysis.model_lab import (
    train_all_models, MODEL_EXPLAINERS,
    compute_leaderboard, compute_strategy, compute_pred_vs_actual,
)
from config import TICKER_NAMES

LOOKBACK = 120  # days of history the models train on

st.set_page_config(page_title="TrendFlow", page_icon="📈", layout="wide",
                   initial_sidebar_state="expanded")

if not os.path.exists("trendflow.db"):
    st.warning("⚠️ No database found. Run `python seed_data.py` (demo) or "
               "`python test_hn_api.py` (live) first.")
    st.stop()

# ── Light data-viz palette ────────────────────────────────────────────────────
SURFACE, PAGE, INK, INK2, MUTED = "#ffffff", "#f9f9f7", "#0b0b0b", "#52514e", "#898781"
GRID, GOOD, BAD, ACCENT = "#e1e0d9", "#0ca30c", "#d03b3b", "#2a78d6"
MC = {"Linear Regression": "#2a78d6", "Random Forest": "#1baf7a", "LSTM": "#4a3aa7"}

st.markdown(f"""
<style>
  .block-container {{ padding-top: 2.2rem; max-width: 1150px; }}
  h1 {{ font-weight: 800; letter-spacing: -0.6px; font-size: 2.0rem; }}
  h2 {{ font-weight: 700; font-size: 1.3rem; margin-top: 0.4rem; }}
  h3 {{ font-weight: 700; font-size: 1.05rem; }}
  .lede {{ color:{INK2}; font-size: 1.0rem; margin: -0.3rem 0 1.2rem 0; }}
  .note {{ color:{MUTED}; font-size: 0.86rem; line-height:1.5; margin: 0.2rem 0 1.1rem 0; }}
  .card {{ background:{SURFACE}; border:1px solid {GRID}; border-radius:12px;
          padding:16px 18px; margin-bottom:12px; }}
  .pill {{ display:inline-block; padding:2px 10px; border-radius:20px;
          font-size:0.78rem; font-weight:700; }}
  .buy  {{ background:#e7f6e7; color:#0a7a0a; }}
  .sell {{ background:#fbe6e6; color:#b52a2a; }}
  .hold {{ background:#eeeeec; color:#52514e; }}
  [data-testid="stMetricValue"] {{ font-weight:800; }}
</style>
""", unsafe_allow_html=True)


def sfig(fig, height=None):
    fig.update_layout(plot_bgcolor=SURFACE, paper_bgcolor="rgba(0,0,0,0)",
                      font_color=INK, font_size=12, margin=dict(l=10, r=10, t=40, b=10),
                      xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID),
                      yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID),
                      legend=dict(font=dict(size=11)))
    if height:
        fig.update_layout(height=height)
    return fig


session = getSession()

@st.cache_resource(show_spinner="Training Linear Regression, Random Forest and the LSTM…")
def _train_cached(lookback_days, data_version):
    return train_all_models(session, days_back=lookback_days)


def _data_version():
    """Cheap signal that changes when the data does, to bust the cache."""
    return session.query(MarketData).count() + session.query(Story).count()


def get_lab():
    return _train_cached(LOOKBACK, _data_version())


# ── Sidebar: brand + asset filter ─────────────────────────────────────────────
_all_names = [TICKER_NAMES[t] for t in TICKER_NAMES]
with st.sidebar:
    st.markdown("### 📈 TrendFlow")
    st.caption("Sentiment → next-day returns")
    st.divider()
    st.markdown("**Assets**")
    picked_names = st.multiselect("Filter assets", _all_names, default=_all_names,
                                  label_visibility="collapsed",
                                  placeholder="Choose assets…")
    st.caption(f"{len(picked_names)} of {len(_all_names)} assets shown.")
    st.divider()
    st.caption("Demo data is synthetic (this box blocks live APIs). Run "
               "`test_hn_api.py` locally for real headlines + prices.")

# name -> ticker, and the set of selected tickers used across pages
_name_to_ticker = {v: k for k, v in TICKER_NAMES.items()}
SELECTED = {_name_to_ticker[n] for n in picked_names if n in _name_to_ticker}


def _need_assets():
    if not SELECTED:
        st.info("👈 Select at least one asset in the sidebar.")
        return True
    return False


# ══════════════════════════════ PAGE 1 · MODELS ══════════════════════════════
def page_models():
    st.title("🧪 Model Lab")
    st.markdown("<div class='lede'>Three models predict each asset's "
                "<b>next-day return</b> — a Linear baseline, a Random Forest, and an "
                "LSTM neural network — tested on a held-out future period.</div>",
                unsafe_allow_html=True)

    lab = get_lab()
    if not lab["ok"]:
        st.warning(f"⚠️ {lab['error']}")
        return
    if _need_assets():
        return
    if not lab["torch_ok"]:
        st.info("ℹ️ PyTorch not available — LSTM skipped. `pip install torch` to enable it.")

    # Re-aggregate everything for the selected assets (no retraining)
    names = lab["model_names"]
    rows = [r for r in lab["test_records"] if r["ticker"] in SELECTED]
    leaderboard = compute_leaderboard(rows, names)
    strat_dates, strategy, buyhold = compute_strategy(rows, names)
    pred_vs_actual = compute_pred_vs_actual(rows, names)
    all_selected = len(SELECTED) == lab["n_tickers"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Test predictions", f"{len(rows):,}")
    c2.metric("Assets shown", len(SELECTED))
    c3.metric("Models", len(names))

    lb = pd.DataFrame(leaderboard)
    winner = lb.iloc[0]
    if not all_selected:
        st.caption(f"📊 Metrics & charts reflect the {len(SELECTED)} selected asset(s). "
                   "Select all to see the full-universe results.")

    # Leaderboard
    st.subheader("🏆 Leaderboard")
    disp = lb.copy()
    disp["Dir. Acc"] = (disp["Dir. Acc"] * 100).map("{:.1f}%".format)
    for col in ["MAE", "RMSE", "R2", "d-prime"]:
        disp[col] = disp[col].map("{:.3f}".format)
    st.dataframe(disp, use_container_width=True, hide_index=True)
    wexp = MODEL_EXPLAINERS.get(winner["Model"], {})
    st.markdown(
        f"<div class='note'>🥇 <b>{winner['Model']}</b> leads — "
        f"<b>{winner['Dir. Acc']*100:.1f}%</b> directional accuracy, d′ "
        f"<b>{winner['d-prime']:.2f}</b>. {wexp.get('read','')}<br>"
        "<b>Reading it:</b> MAE/RMSE = avg error size (lower better) · "
        "Dir. Acc = right up/down direction (50% = coin flip) · "
        "d′ = how cleanly it separates up-days from down-days.</div>",
        unsafe_allow_html=True)

    # Strategy Index — headline
    st.subheader("💰 Would it have made money?")
    st.markdown("<div class='note'>Grow <b>$100</b> over the test period by holding an "
                "asset only when the model predicts a rise, vs. simply buying &amp; "
                "holding everything (dashed). Above the dashed line = beat the market."
                "</div>", unsafe_allow_html=True)
    fig = go.Figure()
    for name, idx in strategy.items():
        fig.add_trace(go.Scatter(x=strat_dates, y=idx, mode="lines",
                                 name=name, line=dict(color=MC.get(name), width=2.5)))
    fig.add_trace(go.Scatter(x=strat_dates, y=buyhold, mode="lines",
                             name="Buy & Hold", line=dict(color=MUTED, width=2, dash="dash")))
    fig.update_layout(title="Growth of $100 (held-out test period)",
                      yaxis_title="Index (start = 100)")
    st.plotly_chart(sfig(fig, 400), use_container_width=True)

    # Predicted vs actual
    st.subheader("🎯 Predicted vs. actual return")
    st.markdown("<div class='note'>Each dot is one test day. The tighter the cloud hugs "
                "the diagonal, the better the prediction.</div>", unsafe_allow_html=True)
    cols = st.columns(len(pred_vs_actual))
    for col, (name, pv) in zip(cols, pred_vs_actual.items()):
        with col:
            a, p = np.array(pv["actual"]), np.array(pv["pred"])
            lim = float(max(np.abs(a).max(), np.abs(p).max())) * 1.05
            f = go.Figure()
            f.add_trace(go.Scatter(x=[-lim, lim], y=[-lim, lim], mode="lines",
                                   line=dict(color=GRID, dash="dash"), showlegend=False))
            f.add_trace(go.Scatter(x=a, y=p, mode="markers", showlegend=False,
                                   marker=dict(color=MC.get(name), size=6, opacity=0.5),
                                   hovertemplate="actual %{x:.2f}%<br>pred %{y:.2f}%<extra></extra>"))
            f.update_layout(title=name, xaxis_title="Actual %", yaxis_title="Predicted %",
                            xaxis=dict(range=[-lim, lim]), yaxis=dict(range=[-lim, lim]))
            st.plotly_chart(sfig(f, 280), use_container_width=True)

    # Model cards
    st.subheader("📖 The three models")
    cols = st.columns(3)
    for col, name in zip(cols, ["Linear Regression", "Random Forest", "LSTM"]):
        info = MODEL_EXPLAINERS[name]
        with col:
            st.markdown(
                f"<div class='card'><div style='color:{MC[name]};font-weight:800'>{name}</div>"
                f"<div style='color:{MUTED};font-size:0.78rem;margin-bottom:6px'>{info['tag']}</div>"
                f"<div style='font-size:0.86rem'>{info['how']}</div></div>",
                unsafe_allow_html=True)


# ══════════════════════════ PAGE 2 · PREDICTIONS ═════════════════════════════
def page_predictions():
    st.title("📡 Predictions")
    st.markdown("<div class='lede'>What the models say to do with each asset "
                "<b>right now</b> — the predicted next-day return drives a "
                "<b>BUY / HOLD / SELL</b> call.</div>", unsafe_allow_html=True)

    lab = get_lab()
    if not lab["ok"]:
        st.warning(f"⚠️ {lab['error']}")
        return
    if _need_assets():
        return
    live = lab["live"][lab["live"]["ticker"].isin(SELECTED)].reset_index(drop=True)
    if live.empty:
        st.info("No predictions for the selected assets.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("🟢 BUY", int((live["Signal"] == "BUY").sum()))
    c2.metric("⚪ HOLD", int((live["Signal"] == "HOLD").sum()))
    c3.metric("🔴 SELL", int((live["Signal"] == "SELL").sum()))

    colL, colR = st.columns([3, 2], gap="large")
    with colL:
        d = live.copy()
        d["clr"] = d["Signal"].map({"BUY": GOOD, "SELL": BAD, "HOLD": MUTED})
        fig = go.Figure(go.Bar(
            x=d["pred_return"], y=d["name"], orientation="h", marker_color=d["clr"],
            text=[f"{v:+.2f}%" for v in d["pred_return"]], textposition="outside",
            hovertemplate="%{y}: %{x:+.2f}%<extra></extra>"))
        fig.add_vline(x=0, line_color=MUTED)
        lo, hi = float(d["pred_return"].min()), float(d["pred_return"].max())
        pad = max(0.12, (hi - lo) * 0.18)
        fig.update_layout(title="Predicted next-day return",
                          yaxis=dict(autorange="reversed"), xaxis_title="Return %",
                          xaxis=dict(range=[lo - pad, hi + pad]))
        st.plotly_chart(sfig(fig, 460), use_container_width=True)
    with colR:
        st.markdown("#### Signals")
        for _, r in live.iterrows():
            pill = r["Signal"].lower()
            st.markdown(
                f"<div class='card' style='padding:10px 14px;margin-bottom:8px'>"
                f"<span style='font-weight:700'>{r['name']}</span> "
                f"<span class='pill {pill}'>{r['Signal']}</span><br>"
                f"<span style='color:{INK2};font-size:0.84rem'>"
                f"predicted <b>{r['pred_return']:+.2f}%</b> · "
                f"sentiment {r['avg_sentiment']:+.2f}</span></div>",
                unsafe_allow_html=True)

    st.markdown(f"<div class='note'>⚠️ Not financial advice. On a noisy real-world "
                "signal these models run near ~55% directional accuracy — treat any single "
                "call as a weak prior, not a guarantee.</div>", unsafe_allow_html=True)


# ═══════════════════════════ PAGE 3 · SENTIMENT ══════════════════════════════
def page_sentiment():
    from analysis.market_features import tickers_in_text, TICKER_NAMES
    st.title("💬 Sentiment")
    st.markdown("<div class='lede'>The input signal: every headline scored from "
                "<b>−1 bearish</b> to <b>+1 bullish</b> (VADER + a finance word-list).</div>",
                unsafe_allow_html=True)

    if _need_assets():
        return
    rows = [{"name": TICKER_NAMES.get(tk, tk), "sentiment": s.sentiment or 0.0,
             "title": s.title}
            for s in session.query(Story).all()
            for tk in tickers_in_text(s.title or "") if tk in SELECTED]
    if not rows:
        st.info("No headlines for the selected assets.")
        return
    sdf = pd.DataFrame(rows)

    agg = sdf.groupby("name")["sentiment"].mean().reset_index().sort_values("sentiment")
    fig = go.Figure(go.Bar(
        x=agg["sentiment"], y=agg["name"], orientation="h",
        marker_color=[GOOD if v > 0 else BAD for v in agg["sentiment"]],
        hovertemplate="%{y}: %{x:+.2f}<extra></extra>"))
    fig.update_layout(title="Average headline sentiment by asset", xaxis_title="Sentiment")
    st.plotly_chart(sfig(fig, 380), use_container_width=True)

    uniq = sdf.drop_duplicates("title")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("#### 🟢 Most bullish")
        for _, r in uniq.nlargest(7, "sentiment").iterrows():
            st.markdown(f"<div class='note' style='margin:4px 0'>"
                        f"<b style='color:{GOOD}'>+{r['sentiment']:.2f}</b> &nbsp;{r['title']}</div>",
                        unsafe_allow_html=True)
    with c2:
        st.markdown("#### 🔴 Most bearish")
        for _, r in uniq.nsmallest(7, "sentiment").iterrows():
            st.markdown(f"<div class='note' style='margin:4px 0'>"
                        f"<b style='color:{BAD}'>{r['sentiment']:.2f}</b> &nbsp;{r['title']}</div>",
                        unsafe_allow_html=True)


# ── Native sidebar navigation ─────────────────────────────────────────────────
nav = st.navigation([
    st.Page(page_models, title="Models", icon="🧪", default=True),
    st.Page(page_predictions, title="Predictions", icon="📡"),
    st.Page(page_sentiment, title="Sentiment", icon="💬"),
])
nav.run()
