import os
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()  # read ANTHROPIC_API_KEY / NEWS_API_KEY etc. from a local .env

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from database.db_setup import getSession
from database.models import Story, MarketData, PipelineRun

st.set_page_config(page_title="TrendFlow — Sentiment → Returns", page_icon="📡",
                   layout="wide", initial_sidebar_state="expanded")

if not os.path.exists("trendflow.db"):
    st.warning("⚠️ No database found. Run `python seed_data.py` (demo) "
               "or `python test_hn_api.py` (live) first.")
    st.stop()

# ── Design tokens (validated data-viz dark palette) ───────────────────────────
SURFACE, GRID, INK, INK2 = "#1a1a19", "#2c2c2a", "#ffffff", "#c3c2b7"
GOOD, BAD, ACCENT = "#0ca30c", "#d03b3b", "#3987e5"
MC = {"Linear Regression": "#3987e5", "Random Forest": "#199e70", "LSTM": "#9085e9"}

st.markdown(f"""
<style>
    .block-container {{ padding-top: 2rem; max-width: 1400px; }}
    h1 {{ color:{INK}; font-weight:800; letter-spacing:-0.5px; }}
    h2,h3 {{ color:{INK}; }}
    .ctx {{
        background:#12233a; border-left:4px solid {ACCENT};
        border-radius:6px; padding:10px 14px; margin:6px 0 14px 0;
        color:{INK2}; font-size:0.9rem; line-height:1.5;
    }}
    .ctx b {{ color:{INK}; }}
    .sig-buy {{ color:{GOOD}; font-weight:700; }}
    .sig-sell {{ color:{BAD}; font-weight:700; }}
    .sig-hold {{ color:{INK2}; font-weight:700; }}
    .mcard {{ background:{SURFACE}; border:1px solid {GRID}; border-radius:10px;
             padding:12px 14px; margin-bottom:10px; }}
</style>
""", unsafe_allow_html=True)


def ctx(text):
    st.markdown(f"<div class='ctx'>{text}</div>", unsafe_allow_html=True)


def style_fig(fig, height=None, legend=True):
    fig.update_layout(plot_bgcolor=SURFACE, paper_bgcolor="rgba(0,0,0,0)",
                      font_color=INK, font_size=12, margin=dict(l=10, r=10, t=44, b=10),
                      xaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
                      yaxis=dict(gridcolor=GRID, zerolinecolor=GRID))
    if height:
        fig.update_layout(height=height)
    fig.update_layout(showlegend=legend, legend=dict(font=dict(size=10)))
    return fig


session = getSession()

st.title("📡 TrendFlow — Sentiment → Next-Day Returns")
st.markdown(
    f"<span style='color:{INK2}'>Predicting the <b>next-day return %</b> of tech stocks "
    f"& crypto from <b>headline sentiment</b> + price momentum — with three models: a "
    f"<b>Linear</b> baseline, a <b>Random Forest</b>, and an <b>LSTM</b> neural network.</span>",
    unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Controls")
lookback = st.sidebar.slider("📅 History used (days)", 40, 130, 120, 10)
if st.sidebar.button("🔄 Retrain models"):
    st.session_state.pop("lab", None)
    st.rerun()
st.sidebar.divider()
st.sidebar.caption("Demo data is synthetic (this box blocks live APIs). Run "
                   "`test_hn_api.py` locally for real headlines + prices via yfinance.")

# ── KPI row ───────────────────────────────────────────────────────────────────
n_stories = session.query(Story).count()
n_prices = session.query(MarketData).count()
n_tickers = session.query(MarketData.ticker).distinct().count()
sents = [s.sentiment for s in session.query(Story.sentiment).all() if s.sentiment is not None]
avg_sent = float(np.mean(sents)) if sents else 0.0
k1, k2, k3, k4 = st.columns(4)
k1.metric("🏦 Assets tracked", n_tickers)
k2.metric("📰 Headlines scored", f"{n_stories:,}")
k3.metric("📈 Price bars", f"{n_prices:,}")
k4.metric("💬 Avg sentiment", f"{avg_sent:+.2f}",
          "Bullish" if avg_sent > 0.1 else ("Bearish" if avg_sent < -0.1 else "Neutral"))
st.divider()

from analysis.model_lab import train_all_models, MODEL_EXPLAINERS


def get_lab():
    if "lab" not in st.session_state or st.session_state.get("lab_lb") != lookback:
        with st.spinner("Training Linear Regression, Random Forest and the LSTM…"):
            st.session_state["lab"] = train_all_models(session, days_back=lookback)
            st.session_state["lab_lb"] = lookback
    return st.session_state["lab"]


tab_signals, tab_lab, tab_sentiment, tab_ai, tab_pipeline = st.tabs([
    "📡 Live Signals", "🧪 Model Lab", "💬 Sentiment", "🧠 AI Briefing", "🔧 Pipeline"])

# ══ TAB 1: LIVE SIGNALS ═══════════════════════════════════════════════════════
with tab_signals:
    st.header("📡 Tomorrow's Signals")
    ctx("<b>What this is:</b> each model predicts tomorrow's return; we average the "
        "tabular models into one number per asset. <b>BUY</b> if the predicted move is "
        "above +0.15%, <b>SELL</b> if below −0.15%, else <b>HOLD</b>. "
        "Not financial advice — see the reliability note in the Model Lab.")
    lab = get_lab()
    if not lab["ok"]:
        st.warning(f"⚠️ {lab['error']}")
    else:
        live = lab["live"]
        c1, c2, c3 = st.columns(3)
        c1.metric("🟢 BUY", int((live["Signal"] == "BUY").sum()))
        c2.metric("⚪ HOLD", int((live["Signal"] == "HOLD").sum()))
        c3.metric("🔴 SELL", int((live["Signal"] == "SELL").sum()))
        col_c, col_l = st.columns([2, 1])
        with col_c:
            d = live.copy()
            d["clr"] = d["Signal"].map({"BUY": GOOD, "SELL": BAD, "HOLD": INK2})
            fig = go.Figure(go.Bar(x=d["pred_return"], y=d["name"], orientation="h",
                                   marker_color=d["clr"],
                                   text=[f"{v:+.2f}%" for v in d["pred_return"]],
                                   textposition="outside",
                                   hovertemplate="%{y}: %{x:+.2f}%<extra></extra>"))
            fig.add_vline(x=0, line_color=INK2)
            fig.update_layout(title="Predicted next-day return by asset",
                              yaxis=dict(autorange="reversed"), xaxis_title="Return %")
            st.plotly_chart(style_fig(fig, 440, legend=False), use_container_width=True)
        with col_l:
            st.subheader("Ranked")
            for _, r in live.iterrows():
                cls = f"sig-{r['Signal'].lower()}"
                st.markdown(f"**{r['name']}** <span class='{cls}'>{r['Signal']}</span><br>"
                            f"<small style='color:{INK2}'>{r['pred_return']:+.2f}% · "
                            f"sentiment {r['avg_sentiment']:+.2f}</small>",
                            unsafe_allow_html=True)
                st.markdown("<hr style='margin:6px 0;border-color:#2c2c2a'>",
                            unsafe_allow_html=True)

# ══ TAB 2: MODEL LAB ══════════════════════════════════════════════════════════
with tab_lab:
    st.header("🧪 Model Lab — Three Models, One Honest Test")
    ctx("<b>The task:</b> predict each asset's <b>next-day return %</b> from today's "
        "sentiment, buzz and price momentum. All three models are tested on the <b>same "
        "held-out later period</b> (trained on the past, tested on the future — no "
        "peeking). Below are only the charts that actually tell you something.")
    lab = get_lab()
    if not lab["ok"]:
        st.warning(f"⚠️ {lab['error']}")
    else:
        if not lab["torch_ok"]:
            st.info("ℹ️ PyTorch isn't available here, so the LSTM was skipped. "
                    "`pip install torch` to enable it.")
        m1, m2, m3 = st.columns(3)
        m1.metric("🧾 Samples", f"{lab['n_samples']:,}")
        m2.metric("🧪 Test set", f"{lab['n_test']:,}")
        m3.metric("🏦 Assets", lab["n_tickers"])

        lb = pd.DataFrame(lab["leaderboard"])
        winner = lb.iloc[0]

        # ── Leaderboard ───────────────────────────────────────────────────────
        st.subheader("🏆 Leaderboard")
        ctx("<b>How to read it —</b> "
            "<b>MAE/RMSE:</b> average size of the prediction error (lower is better). "
            "<b>Dir. Acc:</b> how often the up/down direction is right (50% = coin flip). "
            "<b>d′ (d-prime):</b> how cleanly it separates up-days from down-days "
            "(0 = no skill, higher = better). Ranked by directional accuracy.")
        disp = lb.copy()
        disp["Dir. Acc"] = (disp["Dir. Acc"] * 100).map("{:.1f}%".format)
        for c in ["MAE", "RMSE", "R2", "d-prime"]:
            disp[c] = disp[c].map("{:.3f}".format)
        st.dataframe(disp, use_container_width=True, hide_index=True)
        wexp = MODEL_EXPLAINERS.get(winner["Model"], {})
        st.success(f"🥇 **{winner['Model']}** leads — directional accuracy "
                   f"**{winner['Dir. Acc']*100:.1f}%**, d′ **{winner['d-prime']:.2f}**. "
                   f"{wexp.get('read','')}")

        # ── Strategy Index (headline result) ──────────────────────────────────
        st.subheader("💰 Strategy Index — would it have made money?")
        ctx("<b>The bottom line.</b> Start with <b>$100</b> on the first test day. Each "
            "day, if a model predicts a rise, you hold that asset (else you sit in cash). "
            "The lines show how $100 grows following each model, versus simply "
            "<b>buying and holding</b> everything (dashed). Above the dashed line = the "
            "model beat the market.")
        figs = go.Figure()
        dts = lab["strategy_dates"]
        for name, idx in lab["strategy"].items():
            figs.add_trace(go.Scatter(x=dts, y=idx, mode="lines", name=name,
                                      line=dict(color=MC.get(name), width=2.5)))
        figs.add_trace(go.Scatter(x=dts, y=lab["buyhold"], mode="lines",
                                  name="Buy & Hold", line=dict(color=INK2, width=2, dash="dash")))
        figs.update_layout(title="Growth of $100 (held-out test period)",
                           yaxis_title="Index (start = 100)", xaxis_title="Date")
        st.plotly_chart(style_fig(figs, 400), use_container_width=True)
        finals = {n: v[-1] for n, v in lab["strategy"].items()}
        best_strat = max(finals, key=finals.get)
        st.caption(f"Final: " + " · ".join(f"**{n}** {v:.1f}" for n, v in finals.items())
                   + f" · Buy & Hold {lab['buyhold'][-1]:.1f}  →  "
                   f"best strategy: **{best_strat}**")

        # ── Predicted vs Actual (small multiples) ─────────────────────────────
        st.subheader("🎯 Predicted vs. Actual return")
        ctx("Each dot is one test day for one asset. <b>The closer the dots hug the "
            "dashed diagonal, the better the prediction.</b> A flat cloud means the model "
            "is guessing; a tilted cloud means it's tracking real moves.")
        pcols = st.columns(len(lab["pred_vs_actual"]))
        for col, (name, pv) in zip(pcols, lab["pred_vs_actual"].items()):
            with col:
                a, p = np.array(pv["actual"]), np.array(pv["pred"])
                lim = float(max(np.abs(a).max(), np.abs(p).max())) * 1.05
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=[-lim, lim], y=[-lim, lim], mode="lines",
                                         line=dict(color=INK2, dash="dash"), showlegend=False))
                fig.add_trace(go.Scatter(x=a, y=p, mode="markers",
                                         marker=dict(color=MC.get(name), size=6, opacity=0.55),
                                         showlegend=False,
                                         hovertemplate="actual %{x:.2f}%<br>pred %{y:.2f}%<extra></extra>"))
                fig.update_layout(title=name, xaxis_title="Actual %", yaxis_title="Predicted %",
                                  xaxis=dict(range=[-lim, lim]), yaxis=dict(range=[-lim, lim]))
                st.plotly_chart(style_fig(fig, 300, legend=False), use_container_width=True)

        # ── d-prime + directional accuracy bars ───────────────────────────────
        cA, cB = st.columns(2)
        with cA:
            st.subheader("📏 d′ — signal separation")
            fig = go.Figure(go.Bar(x=lb["Model"], y=lb["d-prime"],
                                   marker_color=[MC.get(m) for m in lb["Model"]],
                                   text=[f"{v:.2f}" for v in lb["d-prime"]], textposition="outside"))
            fig.update_layout(title="Higher = cleaner up/down separation", yaxis_title="d′")
            st.plotly_chart(style_fig(fig, 300, legend=False), use_container_width=True)
        with cB:
            st.subheader("🎯 Directional accuracy")
            fig = go.Figure(go.Bar(x=lb["Model"], y=lb["Dir. Acc"] * 100,
                                   marker_color=[MC.get(m) for m in lb["Model"]],
                                   text=[f"{v*100:.1f}%" for v in lb["Dir. Acc"]],
                                   textposition="outside"))
            fig.add_hline(y=50, line_dash="dash", line_color=INK2,
                          annotation_text="coin flip", annotation_font_color=INK2)
            fig.update_layout(title="% of days with the right direction",
                              yaxis=dict(range=[40, 80], title="%"))
            st.plotly_chart(style_fig(fig, 300, legend=False), use_container_width=True)

        # ── Model explainer ───────────────────────────────────────────────────
        st.subheader("📖 The three models")
        ecols = st.columns(3)
        for col, name in zip(ecols, ["Linear Regression", "Random Forest", "LSTM"]):
            info = MODEL_EXPLAINERS[name]
            with col:
                st.markdown(
                    f"<div class='mcard'>"
                    f"<div style='color:{MC[name]};font-weight:700'>{name}</div>"
                    f"<div style='color:{INK2};font-size:0.8rem;margin-bottom:6px'>{info['tag']}</div>"
                    f"<div style='font-size:0.86rem'><b>How:</b> {info['how']}</div>"
                    f"<div style='font-size:0.86rem;margin-top:6px;color:{INK2}'>"
                    f"<b>Read:</b> {info['read']}</div></div>", unsafe_allow_html=True)

        # ── Reliability note ──────────────────────────────────────────────────
        st.warning(
            "**Reliability check.** On this (synthetic demo) signal the models reach "
            f"~{winner['Dir. Acc']*100:.0f}% directional accuracy. **Real markets are far "
            "noisier** — expect closer to 50–55%, and treat any single prediction as a "
            "weak prior, never a guarantee. The value is in the aggregate edge over many "
            "trades, not any one call.")

# ══ TAB 3: SENTIMENT ══════════════════════════════════════════════════════════
with tab_sentiment:
    st.header("💬 Sentiment — the input signal")
    ctx("Every headline is scored from <b>−1 (bearish)</b> to <b>+1 (bullish)</b> by VADER "
        "plus a finance word-list. This is the raw signal the models turn into predictions.")
    from analysis.market_features import tickers_in_text, TICKER_NAMES
    rows = [{"name": TICKER_NAMES.get(tk, tk), "sentiment": s.sentiment or 0.0,
             "date": s.timestamp.date(), "title": s.title}
            for s in session.query(Story).all() for tk in tickers_in_text(s.title or "")]
    if not rows:
        st.info("No ticker-tagged headlines yet.")
    else:
        sdf = pd.DataFrame(rows)
        agg = sdf.groupby("name")["sentiment"].mean().reset_index().sort_values("sentiment")
        fig = go.Figure(go.Bar(x=agg["sentiment"], y=agg["name"], orientation="h",
                               marker_color=[GOOD if v > 0 else BAD for v in agg["sentiment"]],
                               hovertemplate="%{y}: %{x:+.2f}<extra></extra>"))
        fig.update_layout(title="Average headline sentiment by asset", xaxis_title="Sentiment")
        st.plotly_chart(style_fig(fig, 380, legend=False), use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("🟢 Most bullish headlines")
            for _, r in sdf.nlargest(6, "sentiment").iterrows():
                st.markdown(f"<small style='color:{GOOD}'>+{r['sentiment']:.2f}</small> {r['title']}",
                            unsafe_allow_html=True)
        with c2:
            st.subheader("🔴 Most bearish headlines")
            for _, r in sdf.nsmallest(6, "sentiment").iterrows():
                st.markdown(f"<small style='color:{BAD}'>{r['sentiment']:.2f}</small> {r['title']}",
                            unsafe_allow_html=True)

# ══ TAB 4: AI BRIEFING ════════════════════════════════════════════════════════
with tab_ai:
    st.header("🧠 AI Market Briefing")
    ctx("Claude reads the live signals + sentiment and writes a short analyst briefing.")
    from analysis.trend_summarizer import summarize_market_signals
    if not os.getenv("ANTHROPIC_API_KEY"):
        st.warning("⚠️ Set `ANTHROPIC_API_KEY` in your `.env` to enable AI briefings.")
    else:
        lab = get_lab()
        if lab["ok"] and st.button("📝 Generate briefing", type="primary"):
            with st.spinner("Claude is analyzing…"):
                note = f"Overall market sentiment is {avg_sent:+.2f} across {n_stories} headlines."
                text = summarize_market_signals(lab["live"], lab["leaderboard"], note)
            st.markdown("---")
            st.markdown(text)
            st.caption(f"Generated {datetime.now():%Y-%m-%d %H:%M}")

# ══ TAB 5: PIPELINE ═══════════════════════════════════════════════════════════
with tab_pipeline:
    st.header("🔧 Pipeline Monitor")
    runs = session.query(PipelineRun).order_by(PipelineRun.started_at.desc()).limit(20).all()
    if runs:
        st.dataframe(pd.DataFrame([{
            "Started": r.started_at.strftime("%Y-%m-%d %H:%M"), "Status": r.status,
            "Sources": r.sources_run, "Stories": r.stories_collected,
            "Duration (s)": round((r.finished_at - r.started_at).total_seconds(), 1)
            if r.finished_at else "—"} for r in runs]),
            use_container_width=True, hide_index=True)
    else:
        st.info("No pipeline runs recorded. Run `python test_hn_api.py`.")

st.divider()
st.caption(f"Last updated {datetime.now():%Y-%m-%d %H:%M:%S} · TrendFlow v5.0 — "
           "regression edition (Linear · Random Forest · LSTM)")
