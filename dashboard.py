import os
from datetime import datetime, timedelta
from collections import Counter

from dotenv import load_dotenv
load_dotenv()  # read ANTHROPIC_API_KEY / NEWS_API_KEY etc. from a local .env

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from database.db_setup import getSession
from database.models import Story, MarketData, PipelineRun

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TrendFlow — Sentiment → Market Signals",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

if not os.path.exists("trendflow.db"):
    st.warning("⚠️ No database found. Run `python seed_data.py` (demo data) "
               "or `python test_hn_api.py` (live data) first.")
    st.stop()

# ── Design tokens (validated data-viz dark palette) ───────────────────────────
SURFACE = "#1a1a19"
GRID = "#2c2c2a"
INK = "#ffffff"
INK2 = "#c3c2b7"
GOOD = "#0ca30c"
BAD = "#d03b3b"
ACCENT = "#3987e5"
CAT_COLORS = {
    "Linear & Probabilistic": "#3987e5",
    "Tree Ensembles": "#199e70",
    "Kernel & Neural": "#9085e9",
}

st.markdown(f"""
<style>
    .block-container {{ padding-top: 2rem; }}
    h1 {{ color: {INK}; font-weight: 800; letter-spacing: -0.5px; }}
    h2, h3 {{ color: {INK}; }}
    .cat-banner {{
        border-radius: 10px; padding: 10px 14px; margin-bottom: 10px;
        font-weight: 700; font-size: 1.05rem; color: #fff;
    }}
    .model-card {{
        background: {SURFACE}; border: 1px solid {GRID};
        border-radius: 10px; padding: 12px 14px; margin-bottom: 10px;
    }}
    .model-card .mname {{ font-weight: 700; font-size: 0.98rem; }}
    .model-card .mrow {{ color: {INK2}; font-size: 0.82rem; font-variant-numeric: tabular-nums; }}
    .sig-buy {{ color: {GOOD}; font-weight: 700; }}
    .sig-sell {{ color: {BAD}; font-weight: 700; }}
    .sig-hold {{ color: {INK2}; font-weight: 700; }}
</style>
""", unsafe_allow_html=True)


def _has_statsmodels():
    try:
        import statsmodels  # noqa: F401
        return True
    except Exception:
        return False


def style_fig(fig, height=None, legend=True):
    fig.update_layout(
        plot_bgcolor=SURFACE, paper_bgcolor="rgba(0,0,0,0)",
        font_color=INK, font_size=12,
        margin=dict(l=10, r=10, t=44, b=10),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
    )
    if height:
        fig.update_layout(height=height)
    if not legend:
        fig.update_layout(showlegend=False)
    else:
        fig.update_layout(legend=dict(font=dict(size=10)))
    return fig


session = getSession()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📡 TrendFlow — Sentiment → Market Signals")
st.markdown(
    f"<span style='color:{INK2}'>Predicting <b>next-day price direction</b> for tech "
    f"equities & crypto from <b>headline sentiment</b> + <b>social buzz</b> + price "
    f"momentum — compared across six ML models in three families.</span>",
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Controls")
lookback = st.sidebar.slider("📅 Training lookback (days)", 20, 90, 60, 5)
st.sidebar.caption("How much price/sentiment history to train the models on.")
if st.sidebar.button("🔄 Refresh data"):
    st.session_state.pop("lab", None)
    st.rerun()
st.sidebar.divider()
st.sidebar.markdown(
    f"<small style='color:{INK2}'>Demo data is synthetic (network-restricted box). "
    f"Run <code>test_hn_api.py</code> locally for live headlines + real prices via "
    f"yfinance.</small>", unsafe_allow_html=True)

# ── KPI row ───────────────────────────────────────────────────────────────────
n_stories = session.query(Story).count()
n_prices = session.query(MarketData).count()
n_tickers = session.query(MarketData.ticker).distinct().count()
sents = [s.sentiment for s in session.query(Story.sentiment).all() if s.sentiment is not None]
avg_sent = float(np.mean(sents)) if sents else 0.0

k1, k2, k3, k4 = st.columns(4)
k1.metric("🏦 Tickers tracked", n_tickers)
k2.metric("📰 Headlines analyzed", f"{n_stories:,}")
k3.metric("📈 Price bars", f"{n_prices:,}")
sent_word = "Bullish" if avg_sent > 0.1 else ("Bearish" if avg_sent < -0.1 else "Neutral")
k4.metric("💬 Avg sentiment", f"{avg_sent:+.2f}", sent_word)

st.divider()

# ── Train models once, cache in session ───────────────────────────────────────
from analysis.model_lab import (
    train_all_models, MODEL_COLORS, MODEL_CATEGORY, MODEL_EXPLANATIONS,
)


def get_lab():
    if "lab" not in st.session_state or st.session_state.get("lab_lb") != lookback:
        with st.spinner("Training six models across three families…"):
            st.session_state["lab"] = train_all_models(session, days_back=lookback)
            st.session_state["lab_lb"] = lookback
    return st.session_state["lab"]


tab_signals, tab_lab, tab_sentiment, tab_price, tab_ai, tab_pipeline = st.tabs([
    "📡 Live Signals", "🧪 Model Lab", "💬 Sentiment & Buzz",
    "📈 Price & Correlation", "🧠 AI Briefing", "🔧 Pipeline",
])

# ══ TAB 1: LIVE SIGNALS ═══════════════════════════════════════════════════════
with tab_signals:
    st.header("📡 Today's Signals")
    st.caption("Model consensus probability that each asset closes **up tomorrow**, "
               "with the driving sentiment. Not financial advice — see the reliability "
               "note in the Model Lab.")
    lab = get_lab()
    if not lab["ok"]:
        st.warning(f"⚠️ {lab['error']}")
    else:
        live = lab["live"]
        n_buy = (live["Signal"] == "BUY").sum()
        n_sell = (live["Signal"] == "SELL").sum()
        n_hold = (live["Signal"] == "HOLD").sum()
        c1, c2, c3 = st.columns(3)
        c1.metric("🟢 BUY signals", int(n_buy))
        c2.metric("⚪ HOLD", int(n_hold))
        c3.metric("🔴 SELL signals", int(n_sell))

        col_chart, col_list = st.columns([2, 1])
        with col_chart:
            dfl = live.copy()
            dfl["color"] = dfl["Signal"].map({"BUY": GOOD, "SELL": BAD, "HOLD": INK2})
            fig = go.Figure(go.Bar(
                x=dfl["Consensus"], y=dfl["name"], orientation="h",
                marker_color=dfl["color"],
                text=[f"{v:.0f}%" for v in dfl["Consensus"]],
                textposition="outside",
                hovertemplate="%{y}: %{x:.1f}% up<extra></extra>",
            ))
            fig.add_vline(x=50, line_dash="dash", line_color=INK2)
            fig.update_layout(title="Consensus P(up tomorrow) by asset",
                              yaxis=dict(autorange="reversed"),
                              xaxis=dict(range=[0, 100], title="Consensus %"))
            st.plotly_chart(style_fig(fig, height=420, legend=False), use_container_width=True)
        with col_list:
            st.subheader("Ranked")
            for _, r in live.iterrows():
                cls = f"sig-{r['Signal'].lower()}"
                st.markdown(
                    f"**{r['name']}** <span class='{cls}'>{r['Signal']}</span><br>"
                    f"<small style='color:{INK2}'>{r['Consensus']:.0f}% up · "
                    f"sentiment {r['avg_sentiment']:+.2f} · {int(r['buzz'])} stories</small>",
                    unsafe_allow_html=True)
                st.markdown("<hr style='margin:6px 0;border-color:#2c2c2a'>", unsafe_allow_html=True)

# ══ TAB 2: MODEL LAB (3 categories) ═══════════════════════════════════════════
with tab_lab:
    st.header("🧪 Model Lab — Three Families, One Task")
    st.markdown(
        "**The task:** given a ticker's social sentiment/buzz today plus its price "
        "momentum, *will it close up tomorrow?* Every model trains on the **same** "
        "features and split. Grouped into three algorithm families so it's easy to read."
    )
    lab = get_lab()
    if not lab["ok"]:
        st.warning(f"⚠️ {lab['error']}")
    else:
        bal = lab["class_balance"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🧾 Samples", f"{lab['n_samples']:,}")
        m2.metric("🏦 Tickers", lab["n_tickers"])
        m3.metric("📈 Up days", bal["up"])
        m4.metric("📉 Down days", bal["down"])

        lb_df = pd.DataFrame(lab["leaderboard"])
        winner = lb_df.iloc[0]

        # ── THREE CATEGORY COLUMNS ────────────────────────────────────────────
        st.subheader("Model families")
        cat_cols = st.columns(3)
        for col, (cat_name, cat) in zip(cat_cols, lab["categories"].items()):
            with col:
                st.markdown(
                    f"<div class='cat-banner' style='background:{cat['color']}'>{cat_name}</div>",
                    unsafe_allow_html=True)
                st.markdown(f"<small style='color:{INK2}'>{cat['blurb']}</small>",
                            unsafe_allow_html=True)
                for mname in cat["models"]:
                    row = lb_df[lb_df["Model"] == mname].iloc[0]
                    crown = " 👑" if mname == winner["Model"] else ""
                    st.markdown(
                        f"<div class='model-card'>"
                        f"<div class='mname' style='color:{cat['color']}'>{mname}{crown}</div>"
                        f"<div class='mrow'>F1 <b>{row['F1']:.3f}</b> · "
                        f"AUC <b>{row['ROC-AUC']:.3f}</b></div>"
                        f"<div class='mrow'>Acc {row['Accuracy']:.3f} · "
                        f"CV {row['CV Acc']:.3f} · {row['Train (ms)']:.0f} ms</div>"
                        f"</div>", unsafe_allow_html=True)
                    cm = np.array(lab["confusion"][mname])
                    figcm = px.imshow(
                        cm, text_auto=True, color_continuous_scale="Blues",
                        x=["Down", "Up"], y=["Down", "Up"],
                        labels=dict(x="Predicted", y="Actual"))
                    figcm.update_layout(coloraxis_showscale=False, height=200,
                                        margin=dict(l=6, r=6, t=6, b=6),
                                        paper_bgcolor="rgba(0,0,0,0)", font_color=INK)
                    st.plotly_chart(figcm, use_container_width=True)

        st.divider()

        # ── Leaderboard ───────────────────────────────────────────────────────
        st.subheader("🏆 Leaderboard")
        st.success(
            f"🥇 **{winner['Model']}** ({winner['Category']}) leads — "
            f"F1 {winner['F1']:.3f}, ROC-AUC {winner['ROC-AUC']:.3f}, "
            f"accuracy {winner['Accuracy']:.3f}. "
            f"On a noisy market signal, a simple model often wins — complex models "
            f"overfit the noise.")
        show = lb_df[["Model", "Category", "Accuracy", "Precision", "Recall",
                      "F1", "ROC-AUC", "CV Acc"]].copy()
        for c in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "CV Acc"]:
            show[c] = show[c].map(lambda v: f"{v:.3f}" if pd.notna(v) else "—")
        st.dataframe(show, use_container_width=True, hide_index=True)

        # ── Metric bars + ROC ─────────────────────────────────────────────────
        cA, cB = st.columns(2)
        with cA:
            st.subheader("📊 Metric comparison")
            ml = lb_df.melt(id_vars="Model",
                            value_vars=["Accuracy", "Precision", "Recall", "F1"],
                            var_name="Metric", value_name="Score")
            figm = px.bar(ml, x="Metric", y="Score", color="Model", barmode="group",
                          color_discrete_map=MODEL_COLORS)
            figm.update_layout(yaxis=dict(range=[0, 1]), title="Per-model metrics")
            st.plotly_chart(style_fig(figm, height=360), use_container_width=True)
        with cB:
            st.subheader("📈 ROC curves")
            figr = go.Figure()
            figr.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                      line=dict(dash="dash", color=INK2), name="Random"))
            for name, rc in lab["roc"].items():
                auc = rc["auc"]
                figr.add_trace(go.Scatter(
                    x=rc["fpr"], y=rc["tpr"], mode="lines",
                    name=f"{name} ({auc:.2f})",
                    line=dict(color=MODEL_COLORS.get(name), width=2)))
            figr.update_layout(title="True vs false positive rate",
                               xaxis_title="FPR", yaxis_title="TPR")
            st.plotly_chart(style_fig(figr, height=360), use_container_width=True)

        # ── Feature importance ────────────────────────────────────────────────
        st.subheader("🧩 What drives each model")
        st.caption("Importance (trees) / |coefficient| (linear), normalized per model. "
                   "SVM & MLP don't expose per-feature importances, so they're omitted.")
        fi = lab["feature_importance"]
        if fi:
            rows = []
            for mn, feats in fi.items():
                tot = sum(abs(v) for v in feats.values()) or 1.0
                for fe, va in feats.items():
                    rows.append({"Model": mn, "Feature": fe, "Importance": abs(va) / tot})
            figfi = px.bar(pd.DataFrame(rows), x="Feature", y="Importance",
                           color="Model", barmode="group", color_discrete_map=MODEL_COLORS)
            figfi.update_layout(title="Normalized feature importance", xaxis_tickangle=-40)
            st.plotly_chart(style_fig(figfi, height=380), use_container_width=True)

        # ── Live predictions lined up ─────────────────────────────────────────
        st.subheader("🎯 Every model's call, side by side")
        st.caption("Each model's P(up tomorrow) per ticker. Green = up, red = down.")
        live = lab["live"]
        model_names = [r["Model"] for r in lab["leaderboard"]]
        disp = live[["name"] + model_names + ["Consensus", "Signal"]].copy()

        def rg(val):
            try:
                v = max(0.0, min(100.0, float(val))) / 100.0
            except (TypeError, ValueError):
                return ""
            if v < 0.5:
                r, g = 255, int(200 * v / 0.5)
            else:
                r, g = int(255 * (1 - (v - 0.5) / 0.5)), 200
            return f"background-color: rgba({r},{g},70,0.55); color:#111;"

        pc = model_names + ["Consensus"]
        sty = disp.style.map(rg, subset=pc).format({c: "{:.0f}" for c in pc})
        st.dataframe(sty, use_container_width=True, hide_index=True)

        # ── Explainers grouped by family ──────────────────────────────────────
        st.subheader("📖 Model explainer")
        ecols = st.columns(3)
        for col, (cat_name, cat) in zip(ecols, lab["categories"].items()):
            with col:
                st.markdown(f"<div class='cat-banner' style='background:{cat['color']};"
                            f"font-size:0.92rem'>{cat_name}</div>", unsafe_allow_html=True)
                for mname in cat["models"]:
                    info = MODEL_EXPLANATIONS[mname]
                    with st.expander(mname):
                        st.markdown(f"**How:** {info['how']}")
                        st.markdown(f"**✅ {info['strengths']}**")
                        st.markdown(f"**⚠️ {info['weaknesses']}**")
                        st.markdown(f"**🎯 Best for:** {info['best_for']}")

# ══ TAB 3: SENTIMENT & BUZZ ═══════════════════════════════════════════════════
with tab_sentiment:
    st.header("💬 Sentiment & Buzz")
    st.caption("Headline sentiment (VADER + finance lexicon) is the core feature. "
               "Mention volume = the 'buzz' feature folded into the models.")
    from analysis.market_features import tickers_in_text, TICKER_NAMES

    stories = session.query(Story).all()
    rows = []
    for s in stories:
        for tk in tickers_in_text(s.title or ""):
            rows.append({"ticker": tk, "name": TICKER_NAMES.get(tk, tk),
                         "sentiment": s.sentiment or 0.0, "date": s.timestamp.date(),
                         "title": s.title})
    if not rows:
        st.info("No ticker-tagged headlines yet.")
    else:
        sdf = pd.DataFrame(rows)
        cL, cR = st.columns(2)
        with cL:
            st.subheader("Average sentiment by asset")
            agg = sdf.groupby("name")["sentiment"].mean().reset_index().sort_values("sentiment")
            agg["clr"] = agg["sentiment"].apply(lambda v: GOOD if v > 0 else BAD)
            figs = go.Figure(go.Bar(
                x=agg["sentiment"], y=agg["name"], orientation="h",
                marker_color=agg["clr"],
                hovertemplate="%{y}: %{x:+.2f}<extra></extra>"))
            figs.update_layout(title="Mean headline sentiment", xaxis_title="Sentiment")
            st.plotly_chart(style_fig(figs, height=380, legend=False), use_container_width=True)
        with cR:
            st.subheader("Buzz (mention volume) by asset")
            buzz = sdf.groupby("name").size().reset_index(name="mentions").sort_values("mentions")
            figb = go.Figure(go.Bar(
                x=buzz["mentions"], y=buzz["name"], orientation="h",
                marker_color=ACCENT,
                hovertemplate="%{y}: %{x} mentions<extra></extra>"))
            figb.update_layout(title="Total mentions", xaxis_title="Headlines")
            st.plotly_chart(style_fig(figb, height=380, legend=False), use_container_width=True)

        st.subheader("Sentiment over time (top-buzz assets)")
        top_names = sdf.groupby("name").size().nlargest(5).index.tolist()
        ts = (sdf[sdf["name"].isin(top_names)]
              .groupby(["date", "name"])["sentiment"].mean().reset_index())
        ts["date"] = pd.to_datetime(ts["date"])
        figt = px.line(ts, x="date", y="sentiment", color="name", markers=True)
        figt.add_hline(y=0, line_dash="dash", line_color=INK2)
        figt.update_layout(title="Daily mean sentiment", yaxis_title="Sentiment")
        st.plotly_chart(style_fig(figt, height=360), use_container_width=True)

        cB1, cB2 = st.columns(2)
        with cB1:
            st.subheader("🟢 Most bullish headlines")
            for _, r in sdf.nlargest(6, "sentiment").iterrows():
                st.markdown(f"<small style='color:{GOOD}'>+{r['sentiment']:.2f}</small> "
                            f"{r['title']}", unsafe_allow_html=True)
        with cB2:
            st.subheader("🔴 Most bearish headlines")
            for _, r in sdf.nsmallest(6, "sentiment").iterrows():
                st.markdown(f"<small style='color:{BAD}'>{r['sentiment']:.2f}</small> "
                            f"{r['title']}", unsafe_allow_html=True)

# ══ TAB 4: PRICE & CORRELATION ════════════════════════════════════════════════
with tab_price:
    st.header("📈 Price & Sentiment Correlation")
    from analysis.market_features import TICKER_NAMES, tickers_in_text

    md = session.query(MarketData).all()
    if not md:
        st.info("No price data. Run seed_data.py or the collector.")
    else:
        pdf = pd.DataFrame([{"ticker": m.ticker, "date": m.date, "close": m.close,
                             "return_pct": m.return_pct} for m in md])
        pdf["name"] = pdf["ticker"].map(TICKER_NAMES).fillna(pdf["ticker"])
        pdf["date"] = pd.to_datetime(pdf["date"])
        names = sorted(pdf["name"].unique())
        pick = st.selectbox("Asset", names, index=0)
        sub = pdf[pdf["name"] == pick].sort_values("date")

        # price line
        figp = go.Figure(go.Scatter(x=sub["date"], y=sub["close"], mode="lines",
                                    line=dict(color=ACCENT, width=2), name="Close"))
        figp.update_layout(title=f"{pick} — closing price", yaxis_title="Price")
        st.plotly_chart(style_fig(figp, height=320, legend=False), use_container_width=True)

        # overlay daily sentiment for this ticker
        tk = sub["ticker"].iloc[0]
        srows = [{"date": s.timestamp.date(), "sentiment": s.sentiment or 0.0}
                 for s in session.query(Story).all() if tk in tickers_in_text(s.title or "")]
        if srows:
            sd = pd.DataFrame(srows).groupby("date")["sentiment"].mean().reset_index()
            sd["date"] = pd.to_datetime(sd["date"])
            merged = sub.merge(sd, on="date", how="inner")
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Daily sentiment")
                figsd = go.Figure(go.Bar(
                    x=sd["date"], y=sd["sentiment"],
                    marker_color=[GOOD if v > 0 else BAD for v in sd["sentiment"]]))
                figsd.update_layout(title=f"{pick} sentiment", yaxis_title="Sentiment")
                st.plotly_chart(style_fig(figsd, height=300, legend=False), use_container_width=True)
            with c2:
                st.subheader("Sentiment → next-day return")
                m2 = merged.copy()
                m2["next_return"] = m2["return_pct"].shift(-1)
                m2 = m2.dropna(subset=["next_return"])
                if len(m2) > 2:
                    figsc = px.scatter(m2, x="sentiment", y="next_return",
                                       trendline="ols" if _has_statsmodels() else None,
                                       color_discrete_sequence=[ACCENT])
                    figsc.update_layout(title="Does today's mood predict tomorrow?",
                                        xaxis_title="Sentiment today",
                                        yaxis_title="Next-day return %")
                    st.plotly_chart(style_fig(figsc, height=300, legend=False), use_container_width=True)
                    corr = m2["sentiment"].corr(m2["next_return"])
                    st.caption(f"Correlation (this asset): **{corr:+.2f}** — "
                               "positive means bullish headlines tend to precede gains.")

# ══ TAB 5: AI BRIEFING ════════════════════════════════════════════════════════
with tab_ai:
    st.header("🧠 AI Market Briefing")
    st.caption("Claude reads the live signals + sentiment and writes an analyst briefing.")
    from analysis.trend_summarizer import summarize_market_signals

    if not os.getenv("ANTHROPIC_API_KEY"):
        st.warning("⚠️ Set `ANTHROPIC_API_KEY` in your `.env` to enable AI briefings.")
    else:
        lab = get_lab()
        if lab["ok"] and st.button("📝 Generate briefing", type="primary"):
            with st.spinner("Claude is analyzing the signals…"):
                note = f"Overall market sentiment is {avg_sent:+.2f} across {n_stories} headlines."
                text = summarize_market_signals(lab["live"], lab["leaderboard"], note)
            st.markdown("---")
            st.markdown(text)
            st.caption(f"Generated {datetime.now():%Y-%m-%d %H:%M}")

# ══ TAB 6: PIPELINE ═══════════════════════════════════════════════════════════
with tab_pipeline:
    st.header("🔧 Pipeline Monitor")
    runs = session.query(PipelineRun).order_by(PipelineRun.started_at.desc()).limit(20).all()
    if runs:
        rdf = pd.DataFrame([{
            "Started": r.started_at.strftime("%Y-%m-%d %H:%M"),
            "Status": r.status, "Sources": r.sources_run,
            "Stories": r.stories_collected, "Entities": r.keywords_extracted,
            "Duration (s)": round((r.finished_at - r.started_at).total_seconds(), 1)
            if r.finished_at else "—",
        } for r in runs])
        st.dataframe(rdf, use_container_width=True, hide_index=True)
    else:
        st.info("No pipeline runs recorded. Run `python test_hn_api.py`.")

st.divider()
st.caption(f"Last updated {datetime.now():%Y-%m-%d %H:%M:%S} · TrendFlow v4.0 — "
           "Sentiment→Market edition")
