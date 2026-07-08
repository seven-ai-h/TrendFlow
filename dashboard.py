import streamlit as st
import pandas as pd
import numpy as np
from database.db_setup import getSession
from database.models import Story, Keyword, Article
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter, defaultdict
import os

if not os.path.exists('trendflow.db'):
    st.warning("⚠️ No data available. This is a demo deployment.")
    st.info("To see live data, run the collector locally and connect to a cloud database.")
    st.stop()

st.set_page_config(
    page_title="TrendFlow Dashboard",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { padding: 0rem 1rem; }
    .stMetric {
        background-color: #1E1E1E;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #333;
    }
    h1 { color: #FF6B6B; padding-bottom: 20px; }
    h2 { color: #4ECDC4; padding-top: 20px; }
</style>
""", unsafe_allow_html=True)

st.title("🔥 TrendFlow - Real-Time Trend Detection")
st.markdown("*Tracking trends across Hacker News and major news outlets*")

# Sidebar
st.sidebar.header("⚙️ Settings")
days_back = st.sidebar.slider("📅 Time Range (days)", 1, 365, 7)
min_keyword_count = st.sidebar.slider("🔢 Min Keyword Frequency", 1, 100, 2)
velocity_threshold = st.sidebar.slider("⚡ Velocity Threshold", 0.5, 10.0, 2.0, 0.5)
refresh = st.sidebar.button("🔄 Refresh Data")

cutoff_date = datetime.utcnow() - timedelta(days=days_back)
session = getSession()

# === METRICS ROW ===
col1, col2, col3, col4 = st.columns(4)

total_stories = session.query(Story).filter(Story.timestamp >= cutoff_date).count()
total_keywords = session.query(Keyword).filter(Keyword.timestamp >= cutoff_date).count()
total_articles = session.query(Article).filter(Article.timestamp >= cutoff_date).count()

previous_cutoff = cutoff_date - timedelta(days=days_back)
prev_stories = session.query(Story).filter(
    Story.timestamp >= previous_cutoff,
    Story.timestamp < cutoff_date
).count()
growth = ((total_stories - prev_stories) / prev_stories * 100) if prev_stories > 0 else 0

col1.metric("📰 Total Stories", f"{total_stories:,}", f"{growth:+.1f}%")
col2.metric("🏷️ Unique Keywords", f"{total_keywords:,}")
col3.metric("📄 News Articles", f"{total_articles:,}")
col4.metric("⏱️ Data Range", f"{days_back} days")

st.divider()

# === TABS ===
tab_keywords, tab_trending, tab_predictions, tab_platform, tab_stories = st.tabs([
    "📊 Keywords", "⚡ Trending Now", "🤖 AI Predictions", "🌐 Platform Split", "📰 Stories & News"
])

# ── TAB 1: TOP KEYWORDS ───────────────────────────────────────────────────────
with tab_keywords:
    st.header("📊 Top Trending Keywords")

    keywords_data = session.query(Keyword).filter(
        Keyword.timestamp >= cutoff_date,
        Keyword.count >= min_keyword_count
    ).order_by(Keyword.count.desc()).limit(20).all()

    col_chart, col_list = st.columns([2, 1])

    with col_chart:
        if keywords_data:
            df_keywords = pd.DataFrame([
                {'Keyword': kw.keyword, 'Count': kw.count, 'Platform': kw.platform}
                for kw in keywords_data
            ])
            fig = px.bar(
                df_keywords, x='Keyword', y='Count', color='Platform',
                title='Top 20 Keywords by Frequency',
                color_discrete_map={'hackernews': '#FF6B35', 'news': '#4ECDC4', 'reddit': '#FF4500', 'devto': '#3B49DF', 'github': '#6E40C9'}
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 No keyword data available yet. Run the collector!")

    with col_list:
        st.subheader("🔥 Hot Keywords")
        if keywords_data:
            for i, kw in enumerate(keywords_data[:10], 1):
                emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "🔹"
                st.markdown(f"{emoji} **{kw.keyword}** — {kw.count} mentions")
        else:
            st.info("No data yet")

    # Keyword timeline
    st.subheader("📈 Keyword Trends Over Time")
    all_keywords = session.query(Keyword).filter(Keyword.timestamp >= cutoff_date).all()

    if all_keywords and keywords_data:
        timeline_data = [{'date': kw.timestamp.date(), 'keyword': kw.keyword, 'count': kw.count}
                         for kw in all_keywords]
        df_timeline = pd.DataFrame(timeline_data)
        top_5 = df_keywords.head(5)['Keyword'].tolist()
        df_filtered = df_timeline[df_timeline['keyword'].isin(top_5)]
        df_grouped = df_filtered.groupby(['date', 'keyword'])['count'].sum().reset_index()

        fig_tl = px.line(
            df_grouped, x='date', y='count', color='keyword',
            title='Top 5 Keywords Over Time', markers=True
        )
        fig_tl.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font_color='white', xaxis_title="Date", yaxis_title="Mentions"
        )
        st.plotly_chart(fig_tl, use_container_width=True)

    # Keyword heatmap (top 10 keywords × day of week)
    st.subheader("🗓️ Keyword Activity Heatmap")
    if all_keywords and keywords_data:
        top_10_kws = df_keywords.head(10)['Keyword'].tolist()
        heat_rows = []
        for kw in all_keywords:
            if kw.keyword in top_10_kws:
                heat_rows.append({
                    'keyword': kw.keyword,
                    'day': kw.timestamp.strftime('%a'),
                    'count': kw.count
                })
        if heat_rows:
            df_heat = pd.DataFrame(heat_rows)
            df_heat_pivot = df_heat.groupby(['keyword', 'day'])['count'].sum().unstack(fill_value=0)
            day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            df_heat_pivot = df_heat_pivot.reindex(
                columns=[d for d in day_order if d in df_heat_pivot.columns]
            )
            fig_heat = px.imshow(
                df_heat_pivot,
                color_continuous_scale='Oranges',
                title='Keyword Frequency by Day of Week',
                aspect='auto'
            )
            fig_heat.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', font_color='white'
            )
            st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("📅 Heatmap will appear after multiple collection runs")


# ── TAB 2: TRENDING NOW ───────────────────────────────────────────────────────
with tab_trending:
    st.header("⚡ Currently Trending")
    st.markdown("Keywords with the highest velocity compared to their 7-day baseline.")

    from analysis.trend_detector import detect_trending_keywords

    with st.spinner("Calculating trend velocities…"):
        trending = detect_trending_keywords(session, velocity_threshold=velocity_threshold)

    if trending:
        df_trending = pd.DataFrame(trending)
        df_trending['velocity_display'] = df_trending['velocity'].apply(
            lambda v: "∞" if v == float('inf') else f"{v:.1f}×"
        )
        df_trending['velocity_num'] = df_trending['velocity'].apply(
            lambda v: 999 if v == float('inf') else v
        )

        col_tv, col_tl = st.columns([3, 1])

        with col_tv:
            fig_vel = px.bar(
                df_trending.head(15),
                x='keyword', y='velocity_num',
                title=f'Keyword Velocity (threshold: {velocity_threshold}×)',
                color='velocity_num',
                color_continuous_scale='Reds',
                labels={'velocity_num': 'Velocity', 'keyword': 'Keyword'}
            )
            fig_vel.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font_color='white', showlegend=False
            )
            st.plotly_chart(fig_vel, use_container_width=True)

        with col_tl:
            st.subheader("🚀 Top Movers")
            for row in trending[:10]:
                v = row['velocity']
                label = "∞" if v == float('inf') else f"{v:.1f}×"
                st.markdown(f"**{row['keyword']}** `{label}`  \n"
                            f"<small>{row['recent_count']} now vs {row['baseline_count']} baseline</small>",
                            unsafe_allow_html=True)
                st.markdown("---")
    else:
        st.info(f"No keywords exceeding {velocity_threshold}× velocity right now. "
                "Lower the threshold in the sidebar or wait for more data.")


# ── TAB 3: AI PREDICTIONS ─────────────────────────────────────────────────────
with tab_predictions:
    st.header("🤖 AI-Powered Trend Predictions")
    st.markdown("Random Forest classifier trained on historical keyword velocity patterns "
                "to predict which topics will surge in the next 24–48 hours.")

    from analysis.trend_predictor import load_or_train_model, predict_trending_keywords

    with st.spinner("Loading / training prediction model…"):
        model, result = load_or_train_model(session)

    if model is None:
        st.warning(f"⚠️ Could not train model: {result}")
        st.info("The model needs at least 10 data points (roughly 3+ days of collection runs).")
    else:
        accuracy_label = f"{result:.1%}" if isinstance(result, float) else "cached"
        st.success(f"✅ Model ready — accuracy: **{accuracy_label}**")

        with st.spinner("Running predictions…"):
            predictions = predict_trending_keywords(session, model, top_n=15)

        if predictions:
            df_preds = pd.DataFrame(predictions)

            col_pc, col_pl = st.columns([3, 1])

            with col_pc:
                fig_pred = px.bar(
                    df_preds,
                    x='keyword', y='confidence',
                    color='confidence',
                    color_continuous_scale='Teal',
                    title='Predicted Trending Keywords (confidence %)',
                    labels={'confidence': 'Confidence (%)', 'keyword': 'Keyword'}
                )
                fig_pred.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white', showlegend=False,
                    yaxis=dict(range=[0, 100])
                )
                st.plotly_chart(fig_pred, use_container_width=True)

            with col_pl:
                st.subheader("📈 Top Predicted")
                for pred in predictions[:10]:
                    conf = pred['confidence']
                    bar_filled = int(conf / 10)
                    bar = "█" * bar_filled + "░" * (10 - bar_filled)
                    st.markdown(
                        f"**{pred['keyword']}**  \n"
                        f"`{bar}` {conf:.1f}%  \n"
                        f"<small>velocity: {pred['velocity']:.2f}</small>",
                        unsafe_allow_html=True
                    )
                    st.markdown("---")

            # Scatter: current count vs confidence
            st.subheader("📊 Count vs. Confidence")
            fig_scatter = px.scatter(
                df_preds,
                x='current_count', y='confidence',
                size='confidence', color='velocity',
                hover_name='keyword',
                color_continuous_scale='Viridis',
                title='Current Mention Count vs. Predicted Confidence',
                labels={'current_count': 'Current Mentions', 'confidence': 'Confidence (%)'}
            )
            fig_scatter.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        else:
            st.info("No keywords predicted to trend right now based on recent data.")


# ── TAB 4: PLATFORM SPLIT ─────────────────────────────────────────────────────
with tab_platform:
    st.header("🌐 Platform Analysis")

    kw_all = session.query(Keyword).filter(Keyword.timestamp >= cutoff_date).all()

    if kw_all:
        platform_counts = Counter(kw.platform for kw in kw_all)
        df_plat = pd.DataFrame(list(platform_counts.items()), columns=['Platform', 'Keywords'])

        col_pie, col_bar = st.columns(2)

        with col_pie:
            fig_pie = px.pie(
                df_plat, names='Platform', values='Keywords',
                title='Keyword Distribution by Platform',
                color_discrete_map={'hackernews': '#FF6B35', 'news': '#4ECDC4', 'reddit': '#FF4500', 'devto': '#3B49DF', 'github': '#6E40C9'}
            )
            fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_bar:
            # Top keywords per platform side-by-side
            df_kw_all = pd.DataFrame([
                {'keyword': kw.keyword, 'platform': kw.platform, 'count': kw.count}
                for kw in kw_all
            ])
            top_per_platform = (
                df_kw_all.groupby(['platform', 'keyword'])['count'].sum()
                .reset_index()
                .sort_values('count', ascending=False)
                .groupby('platform').head(8)
            )
            fig_plat_bar = px.bar(
                top_per_platform, x='keyword', y='count', color='platform',
                barmode='group',
                title='Top Keywords per Platform',
                color_discrete_map={'hackernews': '#FF6B35', 'news': '#4ECDC4', 'reddit': '#FF4500', 'devto': '#3B49DF', 'github': '#6E40C9'}
            )
            fig_plat_bar.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig_plat_bar, use_container_width=True)

        # Platform activity over time
        st.subheader("📅 Platform Activity Over Time")
        df_kw_all['date'] = pd.to_datetime([kw.timestamp.date() for kw in kw_all])
        df_daily_plat = df_kw_all.groupby(['date', 'platform'])['count'].sum().reset_index()
        fig_area = px.area(
            df_daily_plat, x='date', y='count', color='platform',
            title='Daily Keyword Volume by Platform',
            color_discrete_map={'hackernews': '#FF6B35', 'news': '#4ECDC4', 'reddit': '#FF4500', 'devto': '#3B49DF', 'github': '#6E40C9'}
        )
        fig_area.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig_area, use_container_width=True)
    else:
        st.info("No keyword data available for the selected time range.")


# ── TAB 5: STORIES & NEWS ─────────────────────────────────────────────────────
with tab_stories:
    st.header("📰 Stories & News")

    col_hn, col_news = st.columns(2)

    with col_hn:
        st.subheader("🔶 Hacker News")
        recent_stories = session.query(Story).filter(
            Story.timestamp >= cutoff_date
        ).order_by(Story.score.desc()).limit(15).all()

        if recent_stories:
            for story in recent_stories:
                with st.expander(f"⬆️ {story.score} | {story.title}"):
                    st.write(f"**Comments:** {story.num_comments}")
                    st.write(f"**Posted:** {story.timestamp.strftime('%Y-%m-%d %H:%M')}")
                    if story.url:
                        st.write(f"**Link:** {story.url}")
        else:
            st.info("No stories in this time range.")

    with col_news:
        st.subheader("📡 News Articles")
        recent_articles = session.query(Article).filter(
            Article.timestamp >= cutoff_date
        ).order_by(Article.timestamp.desc()).limit(15).all()

        if recent_articles:
            for article in recent_articles:
                with st.expander(f"📰 {article.title}"):
                    st.write(f"**Source:** {article.source}")
                    st.write(f"**Published:** {article.published_at or article.timestamp.strftime('%Y-%m-%d %H:%M')}")
                    if article.url:
                        st.write(f"**Link:** {article.url}")
        else:
            st.info("No news articles in this time range. "
                    "Add a NEWS_API_KEY to .env and re-run the collector.")

# Footer
st.divider()
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | TrendFlow v2.0")
