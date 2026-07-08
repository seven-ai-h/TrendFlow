import streamlit as st
import pandas as pd
import numpy as np
from database.db_setup import getSession
from database.models import Story, Keyword, Article, PipelineRun
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
tab_keywords, tab_trending, tab_predictions, tab_platform, tab_stories, tab_insights, tab_pipeline = st.tabs([
    "📊 Keywords", "⚡ Trending Now", "🤖 AI Predictions", "🌐 Platform Split",
    "📰 Stories & News", "🧠 AI Insights", "🔧 Pipeline"
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
    st.header("🤖 ML-Powered Trend Predictions")
    st.markdown(
        "**Gradient Boosting classifier** trained on a rich feature matrix: "
        "cross-source scores, exponential moving averages (3h/6h/24h), "
        "velocity, acceleration, and platform diversity."
    )

    from analysis.trend_predictor import load_or_train_model, predict_trending_keywords, get_feature_importances

    with st.spinner("Loading / training prediction model…"):
        model, scaler, result = load_or_train_model(session)

    if model is None:
        st.warning(f"⚠️ Could not train model: {result}")
        st.info("Run the collector a few times to build training data (needs 15+ keyword records).")
    else:
        accuracy_label = f"{result:.1%}" if isinstance(result, float) else "loaded from cache"
        st.success(f"✅ Model ready — cross-validated accuracy: **{accuracy_label}**")

        col_fi, col_info = st.columns([2, 1])
        with col_fi:
            fi_df = get_feature_importances(model)
            fig_fi = px.bar(
                fi_df, x='importance', y='feature', orientation='h',
                title='Feature Importances (Gradient Boosting)',
                color='importance', color_continuous_scale='Teal',
                labels={'importance': 'Importance', 'feature': 'Feature'}
            )
            fig_fi.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font_color='white', showlegend=False,
                yaxis={'categoryorder': 'total ascending'}
            )
            st.plotly_chart(fig_fi, use_container_width=True)
        with col_info:
            st.subheader("📐 Feature Guide")
            st.markdown("""
| Feature | Meaning |
|---|---|
| `velocity_1h` | Last-1h vs 24h baseline |
| `acceleration` | Δ velocity vs 1h ago |
| `cross_source_score` | Platform-weighted count |
| `platform_diversity` | # distinct platforms |
| `ema_3h / 6h / 24h` | Exponential moving avg |
| `count` | Raw mention count |
| `day_of_week` | Seasonality signal |
""")

        with st.spinner("Running predictions…"):
            predictions = predict_trending_keywords(session, model, scaler, top_n=15)

        if predictions:
            df_preds = pd.DataFrame(predictions)

            col_pc, col_pl = st.columns([3, 1])

            with col_pc:
                fig_pred = px.bar(
                    df_preds,
                    x='keyword', y='confidence',
                    color='confidence',
                    color_continuous_scale='Teal',
                    title='Predicted Trending Keywords — Confidence %',
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
                        f"<small>vel={pred['velocity_1h']:.2f}x · "
                        f"accel={pred['acceleration']:+.2f} · "
                        f"{pred['platform_diversity']} platforms</small>",
                        unsafe_allow_html=True
                    )
                    st.markdown("---")

            st.subheader("📊 Velocity vs. Confidence (coloured by acceleration)")
            fig_scatter = px.scatter(
                df_preds,
                x='velocity_1h', y='confidence',
                size='cross_source_score', color='acceleration',
                hover_name='keyword',
                color_continuous_scale='RdYlGn',
                title='Velocity vs. Predicted Confidence (size = cross-source score)',
                labels={'velocity_1h': 'Velocity (1h)', 'confidence': 'Confidence (%)',
                        'acceleration': 'Acceleration'}
            )
            fig_scatter.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        else:
            st.info("No keywords predicted to trend right now — run the collector to build data.")


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

# ── TAB 6: AI INSIGHTS ────────────────────────────────────────────────────────
with tab_insights:
    st.header("🧠 AI-Generated Trend Insights")
    st.markdown(
        "Claude analyzes the current feature matrix — velocity, cross-source scores, "
        "platform diversity, and acceleration — to write a real-time tech trend report."
    )

    from analysis.feature_engineer import build_feature_matrix
    from analysis.trend_summarizer import summarize_trends

    api_key_set = bool(os.getenv("ANTHROPIC_API_KEY"))
    if not api_key_set:
        st.warning("⚠️ Set `ANTHROPIC_API_KEY` in your `.env` file to enable AI summaries.")
    else:
        if st.button("🔄 Generate Trend Summary", type="primary"):
            with st.spinner("Building feature matrix and calling Claude…"):
                feat_df = build_feature_matrix(session, hours_back=48)

                kw_all = session.query(Keyword).filter(Keyword.timestamp >= cutoff_date).all()
                platform_breakdown = dict(Counter(kw.platform for kw in kw_all))

                summary = summarize_trends(feat_df, platform_breakdown)

            st.markdown("---")
            st.markdown(summary)
            st.caption(f"Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            st.info("Click **Generate Trend Summary** to call Claude and produce an analysis.")

    # Show feature matrix table regardless of API key
    st.subheader("📐 Current Feature Matrix (top 20 by velocity)")
    with st.spinner("Computing features…"):
        feat_preview = build_feature_matrix(session, hours_back=48)

    if not feat_preview.empty:
        display_cols = ['keyword', 'count', 'platform_diversity', 'cross_source_score',
                        'ema_3h', 'ema_6h', 'ema_24h', 'velocity_1h', 'acceleration']
        available = [c for c in display_cols if c in feat_preview.columns]
        top_feat = feat_preview.nlargest(20, 'velocity_1h')[available].reset_index(drop=True)

        for col in ['cross_source_score', 'ema_3h', 'ema_6h', 'ema_24h', 'velocity_1h', 'acceleration']:
            if col in top_feat.columns:
                top_feat[col] = top_feat[col].round(2)

        st.dataframe(top_feat, use_container_width=True)

        # Radar chart for top 5 entities across key dimensions
        if len(top_feat) >= 3:
            st.subheader("🕸️ Multi-Dimensional Entity Comparison (top 5)")
            radar_cols = ['velocity_1h', 'platform_diversity', 'cross_source_score', 'acceleration']
            radar_cols = [c for c in radar_cols if c in top_feat.columns]
            top5 = top_feat.head(5)

            fig_radar = go.Figure()
            for _, row in top5.iterrows():
                values = [max(0, row[c]) for c in radar_cols]
                values_norm = []
                for i, c in enumerate(radar_cols):
                    col_max = top_feat[c].abs().max()
                    values_norm.append(values[i] / col_max if col_max > 0 else 0)
                fig_radar.add_trace(go.Scatterpolar(
                    r=values_norm + [values_norm[0]],
                    theta=radar_cols + [radar_cols[0]],
                    fill='toself',
                    name=str(row['keyword'])[:20],
                    opacity=0.7,
                ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                paper_bgcolor='rgba(0,0,0,0)', font_color='white',
                title='Normalized Feature Comparison (top 5 entities)',
            )
            st.plotly_chart(fig_radar, use_container_width=True)
    else:
        st.info("No feature data available yet — run the collector first.")


# ── TAB 7: PIPELINE MONITOR ───────────────────────────────────────────────────
with tab_pipeline:
    st.header("🔧 Data Pipeline Monitor")
    st.markdown("Observability for each collection run — tracks sources, throughput, and errors.")

    try:
        recent_runs = session.query(PipelineRun).order_by(
            PipelineRun.started_at.desc()
        ).limit(20).all()
    except Exception:
        recent_runs = []

    if recent_runs:
        runs_data = []
        for r in recent_runs:
            duration = (
                (r.finished_at - r.started_at).total_seconds()
                if r.finished_at else None
            )
            runs_data.append({
                'Started': r.started_at.strftime('%Y-%m-%d %H:%M'),
                'Status': r.status or '—',
                'Sources': r.sources_run or '—',
                'New Stories': r.stories_collected or 0,
                'Entities': r.keywords_extracted or 0,
                'Duration (s)': round(duration, 1) if duration else '—',
                'Error': r.error_message or '',
            })
        df_runs = pd.DataFrame(runs_data)
        st.dataframe(df_runs, use_container_width=True)

        # Story collection rate chart
        if len(runs_data) > 1:
            df_chart = pd.DataFrame([
                {'run': r.started_at, 'stories': r.stories_collected or 0,
                 'entities': r.keywords_extracted or 0}
                for r in recent_runs if r.finished_at
            ])
            if not df_chart.empty:
                fig_runs = px.line(
                    df_chart.melt(id_vars='run', value_vars=['stories', 'entities']),
                    x='run', y='value', color='variable',
                    title='Stories & Entities Collected per Run',
                    markers=True,
                    labels={'run': 'Run time', 'value': 'Count', 'variable': 'Metric'}
                )
                fig_runs.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white'
                )
                st.plotly_chart(fig_runs, use_container_width=True)
    else:
        st.info("No pipeline runs recorded yet. Run `python test_hn_api.py` to start a collection.")

    st.subheader("📋 Source Coverage Summary")
    col_s1, col_s2, col_s3 = st.columns(3)
    platforms = ['hackernews', 'reddit', 'devto', 'github', 'rss', 'news']
    platform_story_counts = {
        p: session.query(Story).filter(
            Story.platform == p, Story.timestamp >= cutoff_date
        ).count()
        for p in platforms
    }
    rss_count = platform_story_counts.get('rss', 0)
    news_count = platform_story_counts.get('news', 0)
    hn_count = platform_story_counts.get('hackernews', 0)
    col_s1.metric("HN Stories", hn_count)
    col_s1.metric("Reddit Posts", platform_story_counts.get('reddit', 0))
    col_s2.metric("Dev.to Articles", platform_story_counts.get('devto', 0))
    col_s2.metric("GitHub Repos", platform_story_counts.get('github', 0))
    col_s3.metric("RSS Articles", rss_count)
    col_s3.metric("News (API)", news_count)


# Footer
st.divider()
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | TrendFlow v3.0")
