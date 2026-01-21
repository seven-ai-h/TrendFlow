import streamlit as st
import pandas as pd
from database.db_setup import getSession
from database.models import Story, Keyword, Article
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

# Page config
st.set_page_config(
    page_title="TrendFlow Dashboard",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #1E1E1E;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #333;
    }
    h1 {
        color: #FF6B6B;
        padding-bottom: 20px;
    }
    h2 {
        color: #4ECDC4;
        padding-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("🔥 TrendFlow - Real-Time Trend Detection")
st.markdown("*Tracking trends across Hacker News and major news outlets*")

# Sidebar
st.sidebar.header("⚙️ Settings")
days_back = st.sidebar.slider("📅 Time Range (days)", 1, 365, 7)
min_keyword_count = st.sidebar.slider("🔢 Min Keyword Frequency", 1, 100, 2)
refresh = st.sidebar.button("🔄 Refresh Data")

cutoff_date = datetime.utcnow() - timedelta(days=days_back)

# Get session
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

# === TOP TRENDING KEYWORDS ===
st.header("📊 Top Trending Keywords")

col1, col2 = st.columns([2, 1])

with col1:
    keywords_data = session.query(Keyword).filter(
        Keyword.timestamp >= cutoff_date,
        Keyword.count >= min_keyword_count
    ).order_by(Keyword.count.desc()).limit(20).all()
    
    if keywords_data:
        df_keywords = pd.DataFrame([
            {'Keyword': kw.keyword, 'Count': kw.count, 'Platform': kw.platform}
            for kw in keywords_data
        ])
        
        fig = px.bar(
            df_keywords,
            x='Keyword',
            y='Count',
            color='Platform',
            title='Top 20 Keywords by Frequency',
            color_discrete_map={'hackernews': '#FF6B35', 'news': '#4ECDC4'}
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📊 No keyword data available yet. Run the collector!")

with col2:
    st.subheader("🔥 Hot Keywords")
    if keywords_data:
        for i, kw in enumerate(keywords_data[:10], 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "🔹"
            st.markdown(f"{emoji} **{kw.keyword}** — {kw.count} mentions")
    else:
        st.info("No data yet")

st.divider()

# === KEYWORD TIMELINE ===
st.header("📈 Keyword Trends Over Time")

all_keywords = session.query(Keyword).filter(
    Keyword.timestamp >= cutoff_date
).all()

if all_keywords:
    timeline_data = []
    for kw in all_keywords:
        timeline_data.append({
            'date': kw.timestamp.date(),
            'keyword': kw.keyword,
            'count': kw.count
        })
    
    df_timeline = pd.DataFrame(timeline_data)
    
    top_5_keywords = df_keywords.head(5)['Keyword'].tolist() if 'df_keywords' in locals() else []
    
    if top_5_keywords:
        df_filtered = df_timeline[df_timeline['keyword'].isin(top_5_keywords)]
        df_grouped = df_filtered.groupby(['date', 'keyword'])['count'].sum().reset_index()
        
        fig_timeline = px.line(
            df_grouped,
            x='date',
            y='count',
            color='keyword',
            title='Top 5 Keywords Over Time',
            markers=True
        )
        fig_timeline.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            xaxis_title="Date",
            yaxis_title="Mentions"
        )
        st.plotly_chart(fig_timeline, use_container_width=True)
else:
    st.info("📈 Timeline will appear after multiple collection runs")

st.divider()

# === RECENT STORIES ===
st.header("📰 Recent Hacker News Stories")

recent_stories = session.query(Story).filter(
    Story.timestamp >= cutoff_date
).order_by(Story.timestamp.desc()).limit(10).all()

for story in recent_stories:
    with st.expander(f"⬆️ {story.score} | {story.title}"):
        st.write(f"**Comments:** {story.num_comments}")
        st.write(f"**Posted:** {story.timestamp.strftime('%Y-%m-%d %H:%M')}")
        if story.url:
            st.write(f"**Link:** {story.url}")

# Footer
st.divider()
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | TrendFlow v1.0")