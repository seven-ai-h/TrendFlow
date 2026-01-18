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

# Custom CSS for better styling
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
    .trend-up {
        color: #51CF66;
    }
    .trend-down {
        color: #FF6B6B;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("🔥 TrendFlow - Real-Time Trend Detection")
st.markdown("*Tracking trends across Hacker News and major news outlets*")

# Sidebar
st.sidebar.header("⚙️ Settings")
days_back = st.sidebar.slider("📅 Time Range (days)", 1, 30, 7)
min_keyword_count = st.sidebar.slider("🔢 Min Keyword Frequency", 1, 10, 2)
refresh = st.sidebar.button("🔄 Refresh Data")

cutoff_date = datetime.utcnow() - timedelta(days=days_back)

# Get session
session = getSession()

# === METRICS ROW ===
col1, col2, col3, col4 = st.columns(4)

total_stories = session.query(Story).filter(Story.timestamp >= cutoff_date).count()
total_keywords = session.query(Keyword).filter(Keyword.timestamp >= cutoff_date).count()
total_articles = session.query(Article).filter(Article.timestamp >= cutoff_date).count()

# Calculate growth (compare to previous period)
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

# Get all keywords with timestamps
all_keywords = session.query(Keyword).filter(
    Keyword.timestamp >= cutoff_date
).all()

if all_keywords:
    # Create timeline data
    timeline_data = []
    for kw in all_keywords:
        timeline_data.append({
            'date': kw.timestamp.date(),
            'keyword': kw.keyword,
            'count': kw.count
        })
    
    df_timeline = pd.DataFrame(timeline_data)
    
    # Get top 5 keywords to plot
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

# === CROSS-PLATFORM COMPARISON ===
st.header("🔄 Cross-Platform Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Hacker News Keywords")
    hn_keywords = session.query(Keyword).filter(
        Keyword.platform == 'hackernews',
        Keyword.timestamp >= cutoff_date
    ).order_by(Keyword.count.desc()).limit(10).all()
    
    if hn_keywords:
        for kw in hn_keywords:
            st.markdown(f"• **{kw.keyword}** ({kw.count})")
    else:
        st.info("No HN data")

with col2:
    st.subheader("News Keywords")
    # Extract keywords from news article titles
    news_articles = session.query(Article).filter(
        Article.timestamp >= cutoff_date
    ).all()
    
    if news_articles:
        from analysis.keyword_extractor import extract_keywords
        news_keywords = []
        for article in news_articles:
            keywords = extract_keywords(article.title, top_n=5)
            news_keywords.extend(keywords)
        
        news_counter = Counter(news_keywords).most_common(10)
        for keyword, count in news_counter:
            st.markdown(f"• **{keyword}** ({count})")
    else:
        st.info("No news data")

st.divider()

# === TREND VELOCITY ===
st.header("🚀 Trending Now (Velocity)")

st.markdown("*Keywords with fastest growth compared to previous period*")

# Get current period keywords
current_keywords = {}
for kw in session.query(Keyword).filter(Keyword.timestamp >= cutoff_date).all():
    current_keywords[kw.keyword] = current_keywords.get(kw.keyword, 0) + kw.count

# Get previous period keywords
prev_keywords = {}
for kw in session.query(Keyword).filter(
    Keyword.timestamp >= previous_cutoff,
    Keyword.timestamp < cutoff_date
).all():
    prev_keywords[kw.keyword] = prev_keywords.get(kw.keyword, 0) + kw.count

# Calculate velocity
velocities = []
for keyword, current_count in current_keywords.items():
    prev_count = prev_keywords.get(keyword, 0)
    if prev_count > 0:
        velocity = ((current_count - prev_count) / prev_count) * 100
    else:
        velocity = float('inf') if current_count > 0 else 0
    
    if velocity != float('inf') and current_count >= min_keyword_count:
        velocities.append({
            'keyword': keyword,
            'current': current_count,
            'previous': prev_count,
            'velocity': velocity
        })

velocities.sort(key=lambda x: x['velocity'], reverse=True)

if velocities:
    col1, col2, col3 = st.columns(3)
    
    for i, v in enumerate(velocities[:9]):
        col = [col1, col2, col3][i % 3]
        with col:
            trend_emoji = "📈" if v['velocity'] > 0 else "📉"
            color = "trend-up" if v['velocity'] > 0 else "trend-down"
            st.markdown(f"""
            <div style='padding: 10px; background-color: #1E1E1E; border-radius: 5px; margin-bottom: 10px;'>
                {trend_emoji} <strong>{v['keyword']}</strong><br/>
                <span class='{color}'>{v['velocity']:+.0f}%</span> velocity<br/>
                <small>{v['previous']} → {v['current']} mentions</small>
            </div>
            """, unsafe_allow_html=True)
else:
    st.info("🚀 Velocity data will appear after multiple collections over time")

st.divider()

# === RECENT STORIES ===
st.header("📰 Recent Hacker News Stories")

recent_stories = session.query(Story).filter(
    Story.timestamp >= cutoff_date
).order_by(Story.score.desc()).limit(10).all()

for story in recent_stories:
    with st.expander(f"⬆️ {story.score} | 💬 {story.num_comments} | {story.title}"):
        col1, col2 = st.columns([3, 1])
        with col1:
            if story.url:
                st.markdown(f"[🔗 Read Article]({story.url})")
        with col2:
            st.caption(f"Posted: {story.timestamp.strftime('%Y-%m-%d %H:%M')}")

st.divider()

# === RECENT NEWS ===
st.header("📄 Related News Articles")

recent_articles = session.query(Article).filter(
    Article.timestamp >= cutoff_date
).order_by(Article.timestamp.desc()).limit(10).all()

for article in recent_articles:
    with st.expander(f"📰 {article.source} | {article.title}"):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"[🔗 Read Full Article]({article.url})")
        with col2:
            st.caption(f"Published: {article.published_at.strftime('%Y-%m-%d')}")

# Footer
st.divider()
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | TrendFlow v1.0")