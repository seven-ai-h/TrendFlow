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
    page_title="TrendFlow Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 2rem;
        background-color: #f8f9fa;
    }
    
    /* Metrics cards */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a1a;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        font-weight: 500;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Card styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
    }
    
    /* Headers */
    h1 {
        color: #1a1a1a;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    h2 {
        color: #1a1a1a;
        font-weight: 600;
        font-size: 1.3rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    h3 {
        color: #333;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }
    
    [data-testid="stSidebar"] h2 {
        color: #5a3d99;
        font-weight: 700;
    }
    
    /* Charts */
    .stPlotlyChart {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background-color: white;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        font-weight: 500;
    }
    
    /* Progress bars */
    .stProgress > div > div {
        background-color: #5a3d99;
    }
    
    /* Buttons */
    .stButton button {
        background-color: #5a3d99;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: 600;
    }
    
    .stButton button:hover {
        background-color: #7b5bb8;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 📊 TrendFlow Analytics")
    st.markdown("---")
    
    st.markdown("### ⚙️ Settings")
    days_back = st.slider("Time Range (days)", 1, 365, 7)
    min_keyword_count = st.slider("Min Keyword Frequency", 1, 100, 2)
    
    st.markdown("---")
    
    st.markdown("### 📈 Quick Stats")
    refresh = st.button("🔄 Refresh Data", use_container_width=True)
    
    st.markdown("---")
    st.caption("© 2026 TrendFlow")

# Get session
session = getSession()
cutoff_date = datetime.utcnow() - timedelta(days=days_back)

# Header
st.title("Trend Detection Dashboard")
st.markdown("Real-time analysis of trending topics across Hacker News and news outlets")
st.markdown("---")

# === TOP METRICS ROW ===
col1, col2, col3, col4 = st.columns(4)

total_stories = session.query(Story).filter(Story.timestamp >= cutoff_date).count()
total_keywords = session.query(Keyword).filter(Keyword.timestamp >= cutoff_date).count()
total_articles = session.query(Article).filter(Article.timestamp >= cutoff_date).count()

# Calculate growth
previous_cutoff = cutoff_date - timedelta(days=days_back)
prev_stories = session.query(Story).filter(
    Story.timestamp >= previous_cutoff,
    Story.timestamp < cutoff_date
).count()
growth = ((total_stories - prev_stories) / prev_stories * 100) if prev_stories > 0 else 0

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Total Stories", f"{total_stories:,}", f"{growth:+.1f}%")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Unique Keywords", f"{total_keywords:,}")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("News Articles", f"{total_articles:,}")
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Data Range", f"{days_back} days")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# === MAIN CONTENT AREA ===
col_left, col_right = st.columns([2, 1])

with col_left:
    # Top Keywords Chart
    st.markdown("## Keyword Frequency Analysis")
    
    keywords_data = session.query(Keyword).filter(
        Keyword.timestamp >= cutoff_date,
        Keyword.count >= min_keyword_count
    ).order_by(Keyword.count.desc()).limit(15).all()
    
    if keywords_data:
        df_keywords = pd.DataFrame([
            {'Keyword': kw.keyword, 'Mentions': kw.count, 'Platform': kw.platform}
            for kw in keywords_data
        ])
        
        fig = px.bar(
            df_keywords,
            x='Keyword',
            y='Mentions',
            color='Platform',
            color_discrete_map={'hackernews': '#5a3d99', 'news': '#7b5bb8'},
            height=400
        )
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Inter, sans-serif", size=12, color="#333"),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📊 No keyword data available. Run the collector to gather data!")

with col_right:
    # Top Keywords List
    st.markdown("## Top Trending")
    
    if keywords_data:
        for i, kw in enumerate(keywords_data[:10], 1):
            progress = min(kw.count / keywords_data[0].count, 1.0)
            
            st.markdown(f"""
            <div style='background: white; padding: 12px; margin-bottom: 8px; border-radius: 8px; border: 1px solid #e0e0e0;'>
                <div style='display: flex; justify-content: space-between; margin-bottom: 4px;'>
                    <span style='font-weight: 600; color: #333;'>{kw.keyword}</span>
                    <span style='color: #666; font-size: 0.9rem;'>{kw.count} mentions</span>
                </div>
                <div style='background: #f0f0f0; height: 6px; border-radius: 3px; overflow: hidden;'>
                    <div style='background: linear-gradient(90deg, #5a3d99, #7b5bb8); width: {progress*100}%; height: 100%;'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No data yet")

# === TREND TIMELINE ===
st.markdown("## Keyword Trends Over Time")

all_keywords = session.query(Keyword).filter(
    Keyword.timestamp >= cutoff_date
).all()

if all_keywords and len(all_keywords) > 10:
    timeline_data = []
    for kw in all_keywords:
        timeline_data.append({
            'date': kw.timestamp.date(),
            'keyword': kw.keyword,
            'count': kw.count
        })
    
    df_timeline = pd.DataFrame(timeline_data)
    
    if 'df_keywords' in locals() and len(df_keywords) > 0:
        top_5_keywords = df_keywords.head(5)['Keyword'].tolist()
        df_filtered = df_timeline[df_timeline['keyword'].isin(top_5_keywords)]
        df_grouped = df_filtered.groupby(['date', 'keyword'])['count'].sum().reset_index()
        
        fig_timeline = px.line(
            df_grouped,
            x='date',
            y='count',
            color='keyword',
            markers=True,
            height=350
        )
        fig_timeline.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Inter, sans-serif", size=12, color="#333"),
            xaxis=dict(showgrid=False, title="Date"),
            yaxis=dict(showgrid=True, gridcolor='#f0f0f0', title="Mentions"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_timeline, use_container_width=True)
else:
    st.info("📈 Timeline will appear after multiple collections")

# === CROSS-PLATFORM COMPARISON ===
st.markdown("## Cross-Platform Analysis")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Hacker News")
    hn_keywords = session.query(Keyword).filter(
        Keyword.platform == 'hackernews',
        Keyword.timestamp >= cutoff_date
    ).order_by(Keyword.count.desc()).limit(8).all()
    
    if hn_keywords:
        for kw in hn_keywords:
            st.markdown(f"""
            <div style='background: white; padding: 10px; margin-bottom: 6px; border-radius: 6px; border-left: 3px solid #5a3d99;'>
                <strong>{kw.keyword}</strong> <span style='color: #666; float: right;'>({kw.count})</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No HN data")

with col2:
    st.markdown("### News Outlets")
    news_articles = session.query(Article).filter(
        Article.timestamp >= cutoff_date
    ).all()
    
    if news_articles:
        from analysis.keyword_extractor import extract_keywords
        news_keywords = []
        for article in news_articles:
            keywords = extract_keywords(article.title, top_n=5)
            news_keywords.extend(keywords)
        
        news_counter = Counter(news_keywords).most_common(8)
        for keyword, count in news_counter:
            st.markdown(f"""
            <div style='background: white; padding: 10px; margin-bottom: 6px; border-radius: 6px; border-left: 3px solid #7b5bb8;'>
                <strong>{keyword}</strong> <span style='color: #666; float: right;'>({count})</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No news data")

# === RECENT STORIES ===
st.markdown("## Recent Stories")

recent_stories = session.query(Story).filter(
    Story.timestamp >= cutoff_date
).order_by(Story.score.desc()).limit(8).all()

for story in recent_stories:
    with st.expander(f"⬆️ {story.score} | 💬 {story.num_comments} | {story.title}"):
        col1, col2 = st.columns([3, 1])
        with col1:
            if story.url:
                st.markdown(f"[🔗 Read Article]({story.url})")
        with col2:
            st.caption(f"{story.timestamp.strftime('%Y-%m-%d %H:%M')}")

# Footer
st.markdown("---")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | TrendFlow Analytics v1.0")