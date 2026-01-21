import streamlit as st
import pandas as pd
from database.db_setup import getSession
from database.models import Story, Keyword, Article
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

st.set_page_config(page_title="TrendFlow", page_icon="📊", layout="wide")

# Simple clean CSS
st.markdown("""
<style>
    .main {background-color: #fafafa; padding: 2rem;}
    .stPlotlyChart {background: white; padding: 1rem; border-radius: 8px;}
    div[data-testid="stMetric"] {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("⚙️ Controls")
days_back = st.sidebar.slider("Time Range (days)", 1, 365, 7)
min_count = st.sidebar.slider("Min Mentions", 1, 50, 2)
st.sidebar.divider()
st.sidebar.caption(f"Updated: {datetime.now().strftime('%H:%M')}")

# Get data
session = getSession()
cutoff = datetime.utcnow() - timedelta(days=days_back)

# Header
st.title("🔥 TrendFlow Analytics")
st.markdown("Real-time trend detection across Hacker News and news outlets")
st.markdown("---")

# Metrics
col1, col2, col3, col4 = st.columns(4)
stories = session.query(Story).filter(Story.timestamp >= cutoff).count()
keywords = session.query(Keyword).filter(Story.timestamp >= cutoff).count()
articles = session.query(Article).filter(Article.timestamp >= cutoff).count()

col1.metric("Stories", f"{stories:,}")
col2.metric("Keywords", f"{keywords:,}")
col3.metric("Articles", f"{articles:,}")
col4.metric("Range", f"{days_back}d")

st.markdown("---")

# Main content - 2 columns
left, right = st.columns([2, 1])

with left:
    st.subheader("📊 Top Keywords")
    
    kw_data = session.query(Keyword).filter(
        Keyword.timestamp >= cutoff,
        Keyword.count >= min_count
    ).order_by(Keyword.count.desc()).limit(15).all()
    
    if kw_data:
        df = pd.DataFrame([
            {'Keyword': k.keyword, 'Mentions': k.count, 'Platform': k.platform}
            for k in kw_data
        ])
        
        fig = px.bar(
            df, 
            x='Keyword', 
            y='Mentions', 
            color='Platform',
            color_discrete_map={'hackernews': '#FF6B35', 'news': '#4ECDC4'},
            height=400
        )
        fig.update_layout(
            plot_bgcolor='white',
            xaxis_title="",
            yaxis_title="Mentions",
            margin=dict(l=40, r=20, t=20, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data - run the collector!")

with right:
    st.subheader("🔥 Top 10")
    
    if kw_data:
        for i, kw in enumerate(kw_data[:10], 1):
            pct = (kw.count / kw_data[0].count) * 100
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            
            st.markdown(f"""
            <div style='background:white; padding:12px; margin-bottom:8px; border-radius:6px;'>
                <div style='margin-bottom:4px;'>
                    <strong>{medal} {kw.keyword}</strong>
                    <span style='float:right; color:#666;'>{kw.count}</span>
                </div>
                <div style='background:#eee; height:6px; border-radius:3px;'>
                    <div style='background:#FF6B35; width:{pct}%; height:100%; border-radius:3px;'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")

# Timeline
st.subheader("📈 Trends Over Time")

all_kw = session.query(Keyword).filter(Keyword.timestamp >= cutoff).all()

if all_kw and 'df' in locals():
    timeline = pd.DataFrame([
        {'date': k.timestamp.date(), 'keyword': k.keyword, 'count': k.count}
        for k in all_kw
    ])
    
    top5 = df.head(5)['Keyword'].tolist()
    filtered = timeline[timeline['keyword'].isin(top5)]
    grouped = filtered.groupby(['date', 'keyword'])['count'].sum().reset_index()
    
    if len(grouped) > 0:
        fig2 = px.line(grouped, x='date', y='count', color='keyword', markers=True, height=350)
        fig2.update_layout(
            plot_bgcolor='white',
            xaxis_title="Date",
            yaxis_title="Mentions",
            margin=dict(l=40, r=20, t=20, b=40)
        )
        st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("Timeline needs multiple collections")

st.markdown("---")

# Recent Stories
st.subheader("📰 Top Stories")

stories_list = session.query(Story).filter(
    Story.timestamp >= cutoff
).order_by(Story.score.desc()).limit(10).all()

if stories_list:
    for s in stories_list:
        col1, col2 = st.columns([1, 10])
        with col1:
            st.metric("↑", s.score)
        with col2:
            st.markdown(f"**{s.title}**")
            st.caption(f"💬 {s.num_comments} comments • {s.timestamp.strftime('%b %d, %H:%M')}")
        st.divider()