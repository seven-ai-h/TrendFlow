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
    page_title="TrendFlow",
    page_icon="📊",
    layout="wide"
)

# Clean, minimal CSS
st.markdown("""
<style>
    /* Clean background */
    .main {
        background-color: #fafafa;
    }
    
    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Card style */
    div[data-testid="stMetric"] {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Clean metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        color: #1f1f1f;
    }
    
    [data-testid="stMetricLabel"] {
        color: #666;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# Get data
session = getSession()
days_back = st.sidebar.slider("Days", 1, 365, 7)
cutoff = datetime.utcnow() - timedelta(days=days_back)

# Header
st.title("TrendFlow")
st.caption("Real-time trend detection across platforms")
st.divider()

# Metrics
col1, col2, col3, col4 = st.columns(4)

stories = session.query(Story).filter(Story.timestamp >= cutoff).count()
keywords = session.query(Keyword).filter(Keyword.timestamp >= cutoff).count()
articles = session.query(Article).filter(Article.timestamp >= cutoff).count()

col1.metric("Stories", f"{stories:,}")
col2.metric("Keywords", f"{keywords:,}")
col3.metric("Articles", f"{articles:,}")
col4.metric("Range", f"{days_back}d")

st.divider()

# Main content
tab1, tab2, tab3 = st.tabs(["📊 Keywords", "📈 Trends", "📰 Stories"])

with tab1:
    st.subheader("Top Keywords")
    
    kw_data = session.query(Keyword).filter(
        Keyword.timestamp >= cutoff
    ).order_by(Keyword.count.desc()).limit(20).all()
    
    if kw_data:
        df = pd.DataFrame([
            {'keyword': k.keyword, 'count': k.count, 'platform': k.platform}
            for k in kw_data
        ])
        
        fig = px.bar(df, x='keyword', y='count', color='platform',
                    color_discrete_map={'hackernews': '#FF6B35', 'news': '#004E89'})
        fig.update_layout(
            plot_bgcolor='white',
            height=400,
            showlegend=True,
            xaxis_title="",
            yaxis_title="Mentions"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Table
        st.dataframe(
            df[['keyword', 'count', 'platform']].rename(columns={
                'keyword': 'Keyword',
                'count': 'Count',
                'platform': 'Platform'
            }),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No data yet. Run the collector!")

with tab2:
    st.subheader("Keyword Timeline")
    
    all_kw = session.query(Keyword).filter(Keyword.timestamp >= cutoff).all()
    
    if all_kw:
        timeline = pd.DataFrame([
            {'date': k.timestamp.date(), 'keyword': k.keyword, 'count': k.count}
            for k in all_kw
        ])
        
        if len(timeline) > 0:
            # Top 5 keywords
            top5 = df.head(5)['keyword'].tolist() if 'df' in locals() else []
            
            if top5:
                filtered = timeline[timeline['keyword'].isin(top5)]
                grouped = filtered.groupby(['date', 'keyword'])['count'].sum().reset_index()
                
                fig = px.line(grouped, x='date', y='count', color='keyword', markers=True)
                fig.update_layout(
                    plot_bgcolor='white',
                    height=400,
                    xaxis_title="Date",
                    yaxis_title="Mentions"
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Timeline needs multiple collections")

with tab3:
    st.subheader("Recent Stories")
    
    stories = session.query(Story).filter(
        Story.timestamp >= cutoff
    ).order_by(Story.score.desc()).limit(20).all()
    
    if stories:
        for s in stories:
            col1, col2, col3 = st.columns([1, 8, 2])
            col1.metric("↑", s.score)
            col2.markdown(f"**{s.title}**")
            col3.caption(s.timestamp.strftime('%m/%d %H:%M'))
            
            if s.url:
                st.caption(f"[Link]({s.url}) • {s.num_comments} comments")
            st.divider()
    else:
        st.info("No stories yet")

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.caption("TrendFlow v1.0")
st.sidebar.caption(f"Updated: {datetime.now().strftime('%H:%M')}")