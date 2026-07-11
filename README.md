# 🔥 TrendFlow - Real-Time Trend Detection System

A multi-platform trend detection system that identifies and tracks emerging topics across Hacker News and major news outlets in real-time.

[![Live Demo](https://img.shields.io/badge/Demo-Live-success)](your-streamlit-url-here)
[![Python](https://img.shields.io/badge/Python-3.9+-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 🚀 Live Demo

**[View Live Dashboard →](your-streamlit-url-here)**

## 📸 Screenshots

![Dashboard Overview](link-to-screenshot)

## ✨ Features

- **📊 Multi-Platform Data Collection** - Aggregates content from Hacker News API and NewsAPI
- **🔍 Intelligent Keyword Extraction** - NLP-powered extraction with custom stopword filtering using NLTK
- **📈 Trend Velocity Tracking** - Identifies keywords with highest growth rates over time
- **🔄 Cross-Platform Analysis** - Tracks how trends migrate between tech communities and mainstream news
- **📉 Time-Series Visualization** - Interactive Plotly charts showing trend evolution
- **🤖 Automated Pipeline** - Scheduled hourly data collection and processing
- **⚡ Real-Time Dashboard** - Live Streamlit interface with interactive filters

## 🛠️ Tech Stack

**Backend & Data Processing:**
- Python 3.9+
- SQLAlchemy (ORM)
- SQLite Database
- NLTK (Natural Language Processing)
- Pandas & NumPy (Data manipulation)
- Schedule (Automation)

**APIs:**
- Hacker News Firebase API
- NewsAPI

**Frontend & Visualization:**
- Streamlit
- Plotly (Interactive charts)

**Machine Learning:**
- scikit-learn (Trend prediction - in development)

## 📊 Project Architecture
```
┌─────────────────────────────────────────┐
│     Data Sources                        │
│  (Hacker News API, NewsAPI)            │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│   Data Collection Pipeline              │
│   (Automated hourly collection)         │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│   Keyword Extraction (NLTK)             │
│   (Text processing & filtering)         │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│   SQLite Database                       │
│   (Stories, Keywords, Articles)         │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│   Analysis & Visualization              │
│   (Streamlit + Plotly)                  │
└─────────────────────────────────────────┘
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- pip package manager
- NewsAPI key ([Get free key](https://newsapi.org))

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/seven-ai-h/TrendFlow.git
cd TrendFlow
```

2. **Create and activate virtual environment**
```bash
python -m venv venv

# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download required NLTK data**
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt_tab')"
```

5. **Set up environment variables**

Create a `.env` file in the root directory:
```
NEWS_API_KEY=your_newsapi_key_here
```

6. **Initialize database and collect initial data**
```bash
python test_hn_api.py
```

### Running the Dashboard
```bash
streamlit run dashboard.py
```

Open your browser to `http://localhost:8501`

### Automated Data Collection

To run the collector continuously (collects data every hour):
```bash
python run_collector.py
```

## 📂 Project Structure
```
TrendFlow/
├── dashboard.py              # Streamlit dashboard
├── test_hn_api.py           # Main data collection script
├── run_collector.py         # Automated scheduler
├── database/
│   ├── models.py            # SQLAlchemy models
│   └── db_setup.py          # Database configuration
├── data_collection/
│   └── news_collector.py    # NewsAPI integration
├── analysis/
│   ├── keyword_extractor.py # NLP keyword extraction
│   ├── trend_detector.py    # Trend detection algorithms
│   └── trend_predictor.py   # ML prediction model
├── requirements.txt          # Python dependencies
└── .env                     # Environment variables (not in repo)
```

## 🎯 How It Works

1. **Data Collection**: Every hour, the system fetches the top 100 stories from Hacker News
2. **Keyword Extraction**: NLTK processes titles to extract meaningful keywords, filtering out stopwords
3. **News Correlation**: Top keywords are used to search NewsAPI for related mainstream news
4. **Trend Analysis**: The system calculates velocity (growth rate) by comparing current mentions to historical baselines
5. **Visualization**: Streamlit dashboard displays trends with interactive Plotly charts

## 📈 Key Metrics Tracked

- **Keyword Frequency**: Total mentions across all platforms
- **Trend Velocity**: Growth rate compared to previous time periods
- **Cross-Platform Presence**: Keywords appearing on both HN and news outlets
- **Temporal Patterns**: How keyword frequency changes over time
- **Engagement Metrics**: Story scores and comment counts

## 🔮 Future Enhancements

- [ ] **AWS Deployment**: Migrate to cloud infrastructure (Lambda + RDS)
- [ ] **Enhanced ML Model**: Improve prediction accuracy with more training data
- [ ] **Sentiment Analysis**: Determine positive/negative sentiment for trending topics
- [ ] **Alert System**: Email/Discord notifications for emerging trends
- [ ] **Keyword Clustering**: Group related keywords using NLP similarity
- [ ] **Export Features**: PDF/CSV report generation
- [ ] **User Authentication**: Save custom filters and preferences
- [ ] **Additional Data Sources**: Reddit, Twitter/X integration

## 🎓 What I Learned

This project helped me develop skills in:

- **Data Engineering**: Building ETL pipelines, database design, data normalization
- **API Integration**: Working with RESTful APIs, rate limiting, error handling
- **Natural Language Processing**: Text cleaning, tokenization, keyword extraction
- **Data Visualization**: Creating interactive dashboards with Plotly and Streamlit
- **Automation**: Scheduled tasks, background processes, logging
- **Software Architecture**: Modular design, separation of concerns, scalability planning

## 📝 License

MIT License - feel free to use this project for learning or portfolio purposes!

## 👤 Author

**Hiro**
- GitHub: [@seven-ai-h](https://github.com/seven-ai-h)
- LinkedIn: [www.linkedin.com/in/hiro-xiangyuan]

## 🙏 Acknowledgments

- Hacker News for their excellent API
- NewsAPI for news data access
- Streamlit community for amazing documentation
- NLTK for NLP tools

---

⭐ **If you found this project helpful, please consider giving it a star!**