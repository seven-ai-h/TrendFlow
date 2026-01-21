# 🔥 TrendFlow - Real-Time Trend Detection System

A multi-platform trend detection system that tracks and predicts emerging topics across Hacker News and major news outlets.

![Dashboard Preview](link-to-screenshot-here)

## 🚀 Live Demo

[View Live Dashboard](your-streamlit-url-here)

## ✨ Features

- **Multi-Platform Data Collection**: Aggregates content from Hacker News and NewsAPI
- **Intelligent Keyword Extraction**: NLP-powered keyword extraction with custom stopword filtering
- **Trend Velocity Tracking**: Identifies keywords with fastest growth rates
- **Cross-Platform Analysis**: Tracks how trends migrate between platforms
- **Time-Series Visualization**: Interactive charts showing trend evolution
- **Automated Pipeline**: Hourly data collection and processing
- **Prediction Model**: ML-based prediction of trending topics (in development)

## 🛠️ Tech Stack

**Backend:**
- Python 3.9+
- SQLAlchemy (ORM)
- SQLite Database
- NLTK (Natural Language Processing)
- scikit-learn (Machine Learning)

**Data Sources:**
- Hacker News API
- NewsAPI

**Frontend:**
- Streamlit
- Plotly (Interactive Visualizations)
- Pandas (Data Processing)

**Automation:**
- Schedule library for automated collection

## 📊 Architecture
```
Data Sources (HN API, NewsAPI)
          ↓
Data Collection Pipeline
          ↓
Keyword Extraction (NLTK)
          ↓
SQLite Database
          ↓
Analysis & Prediction (scikit-learn)
          ↓
Streamlit Dashboard (Plotly)
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- pip
- NewsAPI key (free at https://newsapi.org)

### Installation

1. Clone the repository
```bash
git clone https://github.com/seven-ai-h/TrendFlow.git
cd TrendFlow
```

2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Download NLTK data
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt_tab')"
```

5. Set up environment variables
```bash
# Create .env file
echo "NEWS_API_KEY=your_api_key_here" > .env
```

6. Initialize database
```bash
python test_hn_api.py
```

7. Run the dashboard
```bash
streamlit run dashboard.py
```

## 📈 Usage

### Manual Data Collection
```bash
python test_hn_api.py
```

### Automated Collection (runs every hour)
```bash
python run_collector.py
```

### View Dashboard
```bash
streamlit run dashboard.py
```
Open browser to `http://localhost:8501`

## 🎯 Project Goals

This project was built to:
- Learn data engineering pipelines
- Explore real-time data processing
- Practice API integration
- Build predictive ML models
- Create interactive data visualizations

## 📝 Future Enhancements

- [ ] Deploy to AWS for production-scale infrastructure
- [ ] Add sentiment analysis
- [ ] Improve prediction model accuracy with more data
- [ ] Add email/Discord alerts for trending keywords
- [ ] Implement Redis caching layer
- [ ] Add user authentication
- [ ] Export reports as PDF/CSV

## 📄 License

MIT License

## 👤 Author

**Hiro**
- GitHub: [@seven-ai-h](https://github.com/seven-ai-h)
- LinkedIn: [Your LinkedIn](your-linkedin-url)

## 🙏 Acknowledgments

- Hacker News API
- NewsAPI
- Streamlit Community