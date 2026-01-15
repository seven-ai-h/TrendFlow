import requests
from database.models import Story, Keyword, Article  # Add Article
from database.db_setup import db_connection, getSession
from datetime import datetime
from analysis.keyword_extractor import extract_keywords
from data_collection.news_collector import search_news  # Add this
from collections import Counter

def test_hacker_news_api():
    print("Initializing database...")
    db_connection()
    print("Getting session...")
    session = getSession()
    print("Database ready!")
    
    url = "https://hacker-news.firebaseio.com/v0/topstories.json"
    response = requests.get(url)
    story_ids = response.json()
    extracted_words = []

    # Collect HN stories
    for i in story_ids[:100]:
        story_url = f'https://hacker-news.firebaseio.com/v0/item/{i}.json'
        story_response = requests.get(story_url)
        story_data = story_response.json()
        
        story = Story(
            title=story_data['title'], 
            score=story_data['score'], 
            num_comments=story_data.get('descendants', 0), 
            url=story_data.get('url', ''), 
            platform='hackernews', 
            timestamp=datetime.utcnow()
        )
        
        keywords = extract_keywords(story_data['title'], top_n=10)
        extracted_words.extend(keywords)
        session.add(story)
        
        print(f"Story: {story_data['title'][:50]}...")
    
    session.commit()
    print(f"\nSaved {len(story_ids[:100])} HN stories")
    
    # Count keywords
    keyword_counts = Counter(extracted_words)
    
    for keyword, count in keyword_counts.items():
        keyword_obj = Keyword(
            keyword=keyword,
            platform='hackernews',
            count=count,
            timestamp=datetime.utcnow()
        )
        session.add(keyword_obj)
    
    session.commit()
    print(f"Saved {len(keyword_counts)} unique keywords")
    
    # Get top 10 keywords to search news
    top_keywords = [kw for kw, count in keyword_counts.most_common(10)]
    print(f"\nTop keywords: {top_keywords}")
    
    # Search news for trending keywords
    print("\nSearching news articles...")
    news_articles = search_news(top_keywords, max_results=20)

    for article in news_articles:
        try:
            # Parse published_at to datetime
            pub_date = datetime.fromisoformat(article['published_at'].replace('Z', '+00:00'))
        except:
            pub_date = datetime.utcnow()
        
        article_obj = Article(
            title=article['title'],
            url=article['url'],
            source=article['source'],
            published_at=pub_date,
            platform='news',
            timestamp=datetime.utcnow()
        )
        session.add(article_obj)
    
    session.commit()
    print(f"Saved {len(news_articles)} news articles")
    print("\n" + "="*50)
    print("Collection complete!")
    print("="*50)

if __name__ == "__main__":
    test_hacker_news_api()