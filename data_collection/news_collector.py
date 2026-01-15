import os
import requests
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

def search_news(keywords, max_results=10):
    """
    Search for news articles using NewsAPI
    
    Args:
        keywords: List of keywords to search for
        max_results: Maximum number of articles to return
    
    Returns:
        List of article dictionaries
    """
    api_key = os.getenv('NEWS_API_KEY')
    
    if not api_key:
        raise ValueError("NEWS_API_KEY not found in environment variables")
    
    # Join keywords with OR
    query = ' OR '.join(keywords)
    
    url = 'https://newsapi.org/v2/everything'
    params = {
        'q': query,
        'apiKey': api_key,
        'sortBy': 'publishedAt',
        'language': 'en',
        'pageSize': max_results
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        articles = []
        for article in data.get('articles', []):
            articles.append({
                'title': article.get('title'),
                'url': article.get('url'),
                'published_at': article.get('publishedAt'),
                'source': article.get('source', {}).get('name'),
                'description': article.get('description', '')
            })
        
        return articles
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching news: {e}")
        return []

# Test function
if __name__ == "__main__":
    print("Starting news collector...")
    test_keywords = ['python', 'artificial intelligence']
    print(f"Searching for: {test_keywords}")
    articles = search_news(test_keywords, max_results=5)
    
    print(f"Found {len(articles)} articles:\n")
    for article in articles:
        print(f"Title: {article['title']}")
        print(f"Source: {article['source']}")
        print(f"URL: {article['url']}")
        print("-" * 80)
