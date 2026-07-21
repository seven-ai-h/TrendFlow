import os
import requests
from dotenv import load_dotenv
from datetime import datetime
from database.db_setup import getSession
from database.models import Article

load_dotenv()

def _parse_published_at(published_at):
    if not published_at:
        return None
    try:
        # NewsAPI returns ISO format like 2023-08-01T12:34:56Z
        if published_at.endswith('Z'):
            return datetime.fromisoformat(published_at.replace('Z', '+00:00'))
        return datetime.fromisoformat(published_at)
    except Exception:
        return None


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
            published = _parse_published_at(article.get('publishedAt'))
            articles.append({
                'title': article.get('title'),
                'url': article.get('url'),
                'published_at': published,
                'source': article.get('source', {}).get('name'),
                'description': article.get('description', ''),
                'author': article.get('author'),
                'content': article.get('content'),
                'image': article.get('urlToImage')
            })

        return articles

    except requests.exceptions.RequestException as e:
        print(f"Error fetching news: {e}")
        return []


def save_articles(articles, session=None):
    """Save a list of article dicts into the database. Skips if URL already exists."""
    close_session = False
    if session is None:
        session = getSession()
        close_session = True

    created = 0
    for a in articles:
        if not a.get('url'):
            continue
        exists = session.query(Article).filter(Article.url == a['url']).first()
        if exists:
            continue
        art = Article(
            title=a.get('title'),
            url=a.get('url'),
            source=a.get('source'),
            published_at=a.get('published_at'),
            platform='news'
        )
        session.add(art)
        created += 1

    session.commit()
    if close_session:
        session.close()
    return created


if __name__ == "__main__":
    import sys
    kws = sys.argv[1:] or ['technology', 'artificial intelligence', 'python']
    print(f"Searching news for: {kws}")
    arts = search_news(kws, max_results=20)
    print(f"Found {len(arts)} articles; saving to DB...")
    created = save_articles(arts)
    print(f"Saved {created} new articles.")

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
