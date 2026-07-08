import requests
from database.models import Story, Keyword, Article
from database.db_setup import db_connection, getSession
from datetime import datetime
from analysis.keyword_extractor import extract_keywords
from data_collection.news_collector import search_news
from data_collection.reddit_collector import collect_reddit
from data_collection.devto_collector import collect_devto
from data_collection.github_collector import collect_github_trending
from collections import Counter


def _save_stories(session, posts, platform_label):
    """Persist a list of post dicts as Story rows and return extracted keywords."""
    all_keywords = []
    for post in posts:
        story = Story(
            title=post['title'],
            score=post.get('score', 0),
            num_comments=post.get('num_comments', 0),
            url=post.get('url', ''),
            platform=platform_label,
            timestamp=datetime.utcnow(),
        )
        session.add(story)
        all_keywords.extend(extract_keywords(post['title'], top_n=10))
    return all_keywords


def _save_keywords(session, keyword_list, platform_label):
    counts = Counter(keyword_list)
    for keyword, count in counts.items():
        session.add(Keyword(
            keyword=keyword,
            platform=platform_label,
            count=count,
            timestamp=datetime.utcnow(),
        ))
    return counts


def test_hacker_news_api():
    print("Initializing database...")
    db_connection()
    session = getSession()
    print("Database ready!\n")

    all_keywords = []

    # ── Hacker News ───────────────────────────────────────────────────────────
    print("=== Hacker News ===")
    url = "https://hacker-news.firebaseio.com/v0/topstories.json"
    try:
        response = requests.get(url, timeout=10)
        story_ids = response.json()
        hn_posts = []
        for story_id in story_ids[:100]:
            item_url = f'https://hacker-news.firebaseio.com/v0/item/{story_id}.json'
            item = requests.get(item_url, timeout=10).json()
            if not item or not item.get('title'):
                continue
            hn_posts.append({
                'title': item['title'],
                'score': item.get('score', 0),
                'num_comments': item.get('descendants', 0),
                'url': item.get('url', ''),
            })
            print(f"  {item['title'][:60]}...")
        kws = _save_stories(session, hn_posts, 'hackernews')
        all_keywords.extend(kws)
        counts = _save_keywords(session, kws, 'hackernews')
        session.commit()
        print(f"Saved {len(hn_posts)} HN stories, {len(counts)} keywords\n")
    except Exception as e:
        print(f"HN error: {e}\n")

    # ── Reddit ────────────────────────────────────────────────────────────────
    print("=== Reddit ===")
    try:
        reddit_posts = collect_reddit()
        kws = _save_stories(session, reddit_posts, 'reddit')
        all_keywords.extend(kws)
        counts = _save_keywords(session, kws, 'reddit')
        session.commit()
        print(f"Saved {len(reddit_posts)} Reddit posts, {len(counts)} keywords\n")
    except Exception as e:
        print(f"Reddit error: {e}\n")

    # ── Dev.to ────────────────────────────────────────────────────────────────
    print("=== Dev.to ===")
    try:
        devto_posts = collect_devto()
        kws = _save_stories(session, devto_posts, 'devto')
        all_keywords.extend(kws)
        counts = _save_keywords(session, kws, 'devto')
        session.commit()
        print(f"Saved {len(devto_posts)} Dev.to articles, {len(counts)} keywords\n")
    except Exception as e:
        print(f"Dev.to error: {e}\n")

    # ── GitHub Trending ───────────────────────────────────────────────────────
    print("=== GitHub Trending ===")
    try:
        github_posts = collect_github_trending()
        kws = _save_stories(session, github_posts, 'github')
        all_keywords.extend(kws)
        counts = _save_keywords(session, kws, 'github')
        session.commit()
        print(f"Saved {len(github_posts)} GitHub repos, {len(counts)} keywords\n")
    except Exception as e:
        print(f"GitHub error: {e}\n")

    # ── NewsAPI (cross-reference top keywords) ────────────────────────────────
    print("=== NewsAPI ===")
    top_keywords = [kw for kw, _ in Counter(all_keywords).most_common(10)]
    print(f"Cross-referencing top keywords: {top_keywords}")
    try:
        news_articles = search_news(top_keywords, max_results=20)
        for article in news_articles:
            try:
                pub_date = datetime.fromisoformat(article['published_at'].replace('Z', '+00:00'))
            except Exception:
                pub_date = datetime.utcnow()
            session.add(Article(
                title=article['title'],
                url=article['url'],
                source=article['source'],
                published_at=pub_date,
                platform='news',
                timestamp=datetime.utcnow(),
            ))
        session.commit()
        print(f"Saved {len(news_articles)} news articles\n")
    except Exception as e:
        print(f"NewsAPI error (check NEWS_API_KEY in .env): {e}\n")

    print("=" * 50)
    print("Collection complete!")
    print("=" * 50)


if __name__ == "__main__":
    test_hacker_news_api()
