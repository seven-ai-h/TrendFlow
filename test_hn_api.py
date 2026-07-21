import requests
from database.models import Story, Keyword, Article, PipelineRun, MarketData
from database.db_setup import db_connection, getSession
from datetime import datetime
from analysis.entity_extractor import extract_entities
from analysis.sentiment import score_sentiment
from data_collection.news_collector import search_news
from data_collection.reddit_collector import collect_reddit
from data_collection.devto_collector import collect_devto
from data_collection.github_collector import collect_github_trending
from data_collection.rss_collector import collect_rss
from data_collection.market_collector import collect_market_data, collect_ticker_news
from collections import Counter


def _existing_urls(session) -> set:
    """Fetch all URLs already stored (for deduplication)."""
    story_urls = {r[0] for r in session.query(Story.url).filter(Story.url != '').all()}
    article_urls = {r[0] for r in session.query(Article.url).filter(Article.url != '').all()}
    return story_urls | article_urls


def _save_stories(session, posts, platform_label, existing_urls: set):
    """Persist new Story rows (skipping duplicate URLs) and return extracted entities."""
    all_entities = []
    new_count = 0
    for post in posts:
        url = post.get('url', '')
        if url and url in existing_urls:
            continue
        if url:
            existing_urls.add(url)
        story = Story(
            title=post['title'],
            score=post.get('score', 0),
            num_comments=post.get('num_comments', 0),
            url=url,
            platform=platform_label,
            sentiment=score_sentiment(post['title']),
            timestamp=post.get('_timestamp') or datetime.utcnow(),
        )
        session.add(story)
        new_count += 1
        all_entities.extend(extract_entities(post['title'], top_n=10))
    return all_entities, new_count


def _collect_market(session):
    """Fetch daily prices for tracked tickers and upsert into MarketData."""
    print("=== Market Prices (yfinance) ===")
    try:
        bars = collect_market_data()
        existing = {(m.ticker, m.date.date() if hasattr(m.date, 'date') else m.date)
                    for m in session.query(MarketData).all()}
        added = 0
        for b in bars:
            key = (b['ticker'], b['date'].date() if hasattr(b['date'], 'date') else b['date'])
            if key in existing:
                continue
            existing.add(key)
            session.add(MarketData(
                ticker=b['ticker'], date=b['date'], open=b['open'], close=b['close'],
                high=b['high'], low=b['low'], volume=b['volume'], return_pct=b['return_pct']))
            added += 1
        session.commit()
        print(f"Saved {added} new price bars\n")
        return added > 0
    except Exception as e:
        print(f"Market error: {e}\n")
        return False


def _save_keywords(session, entity_list, platform_label):
    counts = Counter(entity_list)
    for entity, count in counts.items():
        session.add(Keyword(
            keyword=entity,
            platform=platform_label,
            count=count,
            timestamp=datetime.utcnow(),
        ))
    return counts


def run_pipeline():
    print("Initializing database…")
    db_connection()
    session = getSession()
    print("Database ready!\n")

    existing_urls = _existing_urls(session)
    print(f"Already stored {len(existing_urls)} unique URLs (will skip duplicates)\n")

    # ── Start pipeline run tracking ───────────────────────────────────────────
    run = PipelineRun(
        started_at=datetime.utcnow(),
        status='running',
    )
    session.add(run)
    session.commit()

    all_entities = []
    total_stories = 0
    sources_run = []

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
            print(f"  {item['title'][:70]}…")
        entities, new_count = _save_stories(session, hn_posts, 'hackernews', existing_urls)
        all_entities.extend(entities)
        counts = _save_keywords(session, entities, 'hackernews')
        session.commit()
        total_stories += new_count
        sources_run.append('hackernews')
        print(f"Saved {new_count} new HN stories, {len(counts)} entity types\n")
    except Exception as e:
        print(f"HN error: {e}\n")

    # ── Reddit ────────────────────────────────────────────────────────────────
    print("=== Reddit ===")
    try:
        reddit_posts = collect_reddit()
        entities, new_count = _save_stories(session, reddit_posts, 'reddit', existing_urls)
        all_entities.extend(entities)
        counts = _save_keywords(session, entities, 'reddit')
        session.commit()
        total_stories += new_count
        sources_run.append('reddit')
        print(f"Saved {new_count} new Reddit posts, {len(counts)} entity types\n")
    except Exception as e:
        print(f"Reddit error: {e}\n")

    # ── Dev.to ────────────────────────────────────────────────────────────────
    print("=== Dev.to ===")
    try:
        devto_posts = collect_devto()
        entities, new_count = _save_stories(session, devto_posts, 'devto', existing_urls)
        all_entities.extend(entities)
        counts = _save_keywords(session, entities, 'devto')
        session.commit()
        total_stories += new_count
        sources_run.append('devto')
        print(f"Saved {new_count} new Dev.to articles, {len(counts)} entity types\n")
    except Exception as e:
        print(f"Dev.to error: {e}\n")

    # ── GitHub Trending ───────────────────────────────────────────────────────
    print("=== GitHub Trending ===")
    try:
        github_posts = collect_github_trending()
        entities, new_count = _save_stories(session, github_posts, 'github', existing_urls)
        all_entities.extend(entities)
        counts = _save_keywords(session, entities, 'github')
        session.commit()
        total_stories += new_count
        sources_run.append('github')
        print(f"Saved {new_count} new GitHub repos, {len(counts)} entity types\n")
    except Exception as e:
        print(f"GitHub error: {e}\n")

    # ── RSS Feeds ────────────────────────────────────────────────────────────
    print("=== RSS Feeds ===")
    try:
        rss_posts = collect_rss()
        entities, new_count = _save_stories(session, rss_posts, 'rss', existing_urls)
        all_entities.extend(entities)
        counts = _save_keywords(session, entities, 'rss')
        session.commit()
        total_stories += new_count
        sources_run.append('rss')
        print(f"Saved {new_count} new RSS articles, {len(counts)} entity types\n")
    except Exception as e:
        print(f"RSS error: {e}\n")

    # ── NewsAPI (cross-reference top entities) ────────────────────────────────
    print("=== NewsAPI ===")
    top_entities = [kw for kw, _ in Counter(all_entities).most_common(10)]
    # Filter to single-word terms for NewsAPI compatibility
    top_keywords = [e for e in top_entities if ' ' not in e][:10]
    print(f"Cross-referencing top entities: {top_keywords}")
    try:
        news_articles = search_news(top_keywords, max_results=20)
        new_articles = 0
        for article in news_articles:
            url = article.get('url', '')
            if url and url in existing_urls:
                continue
            if url:
                existing_urls.add(url)
            try:
                pub_date = datetime.fromisoformat(article['published_at'].replace('Z', '+00:00'))
            except Exception:
                pub_date = datetime.utcnow()
            session.add(Article(
                title=article['title'],
                url=url,
                source=article['source'],
                published_at=pub_date,
                platform='news',
                timestamp=datetime.utcnow(),
            ))
            new_articles += 1
        session.commit()
        sources_run.append('news')
        print(f"Saved {new_articles} new news articles\n")
    except Exception as e:
        print(f"NewsAPI error (check NEWS_API_KEY in .env): {e}\n")

    # ── Per-ticker financial news (yfinance) — real, already ticker-tagged ─────
    print("=== Ticker News (yfinance) ===")
    try:
        news_posts = collect_ticker_news()
        # keep only genuinely new URLs, score sentiment, tag platform 'finance'
        fresh = [p for p in news_posts if not p.get('url') or p['url'] not in existing_urls]
        entities, new_count = _save_stories(session, fresh, 'finance', existing_urls)
        all_entities.extend(entities)
        _save_keywords(session, entities, 'finance')
        session.commit()
        total_stories += new_count
        if new_count:
            sources_run.append('finance')
        print(f"Saved {new_count} new ticker headlines\n")
    except Exception as e:
        print(f"Ticker news error: {e}\n")

    # ── Market prices ─────────────────────────────────────────────────────────
    if _collect_market(session):
        sources_run.append('market')

    # ── Finalize pipeline run ─────────────────────────────────────────────────
    run.finished_at = datetime.utcnow()
    run.status = 'success'
    run.stories_collected = total_stories
    run.keywords_extracted = len(set(all_entities))
    run.sources_run = ','.join(sources_run)
    session.commit()

    elapsed = (run.finished_at - run.started_at).total_seconds()
    print("=" * 60)
    print(f"Pipeline complete in {elapsed:.1f}s")
    print(f"  Sources: {run.sources_run}")
    print(f"  New stories: {run.stories_collected}")
    print(f"  Unique entities extracted: {run.keywords_extracted}")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()
