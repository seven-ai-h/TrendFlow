import requests
from datetime import datetime

BASE_URL = 'https://dev.to/api/articles'


def collect_devto(per_page=50):
    """Fetch trending articles from Dev.to (no API key required)."""
    results = []
    # 'top' endpoint returns articles sorted by reactions in the past week
    for tag in ['', 'python', 'javascript', 'ai', 'webdev']:
        params = {'per_page': per_page, 'top': 7}
        if tag:
            params['tag'] = tag
        try:
            response = requests.get(BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            articles = response.json()
            for a in articles:
                title = a.get('title', '').strip()
                if not title:
                    continue
                results.append({
                    'title': title,
                    'score': a.get('positive_reactions_count', 0),
                    'num_comments': a.get('comments_count', 0),
                    'url': a.get('url', ''),
                    'platform': 'devto',
                })
        except requests.exceptions.RequestException as e:
            print(f"Dev.to fetch error (tag={tag!r}): {e}")

    # Deduplicate by URL
    seen = set()
    unique = []
    for item in results:
        if item['url'] not in seen:
            seen.add(item['url'])
            unique.append(item)

    print(f"  Dev.to: {len(unique)} unique articles")
    return unique
