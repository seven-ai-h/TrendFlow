import requests
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

BASE_URL = 'https://api.github.com/search/repositories'
HEADERS = {'Accept': 'application/vnd.github+json', 'User-Agent': 'TrendFlow/1.0'}


def collect_github_trending(days_back=1, per_page=50):
    """
    Fetch repos created recently and sorted by stars.
    Uses public GitHub search API — no key required (60 req/hr).
    Optionally uses GITHUB_TOKEN from .env for 5000 req/hr.
    """
    token = os.getenv('GITHUB_TOKEN')
    headers = dict(HEADERS)
    if token:
        headers['Authorization'] = f'token {token}'

    since = (datetime.utcnow() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    params = {
        'q': f'created:>{since}',
        'sort': 'stars',
        'order': 'desc',
        'per_page': per_page,
    }

    try:
        response = requests.get(BASE_URL, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        items = response.json().get('items', [])
        results = []
        for repo in items:
            name = repo.get('name', '')
            desc = repo.get('description') or ''
            title = f"{name}: {desc}" if desc else name
            results.append({
                'title': title,
                'score': repo.get('stargazers_count', 0),
                'num_comments': repo.get('open_issues_count', 0),
                'url': repo.get('html_url', ''),
                'platform': 'github',
            })
        print(f"  GitHub trending: {len(results)} repos")
        return results
    except requests.exceptions.RequestException as e:
        print(f"GitHub fetch error: {e}")
        return []
