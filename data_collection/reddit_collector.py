import requests
from datetime import datetime

SUBREDDITS = ['technology', 'programming', 'artificial', 'science', 'worldnews', 'MachineLearning']
HEADERS = {'User-Agent': 'TrendFlow/1.0 (data pipeline project)'}


def fetch_reddit_posts(subreddit, limit=50, time_filter='day'):
    url = f'https://www.reddit.com/r/{subreddit}/top.json'
    params = {'limit': limit, 't': time_filter}
    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=10)
        response.raise_for_status()
        posts = response.json().get('data', {}).get('children', [])
        results = []
        for post in posts:
            d = post.get('data', {})
            if d.get('stickied') or not d.get('title'):
                continue
            results.append({
                'title': d['title'],
                'score': d.get('ups', 0),
                'num_comments': d.get('num_comments', 0),
                'url': d.get('url', ''),
                'subreddit': subreddit,
                'platform': 'reddit',
            })
        return results
    except requests.exceptions.RequestException as e:
        print(f"Reddit fetch error ({subreddit}): {e}")
        return []


def collect_reddit(subreddits=None, limit=50):
    """Fetch top posts from multiple subreddits."""
    all_posts = []
    for sub in (subreddits or SUBREDDITS):
        posts = fetch_reddit_posts(sub, limit=limit)
        all_posts.extend(posts)
        print(f"  Reddit r/{sub}: {len(posts)} posts")
    return all_posts
