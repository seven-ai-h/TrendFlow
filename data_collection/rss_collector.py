"""
RSS collector with two backends:
  1. feedparser (if importable)
  2. requests + xml.etree.ElementTree fallback (always available)
"""
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
from email.utils import parsedate_to_datetime

# Try feedparser; fall back silently if broken (e.g. Python 3.11 + old feedparser)
try:
    import feedparser as _feedparser
    _HAVE_FEEDPARSER = True
except Exception:
    _feedparser = None
    _HAVE_FEEDPARSER = False

RSS_FEEDS = [
    ("TechCrunch", "https://techcrunch.com/feed/"),
    ("Wired", "https://www.wired.com/feed/rss"),
    ("MIT Tech Review", "https://www.technologyreview.com/feed/"),
    ("ArsTechnica", "http://feeds.arstechnica.com/arstechnica/index"),
    ("The Verge Tech", "https://www.theverge.com/tech/rss/index.xml"),
    ("HN Best (RSS)", "https://hnrss.org/best"),
    ("VentureBeat AI", "https://venturebeat.com/category/ai/feed/"),
]

_RSS_NS = {
    'atom': 'http://www.w3.org/2005/Atom',
    'content': 'http://purl.org/rss/1.0/modules/content/',
}


def _parse_date_str(s: str) -> datetime:
    if not s:
        return datetime.utcnow()
    try:
        return parsedate_to_datetime(s).replace(tzinfo=None)
    except Exception:
        return datetime.utcnow()


def _fetch_feedparser(source_name: str, feed_url: str, max_items: int) -> list:
    feed = _feedparser.parse(feed_url)
    results = []
    for entry in feed.entries[:max_items]:
        title = entry.get('title', '').strip()
        if not title:
            continue
        pub = datetime.utcnow()
        if hasattr(entry, 'published_parsed') and entry.published_parsed:
            try:
                pub = datetime(*entry.published_parsed[:6])
            except Exception:
                pass
        results.append({
            'title': title,
            'url': entry.get('link', ''),
            'score': 0,
            'num_comments': 0,
            'platform': 'rss',
            '_source': source_name,
        })
    return results


def _fetch_xml(source_name: str, feed_url: str, max_items: int) -> list:
    headers = {'User-Agent': 'TrendFlow/1.0'}
    resp = requests.get(feed_url, timeout=10, headers=headers)
    resp.raise_for_status()
    root = ET.fromstring(resp.content)

    # Handle both RSS 2.0 and Atom
    items = root.findall('.//item') or root.findall('.//{http://www.w3.org/2005/Atom}entry')

    results = []
    for item in items[:max_items]:
        # Title
        title_el = item.find('title') or item.find('{http://www.w3.org/2005/Atom}title')
        title = (title_el.text or '').strip() if title_el is not None else ''
        if not title:
            continue

        # Link
        link_el = item.find('link') or item.find('{http://www.w3.org/2005/Atom}link')
        link = ''
        if link_el is not None:
            link = link_el.get('href') or link_el.text or ''

        results.append({
            'title': title,
            'url': link.strip(),
            'score': 0,
            'num_comments': 0,
            'platform': 'rss',
            '_source': source_name,
        })
    return results


def collect_rss(max_per_feed: int = 20) -> list:
    """Fetch articles from curated RSS feeds. Returns Story-compatible dicts."""
    results = []
    for source_name, feed_url in RSS_FEEDS:
        try:
            if _HAVE_FEEDPARSER:
                items = _fetch_feedparser(source_name, feed_url, max_per_feed)
            else:
                items = _fetch_xml(source_name, feed_url, max_per_feed)
            results.extend(items)
        except Exception as e:
            print(f"RSS fetch error ({source_name}): {e}")

    # Deduplicate by URL
    seen: set = set()
    unique = []
    for item in results:
        key = item.get('url') or item.get('title', '')
        if key and key not in seen:
            seen.add(key)
            unique.append(item)

    backend = "feedparser" if _HAVE_FEEDPARSER else "xml fallback"
    print(f"  RSS [{backend}]: {len(unique)} unique articles from {len(RSS_FEEDS)} feeds")
    return unique
