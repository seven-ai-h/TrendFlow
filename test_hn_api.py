import requests
from database.models import Story, Keyword
from database.db_setup import db_connection, getSession
from datetime import datetime
from analysis.keyword_extractor import extract_keywords
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
    extracted_keywords = []

    for i in story_ids[:100]:
        story_url = f'https://hacker-news.firebaseio.com/v0/item/{i}.json'
        story_response = requests.get(story_url)
        story_data = story_response.json()
        story = Story(title=story_data['title'], score=story_data['score'], num_comments=story_data.get('descendants', 0), url=story_data.get('url', ''), platform='hackernews', timestamp=datetime.utcnow())
        keywords = extract_keywords(story_data['title'], top_n=10)
        print(f"Keywords for '{story_data['title']}': {keywords}")  # Add this
        extracted_keywords.extend(keywords)
        session.add(story)
        print(f"\nStory ID: {i}")
        print(f"Title: {story_data['title']}")
        print(f"Score: {story_data['score']}")
        print(f"Comments: {story_data.get('descendants', 0)}")
    session.commit()
    keywords_counts = Counter(extracted_keywords)
    print(f"Debug - extracted_words: {extracted_keywords}")  # Add this
    print(f"Debug - keyword_counts: {keywords_counts}")   # Add this
    for keyword, count in keywords_counts.items():
        keyword_obj = Keyword(
        keyword=keyword,
        platform='hackernews',
        count=count,
        timestamp=datetime.utcnow()
    )
        session.add(keyword_obj)

    session.commit()
    print(f"Saved {len(keywords_counts)} unique keywords!")
        
        


        # Call the function
if __name__ == "__main__":
    test_hacker_news_api()


    