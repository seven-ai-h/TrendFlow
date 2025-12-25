import requests
from database.models import Story
from database.db_setup import db_connection, getSession
from datetime import datetime

def test_hacker_news_api():
    print("Initializing database...")
    db_connection()
    print("Getting session...")
    session = getSession()
    print("Database ready!")
    url = "https://hacker-news.firebaseio.com/v0/topstories.json"
    response = requests.get(url)
    story_ids = response.json()
    for i in story_ids[:10]:
        story_url = f'https://hacker-news.firebaseio.com/v0/item/{i}.json'
        story_response = requests.get(story_url)
        story_data = story_response.json()
        story = Story(title=story_data['title'], score=story_data['score'], num_comments=story_data.get('descendants', 0), url=story_data.get('url', ''), platform='hackernews', timestamp=datetime.utcnow())
        session.add(story)
        print(f"\nStory ID: {i}")
        print(f"Title: {story_data['title']}")
        print(f"Score: {story_data['score']}")
        print(f"Comments: {story_data.get('descendants', 0)}")
    session.commit()
        # Call the function
if __name__ == "__main__":
    test_hacker_news_api()


    