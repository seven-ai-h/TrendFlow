from database.db_setup import getSession
from database.models import Story

session = getSession()

# Query all stories
stories = session.query(Story).all()

print(f"Total stories in database: {len(stories)}\n")

for story in stories:
    print(f"ID: {story.id}")
    print(f"Title: {story.title}")
    print(f"Score: {story.score}")
    print(f"Comments: {story.num_comments}")
    print(f"Platform: {story.platform}")
    print(f"Timestamp: {story.timestamp}")
    print("-" * 50)