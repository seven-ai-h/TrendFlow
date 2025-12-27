from database.db_setup import getSession
from database.models import Keyword

session = getSession()

keywords = session.query(Keyword).order_by(Keyword.count.desc()).all()

print(f"Total unique keywords: {len(keywords)}\n")
print("Top 10 keywords:")
for kw in keywords[:10]:
    print(f"{kw.keyword}: {kw.count} times")