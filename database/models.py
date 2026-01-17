from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Text
from datetime import datetime
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

#a story class to store fetched api and turn it into a database form so easier to modify
class Story(Base):
    __tablename__ = 'stories'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(500))
    score = Column(Integer)
    num_comments = Column(Integer)
    timestamp = Column(DateTime,  default=datetime.utcnow)
    url = Column(Text)
    platform = Column(String(50))

class Keyword(Base):
    __tablename__ = 'keywords'

    id = Column(Integer, primary_key= True)
    keyword = Column(String(200))
    platform = Column(String(50))
    count = Column(Integer)
    timestamp = Column(DateTime, default= datetime.utcnow)
    
class Article(Base):
    __tablename__ = 'articles'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(500))
    url = Column(Text)
    source = Column(String(200))
    published_at = Column(DateTime)
    platform = Column(String(50))  # 'news'
    timestamp = Column(DateTime, default=datetime.utcnow)  # When we collected it