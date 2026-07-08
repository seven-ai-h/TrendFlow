from sqlalchemy import Column, Integer, String, Float, DateTime, Text, UniqueConstraint
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
    platform = Column(String(50))  # 'news' or 'rss'
    timestamp = Column(DateTime, default=datetime.utcnow)


class PipelineRun(Base):
    """Tracks each data collection run for data engineering observability."""
    __tablename__ = 'pipeline_runs'

    id = Column(Integer, primary_key=True)
    started_at = Column(DateTime, default=datetime.utcnow)
    finished_at = Column(DateTime, nullable=True)
    status = Column(String(20))          # 'running' | 'success' | 'failed'
    stories_collected = Column(Integer, default=0)
    keywords_extracted = Column(Integer, default=0)
    sources_run = Column(String(500), nullable=True)   # comma-separated
    error_message = Column(Text, nullable=True)