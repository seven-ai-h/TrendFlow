from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Text
from datetime import datetime
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Story(Base):
    __tablename__ = 'stories'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(500))
    score = Column(Integer)
    num_comments = Column(Integer)
    timestamp = Column(DateTime,  default=datetime.utcnow)
    url = Column(Text)
    platform = Column(String(50))