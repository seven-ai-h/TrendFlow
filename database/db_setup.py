from sqlalchemy import create_engine
from .models import Base
from sqlalchemy.orm import sessionmaker
 
def db_connection(): #creates database file and all the tables defined in models
    engine = create_engine('sqlite:///trendflow.db')
    Base.metadata.create_all(engine)

def getSession(): # creates connection to the db and return session object to let you use that db
    engine = create_engine('sqlite:///trendflow.db')
    Session = sessionmaker(bind=engine) # returns session class
    return Session()




