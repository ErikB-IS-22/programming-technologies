from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class Prompt(Base):
    
    __tablename__ = 'prompts'  
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(100), nullable=False, index=True)
    text = Column(Text, nullable=False)  
    created_at = Column(DateTime, default=datetime.utcnow)
    

class Dialog(Base):

    __tablename__ = 'dialogs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(100), nullable=False, index=True)
    data = Column(Text, nullable=False)  
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    

def init_db(db_url: str = "sqlite:///app.db"):
    engine = create_engine(db_url, echo=False)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


SessionLocal = init_db()

def get_session():
    return SessionLocal()
