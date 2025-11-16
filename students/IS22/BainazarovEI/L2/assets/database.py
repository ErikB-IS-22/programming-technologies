from sqlalchemy import create_engine
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()

Base = declarative_base()

class DatabaseManager:
    def __init__(self, db_path='sqlite:///chat_history.db'):
        self.engine = create_engine(db_path)
        self.Session = sessionmaker(bind=self.engine)
        Base.metadata.create_all(self.engine)
    
    def get_session(self):
        return self.Session()

class ChatMessage(Base):
    __tablename__ = 'chat_history'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

def init_db():
    db_manager = DatabaseManager()
    return db_manager

def save_message(db_manager, role, content):
    session = db_manager.get_session()
    try:
        new_message = ChatMessage(role=role, content=content)
        session.add(new_message)
        session.commit()
    except Exception as e:
        session.rollback()
        print(f"Ошибка при сохранении сообщения: {e}")
    finally:
        session.close()

def get_recent_history(db_manager):
    session = db_manager.get_session()
    try:
        messages = session.query(ChatMessage).order_by(ChatMessage.timestamp.desc()).limit(6).all()
        print(messages)
        
        history = [{"role": msg.role, "content": msg.content} for msg in reversed(messages)]
        return history
    finally:
        session.close()

def clear_history(db_manager):
    session = db_manager.get_session()
    try:
        session.query(ChatMessage).delete()
        session.commit()
        print("История диалога очищена!")
    except Exception as e:
        session.rollback()
        print(f"Ошибка при очистке истории: {e}")
    finally:
        session.close()



# import sqlite3

# def init_db():
#     conn = sqlite3.connect('chat_history.db')
#     cursor = conn.cursor()
    
#     cursor.execute('''
#         CREATE TABLE IF NOT EXISTS chat_history (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             role TEXT NOT NULL,
#             content TEXT NOT NULL,
#             timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
#         )
#     ''')
    
#     conn.commit()
#     conn.close()

# def save_message(role, content):
#     conn = sqlite3.connect('chat_history.db')
#     cursor = conn.cursor()
    
#     cursor.execute('''
#         INSERT INTO chat_history (role, content)
#         VALUES (?, ?)
#     ''', (role, content))
    
#     conn.commit()
#     conn.close()

# def get_recent_history():
#     conn = sqlite3.connect('chat_history.db')
#     cursor = conn.cursor()
    
#     cursor.execute('''
#         SELECT role, content FROM chat_history 
#         ORDER BY timestamp DESC 
#         LIMIT 6
#     ''')
    
#     messages = cursor.fetchall()
#     conn.close()
    
#     history = [{"role": role, "content": content} for role, content in reversed(messages)]
#     return history

# def clear_history():
#     conn = sqlite3.connect('chat_history.db')
#     cursor = conn.cursor()
    
#     cursor.execute('DELETE FROM chat_history')
    
#     conn.commit()
#     conn.close()