import logging
from typing import Optional

from sqlalchemy import (Column, DateTime, Integer, String, Text, create_engine,
                        func)
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

Base = declarative_base()


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    username = Column(String, index=True)
    message_text = Column(Text)
    response_text = Column(Text)
    timestamp = Column(DateTime, default=func.now())

    def __repr__(self):
        return f"<Message(user_id={self.user_id}, username='{self.username}', message='{self.message_text[:50]}...')>"


class DialogContext(Base):
    __tablename__ = "dialog_contexts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, unique=True, index=True)
    context_data = Column(Text)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    def __repr__(self):
        return f"<DialogContext(user_id={self.user_id})>"


class DatabaseManager:
    def __init__(self, database_url: str = "sqlite:///bot_database.db"):
        self.engine = create_engine(database_url, echo=False)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )

    def get_session(self) -> Session:
        """Получить сессию базы данных"""
        return self.SessionLocal()

    def add_message(
        self, user_id: int, username: str, message_text: str, response_text: str
    ) -> bool:
        """Добавить сообщение в базу данных"""
        session = self.get_session()
        try:
            message = Message(
                user_id=user_id,
                username=username,
                message_text=message_text,
                response_text=response_text,
            )
            session.add(message)
            session.commit()
            return True
        except SQLAlchemyError as e:
            logging.error(f"Error adding message to database: {e}")
            session.rollback()
            return False
        finally:
            session.close()

    def get_user_context(self, user_id: int) -> Optional[str]:
        """Получить контекст диалога пользователя"""
        session = self.get_session()
        try:
            context = (
                session.query(DialogContext)
                .filter(DialogContext.user_id == user_id)
                .first()
            )
            return context.context_data if context else None
        except SQLAlchemyError as e:
            logging.error(f"Error getting user context: {e}")
            return None
        finally:
            session.close()

    def update_user_context(self, user_id: int, context_data: str) -> bool:
        """Обновить контекст диалога пользователя"""
        session = self.get_session()
        try:
            context = (
                session.query(DialogContext)
                .filter(DialogContext.user_id == user_id)
                .first()
            )
            if context:
                context.context_data = context_data
            else:
                context = DialogContext(user_id=user_id, context_data=context_data)
                session.add(context)
            session.commit()
            return True
        except SQLAlchemyError as e:
            logging.error(f"Error updating user context: {e}")
            session.rollback()
            return False
        finally:
            session.close()

    def reset_user_context(self, user_id: int) -> bool:
        """Сбросить контекст диалога пользователя"""
        session = self.get_session()
        try:
            context = (
                session.query(DialogContext)
                .filter(DialogContext.user_id == user_id)
                .first()
            )
            if context:
                session.delete(context)
                session.commit()
            return True
        except SQLAlchemyError as e:
            logging.error(f"Error resetting user context: {e}")
            session.rollback()
            return False
        finally:
            session.close()


db_manager = DatabaseManager()
