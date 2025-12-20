from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Integer,
    String,
    Text,
    create_engine,
    func,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    username = Column(String)
    user_message = Column(Text)
    assistant_message = Column(Text)
    created_at = Column(DateTime, default=func.now())
    in_history = Column(Boolean, default=True)


class DatabaseManager:
    def __init__(self, database_url: str = "sqlite:///bot_database.db"):
        self.engine = create_engine(database_url, echo=False)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )

    def add_message(
        self,
        user_id: int,
        username: str | None,
        user_message: str | None = None,
        assistant_message: str | None = None,
        in_history: bool = True,
    ) -> Message:
        session = self.SessionLocal()
        try:
            msg = Message(
                user_id=user_id,
                username=username,
                user_message=user_message,
                assistant_message=assistant_message,
                in_history=in_history,
            )
            session.add(msg)
            session.commit()
            session.refresh(msg)
            return msg
        finally:
            session.close()

    def clear_history(self, user_id: int) -> int:
        session = self.SessionLocal()
        try:
            updated_count = (
                session.query(Message)
                .filter(Message.user_id == user_id, Message.in_history == True)
                .update({"in_history": False})
            )
            session.commit()
            return updated_count
        finally:
            session.close()

    def get_last_messages(self, user_id: int, limit: int = 5) -> list[dict]:
        session = self.SessionLocal()
        try:
            msgs = (
                session.query(Message)
                .filter(Message.user_id == user_id, Message.in_history == True)
                .order_by(Message.created_at.desc())
                .limit(limit)
                .all()
            )

            msgs = list(reversed(msgs))

            result = []
            for msg in msgs:
                result.append(
                    {
                        "user_message": msg.user_message,
                        "assistant_message": msg.assistant_message,
                    }
                )
            return result
        finally:
            session.close()


db_manager = DatabaseManager()
