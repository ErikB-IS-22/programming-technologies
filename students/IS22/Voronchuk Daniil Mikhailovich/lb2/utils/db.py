from models import Message
from db import SessionLocal

def save_message(user_id: int, role: str, content: str):
    session = SessionLocal()
    msg = Message(user_id=user_id, role=role, content=content)
    session.add(msg)
    session.commit()
    session.close()


def get_user_messages(user_id: int):
    session = SessionLocal()
    messages = (
        session.query(Message)
        .filter(Message.user_id == user_id)
        .order_by(Message.id)
        .all()
    )
    session.close()
    return [(m.role, m.content) for m in messages]


def reset_user_context(user_id: int):
    session = SessionLocal()
    session.query(Message).filter(Message.user_id == user_id).delete()
    session.commit()
    session.close()
