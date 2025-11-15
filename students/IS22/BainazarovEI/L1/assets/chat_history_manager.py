from database import ChatMessage

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