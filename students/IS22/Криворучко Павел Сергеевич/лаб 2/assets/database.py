from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, String, Text
from config import DATABASE_URL
import json

engine = create_async_engine(DATABASE_URL, echo=False)

AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)

Base = declarative_base()

class DialogHistory(Base):
    __tablename__ = "dialog_history"

    user_id = Column(String, primary_key=True)
    history = Column(Text)  

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def get_dialog_history(user_id: int):
    async with AsyncSessionLocal() as session:
        user_id_str = str(user_id)  
        result = await session.get(DialogHistory, user_id_str)
        if result and result.history:
            return json.loads(result.history)
        return []

async def save_dialog_history(user_id: int, dialog_history_actual: list):
    async with AsyncSessionLocal() as session:
        user_id_str = str(user_id) 
        obj = await session.get(DialogHistory, user_id_str)

        if obj:
            obj.history = json.dumps(dialog_history_actual)
        else:
            obj = DialogHistory(
                user_id=user_id_str,
                history=json.dumps(dialog_history_actual)
            )
            session.add(obj)
        await session.commit()