import logging
import datetime
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import ForeignKey, String, Text, DateTime, Integer, select, desc
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker


engine = create_async_engine(
    "sqlite+aiosqlite:///messages.db",
    echo=False
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(unique=True, index=True)
    username: Mapped[Optional[str]] = mapped_column(String(100))
    first_name: Mapped[Optional[str]] = mapped_column(String(100))
    last_name: Mapped[Optional[str]] = mapped_column(String(100))
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=datetime.datetime.utcnow)

    messages: Mapped[List["Message"]] = relationship("Message", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self):
        return f"User(id={self.id}, user_id={self.user_id}, username={self.username})"


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.user_id"))
    role: Mapped[str] = mapped_column(String(20))
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=datetime.datetime.utcnow)

    user: Mapped["User"] = relationship("User", back_populates="messages")

    def __repr__(self):
        return f"Message(id={self.id}, user_id={self.user_id}, role={self.role})"


class Database:
    def __init__(self):
        self.engine = engine
        self.session_factory = AsyncSessionLocal


    async def create_tables(self):
        """Создание всех таблиц в базе данных"""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)


    async def get_user_by_telegram_id(self, user_id: int) -> Optional[User]:
        """Получение пользователя по Telegram ID"""
        async with self.session_factory() as session:
            stmt = select(User).where(User.user_id == user_id)
            result = await session.execute(stmt)
            return result.scalar_one_or_none()


    async def add_user(self, user_id: int, username: str = None, 
                      first_name: str = None, last_name: str = None) -> User:
        """Добавление или обновление пользователя"""
        async with self.session_factory() as session:
            existing_user = await self.get_user_by_telegram_id(user_id)
            
            if existing_user:
                existing_user.username = username
                existing_user.first_name = first_name
                existing_user.last_name = last_name
                user = existing_user
            else:
                user = User(
                    user_id=user_id,
                    username=username,
                    first_name=first_name,
                    last_name=last_name
                )
                session.add(user)
            
            await session.commit()
            return user


    async def add_message(self, user_id: int, role: str, content: str) -> Message:
        """Добавление сообщения в базу данных"""
        async with self.session_factory() as session:
            user = await self.get_user_by_telegram_id(user_id)
            if not user:
                user = User(user_id=user_id)
                session.add(user)
                await session.commit()
                await session.refresh(user)

            message = Message(
                user_id=user_id,
                role=role,
                content=content
            )
            session.add(message)
            await session.commit()
            await session.refresh(message)
            return message


    async def get_user_messages(self, user_id: int, limit: int = 10) -> List[Message]:
        """Получение последних сообщений пользователя"""
        async with self.session_factory() as session:
            stmt = (
                select(Message)
                .where(Message.user_id == user_id)
                .order_by(Message.created_at.desc())
                .limit(limit)
            )
            result = await session.execute(stmt)
            messages = result.scalars().all()
            return list(reversed(messages))


    async def get_message_count(self, user_id: int) -> int:
        """Получение количества сообщений пользователя"""
        async with self.session_factory() as session:
            stmt = select(Message).where(Message.user_id == user_id)
            result = await session.execute(stmt)
            messages = result.scalars().all()
            return len(messages)


    async def get_conversation_context(self, user_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """Получение контекста диалога для AI-модели"""
        messages = await self.get_user_messages(user_id, limit)
        
        context = []
        for message in messages:
            context.append({
                "role": message.role,
                "content": message.content
            })
        
        return context


    async def clear_user_messages(self, user_id: int) -> int:
        """Очистка истории сообщений пользователя"""
        async with self.session_factory() as session:
            stmt = select(Message).where(Message.user_id == user_id)
            result = await session.execute(stmt)
            messages = result.scalars().all()
            
            count = len(messages)
            for message in messages:
                await session.delete(message)
            
            await session.commit()
            return count

db = Database()