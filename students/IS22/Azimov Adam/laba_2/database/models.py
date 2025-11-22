
import datetime
import sqlalchemy as sa

from typing import Annotated
from sqlalchemy import JSON, ForeignKey, String, ARRAY
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "%(table_name)s_%(column_0_name)s_key",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "%(table_name)s_%(column_0_name)s_fkey",
    "pk": "%(table_name)s_pkey",
}

metadata = sa.MetaData(naming_convention=convention)


class Base(DeclarativeBase):
    """Базовая модель"""
    uuid_pk = Annotated[UUID,
                        mapped_column(UUID(as_uuid=True),
                                      primary_key=True,
                                      server_default=sa.text("gen_random_uuid()"))]

    created_at = Annotated[datetime.datetime, mapped_column(
        server_default=sa.text("TIMEZONE('Europe/Moscow', NOW())"))]
    updated_at = Annotated[datetime.datetime, mapped_column(
        server_default=sa.text("TIMEZONE('Europe/Moscow', NOW())"),
        onupdate=datetime.datetime.utcnow)]


class Users(Base):
    __tablename__ = "users"

    id: Mapped[Base.uuid_pk]
    username: Mapped[str] = mapped_column(String(255), nullable=False)
    telegram_id: Mapped[str] = mapped_column(String(10),
                                             nullable=False, unique=True)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)

    created_at: Mapped[Base.created_at] = mapped_column(nullable=True)
    updated_at: Mapped[Base.updated_at] = mapped_column(nullable=True)


class Dialogs(Base):
    __tablename__ = "dialogs"

    id: Mapped[Base.uuid_pk]
    is_active: Mapped[bool] = mapped_column(default=True)
    username: Mapped[str] = mapped_column(String(255), nullable=False)
    user_id: Mapped[UUID] = mapped_column(
        ForeignKey('users.id', ondelete='CASCADE'),
        nullable=False
    )

    list_messages: Mapped[list[dict]] = mapped_column(
        ARRAY(JSON),  # Массив JSON объектов
        nullable=True,
        default=list
    )

    created_at: Mapped[Base.created_at] = mapped_column(nullable=True)
    updated_at: Mapped[Base.updated_at] = mapped_column(nullable=True)
    user: Mapped["Users"] = relationship()
