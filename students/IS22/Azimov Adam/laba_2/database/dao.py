from sqlalchemy import insert, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from .models import Users, Dialogs


class BaseDAO:
    model = None

    @classmethod
    async def find_by_id(cls, db: AsyncSession, model_id: int):
        query = select(cls.model).filter_by(id=model_id)
        result = await db.execute(query)
        return result.scalar_one_or_none()

    @classmethod
    async def find_one_or_none(cls, db: AsyncSession, **filter_by):
        query = select(cls.model).filter_by(**filter_by)
        result = await db.execute(query)
        return result.scalar_one_or_none()

    @classmethod
    async def find_all(cls, db: AsyncSession, **filter_by):
        query = select(cls.model).filter_by(**filter_by)
        result = await db.execute(query)
        return result.scalars().all()

    @classmethod
    async def add(cls, db: AsyncSession, **data):
        query = insert(cls.model).values(**data).returning(cls.model.id)
        result = await db.execute(query)
        await db.commit()
        return result.scalar()

    @classmethod
    async def update(cls, db: AsyncSession, model_id, **data):
        query = (
            update(cls.model).where(cls.model.id == model_id).values(
                **data)
        )
        await db.execute(query)
        await db.commit()


class UserDAO(BaseDAO):
    model = Users


class DialogsDAO(BaseDAO):
    model = Dialogs

    @classmethod
    async def update_many(cls, db: AsyncSession, updates: list[dict]):
        for item in updates:
            model_id = item.pop("id")
            query = update(cls.model).where(
                cls.model.id == model_id).values(**item)
            await db.execute(query)
        await db.commit()
