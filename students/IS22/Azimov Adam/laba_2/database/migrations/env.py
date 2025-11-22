import asyncio
from logging.config import fileConfig
import os
import sys

from sqlalchemy import engine_from_config
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import create_async_engine

from alembic import context
from database.models import Base

from config import database_url

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

config.set_main_option("sqlalchemy.url", database_url)
target_metadata = Base.metadata


def get_async_engine():
    return create_async_engine(
        database_url,
        poolclass=pool.NullPool
    )


async def run_migrations_online():
    """Выполнение миграций в 'online' режиме (с реальным подключением к базе данных)."""
    connectable = get_async_engine()

    async with connectable.connect() as connection:
        # Передаем функцию в run_sync без await
        await connection.run_sync(do_run_migrations)


def do_run_migrations(connection):
    """Функция, выполняющая конфигурацию и запуск миграций."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_offline():
    """Выполнение миграций в 'offline' режиме (без реального подключения к базе данных)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


# Определение режима работы (offline/online) и выполнение миграций
if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
