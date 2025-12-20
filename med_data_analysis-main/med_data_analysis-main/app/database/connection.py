"""
Настройка подключения к базе данных.
"""

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core import settings

# создаём асинхронный движок
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DATABASE_ECHO,
)

# создаём пул сессий
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Получение асинхронной сессии БД."""
    async with async_session_maker() as session:
        yield session


async def create_db_and_tables():
    """Создание таблиц в БД."""
    from app.database.models import Base
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
