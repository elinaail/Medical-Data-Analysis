"""
History эндпоинты.
"""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import logger, get_current_admin
from app.database import get_async_session
from app.models import HistoryResponse, HistoryRecord, DeleteHistoryResponse, StatsResponse
from app.services import HistoryService

router = APIRouter()


@router.get("/history", response_model=HistoryResponse)
async def get_history(
    limit: int = Query(default=100, ge=1, le=1000, description="Количество записей"),
    offset: int = Query(default=0, ge=0, description="Смещение для пагинации"),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Возвращает историю всех запросов.
    
    Args:
        limit: Максимальное количество записей (по умолчанию 100)
        offset: Смещение для пагинации
        
    Returns:
        Список записей истории с общим количеством
    """
    logger.info(f"Запрос истории: limit={limit}, offset={offset}")
    
    history_service = HistoryService(session)
    records, total = await history_service.get_history(limit=limit, offset=offset)
    
    return HistoryResponse(
        total=total,
        records=[HistoryRecord.model_validate(record) for record in records]
    )


@router.delete("/history", response_model=DeleteHistoryResponse)
async def delete_history(
    current_admin: dict = Depends(get_current_admin),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Удаляет всю историю запросов.
    
    Важно: Требует авторизации администратора (JWT токен в заголовке Authorization).
    
    Returns:
        Информация об удалённых записях
        
    Raises:
        HTTPException 401: Не авторизован
        HTTPException 403: Недостаточно прав (требуется admin)
    """
    logger.info(f"Запрос на удаление истории от администратора: {current_admin['username']}")
    
    history_service = HistoryService(session)
    deleted_count = await history_service.delete_all_history()
    
    return DeleteHistoryResponse(
        message="История успешно удалена",
        deleted_count=deleted_count
    )


@router.get("/stats", response_model=StatsResponse)
async def get_stats(
    current_admin: dict = Depends(get_current_admin),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Возвращает статистику запросов.
    
    Важно: Требует авторизации администратора (JWT токен в заголовке Authorization).
    
    Включает:
    - Общую статистику запросов (успешные/неуспешные)
    - Статистику времени обработки (mean, p50, p95, p99)
    - Характеристики входных данных (возраст, рост, вес, пол, ось сердца)
    - Распределение предсказанных классов
    
    Returns:
        Объект со статистикой
    """
    logger.info(f"Запрос статистики от администратора: {current_admin['username']}")
    
    history_service = HistoryService(session)
    stats = await history_service.get_stats()
    
    return StatsResponse(**stats)
