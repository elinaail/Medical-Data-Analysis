"""
Health check эндпоинт для проверки состояния сервиса.
"""

from fastapi import APIRouter
from app.models import HealthResponse
from app.services import prediction_service

router = APIRouter()

# создаем эндпоинт для проверки состояния сервиса
@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Проверка состояния сервиса."""
    return HealthResponse(
        status="healthy",
        model_loaded=prediction_service.is_model_loaded()
    )
