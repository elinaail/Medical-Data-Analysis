"""
Зависимости для API endpoints.
"""

from fastapi import Depends, HTTPException

from app.services import prediction_service


def get_prediction_service():
    """Возвращает сервис предсказания."""
    return prediction_service


def require_model_loaded():
    """Проверяет, что модель загружена."""
    if not prediction_service.is_model_loaded():
        raise HTTPException(
            status_code=403,
            detail="модель не смогла обработать данные"
        )
    return prediction_service
