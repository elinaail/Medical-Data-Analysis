"""
Router для API v1.
"""

from fastapi import APIRouter

from app.api.v1.endpoints import health, prediction, history, auth

router = APIRouter()

router.include_router(auth.router, prefix="/auth", tags=["auth"])
router.include_router(health.router, tags=["health"])
router.include_router(prediction.router, tags=["prediction"])
router.include_router(history.router, tags=["history"])
