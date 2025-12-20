"""
FastAPI приложение для классификации ЭКГ.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from app.core import settings, logger
from app.api.v1.router import router as api_v1_router
from app.services import prediction_service
from app.models import RootResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Жизненный цикл приложения."""
    # startup
    logger.info("Запуск приложения...")
    
    # база данных управляется через Alembic миграции
    # для применения миграций используйте: uv run alembic upgrade head
    logger.info("Проверка базы данных...")
    logger.info("Для управления схемой БД используйте: uv run alembic upgrade head")
    
    try:
        prediction_service.load_model()
    except FileNotFoundError:
        logger.warning("Модель не найдена при запуске. Загрузите модель вручную.")
    
    yield
    
    # shutdown
    logger.info("Остановка приложения...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.PROJECT_DESCRIPTION,
    version=settings.VERSION,
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Кастомный обработчик ошибок валидации.
    Возвращает 400 Bad Request вместо 422 Validation Error.
    """
    logger.error(f"Ошибка валидации запроса: {exc.errors()}")
    return JSONResponse(
        status_code=400,
        content={"detail": "bad request"}
    )


# подключаем API v1 router
app.include_router(api_v1_router, prefix=settings.API_V1_PREFIX)


@app.get("/", response_model=RootResponse)
async def root():
    """Корневой эндпоинт."""
    return RootResponse(
        message=settings.PROJECT_NAME,
        version=settings.VERSION,
        endpoints={
            f"{settings.API_V1_PREFIX}/forward": "POST - Предсказание класса ЭКГ",
            f"{settings.API_V1_PREFIX}/forward/batch": "POST - Batch предсказание",
            f"{settings.API_V1_PREFIX}/health": "GET - Проверка состояния сервиса",
            f"{settings.API_V1_PREFIX}/history": "GET - История всех запросов",
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True
    )
