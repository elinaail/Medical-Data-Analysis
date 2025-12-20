"""
Prediction эндпоинты.
"""

import time
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import logger
from app.models import (
    ECGInput,
    PredictionResponse,
    BatchECGInput,
    BatchPredictionResponse,
)
from app.api.deps import require_model_loaded
from app.services import PredictionService, HistoryService
from app.database import get_async_session

router = APIRouter()


@router.post("/forward", response_model=PredictionResponse)
async def forward(
    input_data: ECGInput,
    service: PredictionService = Depends(require_model_loaded),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Выполняет предсказание класса ЭКГ.
    
    Args:
        input_data: Входные данные для модели
        
    Returns:
        Результат предсказания в формате JSON
        
    Raises:
        HTTPException 400: Неверный формат запроса
        HTTPException 403: Модель не смогла обработать данные
    """
    logger.info("Получен запрос на предсказание")
    history_service = HistoryService(session)
    start_time = time.time()
    
    try:
        prediction = service.predict(input_data)
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Сохраняем успешный запрос
        await history_service.save_request(
            input_data=input_data,
            prediction=prediction,
            success=True,
            request_type="single",
            processing_time_ms=processing_time_ms
        )
        return prediction
    except ValueError as e:
        processing_time_ms = (time.time() - start_time) * 1000
        logger.error(f"Ошибка валидации данных: {e}")
        # Сохраняем неуспешный запрос
        await history_service.save_request(
            input_data=input_data,
            success=False,
            error_message=str(e),
            request_type="single",
            processing_time_ms=processing_time_ms
        )
        raise HTTPException(
            status_code=400,
            detail="bad request"
        )
    except Exception as e:
        processing_time_ms = (time.time() - start_time) * 1000
        logger.error(f"Ошибка при выполнении предсказания: {e}")
        # Сохраняем неуспешный запрос
        await history_service.save_request(
            input_data=input_data,
            success=False,
            error_message=str(e),
            request_type="single",
            processing_time_ms=processing_time_ms
        )
        raise HTTPException(
            status_code=403,
            detail="модель не смогла обработать данные"
        )


@router.post("/forward/batch", response_model=BatchPredictionResponse)
async def forward_batch(
    input_data: BatchECGInput,
    service: PredictionService = Depends(require_model_loaded),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Выполняет batch предсказание для нескольких записей ЭКГ.
    
    Args:
        input_data: Список входных данных для модели
        
    Returns:
        Список результатов предсказаний в формате JSON
    """
    logger.info(f"Получен batch запрос на {len(input_data.samples)} записей")
    history_service = HistoryService(session)
    
    predictions = []
    
    for i, sample in enumerate(input_data.samples):
        start_time = time.time()
        try:
            prediction = service.predict(sample)
            processing_time_ms = (time.time() - start_time) * 1000
            predictions.append(prediction)
            # Сохраняем успешный запрос
            await history_service.save_request(
                input_data=sample,
                prediction=prediction,
                success=True,
                request_type="batch",
                processing_time_ms=processing_time_ms
            )
        except ValueError as e:
            processing_time_ms = (time.time() - start_time) * 1000
            logger.error(f"Ошибка валидации данных для записи {i}: {e}")
            await history_service.save_request(
                input_data=sample,
                success=False,
                error_message=str(e),
                request_type="batch",
                processing_time_ms=processing_time_ms
            )
            raise HTTPException(
                status_code=400,
                detail="bad request"
            )
        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            logger.error(f"Ошибка обработки записи {i}: {e}")
            await history_service.save_request(
                input_data=sample,
                success=False,
                error_message=str(e),
                request_type="batch",
                processing_time_ms=processing_time_ms
            )
            raise HTTPException(
                status_code=403,
                detail="модель не смогла обработать данные"
            )
    
    logger.info(f"Batch предсказание выполнено для {len(predictions)} записей")
    
    return BatchPredictionResponse(predictions=predictions)
