"""
Сервис для работы с историей запросов.
"""

from typing import List, Optional

from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import RequestHistory
from app.models.schemas import ECGInput, PredictionResponse
from app.core import logger


class HistoryService:
    """Сервис для работы с историей запросов."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def save_request(
        self,
        input_data: ECGInput,
        prediction: Optional[PredictionResponse] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        request_type: str = "single",
        processing_time_ms: Optional[float] = None
    ) -> RequestHistory:
        """
        Сохраняет запрос в историю.
        
        Args:
            input_data: Входные данные запроса
            prediction: Результат предсказания (если успешно)
            success: Успешность запроса
            error_message: Сообщение об ошибке (если неуспешно)
            request_type: Тип запроса (single/batch)
            processing_time_ms: Время обработки в миллисекундах
            
        Returns:
            Созданная запись истории
        """
        record = RequestHistory(
            age=input_data.age,
            sex=input_data.sex,
            height=input_data.height,
            weight=input_data.weight,
            heart_axis=input_data.heart_axis,
            predicted_class_index=prediction.class_index if prediction else None,
            predicted_class_name=prediction.class_name if prediction else None,
            predicted_class_description=prediction.class_description if prediction else None,
            success=success,
            error_message=error_message,
            request_type=request_type,
            processing_time_ms=processing_time_ms,
        )
        
        self.session.add(record)
        await self.session.commit()
        await self.session.refresh(record)
        
        logger.info(f"Сохранена запись истории: id={record.id}")
        return record
    
    async def get_history(
        self,
        limit: int = 100,
        offset: int = 0
    ) -> tuple[List[RequestHistory], int]:
        """
        Получает историю запросов.
        
        Args:
            limit: Максимальное количество записей
            offset: Смещение для пагинации
            
        Returns:
            Кортеж (список записей, общее количество)
        """
        # получаем общее количество
        count_query = select(RequestHistory)
        result = await self.session.execute(count_query)
        total = len(result.scalars().all())
        
        # получаем записи с пагинацией
        query = (
            select(RequestHistory)
            .order_by(desc(RequestHistory.created_at))
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.execute(query)
        records = result.scalars().all()
        
        return list(records), total
    
    async def delete_all_history(self) -> int:
        """
        Удаляет всю историю запросов.
        
        Returns:
            Количество удалённых записей
        """
        # получаем количество записей перед удалением
        count_query = select(RequestHistory)
        result = await self.session.execute(count_query)
        count = len(result.scalars().all())
        
        # удаляем все записи
        from sqlalchemy import delete
        delete_query = delete(RequestHistory)
        await self.session.execute(delete_query)
        await self.session.commit()
        
        logger.info(f"Удалено {count} записей из истории")
        return count
    
    async def get_stats(self) -> dict:
        """
        Получает статистику запросов.
        
        Returns:
            Словарь со статистикой
        """
        import numpy as np
        from collections import Counter
        
        # получаем все записи
        query = select(RequestHistory)
        result = await self.session.execute(query)
        records = result.scalars().all()
        
        if not records:
            return {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "processing_time": None,
                "input_stats": None,
                "class_distribution": None,
            }
        
        # базовая статистика
        total = len(records)
        successful = sum(1 for r in records if r.success)
        failed = total - successful
        
        # статистика времени обработки
        processing_times = [r.processing_time_ms for r in records if r.processing_time_ms is not None]
        
        processing_time_stats = None
        if processing_times:
            times_array = np.array(processing_times)
            processing_time_stats = {
                "mean_ms": float(np.mean(times_array)),
                "p50_ms": float(np.percentile(times_array, 50)),
                "p95_ms": float(np.percentile(times_array, 95)),
                "p99_ms": float(np.percentile(times_array, 99)),
                "min_ms": float(np.min(times_array)),
                "max_ms": float(np.max(times_array)),
            }
        
        # статистика входных данных
        ages = [r.age for r in records]
        heights = [r.height for r in records]
        weights = [r.weight for r in records]
        sexes = [r.sex for r in records]
        heart_axes = [r.heart_axis for r in records]
        
        input_stats = {
            "age": {
                "mean": float(np.mean(ages)),
                "min": float(np.min(ages)),
                "max": float(np.max(ages)),
                "std": float(np.std(ages)),
            },
            "height": {
                "mean": float(np.mean(heights)),
                "min": float(np.min(heights)),
                "max": float(np.max(heights)),
                "std": float(np.std(heights)),
            },
            "weight": {
                "mean": float(np.mean(weights)),
                "min": float(np.min(weights)),
                "max": float(np.max(weights)),
                "std": float(np.std(weights)),
            },
            "sex_distribution": dict(Counter(sexes)),
            "heart_axis_distribution": dict(Counter(heart_axes)),
        }
        
        # распределение предсказанных классов
        predicted_classes = [r.predicted_class_name for r in records if r.predicted_class_name]
        class_distribution = dict(Counter(predicted_classes))
        
        # распределение по типам запросов
        request_types = [r.request_type for r in records]
        request_type_distribution = dict(Counter(request_types))
        
        return {
            "total_requests": total,
            "successful_requests": successful,
            "failed_requests": failed,
            "success_rate": round(successful / total * 100, 2),
            "request_type_distribution": request_type_distribution,
            "processing_time": processing_time_stats,
            "input_stats": input_stats,
            "class_distribution": class_distribution,
        }
