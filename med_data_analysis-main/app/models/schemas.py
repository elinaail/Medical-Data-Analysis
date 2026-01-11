"""
Pydantic модели (схемы) для API.
"""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator

from app.core.constants import VALID_HEART_AXIS


class ECGInput(BaseModel):
    """Входные данные для предсказания модели."""
    
    age: float = Field(..., description="Возраст пациента")
    sex: int = Field(..., description="Пол пациента (0 - женский, 1 - мужской)")
    height: float = Field(..., description="Рост пациента в см")
    weight: float = Field(..., description="Вес пациента в кг")
    heart_axis: str = Field(
        ..., 
        description=f"Электрическая ось сердца ({', '.join(VALID_HEART_AXIS)})"
    )
    ecg_signal: List[List[float]] = Field(
        ..., 
        description="ЭКГ сигнал: массив размером [5000, 12] - 5000 временных точек для 12 отведений"
    )
    
    # валидаторы полей будущего JSON входных данных
    @field_validator("heart_axis")
    @classmethod
    def validate_heart_axis(cls, v: str) -> str:
        if v not in VALID_HEART_AXIS:
            raise ValueError(f"heart_axis должен быть одним из: {VALID_HEART_AXIS}")
        return v
    
    @field_validator("ecg_signal")
    @classmethod
    def validate_ecg_signal(cls, v: List[List[float]]) -> List[List[float]]:
        if len(v) == 0:
            raise ValueError("ecg_signal не может быть пустым!")
        if len(v[0]) != 12:
            raise ValueError(f"ecg_signal должен иметь 12 отведений, получено: {len(v[0])}!")
        return v
    
    @field_validator("sex")
    @classmethod
    def validate_sex(cls, v: int) -> int:
        if v not in [0, 1]:
            raise ValueError("sex должен быть 0 или 1")
        return v


class PredictionResponse(BaseModel):
    """Ответ с результатом предсказания."""
    
    class_index: int = Field(..., description="Числовой индекс предсказанного класса")
    class_name: str = Field(..., description="Название предсказанного класса")
    class_description: str = Field(..., description="Описание предсказанного класса")


class HealthResponse(BaseModel):
    """Ответ на проверку здоровья сервиса."""
    
    status: str = Field(..., description="Статус сервиса")
    model_loaded: bool = Field(..., description="Загружена ли модель")


class RootResponse(BaseModel):
    """Ответ корневого эндпоинта."""
    
    message: str
    version: str
    endpoints: dict


class BatchECGInput(BaseModel):
    """Входные данные для batch предсказания."""
    
    samples: List[ECGInput] = Field(..., description="Список входных данных")


class BatchPredictionResponse(BaseModel):
    """Ответ с результатами batch предсказания."""
    
    predictions: List[PredictionResponse] = Field(..., description="Список предсказаний")


class HistoryRecord(BaseModel):
    """Запись из истории запросов."""
    
    id: int = Field(..., description="ID записи")
    created_at: datetime = Field(..., description="Время запроса")
    age: float = Field(..., description="Возраст пациента")
    sex: int = Field(..., description="Пол пациента")
    height: float = Field(..., description="Рост пациента")
    weight: float = Field(..., description="Вес пациента")
    heart_axis: str = Field(..., description="Электрическая ось сердца")
    predicted_class_index: Optional[int] = Field(None, description="Индекс предсказанного класса")
    predicted_class_name: Optional[str] = Field(None, description="Название предсказанного класса")
    predicted_class_description: Optional[str] = Field(None, description="Описание предсказанного класса")
    success: bool = Field(..., description="Успешность запроса")
    error_message: Optional[str] = Field(None, description="Сообщение об ошибке")
    request_type: str = Field(..., description="Тип запроса (single/batch)")
    
    class Config:
        from_attributes = True


class HistoryResponse(BaseModel):
    """Ответ с историей запросов."""
    
    total: int = Field(..., description="Общее количество записей")
    records: List[HistoryRecord] = Field(..., description="Список записей")


class DeleteHistoryResponse(BaseModel):
    """Ответ на удаление истории."""
    
    message: str = Field(..., description="Сообщение о результате")
    deleted_count: int = Field(..., description="Количество удалённых записей")


class ProcessingTimeStats(BaseModel):
    """Статистика времени обработки."""
    
    mean_ms: float = Field(..., description="Среднее время (мс)")
    p50_ms: float = Field(..., description="Медиана / 50-й перцентиль (мс)")
    p95_ms: float = Field(..., description="95-й перцентиль (мс)")
    p99_ms: float = Field(..., description="99-й перцентиль (мс)")
    min_ms: float = Field(..., description="Минимальное время (мс)")
    max_ms: float = Field(..., description="Максимальное время (мс)")


class NumericStats(BaseModel):
    """Статистика числового параметра."""
    
    mean: float = Field(..., description="Среднее значение")
    min: float = Field(..., description="Минимум")
    max: float = Field(..., description="Максимум")
    std: float = Field(..., description="Стандартное отклонение")


class InputStats(BaseModel):
    """Статистика входных данных."""
    
    age: NumericStats = Field(..., description="Статистика возраста")
    height: NumericStats = Field(..., description="Статистика роста")
    weight: NumericStats = Field(..., description="Статистика веса")
    sex_distribution: dict = Field(..., description="Распределение по полу")
    heart_axis_distribution: dict = Field(..., description="Распределение по оси сердца")


class StatsResponse(BaseModel):
    """Ответ со статистикой запросов."""
    
    total_requests: int = Field(..., description="Общее количество запросов")
    successful_requests: int = Field(..., description="Количество успешных запросов")
    failed_requests: int = Field(..., description="Количество неуспешных запросов")
    success_rate: Optional[float] = Field(None, description="Процент успешных запросов")
    request_type_distribution: Optional[dict] = Field(None, description="Распределение по типам запросов")
    processing_time: Optional[ProcessingTimeStats] = Field(None, description="Статистика времени обработки")
    input_stats: Optional[InputStats] = Field(None, description="Статистика входных данных")
    class_distribution: Optional[dict] = Field(None, description="Распределение предсказанных классов")


# схемы для авторизации

class Token(BaseModel):
    """Ответ с JWT токеном."""
    
    access_token: str = Field(..., description="JWT токен доступа")
    token_type: str = Field(default="bearer", description="Тип токена")


class TokenData(BaseModel):
    """Данные из токена."""
    
    username: Optional[str] = None
    role: Optional[str] = None


class UserLogin(BaseModel):
    """Данные для входа."""
    
    username: str = Field(..., description="Имя пользователя")
    password: str = Field(..., description="Пароль")


class UserResponse(BaseModel):
    """Информация о пользователе."""
    
    username: str = Field(..., description="Имя пользователя")
    role: str = Field(..., description="Роль пользователя")
