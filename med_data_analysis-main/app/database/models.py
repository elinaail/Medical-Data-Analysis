"""
SQLAlchemy модели для базы данных.
"""

from datetime import datetime

from sqlalchemy import Integer, String, Float, DateTime, Boolean
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Базовый класс для всех моделей."""
    pass


class RequestHistory(Base):
    """Модель для хранения истории запросов."""
    
    __tablename__ = "request_history"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    
    # время запроса
    created_at: Mapped[datetime] = mapped_column(
        DateTime, 
        default=datetime.utcnow,
        nullable=False
    )
    
    # входные данные (без ecg_signal для экономии места)
    age: Mapped[float] = mapped_column(Float, nullable=False)
    sex: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[float] = mapped_column(Float, nullable=False)
    weight: Mapped[float] = mapped_column(Float, nullable=False)
    heart_axis: Mapped[str] = mapped_column(String(20), nullable=False)
    
    # результат предсказания
    predicted_class_index: Mapped[int] = mapped_column(Integer, nullable=True)
    predicted_class_name: Mapped[str] = mapped_column(String(50), nullable=True)
    predicted_class_description: Mapped[str] = mapped_column(String(200), nullable=True)
    
    # статус запроса
    success: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    error_message: Mapped[str] = mapped_column(String(500), nullable=True)
    
    # тип запроса (single / batch)
    request_type: Mapped[str] = mapped_column(String(20), default="single", nullable=False)
    
    # время обработки запроса в миллисекундах
    processing_time_ms: Mapped[float] = mapped_column(Float, nullable=True)
    
    def __repr__(self) -> str:
        return f"<RequestHistory(id={self.id}, created_at={self.created_at}, class={self.predicted_class_name})>"
