"""
Конфигурация приложения.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Настройки приложения."""
    
    # API
    API_V1_PREFIX: str = "/api/v1"
    PROJECT_NAME: str = "ECG ML Classification API"
    PROJECT_DESCRIPTION: str = "API для классификации ЭКГ сигналов с использованием машинного обучения."
    VERSION: str = "1.0.0"
    
    # ML модель
    MODEL_PATH: str = "app/ml_artifacts/model.pkl"
    
    # конифгурация сервера
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # конфигурация логирования
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(levelname)s - %(message)s"
    LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
    
    # база данных
    DATABASE_URL: str = "sqlite+aiosqlite:///./request_history.db"
    DATABASE_ECHO: bool = False
    
    # JWT настройки
    JWT_SECRET_KEY: str = "your-super-secret-key-change-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # администратор по умолчанию
    ADMIN_USERNAME: str = "admin"
    ADMIN_PASSWORD: str = "admin123"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# инициализация настроек приложения FastAPI
settings = Settings()
