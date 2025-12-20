"""
Настройка логирования.
"""

import logging
import sys

from app.core.config import settings


def setup_logging() -> logging.Logger:
    """Настраивает и возвращает логгер."""
    
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL),
        format=settings.LOG_FORMAT,
        datefmt=settings.LOG_DATE_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger("ecg_api")

# инициализация логгера
logger = setup_logging()
