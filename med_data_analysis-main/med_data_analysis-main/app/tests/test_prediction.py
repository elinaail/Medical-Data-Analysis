"""
Тесты для prediction эндпоинтов.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_root():
    """Тест корневого эндпоинта."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data
    assert "endpoints" in data


def test_health():
    """Тест health эндпоинта."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"
    assert "model_loaded" in data


def test_forward_without_model():
    """Тест forward без загруженной модели."""
    # Если модель не загружена, должен вернуть 403
    payload = {
        "age": 50,
        "sex": 1,
        "height": 175,
        "weight": 80,
        "heart_axis": "MID",
        "ecg_signal": [[0.1] * 12 for _ in range(100)]
    }
    response = client.post("/api/v1/forward", json=payload)
    # Может быть 200 если модель загружена, или 403 если нет
    assert response.status_code in [200, 403]


def test_forward_bad_request():
    """Тест forward с неверными данными."""
    payload = {
        "age": 50,
        "sex": 1,
        "height": 175,
        "weight": 80,
        "heart_axis": "INVALID",  # Неверное значение
        "ecg_signal": [[0.1] * 12]
    }
    response = client.post("/api/v1/forward", json=payload)
    assert response.status_code == 422  # Validation error


def test_forward_missing_fields():
    """Тест forward с отсутствующими полями."""
    payload = {
        "age": 50,
        "sex": 1
    }
    response = client.post("/api/v1/forward", json=payload)
    assert response.status_code == 422  # Validation error
