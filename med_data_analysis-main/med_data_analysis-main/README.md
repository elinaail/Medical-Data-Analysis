# ECG ML Classification Service

FastAPI сервис для классификации ЭКГ сигналов с использованием машинного обучения.

## Структура проекта

```
med_data_analysis/
├── alembic/                    # миграции базы данных
│   ├── versions/               # файлы миграций
│   └── env.py                  # конфигурация Alembic
├── app/                        # основное приложение
│   ├── api/                    # API слой
│   │   ├── deps.py             # зависимости (Depends)
│   │   └── v1/                 # API версии 1
│   │       ├── router.py       # главный роутер
│   │       └── endpoints/      # эндпоинты
│   │           ├── auth.py     # авторизация
│   │           ├── health.py   # Health check
│   │           ├── history.py  # история запросов
│   │           └── prediction.py # предсказания
│   ├── core/                   # ядро приложения
│   │   ├── config.py           # настройки
│   │   ├── constants.py        # константы
│   │   ├── logging.py          # логирование
│   │   └── security.py         # JWT авторизация
│   ├── database/               # база данных
│   │   ├── connection.py       # подключение к БД
│   │   └── models.py           # SQLAlchemy модели
│   ├── models/                 # Pydantic схемы
│   │   └── schemas.py          # схемы запросов/ответов
│   ├── services/               # бизнес-логика
│   │   ├── prediction.py       # сервис предсказаний
│   │   └── history.py          # сервис истории
│   ├── ml_artifacts/           # ML модели
│   │   └── model.pkl           # обученная модель
│   ├── tests/                  # тесты
│   └── main.py                 # точка входа
├── datasets/                   # датасеты
├── model_training/             # скрипты обучения
├── notebooks/                  # ноутбуки для тестирования приложения
├── alembic.ini                 # конфиг Alembic
├── pyproject.toml              # зависимости проекта
└── request_history.db          # SQLite база данных
```

## Установка зависимостей


**1. Установка зависимостей**

Для успешной установки проекта рекомендуется использовать современный менеджр пакетов uv и Python >= 3.12:
- [Ссылка на установку uv](https://docs.astral.sh/uv/)


```bash
# вариант 1: с помощью uv (рекомендуется)
uv sync

# вариант 2: классический pip + venv
python -m venv venv

# активация виртуальной среды
venv\Scripts\activate      # Windows
source .venv/bin/activate   # Linux / macOS

# установка зависимостей проекта
pip install -e .
```
**Важно**
- Если вы используете uv, то запуск всех скриптов рекомендуется выполнять при помощи uv, например:
```bash
uv run python -m app.main
```

**2. Применение миграций базы данных**

После установки зависимостей необходимо создать структуру базы данных. Проект использует SQLite для хранения истории запросов и Alembic для управления схемой БД (миграциями)

Миграции создают все необходимые таблицы, например, `request_history` с правильной структурой столбцов. Без этого шага сервис не сможет сохранять историю запросов и будет выдавать ошибки при обращении к БД.

```bash
# при помощи uv
uv run alembic upgrade head

# без uv
alembic upgrade head
```

После выполнения команды будет создан файл `request_history.db` с настроенной схемой.

## Запуск сервиса

Перед запуском сервера необходимо обучить ML модель и сохранить ее. Для этого необходимо запустить файл `train_model.py`

```bash
# при помощи uv
uv run python model_training/train_model.py

# без uv
python model_training/train_model.py
```

Убедитесь, что модель успешно сохранена в `app/ml_artifacts`

После установки зависимостей и применения миграций можно запустить сервер. Приложение использует Uvicorn — высокопроизводительный ASGI сервер для FastAPI.

```bash
# при помощи uv
uv run python -m app.main

# без uv
python -m app.main
```

При запуске сервер:
1. Загружает ML модель из `app/ml_artifacts/model.pkl`
2. Подключается к базе данных SQLite
3. Запускает HTTP сервер по адресу `http://0.0.0.0:8000`

После успешного запуска в консоли появится сообщение:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

**Документация API** генерируется автоматически и доступна по адресам:
- **Swagger UI:** http://localhost:8000/docs — интерактивная документация с возможностью тестирования
- **ReDoc:** http://localhost:8000/redoc — альтернативный формат документации


## FastAPI эндпоинты

Сервис предоставляет REST API для классификации ЭКГ сигналов. Все эндпоинты доступны по базовому адресу `http://localhost:8000/api/v1`.

### Обзор эндпоинтов

| Эндпоинт | Метод | Описание | Авторизация |
|----------|-------|----------|-------------|
| `/health` | GET | Проверка состояния сервиса | Нет |
| `/forward` | POST | Предсказание для одной ЭКГ | Нет |
| `/forward/batch` | POST | Batch предсказание | Нет |
| `/history` | GET | Получение истории запросов | Нет |
| `/history` | DELETE | Удаление истории | Admin |
| `/stats` | GET | Статистика запросов | Admin |
| `/auth/login` | POST | Получение JWT токена | Нет |
| `/auth/me` | GET | Информация о пользователе | Да |


### Health Check (GET /health)
Эндпоинт для проверки работоспособности сервиса. Используется для мониторинга и health checks в контейнерных окружениях (Docker, Kubernetes).

**Пример запроса:**
```bash
curl http://localhost:8000/api/v1/health
```

**Ответ:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

| Поле | Тип | Описание |
|------|-----|----------|
| `status` | string | Статус сервиса (`healthy`) |
| `model_loaded` | boolean | Загружена ли ML модель |


### Предсказания (POST /forward)
Основной эндпоинт для классификации ЭКГ. Принимает данные пациента и ЭКГ сигнал, возвращает предсказанный класс заболевания.

**Параметры запроса:**
| Поле | Тип | Описание |
|------|-----|----------|
| `age` | float | Возраст пациента |
| `sex` | int | Пол (0 - женский, 1 - мужской) |
| `height` | float | Рост в см |
| `weight` | float | Вес в кг |
| `heart_axis` | string | Ось сердца: `ALAD`, `ARAD`, `AXL`, `AXR`, `LAD`, `MID`, `RAD`, `SAG`, `NO_DATA` |
| `ecg_signal` | array | ЭКГ сигнал [5000, 12] - 5000 точек × 12 отведений |

**Пример запроса:**
```bash
curl -X POST http://localhost:8000/api/v1/forward \
  -H "Content-Type: application/json" \
  -d '{
    "age": 55,
    "sex": 1,
    "height": 175,
    "weight": 80,
    "heart_axis": "MID",
    "ecg_signal": [[0.1, 0.2, ...], ...]
  }'
```

**Успешный ответ (200):**
```json
{
  "class_index": 0,
  "class_name": "N",
  "class_description": "Normal ECG"
}
```

**Коды ошибок:**
- `400 Bad Request` - Неверный формат данных (отсутствуют обязательные поля, неверное значение `heart_axis`, неверное количество отведений)
- `403 Forbidden` - Модель не смогла обработать данные (внутренняя ошибка ML модели)

### Предсказания (POST /forward/batch)
Эндпоинт для batch обработки нескольких ЭКГ записей за один запрос. Полезен для массовой обработки данных.

**Пример запроса:**
```bash
curl -X POST http://localhost:8000/api/v1/forward/batch \
  -H "Content-Type: application/json" \
  -d '{
    "samples": [
      {"age": 55, "sex": 1, "height": 175, "weight": 80, "heart_axis": "MID", "ecg_signal": [...]},
      {"age": 60, "sex": 0, "height": 165, "weight": 65, "heart_axis": "LAD", "ecg_signal": [...]}
    ]
  }'
```

**Ответ:**
```json
{
  "predictions": [
    {"class_index": 0, "class_name": "N", "class_description": "Normal ECG"},
    {"class_index": 1, "class_name": "S", "class_description": "STTC"}
  ]
}
```

### История запросов (GET /history)
Сервис автоматически сохраняет все запросы на предсказание в базу данных SQLite. Это позволяет отслеживать использование API, анализировать входные данные и диагностировать ошибки.

Получение истории запросов с поддержкой пагинации. Публичный эндпоинт, не требует авторизации.

**Query параметры:**
| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `limit` | int | 100 | Количество записей (1-1000) |
| `offset` | int | 0 | Смещение для пагинации |

**Пример запроса:**
```bash
curl "http://localhost:8000/api/v1/history?limit=10&offset=0"
```

**Ответ:**
```json
{
  "total": 150,
  "records": [
    {
      "id": 1,
      "created_at": "2025-12-14T10:30:00",
      "age": 55.0,
      "sex": 1,
      "height": 175.0,
      "weight": 80.0,
      "heart_axis": "MID",
      "predicted_class_index": 0,
      "predicted_class_name": "N",
      "predicted_class_description": "Normal ECG",
      "success": true,
      "error_message": null,
      "request_type": "single"
    }
  ]
}
```

| Поле ответа | Описание |
|-------------|----------|
| `total` | Общее количество записей в БД |
| `records` | Массив записей истории |
| `request_type` | Тип запроса: `single` или `batch` |
| `success` | Успешность выполнения предсказания |
| `error_message` | Сообщение об ошибке (если `success=false`) |

### История запросов (DELETE /history)
Удаление всей истории запросов. **Требует авторизации администратора!**. Используется для очистки базы данных, например, перед production деплоем или для соответствия требованиям GDPR.

**Пример запроса:**
```bash
curl -X DELETE http://localhost:8000/api/v1/history \
  -H "Authorization: Bearer <JWT_TOKEN>"
```

**Ответ:**
```json
{
  "message": "История успешно удалена",
  "deleted_count": 150
}
```

### Статистика запрсов (GET /stats)
Агрегированная статистика по всем запросам. **Требует авторизации администратора.** Эндпоинт предоставляет:
- Общую статистику запросов (успешные/неуспешные)
- Статистику времени обработки (mean, медиана, перцентили p95, p99)
- Характеристики входных данных (возраст, рост, вес, распределение по полу и оси сердца)
- Распределение предсказанных классов

**Пример запроса:**
```bash
curl http://localhost:8000/api/v1/stats \
  -H "Authorization: Bearer <JWT_TOKEN>"
```

**Ответ:**
```json
{
  "total_requests": 1000,
  "successful_requests": 980,
  "failed_requests": 20,
  "processing_time": {
    "mean_ms": 45.5,
    "p50_ms": 42.0,
    "p95_ms": 78.0,
    "p99_ms": 120.0,
    "min_ms": 15.0,
    "max_ms": 250.0
  },
  "input_stats": {
    "age": {"mean": 55.2, "min": 18.0, "max": 90.0, "std": 15.3},
    "height": {"mean": 170.5, "min": 150.0, "max": 195.0, "std": 10.2},
    "weight": {"mean": 75.0, "min": 45.0, "max": 120.0, "std": 15.0},
    "sex_distribution": {"0": 450, "1": 550},
    "heart_axis_distribution": {"MID": 300, "LAD": 250, "RAD": 200, ...}
  },
  "class_distribution": {"N": 500, "S": 200, "M": 150, ...}
}
```

| Поле | Описание |
|------|----------|
| `processing_time.p50_ms` | Медиана времени обработки (50% запросов быстрее) |
| `processing_time.p95_ms` | 95-й перцентиль (5% запросов медленнее) |
| `processing_time.p99_ms` | 99-й перцентиль (1% запросов медленнее) |
| `class_distribution` | Количество предсказаний каждого класса |

## Авторизация JWT

API использует JWT (JSON Web Tokens) для авторизации защищённых эндпоинтов. JWT — это компактный, безопасный способ передачи информации между сторонами в виде JSON-объекта с цифровой подписью.

### Как работает JWT авторизация

```
┌───────────────┐      1. POST /auth/login       ┌───────────────┐
│               │ ─────────────────────────────▶               
│    Клиент     │      username + password            Сервер     
│               │ ◀─────────────────────────────               
└───────────────┘        2. JWT токен            └───────────────┘
        │
        │  3. Запрос с заголовком
        │     Authorization: Bearer <token>
        ▼
┌───────────────┐      4. Проверка токена        ┌───────────────┐
│               │ ─────────────────────────────▶               
│    Клиент     │                                     Сервер     
│               │ ◀─────────────────────────────               
└───────────────┘    5. Ответ или 401 / 403      └───────────────┘
```

**Процесс авторизации:**
1. Клиент отправляет `username` и `password` на эндпоинт `/auth/login`
2. Сервер проверяет учётные данные и возвращает JWT токен
3. Клиент сохраняет токен и добавляет его в заголовок `Authorization` для защищённых запросов
4. Сервер проверяет подпись токена, срок действия и роль пользователя
5. Если проверка успешна — выполняет запрос, иначе возвращает ошибку 401/403

### Учётные данные по умолчанию

| Параметр | Значение | Описание |
|----------|----------|----------|
| Username | `admin` | Имя пользователя-администратора |
| Password | `admin123` | Пароль по умолчанию |
| Роль | `admin` | Полный доступ ко всем эндпоинтам |

**Важно для production:** Обязательно измените пароль через переменные окружения `ADMIN_USERNAME` и `ADMIN_PASSWORD` или в файле `.env`

### Получение токена (POST /auth/login)
Эндпоинт для аутентификации пользователя. Принимает данные в формате `application/x-www-form-urlencoded` (стандарт OAuth2).

**Параметры запроса:**
| Параметр | Тип | Описание |
|----------|-----|----------|
| `username` | string | Имя пользователя |
| `password` | string | Пароль |

**Пример запроса с curl:**
```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -d "username=admin&password=admin123"
```

**Пример запроса с Python (requests):**
```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/auth/login",
    data={"username": "admin", "password": "admin123"}
)
token = response.json()["access_token"]
print(f"Token: {token}")
```

**Успешный ответ (200):**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsInJvbGUiOiJhZG1pbiIsImV4cCI6MTczNDE4MDAwMH0.xxxxx",
  "token_type": "bearer"
}
```

**Ошибка авторизации (401):**
```json
{
  "detail": "Неверное имя пользователя или пароль"
}
```

### Структура JWT токена
JWT токен состоит из трёх частей, разделённых точками:

```
eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsInJvbGUiOiJhZG1pbiIsImV4cCI6MTczNDE4MDAwMH0.signature
└──────────── Header ────────────┘ └─────────────── Payload ───────────────┘ └─ Signature ─┘
```

**Payload (полезная нагрузка) содержит:**
| Поле | Описание |
|------|----------|
| `sub` | Subject — имя пользователя |
| `role` | Роль пользователя (`admin`) |
| `exp` | Expiration — время истечения токена (Unix timestamp) |

### Использование токена

После получения токена добавляйте его в заголовок `Authorization` с префиксом `Bearer`:

**Формат заголовка:**
```
Authorization: Bearer <ваш_токен>
```

**Пример запроса с curl:**
```bash
# сохраняем токен в переменную
TOKEN="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."

# используем токен для защищённого запроса
curl http://localhost:8000/api/v1/stats \
  -H "Authorization: Bearer $TOKEN"
```

**Пример запроса с Python:**
```python
import requests

token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."

headers = {"Authorization": f"Bearer {token}"}
response = requests.get(
    "http://localhost:8000/api/v1/stats",
    headers=headers
)
print(response.json())
```

### Проверка текущего пользователя (GET /auth/me)
Эндпоинт для проверки валидности токена и получения информации о текущем пользователе. Полезен для отладки и проверки авторизации.

**Пример запроса:**
```bash
curl http://localhost:8000/api/v1/auth/me \
  -H "Authorization: Bearer <JWT_TOKEN>"
```

**Успешный ответ (200):**
```json
{
  "username": "admin",
  "role": "admin"
}
```

**Если токен невалиден (401):**
```json
{
  "detail": "Could not validate credentials"
}
```

### Время жизни токена

| Параметр | Значение | Описание |
|----------|----------|----------|
| Время жизни | **30 минут** | Настраивается через `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` |
| Алгоритм | HS256 | HMAC с SHA-256 |

**Что делать когда токен истёк:**
1. Клиент получает ошибку `401 Unauthorized` с сообщением "Token has expired"
2. Необходимо повторно вызвать `/auth/login` для получения нового токена
3. Использовать новый токен для последующих запросов

- **Совет:** В production рекомендуется реализовать refresh токены для автоматического обновления без повторного ввода пароля.

### Защищённые эндпоинты

| Эндпоинт | Метод | Требуемая роль | Описание |
|----------|-------|----------------|----------|
| `/history` | DELETE | `admin` | Удаление всей истории запросов |
| `/stats` | GET | `admin` | Получение статистики запросов |
| `/auth/me` | GET | любой авторизованный | Информация о текущем пользователе |

**Уровни доступа:**
- **Публичные эндпоинты** — доступны без авторизации (`/health`, `/forward`, `/history GET`)
- **Авторизованные** — требуют валидный JWT токен (`/auth/me`)
- **Административные** — требуют JWT токен с ролью `admin` (`/stats`, `/history DELETE`)

### Коды ошибок авторизации

| Код | Ошибка | Причина | Решение |
|-----|--------|---------|---------|
| `401` | Unauthorized | Токен отсутствует | Добавьте заголовок `Authorization: Bearer <token>` |
| `401` | Unauthorized | Токен истёк | Получите новый токен через `/auth/login` |
| `401` | Unauthorized | Невалидная подпись | Проверьте правильность токена |
| `403` | Forbidden | Недостаточно прав | Требуется роль `admin` |

**Пример ошибки 401:**
```json
{
  "detail": "Could not validate credentials"
}
```

**Пример ошибки 403:**
```json
{
  "detail": "Требуются права администратора"
}
```

## Миграции базы данных
Проект использует **Alembic** для управления схемой базы данных SQLite.

### Основные команды

**Применить все миграции**
```bash
# при помощи uv
uv run alembic upgrade head

# без uv
alembic upgrade head
```

**Откатить последнюю миграцию**
```bash
# при помощи uv
uv run alembic downgrade -1

# без uv
alembic downgrade -1
```

**Откатить все миграции**
```bash
# при помощи uv
uv run alembic downgrade base

# без uv
alembic downgrade base
```

**Показать текущую версию БД**
```bash
# при помощи uv
uv run alembic current

# без uv
alembic current
```

**Показать историю миграций**
```bash
# при помощи uv
uv run alembic history

# без uv
alembic history
```

### Создание новой миграции
После изменения моделей в `app/database/models.py`:

```bash
# при помощи uv
uv run alembic revision --autogenerate -m "Описание изменений"

# без uv
alembic revision --autogenerate -m "Описание изменений"
```

**Пример:** добавление нового поля в модель:

1. Измените модель в `app/database/models.py`:
```python
class RequestHistory(Base):
    # ... существующие поля
    new_field: Mapped[str] = mapped_column(String(100), nullable=True)
```

2. Создайте миграцию:
```bash
# при помощи uv
uv run alembic revision --autogenerate -m "Add new_field to request_history"

# без uv
alembic revision --autogenerate -m "Add new_field to request_history"
```

3. Примените миграцию:
```bash
# при помощи uv
uv run alembic upgrade head

# без uv
alembic upgrade head
```

### Структура файла миграции
Миграции хранятся в `alembic/versions/`:

```python
# alembic/versions/c48a9029ad04_initial_migration.py

def upgrade() -> None:
    """Применение миграции."""
    op.create_table('request_history',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        # ... другие колонки
    )

def downgrade() -> None:
    """Откат миграции."""
    op.drop_table('request_history')
```

### Работа с существующей базой данных
Если база данных уже существует и вы хотите начать использовать миграции:

1. Удалите старую базу:
```bash
rm request_history.db
```

2. Примените миграции:
```bash
# при помощи uv
uv run alembic upgrade head

# без uv
alembic upgrade head
```

## Конфигурация FastAPI сервиса
Настройки задаются через  файл `core.config` или переменные окружения или файл `.env`:

| Переменная | По умолчанию | Описание |
|------------|--------------|----------|
| `HOST` | `0.0.0.0` | Хост сервера |
| `PORT` | `8000` | Порт сервера |
| `DATABASE_URL` | `sqlite+aiosqlite:///./request_history.db` | URL базы данных |
| `JWT_SECRET_KEY` | `your-super-secret-key...` | Секретный ключ JWT |
| `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` | `30` | Время жизни токена (мин) |
| `ADMIN_USERNAME` | `admin` | Имя администратора |
| `ADMIN_PASSWORD` | `admin123` | Пароль администратора |
| `MODEL_PATH` | `app/ml_artifacts/model.pkl` | Путь к ML модели |
| `LOG_LEVEL` | `INFO` | Уровень логирования |

### Пример .env файла
Если вы хотите использовать `.env` (рекомендуется), то вот пример его описания: 

```env
JWT_SECRET_KEY=my-super-secret-production-key
ADMIN_USERNAME=admin
ADMIN_PASSWORD=secure-password-123
LOG_LEVEL=WARNING
```