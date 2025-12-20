"""
Модуль для работы с JWT токенами и авторизацией.
"""

from datetime import datetime, timedelta
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from app.core import settings, logger

# контекст для хэширования паролей
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 схема для получения токена из заголовка
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_V1_PREFIX}/auth/login")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Проверяет пароль."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Хэширует пароль."""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Создаёт JWT токен.
    
    Args:
        data: Данные для включения в токен
        expires_delta: Время жизни токена
        
    Returns:
        JWT токен
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode, 
        settings.JWT_SECRET_KEY, 
        algorithm=settings.JWT_ALGORITHM
    )
    return encoded_jwt


def decode_token(token: str) -> Optional[dict]:
    """
    Декодирует JWT токен.
    
    Args:
        token: JWT токен
        
    Returns:
        Данные из токена или None при ошибке
    """
    try:
        payload = jwt.decode(
            token, 
            settings.JWT_SECRET_KEY, 
            algorithms=[settings.JWT_ALGORITHM]
        )
        return payload
    except JWTError:
        return None


def authenticate_user(username: str, password: str) -> Optional[dict]:
    """
    Аутентифицирует пользователя.
    
    Простая реализация с одним администратором из настроек.
    
    Args:
        username: Имя пользователя
        password: Пароль
        
    Returns:
        Данные пользователя или None
    """
    # Простая проверка администратора из настроек
    if username == settings.ADMIN_USERNAME and password == settings.ADMIN_PASSWORD:
        return {
            "username": username,
            "role": "admin"
        }
    return None


async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    """
    Получает текущего пользователя из токена.
    
    Args:
        token: JWT токен из заголовка Authorization
        
    Returns:
        Данные пользователя
        
    Raises:
        HTTPException 401: Неверный токен
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Неверные учётные данные",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    payload = decode_token(token)
    if payload is None:
        logger.warning("Попытка доступа с невалидным токеном")
        raise credentials_exception
    
    username: str = payload.get("sub")
    if username is None:
        raise credentials_exception
    
    return {"username": username, "role": payload.get("role", "user")}


async def get_current_admin(current_user: dict = Depends(get_current_user)) -> dict:
    """
    Проверяет, что текущий пользователь - администратор.
    
    Args:
        current_user: Текущий пользователь
        
    Returns:
        Данные администратора
        
    Raises:
        HTTPException 403: Недостаточно прав
    """
    if current_user.get("role") != "admin":
        logger.warning(f"Пользователь {current_user.get('username')} попытался получить доступ к admin-ресурсу")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Недостаточно прав. Требуется роль администратора."
        )
    return current_user
