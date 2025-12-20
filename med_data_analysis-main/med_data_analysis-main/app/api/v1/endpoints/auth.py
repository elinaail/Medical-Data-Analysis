"""
Auth эндпоинты для JWT авторизации.
"""

from datetime import timedelta

from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.security import OAuth2PasswordRequestForm

from app.core import logger, settings, authenticate_user, create_access_token, get_current_user
from app.models import Token, UserResponse

router = APIRouter()


@router.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Авторизация пользователя.
    
    Принимает username и password в формате form-data.
    Возвращает JWT токен для доступа к защищённым эндпоинтам.
    
    Args:
        form_data: Данные формы с username и password
        
    Returns:
        JWT токен
        
    Raises:
        HTTPException 401: Неверные учётные данные
    """
    user = authenticate_user(form_data.username, form_data.password)
    
    if not user:
        logger.warning(f"Неудачная попытка входа: {form_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверное имя пользователя или пароль",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"], "role": user["role"]},
        expires_delta=access_token_expires
    )
    
    logger.info(f"Пользователь {user['username']} успешно авторизован")
    
    return Token(access_token=access_token, token_type="bearer")


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: dict = Depends(get_current_user)):
    """
    Получение информации о текущем пользователе.
    
    Требует авторизации через JWT токен в заголовке Authorization.
    
    Returns:
        Информация о пользователе
    """
    return UserResponse(
        username=current_user["username"],
        role=current_user["role"]
    )
