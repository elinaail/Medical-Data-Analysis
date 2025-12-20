"""
Сервис для предсказания ЭКГ классов.
"""

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.signal import welch

from app.core import (
    logger,
    settings,
    CLASS_MAPPING,
    CLASS_DESCRIPTIONS,
    VALID_HEART_AXIS,
    ALL_FEATURES,
)
from app.models.schemas import ECGInput, PredictionResponse


class PredictionService:
    """Сервис для работы с моделью предсказания."""
    
    def __init__(self):
        self.model = None
    
    def load_model(self, model_path: Optional[str] = None) -> None:
        """
        Загружает модель из pickle файла.
        
        Args:
            model_path: Путь к файлу модели
        """
        path = Path(model_path or settings.MODEL_PATH)
        
        if not path.exists():
            logger.error(f"Файл модели не найден: {path}")
            raise FileNotFoundError(f"Файл модели не найден: {path}")
        
        logger.info(f"Загрузка модели из: {path}")
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        logger.info("Модель успешно загружена")
    
    def is_model_loaded(self) -> bool:
        """Проверяет, загружена ли модель."""
        return self.model is not None
    
    def extract_ecg_features(self, ecg: np.ndarray) -> dict:
        """
        Извлекает признаки из ЭКГ сигнала.
        
        Args:
            ecg: numpy array размером (N, 12) - N временных точек для 12 отведений
            
        Returns:
            Словарь с признаками для всех отведений
        """
        features = {}
        num_leads = ecg.shape[1]

        for lead_idx in range(num_leads):
            lead = ecg[:, lead_idx]
            prefix = f"lead{lead_idx+1}"

            # базовые статистики
            features[f"{prefix}_mean"] = np.mean(lead)
            features[f"{prefix}_median"] = np.median(lead)
            features[f"{prefix}_std"] = np.std(lead)
            features[f"{prefix}_min"] = np.min(lead)
            features[f"{prefix}_max"] = np.max(lead)
            features[f"{prefix}_range"] = np.max(lead) - np.min(lead)
            features[f"{prefix}_rms"] = np.sqrt(np.mean(lead**2))

            # форма сигнала
            features[f"{prefix}_skew"] = skew(lead)
            features[f"{prefix}_kurt"] = kurtosis(lead)

            # Энергетические признаки
            features[f"{prefix}_energy"] = np.sum(lead**2)
            features[f"{prefix}_norm_energy"] = np.sum(lead**2) / len(lead)

            # частотные признаки на основе Welch
            freqs, psd = welch(lead, fs=500)
            features[f"{prefix}_dom_freq"] = freqs[np.argmax(psd)]
            features[f"{prefix}_spec_entropy"] = -np.sum(
                (psd / np.sum(psd)) * np.log(psd / np.sum(psd) + 1e-12)
            )

            # LF/HF диапазоны
            lf_band = (freqs >= 0.04) & (freqs <= 0.15)
            hf_band = (freqs >= 0.15) & (freqs <= 0.40)

            lf_energy = np.sum(psd[lf_band])
            hf_energy = np.sum(psd[hf_band])

            features[f"{prefix}_lf_energy"] = lf_energy
            features[f"{prefix}_hf_energy"] = hf_energy
            features[f"{prefix}_lf_hf_ratio"] = lf_energy / (hf_energy + 1e-6)

        return features
    
    def prepare_features(self, input_data: ECGInput) -> pd.DataFrame:
        """
        Подготавливает входные данные для модели.
        
        Args:
            input_data: Входные данные
            
        Returns:
            DataFrame с признаками в правильном порядке
        """
        features = {}
        
        # базовые признаки
        features["age"] = input_data.age
        features["sex"] = input_data.sex
        features["height"] = input_data.height
        features["weight"] = input_data.weight
        
        # One Hot Encoding для heart_axis
        for axis in VALID_HEART_AXIS:
            features[f"heart_axis_{axis}"] = 1 if input_data.heart_axis == axis else 0
        
        # извлекаем ECG признаки из сигнала
        ecg_array = np.array(input_data.ecg_signal)
        ecg_features = self.extract_ecg_features(ecg_array)
        features.update(ecg_features)
        
        # создаем DataFrame с правильным порядком колонок
        df = pd.DataFrame([features])
        df = df[ALL_FEATURES]
        
        return df
    
    def predict(self, input_data: ECGInput) -> PredictionResponse:
        """
        Выполняет предсказание для одной записи ЭКГ.
        
        Args:
            input_data: Входные данные
            
        Returns:
            Результат предсказания
        """
        if not self.is_model_loaded():
            raise RuntimeError("Модель не загружена")
        
        features_df = self.prepare_features(input_data)
        logger.info(f"Признаки подготовлены. Shape: {features_df.shape}")
        
        prediction = self.model.predict(features_df)
        class_index = int(prediction[0])
        class_name = CLASS_MAPPING.get(class_index, "UNKNOWN")
        class_description = CLASS_DESCRIPTIONS.get(class_name, "Unknown class")
        
        logger.info(f"Предсказание выполнено: {class_name} ({class_index})")
        
        return PredictionResponse(
            class_index=class_index,
            class_name=class_name,
            class_description=class_description
        )


# создаем глобальный экземпляр сервиса предсказания
prediction_service = PredictionService()
