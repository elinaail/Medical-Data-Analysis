"""
Модуль для обучения модели LinearSVC_OvR для мультиклассовой классификации ЭКГ.
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Предобработка данных для обучения модели.
    
    - Удаляет строки, где NORM сочетается с другими диагнозами
    - Создает комбинированные классы (combo_class)
    - Кодирует combo_class в числовой индекс (combo_idx)
    - Применяет One Hot Encoding к столбцу heart_axis
    
    Args:
        df: Исходный DataFrame с данными
        
    Returns:
        Обработанный DataFrame
    """
    logger.info("=" * 60)
    logger.info("Начало предобработки данных")
    logger.info("=" * 60)

    def make_combo_class(row):
        letters = "".join(
            mapping[col] for col in ["MI", "HYP", "CD", "STTC"] if row[col] == 1
        )
        if letters == "":
            return "N"
        return letters

    def fill_rand_based_q(df, cols=["height", "weight"], group="sex"):
        df = df.copy()
        np.random.seed(42)
        for col in cols:

            q = df.groupby(group)[col].quantile([0.1, 0.9]).unstack()


            for s in q.index:
                low, high = q.loc[s, 0.1], q.loc[s, 0.9]
                mask = (df[group] == s) & (df[col].isna())

                if pd.isna(low) or pd.isna(high) or low == high:
                    continue

                df.loc[mask, col] = np.random.uniform(low, high, mask.sum())

        return df

    # предобработка данных с этапа EDA
    df = df.drop(["pacemaker", "extra_beats", "infarction_stadium1", "infarction_stadium2"], axis=1)
    df = fill_rand_based_q(df)
    df["heart_axis"] = df.heart_axis.fillna("NO_DATA")
    df['age'] = df['age'].replace(300, 98)
    df = df[df["height"] >= 90]
    df = df[df["weight"] > 25]
    df = df[df["diagnostic_superclass"].apply(lambda x: len(x) > 0)]
    unique_diagnostic_superclasses = sorted({lbl for row in df["diagnostic_superclass"] for lbl in row})
    for lbl in unique_diagnostic_superclasses:
        df[f"{lbl}"] = df["diagnostic_superclass"].apply(lambda x: int(lbl in x))
    
    final_columns = [
        "ecg_id", "patient_id", "recording_date", "age", "sex", "height", "weight",
        "heart_axis", "ecg_signals", "CD", "HYP", "MI", "NORM",
        "STTC", "diagnostic_superclass", "strat_fold"
    ]
    df = df[final_columns]

    # удаляем строки, где NORM сочетается с другими диагнозами
    mask_bad = (
        (df["NORM"] == 1)
        & ((df["CD"] == 1) | (df["HYP"] == 1) | (df["MI"] == 1) | (df["STTC"] == 1))
    )
    df = df.loc[~mask_bad].reset_index(drop=True)
    logger.info(f"Удалено строк с некорректным сочетанием NORM: {mask_bad.sum()}")
    
    # создаем комбинированные классы
    mapping = {
        "CD": "C",
        "HYP": "H",
        "MI": "M",
        "STTC": "S",
    }

    df["combo_class"] = df.apply(make_combo_class, axis=1)
    
    # кодируем combo_class в числовой индекс
    df["combo_idx"], combo_uniques = pd.factorize(df["combo_class"])
        
    # One Hot Encoding для heart_axis
    df = pd.get_dummies(df, columns=["heart_axis"], prefix="heart_axis")
    
    logger.info(f"Финальный размер данных после предобработки: {df.shape}")
    logger.info("Предобработка данных завершена")
    
    return df


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Извлекает признаки из ЭКГ сигналов.
    
    Создает статистические, частотные и энергетические признаки
    для каждого из 12 отведений ЭКГ.
    
    Args:
        df: DataFrame с колонкой ecg_signals
        
    Returns:
        DataFrame с извлеченными признаками (без колонки ecg_signals)
    """
    logger.info("=" * 60)
    logger.info("Начало извлечения признаков из ЭКГ сигналов")
    logger.info("=" * 60)
    
    def extract_ecg_features(ecg):
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

            # энергетические признаки
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

    # извлекаем признаки для всех записей
    feature_rows = []
    total_rows = len(df)
    
    logger.info(f"Извлечение признаков для {total_rows} записей...")
    
    for idx, row in df.iterrows():
        ecg = row["ecg_signals"]
        feats = extract_ecg_features(ecg)
        feature_rows.append(feats)
        
        # Логируем прогресс каждые 1000 записей
        if (idx + 1) % 1000 == 0:
            logger.info(f"Обработано {idx + 1}/{total_rows} записей")

    ecg_features_df = pd.DataFrame(feature_rows)

    df_final = pd.concat(
        [
            df.drop(columns=["ecg_signals"]).reset_index(drop=True),
            ecg_features_df.reset_index(drop=True),
        ],
        axis=1,
    )

    logger.info(f"Финальный датасет с признаками: {df_final.shape}")
    logger.info("Извлечение признаков завершено")

    return df_final


def get_train_test(df_final: pd.DataFrame) -> tuple:
    """
    Разбивает данные на train и test по столбцу strat_fold.
    
    1-8 фолды -> train
    9-10 фолды -> test
    
    Args:
        df_final: DataFrame с признаками
        
    Returns:
        Tuple (X_train, X_test, y_train, y_test)
    """
    logger.info("=" * 60)
    logger.info("Разбиение данных на train/test")
    logger.info("=" * 60)
    
    df_final["strat_fold"] = df_final["strat_fold"].astype(int)

    train_mask = df_final["strat_fold"].between(1, 8)
    test_mask = df_final["strat_fold"].between(9, 10)

    df_train = df_final[train_mask].copy()
    df_test = df_final[test_mask].copy()

    n_train = len(df_train)
    n_test = len(df_test)
    n_total = n_train + n_test

    logger.info(f"Размер TRAIN: {df_train.shape}")
    logger.info(f"Размер TEST: {df_test.shape}")
    logger.info(f"Доля train: {(n_train / n_total):.2f}")
    logger.info(f"Доля test: {(n_test / n_total):.2f}")

    # проверка на пересечение пациентов
    train_patients = set(df_train["patient_id"])
    test_patients = set(df_test["patient_id"])
    cross_of_train_test = train_patients & test_patients
    logger.info(f"Пересечение пациентов TRAIN & TEST: {len(cross_of_train_test)}")

    # формируем признаки
    base_feats = ["age", "sex", "height", "weight"]
    heart_feats = [c for c in df_final.columns if c.startswith("heart_axis_")]
    ecg_feats = [
        c
        for c in df_final.columns
        if c.startswith("lead") or c in ["rr_mean", "rr_std", "hr_mean"]
    ]
    feature_cols = base_feats + heart_feats + ecg_feats

    logger.info(f"Количество признаков: {len(feature_cols)}")

    X_train = df_train[feature_cols]
    y_train = df_train["combo_idx"]

    X_test = df_test[feature_cols]
    y_test = df_test["combo_idx"]

    logger.info(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    logger.info(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    logger.info("Разбиение данных завершено")

    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """
    Обучает модель LinearSVC_OvR для мультиклассовой классификации.
    
    Args:
        X_train: Признаки для обучения
        y_train: Целевая переменная для обучения
        
    Returns:
        Обученный Pipeline с моделью
    """
    logger.info("=" * 60)
    logger.info("Обучение модели LinearSVC_OvR")
    logger.info("=" * 60)
    
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", OneVsRestClassifier(
            LinearSVC(
                class_weight="balanced",
                random_state=42
            )
        ))
    ])
    
    logger.info("Начало обучения модели...")
    model.fit(X_train, y_train)
    logger.info("Обучение модели завершено")
    
    return model


def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_train: pd.Series
) -> dict:
    """
    Оценивает качество модели на тестовых данных.
    
    Args:
        model: Обученная модель
        X_test: Признаки для тестирования
        y_test: Целевая переменная для тестирования
        y_train: Целевая переменная обучения (для определения всех классов)
        
    Returns:
        Словарь с метриками качества
    """
    logger.info("=" * 60)
    logger.info("Оценка качества модели")
    logger.info("=" * 60)
    
    y_pred = model.predict(X_test)
    
    actual_labels = sorted(np.unique(np.concatenate([y_train, y_test, y_pred])))
    logger.info(f"Количество классов: {len(actual_labels)}")

    accuracy = accuracy_score(y_test, y_pred)
    
    prec_micro = precision_score(y_test, y_pred, average="micro", zero_division=0)
    rec_micro = recall_score(y_test, y_pred, average="micro", zero_division=0)
    f1_micro = f1_score(y_test, y_pred, average="micro", zero_division=0)
    
    prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    
    prec_weighted = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec_weighted = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    logger.info(f"Accuracy:           {accuracy:.3f}")
    logger.info(f"F1-micro:           {f1_micro:.3f}")
    logger.info(f"F1-macro:           {f1_macro:.3f}")
    logger.info(f"F1-weighted:        {f1_weighted:.3f}")
    logger.info(f"Precision-micro:    {prec_micro:.3f}")
    logger.info(f"Recall-micro:       {rec_micro:.3f}")

    logger.info("\nClassification report:")
    report = classification_report(
        y_test, y_pred, labels=actual_labels, zero_division=0, digits=3
    )
    logger.info(f"\n{report}")

    results = {
        "model": "LinearSVC_OvR",
        "accuracy": accuracy,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "prec_micro": prec_micro,
        "rec_micro": rec_micro,
        "prec_macro": prec_macro,
        "rec_macro": rec_macro,
        "num_classes": len(actual_labels),
    }
    
    logger.info("Оценка модели завершена")
    
    return results


def main(data_path: str, model_output_path: str = "model.pkl") -> None:
    """
    Оркестрирует весь процесс обучения модели.
    
    Args:
        data_path: Путь к файлу с данными (pickle)
        model_output_path: Путь для сохранения обученной модели
    """
    logger.info("=" * 60)
    logger.info("ЗАПУСК ПАЙПЛАЙНА ОБУЧЕНИЯ МОДЕЛИ")
    logger.info("=" * 60)
    
    logger.info(f"Загрузка данных ...")
    df = pd.read_pickle(data_path)
    logger.info(f"Данные загружены. Размер: {df.shape}")
    
    df = preprocess(df)
    df_final = extract_features(df)
    X_train, X_test, y_train, y_test = get_train_test(df_final)
    model = train_model(X_train, y_train)
    results = evaluate_model(model, X_test, y_test, y_train)
    
    logger.info("=" * 60)
    logger.info("сохранение модели")
    logger.info("=" * 60)
    
    model_path = Path(model_output_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    logger.info(f"Модель сохранена в: {model_path.absolute()}")
    
    logger.info("=" * 60)
    logger.info("ПАЙПЛАЙН ОБУЧЕНИЯ ЗАВЕРШЕН УСПЕШНО")
    logger.info("=" * 60)
    
    return results


if __name__ == "__main__":
    # вычисляем пути относительно расположения скрипта
    SCRIPT_DIR = Path(__file__).parent
    DATA_PATH = SCRIPT_DIR / "../datasets/clean_ptbxl_with_ecg_n_diagnostic_superclass.pkl"
    MODEL_OUTPUT_PATH = SCRIPT_DIR / "../app/ml_artifacts/model.pkl"

    DATA_PATH = str(DATA_PATH)
    MODEL_OUTPUT_PATH = str(MODEL_OUTPUT_PATH)
    
    main(DATA_PATH, MODEL_OUTPUT_PATH)
