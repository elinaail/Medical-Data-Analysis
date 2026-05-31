"""Модуль предобработки данных PTB-XL."""

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from pipeline.constants import (
    CAT_COL,
    FEATURE_COLS,
    FOLD_COL,
    TARGET_COL,
    TEST_FOLDS,
    TRAIN_FOLDS,
    VAL_FOLDS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _make_combo_class(row: pd.Series) -> str:
    """Создает метку комбинированного класса из бинарных столбцов диагнозов.

    Parameters
    ----------
    row : pd.Series
        Строка датафрейма с полями MI, HYP, CD, STTC.

    Returns
    -------
    str
        Метка вида 'N', 'M', 'H', 'C', 'S' или их комбинации.
    """
    mapping = {"CD": "C", "HYP": "H", "MI": "M", "STTC": "S"}
    letters = "".join(mapping[col] for col in ["MI", "HYP", "CD", "STTC"] if row[col] == 1)
    return letters if letters else "N"


def _fill_rand_based_q(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    group: str = "sex",
) -> pd.DataFrame:
    """Заполняет пропуски случайными значениями в диапазоне [Q10, Q90] по группе.

    Parameters
    ----------
    df : pd.DataFrame
        Исходный датафрейм.
    cols : list[str] | None
        Столбцы для заполнения. По умолчанию ['height', 'weight'].
    group : str
        Столбец группировки.

    Returns
    -------
    pd.DataFrame
        Датафрейм с заполненными пропусками.
    """
    if cols is None:
        cols = ["height", "weight"]

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


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Предобработка данных PTB-XL для обучения модели.

    Выполняет следующие шаги:
    - Удаляет шумовые и служебные столбцы
    - Заполняет пропуски в height/weight случайными значениями
    - Фильтрует выбросы и некорректные записи
    - Удаляет строки, где NORM сочетается с другими диагнозами
    - Создаёт комбинированные классы (combo_class)
    - Кодирует combo_class в числовой индекс (combo_idx)
    - Label-кодирует heart_axis для использования в TabNet

    Parameters
    ----------
    df : pd.DataFrame
        Исходный датафрейм PTB-XL.

    Returns
    -------
    pd.DataFrame
        Предобработанный датафрейм, готовый к обучению.
    """
    logger.info("Начало предобработки данных, размер: %s", df.shape)

    df = df.drop(
        ["pacemaker", "extra_beats", "infarction_stadium1", "infarction_stadium2"],
        axis=1,
    )
    df = _fill_rand_based_q(df)
    df["heart_axis"] = df["heart_axis"].fillna("NO_DATA")
    df["age"] = df["age"].replace(300, 98)
    df = df[df["height"] >= 90]
    df = df[df["weight"] > 25]
    df = df[df["diagnostic_superclass"].apply(lambda x: len(x) > 0)]

    unique_classes = sorted({lbl for row in df["diagnostic_superclass"] for lbl in row})
    for lbl in unique_classes:
        df[lbl] = df["diagnostic_superclass"].apply(lambda x, _lbl=lbl: int(_lbl in x))

    final_columns = [
        "ecg_id",
        "patient_id",
        "recording_date",
        "age",
        "sex",
        "height",
        "weight",
        "heart_axis",
        "ecg_signals",
        "CD",
        "HYP",
        "MI",
        "NORM",
        "STTC",
        "diagnostic_superclass",
        "strat_fold",
    ]
    df = df[final_columns]

    mask_bad = (df["NORM"] == 1) & (
        (df["CD"] == 1) | (df["HYP"] == 1) | (df["MI"] == 1) | (df["STTC"] == 1)
    )
    n_bad = mask_bad.sum()
    df = df.loc[~mask_bad].reset_index(drop=True)
    logger.info("Удалено строк с некорректным сочетанием NORM: %d", n_bad)

    df["combo_class"] = df.apply(_make_combo_class, axis=1)
    df["combo_idx"], _ = pd.factorize(df["combo_class"])

    le = LabelEncoder()
    df["heart_axis_enc"] = le.fit_transform(df["heart_axis"])

    logger.info("Предобработка завершена, финальный размер: %s", df.shape)
    logger.info(
        "Распределение классов:\n%s",
        df["combo_class"].value_counts().to_string(),
    )
    return df


def get_cat_idxs_dims(features: list[str], df: pd.DataFrame) -> tuple[list[int], list[int]]:
    """Получает индексы категориальных признаков и их мощности для TabNet.

    Parameters
    ----------
    features : list[str]
        Список названий признаков в том порядке, в котором они подаются в модель.
    df : pd.DataFrame
        Предобработанный датафрейм.

    Returns
    -------
    tuple[list[int], list[int]]
        Кортеж (cat_idxs, cat_dims), где cat_idxs — индексы категориальных
        столбцов в списке features, cat_dims — количество уникальных значений.
    """
    cat_idxs: list[int] = []
    cat_dims: list[int] = []

    cat_cols = [CAT_COL]
    for i, col in enumerate(features):
        if col in cat_cols:
            cat_idxs.append(i)
            cat_dims.append(int(df[col].nunique()))

    return cat_idxs, cat_dims


def split_data(
    df: pd.DataFrame,
    train_folds: list[int] | None = None,
    val_folds: list[int] | None = None,
    test_folds: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame]:
    """Разделяет данные на обучающую, валидационную и тестовую выборки по strat_fold.

    Разбивка соответствует ml_experiments.ipynb:
    - фолды 1–7 → train
    - фолд 8   → val (опционально)
    - фолды 9–10 → test

    Parameters
    ----------
    df : pd.DataFrame
        Предобработанный датафрейм.
    train_folds : list[int] | None
        Фолды для обучающей выборки. По умолчанию TRAIN_FOLDS (1–7).
    val_folds : list[int] | None
        Фолды для валидационной выборки. По умолчанию VAL_FOLDS ([8]).
        Если передать пустой список — val вернётся как None.
    test_folds : list[int] | None
        Фолды для тестовой выборки. По умолчанию TEST_FOLDS ([9, 10]).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame]
        Кортеж (train_df, val_df, test_df).
        val_df равен None, если val_folds пустой.
    """
    if train_folds is None:
        train_folds = TRAIN_FOLDS
    if val_folds is None:
        val_folds = VAL_FOLDS
    if test_folds is None:
        test_folds = TEST_FOLDS

    train_df = df[df[FOLD_COL].isin(train_folds)].reset_index(drop=True)
    test_df = df[df[FOLD_COL].isin(test_folds)].reset_index(drop=True)
    val_df: pd.DataFrame | None = None
    if val_folds:
        val_df = df[df[FOLD_COL].isin(val_folds)].reset_index(drop=True)
        logger.info(
            "Разбивка: train=%d, val=%d, test=%d строк",
            len(train_df),
            len(val_df),
            len(test_df),
        )
    else:
        logger.info("Разбивка: train=%d, test=%d строк", len(train_df), len(test_df))

    return train_df, val_df, test_df


__all__ = [
    "preprocess",
    "get_cat_idxs_dims",
    "split_data",
    "FEATURE_COLS",
    "TARGET_COL",
]
