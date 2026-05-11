"""Модуль вычисления метрик качества классификации."""

import logging

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Вычисляет набор метрик классификации.

    Вычисляются следующие метрики:
    - accuracy
    - f1_micro, f1_macro, f1_weighted
    - prec_micro, prec_macro
    - rec_micro, rec_macro

    Parameters
    ----------
    y_true : np.ndarray
        Истинные метки классов.
    y_pred : np.ndarray
        Предсказанные метки классов.

    Returns
    -------
    dict[str, float]
        Словарь метрик вида {название_метрики: значение}.
    """
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "prec_micro": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "rec_micro": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "prec_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "rec_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    return metrics


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, float]:
    """Вычисляет метрики бинарной классификации.

    Вычисляются следующие метрики:
    - accuracy
    - f1_macro (основная метрика сравнения с CatBoost)
    - f1_micro
    - precision_macro, recall_macro
    - pr_auc (average precision score) — если переданы вероятности
    - roc_auc — если переданы вероятности

    Parameters
    ----------
    y_true : np.ndarray
        Истинные бинарные метки (0/1).
    y_pred : np.ndarray
        Предсказанные бинарные метки (0/1).
    y_proba : np.ndarray | None
        Вероятности класса 1 (для PR-AUC и ROC-AUC).

    Returns
    -------
    dict[str, float]
        Словарь метрик.
    """
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if y_proba is not None:
        try:
            metrics["pr_auc"] = float(average_precision_score(y_true, y_proba))
        except Exception:
            metrics["pr_auc"] = float("nan")
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except Exception:
            metrics["roc_auc"] = float("nan")
    else:
        metrics["pr_auc"] = float("nan")
        metrics["roc_auc"] = float("nan")
    return metrics


def aggregate_fold_metrics(fold_metrics: list[dict[str, float]]) -> dict[str, float]:
    """Усредняет метрики по фолдам кросс-валидации.

    Parameters
    ----------
    fold_metrics : list[dict[str, float]]
        Список словарей метрик по каждому фолду.

    Returns
    -------
    dict[str, float]
        Словарь усреднённых метрик.
    """
    if not fold_metrics:
        return {}
    keys = fold_metrics[0].keys()
    return {k: float(np.mean([m[k] for m in fold_metrics])) for k in keys}


def format_metrics_table(metrics: dict[str, float]) -> str:
    """Форматирует метрики в виде строки таблицы для вывода.

    Parameters
    ----------
    metrics : dict[str, float]
        Словарь метрик.

    Returns
    -------
    str
        Форматированная строка с метриками.
    """
    lines = [f"  {k:<15}: {v:.4f}" for k, v in metrics.items()]
    return "\n".join(lines)
