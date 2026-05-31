"""Модуль обучения модели TabNet (одиночный сплит train/val/test)."""

import logging
import time
from typing import Any

import numpy as np
import pandas as pd
from pytorch_tabnet.tab_model import TabNetClassifier

from pipeline import constants as C  # noqa: N812
from pipeline.metrics import compute_binary_metrics, compute_metrics, format_metrics_table
from pipeline.utils import count_tabnet_params, get_tabnet_init_params

logger = logging.getLogger(__name__)


def train_tabnet(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    params: dict,
    features: list[str],
    target: str,
    experiment_id: str = "exp",
    verbose: int = 0,
    cat_idxs: list[int] | None = None,
    cat_dims: list[int] | None = None,
) -> dict[str, Any]:
    """Обучить TabNet на фиксированном сплите train/val/test.

    Обучение проводится на ``train_df``, ранняя остановка — по ``val_df``.
    Финальные метрики вычисляются на ``val_df`` и ``test_df``.

    Parameters
    ----------
    train_df : pd.DataFrame
        Обучающий датафрейм (фолды 1–7).
    val_df : pd.DataFrame
        Валидационный датафрейм (фолд 8).
    test_df : pd.DataFrame
        Тестовый датафрейм (фолды 9–10).
    params : dict
        Гиперпараметры эксперимента.
    features : list[str]
        Список признаков для обучения.
    target : str
        Название целевого столбца.
    experiment_id : str
        Идентификатор эксперимента для логирования.
    verbose : int
        Уровень вывода TabNet (0 — без вывода).
    cat_idxs : list[int] | None
        Индексы категориальных признаков.
    cat_dims : list[int] | None
        Мощности категориальных признаков.

    Returns
    -------
    dict[str, Any]
        Словарь с ключами:
        - ``val_metrics``: метрики на валидационной выборке
        - ``test_metrics``: метрики на тестовой выборке
        - ``history``: история обучения (loss и accuracy по эпохам)
        - ``train_time``: время обучения (сек)
        - ``val_time``: время инференса на val + test (сек)
        - ``n_params``: количество обучаемых параметров
        - ``best_epoch``: лучшая эпоха (по val accuracy)
        - ``test_preds``: предсказания на тестовой выборке
    """
    logger.info("[%s] Начало обучения", experiment_id)

    x_tr = train_df[features].values  # noqa: N806
    y_tr = train_df[target].values
    x_val = val_df[features].values  # noqa: N806
    y_val = val_df[target].values
    x_test = test_df[features].values  # noqa: N806
    y_test = test_df[target].values

    init_params = get_tabnet_init_params(
        params, verbose=verbose, cat_idxs=cat_idxs, cat_dims=cat_dims
    )
    clf = TabNetClassifier(**init_params)

    t0 = time.perf_counter()
    clf.fit(
        X_train=x_tr,
        y_train=y_tr,
        eval_set=[(x_val, y_val)],
        eval_name=["valid"],
        eval_metric=["accuracy"],
        max_epochs=params.get("max_epochs", C.MAX_EPOCHS),
        patience=params.get("patience", C.PATIENCE),
        batch_size=C.BATCH_SIZE,
        virtual_batch_size=C.VIRTUAL_BATCH_SIZE,
        num_workers=C.NUM_WORKERS,
        drop_last=False,
    )
    train_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    y_val_pred = clf.predict(x_val)
    y_test_pred = clf.predict(x_test)
    inference_time = time.perf_counter() - t1

    val_metrics = compute_metrics(y_val, y_val_pred)
    test_metrics = compute_metrics(y_test, y_test_pred)
    n_params = count_tabnet_params(clf)

    logger.info("[%s] Val метрики:\n%s", experiment_id, format_metrics_table(val_metrics))
    logger.info("[%s] Test метрики:\n%s", experiment_id, format_metrics_table(test_metrics))

    return {
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "history": clf.history.history,
        "train_time": train_time,
        "val_time": inference_time,
        "n_params": n_params,
        "best_epoch": clf.best_epoch,
        "test_preds": y_test_pred,
    }


def train_tabnet_per_class(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    params: dict,
    features: list[str],
    superclasses: list[str] | None = None,
    experiment_id: str = "exp",
    verbose: int = 0,
    cat_idxs: list[int] | None = None,
    cat_dims: list[int] | None = None,
) -> dict[str, Any]:
    """Обучить 5 бинарных TabNet-классификаторов (по одному на суперкласс).

    Соответствует подходу из ml_experiments.ipynb (CatBoost per-class).
    Каждый классификатор обучается как бинарная задача (класс есть / нет)
    с автоматической балансировкой весов классов (``weights=1``).

    Parameters
    ----------
    train_df : pd.DataFrame
        Обучающий датафрейм (фолды 1–7).
    val_df : pd.DataFrame
        Валидационный датафрейм (фолд 8).
    test_df : pd.DataFrame
        Тестовый датафрейм (фолды 9–10).
    params : dict
        Гиперпараметры для всех моделей.
    features : list[str]
        Список признаков.
    superclasses : list[str] | None
        Список суперклассов для бинарной классификации.
        По умолчанию ``C.SUPERCLASSES`` = ['CD', 'HYP', 'MI', 'NORM', 'STTC'].
    experiment_id : str
        Идентификатор эксперимента для логирования.
    verbose : int
        Уровень вывода TabNet (0 — без вывода).
    cat_idxs : list[int] | None
        Индексы категориальных признаков.
    cat_dims : list[int] | None
        Мощности категориальных признаков.

    Returns
    -------
    dict[str, Any]
        Словарь с ключами:
        - ``per_class``: dict {class_name: {val_metrics, test_metrics, history,
          best_epoch, n_params, train_time}}
        - ``mean_val_f1_macro``: среднее F1-macro по классам на валидации
        - ``mean_test_f1_macro``: среднее F1-macro по классам на тесте
        - ``mean_val_pr_auc``: среднее PR-AUC по классам на валидации
        - ``mean_test_pr_auc``: среднее PR-AUC по классам на тесте
        - ``mean_val_roc_auc``: среднее ROC-AUC по классам на валидации
        - ``mean_test_roc_auc``: среднее ROC-AUC по классам на тесте
        - ``train_time``: суммарное время обучения (сек)
        - ``val_time``: суммарное время инференса (сек)
        - ``n_params``: число параметров одной модели
    """
    if superclasses is None:
        superclasses = C.SUPERCLASSES

    logger.info("[%s] Обучение %d бинарных TabNet-моделей", experiment_id, len(superclasses))

    x_tr = train_df[features].values  # noqa: N806
    x_val = val_df[features].values  # noqa: N806
    x_test = test_df[features].values  # noqa: N806

    per_class: dict[str, dict] = {}
    total_train_time = 0.0
    total_inference_time = 0.0
    n_params = 0

    for class_name in superclasses:
        y_tr = train_df[class_name].values
        y_val_cls = val_df[class_name].values
        y_test_cls = test_df[class_name].values

        init_params = get_tabnet_init_params(
            params, verbose=verbose, cat_idxs=cat_idxs, cat_dims=cat_dims
        )
        clf = TabNetClassifier(**init_params)

        t0 = time.perf_counter()
        clf.fit(
            X_train=x_tr,
            y_train=y_tr,
            eval_set=[(x_val, y_val_cls)],
            eval_name=["valid"],
            eval_metric=["accuracy"],
            max_epochs=params.get("max_epochs", C.MAX_EPOCHS),
            patience=params.get("patience", C.PATIENCE),
            batch_size=C.BATCH_SIZE,
            virtual_batch_size=C.VIRTUAL_BATCH_SIZE,
            num_workers=C.NUM_WORKERS,
            drop_last=False,
            weights=1,  # автоматическая балансировка классов
        )
        total_train_time += time.perf_counter() - t0

        t1 = time.perf_counter()
        y_val_pred = clf.predict(x_val)
        y_val_proba = clf.predict_proba(x_val)[:, 1]
        y_test_pred = clf.predict(x_test)
        y_test_proba = clf.predict_proba(x_test)[:, 1]
        total_inference_time += time.perf_counter() - t1

        val_metrics = compute_binary_metrics(y_val_cls, y_val_pred, y_val_proba)
        test_metrics = compute_binary_metrics(y_test_cls, y_test_pred, y_test_proba)
        n_params = count_tabnet_params(clf)

        logger.info(
            "[%s/%s] Val F1-macro=%.4f  PR-AUC=%.4f  best_epoch=%d",
            experiment_id,
            class_name,
            val_metrics["f1_macro"],
            val_metrics["pr_auc"],
            clf.best_epoch,
        )

        per_class[class_name] = {
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "history": clf.history.history,
            "best_epoch": clf.best_epoch,
            "n_params": n_params,
            "train_time": time.perf_counter() - t0,
        }

    def _mean(key: str, split: str) -> float:
        return float(np.mean([per_class[c][f"{split}_metrics"][key] for c in superclasses]))

    result = {
        "per_class": per_class,
        "mean_val_f1_macro": _mean("f1_macro", "val"),
        "mean_test_f1_macro": _mean("f1_macro", "test"),
        "mean_val_pr_auc": _mean("pr_auc", "val"),
        "mean_test_pr_auc": _mean("pr_auc", "test"),
        "mean_val_roc_auc": _mean("roc_auc", "val"),
        "mean_test_roc_auc": _mean("roc_auc", "test"),
        "train_time": total_train_time,
        "val_time": total_inference_time,
        "n_params": n_params,
    }

    logger.info(
        "[%s] Mean Val F1-macro=%.4f  Mean Test F1-macro=%.4f",
        experiment_id,
        result["mean_val_f1_macro"],
        result["mean_test_f1_macro"],
    )
    return result
