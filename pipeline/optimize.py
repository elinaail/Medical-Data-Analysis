"""Модуль гиперпараметрической оптимизации с использованием Optuna."""

import functools
import logging

import optuna
import pandas as pd

from pipeline import constants as C  # noqa: N812
from pipeline.train import train_tabnet_per_class
from pipeline.train_pytorch_tabular import train_pytorch_tabular_per_class

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)


def _objective_tabnet(
    trial: optuna.Trial,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    cat_idxs: list[int] | None = None,
    cat_dims: list[int] | None = None,
) -> float:
    """Целевая функция Optuna для оптимизации гиперпараметров TabNet.

    Parameters
    ----------
    trial : optuna.Trial
        Объект пробы Optuna.
    train_df : pd.DataFrame
        Обучающий датафрейм.
    val_df : pd.DataFrame
        Валидационный датафрейм.
    test_df : pd.DataFrame
        Тестовый датафрейм.
    features : list[str]
        Список признаков.
    cat_idxs : list[int] | None
        Индексы категориальных признаков.
    cat_dims : list[int] | None
        Мощности категориальных признаков.

    Returns
    -------
    float
        Среднее F1-macro по суперклассам на валидации (максимизируется).
    """
    n_d = trial.suggest_int("n_d", 8, 32, step=4)
    params = {
        "n_d": n_d,
        "n_a": n_d,
        "n_steps": trial.suggest_int("n_steps", 3, 6),
        "n_shared": trial.suggest_int("n_shared", 1, 4),
        "cat_emb_dim": trial.suggest_int("cat_emb_dim", 1, 4),
        "lr": trial.suggest_float("lr", 1e-4, 5e-2, log=True),
        "mask_type": trial.suggest_categorical("mask_type", ["entmax", "sparsemax"]),
        "lambda_sparse": trial.suggest_float("lambda_sparse", 1e-4, 1e-2, log=True),
        "patience": trial.suggest_int("patience", 5, 20, step=5),
        "max_epochs": trial.suggest_int("max_epochs", 20, C.MAX_EPOCHS, step=10),
    }
    result = train_tabnet_per_class(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        params=params,
        features=features,
        experiment_id=f"optuna_trial_{trial.number}",
        verbose=0,
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
    )
    return result["mean_val_f1_macro"]


def hyperparameter_search(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    cat_idxs: list[int] | None = None,
    cat_dims: list[int] | None = None,
    n_trials: int = C.N_TRIALS,
    default_params: dict | None = None,
) -> tuple[dict, optuna.Study]:
    """Проводит поиск гиперпараметров TabNet с помощью Optuna (TPE).

    Оптимизируется среднее F1-macro по 5 суперклассам на валидационной выборке.
    В начало поиска добавляется проба с дефолтными параметрами.

    Parameters
    ----------
    train_df : pd.DataFrame
        Обучающий датафрейм.
    val_df : pd.DataFrame
        Валидационный датафрейм.
    test_df : pd.DataFrame
        Тестовый датафрейм.
    features : list[str]
        Список признаков.
    cat_idxs : list[int] | None
        Индексы категориальных признаков.
    cat_dims : list[int] | None
        Мощности категориальных признаков.
    n_trials : int
        Количество проб оптимизации.
    default_params : dict | None
        Дефолтные параметры для первой пробы. По умолчанию TABNET_PARAMS_DEFAULT.

    Returns
    -------
    tuple[dict, optuna.Study]
        Кортеж (best_params, study).
    """
    if default_params is None:
        default_params = C.TABNET_PARAMS_DEFAULT.copy()

    objective = functools.partial(
        _objective_tabnet,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        features=features,
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
    )

    sampler = optuna.samplers.TPESampler(seed=C.SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    # добавляем стартовую точку с дефолтными параметрами
    enqueue_params = {
        k: default_params[k]
        for k in [
            "n_d",
            "n_steps",
            "n_shared",
            "cat_emb_dim",
            "lr",
            "mask_type",
            "lambda_sparse",
            "patience",
            "max_epochs",
        ]
        if k in default_params
    }
    study.enqueue_trial(enqueue_params)

    logger.info("Запуск Optuna поиска, n_trials=%d", n_trials)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info("Лучшие параметры: %s", study.best_params)
    logger.info("Лучшее Mean Val F1-macro: %.4f", study.best_value)

    best_params = study.best_params.copy()
    best_params["n_a"] = best_params["n_d"]
    return best_params, study


def _objective_gandalf(
    trial: optuna.Trial,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
) -> float:
    """Целевая функция Optuna для оптимизации гиперпараметров GANDALF."""
    params = {
        "gflu_stages": trial.suggest_int("gflu_stages", 3, 12),
        "gflu_dropout": trial.suggest_float("gflu_dropout", 0.0, 0.3),
        "gflu_feature_init_sparsity": trial.suggest_float("gflu_feature_init_sparsity", 0.1, 0.7),
        "learnable_sparsity": True,
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "patience": trial.suggest_int("patience", 5, 20, step=5),
        "max_epochs": trial.suggest_int("max_epochs", 20, C.MAX_EPOCHS, step=10),
        "batch_size": C.BATCH_SIZE,
    }
    result = train_pytorch_tabular_per_class(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        params=params,
        features=features,
        model_type="gandalf",
        experiment_id=f"optuna_gandalf_trial_{trial.number}",
    )
    return result["mean_val_f1_macro"]


def hyperparameter_search_gandalf(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    n_trials: int = C.N_TRIALS,
    default_params: dict | None = None,
) -> tuple[dict, optuna.Study]:
    """Проводит поиск гиперпараметров GANDALF с помощью Optuna (TPE).

    Parameters
    ----------
    train_df : pd.DataFrame
        Обучающий датафрейм.
    val_df : pd.DataFrame
        Валидационный датафрейм.
    test_df : pd.DataFrame
        Тестовый датафрейм.
    features : list[str]
        Список признаков.
    n_trials : int
        Количество проб оптимизации.
    default_params : dict | None
        Дефолтные параметры для первой пробы. По умолчанию GANDALF_PARAMS_DEFAULT.

    Returns
    -------
    tuple[dict, optuna.Study]
        Кортеж (best_params, study).
    """
    if default_params is None:
        default_params = C.GANDALF_PARAMS_DEFAULT.copy()

    objective = functools.partial(
        _objective_gandalf,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        features=features,
    )

    sampler = optuna.samplers.TPESampler(seed=C.SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    # стартовая точка с дефолтными параметрами
    enqueue_keys = [
        "gflu_stages", "gflu_dropout", "gflu_feature_init_sparsity",
        "lr", "patience", "max_epochs",
    ]
    enqueue_params = {k: default_params[k] for k in enqueue_keys if k in default_params}
    study.enqueue_trial(enqueue_params)

    logger.info("Запуск Optuna GANDALF поиска, n_trials=%d", n_trials)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info("Лучшие параметры GANDALF: %s", study.best_params)
    logger.info("Лучшее Mean Val F1-macro: %.4f", study.best_value)

    best_params = study.best_params.copy()
    best_params["learnable_sparsity"] = True
    best_params["batch_size"] = C.BATCH_SIZE
    return best_params, study
