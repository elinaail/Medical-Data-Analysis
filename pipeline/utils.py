"""Вспомогательные утилиты проекта."""

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def seed_everything(seed: int = 42) -> None:
    """Установить фиксированные значения всех генераторов случайных чисел.

    Parameters
    ----------
    seed : int
        Значение зерна случайности.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Seed установлен: %d", seed)


def get_tabnet_init_params(
    params: dict,
    verbose: int = 1,
    cat_idxs: list[int] | None = None,
    cat_dims: list[int] | None = None,
) -> dict:
    """Сформировать словарь параметров инициализации TabNetClassifier.

    Parameters
    ----------
    params : dict
        Словарь гиперпараметров эксперимента.
    verbose : int
        Уровень логирования TabNet (0 — без вывода, 1 — вывод).
    cat_idxs : list[int] | None
        Индексы категориальных признаков.
    cat_dims : list[int] | None
        Мощности категориальных признаков.

    Returns
    -------
    dict
        Словарь параметров, пригодный для передачи в TabNetClassifier(**...).
    """
    n_d = params.get("n_d", 8)
    return {
        "n_d": n_d,
        "n_a": params.get("n_a", n_d),
        "n_steps": params.get("n_steps", 3),
        "n_shared": params.get("n_shared", 2),
        "cat_emb_dim": params.get("cat_emb_dim", 1),
        "optimizer_params": {"lr": params.get("lr", 2e-2)},
        "mask_type": params.get("mask_type", "sparsemax"),
        "lambda_sparse": params.get("lambda_sparse", 1e-3),
        "optimizer_fn": torch.optim.Adam,
        "cat_idxs": cat_idxs or [],
        "cat_dims": cat_dims or [],
        "verbose": verbose,
        "seed": 42,
    }


def count_tabnet_params(model: object) -> int:
    """Подсчитать количество обучаемых параметров модели TabNet.

    Parameters
    ----------
    model : TabNetClassifier
        Обученная модель TabNet.

    Returns
    -------
    int
        Количество обучаемых параметров.
    """
    try:
        return sum(p.numel() for p in model.network.parameters() if p.requires_grad)
    except AttributeError:
        logger.warning("Не удалось подсчитать параметры модели")
        return 0
