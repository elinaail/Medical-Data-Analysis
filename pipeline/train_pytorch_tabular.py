"""Обучение FT-Transformer и GANDALF через pytorch-tabular (per-class бинарная классификация)."""

import logging
import time
from typing import Any

import numpy as np
import pandas as pd

from pipeline import constants as C
from pipeline.metrics import compute_binary_metrics

logger = logging.getLogger(__name__)


try:
    from pytorch_lightning import Callback as _Callback
except ImportError:
    from lightning.pytorch import Callback as _Callback


class _EpochHistoryCallback(_Callback):
    """Собирает train_loss и valid_accuracy по эпохам."""

    def __init__(self) -> None:
        self.loss: list[float] = []
        self.valid_accuracy: list[float] = []

    def on_train_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        v = trainer.callback_metrics.get("train_loss")
        if v is not None:
            self.loss.append(float(v))

    def on_validation_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        v = trainer.callback_metrics.get("valid_accuracy")
        if v is not None:
            self.valid_accuracy.append(float(v))


_FT_TRANSFORMER_MODEL_KEYS = {
    "input_embed_dim",
    "num_heads",
    "num_attn_blocks",
    "attn_dropout",
    "add_norm_dropout",
    "ff_dropout",
    "ff_hidden_multiplier",
    "transformer_activation",
    "share_embedding",
    "attn_feature_importance",
    "embedding_initialization",
    "embedding_bias",
}

_GANDALF_MODEL_KEYS = {
    "gflu_stages",
    "gflu_dropout",
    "gflu_feature_init_sparsity",
    "learnable_sparsity",
}


def train_pytorch_tabular_per_class(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    params: dict,
    features: list[str],
    model_type: str = "ft_transformer",
    superclasses: list[str] | None = None,
    experiment_id: str = "exp",
) -> dict[str, Any]:
    """Обучить 5 бинарных классификаторов через pytorch-tabular (FT-Transformer или GANDALF).

    Интерфейс аналогичен ``train_tabnet_per_class`` для совместимости с ноутбуком.

    Parameters
    ----------
    train_df : pd.DataFrame
        Обучающий датафрейм (фолды 1–7).
    val_df : pd.DataFrame
        Валидационный датафрейм (фолд 8).
    test_df : pd.DataFrame
        Тестовый датафрейм (фолды 9–10).
    params : dict
        Гиперпараметры. Общие ключи: ``lr``, ``max_epochs``, ``patience``, ``batch_size``.
        Специфичные для модели ключи передаются в конфиг модели.
    features : list[str]
        Список непрерывных признаков.
    model_type : str
        ``"ft_transformer"`` или ``"gandalf"``.
    superclasses : list[str] | None
        Список суперклассов. По умолчанию ``C.SUPERCLASSES``.
    experiment_id : str
        Идентификатор эксперимента для логирования.

    Returns
    -------
    dict[str, Any]
        Словарь с теми же ключами, что и у ``train_tabnet_per_class``:
        ``per_class``, ``mean_val_f1_macro``, ``mean_test_f1_macro``,
        ``mean_val_pr_auc``, ``mean_test_pr_auc``, ``mean_val_roc_auc``,
        ``mean_test_roc_auc``, ``train_time``, ``val_time``, ``n_params``.
        История: ``{"loss": [...], "valid_accuracy": [...]}``.
    """
    from pytorch_tabular import TabularModel
    from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig

    if model_type == "ft_transformer":
        from pytorch_tabular.models import FTTransformerConfig as _ModelConfig
        _model_keys = _FT_TRANSFORMER_MODEL_KEYS
    elif model_type == "gandalf":
        from pytorch_tabular.models import GANDALFConfig as _ModelConfig
        _model_keys = _GANDALF_MODEL_KEYS
    else:
        raise ValueError(f"Неизвестный model_type={model_type!r}. Используйте 'ft_transformer' или 'gandalf'.")

    if superclasses is None:
        superclasses = C.SUPERCLASSES

    max_epochs = params.get("max_epochs", C.MAX_EPOCHS)
    patience = params.get("patience", C.PATIENCE)
    batch_size = params.get("batch_size", C.BATCH_SIZE)
    lr = params.get("lr", 1e-3)
    model_specific = {k: v for k, v in params.items() if k in _model_keys}

    per_class: dict[str, dict] = {}
    total_train_time = 0.0
    total_inference_time = 0.0
    n_params = 0

    for class_name in superclasses:
        logger.info("[%s/%s] Начало обучения (%s)", experiment_id, class_name, model_type)

        # Cast features to float32 — required for pytorch-tabular's in-place normalization
        tr = train_df[features].astype("float32").copy()
        tr[class_name] = train_df[class_name].astype(int).values
        vl = val_df[features].astype("float32").copy()
        vl[class_name] = val_df[class_name].astype(int).values
        ts = test_df[features].astype("float32").copy()
        ts[class_name] = test_df[class_name].astype(int).values

        data_config = DataConfig(
            target=[class_name],
            continuous_cols=list(features),
            categorical_cols=[],
        )

        trainer_config = TrainerConfig(
            max_epochs=max_epochs,
            batch_size=batch_size,
            early_stopping="valid_loss",
            early_stopping_patience=patience,
            early_stopping_min_delta=1e-5,
            checkpoints=None,
            load_best=False,
            progress_bar="none",
            seed=C.SEED,
            precision=32,
            accelerator="cpu",
            devices=1,
            trainer_kwargs={"enable_model_summary": False},
        )

        optimizer_config = OptimizerConfig(optimizer="Adam")

        model_config = _ModelConfig(
            task="classification",
            learning_rate=lr,
            **model_specific,
        )

        hist_cb = _EpochHistoryCallback()

        model = TabularModel(
            data_config=data_config,
            model_config=model_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config,
        )

        t0 = time.perf_counter()
        model.fit(train=tr, validation=vl, callbacks=[hist_cb])
        train_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        val_preds_df = model.predict(vl, progress_bar=None)
        test_preds_df = model.predict(ts, progress_bar=None)
        inference_time = time.perf_counter() - t1

        y_val_true = vl[class_name].values
        y_test_true = ts[class_name].values
        y_val_pred = val_preds_df["prediction"].values.astype(int)
        y_test_pred = test_preds_df["prediction"].values.astype(int)
        y_val_proba = val_preds_df["1_probability"].values
        y_test_proba = test_preds_df["1_probability"].values

        val_metrics = compute_binary_metrics(y_val_true, y_val_pred, y_val_proba)
        test_metrics = compute_binary_metrics(y_test_true, y_test_pred, y_test_proba)
        n_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)

        best_epoch = len(hist_cb.loss)
        history = {
            "loss": hist_cb.loss,
            "valid_accuracy": hist_cb.valid_accuracy,
        }

        logger.info(
            "[%s/%s] Val F1-macro=%.4f  PR-AUC=%.4f  epochs=%d",
            experiment_id,
            class_name,
            val_metrics["f1_macro"],
            val_metrics["pr_auc"],
            best_epoch,
        )

        per_class[class_name] = {
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "history": history,
            "best_epoch": best_epoch,
            "n_params": n_params,
            "train_time": train_time,
        }

        total_train_time += train_time
        total_inference_time += inference_time

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
