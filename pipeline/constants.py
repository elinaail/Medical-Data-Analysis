"""Константы и настройки проекта."""

# путь к данным
RAW_DATA_FILE: str = "datasets/raw/clean_ptbxl_with_ecg_n_diagnostic_superclass.pkl"

# глобальные параметры моделей NN
SEED: int = 42
MAX_EPOCHS: int = 100
N_TRIALS: int = 5
PATIENCE: int = 15
BATCH_SIZE: int = 1024
VIRTUAL_BATCH_SIZE: int = 128
NUM_WORKERS: int = 0

# настройка датасета
TARGET_COL: str = "combo_idx"
CLASS_COL: str = "combo_class"
FOLD_COL: str = "strat_fold"
HEART_AXIS_COL: str = "heart_axis"
SUPERCLASSES: list[str] = ["CD", "HYP", "MI", "NORM", "STTC"]
FEATURE_COLS: list[str] = ["age", "sex", "height", "weight", "heart_axis_enc"]
CAT_COL: str = "heart_axis_enc"

# train/val/test split
TRAIN_FOLDS: list[int] = [1, 2, 3, 4, 5, 6, 7]
VAL_FOLDS: list[int] = [8]
TEST_FOLDS: list[int] = [9, 10]

# параметры TabNet — конфигурация 1 (базовая)
TABNET_PARAMS_DEFAULT: dict = {
    "n_d": 8,
    "n_a": 8,
    "n_steps": 3,
    "n_shared": 2,
    "cat_emb_dim": 1,
    "lr": 2e-2,
    "mask_type": "entmax",
    "lambda_sparse": 1e-3,
    "max_epochs": MAX_EPOCHS,
    "patience": PATIENCE,
}

# параметры TabNet — конфигурация 2 (более глубокая)
TABNET_PARAMS_LARGE: dict = {
    "n_d": 16,
    "n_a": 16,
    "n_steps": 5,
    "n_shared": 3,
    "cat_emb_dim": 2,
    "lr": 5e-3,
    "mask_type": "sparsemax",
    "lambda_sparse": 5e-4,
    "max_epochs": MAX_EPOCHS,
    "patience": PATIENCE,
}

# параметры GANDALF — конфигурация 1 (базовая)
GANDALF_PARAMS_DEFAULT: dict = {
    "gflu_stages": 6,
    "gflu_dropout": 0.0,
    "gflu_feature_init_sparsity": 0.3,
    "learnable_sparsity": True,
    "lr": 1e-3,
    "max_epochs": MAX_EPOCHS,
    "patience": PATIENCE,
    "batch_size": BATCH_SIZE,
}

# параметры GANDALF — конфигурация 2 (более глубокая)
GANDALF_PARAMS_LARGE: dict = {
    "gflu_stages": 8,
    "gflu_dropout": 0.1,
    "gflu_feature_init_sparsity": 0.5,
    "learnable_sparsity": True,
    "lr": 5e-4,
    "max_epochs": MAX_EPOCHS,
    "patience": PATIENCE,
    "batch_size": BATCH_SIZE,
}

# метрики валидации
METRICS: list[str] = [
    "accuracy",
    "f1_micro",
    "f1_macro",
    "f1_weighted",
    "prec_micro",
    "rec_micro",
    "prec_macro",
    "rec_macro",
]
