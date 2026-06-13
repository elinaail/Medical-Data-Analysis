"""Run a separate CatBoost experiment with clinical ECG features.

This script does not touch the FastAPI service or production model artifacts.
It trains 5 binary CatBoost models for PTB-XL diagnostic superclasses and
compares baseline ECG features with baseline + clinical features.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

from pipeline import constants as C
from pipeline.clinical_features import extract_clinical_ecg_features
from pipeline.features import generate_features
from pipeline.preprocess import preprocess, split_data

logger = logging.getLogger(__name__)


def _load_catboost():
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise RuntimeError(
            "CatBoost is not installed. Install experiment dependencies with: "
            "uv pip install -r requirements-clinical.txt"
        ) from exc
    return CatBoostClassifier


def _save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _finite_feature_frame(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    result = df.copy()
    result[features] = result[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return result


def generate_clinical_feature_frame(
    df: pd.DataFrame,
    sampling_rate: int = 500,
    cache_path: str | Path | None = "datasets/clinical_features_v2_cache.pkl",
) -> pd.DataFrame:
    """Generate clinical features for each row with optional caching."""

    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            logger.info("Loading clinical features from cache: %s", cache_path)
            with cache_path.open("rb") as fh:
                return pd.read_pickle(fh)

    records = []
    for ecg in tqdm(df["ecg_signals"], desc="Clinical ECG features"):
        records.append(extract_clinical_ecg_features(ecg, sampling_rate=sampling_rate))

    clinical_df = pd.DataFrame(records).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        clinical_df.to_pickle(cache_path)
        logger.info("Saved clinical features cache: %s", cache_path)

    return clinical_df


def build_feature_dataset(
    data_path: str | Path,
    baseline_cache: str | Path | None,
    clinical_cache: str | Path | None,
    sample_rows: int | None,
    sampling_rate: int,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Load raw PTB-XL data and build baseline + clinical feature matrix."""

    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}. Put the PTB-XL pickle there or pass --data-path."
        )

    logger.info("Loading data: %s", data_path)
    raw_df = pd.read_pickle(data_path)

    logger.info("Preprocessing raw data")
    base_df = preprocess(raw_df)
    if sample_rows is not None:
        base_df = base_df.head(sample_rows).copy()
        logger.info("Using preprocessed sample rows: %d", len(base_df))
        baseline_cache = None
        clinical_cache = None

    logger.info("Generating baseline statistical/frequency features")
    feature_df, feature_cols = generate_features(base_df, n_jobs=1, cache_path=baseline_cache)

    logger.info("Generating clinical ECG features")
    clinical_df = generate_clinical_feature_frame(
        base_df,
        sampling_rate=sampling_rate,
        cache_path=clinical_cache,
    )
    clinical_cols = list(clinical_df.columns)

    full_df = pd.concat(
        [feature_df.reset_index(drop=True), clinical_df.reset_index(drop=True)],
        axis=1,
    )
    baseline_features = feature_cols.all
    clinical_features = baseline_features + clinical_cols
    full_df = _finite_feature_frame(full_df, clinical_features)

    logger.info("Baseline features: %d", len(baseline_features))
    logger.info("Clinical additional features: %d", len(clinical_cols))
    logger.info("Total clinical experiment features: %d", len(clinical_features))
    return full_df, baseline_features, clinical_features


def _class_weight_scale(y: pd.Series) -> float:
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    if positives == 0:
        return 1.0
    return max(1.0, negatives / positives)


def train_catboost_per_class(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    output_dir: Path,
    experiment_name: str,
    iterations: int,
    learning_rate: float,
    depth: int,
    catboost_verbose: int,
) -> dict[str, Any]:
    """Train one binary CatBoost model per diagnostic superclass."""

    CatBoostClassifier = _load_catboost()
    output_dir.mkdir(parents=True, exist_ok=True)

    per_class: dict[str, dict[str, float]] = {}
    for class_name in C.SUPERCLASSES:
        logger.info("[%s/%s] Training CatBoost", experiment_name, class_name)
        y_train = train_df[class_name].astype(int)
        y_val = val_df[class_name].astype(int)
        y_test = test_df[class_name].astype(int)
        if y_train.nunique() < 2:
            logger.warning(
                "[%s/%s] Skipped: train target contains only one class",
                experiment_name,
                class_name,
            )
            per_class[class_name] = {
                "f1_macro": float("nan"),
                "pr_auc": float("nan"),
                "roc_auc": float("nan"),
                "skipped": True,
            }
            continue

        model = CatBoostClassifier(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            loss_function="Logloss",
            eval_metric="F1",
            random_seed=C.SEED,
            verbose=catboost_verbose if catboost_verbose else False,
            allow_writing_files=False,
            scale_pos_weight=_class_weight_scale(y_train),
        )
        model.fit(
            train_df[features],
            y_train,
            eval_set=(val_df[features], y_val),
            use_best_model=True,
        )

        y_pred = model.predict(test_df[features]).astype(int)
        y_proba = model.predict_proba(test_df[features])[:, 1]
        metrics = {
            "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
            "pr_auc": float(average_precision_score(y_test, y_proba)),
            "roc_auc": float(roc_auc_score(y_test, y_proba)) if len(np.unique(y_test)) > 1 else float("nan"),
            "skipped": False,
        }
        per_class[class_name] = metrics

        model_path = output_dir / f"{experiment_name}_{class_name}.cbm"
        model.save_model(model_path)
        logger.info(
            "[%s/%s] Test F1-macro=%.4f PR-AUC=%.4f",
            experiment_name,
            class_name,
            metrics["f1_macro"],
            metrics["pr_auc"],
        )

    summary = {
        "experiment": experiment_name,
        "feature_count": len(features),
        "per_class": per_class,
        "mean_f1_macro": float(np.nanmean([m["f1_macro"] for m in per_class.values()])),
        "mean_pr_auc": float(np.nanmean([m["pr_auc"] for m in per_class.values()])),
        "mean_roc_auc": float(np.nanmean([m["roc_auc"] for m in per_class.values()])),
    }
    _save_json(output_dir / f"{experiment_name}_metrics.json", summary)
    return summary


def train_catboost_multiclass(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    output_dir: Path,
    experiment_name: str,
    iterations: int,
    learning_rate: float,
    depth: int,
    catboost_verbose: int,
) -> dict[str, Any]:
    """Train one multiclass CatBoost model for 16 combo ECG classes."""

    CatBoostClassifier = _load_catboost()
    output_dir.mkdir(parents=True, exist_ok=True)

    y_train = train_df[C.CLASS_COL].astype(str)
    y_val = val_df[C.CLASS_COL].astype(str)
    y_test = test_df[C.CLASS_COL].astype(str)

    model = CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        loss_function="MultiClass",
        eval_metric="TotalF1",
        auto_class_weights="Balanced",
        random_seed=C.SEED,
        verbose=catboost_verbose if catboost_verbose else False,
        allow_writing_files=False,
    )
    logger.info("[%s] Training multiclass CatBoost on %d classes", experiment_name, y_train.nunique())
    fit_kwargs: dict[str, Any] = {}
    if set(y_val).issubset(set(y_train)):
        fit_kwargs["eval_set"] = (val_df[features], y_val)
        fit_kwargs["use_best_model"] = True
    else:
        logger.warning(
            "[%s] Validation contains classes absent from the sample train split; "
            "training without eval_set",
            experiment_name,
        )

    model.fit(train_df[features], y_train, **fit_kwargs)

    y_pred = np.asarray(model.predict(test_df[features])).reshape(-1).astype(str)
    y_proba = model.predict_proba(test_df[features])
    labels = sorted(set(y_train) | set(y_val) | set(y_test) | set(y_pred))
    model_classes = [str(label) for label in model.classes_]
    y_score = np.zeros((len(y_test), len(labels)))
    for model_idx, label in enumerate(model_classes):
        if label in labels:
            y_score[:, labels.index(label)] = y_proba[:, model_idx]
    y_test_binary = label_binarize(y_test, classes=labels)

    metrics = {
        "experiment": experiment_name,
        "feature_count": len(features),
        "class_count_train": int(y_train.nunique()),
        "class_count_test": int(y_test.nunique()),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "pr_auc_macro": float(average_precision_score(y_test_binary, y_score, average="macro")),
        "pr_auc_weighted": float(average_precision_score(y_test_binary, y_score, average="weighted")),
    }

    per_class_f1 = f1_score(y_test, y_pred, labels=labels, average=None, zero_division=0)
    per_class_pr_auc = average_precision_score(y_test_binary, y_score, average=None)
    metrics["per_class_f1"] = {
        label: float(score)
        for label, score in zip(labels, per_class_f1, strict=True)
    }
    metrics["per_class_pr_auc"] = {
        label: float(score)
        for label, score in zip(labels, per_class_pr_auc, strict=True)
    }

    model.save_model(output_dir / f"{experiment_name}.cbm")
    _save_json(output_dir / f"{experiment_name}_metrics.json", metrics)
    logger.info(
        "[%s] Test accuracy=%.4f F1-macro=%.4f F1-weighted=%.4f",
        experiment_name,
        metrics["accuracy"],
        metrics["f1_macro"],
        metrics["f1_weighted"],
    )
    return metrics


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    df, baseline_features, clinical_features = build_feature_dataset(
        data_path=args.data_path,
        baseline_cache=args.baseline_cache,
        clinical_cache=args.clinical_cache,
        sample_rows=args.sample_rows,
        sampling_rate=args.sampling_rate,
    )
    train_df, val_df, test_df = split_data(df)
    if val_df is None:
        raise RuntimeError("Validation split is required for this experiment")

    output_dir = Path(args.output_dir)
    results: dict[str, Any] = {}
    if args.task in {"binary", "both"} and not args.clinical_only:
        results["baseline"] = train_catboost_per_class(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            features=baseline_features,
            output_dir=output_dir,
            experiment_name="baseline_catboost",
            iterations=args.iterations,
            learning_rate=args.learning_rate,
            depth=args.depth,
            catboost_verbose=args.catboost_verbose,
        )

    if args.task in {"binary", "both"}:
        results["clinical"] = train_catboost_per_class(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            features=clinical_features,
            output_dir=output_dir,
            experiment_name="clinical_catboost",
            iterations=args.iterations,
            learning_rate=args.learning_rate,
            depth=args.depth,
            catboost_verbose=args.catboost_verbose,
        )

    if args.task in {"multiclass", "both"} and not args.clinical_only:
        results["baseline_multiclass"] = train_catboost_multiclass(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            features=baseline_features,
            output_dir=output_dir,
            experiment_name="baseline_catboost_multiclass",
            iterations=args.iterations,
            learning_rate=args.learning_rate,
            depth=args.depth,
            catboost_verbose=args.catboost_verbose,
        )

    if args.task in {"multiclass", "both"}:
        results["clinical_multiclass"] = train_catboost_multiclass(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            features=clinical_features,
            output_dir=output_dir,
            experiment_name="clinical_catboost_multiclass",
            iterations=args.iterations,
            learning_rate=args.learning_rate,
            depth=args.depth,
            catboost_verbose=args.catboost_verbose,
        )

    _save_json(output_dir / "comparison_metrics.json", results)
    rows = []
    for name, metrics in results.items():
        rows.append(
            {
                "experiment": name,
                "feature_count": metrics["feature_count"],
                "mean_f1_macro": metrics.get("mean_f1_macro"),
                "mean_pr_auc": metrics.get("mean_pr_auc"),
                "mean_roc_auc": metrics.get("mean_roc_auc"),
                "accuracy": metrics.get("accuracy"),
                "f1_macro": metrics.get("f1_macro"),
                "f1_weighted": metrics.get("f1_weighted"),
                "pr_auc_macro": metrics.get("pr_auc_macro"),
                "pr_auc_weighted": metrics.get("pr_auc_weighted"),
                "primary_pr_auc": metrics.get("mean_pr_auc", metrics.get("pr_auc_macro")),
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "comparison_metrics.csv", index=False)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default=C.RAW_DATA_FILE)
    parser.add_argument("--baseline-cache", default="datasets/features_cache.pkl")
    parser.add_argument("--clinical-cache", default="datasets/clinical_features_v2_cache.pkl")
    parser.add_argument("--output-dir", default="outputs/clinical_catboost")
    parser.add_argument("--sample-rows", type=int, default=None)
    parser.add_argument("--sampling-rate", type=int, default=500)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--catboost-verbose", type=int, default=0)
    parser.add_argument("--clinical-only", action="store_true")
    parser.add_argument("--task", choices=["binary", "multiclass", "both"], default="both")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    results = run_experiment(args)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
