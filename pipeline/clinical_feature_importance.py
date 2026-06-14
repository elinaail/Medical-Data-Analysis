"""Compute feature importance for the clinical CatBoost experiment.

The script uses saved CatBoost binary models and the validation split.
It reports permutation importance with PR-AUC and CatBoost built-in importance,
averaged over the five PTB-XL diagnostic superclasses.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score

from pipeline import constants as C
from pipeline.clinical_catboost_experiment import build_feature_dataset
from pipeline.preprocess import split_data

logger = logging.getLogger(__name__)


def _load_model(model_dir: Path, experiment_name: str, class_name: str) -> CatBoostClassifier:
    model = CatBoostClassifier()
    model.load_model(model_dir / f"{experiment_name}_{class_name}.cbm")
    return model


def _positive_class_index(model: CatBoostClassifier) -> int:
    classes = [int(label) for label in model.classes_]
    return classes.index(1) if 1 in classes else 1


def _feature_type(feature_name: str, baseline_features: set[str]) -> str:
    return "baseline" if feature_name in baseline_features else "clinical"


def compute_importance(
    data_path: str | Path,
    baseline_cache: str | Path,
    clinical_cache: str | Path,
    model_dir: str | Path,
    experiment_name: str,
    output_dir: str | Path,
    n_repeats: int,
    top_k: int,
    sampling_rate: int,
) -> dict[str, str]:
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df, baseline_features, clinical_features = build_feature_dataset(
        data_path=data_path,
        baseline_cache=baseline_cache,
        clinical_cache=clinical_cache,
        sample_rows=None,
        sampling_rate=sampling_rate,
    )
    _, val_df, _ = split_data(df)
    if val_df is None:
        raise RuntimeError("Validation split is required for permutation importance")

    features = clinical_features
    baseline_feature_set = set(baseline_features)
    x_val = val_df[features].to_numpy()
    rng = np.random.default_rng(C.SEED)

    builtin_df = pd.DataFrame(index=features)
    perm_df = pd.DataFrame(index=features)

    for class_name in C.SUPERCLASSES:
        logger.info("[%s] Feature importance", class_name)
        model = _load_model(model_dir, experiment_name, class_name)
        positive_idx = _positive_class_index(model)
        y_val = val_df[class_name].astype(int).to_numpy()

        builtin_df[class_name] = model.get_feature_importance()

        base_score = average_precision_score(y_val, model.predict_proba(x_val)[:, positive_idx])
        class_importances = np.zeros((len(features), n_repeats))

        for feature_idx in range(len(features)):
            for repeat_idx in range(n_repeats):
                x_perm = x_val.copy()
                x_perm[:, feature_idx] = rng.permutation(x_perm[:, feature_idx])
                score = average_precision_score(y_val, model.predict_proba(x_perm)[:, positive_idx])
                class_importances[feature_idx, repeat_idx] = base_score - score

        perm_df[f"perm_mean_{class_name}"] = class_importances.mean(axis=1)
        perm_df[f"perm_std_{class_name}"] = class_importances.std(axis=1)

    builtin_df["mean_importance"] = builtin_df[C.SUPERCLASSES].mean(axis=1)
    builtin_df["feature_type"] = [
        _feature_type(feature_name, baseline_feature_set)
        for feature_name in builtin_df.index
    ]
    builtin_df = builtin_df.sort_values("mean_importance", ascending=False)

    mean_cols = [f"perm_mean_{class_name}" for class_name in C.SUPERCLASSES]
    std_cols = [f"perm_std_{class_name}" for class_name in C.SUPERCLASSES]
    perm_df["perm_mean_global"] = perm_df[mean_cols].mean(axis=1)
    perm_df["perm_std_global"] = perm_df[std_cols].mean(axis=1)
    perm_df["builtin_mean_importance"] = builtin_df["mean_importance"].reindex(perm_df.index)
    perm_df["feature_type"] = [
        _feature_type(feature_name, baseline_feature_set)
        for feature_name in perm_df.index
    ]
    perm_df = perm_df.sort_values("perm_mean_global", ascending=False)

    perm_path = output_dir / f"{experiment_name}_permutation_importance.csv"
    builtin_path = output_dir / f"{experiment_name}_builtin_importance.csv"
    top_path = output_dir / f"{experiment_name}_top_{top_k}_importance.csv"
    plot_path = output_dir / f"{experiment_name}_top_{top_k}_importance.png"
    summary_path = output_dir / f"{experiment_name}_importance_summary.json"

    perm_df.to_csv(perm_path, encoding="utf-8", index_label="feature")
    builtin_df.to_csv(builtin_path, encoding="utf-8", index_label="feature")
    perm_df.head(top_k).to_csv(top_path, encoding="utf-8", index_label="feature")

    top_perm = perm_df.head(top_k)
    fig, axes = plt.subplots(1, 2, figsize=(18, 10))
    feature_labels = top_perm.index[::-1]
    perm_means = top_perm["perm_mean_global"].to_numpy()[::-1]
    perm_stds = top_perm["perm_std_global"].to_numpy()[::-1]
    perm_colors = [
        "#3498db" if value >= 0 else "#e74c3c"
        for value in perm_means
    ]

    axes[0].barh(
        feature_labels,
        perm_means,
        xerr=perm_stds,
        color=perm_colors,
        alpha=0.85,
        capsize=3,
    )
    axes[0].axvline(0, color="black", linewidth=0.8, linestyle="--")
    axes[0].set_title(
        f"Top {top_k} features: Permutation Importance\n"
        "(mean over 5 classes, metric: PR-AUC)"
    )
    axes[0].set_xlabel("Mean PR-AUC decrease after permutation")

    builtin_values = (
        builtin_df["mean_importance"]
        .reindex(top_perm.index)
        .fillna(0.0)
        .to_numpy()[::-1]
    )
    builtin_colors = [
        "#2ecc71" if _feature_type(feature_name, baseline_feature_set) == "baseline" else "#9b59b6"
        for feature_name in feature_labels
    ]
    axes[1].barh(feature_labels, builtin_values, color=builtin_colors, alpha=0.85)
    axes[1].set_title("CatBoost Importance\n(built-in importance, mean over 5 classes)")
    axes[1].set_xlabel("Feature importance (CatBoost built-in)")

    plt.tight_layout()
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "experiment": experiment_name,
        "feature_count": len(features),
        "baseline_feature_count": len(baseline_features),
        "clinical_added_feature_count": len(features) - len(baseline_features),
        "n_repeats": n_repeats,
        "split": "validation",
        "metric": "PR-AUC",
        "top_k": top_k,
        "top_features": [
            {
                "feature": feature_name,
                "feature_type": str(row["feature_type"]),
                "permutation_pr_auc_decrease": float(row["perm_mean_global"]),
                "permutation_std": float(row["perm_std_global"]),
                "catboost_builtin_importance": float(row["builtin_mean_importance"]),
            }
            for feature_name, row in top_perm.iterrows()
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "permutation_csv": str(perm_path),
        "builtin_csv": str(builtin_path),
        "top_csv": str(top_path),
        "plot_png": str(plot_path),
        "summary_json": str(summary_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default=C.RAW_DATA_FILE)
    parser.add_argument("--baseline-cache", default="datasets/features_cache.pkl")
    parser.add_argument("--clinical-cache", default="datasets/clinical_features_v2_cache.pkl")
    parser.add_argument("--model-dir", default="outputs/clinical_catboost_full_v2")
    parser.add_argument("--experiment-name", default="clinical_catboost")
    parser.add_argument("--output-dir", default="outputs/clinical_catboost_full_v2")
    parser.add_argument("--n-repeats", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--sampling-rate", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    outputs = compute_importance(**vars(parse_args()))
    print(json.dumps(outputs, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
