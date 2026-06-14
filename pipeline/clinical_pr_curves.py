"""Plot PR curves for the saved clinical CatBoost experiment."""

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
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.preprocessing import label_binarize

from pipeline import constants as C
from pipeline.clinical_catboost_experiment import build_feature_dataset
from pipeline.preprocess import split_data

logger = logging.getLogger(__name__)


def _load_model(path: Path) -> CatBoostClassifier:
    model = CatBoostClassifier()
    model.load_model(path)
    return model


def _positive_class_index(model: CatBoostClassifier) -> int:
    classes = [int(label) for label in model.classes_]
    return classes.index(1) if 1 in classes else 1


def _predict_binary_proba(model_dir: Path, experiment_name: str, class_name: str, x: pd.DataFrame) -> np.ndarray:
    model = _load_model(model_dir / f"{experiment_name}_{class_name}.cbm")
    positive_idx = _positive_class_index(model)
    return model.predict_proba(x)[:, positive_idx]


def _predict_multiclass_scores(model_path: Path, x: pd.DataFrame, labels: list[str]) -> np.ndarray:
    model = _load_model(model_path)
    proba = model.predict_proba(x)
    scores = np.zeros((len(x), len(labels)))
    for model_idx, label in enumerate(str(label) for label in model.classes_):
        if label in labels:
            scores[:, labels.index(label)] = proba[:, model_idx]
    return scores


def plot_binary_pr_curves(
    test_df: pd.DataFrame,
    baseline_features: list[str],
    clinical_features: list[str],
    model_dir: Path,
    output_dir: Path,
) -> pd.DataFrame:
    rows = []
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
    axes_flat = axes.ravel()

    for idx, class_name in enumerate(C.SUPERCLASSES):
        ax = axes_flat[idx]
        y_true = test_df[class_name].astype(int).to_numpy()
        prevalence = float(y_true.mean())

        for experiment_name, features, color in [
            ("baseline_catboost", baseline_features, "#95a5a6"),
            ("clinical_catboost", clinical_features, "#2980b9"),
        ]:
            y_score = _predict_binary_proba(model_dir, experiment_name, class_name, test_df[features])
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            pr_auc = average_precision_score(y_true, y_score)
            label = experiment_name.replace("_catboost", "")
            ax.plot(recall, precision, color=color, linewidth=2, label=f"{label} AP={pr_auc:.3f}")
            rows.append(
                {
                    "task": "binary_superclass",
                    "class": class_name,
                    "experiment": label,
                    "pr_auc": float(pr_auc),
                    "prevalence": prevalence,
                }
            )

        ax.axhline(prevalence, color="#7f8c8d", linestyle="--", linewidth=1, label=f"prevalence={prevalence:.3f}")
        ax.set_title(class_name)
        ax.grid(alpha=0.2)
        ax.legend(loc="lower left", fontsize=9)

    axes_flat[-1].axis("off")
    fig.suptitle("Precision-Recall Curves: 5 Diagnostic Superclasses", fontsize=16)
    for ax in axes_flat:
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_dir / "binary_superclass_pr_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rows)


def plot_multiclass_pr_curves(
    test_df: pd.DataFrame,
    baseline_features: list[str],
    clinical_features: list[str],
    model_dir: Path,
    output_dir: Path,
) -> pd.DataFrame:
    y_true = test_df[C.CLASS_COL].astype(str)
    labels = sorted(set(y_true))
    y_binary = label_binarize(y_true, classes=labels)

    baseline_scores = _predict_multiclass_scores(
        model_dir / "baseline_catboost_multiclass.cbm",
        test_df[baseline_features],
        labels,
    )
    clinical_scores = _predict_multiclass_scores(
        model_dir / "clinical_catboost_multiclass.cbm",
        test_df[clinical_features],
        labels,
    )

    rows = []
    fig, axes = plt.subplots(4, 4, figsize=(20, 16), sharex=True, sharey=True)
    axes_flat = axes.ravel()

    for idx, label_name in enumerate(labels):
        ax = axes_flat[idx]
        y_class = y_binary[:, idx]
        prevalence = float(y_class.mean())

        for experiment_name, scores, color in [
            ("baseline", baseline_scores, "#95a5a6"),
            ("clinical", clinical_scores, "#8e44ad"),
        ]:
            precision, recall, _ = precision_recall_curve(y_class, scores[:, idx])
            pr_auc = average_precision_score(y_class, scores[:, idx])
            ax.plot(recall, precision, color=color, linewidth=1.8, label=f"{experiment_name} AP={pr_auc:.3f}")
            rows.append(
                {
                    "task": "multiclass_one_vs_rest",
                    "class": label_name,
                    "experiment": experiment_name,
                    "pr_auc": float(pr_auc),
                    "prevalence": prevalence,
                }
            )

        ax.axhline(prevalence, color="#7f8c8d", linestyle="--", linewidth=0.8)
        ax.set_title(label_name)
        ax.grid(alpha=0.2)
        ax.legend(loc="lower left", fontsize=8)

    fig.suptitle("Precision-Recall Curves: 16 Classes, One-vs-Rest", fontsize=16)
    for ax in axes_flat:
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_dir / "multiclass_16_pr_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    result = pd.DataFrame(rows)
    summary = (
        result
        .pivot(index="class", columns="experiment", values="pr_auc")
        .reset_index()
    )
    summary["delta_clinical_minus_baseline"] = summary["clinical"] - summary["baseline"]
    summary.sort_values("delta_clinical_minus_baseline", ascending=False).to_csv(
        output_dir / "multiclass_16_pr_auc_by_class.csv",
        index=False,
    )
    return result


def run(args: argparse.Namespace) -> dict[str, str]:
    output_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df, baseline_features, clinical_features = build_feature_dataset(
        data_path=args.data_path,
        baseline_cache=args.baseline_cache,
        clinical_cache=args.clinical_cache,
        sample_rows=None,
        sampling_rate=args.sampling_rate,
    )
    _, _, test_df = split_data(df)

    binary_metrics = plot_binary_pr_curves(
        test_df=test_df,
        baseline_features=baseline_features,
        clinical_features=clinical_features,
        model_dir=model_dir,
        output_dir=output_dir,
    )
    multiclass_metrics = plot_multiclass_pr_curves(
        test_df=test_df,
        baseline_features=baseline_features,
        clinical_features=clinical_features,
        model_dir=model_dir,
        output_dir=output_dir,
    )
    metrics = pd.concat([binary_metrics, multiclass_metrics], ignore_index=True)
    metrics_path = output_dir / "pr_curve_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    summary = {
        "binary_superclass_mean_pr_auc": (
            binary_metrics.groupby("experiment")["pr_auc"].mean().to_dict()
        ),
        "multiclass_one_vs_rest_macro_pr_auc": (
            multiclass_metrics.groupby("experiment")["pr_auc"].mean().to_dict()
        ),
        "outputs": {
            "binary_pr_curves": str(output_dir / "binary_superclass_pr_curves.png"),
            "multiclass_pr_curves": str(output_dir / "multiclass_16_pr_curves.png"),
            "metrics_csv": str(metrics_path),
            "multiclass_by_class_csv": str(output_dir / "multiclass_16_pr_auc_by_class.csv"),
        },
    }
    summary_path = output_dir / "pr_curve_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary["outputs"] | {"summary_json": str(summary_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default=C.RAW_DATA_FILE)
    parser.add_argument("--baseline-cache", default="datasets/features_cache.pkl")
    parser.add_argument("--clinical-cache", default="datasets/clinical_features_v2_cache.pkl")
    parser.add_argument("--model-dir", default="outputs/clinical_catboost_full_v2")
    parser.add_argument("--output-dir", default="outputs/clinical_catboost_full_v2")
    parser.add_argument("--sampling-rate", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
