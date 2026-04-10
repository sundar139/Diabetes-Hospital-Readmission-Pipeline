from __future__ import annotations

import argparse
from pathlib import Path

from src.models.pipeline_factory import ModelFamily
from src.models.train import (
    SamplingStrategy,
    train_task,
)

ALLOWED_MODEL_FAMILIES: tuple[ModelFamily, ...] = (
    "logistic_regression",
    "random_forest",
    "xgboost",
)
ALLOWED_SAMPLING: tuple[SamplingStrategy, ...] = ("none", "over", "under")


def _parse_csv_values(raw: str) -> tuple[str, ...]:
    return tuple(value.strip() for value in raw.split(",") if value.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train binary readmission models using readmitted_30d target."
    )
    parser.add_argument(
        "--model-families",
        default="logistic_regression,random_forest,xgboost",
        help="Comma-separated model families.",
    )
    parser.add_argument(
        "--sampling-strategies",
        default="none,over,under",
        help="Comma-separated sampling strategies for binary task.",
    )
    parser.add_argument(
        "--feature-selection",
        choices=["none", "boruta"],
        default="none",
        help="Optional feature selection strategy.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help="Optional processed-data directory containing *_features.parquet files.",
    )
    parser.add_argument(
        "--feature-metadata-path",
        type=Path,
        default=None,
        help="Optional path to feature_metadata.json.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help="Optional artifact output directory.",
    )
    parser.add_argument(
        "--disable-mlflow",
        action="store_true",
        help="Disable MLflow tracking for this run.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    model_families = _parse_csv_values(args.model_families)
    bad_families = [value for value in model_families if value not in ALLOWED_MODEL_FAMILIES]
    if bad_families:
        raise ValueError("Unsupported model family values: " + ", ".join(bad_families))

    sampling_values = _parse_csv_values(args.sampling_strategies)
    bad_sampling = [value for value in sampling_values if value not in ALLOWED_SAMPLING]
    if bad_sampling:
        raise ValueError("Unsupported sampling strategy values: " + ", ".join(bad_sampling))

    result = train_task(
        task_type="binary",
        model_families=tuple(model_families),
        sampling_strategies=tuple(sampling_values),
        feature_selection_strategy=args.feature_selection,
        enable_mlflow=not args.disable_mlflow,
        processed_dir=args.processed_dir.resolve() if args.processed_dir else None,
        feature_metadata_path=(
            args.feature_metadata_path.resolve() if args.feature_metadata_path else None
        ),
        artifacts_dir=args.artifacts_dir.resolve() if args.artifacts_dir else None,
    )

    val_summary = {
        "roc_auc": result.best_val_metrics.get("roc_auc"),
        "precision": result.best_val_metrics.get("precision"),
        "recall": result.best_val_metrics.get("recall"),
        "f1": result.best_val_metrics.get("f1"),
    }
    test_summary = {
        "roc_auc": result.best_test_metrics.get("roc_auc"),
        "precision": result.best_test_metrics.get("precision"),
        "recall": result.best_test_metrics.get("recall"),
        "f1": result.best_test_metrics.get("f1"),
    }

    print("Binary training completed.")
    print(f"- best_model: {result.best_model_path}")
    print(f"- best_metadata: {result.best_metadata_path}")
    print(f"- training_results: {result.training_results_path}")
    print(f"- best_model_family: {result.model_family}")
    print(f"- best_sampling_strategy: {result.sampling_strategy}")
    print(f"- best_feature_selection: {result.feature_selection_strategy}")
    print(f"- best_val_metrics: {val_summary}")
    print(f"- best_test_metrics: {test_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
