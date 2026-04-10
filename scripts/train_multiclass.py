from __future__ import annotations

import argparse
from pathlib import Path

from src.models.pipeline_factory import ModelFamily
from src.models.train import train_task

ALLOWED_MODEL_FAMILIES: tuple[ModelFamily, ...] = (
    "logistic_regression",
    "random_forest",
    "xgboost",
)


def _parse_csv_values(raw: str) -> tuple[str, ...]:
    return tuple(value.strip() for value in raw.split(",") if value.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train multiclass readmission models using readmitted target."
    )
    parser.add_argument(
        "--model-families",
        default="logistic_regression,random_forest,xgboost",
        help="Comma-separated model families.",
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

    result = train_task(
        task_type="multiclass",
        model_families=tuple(model_families),
        sampling_strategies=("none",),
        feature_selection_strategy=args.feature_selection,
        enable_mlflow=not args.disable_mlflow,
        processed_dir=args.processed_dir.resolve() if args.processed_dir else None,
        feature_metadata_path=(
            args.feature_metadata_path.resolve() if args.feature_metadata_path else None
        ),
        artifacts_dir=args.artifacts_dir.resolve() if args.artifacts_dir else None,
    )

    val_summary = {
        "accuracy": result.best_val_metrics.get("accuracy"),
        "macro_f1": result.best_val_metrics.get("macro_f1"),
    }
    test_summary = {
        "accuracy": result.best_test_metrics.get("accuracy"),
        "macro_f1": result.best_test_metrics.get("macro_f1"),
    }

    print("Multiclass training completed.")
    print(f"- best_model: {result.best_model_path}")
    print(f"- best_metadata: {result.best_metadata_path}")
    print(f"- training_results: {result.training_results_path}")
    print(f"- best_model_family: {result.model_family}")
    print(f"- best_feature_selection: {result.feature_selection_strategy}")
    print(f"- best_xgboost_device: {result.xgboost_device_used or 'n/a'}")
    print(
        "- best_xgboost_device_used_for_training: "
        f"{result.xgboost_device_used_for_training or 'n/a'}"
    )
    print(
        "- best_xgboost_device_used_for_inference: "
        f"{result.xgboost_device_used_for_inference or 'n/a'}"
    )
    print(
        "- best_xgboost_inference_fallback_path: "
        f"{result.xgboost_inference_used_fallback_path}"
    )
    print(f"- best_val_metrics: {val_summary}")
    print(f"- best_test_metrics: {test_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
