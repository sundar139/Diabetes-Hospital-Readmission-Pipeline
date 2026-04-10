from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from src.config.settings import get_settings
from src.models.evaluate import EvaluationResult, evaluate_model
from src.models.predict import load_model, load_model_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate saved binary and multiclass models and generate comparison report."
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help="Optional processed data directory containing test_features.parquet.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help="Optional artifacts directory containing saved models and metadata.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=None,
        help="Optional reports output directory.",
    )
    return parser.parse_args()


def _load_training_results(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _best_f1_by_sampling(
    binary_results: dict[str, Any] | None,
    sampling_group: set[str],
) -> float | None:
    if not binary_results:
        return None

    f1_values: list[float] = []
    for run in binary_results.get("runs", []):
        if str(run.get("sampling_strategy")) not in sampling_group:
            continue
        test_metrics = run.get("test_metrics", {})
        metric_value = test_metrics.get("f1")
        if isinstance(metric_value, (int, float)):
            f1_values.append(float(metric_value))

    if not f1_values:
        return None
    return max(f1_values)


def _table(headers: list[str], rows: list[list[str]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    divider_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    row_lines = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, divider_line, *row_lines])


def _render_model_comparison_report(
    *,
    binary_metadata: dict[str, Any],
    multiclass_metadata: dict[str, Any],
    binary_eval: EvaluationResult,
    multiclass_eval: EvaluationResult,
    binary_training_results: dict[str, Any] | None,
    multiclass_training_results: dict[str, Any] | None,
) -> str:
    binary_models_trained = []
    if binary_training_results is not None:
        binary_models_trained = [
            f"{run['model_family']} (sampling={run['sampling_strategy']})"
            for run in binary_training_results.get("runs", [])
        ]

    multiclass_models_trained = []
    if multiclass_training_results is not None:
        multiclass_models_trained = [
            f"{run['model_family']} (sampling={run['sampling_strategy']})"
            for run in multiclass_training_results.get("runs", [])
        ]

    baseline_f1 = _best_f1_by_sampling(binary_training_results, {"none"})
    sampled_f1 = _best_f1_by_sampling(binary_training_results, {"over", "under"})

    if baseline_f1 is not None and sampled_f1 is not None:
        improved = sampled_f1 > baseline_f1
        imbalance_note = (
            "Sampling best test F1="
            f"{sampled_f1:.4f} vs no-sampling best test F1={baseline_f1:.4f}; "
            f"improved={improved}."
        )
    else:
        imbalance_note = "Insufficient binary run history to quantify sampling impact."

    feature_selection_used = (
        str(binary_metadata.get("feature_selection_strategy", "none")) != "none"
        or str(multiclass_metadata.get("feature_selection_strategy", "none")) != "none"
    )

    binary_shap_note = "not generated"
    if binary_metadata.get("shap_artifacts"):
        binary_shap_note = "generated"

    multiclass_shap_note = "not generated"
    if multiclass_metadata.get("shap_artifacts"):
        multiclass_shap_note = "generated"

    def runtime_note(payload: dict[str, Any], eval_metrics: dict[str, Any]) -> str:
        runtime_payload = eval_metrics.get("inference_runtime", {})
        if not isinstance(runtime_payload, dict):
            runtime_payload = {}

        requested = runtime_payload.get("xgboost_device_requested")
        if requested is None:
            requested = payload.get("xgboost_device_requested")

        used_training = payload.get("xgboost_device_used_for_training")
        if used_training is None:
            used_training = payload.get("xgboost_device_used")

        used_inference = runtime_payload.get("xgboost_device_used_for_inference")
        if used_inference is None:
            used_inference = payload.get("xgboost_device_used_for_inference")
        if used_inference is None:
            used_inference = used_training

        fallback_path = runtime_payload.get("inference_used_fallback_path")
        if fallback_path is None:
            fallback_path = payload.get("xgboost_inference_used_fallback_path")
        if fallback_path is None:
            fallback_path = bool(used_training == "cuda" and used_inference == "cpu")

        return (
            f"requested={requested}, training={used_training}, "
            f"inference={used_inference}, fallback_path={fallback_path}"
        )

    recommended_model = (
        "binary/"
        f"{binary_metadata.get('model_family', 'unknown')} "
        "for early-intervention workflows, with multiclass/"
        f"{multiclass_metadata.get('model_family', 'unknown')} "
        "as secondary triage support."
    )

    comparison_table = _table(
        ["Task", "Best Model", "Primary Metric", "Value"],
        [
            [
                "Binary (readmitted_30d)",
                str(binary_metadata.get("model_family", "unknown")),
                "F1",
                f"{float(binary_eval.metrics.get('f1', 0.0)):.4f}",
            ],
            [
                "Multiclass (readmitted)",
                str(multiclass_metadata.get("model_family", "unknown")),
                "Macro F1",
                f"{float(multiclass_eval.metrics.get('macro_f1', 0.0)):.4f}",
            ],
        ],
    )

    return "\n".join(
        [
            "# Model Comparison Report",
            "",
            "## Task Framing",
            "",
            "- Binary objective predicts early readmission risk (`readmitted_30d`).",
            (
                "- Multiclass objective predicts full horizon readmission category "
                "(`readmitted`: NO, >30, <30)."
            ),
            "",
            "## Models Trained",
            "",
            "- Binary candidates: "
            + (", ".join(binary_models_trained) if binary_models_trained else "N/A"),
            "- Multiclass candidates: "
            + (", ".join(multiclass_models_trained) if multiclass_models_trained else "N/A"),
            "",
            "## Best Model Per Task",
            "",
            (
                f"- Binary best: {binary_metadata.get('model_family', 'unknown')} "
                f"(sampling={binary_metadata.get('sampling_strategy', 'unknown')})."
            ),
            f"- Multiclass best: {multiclass_metadata.get('model_family', 'unknown')}.",
            "",
            "## Key Metrics Comparison",
            "",
            comparison_table,
            "",
            "## Binary Imbalance Handling Impact",
            "",
            f"- {imbalance_note}",
            "",
            "## Feature Selection Usage",
            "",
            f"- Optional feature selection used: {feature_selection_used}.",
            (
                "- Binary feature selection strategy: "
                f"{binary_metadata.get('feature_selection_strategy', 'none')}."
            ),
            (
                "- Multiclass feature selection strategy: "
                f"{multiclass_metadata.get('feature_selection_strategy', 'none')}."
            ),
            "",
            "## Interpretability Artifacts",
            "",
            f"- Binary SHAP artifacts: {binary_shap_note}.",
            f"- Multiclass SHAP artifacts: {multiclass_shap_note}.",
            "",
            "## XGBoost Runtime Notes",
            "",
            f"- Binary runtime: {runtime_note(binary_metadata, binary_eval.metrics)}.",
            f"- Multiclass runtime: {runtime_note(multiclass_metadata, multiclass_eval.metrics)}.",
            "",
            "## Recommended Production Candidate",
            "",
            f"- {recommended_model}",
            "",
        ]
    )


def main() -> int:
    args = parse_args()
    settings = get_settings()

    processed_dir = (
        args.processed_dir.resolve() if args.processed_dir else settings.processed_data_dir_path
    )
    artifacts_dir = (
        args.artifacts_dir.resolve() if args.artifacts_dir else settings.artifacts_dir_path
    )
    reports_dir = args.reports_dir.resolve() if args.reports_dir else settings.reports_dir_path

    test_features_path = processed_dir / "test_features.parquet"
    if not test_features_path.exists():
        raise FileNotFoundError(
            "Feature test split not found at "
            f"{test_features_path}. Run feature build and training first."
        )

    frame = pd.read_parquet(test_features_path)

    binary_model_path = artifacts_dir / "binary_model.joblib"
    multiclass_model_path = artifacts_dir / "multiclass_model.joblib"
    binary_metadata_path = artifacts_dir / "binary_model_metadata.json"
    multiclass_metadata_path = artifacts_dir / "multiclass_model_metadata.json"

    binary_model = load_model(binary_model_path)
    multiclass_model = load_model(multiclass_model_path)
    binary_metadata = load_model_metadata(binary_metadata_path)
    multiclass_metadata = load_model_metadata(multiclass_metadata_path)

    binary_features = [str(column) for column in binary_metadata.get("feature_columns", [])]
    multiclass_features = [
        str(column) for column in multiclass_metadata.get("feature_columns", [])
    ]

    binary_eval = evaluate_model(
        model=binary_model,
        x=frame[binary_features],
        y=frame[str(binary_metadata.get("target_column", "readmitted_30d"))],
        task_type="binary",
        output_dir=artifacts_dir / "evaluations" / "binary" / "final",
        run_name="test_final",
        class_labels=None,
        label_decoder=None,
    )

    mapping_raw = multiclass_metadata.get("label_mapping", {})
    label_decoder = (
        {int(k): str(v) for k, v in mapping_raw.items()}
        if isinstance(mapping_raw, dict)
        else None
    )
    class_labels = tuple(str(value) for value in multiclass_metadata.get("class_labels", []))

    multiclass_eval = evaluate_model(
        model=multiclass_model,
        x=frame[multiclass_features],
        y=frame[str(multiclass_metadata.get("target_column", "readmitted"))],
        task_type="multiclass",
        output_dir=artifacts_dir / "evaluations" / "multiclass" / "final",
        run_name="test_final",
        class_labels=class_labels,
        label_decoder=label_decoder,
    )

    binary_training_results = _load_training_results(
        artifacts_dir / "binary_training_results.json"
    )
    multiclass_training_results = _load_training_results(
        artifacts_dir / "multiclass_training_results.json"
    )

    report_text = _render_model_comparison_report(
        binary_metadata=binary_metadata,
        multiclass_metadata=multiclass_metadata,
        binary_eval=binary_eval,
        multiclass_eval=multiclass_eval,
        binary_training_results=binary_training_results,
        multiclass_training_results=multiclass_training_results,
    )

    report_path = reports_dir / "model_comparison_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")

    print("Final evaluation completed.")
    print(f"- binary_metrics: {binary_eval.metrics_path}")
    print(f"- multiclass_metrics: {multiclass_eval.metrics_path}")
    print(f"- comparison_report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
