from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from src.config.settings import get_settings
from src.models.predict import load_model, load_model_metadata, predict_from_frame
from src.monitoring.drift_monitor import (
    DEFAULT_NUMERIC_DRIFT_COLUMNS,
    DEFAULT_SELECTED_INPUT_COLUMNS,
    build_monitoring_summary,
    build_prediction_records,
    generate_monitoring_narrative,
    load_model_version_info,
    load_prediction_records_jsonl,
    records_selected_inputs_to_frame,
    render_monitoring_report,
    write_prediction_records_jsonl,
)


def _parse_csv(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def parse_args() -> argparse.Namespace:
    settings = get_settings()

    parser = argparse.ArgumentParser(
        description="Generate local drift and prediction monitoring reports."
    )
    parser.add_argument(
        "--reference-path",
        type=Path,
        default=settings.processed_data_dir_path / "val_features.parquet",
        help="Reference feature parquet path.",
    )
    parser.add_argument(
        "--current-path",
        type=Path,
        default=settings.processed_data_dir_path / "test_features.parquet",
        help="Current feature parquet path used for prediction logging.",
    )
    parser.add_argument(
        "--prediction-log-path",
        type=Path,
        default=settings.artifacts_dir_path / "monitoring" / "prediction_log.jsonl",
        help="Prediction log output path in JSONL format.",
    )
    parser.add_argument(
        "--append-log",
        action="store_true",
        help="Append to prediction log instead of overwriting.",
    )
    parser.add_argument(
        "--selected-input-columns",
        default=",".join(DEFAULT_SELECTED_INPUT_COLUMNS),
        help="Comma-separated feature columns to log in selected_inputs.",
    )
    parser.add_argument(
        "--numeric-drift-columns",
        default=",".join(DEFAULT_NUMERIC_DRIFT_COLUMNS),
        help="Comma-separated numeric feature columns for drift checks.",
    )
    parser.add_argument(
        "--true-label-column",
        default="readmitted",
        help="Optional true-label column name to include in logs when present.",
    )
    parser.add_argument(
        "--true-label-binary-column",
        default="readmitted_30d",
        help="Optional binary true-label column name to include in logs when present.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=5000,
        help="Maximum rows to read from each feature input.",
    )
    parser.add_argument(
        "--min-sample-size",
        type=int,
        default=50,
        help="Minimum sample size threshold for drift checks.",
    )
    parser.add_argument(
        "--psi-bins",
        type=int,
        default=10,
        help="Number of bins used for PSI drift calculations.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=settings.reports_dir_path / "monitoring_summary.json",
        help="Monitoring summary JSON output path.",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=settings.reports_dir_path / "monitoring_report.md",
        help="Monitoring markdown report output path.",
    )
    parser.add_argument(
        "--prefer-ollama",
        action="store_true",
        help="Attempt optional Ollama narrative summary before fallback.",
    )
    return parser.parse_args()


def _load_frame(path: Path, *, max_rows: int) -> pd.DataFrame | None:
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    if max_rows > 0:
        return frame.head(max_rows).copy()
    return frame.copy()


def _parse_label_decoder(metadata: dict[str, Any]) -> dict[int, str] | None:
    mapping_raw = metadata.get("label_mapping", {})
    if not isinstance(mapping_raw, dict) or not mapping_raw:
        return None
    return {int(key): str(value) for key, value in mapping_raw.items()}


def _reference_binary_probabilities(
    *,
    reference_frame: pd.DataFrame | None,
    binary_model: Any,
    binary_metadata: dict[str, Any],
) -> list[float] | None:
    if reference_frame is None or reference_frame.empty:
        return None

    feature_columns = [str(column) for column in binary_metadata.get("feature_columns", [])]
    if not feature_columns:
        return None

    prediction_result = predict_from_frame(
        model=binary_model,
        frame=reference_frame,
        feature_columns=feature_columns,
        task_type="binary",
        label_decoder=_parse_label_decoder(binary_metadata),
    )

    if prediction_result.positive_class_probability is not None:
        return [float(value) for value in prediction_result.positive_class_probability]

    probs = prediction_result.probabilities_by_class
    for label in ("1", "<30"):
        if label in probs:
            return [float(value) for value in probs[label]]

    return None


def main() -> int:
    args = parse_args()
    settings = get_settings()

    artifacts_dir = settings.artifacts_dir_path

    binary_model = load_model(artifacts_dir / "binary_model.joblib")
    multiclass_model = load_model(artifacts_dir / "multiclass_model.joblib")
    binary_metadata = load_model_metadata(artifacts_dir / "binary_model_metadata.json")
    multiclass_metadata = load_model_metadata(artifacts_dir / "multiclass_model_metadata.json")

    selected_input_columns = _parse_csv(args.selected_input_columns)
    numeric_drift_columns = _parse_csv(args.numeric_drift_columns)

    reference_frame = _load_frame(args.reference_path, max_rows=args.max_rows)
    current_frame = _load_frame(args.current_path, max_rows=args.max_rows)

    model_version_info = load_model_version_info(artifacts_dir)

    current_records: list[dict[str, Any]]
    if current_frame is not None and not current_frame.empty:
        current_records = build_prediction_records(
            frame=current_frame,
            binary_model=binary_model,
            multiclass_model=multiclass_model,
            binary_metadata=binary_metadata,
            multiclass_metadata=multiclass_metadata,
            model_version_info=model_version_info,
            selected_input_columns=selected_input_columns,
            prediction_source="monitoring_report_batch",
            true_label_column=args.true_label_column,
            true_label_binary_column=args.true_label_binary_column,
        )
        write_prediction_records_jsonl(
            records=current_records,
            output_path=args.prediction_log_path,
            append=args.append_log,
        )
    else:
        current_records = load_prediction_records_jsonl(args.prediction_log_path)

    current_feature_frame = current_frame
    if current_feature_frame is None or current_feature_frame.empty:
        current_feature_frame = records_selected_inputs_to_frame(current_records)

    summary = build_monitoring_summary(
        model_version_info=model_version_info,
        current_records=current_records,
        reference_frame=reference_frame,
        current_feature_frame=current_feature_frame,
        numeric_feature_columns=numeric_drift_columns,
        reference_binary_probabilities=_reference_binary_probabilities(
            reference_frame=reference_frame,
            binary_model=binary_model,
            binary_metadata=binary_metadata,
        ),
        min_sample_size=args.min_sample_size,
        psi_bins=args.psi_bins,
    )

    narrative_text, narrative_mode, narrative_warnings = generate_monitoring_narrative(
        summary=summary,
        settings=settings,
        prefer_ollama=args.prefer_ollama,
    )

    summary["monitoring_narrative"] = narrative_text
    summary["monitoring_narrative_mode"] = narrative_mode
    summary["warnings"].extend(narrative_warnings)
    summary["prediction_log_path"] = str(args.prediction_log_path)

    report_markdown = render_monitoring_report(summary)

    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)

    args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    args.report_output.write_text(report_markdown, encoding="utf-8")

    print("Monitoring report generated.")
    print(f"- summary: {args.summary_output}")
    print(f"- report: {args.report_output}")
    print(f"- prediction_log: {args.prediction_log_path}")
    print(f"- narrative_mode: {narrative_mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
