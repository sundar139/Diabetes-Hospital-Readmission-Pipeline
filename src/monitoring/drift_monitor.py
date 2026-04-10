from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd

from src.config.settings import Settings
from src.models.predict import load_model_metadata, predict_from_frame

DEFAULT_SELECTED_INPUT_COLUMNS: tuple[str, ...] = (
    "time_in_hospital",
    "num_medications",
    "number_diagnoses",
    "utilization_intensity",
    "patient_severity",
    "medication_change_ratio",
    "recurrency",
    "complex_discharge_flag",
    "age_bucket_risk",
)

DEFAULT_NUMERIC_DRIFT_COLUMNS: tuple[str, ...] = (
    "time_in_hospital",
    "num_medications",
    "number_diagnoses",
    "utilization_intensity",
    "patient_severity",
    "medication_change_ratio",
    "recurrency",
    "complex_discharge_flag",
    "age_bucket_risk",
)


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    if isinstance(value, np.generic):
        return _json_safe(value.item())

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    return str(value)


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        parsed = float(value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(parsed):
        return None
    return parsed


def _parse_label_decoder(metadata: Mapping[str, Any]) -> dict[int, str] | None:
    mapping_raw = metadata.get("label_mapping", {})
    if not isinstance(mapping_raw, Mapping) or not mapping_raw:
        return None

    decoder: dict[int, str] = {}
    for key, value in mapping_raw.items():
        decoder[int(key)] = str(value)
    return decoder


def _binary_prediction_to_int(raw_prediction: str) -> int:
    normalized = str(raw_prediction).strip().lower()
    if normalized in {"1", "true", "yes", "<30"}:
        return 1
    return 0


def _binary_probability_at(
    probabilities_by_class: Mapping[str, list[float]],
    *,
    index: int,
    positive_labels: tuple[str, ...] = ("1", "<30"),
) -> float:
    for label in positive_labels:
        values = probabilities_by_class.get(label)
        if values is not None and index < len(values):
            return float(values[index])
    return 0.0


def load_model_version_info(artifacts_dir: Path) -> dict[str, Any]:
    binary_metadata_path = artifacts_dir / "binary_model_metadata.json"
    multiclass_metadata_path = artifacts_dir / "multiclass_model_metadata.json"

    binary_metadata: dict[str, Any] = {}
    multiclass_metadata: dict[str, Any] = {}

    if binary_metadata_path.exists():
        binary_metadata = load_model_metadata(binary_metadata_path)
    if multiclass_metadata_path.exists():
        multiclass_metadata = load_model_metadata(multiclass_metadata_path)

    return {
        "binary_model": {
            "model_family": binary_metadata.get("model_family"),
            "training_timestamp_utc": binary_metadata.get("training_timestamp_utc"),
            "feature_count": len(binary_metadata.get("feature_columns", [])),
            "metadata_path": str(binary_metadata_path),
        },
        "multiclass_model": {
            "model_family": multiclass_metadata.get("model_family"),
            "training_timestamp_utc": multiclass_metadata.get("training_timestamp_utc"),
            "feature_count": len(multiclass_metadata.get("feature_columns", [])),
            "metadata_path": str(multiclass_metadata_path),
        },
    }


def build_prediction_records(
    *,
    frame: pd.DataFrame,
    binary_model: Any,
    multiclass_model: Any,
    binary_metadata: Mapping[str, Any],
    multiclass_metadata: Mapping[str, Any],
    model_version_info: Mapping[str, Any],
    selected_input_columns: Sequence[str] = DEFAULT_SELECTED_INPUT_COLUMNS,
    prediction_source: str = "monitoring_batch",
    true_label_column: str | None = None,
    true_label_binary_column: str | None = None,
) -> list[dict[str, Any]]:
    binary_feature_columns = [str(column) for column in binary_metadata.get("feature_columns", [])]
    multiclass_feature_columns = [
        str(column) for column in multiclass_metadata.get("feature_columns", [])
    ]

    if not binary_feature_columns:
        raise ValueError("Binary metadata is missing feature_columns.")
    if not multiclass_feature_columns:
        raise ValueError("Multiclass metadata is missing feature_columns.")

    binary_decoder = _parse_label_decoder(binary_metadata)
    multiclass_decoder = _parse_label_decoder(multiclass_metadata)

    binary_output = predict_from_frame(
        model=binary_model,
        frame=frame,
        feature_columns=binary_feature_columns,
        task_type="binary",
        label_decoder=binary_decoder,
    )
    multiclass_output = predict_from_frame(
        model=multiclass_model,
        frame=frame,
        feature_columns=multiclass_feature_columns,
        task_type="multiclass",
        label_decoder=multiclass_decoder,
    )

    timestamp_utc = datetime.now(UTC).isoformat()
    records: list[dict[str, Any]] = []

    for idx in range(len(frame)):
        row = frame.iloc[idx]

        selected_inputs: dict[str, Any] = {}
        for column in selected_input_columns:
            if column in frame.columns:
                selected_inputs[column] = _json_safe(row[column])

        if binary_output.positive_class_probability is not None:
            binary_probability = float(binary_output.positive_class_probability[idx])
        else:
            binary_probability = _binary_probability_at(
                binary_output.probabilities_by_class,
                index=idx,
            )

        multiclass_probabilities = {
            label: float(values[idx])
            for label, values in multiclass_output.probabilities_by_class.items()
            if idx < len(values)
        }

        record: dict[str, Any] = {
            "timestamp_utc": timestamp_utc,
            "prediction_source": prediction_source,
            "selected_inputs": selected_inputs,
            "binary_prediction": _binary_prediction_to_int(binary_output.predictions[idx]),
            "binary_probability": binary_probability,
            "multiclass_prediction": str(multiclass_output.predictions[idx]),
            "multiclass_probabilities": multiclass_probabilities,
            "model_version": model_version_info,
        }

        if true_label_column and true_label_column in frame.columns:
            record["true_label"] = _json_safe(row[true_label_column])
        if true_label_binary_column and true_label_binary_column in frame.columns:
            record["true_label_30d"] = _json_safe(row[true_label_binary_column])

        records.append(record)

    return records


def write_prediction_records_jsonl(
    *,
    records: Sequence[Mapping[str, Any]],
    output_path: Path,
    append: bool = False,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with output_path.open(mode, encoding="utf-8", newline="\n") as file_handle:
        for record in records:
            file_handle.write(json.dumps(dict(record), sort_keys=True) + "\n")


def load_prediction_records_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def records_selected_inputs_to_frame(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        selected_inputs = record.get("selected_inputs", {})
        if isinstance(selected_inputs, Mapping):
            rows.append({str(key): value for key, value in selected_inputs.items()})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def compute_psi(
    reference_values: Sequence[float],
    current_values: Sequence[float],
    *,
    bins: int = 10,
) -> float | None:
    reference = np.asarray([value for value in reference_values if np.isfinite(value)], dtype=float)
    current = np.asarray([value for value in current_values if np.isfinite(value)], dtype=float)

    if reference.size == 0 or current.size == 0:
        return None

    quantiles = np.linspace(0.0, 1.0, bins + 1)
    bin_edges = np.unique(np.quantile(reference, quantiles))

    if bin_edges.size <= 1:
        return 0.0

    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    reference_hist, _ = np.histogram(reference, bins=bin_edges)
    current_hist, _ = np.histogram(current, bins=bin_edges)

    reference_ratio = reference_hist / max(reference_hist.sum(), 1)
    current_ratio = current_hist / max(current_hist.sum(), 1)

    eps = 1e-6
    reference_ratio = np.clip(reference_ratio, eps, None)
    current_ratio = np.clip(current_ratio, eps, None)

    psi_terms = (current_ratio - reference_ratio) * np.log(current_ratio / reference_ratio)
    return float(np.sum(psi_terms))


def _psi_band(psi_value: float | None) -> str:
    if psi_value is None:
        return "unavailable"
    if psi_value < 0.10:
        return "stable"
    if psi_value < 0.20:
        return "moderate"
    return "high"


def summarize_probability_distribution(
    probabilities: Sequence[float],
) -> dict[str, float | int | None]:
    values = np.asarray([value for value in probabilities if np.isfinite(value)], dtype=float)
    if values.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "p10": None,
            "p50": None,
            "p90": None,
        }

    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "min": float(values.min()),
        "max": float(values.max()),
        "p10": float(np.quantile(values, 0.10)),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
    }


def summarize_prediction_distribution(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    binary_counter = Counter()
    multiclass_counter = Counter()

    for record in records:
        binary_counter[str(record.get("binary_prediction", "unknown"))] += 1
        multiclass_counter[str(record.get("multiclass_prediction", "unknown"))] += 1

    total = max(len(records), 1)

    return {
        "n_records": len(records),
        "binary_prediction_counts": dict(sorted(binary_counter.items())),
        "binary_prediction_rates": {
            key: float(value / total) for key, value in sorted(binary_counter.items())
        },
        "multiclass_prediction_counts": dict(sorted(multiclass_counter.items())),
        "multiclass_prediction_rates": {
            key: float(value / total) for key, value in sorted(multiclass_counter.items())
        },
    }


def compute_numeric_feature_drift_summary(
    *,
    reference_frame: pd.DataFrame,
    current_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    min_sample_size: int = 50,
    psi_bins: int = 10,
) -> tuple[dict[str, Any], list[str]]:
    drift_by_feature: dict[str, Any] = {}
    warnings: list[str] = []

    for feature_name in feature_columns:
        if feature_name not in reference_frame.columns or feature_name not in current_frame.columns:
            warnings.append(
                f"Feature '{feature_name}' is missing from reference or current frame; "
                "skipped drift check."
            )
            continue

        ref_values = (
            pd.to_numeric(reference_frame[feature_name], errors="coerce").dropna().to_numpy()
        )
        cur_values = (
            pd.to_numeric(current_frame[feature_name], errors="coerce").dropna().to_numpy()
        )

        if len(ref_values) < min_sample_size or len(cur_values) < min_sample_size:
            warnings.append(
                f"Feature '{feature_name}' has insufficient sample size for drift check "
                f"(reference={len(ref_values)}, current={len(cur_values)})."
            )
            drift_by_feature[feature_name] = {
                "status": "insufficient_data",
                "reference_count": int(len(ref_values)),
                "current_count": int(len(cur_values)),
            }
            continue

        psi_value = compute_psi(ref_values, cur_values, bins=psi_bins)
        drift_by_feature[feature_name] = {
            "status": _psi_band(psi_value),
            "psi": psi_value,
            "reference_count": int(len(ref_values)),
            "current_count": int(len(cur_values)),
            "reference_mean": float(np.mean(ref_values)),
            "current_mean": float(np.mean(cur_values)),
            "absolute_mean_shift": float(abs(np.mean(cur_values) - np.mean(ref_values))),
        }

    return drift_by_feature, warnings


def summarize_label_availability(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    label_count = 0
    binary_label_count = 0

    for record in records:
        if record.get("true_label") is not None:
            label_count += 1
        if record.get("true_label_30d") is not None:
            binary_label_count += 1

    return {
        "labels_available": label_count > 0 or binary_label_count > 0,
        "true_label_count": label_count,
        "true_label_30d_count": binary_label_count,
        "performance_monitoring": (
            "unavailable"
            if label_count == 0 and binary_label_count == 0
            else "labels_present_but_not_scored_in_this_report"
        ),
    }


def build_monitoring_summary(
    *,
    model_version_info: Mapping[str, Any],
    current_records: Sequence[Mapping[str, Any]],
    reference_frame: pd.DataFrame | None,
    current_feature_frame: pd.DataFrame | None,
    numeric_feature_columns: Sequence[str] = DEFAULT_NUMERIC_DRIFT_COLUMNS,
    reference_binary_probabilities: Sequence[float] | None = None,
    min_sample_size: int = 50,
    psi_bins: int = 10,
) -> dict[str, Any]:
    warnings: list[str] = []

    binary_probabilities = [
        value
        for value in (_to_float(record.get("binary_probability")) for record in current_records)
        if value is not None
    ]

    probability_summary = summarize_probability_distribution(binary_probabilities)
    prediction_distribution = summarize_prediction_distribution(current_records)

    probability_drift: dict[str, Any]
    if reference_binary_probabilities is None:
        probability_drift = {"status": "unavailable", "psi": None}
        warnings.append(
            "Reference binary probabilities are unavailable; probability drift skipped."
        )
    else:
        reference_values = [value for value in reference_binary_probabilities if np.isfinite(value)]
        if len(reference_values) < min_sample_size or len(binary_probabilities) < min_sample_size:
            probability_drift = {
                "status": "insufficient_data",
                "psi": None,
                "reference_count": len(reference_values),
                "current_count": len(binary_probabilities),
            }
            warnings.append(
                "Insufficient sample size for probability drift check "
                f"(reference={len(reference_values)}, current={len(binary_probabilities)})."
            )
        else:
            psi_value = compute_psi(reference_values, binary_probabilities, bins=psi_bins)
            probability_drift = {
                "status": _psi_band(psi_value),
                "psi": psi_value,
                "reference_count": len(reference_values),
                "current_count": len(binary_probabilities),
            }

    feature_drift_summary: dict[str, Any] = {}
    if reference_frame is None or current_feature_frame is None:
        warnings.append(
            "Reference or current feature frame is unavailable; "
            "feature drift checks skipped."
        )
    elif reference_frame.empty or current_feature_frame.empty:
        warnings.append(
            "Reference or current feature frame is empty; feature drift checks skipped."
        )
    else:
        feature_drift_summary, feature_warnings = compute_numeric_feature_drift_summary(
            reference_frame=reference_frame,
            current_frame=current_feature_frame,
            feature_columns=numeric_feature_columns,
            min_sample_size=min_sample_size,
            psi_bins=psi_bins,
        )
        warnings.extend(feature_warnings)

    label_monitoring = summarize_label_availability(current_records)
    if not label_monitoring["labels_available"]:
        warnings.append(
            "True labels are unavailable in monitoring records; "
            "live performance monitoring is not computed."
        )

    if len(current_records) < min_sample_size:
        warnings.append(
            f"Current prediction record count ({len(current_records)}) is "
            f"below recommended minimum ({min_sample_size})."
        )

    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "model_version": dict(model_version_info),
        "sample_sizes": {
            "reference_rows": int(len(reference_frame)) if reference_frame is not None else 0,
            "current_feature_rows": (
                int(len(current_feature_frame)) if current_feature_frame is not None else 0
            ),
            "prediction_records": int(len(current_records)),
        },
        "prediction_distribution": prediction_distribution,
        "binary_probability_summary": probability_summary,
        "binary_probability_drift": probability_drift,
        "feature_drift": feature_drift_summary,
        "label_monitoring": label_monitoring,
        "warnings": warnings,
    }


def build_monitoring_fallback_summary(summary: Mapping[str, Any]) -> str:
    sample_sizes = summary.get("sample_sizes", {})
    probability_drift = summary.get("binary_probability_drift", {})
    warnings = summary.get("warnings", [])

    drift_status = probability_drift.get("status", "unavailable")
    record_count = sample_sizes.get("prediction_records", 0)
    reference_count = sample_sizes.get("reference_rows", 0)

    return (
        "Monitoring summary generated for local model operations. "
        f"Compared {record_count} prediction records against {reference_count} reference rows. "
        f"Binary-probability drift status is '{drift_status}'. "
        f"Warnings reported: {len(warnings)}."
    )


def generate_monitoring_narrative(
    *,
    summary: Mapping[str, Any],
    settings: Settings,
    prefer_ollama: bool,
) -> tuple[str, str, list[str]]:
    fallback_text = build_monitoring_fallback_summary(summary)
    if not prefer_ollama:
        return fallback_text, "fallback", []

    system_prompt = (
        "You summarize local machine-learning monitoring outputs for engineering stakeholders. "
        "Be concise and factual. Mention uncertainty and avoid unsupported claims."
    )

    summary_fragment = {
        "sample_sizes": summary.get("sample_sizes", {}),
        "binary_probability_drift": summary.get("binary_probability_drift", {}),
        "prediction_distribution": summary.get("prediction_distribution", {}),
        "warning_count": len(summary.get("warnings", [])),
    }
    user_prompt = (
        "Write a short monitoring summary (3-5 sentences) from this JSON payload:\n"
        f"{json.dumps(summary_fragment, sort_keys=True)}\n"
        "Include whether labels are available and whether drift appears stable/moderate/high."
    )

    endpoint = settings.ollama_host.rstrip("/") + "/api/generate"
    payload = {
        "model": settings.ollama_model,
        "prompt": system_prompt + "\n\n" + user_prompt,
        "stream": False,
    }

    timeout = httpx.Timeout(settings.ollama_timeout_seconds)
    warnings: list[str] = []

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(endpoint, json=payload)
        response.raise_for_status()
        response_json = response.json()
        narrative = str(response_json.get("response", "")).strip()
        if narrative:
            return narrative, "ollama", warnings
        warnings.append("Ollama monitoring summary response was empty.")
    except Exception as exc:
        warnings.append(f"Ollama monitoring summary unavailable: {exc}")

    return fallback_text, "fallback", warnings


def render_monitoring_report(summary: Mapping[str, Any]) -> str:
    model_version = summary.get("model_version", {})
    sample_sizes = summary.get("sample_sizes", {})
    prediction_distribution = summary.get("prediction_distribution", {})
    probability_summary = summary.get("binary_probability_summary", {})
    probability_drift = summary.get("binary_probability_drift", {})
    feature_drift = summary.get("feature_drift", {})
    label_monitoring = summary.get("label_monitoring", {})
    warnings = summary.get("warnings", [])
    binary_counts_json = json.dumps(
        prediction_distribution.get("binary_prediction_counts", {}),
        sort_keys=True,
    )
    multiclass_counts_json = json.dumps(
        prediction_distribution.get("multiclass_prediction_counts", {}),
        sort_keys=True,
    )

    lines: list[str] = [
        "# Monitoring Report",
        "",
        f"Generated at (UTC): {summary.get('generated_at_utc', 'unknown')}",
        "",
        "## Model Version",
        "",
        f"- Binary model family: {model_version.get('binary_model', {}).get('model_family')}",
        (
            "- Binary training timestamp: "
            f"{model_version.get('binary_model', {}).get('training_timestamp_utc')}"
        ),
        (
            "- Multiclass model family: "
            f"{model_version.get('multiclass_model', {}).get('model_family')}"
        ),
        (
            "- Multiclass training timestamp: "
            f"{model_version.get('multiclass_model', {}).get('training_timestamp_utc')}"
        ),
        "",
        "## Sample Sizes",
        "",
        f"- Reference rows: {sample_sizes.get('reference_rows', 0)}",
        f"- Current feature rows: {sample_sizes.get('current_feature_rows', 0)}",
        f"- Prediction records: {sample_sizes.get('prediction_records', 0)}",
        "",
        "## Prediction Distribution",
        "",
        f"- Binary counts: {binary_counts_json}",
        f"- Multiclass counts: {multiclass_counts_json}",
        "",
        "## Binary Probability Summary",
        "",
        f"- Mean: {probability_summary.get('mean')}",
        f"- Std: {probability_summary.get('std')}",
        f"- Min: {probability_summary.get('min')}",
        f"- Max: {probability_summary.get('max')}",
        "",
        "## Drift Checks",
        "",
        f"- Binary probability drift status: {probability_drift.get('status')}",
        f"- Binary probability PSI: {probability_drift.get('psi')}",
        "",
        "### Feature Drift",
        "",
    ]

    if feature_drift:
        for feature_name in sorted(feature_drift):
            payload = feature_drift[feature_name]
            lines.append(
                f"- {feature_name}: status={payload.get('status')}, psi={payload.get('psi')}"
            )
    else:
        lines.append("- No feature drift metrics were generated.")

    lines.extend(
        [
            "",
            "## Label Availability",
            "",
            f"- Labels available: {label_monitoring.get('labels_available')}",
            f"- true_label count: {label_monitoring.get('true_label_count')}",
            f"- true_label_30d count: {label_monitoring.get('true_label_30d_count')}",
            (
                "- Performance monitoring: "
                f"{label_monitoring.get('performance_monitoring')}"
            ),
            "",
            "## Warnings",
            "",
        ]
    )

    if warnings:
        for warning in warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- No warnings.")

    narrative = summary.get("monitoring_narrative")
    if isinstance(narrative, str) and narrative:
        lines.extend(
            [
                "",
                "## Optional Narrative Summary",
                "",
                f"Mode: {summary.get('monitoring_narrative_mode', 'fallback')}",
                "",
                narrative,
            ]
        )

    return "\n".join(lines) + "\n"
