from __future__ import annotations

import importlib
import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_prompting_module = importlib.import_module("src.llm.prompting")
build_fallback_explanation_text = _prompting_module.build_fallback_explanation_text
build_prediction_summary = _prompting_module.build_prediction_summary

_predict_module = importlib.import_module("src.models.predict")
load_model = _predict_module.load_model
load_model_metadata = _predict_module.load_model_metadata
predict_from_frame = _predict_module.predict_from_frame

JsonDict = dict[str, Any]
ModelMetadataSummary = dict[str, str | float | int | bool | None]


@dataclass(frozen=True)
class DeploymentPaths:
    app_dir: Path
    project_root: Path
    artifacts_dir: Path
    reports_dir: Path
    docs_dir: Path
    readme_path: Path
    binary_model_path: Path
    multiclass_model_path: Path
    binary_metadata_path: Path
    multiclass_metadata_path: Path
    sample_payload_path: Path
    sample_batch_payload_path: Path
    sample_explain_payload_path: Path
    monitoring_summary_path: Path
    monitoring_report_path: Path
    model_comparison_report_path: Path
    demo_summary_path: Path

    @property
    def required_prediction_paths(self) -> dict[str, Path]:
        return {
            "binary_model": self.binary_model_path,
            "multiclass_model": self.multiclass_model_path,
            "binary_metadata": self.binary_metadata_path,
            "multiclass_metadata": self.multiclass_metadata_path,
        }


@dataclass(frozen=True)
class DeploymentArtifacts:
    paths: DeploymentPaths
    binary_model: Any
    multiclass_model: Any
    binary_metadata: JsonDict
    multiclass_metadata: JsonDict

    @property
    def binary_feature_columns(self) -> list[str]:
        raw = self.binary_metadata.get("feature_columns", [])
        if not isinstance(raw, list):
            return []
        return [str(column) for column in raw]

    @property
    def multiclass_feature_columns(self) -> list[str]:
        raw = self.multiclass_metadata.get("feature_columns", [])
        if not isinstance(raw, list):
            return []
        return [str(column) for column in raw]


@dataclass(frozen=True)
class DeploymentContext:
    paths: DeploymentPaths
    baseline_features: JsonDict
    dummy_features: JsonDict
    batch_example_rows: list[JsonDict]
    monitoring_summary: JsonDict | None
    monitoring_report_text: str | None
    model_comparison_report_text: str | None
    demo_summary_text: str | None
    readme_excerpt: str | None
    load_warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class PredictionOutput:
    binary_prediction: int
    binary_probability: float
    multiclass_prediction: str
    multiclass_probabilities: dict[str, float]
    model_metadata_summary: ModelMetadataSummary


@dataclass(frozen=True)
class ExplanationOutput:
    prediction_summary: str
    top_risk_increasing_factors: list[str]
    top_risk_decreasing_factors: list[str]
    explanation_text: str
    explanation_mode: str
    warnings: list[str]


@dataclass(frozen=True)
class PredictionBundle:
    prediction: PredictionOutput
    explanation: ExplanationOutput


def resolve_deployment_paths(project_root: Path | None = None) -> DeploymentPaths:
    resolved_root = (project_root or PROJECT_ROOT).resolve()
    artifacts_dir = (resolved_root / "artifacts").resolve()
    reports_dir = (resolved_root / "reports").resolve()

    return DeploymentPaths(
        app_dir=APP_DIR,
        project_root=resolved_root,
        artifacts_dir=artifacts_dir,
        reports_dir=reports_dir,
        docs_dir=(resolved_root / "docs").resolve(),
        readme_path=(resolved_root / "README.md").resolve(),
        binary_model_path=(artifacts_dir / "binary_model.joblib").resolve(),
        multiclass_model_path=(artifacts_dir / "multiclass_model.joblib").resolve(),
        binary_metadata_path=(artifacts_dir / "binary_model_metadata.json").resolve(),
        multiclass_metadata_path=(artifacts_dir / "multiclass_model_metadata.json").resolve(),
        sample_payload_path=(artifacts_dir / "sample_payload.json").resolve(),
        sample_batch_payload_path=(artifacts_dir / "sample_batch_payload.json").resolve(),
        sample_explain_payload_path=(artifacts_dir / "sample_explain_payload.json").resolve(),
        monitoring_summary_path=(reports_dir / "monitoring_summary.json").resolve(),
        monitoring_report_path=(reports_dir / "monitoring_report.md").resolve(),
        model_comparison_report_path=(reports_dir / "model_comparison_report.md").resolve(),
        demo_summary_path=(reports_dir / "demo_summary.md").resolve(),
    )


def missing_required_artifacts(paths: DeploymentPaths | None = None) -> list[Path]:
    resolved_paths = paths or resolve_deployment_paths()
    return [path for path in resolved_paths.required_prediction_paths.values() if not path.exists()]


def _read_optional_json(path: Path) -> JsonDict | None:
    if not path.exists():
        return None

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return payload
    return None


def _read_required_json(path: Path) -> JsonDict:
    if not path.exists():
        raise FileNotFoundError(f"Expected JSON artifact is missing: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON in {path}, got {type(payload).__name__}.")
    return payload


def _read_optional_text(path: Path, *, max_chars: int = 6000) -> str | None:
    if not path.exists():
        return None

    text = path.read_text(encoding="utf-8")
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n\n..."


def _extract_features(payload: Mapping[str, Any] | None) -> JsonDict:
    if payload is None:
        return {}

    features = payload.get("features")
    if not isinstance(features, Mapping):
        return {}

    return {str(key): value for key, value in features.items()}


def _extract_batch_rows(payload: Mapping[str, Any] | None) -> list[JsonDict]:
    if payload is None:
        return []

    rows = payload.get("rows")
    if not isinstance(rows, list):
        return []

    output_rows: list[JsonDict] = []
    for row in rows:
        if isinstance(row, Mapping):
            output_rows.append({str(key): value for key, value in row.items()})
    return output_rows


def build_feature_defaults(
    *,
    feature_columns: list[str],
    base_features: Mapping[str, Any],
) -> JsonDict:
    return {feature_name: base_features.get(feature_name) for feature_name in feature_columns}


def load_deployment_context(paths: DeploymentPaths | None = None) -> DeploymentContext:
    resolved_paths = paths or resolve_deployment_paths()

    warnings: list[str] = []

    try:
        sample_payload = _read_required_json(resolved_paths.sample_payload_path)
    except Exception:
        sample_payload = _read_optional_json(resolved_paths.sample_payload_path)
        warnings.append(
            "sample_payload.json is missing or invalid; using fallback defaults where possible."
        )

    try:
        sample_explain_payload = _read_required_json(resolved_paths.sample_explain_payload_path)
    except Exception:
        sample_explain_payload = _read_optional_json(resolved_paths.sample_explain_payload_path)
        warnings.append(
            "sample_explain_payload.json is missing or invalid; dummy example may be limited."
        )

    sample_batch_payload = _read_optional_json(resolved_paths.sample_batch_payload_path)

    batch_rows = _extract_batch_rows(sample_batch_payload)
    baseline_features = _extract_features(sample_payload)
    if not baseline_features and batch_rows:
        baseline_features = dict(batch_rows[0])

    dummy_features = _extract_features(sample_explain_payload)
    if not dummy_features and len(batch_rows) >= 2:
        dummy_features = dict(batch_rows[1])
    if not dummy_features:
        dummy_features = dict(baseline_features)

    if not baseline_features:
        warnings.append(
            "No baseline feature payload was loaded; form defaults may require manual input."
        )

    return DeploymentContext(
        paths=resolved_paths,
        baseline_features=baseline_features,
        dummy_features=dummy_features,
        batch_example_rows=batch_rows,
        monitoring_summary=_read_optional_json(resolved_paths.monitoring_summary_path),
        monitoring_report_text=_read_optional_text(resolved_paths.monitoring_report_path),
        model_comparison_report_text=_read_optional_text(
            resolved_paths.model_comparison_report_path
        ),
        demo_summary_text=_read_optional_text(resolved_paths.demo_summary_path),
        readme_excerpt=_read_optional_text(resolved_paths.readme_path, max_chars=3500),
        load_warnings=tuple(warnings),
    )


def load_deployment_artifacts(paths: DeploymentPaths | None = None) -> DeploymentArtifacts:
    resolved_paths = paths or resolve_deployment_paths()
    missing = missing_required_artifacts(resolved_paths)
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Required model artifacts for Streamlit deployment are missing: "
            f"{missing_text}. Run training/evaluation first."
        )

    return DeploymentArtifacts(
        paths=resolved_paths,
        binary_model=load_model(resolved_paths.binary_model_path),
        multiclass_model=load_model(resolved_paths.multiclass_model_path),
        binary_metadata=load_model_metadata(resolved_paths.binary_metadata_path),
        multiclass_metadata=load_model_metadata(resolved_paths.multiclass_metadata_path),
    )


def _parse_label_decoder(metadata: Mapping[str, Any]) -> dict[int, str] | None:
    mapping_raw = metadata.get("label_mapping", {})
    if not isinstance(mapping_raw, Mapping) or not mapping_raw:
        return None

    parsed: dict[int, str] = {}
    for key, value in mapping_raw.items():
        parsed[int(key)] = str(value)
    return parsed


def _binary_prediction_to_int(raw_prediction: str) -> int:
    normalized = raw_prediction.strip().lower()
    if normalized in {"1", "true", "yes", "<30"}:
        return 1
    return 0


def _first_probability_or_default(
    probabilities_by_class: Mapping[str, list[float]],
    index: int,
    *,
    positive_labels: tuple[str, ...],
) -> float:
    for label in positive_labels:
        values = probabilities_by_class.get(label)
        if values is not None and index < len(values):
            return float(values[index])
    return 0.0


def _model_metadata_summary(artifacts: DeploymentArtifacts) -> ModelMetadataSummary:
    binary_training_device = artifacts.binary_metadata.get("xgboost_device_used_for_training")
    if binary_training_device is None:
        binary_training_device = artifacts.binary_metadata.get("xgboost_device_used")

    binary_inference_device = artifacts.binary_metadata.get("xgboost_device_used_for_inference")
    if binary_inference_device is None:
        binary_inference_device = binary_training_device

    multiclass_training_device = artifacts.multiclass_metadata.get(
        "xgboost_device_used_for_training"
    )
    if multiclass_training_device is None:
        multiclass_training_device = artifacts.multiclass_metadata.get("xgboost_device_used")

    multiclass_inference_device = artifacts.multiclass_metadata.get(
        "xgboost_device_used_for_inference"
    )
    if multiclass_inference_device is None:
        multiclass_inference_device = multiclass_training_device

    return {
        "binary_model_family": str(artifacts.binary_metadata.get("model_family")),
        "multiclass_model_family": str(artifacts.multiclass_metadata.get("model_family")),
        "binary_training_timestamp_utc": artifacts.binary_metadata.get("training_timestamp_utc"),
        "multiclass_training_timestamp_utc": artifacts.multiclass_metadata.get(
            "training_timestamp_utc"
        ),
        "binary_xgboost_device_requested": artifacts.binary_metadata.get(
            "xgboost_device_requested"
        ),
        "binary_xgboost_device_used_for_training": binary_training_device,
        "binary_xgboost_device_used_for_inference": binary_inference_device,
        "binary_xgboost_inference_fallback_path": artifacts.binary_metadata.get(
            "xgboost_inference_used_fallback_path"
        ),
        "multiclass_xgboost_device_requested": artifacts.multiclass_metadata.get(
            "xgboost_device_requested"
        ),
        "multiclass_xgboost_device_used_for_training": multiclass_training_device,
        "multiclass_xgboost_device_used_for_inference": multiclass_inference_device,
        "multiclass_xgboost_inference_fallback_path": artifacts.multiclass_metadata.get(
            "xgboost_inference_used_fallback_path"
        ),
    }


def predict_single_row(
    *,
    features: Mapping[str, Any],
    artifacts: DeploymentArtifacts,
) -> PredictionOutput:
    frame = pd.DataFrame([dict(features)])

    binary_output = predict_from_frame(
        model=artifacts.binary_model,
        frame=frame,
        feature_columns=artifacts.binary_feature_columns,
        task_type="binary",
        label_decoder=_parse_label_decoder(artifacts.binary_metadata),
    )
    multiclass_output = predict_from_frame(
        model=artifacts.multiclass_model,
        frame=frame,
        feature_columns=artifacts.multiclass_feature_columns,
        task_type="multiclass",
        label_decoder=_parse_label_decoder(artifacts.multiclass_metadata),
    )

    metadata_summary = _model_metadata_summary(artifacts)

    if binary_output.positive_class_probability is not None:
        binary_probability = float(binary_output.positive_class_probability[0])
    else:
        binary_probability = _first_probability_or_default(
            binary_output.probabilities_by_class,
            0,
            positive_labels=("1", "<30"),
        )

    multiclass_probabilities = {
        label: float(values[0])
        for label, values in multiclass_output.probabilities_by_class.items()
        if values
    }

    return PredictionOutput(
        binary_prediction=_binary_prediction_to_int(binary_output.predictions[0]),
        binary_probability=binary_probability,
        multiclass_prediction=multiclass_output.predictions[0],
        multiclass_probabilities=multiclass_probabilities,
        model_metadata_summary=metadata_summary,
    )


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fallback_factor_lists(
    features: Mapping[str, Any],
    *,
    top_n_factors: int,
) -> tuple[list[str], list[str]]:
    increasing: list[str] = []
    decreasing: list[str] = []

    recurrency = _to_float(features.get("recurrency"))
    if recurrency is not None:
        if recurrency >= 1:
            increasing.append(f"Repeated encounter burden (recurrency={recurrency:.0f})")
        else:
            decreasing.append("No repeated encounter signal (recurrency near 0)")

    severity = _to_float(features.get("patient_severity"))
    if severity is not None:
        if severity >= 0.40:
            increasing.append(f"Higher severity index (patient_severity={severity:.2f})")
        elif severity <= 0.20:
            decreasing.append(f"Lower severity index (patient_severity={severity:.2f})")

    utilization = _to_float(features.get("utilization_intensity"))
    if utilization is not None:
        if utilization >= 3:
            increasing.append(f"Higher utilization intensity ({utilization:.0f})")
        elif utilization <= 1:
            decreasing.append(f"Lower utilization intensity ({utilization:.0f})")

    medication_change = _to_float(features.get("medication_change_ratio"))
    if medication_change is not None:
        if medication_change >= 0.20:
            increasing.append(
                f"More medication changes (medication_change_ratio={medication_change:.2f})"
            )
        elif medication_change <= 0.05:
            decreasing.append(
                f"Limited medication changes (medication_change_ratio={medication_change:.2f})"
            )

    complex_discharge = _to_float(features.get("complex_discharge_flag"))
    if complex_discharge is not None:
        if complex_discharge >= 1:
            increasing.append("Complex discharge disposition signal present")
        else:
            decreasing.append("Home-like discharge disposition signal present")

    age_bucket_risk = _to_float(features.get("age_bucket_risk"))
    if age_bucket_risk is not None:
        if age_bucket_risk >= 7:
            increasing.append(f"Higher age-risk bucket (age_bucket_risk={age_bucket_risk:.0f})")
        elif age_bucket_risk >= 0 and age_bucket_risk <= 4:
            decreasing.append(f"Lower age-risk bucket (age_bucket_risk={age_bucket_risk:.0f})")

    if not increasing:
        increasing.append(
            "Predicted risk level is mostly driven by overall model feature interactions"
        )
    if not decreasing:
        decreasing.append("Limited counterbalancing low-risk signals were detected")

    return increasing[:top_n_factors], decreasing[:top_n_factors]


def build_deterministic_explanation(
    *,
    features: Mapping[str, Any],
    prediction: PredictionOutput,
    top_n_factors: int,
) -> ExplanationOutput:
    top_n = max(1, min(top_n_factors, 10))

    prediction_summary = build_prediction_summary(
        binary_prediction=prediction.binary_prediction,
        binary_probability=prediction.binary_probability,
        multiclass_prediction=prediction.multiclass_prediction,
        multiclass_probabilities=prediction.multiclass_probabilities,
    )

    increasing_factors, decreasing_factors = _fallback_factor_lists(
        features,
        top_n_factors=top_n,
    )

    explanation_text = build_fallback_explanation_text(
        prediction_summary=prediction_summary,
        increasing_factors=increasing_factors,
        decreasing_factors=decreasing_factors,
    )

    return ExplanationOutput(
        prediction_summary=prediction_summary,
        top_risk_increasing_factors=increasing_factors,
        top_risk_decreasing_factors=decreasing_factors,
        explanation_text=explanation_text,
        explanation_mode="fallback",
        warnings=[],
    )


def predict_with_deterministic_explanation(
    *,
    features: Mapping[str, Any],
    artifacts: DeploymentArtifacts,
    top_n_factors: int = 3,
) -> PredictionBundle:
    prediction = predict_single_row(features=features, artifacts=artifacts)
    explanation = build_deterministic_explanation(
        features=features,
        prediction=prediction,
        top_n_factors=top_n_factors,
    )
    return PredictionBundle(prediction=prediction, explanation=explanation)


def resolve_existing_path(raw_path: Any, *, project_root: Path) -> Path | None:
    if raw_path is None:
        return None

    text = str(raw_path).strip()
    if not text:
        return None

    direct_candidate = Path(text)
    if direct_candidate.exists():
        return direct_candidate.resolve()

    normalized = text.replace("\\", "/")

    if normalized.startswith("./"):
        relative_candidate = (project_root / normalized[2:]).resolve()
        if relative_candidate.exists():
            return relative_candidate

    for anchor in ("artifacts/", "reports/", "docs/", "data/"):
        anchor_index = normalized.lower().find(anchor)
        if anchor_index >= 0:
            suffix = normalized[anchor_index:]
            relative_candidate = (project_root / Path(suffix)).resolve()
            if relative_candidate.exists():
                return relative_candidate

    if normalized.startswith("/") and len(normalized) > 2 and normalized[2] == ":":
        windows_drive_candidate = Path(normalized[1:])
        if windows_drive_candidate.exists():
            return windows_drive_candidate.resolve()

    return None
