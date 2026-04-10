from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.frontend.loaders import PredictionArtifacts
from src.llm.explain import _fallback_factor_lists, _shap_factor_lists
from src.llm.prompting import build_fallback_explanation_text, build_prediction_summary
from src.models.predict import predict_from_frame

ModelMetadataSummary = dict[str, str | float | int | bool | None]


@dataclass(frozen=True)
class FrontendPrediction:
    binary_prediction: int
    binary_probability: float
    multiclass_prediction: str
    multiclass_probabilities: dict[str, float]
    model_metadata_summary: ModelMetadataSummary


@dataclass(frozen=True)
class DeterministicExplanation:
    prediction_summary: str
    top_risk_increasing_factors: list[str]
    top_risk_decreasing_factors: list[str]
    explanation_text: str
    explanation_mode: str
    warnings: list[str]


@dataclass(frozen=True)
class PredictionWithExplanation:
    prediction: FrontendPrediction
    explanation: DeterministicExplanation


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


def _model_metadata_summary(artifacts: PredictionArtifacts) -> ModelMetadataSummary:
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
    artifacts: PredictionArtifacts,
) -> FrontendPrediction:
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
    binary_runtime = binary_output.inference_runtime or {}
    multiclass_runtime = multiclass_output.inference_runtime or {}
    metadata_summary["binary_runtime_device"] = binary_runtime.get(
        "xgboost_device_used_for_inference"
    )
    metadata_summary["binary_runtime_fallback_path"] = binary_runtime.get(
        "inference_used_fallback_path"
    )
    metadata_summary["multiclass_runtime_device"] = multiclass_runtime.get(
        "xgboost_device_used_for_inference"
    )
    metadata_summary["multiclass_runtime_fallback_path"] = multiclass_runtime.get(
        "inference_used_fallback_path"
    )

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

    return FrontendPrediction(
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


def _normalize_multiclass_probabilities(raw_value: Any) -> dict[str, float]:
    if not isinstance(raw_value, Mapping):
        return {}

    parsed: dict[str, float] = {}
    for key, value in raw_value.items():
        numeric_value = _to_float(value)
        if numeric_value is not None:
            parsed[str(key)] = numeric_value
    return parsed


def build_deterministic_explanation(
    *,
    features: Mapping[str, Any],
    binary_model: Any,
    prediction_payload: Mapping[str, Any],
    top_n_factors: int = 3,
) -> DeterministicExplanation:
    top_n = max(1, min(top_n_factors, 10))

    binary_prediction = int(prediction_payload.get("binary_prediction", 0))
    binary_probability = float(prediction_payload.get("binary_probability", 0.0))
    multiclass_prediction = str(prediction_payload.get("multiclass_prediction", "unknown"))
    multiclass_probabilities = _normalize_multiclass_probabilities(
        prediction_payload.get("multiclass_probabilities", {})
    )

    prediction_summary = build_prediction_summary(
        binary_prediction=binary_prediction,
        binary_probability=binary_probability,
        multiclass_prediction=multiclass_prediction,
        multiclass_probabilities=multiclass_probabilities,
    )

    row_frame = pd.DataFrame([dict(features)])
    shap_increasing, shap_decreasing, shap_warnings = _shap_factor_lists(
        binary_model=binary_model,
        row_frame=row_frame,
        top_n_factors=top_n,
    )

    if shap_increasing or shap_decreasing:
        increasing_factors = shap_increasing or [
            "No positive contribution factors were available for this row"
        ]
        decreasing_factors = shap_decreasing or [
            "No negative contribution factors were available for this row"
        ]
    else:
        increasing_factors, decreasing_factors = _fallback_factor_lists(
            features,
            top_n_factors=top_n,
        )

    explanation_text = build_fallback_explanation_text(
        prediction_summary=prediction_summary,
        increasing_factors=increasing_factors,
        decreasing_factors=decreasing_factors,
    )

    return DeterministicExplanation(
        prediction_summary=prediction_summary,
        top_risk_increasing_factors=increasing_factors,
        top_risk_decreasing_factors=decreasing_factors,
        explanation_text=explanation_text,
        explanation_mode="fallback",
        warnings=shap_warnings,
    )


def predict_with_deterministic_explanation(
    *,
    features: Mapping[str, Any],
    artifacts: PredictionArtifacts,
    top_n_factors: int = 3,
) -> PredictionWithExplanation:
    prediction = predict_single_row(features=features, artifacts=artifacts)
    explanation = build_deterministic_explanation(
        features=features,
        binary_model=artifacts.binary_model,
        prediction_payload={
            "binary_prediction": prediction.binary_prediction,
            "binary_probability": prediction.binary_probability,
            "multiclass_prediction": prediction.multiclass_prediction,
            "multiclass_probabilities": prediction.multiclass_probabilities,
        },
        top_n_factors=top_n_factors,
    )

    return PredictionWithExplanation(prediction=prediction, explanation=explanation)
