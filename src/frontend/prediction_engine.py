from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.frontend.loaders import PredictionArtifacts
from src.llm.prompting import build_fallback_explanation_text, build_prediction_summary

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


def _normalize_xgboost_device(raw_value: Any) -> str | None:
    if raw_value is None:
        return None

    normalized = str(raw_value).strip().lower()
    if not normalized:
        return None

    if normalized.startswith("cuda") or normalized.startswith("gpu"):
        return "cuda"
    if normalized == "cpu":
        return "cpu"
    return normalized


def _extract_xgboost_classifier(model: Any) -> Any | None:
    if model.__class__.__name__ == "XGBClassifier":
        return model

    named_steps = getattr(model, "named_steps", None)
    if not isinstance(named_steps, Mapping):
        return None

    classifier = named_steps.get("classifier")
    if classifier is None:
        return None
    if classifier.__class__.__name__ != "XGBClassifier":
        return None
    return classifier


def _ensure_cpu_for_xgboost_inference(model: Any) -> dict[str, Any]:
    classifier = _extract_xgboost_classifier(model)
    if classifier is None:
        return {
            "xgboost_device_used_for_inference": None,
            "inference_used_fallback_path": False,
        }

    params = classifier.get_params(deep=False) if hasattr(classifier, "get_params") else {}
    requested_device = _normalize_xgboost_device(params.get("device"))
    if requested_device is None and hasattr(classifier, "get_xgb_params"):
        try:
            requested_device = _normalize_xgboost_device(classifier.get_xgb_params().get("device"))
        except Exception:
            requested_device = None

    if requested_device is None:
        requested_device = "cpu"

    if requested_device == "cpu":
        return {
            "xgboost_device_used_for_inference": "cpu",
            "inference_used_fallback_path": False,
        }

    try:
        classifier.set_params(device="cpu")
        return {
            "xgboost_device_used_for_inference": "cpu",
            "inference_used_fallback_path": True,
        }
    except Exception:
        return {
            "xgboost_device_used_for_inference": requested_device,
            "inference_used_fallback_path": False,
        }


def _select_model_features(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    missing_columns = [column for column in feature_columns if column not in frame.columns]
    if missing_columns:
        raise KeyError("Missing feature columns for inference: " + ", ".join(missing_columns))
    return frame[feature_columns].copy()


def _decode_predictions(
    values: np.ndarray,
    *,
    label_decoder: dict[int, str] | None,
) -> list[str]:
    if label_decoder is None:
        return [str(value) for value in values.tolist()]

    decoded: list[str] = []
    for value in values.tolist():
        try:
            decoded.append(label_decoder[int(value)])
        except Exception:
            decoded.append(str(value))
    return decoded


def _predict_model(
    *,
    model: Any,
    frame: pd.DataFrame,
    feature_columns: list[str],
    task_type: str,
    label_decoder: dict[int, str] | None,
) -> dict[str, Any]:
    inference_runtime = _ensure_cpu_for_xgboost_inference(model)
    x = _select_model_features(frame, feature_columns)

    raw_predictions = np.asarray(model.predict(x))
    decoded_predictions = _decode_predictions(raw_predictions, label_decoder=label_decoder)

    probabilities_by_class: dict[str, list[float]] = {}
    positive_probability: list[float] | None = None

    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(x), dtype=float)
        classes = np.asarray(getattr(model, "classes_", np.arange(probabilities.shape[1])))

        for idx, class_value in enumerate(classes.tolist()):
            class_key = str(class_value)
            if label_decoder is not None:
                try:
                    class_key = label_decoder[int(class_value)]
                except Exception:
                    class_key = str(class_value)
            probabilities_by_class[class_key] = probabilities[:, idx].tolist()

        if task_type == "binary" and probabilities.shape[1] >= 2:
            positive_index = 1
            if 1 in classes:
                positive_index = int(np.where(classes == 1)[0][0])
            positive_probability = probabilities[:, positive_index].tolist()

    return {
        "predictions": decoded_predictions,
        "probabilities_by_class": probabilities_by_class,
        "positive_class_probability": positive_probability,
        "inference_runtime": inference_runtime,
    }


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
        elif 0 <= age_bucket_risk <= 4:
            decreasing.append(f"Lower age-risk bucket (age_bucket_risk={age_bucket_risk:.0f})")

    if not increasing:
        increasing.append(
            "Predicted risk level is mostly driven by overall model feature interactions"
        )
    if not decreasing:
        decreasing.append("Limited counterbalancing low-risk signals were detected")

    return increasing[:top_n_factors], decreasing[:top_n_factors]


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

    binary_output = _predict_model(
        model=artifacts.binary_model,
        frame=frame,
        feature_columns=artifacts.binary_feature_columns,
        task_type="binary",
        label_decoder=_parse_label_decoder(artifacts.binary_metadata),
    )
    multiclass_output = _predict_model(
        model=artifacts.multiclass_model,
        frame=frame,
        feature_columns=artifacts.multiclass_feature_columns,
        task_type="multiclass",
        label_decoder=_parse_label_decoder(artifacts.multiclass_metadata),
    )

    metadata_summary = _model_metadata_summary(artifacts)
    binary_runtime = binary_output.get("inference_runtime", {}) or {}
    multiclass_runtime = multiclass_output.get("inference_runtime", {}) or {}
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

    binary_positive_probability = binary_output.get("positive_class_probability")
    binary_probabilities_by_class = binary_output.get("probabilities_by_class", {})
    multiclass_probabilities_by_class = multiclass_output.get("probabilities_by_class", {})

    if binary_positive_probability is not None:
        binary_probability = float(binary_positive_probability[0])
    else:
        binary_probability = _first_probability_or_default(
            binary_probabilities_by_class,
            0,
            positive_labels=("1", "<30"),
        )

    multiclass_probabilities = {
        label: float(values[0])
        for label, values in multiclass_probabilities_by_class.items()
        if values
    }

    binary_predictions = binary_output.get("predictions", ["0"])
    multiclass_predictions = multiclass_output.get("predictions", ["unknown"])

    return FrontendPrediction(
        binary_prediction=_binary_prediction_to_int(str(binary_predictions[0])),
        binary_probability=binary_probability,
        multiclass_prediction=str(multiclass_predictions[0]),
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
        warnings=[],
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
        prediction_payload={
            "binary_prediction": prediction.binary_prediction,
            "binary_probability": prediction.binary_probability,
            "multiclass_prediction": prediction.multiclass_prediction,
            "multiclass_probabilities": prediction.multiclass_probabilities,
        },
        top_n_factors=top_n_factors,
    )

    return PredictionWithExplanation(prediction=prediction, explanation=explanation)
