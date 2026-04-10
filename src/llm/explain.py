from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import httpx
import pandas as pd

from src.config.settings import Settings
from src.llm.prompting import (
    build_fallback_explanation_text,
    build_ollama_system_prompt,
    build_ollama_user_prompt,
    build_prediction_summary,
)
from src.models.predict import explain_single_prediction


@dataclass(frozen=True)
class ExplanationOutput:
    prediction_summary: str
    top_risk_increasing_factors: list[str]
    top_risk_decreasing_factors: list[str]
    explanation_text: str
    explanation_mode: str
    warnings: list[str]


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


def _shap_factor_lists(
    *,
    binary_model: Any,
    row_frame: pd.DataFrame,
    top_n_factors: int,
) -> tuple[list[str], list[str], list[str]]:
    try:
        contributions = explain_single_prediction(
            model=binary_model,
            row_frame=row_frame,
            top_n=max(2 * top_n_factors, top_n_factors),
        )
    except Exception as exc:
        return [], [], [f"Per-row contribution analysis unavailable: {exc}"]

    increasing: list[str] = []
    decreasing: list[str] = []
    for item in contributions:
        feature_name = str(item.get("feature", "unknown_feature"))
        contribution = _to_float(item.get("contribution"))
        if contribution is None:
            continue

        label = f"{feature_name} ({contribution:+.3f})"
        if contribution > 0:
            increasing.append(label)
        elif contribution < 0:
            decreasing.append(label)

    return increasing[:top_n_factors], decreasing[:top_n_factors], []


async def _request_ollama_explanation(
    *,
    settings: Settings,
    system_prompt: str,
    user_prompt: str,
) -> tuple[str | None, str | None]:
    endpoint = settings.ollama_host.rstrip("/") + "/api/generate"
    prompt = system_prompt + "\n\n" + user_prompt
    payload = {
        "model": settings.ollama_model,
        "prompt": prompt,
        "stream": False,
    }

    timeout = httpx.Timeout(settings.ollama_timeout_seconds)
    last_error: str | None = None

    for _ in range(2):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(endpoint, json=payload)
            response.raise_for_status()
            response_json = response.json()
            text = str(response_json.get("response", "")).strip()
            if not text:
                last_error = "Ollama response was empty."
                continue
            return text, None
        except Exception as exc:
            last_error = f"Ollama request failed: {exc}"

    return None, last_error


async def generate_explanation(
    *,
    features: Mapping[str, Any],
    binary_model: Any,
    prediction_payload: Mapping[str, Any],
    settings: Settings,
    prefer_ollama: bool,
    top_n_factors: int,
) -> ExplanationOutput:
    warnings: list[str] = []

    binary_prediction = int(prediction_payload.get("binary_prediction", 0))
    binary_probability = float(prediction_payload.get("binary_probability", 0.0))
    multiclass_prediction = str(prediction_payload.get("multiclass_prediction", "unknown"))
    multiclass_probabilities_raw = prediction_payload.get("multiclass_probabilities", {})

    multiclass_probabilities: dict[str, float] = {}
    if isinstance(multiclass_probabilities_raw, Mapping):
        for key, value in multiclass_probabilities_raw.items():
            try:
                multiclass_probabilities[str(key)] = float(value)
            except (TypeError, ValueError):
                continue

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
        top_n_factors=top_n_factors,
    )
    warnings.extend(shap_warnings)

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
            top_n_factors=top_n_factors,
        )

    fallback_text = build_fallback_explanation_text(
        prediction_summary=prediction_summary,
        increasing_factors=increasing_factors,
        decreasing_factors=decreasing_factors,
    )

    explanation_mode = "fallback"
    explanation_text = fallback_text

    if prefer_ollama:
        system_prompt = build_ollama_system_prompt()
        user_prompt = build_ollama_user_prompt(
            prediction_summary=prediction_summary,
            increasing_factors=increasing_factors,
            decreasing_factors=decreasing_factors,
        )
        ollama_text, ollama_warning = await _request_ollama_explanation(
            settings=settings,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        if ollama_text:
            explanation_mode = "ollama"
            explanation_text = ollama_text
        elif ollama_warning:
            warnings.append(ollama_warning)

    return ExplanationOutput(
        prediction_summary=prediction_summary,
        top_risk_increasing_factors=increasing_factors,
        top_risk_decreasing_factors=decreasing_factors,
        explanation_text=explanation_text,
        explanation_mode=explanation_mode,
        warnings=warnings,
    )
