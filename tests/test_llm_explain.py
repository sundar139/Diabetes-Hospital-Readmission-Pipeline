from __future__ import annotations

import asyncio
from typing import Any

from src.config.settings import Settings
from src.llm import explain as llm_explain


def _sample_features() -> dict[str, Any]:
    return {
        "recurrency": 2,
        "patient_severity": 0.52,
        "utilization_intensity": 4,
        "medication_change_ratio": 0.24,
        "complex_discharge_flag": 1,
        "age_bucket_risk": 8,
    }


def _sample_prediction_payload() -> dict[str, Any]:
    return {
        "binary_prediction": 1,
        "binary_probability": 0.81,
        "multiclass_prediction": "<30",
        "multiclass_probabilities": {
            "NO": 0.10,
            ">30": 0.09,
            "<30": 0.81,
        },
    }


def test_generate_explanation_uses_fallback_when_ollama_disabled(monkeypatch: Any) -> None:
    def fake_shap_factor_lists(**_: Any) -> tuple[list[str], list[str], list[str]]:
        return ["recurrency (+0.51)"], ["complex_discharge_flag (-0.10)"], []

    monkeypatch.setattr(llm_explain, "_shap_factor_lists", fake_shap_factor_lists)

    output = asyncio.run(
        llm_explain.generate_explanation(
            features=_sample_features(),
            binary_model=object(),
            prediction_payload=_sample_prediction_payload(),
            settings=Settings(),
            prefer_ollama=False,
            top_n_factors=3,
        )
    )

    assert output.explanation_mode == "fallback"
    assert "not medical advice" in output.explanation_text
    assert output.top_risk_increasing_factors == ["recurrency (+0.51)"]


def test_generate_explanation_uses_ollama_text_when_available(monkeypatch: Any) -> None:
    def fake_shap_factor_lists(**_: Any) -> tuple[list[str], list[str], list[str]]:
        return ["recurrency (+0.51)"], ["complex_discharge_flag (-0.10)"], []

    async def fake_request_ollama_explanation(**_: Any) -> tuple[str | None, str | None]:
        return "Model-generated non-diagnostic explanation from Ollama.", None

    monkeypatch.setattr(llm_explain, "_shap_factor_lists", fake_shap_factor_lists)
    monkeypatch.setattr(
        llm_explain,
        "_request_ollama_explanation",
        fake_request_ollama_explanation,
    )

    output = asyncio.run(
        llm_explain.generate_explanation(
            features=_sample_features(),
            binary_model=object(),
            prediction_payload=_sample_prediction_payload(),
            settings=Settings(),
            prefer_ollama=True,
            top_n_factors=3,
        )
    )

    assert output.explanation_mode == "ollama"
    assert output.explanation_text.startswith("Model-generated")
    assert output.warnings == []


def test_generate_explanation_falls_back_when_ollama_unavailable(monkeypatch: Any) -> None:
    def fake_shap_factor_lists(**_: Any) -> tuple[list[str], list[str], list[str]]:
        return [], [], []

    async def fake_request_ollama_explanation(**_: Any) -> tuple[str | None, str | None]:
        return None, "Ollama request failed: connection refused"

    monkeypatch.setattr(llm_explain, "_shap_factor_lists", fake_shap_factor_lists)
    monkeypatch.setattr(
        llm_explain,
        "_request_ollama_explanation",
        fake_request_ollama_explanation,
    )

    output = asyncio.run(
        llm_explain.generate_explanation(
            features=_sample_features(),
            binary_model=object(),
            prediction_payload=_sample_prediction_payload(),
            settings=Settings(),
            prefer_ollama=True,
            top_n_factors=2,
        )
    )

    assert output.explanation_mode == "fallback"
    assert len(output.top_risk_increasing_factors) == 2
    assert any("Ollama request failed" in warning for warning in output.warnings)
