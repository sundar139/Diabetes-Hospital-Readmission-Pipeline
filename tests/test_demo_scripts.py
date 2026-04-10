from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any


def _load_script_module(script_name: str) -> ModuleType:
    module_path = Path(__file__).resolve().parents[1] / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load script module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_demo_scripts_are_import_safe() -> None:
    for script_name in (
        "demo_smoke_run.py",
        "demo_prediction_examples.py",
        "demo_generate_showcase_artifacts.py",
    ):
        module = _load_script_module(script_name)
        assert hasattr(module, "main")


def test_demo_prediction_examples_writes_response_files(tmp_path: Path) -> None:
    module = _load_script_module("demo_prediction_examples.py")
    artifacts_dir = Path(__file__).resolve().parents[1] / "artifacts"

    def fake_request(
        method: str,
        url: str,
        payload: dict[str, Any] | None,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        assert timeout_seconds > 0

        if url.endswith("/health"):
            return {"status": "ok", "models_loaded": True}

        if url.endswith("/predict"):
            assert payload is not None
            return {
                "binary_prediction": 1,
                "binary_probability": 0.77,
                "multiclass_prediction": "<30",
                "multiclass_probabilities": {"NO": 0.1, ">30": 0.13, "<30": 0.77},
                "model_metadata_summary": {},
            }

        if url.endswith("/predict-batch"):
            return {
                "n_rows": 2,
                "predictions": [
                    {
                        "binary_prediction": 0,
                        "binary_probability": 0.2,
                        "multiclass_prediction": "NO",
                        "multiclass_probabilities": {"NO": 0.8, ">30": 0.15, "<30": 0.05},
                        "model_metadata_summary": {},
                    },
                    {
                        "binary_prediction": 1,
                        "binary_probability": 0.7,
                        "multiclass_prediction": "<30",
                        "multiclass_probabilities": {"NO": 0.1, ">30": 0.2, "<30": 0.7},
                        "model_metadata_summary": {},
                    },
                ],
            }

        if url.endswith("/explain"):
            return {
                "binary_prediction": 1,
                "binary_probability": 0.72,
                "multiclass_prediction": "<30",
                "multiclass_probabilities": {"NO": 0.1, ">30": 0.18, "<30": 0.72},
                "model_metadata_summary": {},
                "prediction_summary": "Example explanation summary.",
                "top_risk_increasing_factors": ["recurrency (+0.6)"],
                "top_risk_decreasing_factors": ["complex_discharge_flag (-0.2)"],
                "explanation_text": "Example explanation text.",
                "explanation_mode": "fallback",
                "warnings": [],
            }

        raise AssertionError(f"Unexpected request URL: {method} {url}")

    generated = module.run_demo_prediction_examples(
        api_base_url="http://127.0.0.1:8000",
        output_dir=tmp_path,
        single_payload_path=artifacts_dir / "sample_payload.json",
        batch_payload_path=artifacts_dir / "sample_batch_payload.json",
        explain_payload_path=artifacts_dir / "sample_explain_payload.json",
        timeout_seconds=5.0,
        mode="all",
        requester=fake_request,
        open_docs=False,
    )

    assert generated["health_response"].exists()
    assert generated["predict_response"].exists()
    assert generated["predict_batch_response"].exists()
    assert generated["explain_response"].exists()
    assert generated["request_manifest"].exists()


def test_showcase_summary_builder_produces_expected_sections(tmp_path: Path) -> None:
    module = _load_script_module("demo_generate_showcase_artifacts.py")

    generated_paths = {
        "predict_response": tmp_path / "sample_prediction_response.json",
        "explain_response": tmp_path / "sample_explanation_response.json",
        "monitoring_summary": tmp_path / "monitoring_summary.json",
    }

    summary = module.build_demo_summary_markdown(
        api_base_url="http://127.0.0.1:8000",
        prediction_response={
            "binary_probability": 0.77,
            "multiclass_prediction": "<30",
        },
        batch_response={"n_rows": 2},
        explain_response={"explanation_mode": "fallback"},
        monitoring_summary={
            "binary_probability_drift": {"status": "stable", "psi": 0.01},
            "warnings": [],
        },
        generated_paths=generated_paths,
    )

    assert "# Demo Summary" in summary
    assert "## Local Endpoints" in summary
    assert "## Demo Response Highlights" in summary
    assert "## Monitoring Highlights" in summary

    manifest = module.build_demo_manifest_payload(
        api_base_url="http://127.0.0.1:8000",
        generated_paths=generated_paths,
    )

    assert manifest["api_base_url"] == "http://127.0.0.1:8000"
    assert "generated_files" in manifest


def test_demo_smoke_prerequisite_validator_detects_missing_files(tmp_path: Path) -> None:
    module = _load_script_module("demo_smoke_run.py")

    missing_before = module.validate_demo_prerequisites(tmp_path)
    assert missing_before

    required_paths = module._required_demo_artifacts(tmp_path)
    for path in required_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")

    missing_after = module.validate_demo_prerequisites(tmp_path)
    assert missing_after == []
