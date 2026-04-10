from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

from src.frontend.loaders import missing_prediction_artifacts, resolve_frontend_paths
from src.frontend.prediction_engine import build_deterministic_explanation


def test_frontend_module_imports() -> None:
    module_names = (
        "src.frontend",
        "src.frontend.loaders",
        "src.frontend.prediction_engine",
        "src.frontend.prediction_ui",
        "src.frontend.analytics_ui",
        "src.frontend.monitoring_ui",
        "src.frontend.explanation_ui",
        "src.frontend.utils",
    )

    for module_name in module_names:
        imported = importlib.import_module(module_name)
        assert imported is not None


def test_streamlit_entrypoint_imports_main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    entrypoint = project_root / "streamlit_app.py"

    spec = importlib.util.spec_from_file_location("streamlit_app", entrypoint)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert hasattr(module, "main")


def test_deterministic_explanation_fallback_contract() -> None:
    explanation = build_deterministic_explanation(
        features={
            "recurrency": 2,
            "patient_severity": 0.52,
            "utilization_intensity": 4,
            "medication_change_ratio": 0.24,
            "complex_discharge_flag": 1,
            "age_bucket_risk": 8,
        },
        prediction_payload={
            "binary_prediction": 1,
            "binary_probability": 0.73,
            "multiclass_prediction": "<30",
            "multiclass_probabilities": {"NO": 0.12, ">30": 0.15, "<30": 0.73},
        },
        top_n_factors=3,
    )

    assert explanation.explanation_mode == "fallback"
    assert explanation.top_risk_increasing_factors
    assert explanation.top_risk_decreasing_factors
    assert "not medical advice" in explanation.explanation_text.lower()


def test_frontend_paths_resolve_without_full_stack_imports() -> None:
    paths = resolve_frontend_paths()
    missing = missing_prediction_artifacts(paths)

    assert paths.project_root.exists()
    assert isinstance(missing, list)
