from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

from src.frontend.loaders import (
    build_feature_defaults,
    load_frontend_context,
    load_prediction_artifacts,
    missing_prediction_artifacts,
)
from src.frontend.prediction_engine import (
    FrontendPrediction,
    PredictionWithExplanation,
    predict_with_deterministic_explanation,
)


def test_streamlit_frontend_artifacts_are_loadable() -> None:
    context = load_frontend_context()
    missing = missing_prediction_artifacts(context.paths)
    assert not missing

    artifacts = load_prediction_artifacts(context.paths)
    assert artifacts.binary_feature_columns
    assert artifacts.multiclass_feature_columns


def test_streamlit_sample_payload_is_compatible_with_feature_columns() -> None:
    context = load_frontend_context()
    artifacts = load_prediction_artifacts(context.paths)

    defaults = build_feature_defaults(
        feature_columns=artifacts.binary_feature_columns,
        base_features=context.baseline_features,
    )

    assert set(defaults.keys()) == set(artifacts.binary_feature_columns)


def test_streamlit_prediction_helper_output_structure() -> None:
    context = load_frontend_context()
    artifacts = load_prediction_artifacts(context.paths)

    features = build_feature_defaults(
        feature_columns=artifacts.binary_feature_columns,
        base_features=context.dummy_features or context.baseline_features,
    )

    output = predict_with_deterministic_explanation(features=features, artifacts=artifacts)

    assert isinstance(output, PredictionWithExplanation)
    assert isinstance(output.prediction, FrontendPrediction)
    assert output.prediction.binary_prediction in {0, 1}
    assert 0.0 <= output.prediction.binary_probability <= 1.0
    assert output.prediction.multiclass_probabilities
    assert output.explanation.explanation_mode == "fallback"
    assert "not medical advice" in output.explanation.explanation_text


def test_streamlit_app_module_imports_safely() -> None:
    project_root = Path(__file__).resolve().parents[1]
    streamlit_entrypoint = project_root / "streamlit_app.py"
    spec = importlib.util.spec_from_file_location("streamlit_app", streamlit_entrypoint)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "main")
