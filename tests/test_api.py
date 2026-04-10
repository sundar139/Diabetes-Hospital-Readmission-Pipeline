from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from src.serving.api import PredictionServingService, create_app
from src.serving.schemas import ExplainResponse, SinglePredictionResponse


class StubServingService:
    def __init__(self) -> None:
        self.initialized = False

    def initialize(self) -> None:
        self.initialized = True

    def health_payload(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "models_loaded": True,
            "load_error": None,
            "required_artifacts": {},
            "ollama_host": "http://localhost:11434",
            "ollama_model": "llama3.1:8b",
        }

    def _single_prediction(self) -> SinglePredictionResponse:
        return SinglePredictionResponse(
            binary_prediction=1,
            binary_probability=0.83,
            multiclass_prediction="<30",
            multiclass_probabilities={"NO": 0.08, ">30": 0.09, "<30": 0.83},
            model_metadata_summary={
                "binary_model_family": "xgboost",
                "multiclass_model_family": "xgboost",
                "binary_training_timestamp_utc": "2026-04-10T00:00:00+00:00",
                "multiclass_training_timestamp_utc": "2026-04-10T00:00:00+00:00",
            },
        )

    def predict_single(self, features: Mapping[str, Any]) -> SinglePredictionResponse:
        if not features:
            raise ValueError("features must not be empty")
        return self._single_prediction()

    def predict_batch(self, rows: list[Mapping[str, Any]]) -> list[SinglePredictionResponse]:
        if not rows:
            raise ValueError("rows must not be empty")
        return [self._single_prediction() for _ in rows]

    async def explain_single(
        self,
        *,
        features: Mapping[str, Any],
        prefer_ollama: bool,
        top_n_factors: int,
    ) -> ExplainResponse:
        if not features:
            raise ValueError("features must not be empty")
        prediction = self._single_prediction()
        mode = "ollama" if prefer_ollama else "fallback"

        return ExplainResponse(
            **prediction.model_dump(),
            prediction_summary="Binary prediction=1 with elevated probability.",
            top_risk_increasing_factors=["recurrency (+0.6)"][:top_n_factors],
            top_risk_decreasing_factors=["complex_discharge_flag (-0.2)"][:top_n_factors],
            explanation_text="Model-generated explanation for local testing only.",
            explanation_mode=mode,
            warnings=[],
        )


def test_health_endpoint_returns_ok_status() -> None:
    app = create_app(service=StubServingService())

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["models_loaded"] is True


def test_root_route_redirects_to_docs() -> None:
    app = create_app(service=StubServingService())

    with TestClient(app) as client:
        response = client.get("/", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == "/docs"


def test_predict_endpoint_returns_single_prediction() -> None:
    app = create_app(service=StubServingService())

    with TestClient(app) as client:
        response = client.post("/predict", json={"features": {"race": "Caucasian"}})

    assert response.status_code == 200
    payload = response.json()
    assert payload["binary_prediction"] == 1
    assert payload["multiclass_prediction"] == "<30"


def test_predict_batch_endpoint_returns_n_rows() -> None:
    app = create_app(service=StubServingService())

    with TestClient(app) as client:
        response = client.post(
            "/predict-batch",
            json={
                "rows": [
                    {"race": "Caucasian", "gender": "Male"},
                    {"race": "AfricanAmerican", "gender": "Female"},
                ]
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["n_rows"] == 2
    assert len(payload["predictions"]) == 2


def test_explain_endpoint_returns_explanation() -> None:
    app = create_app(service=StubServingService())

    with TestClient(app) as client:
        response = client.post(
            "/explain",
            json={
                "features": {"recurrency": 2},
                "prefer_ollama": False,
                "top_n_factors": 1,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["explanation_mode"] == "fallback"
    assert len(payload["top_risk_increasing_factors"]) == 1


def test_predict_validation_error_for_empty_features() -> None:
    app = create_app(service=StubServingService())

    with TestClient(app) as client:
        response = client.post("/predict", json={"features": {}})

    assert response.status_code == 422


def test_service_health_degraded_when_artifacts_missing(tmp_path: Path) -> None:
    service = PredictionServingService(artifacts_dir=tmp_path / "missing_artifacts")
    service.initialize()

    payload = service.health_payload()
    assert payload["status"] == "degraded"
    assert payload["models_loaded"] is False
    assert payload["load_error"]


def test_predict_returns_503_when_service_is_not_ready(tmp_path: Path) -> None:
    service = PredictionServingService(artifacts_dir=tmp_path / "missing_artifacts")
    app = create_app(service=service)

    with TestClient(app) as client:
        response = client.post("/predict", json={"features": {"race": "Caucasian"}})

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "Serving artifacts are missing" in detail
