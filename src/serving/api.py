from __future__ import annotations

from collections.abc import Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse

from src.config.settings import Settings, get_settings
from src.llm.explain import generate_explanation
from src.models.predict import (
    load_model,
    load_model_metadata,
    predict_from_frame,
)
from src.serving.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ExplainRequest,
    ExplainResponse,
    SinglePredictionRequest,
    SinglePredictionResponse,
)


@dataclass(frozen=True)
class LoadedModelArtifacts:
    binary_model: Any
    multiclass_model: Any
    binary_metadata: dict[str, Any]
    multiclass_metadata: dict[str, Any]


class PredictionServingService:
    def __init__(
        self,
        *,
        settings: Settings | None = None,
        artifacts_dir: Path | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.artifacts_dir = artifacts_dir or self.settings.artifacts_dir_path
        self._artifacts: LoadedModelArtifacts | None = None
        self._load_error: str | None = None

    def initialize(self) -> None:
        try:
            self._artifacts = self._load_artifacts()
            self._load_error = None
        except Exception as exc:
            self._artifacts = None
            self._load_error = str(exc)

    @property
    def is_ready(self) -> bool:
        return self._artifacts is not None

    def _required_artifact_paths(self) -> dict[str, Path]:
        return {
            "binary_model": self.artifacts_dir / "binary_model.joblib",
            "multiclass_model": self.artifacts_dir / "multiclass_model.joblib",
            "binary_metadata": self.artifacts_dir / "binary_model_metadata.json",
            "multiclass_metadata": self.artifacts_dir / "multiclass_model_metadata.json",
        }

    def _load_artifacts(self) -> LoadedModelArtifacts:
        paths = self._required_artifact_paths()
        missing = [name for name, path in paths.items() if not path.exists()]
        if missing:
            missing_names = ", ".join(missing)
            raise FileNotFoundError(
                f"Serving artifacts are missing: {missing_names}. Run training scripts first."
            )

        binary_model = load_model(paths["binary_model"])
        multiclass_model = load_model(paths["multiclass_model"])
        binary_metadata = load_model_metadata(paths["binary_metadata"])
        multiclass_metadata = load_model_metadata(paths["multiclass_metadata"])

        binary_features = binary_metadata.get("feature_columns", [])
        multiclass_features = multiclass_metadata.get("feature_columns", [])
        if not binary_features or not multiclass_features:
            raise ValueError("Model metadata does not contain feature_columns.")

        return LoadedModelArtifacts(
            binary_model=binary_model,
            multiclass_model=multiclass_model,
            binary_metadata=binary_metadata,
            multiclass_metadata=multiclass_metadata,
        )

    def _ensure_ready(self) -> LoadedModelArtifacts:
        if self._artifacts is None:
            detail = self._load_error or "Model artifacts are not initialized."
            raise RuntimeError(detail)
        return self._artifacts

    def health_payload(self) -> dict[str, Any]:
        paths = self._required_artifact_paths()
        return {
            "status": "ok" if self.is_ready else "degraded",
            "models_loaded": self.is_ready,
            "load_error": self._load_error,
            "required_artifacts": {key: str(path) for key, path in paths.items()},
            "ollama_host": self.settings.ollama_host,
            "ollama_model": self.settings.ollama_model,
        }

    @staticmethod
    def _parse_label_decoder(metadata: Mapping[str, Any]) -> dict[int, str] | None:
        mapping_raw = metadata.get("label_mapping", {})
        if not isinstance(mapping_raw, Mapping) or not mapping_raw:
            return None

        parsed: dict[int, str] = {}
        for key, value in mapping_raw.items():
            parsed[int(key)] = str(value)
        return parsed

    @staticmethod
    def _binary_prediction_to_int(raw_prediction: str) -> int:
        normalized = raw_prediction.strip().lower()
        if normalized in {"1", "true", "yes", "<30"}:
            return 1
        return 0

    @staticmethod
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

    def _model_metadata_summary(
        self,
        artifacts: LoadedModelArtifacts,
    ) -> dict[str, str | float | int | None]:
        return {
            "binary_model_family": str(artifacts.binary_metadata.get("model_family")),
            "multiclass_model_family": str(artifacts.multiclass_metadata.get("model_family")),
            "binary_training_timestamp_utc": artifacts.binary_metadata.get(
                "training_timestamp_utc"
            ),
            "multiclass_training_timestamp_utc": artifacts.multiclass_metadata.get(
                "training_timestamp_utc"
            ),
        }

    def _predict_rows(self, frame: pd.DataFrame) -> list[SinglePredictionResponse]:
        artifacts = self._ensure_ready()

        binary_feature_columns = [
            str(column) for column in artifacts.binary_metadata.get("feature_columns", [])
        ]
        multiclass_feature_columns = [
            str(column) for column in artifacts.multiclass_metadata.get("feature_columns", [])
        ]

        binary_decoder = self._parse_label_decoder(artifacts.binary_metadata)
        multiclass_decoder = self._parse_label_decoder(artifacts.multiclass_metadata)

        binary_output = predict_from_frame(
            model=artifacts.binary_model,
            frame=frame,
            feature_columns=binary_feature_columns,
            task_type="binary",
            label_decoder=binary_decoder,
        )
        multiclass_output = predict_from_frame(
            model=artifacts.multiclass_model,
            frame=frame,
            feature_columns=multiclass_feature_columns,
            task_type="multiclass",
            label_decoder=multiclass_decoder,
        )

        metadata_summary = self._model_metadata_summary(artifacts)

        responses: list[SinglePredictionResponse] = []
        for idx in range(len(frame)):
            raw_binary_prediction = binary_output.predictions[idx]
            binary_prediction = self._binary_prediction_to_int(raw_binary_prediction)

            if binary_output.positive_class_probability is not None:
                binary_probability = float(binary_output.positive_class_probability[idx])
            else:
                binary_probability = self._first_probability_or_default(
                    binary_output.probabilities_by_class,
                    idx,
                    positive_labels=("1", "<30"),
                )

            multiclass_probabilities = {
                label: float(values[idx])
                for label, values in multiclass_output.probabilities_by_class.items()
                if idx < len(values)
            }

            responses.append(
                SinglePredictionResponse(
                    binary_prediction=binary_prediction,
                    binary_probability=binary_probability,
                    multiclass_prediction=multiclass_output.predictions[idx],
                    multiclass_probabilities=multiclass_probabilities,
                    model_metadata_summary=metadata_summary,
                )
            )

        return responses

    def predict_single(self, features: Mapping[str, Any]) -> SinglePredictionResponse:
        frame = pd.DataFrame([dict(features)])
        return self._predict_rows(frame)[0]

    def predict_batch(self, rows: list[Mapping[str, Any]]) -> list[SinglePredictionResponse]:
        frame = pd.DataFrame([dict(row) for row in rows])
        return self._predict_rows(frame)

    async def explain_single(
        self,
        *,
        features: Mapping[str, Any],
        prefer_ollama: bool,
        top_n_factors: int,
    ) -> ExplainResponse:
        artifacts = self._ensure_ready()
        prediction = self.predict_single(features)

        explanation = await generate_explanation(
            features=features,
            binary_model=artifacts.binary_model,
            prediction_payload=prediction.model_dump(),
            settings=self.settings,
            prefer_ollama=prefer_ollama,
            top_n_factors=top_n_factors,
        )

        return ExplainResponse(
            **prediction.model_dump(),
            prediction_summary=explanation.prediction_summary,
            top_risk_increasing_factors=explanation.top_risk_increasing_factors,
            top_risk_decreasing_factors=explanation.top_risk_decreasing_factors,
            explanation_text=explanation.explanation_text,
            explanation_mode=explanation.explanation_mode,
            warnings=explanation.warnings,
        )


def create_app(service: PredictionServingService | None = None) -> FastAPI:
    prediction_service = service or PredictionServingService()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.prediction_service.initialize()
        yield

    app = FastAPI(
        title="Diabetes Readmission Local API",
        description="Local-first prediction and explanation API for diabetic readmission models.",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.state.prediction_service = prediction_service

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return app.state.prediction_service.health_payload()

    @app.post("/predict", response_model=SinglePredictionResponse)
    async def predict(payload: SinglePredictionRequest) -> SinglePredictionResponse:
        try:
            return app.state.prediction_service.predict_single(payload.features)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/predict-batch", response_model=BatchPredictionResponse)
    async def predict_batch(payload: BatchPredictionRequest) -> BatchPredictionResponse:
        try:
            predictions = app.state.prediction_service.predict_batch(payload.rows)
            return BatchPredictionResponse(n_rows=len(predictions), predictions=predictions)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/explain", response_model=ExplainResponse)
    async def explain(payload: ExplainRequest) -> ExplainResponse:
        try:
            return await app.state.prediction_service.explain_single(
                features=payload.features,
                prefer_ollama=payload.prefer_ollama,
                top_n_factors=payload.top_n_factors,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app


app = create_app()
