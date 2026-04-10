from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

FeatureValue = str | int | float | bool | None


class SinglePredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: dict[str, FeatureValue] = Field(
        ...,
        min_length=1,
        description="Single-row feature payload keyed by model input column names.",
    )


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rows: list[dict[str, FeatureValue]] = Field(
        ...,
        min_length=1,
        description="List of feature dictionaries for batch prediction.",
    )


class SinglePredictionResponse(BaseModel):
    binary_prediction: int
    binary_probability: float
    multiclass_prediction: str
    multiclass_probabilities: dict[str, float]
    model_metadata_summary: dict[str, str | float | int | bool | None]


class BatchPredictionResponse(BaseModel):
    n_rows: int
    predictions: list[SinglePredictionResponse]


class ExplainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: dict[str, FeatureValue] = Field(
        ...,
        min_length=1,
        description="Single-row feature payload keyed by model input column names.",
    )
    prefer_ollama: bool = Field(
        default=True,
        description="When true, explanation generation attempts Ollama before fallback.",
    )
    top_n_factors: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of risk-increasing/decreasing factors to return.",
    )


class ExplainResponse(SinglePredictionResponse):
    prediction_summary: str
    top_risk_increasing_factors: list[str]
    top_risk_decreasing_factors: list[str]
    explanation_text: str
    explanation_mode: Literal["ollama", "fallback"]
    warnings: list[str] = Field(default_factory=list)
