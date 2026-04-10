from __future__ import annotations

import json

from src.frontend.loaders import (
    build_feature_defaults,
    load_frontend_context,
    load_prediction_artifacts,
)
from src.frontend.prediction_engine import predict_with_deterministic_explanation


def main() -> int:
    context = load_frontend_context()
    artifacts = load_prediction_artifacts(context.paths)

    features = build_feature_defaults(
        feature_columns=artifacts.binary_feature_columns,
        base_features=context.dummy_features or context.baseline_features,
    )

    result = predict_with_deterministic_explanation(
        features=features,
        artifacts=artifacts,
        top_n_factors=3,
    )

    payload = {
        "binary_prediction": result.prediction.binary_prediction,
        "binary_probability": result.prediction.binary_probability,
        "multiclass_prediction": result.prediction.multiclass_prediction,
        "multiclass_probabilities": result.prediction.multiclass_probabilities,
        "explanation_mode": result.explanation.explanation_mode,
        "increasing_factors": result.explanation.top_risk_increasing_factors,
        "decreasing_factors": result.explanation.top_risk_decreasing_factors,
        "warnings": result.explanation.warnings,
    }

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
