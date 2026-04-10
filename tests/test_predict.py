from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from src.models.predict import load_model, predict_with_artifacts
from src.models.train import train_task


def _build_split_frame(*, seed: int, start_encounter: int, n_rows: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    labels = rng.choice(["NO", ">30", "<30"], size=n_rows, p=[0.55, 0.25, 0.20])

    frame = pd.DataFrame(
        {
            "encounter_id": np.arange(start_encounter, start_encounter + n_rows),
            "patient_nbr": rng.integers(1000, 3000, size=n_rows),
            "readmitted": labels,
            "readmitted_30d": (labels == "<30").astype(int),
            "num_feature": rng.normal(size=n_rows),
            "cat_feature": rng.choice(["X", "Y", "Z"], size=n_rows),
        }
    )
    return frame


def _prepare_training_artifacts(tmp_path: Path) -> tuple[pd.DataFrame, Path, Path, Path]:
    processed_dir = tmp_path / "data" / "processed"
    artifacts_dir = tmp_path / "artifacts"
    processed_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    train = _build_split_frame(seed=5, start_encounter=1000)
    val = _build_split_frame(seed=6, start_encounter=2000)
    test = _build_split_frame(seed=9, start_encounter=3000)

    train.to_parquet(processed_dir / "train_features.parquet", index=False)
    val.to_parquet(processed_dir / "val_features.parquet", index=False)
    test.to_parquet(processed_dir / "test_features.parquet", index=False)

    metadata = {
        "target_columns": ["readmitted", "readmitted_30d"],
        "identifier_columns": ["encounter_id", "patient_nbr"],
        "excluded_feature_columns": ["encounter_id", "patient_nbr"],
        "model_candidate_columns": [
            "num_feature",
            "cat_feature",
            "encounter_id",
            "patient_nbr",
        ],
    }
    metadata_path = artifacts_dir / "feature_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    result = train_task(
        task_type="binary",
        model_families=("logistic_regression",),
        sampling_strategies=("none",),
        feature_selection_strategy="none",
        enable_mlflow=False,
        processed_dir=processed_dir,
        feature_metadata_path=metadata_path,
        artifacts_dir=artifacts_dir,
    )

    return test, result.best_model_path, result.best_metadata_path, artifacts_dir


def test_saved_model_loads_and_predicts(tmp_path: Path) -> None:
    test_frame, model_path, metadata_path, _ = _prepare_training_artifacts(tmp_path)

    loaded_model = load_model(model_path)
    assert loaded_model is not None

    inference_frame = test_frame.head(8).copy()
    prediction_result = predict_with_artifacts(
        frame=inference_frame,
        model_path=model_path,
        metadata_path=metadata_path,
    )

    assert len(prediction_result.predictions) == 8
    assert prediction_result.positive_class_probability is not None
    assert len(prediction_result.positive_class_probability) == 8
    assert prediction_result.probabilities_by_class
