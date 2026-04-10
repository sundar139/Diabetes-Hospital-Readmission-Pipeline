from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from src.models.train import train_task


def _build_split_frame(*, seed: int, start_encounter: int, n_rows: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    labels = rng.choice(["NO", ">30", "<30"], size=n_rows, p=[0.55, 0.25, 0.20])

    frame = pd.DataFrame(
        {
            "encounter_id": np.arange(start_encounter, start_encounter + n_rows),
            "patient_nbr": rng.integers(1000, 2000, size=n_rows),
            "readmitted": labels,
            "readmitted_30d": (labels == "<30").astype(int),
            "num_feature": rng.normal(loc=0.0, scale=1.0, size=n_rows),
            "util_feature": rng.integers(0, 12, size=n_rows),
            "cat_feature": rng.choice(["A", "B", "C", None], size=n_rows),
        }
    )
    return frame


def _prepare_feature_splits(tmp_path: Path) -> tuple[Path, Path, Path]:
    processed_dir = tmp_path / "data" / "processed"
    artifacts_dir = tmp_path / "artifacts"
    processed_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    train = _build_split_frame(seed=7, start_encounter=10_000)
    val = _build_split_frame(seed=11, start_encounter=20_000)
    test = _build_split_frame(seed=13, start_encounter=30_000)

    for label in ("NO", ">30", "<30"):
        if label not in train["readmitted"].values:
            train.loc[0, "readmitted"] = label
            train.loc[0, "readmitted_30d"] = int(label == "<30")

    train.to_parquet(processed_dir / "train_features.parquet", index=False)
    val.to_parquet(processed_dir / "val_features.parquet", index=False)
    test.to_parquet(processed_dir / "test_features.parquet", index=False)

    feature_metadata = {
        "engineered_feature_names": [],
        "target_columns": ["readmitted", "readmitted_30d"],
        "identifier_columns": ["encounter_id", "patient_nbr"],
        "excluded_feature_columns": ["encounter_id", "patient_nbr"],
        "model_candidate_columns": [
            "encounter_id",
            "patient_nbr",
            "num_feature",
            "util_feature",
            "cat_feature",
        ],
    }
    feature_metadata_path = artifacts_dir / "feature_metadata.json"
    feature_metadata_path.write_text(json.dumps(feature_metadata, indent=2), encoding="utf-8")

    return processed_dir, feature_metadata_path, artifacts_dir


def test_binary_training_produces_fitted_artifact_and_excludes_identifiers(tmp_path: Path) -> None:
    processed_dir, metadata_path, artifacts_dir = _prepare_feature_splits(tmp_path)

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

    assert result.best_model_path.exists()
    assert result.best_metadata_path.exists()

    metadata = json.loads(result.best_metadata_path.read_text(encoding="utf-8"))
    assert metadata["feature_selection_strategy"] == "none"
    assert "encounter_id" not in metadata["feature_columns"]
    assert "patient_nbr" not in metadata["feature_columns"]


def test_multiclass_training_produces_expected_metadata(tmp_path: Path) -> None:
    processed_dir, metadata_path, artifacts_dir = _prepare_feature_splits(tmp_path)

    result = train_task(
        task_type="multiclass",
        model_families=("logistic_regression",),
        sampling_strategies=("none",),
        feature_selection_strategy="none",
        enable_mlflow=False,
        processed_dir=processed_dir,
        feature_metadata_path=metadata_path,
        artifacts_dir=artifacts_dir,
    )

    assert result.best_model_path.exists()
    metadata = json.loads(result.best_metadata_path.read_text(encoding="utf-8"))

    assert metadata["target_column"] == "readmitted"
    assert metadata["class_labels"] == ["NO", ">30", "<30"]
    assert metadata["feature_selection_strategy"] == "none"
