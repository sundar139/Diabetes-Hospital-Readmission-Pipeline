from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from src.config.settings import Settings
from src.monitoring import drift_monitor


def test_compute_psi_detects_shift() -> None:
    rng = np.random.default_rng(123)
    reference = rng.normal(loc=0.0, scale=1.0, size=500)
    shifted = rng.normal(loc=1.8, scale=1.0, size=500)

    psi_value = drift_monitor.compute_psi(reference.tolist(), shifted.tolist(), bins=10)

    assert psi_value is not None
    assert psi_value > 0.10


def test_build_monitoring_summary_warns_for_small_samples_and_missing_labels() -> None:
    reference_frame = pd.DataFrame({"patient_severity": [0.1, 0.2, 0.3]})
    current_frame = pd.DataFrame({"patient_severity": [0.2, 0.25, 0.4]})
    records = [
        {
            "binary_prediction": 0,
            "binary_probability": 0.21,
            "multiclass_prediction": "NO",
        },
        {
            "binary_prediction": 1,
            "binary_probability": 0.77,
            "multiclass_prediction": "<30",
        },
    ]

    summary = drift_monitor.build_monitoring_summary(
        model_version_info={"binary_model": {}, "multiclass_model": {}},
        current_records=records,
        reference_frame=reference_frame,
        current_feature_frame=current_frame,
        numeric_feature_columns=("patient_severity",),
        reference_binary_probabilities=[0.1, 0.4, 0.2],
        min_sample_size=5,
        psi_bins=5,
    )

    assert summary["label_monitoring"]["labels_available"] is False
    assert summary["label_monitoring"]["performance_monitoring"] == "unavailable"
    assert any("insufficient" in warning.lower() for warning in summary["warnings"])


def test_prediction_log_jsonl_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "prediction_log.jsonl"
    records = [
        {
            "timestamp_utc": "2026-04-09T00:00:00+00:00",
            "selected_inputs": {"patient_severity": 0.22},
            "binary_prediction": 1,
            "binary_probability": 0.74,
            "multiclass_prediction": "<30",
            "multiclass_probabilities": {"NO": 0.12, ">30": 0.14, "<30": 0.74},
            "model_version": {},
        }
    ]

    drift_monitor.write_prediction_records_jsonl(records=records, output_path=path, append=False)
    loaded = drift_monitor.load_prediction_records_jsonl(path)

    assert len(loaded) == 1
    assert loaded[0]["binary_prediction"] == 1
    assert loaded[0]["selected_inputs"]["patient_severity"] == 0.22


def test_monitoring_narrative_fallback_mode() -> None:
    summary = {
        "sample_sizes": {"prediction_records": 10, "reference_rows": 20},
        "binary_probability_drift": {"status": "stable"},
        "warnings": [],
    }

    text, mode, warnings = drift_monitor.generate_monitoring_narrative(
        summary=summary,
        settings=Settings(_env_file=None),
        prefer_ollama=False,
    )

    assert mode == "fallback"
    assert warnings == []
    assert "Monitoring summary generated" in text


def test_build_prediction_records_include_inference_runtime(monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "time_in_hospital": [4, 6],
            "num_medications": [10, 12],
        }
    )

    def fake_predict_from_frame(*, task_type: str, **_: object) -> SimpleNamespace:
        if task_type == "binary":
            return SimpleNamespace(
                predictions=["1", "0"],
                probabilities_by_class={"0": [0.25, 0.75], "1": [0.75, 0.25]},
                positive_class_probability=[0.75, 0.25],
                inference_runtime={
                    "xgboost_device_used_for_inference": "cpu",
                    "inference_used_fallback_path": True,
                },
            )

        return SimpleNamespace(
            predictions=["<30", "NO"],
            probabilities_by_class={"NO": [0.2, 0.8], ">30": [0.1, 0.1], "<30": [0.7, 0.1]},
            positive_class_probability=None,
            inference_runtime={
                "xgboost_device_used_for_inference": "cpu",
                "inference_used_fallback_path": True,
            },
        )

    monkeypatch.setattr(drift_monitor, "predict_from_frame", fake_predict_from_frame)

    records = drift_monitor.build_prediction_records(
        frame=frame,
        binary_model=object(),
        multiclass_model=object(),
        binary_metadata={"feature_columns": ["time_in_hospital", "num_medications"]},
        multiclass_metadata={"feature_columns": ["time_in_hospital", "num_medications"]},
        model_version_info={
            "binary_model": {
                "xgboost_device_requested": "cuda",
                "xgboost_device_used_for_training": "cuda",
                "xgboost_device_used_for_inference": "cpu",
                "xgboost_inference_used_fallback_path": True,
            },
            "multiclass_model": {
                "xgboost_device_requested": "cuda",
                "xgboost_device_used_for_training": "cuda",
                "xgboost_device_used_for_inference": "cpu",
                "xgboost_inference_used_fallback_path": True,
            },
        },
    )

    assert records
    assert records[0]["inference_runtime"]["binary"]["inference_used_fallback_path"] is True

    runtime = drift_monitor.summarize_inference_runtime(
        model_version_info={
            "binary_model": {
                "xgboost_device_requested": "cuda",
                "xgboost_device_used_for_training": "cuda",
                "xgboost_device_used_for_inference": "cpu",
                "xgboost_inference_used_fallback_path": True,
            },
            "multiclass_model": {
                "xgboost_device_requested": "cuda",
                "xgboost_device_used_for_training": "cuda",
                "xgboost_device_used_for_inference": "cpu",
                "xgboost_inference_used_fallback_path": True,
            },
        },
        current_records=records,
    )

    assert runtime["binary_model"]["inference_used_fallback_path"] is True
    assert runtime["binary_model"]["record_runtime_device"] == "cpu"
