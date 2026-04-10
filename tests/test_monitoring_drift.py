from __future__ import annotations

from pathlib import Path

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
