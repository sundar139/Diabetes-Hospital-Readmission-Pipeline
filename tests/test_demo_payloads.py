from __future__ import annotations

import json
from pathlib import Path

from src.serving.schemas import (
    BatchPredictionRequest,
    ExplainRequest,
    SinglePredictionRequest,
)


def _read_payload(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_sample_payload_files_match_api_schema() -> None:
    artifacts_dir = Path(__file__).resolve().parents[1] / "artifacts"

    single_payload = _read_payload(artifacts_dir / "sample_payload.json")
    batch_payload = _read_payload(artifacts_dir / "sample_batch_payload.json")
    explain_payload = _read_payload(artifacts_dir / "sample_explain_payload.json")

    SinglePredictionRequest.model_validate(single_payload)
    BatchPredictionRequest.model_validate(batch_payload)
    ExplainRequest.model_validate(explain_payload)


def test_sample_payload_feature_keys_are_consistent_across_demo_requests() -> None:
    artifacts_dir = Path(__file__).resolve().parents[1] / "artifacts"

    single_payload = _read_payload(artifacts_dir / "sample_payload.json")
    batch_payload = _read_payload(artifacts_dir / "sample_batch_payload.json")
    explain_payload = _read_payload(artifacts_dir / "sample_explain_payload.json")

    single_features = single_payload["features"]
    explain_features = explain_payload["features"]
    batch_rows = batch_payload["rows"]

    assert isinstance(single_features, dict)
    assert isinstance(explain_features, dict)
    assert isinstance(batch_rows, list)
    assert batch_rows

    expected_keys = set(single_features.keys())
    assert set(explain_features.keys()) == expected_keys

    for row in batch_rows:
        assert isinstance(row, dict)
        assert set(row.keys()) == expected_keys
