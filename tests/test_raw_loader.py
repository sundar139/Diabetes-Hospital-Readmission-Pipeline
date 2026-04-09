from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from src.data.load_raw import detect_column_roles, load_raw_data


def test_load_raw_data_missing_file_raises_clear_error(tmp_path: Path) -> None:
    missing_path = tmp_path / "not_found.csv"
    with pytest.raises(FileNotFoundError, match="Raw dataset file not found"):
        load_raw_data(csv_path=missing_path)


def test_load_raw_data_preserves_original_column_names(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "encounter_id": [1, 2],
            "patient_nbr": [11, 12],
            "readmitted": ["NO", "<30"],
            "Medication Flag": ["No", "Up"],
        }
    )
    csv_path = tmp_path / "raw.csv"
    frame.to_csv(csv_path, index=False)

    loaded = load_raw_data(csv_path=csv_path)

    assert list(loaded.columns) == list(frame.columns)


def test_detect_column_roles_infers_expected_roles() -> None:
    frame = pd.DataFrame(
        {
            "encounter_id": [1, 2],
            "readmitted": ["NO", "<30"],
            "num_medications": [5, 6],
            "diag_1": ["250.83", "401"],
            "metformin": ["No", "Up"],
            "race": ["Caucasian", "AfricanAmerican"],
        }
    )

    roles = detect_column_roles(frame)

    assert roles["encounter_id"] == "identifier"
    assert roles["readmitted"] == "target"
    assert roles["num_medications"] == "numeric"
    assert roles["diag_1"] == "diagnosis-like"
    assert roles["metformin"] == "medication-status"
    assert roles["race"] == "categorical"
