from __future__ import annotations

import pandas as pd
from src.data.validate_raw import build_raw_validation_summary


def _base_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "encounter_id": [1, 2, 3],
            "patient_nbr": [10, 10, 11],
            "readmitted": ["NO", ">30", "<30"],
            "age": ["[50-60)", "[60-70)", "[70-80)"],
            "race": ["Caucasian", "AfricanAmerican", "Asian"],
            "gender": ["Female", "Male", "Female"],
            "time_in_hospital": [2, 5, 7],
            "num_lab_procedures": [40, 55, 65],
            "num_medications": [12, 15, 18],
            "number_outpatient": [0, 1, 2],
            "number_emergency": [0, 0, 1],
            "number_inpatient": [0, 1, 1],
        }
    )


def test_missing_required_columns_are_flagged() -> None:
    frame = _base_frame().drop(columns=["age"])

    summary = build_raw_validation_summary(frame)

    assert summary["required_columns_present"]["age"] is False
    assert any("Missing required columns" in warning for warning in summary["warnings"])


def test_target_validation_reports_invalid_labels() -> None:
    frame = _base_frame()
    frame.loc[0, "readmitted"] = "INVALID"

    summary = build_raw_validation_summary(frame)

    assert "INVALID" in summary["target_distribution"]
    assert any("Unexpected target values" in warning for warning in summary["warnings"])


def test_summary_contains_expected_keys_and_null_like_counts() -> None:
    frame = _base_frame()
    frame.loc[1, "race"] = "?"
    frame.loc[2, "gender"] = "   "

    summary = build_raw_validation_summary(frame)

    expected_keys = {
        "n_rows",
        "n_columns",
        "required_columns_present",
        "duplicate_rows",
        "encounter_id_unique",
        "patient_nbr_unique_count",
        "target_distribution",
        "null_like_counts",
        "numeric_columns",
        "categorical_columns",
        "warnings",
    }
    assert expected_keys.issubset(summary.keys())
    assert summary["null_like_counts"]["question_mark"] >= 1

    missingness_by_column = {
        item["column"]: item for item in summary["missingness_summary"]
    }
    assert missingness_by_column["race"]["missing_count"] >= 1
