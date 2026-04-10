from __future__ import annotations

import pandas as pd
from src.data.split import (
    GroupedSplitResult,
    assert_no_group_overlap,
    build_split_manifest,
    grouped_split_by_patient,
)


def _synthetic_frame(n_patients: int = 60, visits_per_patient: int = 2) -> pd.DataFrame:
    rows: list[dict[str, int | str]] = []
    for patient_idx in range(n_patients):
        patient_id = 1000 + patient_idx
        for visit_idx in range(visits_per_patient):
            rows.append(
                {
                    "encounter_id": patient_idx * 10 + visit_idx,
                    "patient_nbr": patient_id,
                    "readmitted": "<30" if (patient_idx + visit_idx) % 5 == 0 else "NO",
                    "readmitted_30d": 1 if (patient_idx + visit_idx) % 5 == 0 else 0,
                    "num_medications": 5 + visit_idx,
                }
            )
    return pd.DataFrame(rows)


def test_grouped_split_has_no_patient_overlap() -> None:
    frame = _synthetic_frame()

    split_result = grouped_split_by_patient(frame, random_state=42)

    assert_no_group_overlap(
        split_result.train,
        split_result.val,
        split_result.test,
        group_column="patient_nbr",
    )


def test_grouped_split_has_reasonable_row_proportions() -> None:
    frame = _synthetic_frame(n_patients=80, visits_per_patient=3)
    split_result = grouped_split_by_patient(frame, random_state=7)

    total_rows = len(frame)
    train_ratio = len(split_result.train) / total_rows
    val_ratio = len(split_result.val) / total_rows
    test_ratio = len(split_result.test) / total_rows

    assert 0.60 <= train_ratio <= 0.80
    assert 0.08 <= val_ratio <= 0.25
    assert 0.08 <= test_ratio <= 0.25


def test_processed_split_contains_required_targets_and_manifest_keys() -> None:
    frame = _synthetic_frame()
    split_result = grouped_split_by_patient(frame, random_state=123)

    for split_frame in (split_result.train, split_result.val, split_result.test):
        assert "readmitted" in split_frame.columns
        assert "readmitted_30d" in split_frame.columns

    manifest = build_split_manifest(
        split_result,
        excluded_feature_columns=["encounter_id", "patient_nbr"],
        random_state=123,
    )

    expected_keys = {
        "n_rows_total",
        "n_rows_train",
        "n_rows_val",
        "n_rows_test",
        "n_patients_total",
        "n_patients_train",
        "n_patients_val",
        "n_patients_test",
        "readmitted_distribution_by_split",
        "readmitted_30d_distribution_by_split",
        "leakage_check_passed",
        "excluded_feature_columns",
        "random_state",
    }
    assert expected_keys.issubset(manifest.keys())


def test_manifest_builder_accepts_grouped_split_result() -> None:
    frame = _synthetic_frame(n_patients=30, visits_per_patient=2)
    split_result = grouped_split_by_patient(frame, random_state=17)

    manifest = build_split_manifest(
        GroupedSplitResult(split_result.train, split_result.val, split_result.test),
        excluded_feature_columns=["encounter_id", "patient_nbr"],
        random_state=17,
    )

    assert manifest["leakage_check_passed"] is True
    assert manifest["n_rows_total"] == len(frame)
