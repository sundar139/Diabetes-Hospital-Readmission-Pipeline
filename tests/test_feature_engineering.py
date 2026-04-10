from __future__ import annotations

import pandas as pd
from src.features.build_features import (
    ENGINEERED_FEATURE_NAMES,
    build_age_bucket_risk_feature,
    build_utilization_intensity_feature,
    detect_medication_status_columns,
    engineer_clinical_features,
)


def _synthetic_frame() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "encounter_id": [1, 2, 3, 4],
            "patient_nbr": [100, 100, 101, 102],
            "readmitted": ["NO", "<30", ">30", "NO"],
            "readmitted_30d": [0, 1, 0, 0],
            "time_in_hospital": [2, 6, 3, 5],
            "num_diagnoses": [4, 9, 6, 7],
            "number_inpatient": [0, 2, 1, 0],
            "number_outpatient": [1, 2, 0, 0],
            "number_emergency": [0, 1, 0, 2],
            "discharge_disposition_id": [1, 2, 6, 99],
            "age": ["[30-40)", "[70-80)", "[80-90)", "bad"],
            "metformin": ["No", "Up", "Steady", pd.NA],
            "insulin": ["No", "Down", "No", pd.NA],
            "num_medications": [8, 15, 11, 9],
        }
    )
    frame.index = pd.Index([10, 11, 12, 13])
    return frame


def test_each_engineered_feature_is_created() -> None:
    frame = _synthetic_frame()
    result = engineer_clinical_features(frame)

    for feature_name in ENGINEERED_FEATURE_NAMES:
        assert feature_name in result.frame.columns


def test_row_count_and_index_are_preserved() -> None:
    frame = _synthetic_frame()
    result = engineer_clinical_features(frame)

    assert len(result.frame) == len(frame)
    assert result.frame.index.equals(frame.index)


def test_medication_change_ratio_handles_zero_denominator_safely() -> None:
    frame = _synthetic_frame()
    frame["metformin"] = pd.NA
    frame["insulin"] = pd.NA

    result = engineer_clinical_features(frame, medication_status_columns=("metformin", "insulin"))

    assert (result.frame["medication_change_ratio"] == 0.0).all()
    assert result.fallback_counts["medication_change_ratio_zero_denominator"] == len(frame)


def test_age_bucket_mapping_and_fallback_work() -> None:
    frame = _synthetic_frame()
    risk_feature, fallback_count = build_age_bucket_risk_feature(frame)

    assert risk_feature.tolist() == [3, 7, 8, -1]
    assert fallback_count == 1


def test_utilization_intensity_is_computed_correctly() -> None:
    frame = _synthetic_frame()
    utilization = build_utilization_intensity_feature(frame)

    assert utilization.tolist() == [1.0, 5.0, 1.0, 2.0]


def test_patient_severity_supports_number_diagnoses_alias() -> None:
    frame = _synthetic_frame().drop(columns=["num_diagnoses"])
    frame["number_diagnoses"] = [4, 9, 6, 7]

    result = engineer_clinical_features(frame)

    assert "patient_severity" in result.frame.columns
    assert "number_diagnoses" in result.bookkeeping.feature_source_map["patient_severity"]


def test_identifier_exclusion_metadata_remains_correct() -> None:
    frame = _synthetic_frame()
    result = engineer_clinical_features(frame)

    assert "encounter_id" in result.bookkeeping.excluded_feature_columns
    assert "patient_nbr" in result.bookkeeping.excluded_feature_columns
    assert "encounter_id" not in result.bookkeeping.model_candidate_columns
    assert "patient_nbr" not in result.bookkeeping.model_candidate_columns


def test_output_schema_contains_required_engineered_columns() -> None:
    frame = _synthetic_frame()
    medication_columns = detect_medication_status_columns(frame)
    result = engineer_clinical_features(frame, medication_status_columns=medication_columns)

    required_columns = {
        "readmitted",
        "readmitted_30d",
        "recurrency",
        "patient_severity",
        "medication_change_ratio",
        "utilization_intensity",
        "complex_discharge_flag",
        "age_bucket_risk",
    }
    assert required_columns.issubset(result.frame.columns)
