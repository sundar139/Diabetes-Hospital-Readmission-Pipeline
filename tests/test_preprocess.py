from __future__ import annotations

import pandas as pd
import pytest
from src.data.preprocess import (
    build_feature_candidacy,
    build_preprocessing_summary,
    derive_binary_readmission_target,
    replace_null_like_tokens,
)


def test_replace_null_like_tokens_replaces_question_and_whitespace() -> None:
    frame = pd.DataFrame(
        {
            "race": ["Caucasian", "?", "  "],
            "gender": ["Female", "Male", ""],
        }
    )

    normalized = replace_null_like_tokens(frame)

    assert pd.isna(normalized.loc[1, "race"])
    assert pd.isna(normalized.loc[2, "race"])
    assert pd.isna(normalized.loc[2, "gender"])


def test_derive_binary_readmission_target_maps_expected_values() -> None:
    frame = pd.DataFrame(
        {
            "readmitted": ["NO", ">30", "<30", "NO"],
        }
    )

    transformed = derive_binary_readmission_target(frame)

    assert transformed["readmitted_30d"].tolist() == [0, 0, 1, 0]


def test_derive_binary_readmission_target_raises_for_unexpected_values() -> None:
    frame = pd.DataFrame(
        {
            "readmitted": ["NO", "INVALID"],
        }
    )

    with pytest.raises(ValueError, match="Unexpected labels"):
        derive_binary_readmission_target(frame)


def test_feature_candidacy_excludes_identifier_columns() -> None:
    frame = pd.DataFrame(
        {
            "encounter_id": [1, 2],
            "patient_nbr": [10, 11],
            "readmitted": ["NO", "<30"],
            "readmitted_30d": [0, 1],
            "num_medications": [5, 7],
        }
    )

    candidacy = build_feature_candidacy(frame)

    assert "encounter_id" in candidacy.excluded_feature_columns
    assert "patient_nbr" in candidacy.excluded_feature_columns
    assert "num_medications" in candidacy.candidate_feature_columns
    assert "encounter_id" not in candidacy.candidate_feature_columns


def test_preprocessing_summary_contains_required_target_columns() -> None:
    raw_frame = pd.DataFrame(
        {
            "encounter_id": [1, 2],
            "patient_nbr": [10, 11],
            "readmitted": ["NO", "<30"],
            "race": ["Caucasian", "?"],
        }
    )
    normalized = replace_null_like_tokens(raw_frame)
    preprocessed = derive_binary_readmission_target(normalized)
    candidacy = build_feature_candidacy(preprocessed)

    summary = build_preprocessing_summary(raw_frame, preprocessed, candidacy)

    assert "target_distribution_readmitted" in summary
    assert "target_distribution_readmitted_30d" in summary
    assert summary["shape_after_preprocessing"]["n_columns"] >= raw_frame.shape[1] + 1
