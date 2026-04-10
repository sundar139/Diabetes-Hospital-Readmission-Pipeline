from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

DEFAULT_TARGET_COLUMN = "readmitted"
DEFAULT_BINARY_TARGET_COLUMN = "readmitted_30d"
DEFAULT_ALLOWED_TARGET_LABELS: tuple[str, str, str] = ("NO", ">30", "<30")
DEFAULT_POSITIVE_TARGET_LABEL = "<30"
LEAKAGE_PRONE_IDENTIFIER_COLUMNS: tuple[str, str] = ("encounter_id", "patient_nbr")


@dataclass(frozen=True)
class FeatureCandidacy:
    identifier_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    excluded_feature_columns: tuple[str, ...]
    candidate_feature_columns: tuple[str, ...]


def _string_token_masks(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    question_mark_mask = pd.DataFrame(False, index=frame.index, columns=frame.columns)
    blank_or_whitespace_mask = pd.DataFrame(False, index=frame.index, columns=frame.columns)

    for column in frame.columns:
        text_series = frame[column].astype("string")
        non_null_mask = text_series.notna()
        stripped = text_series.str.strip()

        question_mark_mask[column] = non_null_mask & (stripped == "?")
        blank_or_whitespace_mask[column] = non_null_mask & (stripped == "")

    return question_mark_mask, blank_or_whitespace_mask


def replace_null_like_tokens(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy(deep=True)
    question_mark_mask, blank_or_whitespace_mask = _string_token_masks(normalized)
    replacement_mask = question_mark_mask | blank_or_whitespace_mask

    for column in normalized.columns:
        if bool(replacement_mask[column].any()):
            normalized[column] = normalized[column].mask(replacement_mask[column], pd.NA)

    return normalized


def validate_multiclass_target_labels(
    frame: pd.DataFrame,
    *,
    target_column: str = DEFAULT_TARGET_COLUMN,
    allowed_labels: tuple[str, str, str] = DEFAULT_ALLOWED_TARGET_LABELS,
) -> None:
    if target_column not in frame.columns:
        raise KeyError(f"Target column '{target_column}' is not present in the dataset.")

    target_values = frame[target_column].astype("string").str.strip()
    missing_target_count = int(target_values.isna().sum())
    if missing_target_count > 0:
        raise ValueError(
            f"Target column '{target_column}' contains {missing_target_count} missing value(s)."
        )

    observed_values = {str(value) for value in target_values.dropna().unique().tolist()}
    unexpected = sorted(observed_values - set(allowed_labels))
    if unexpected:
        raise ValueError(
            f"Unexpected labels found in '{target_column}': {', '.join(unexpected)}. "
            f"Expected labels: {', '.join(allowed_labels)}."
        )


def derive_binary_readmission_target(
    frame: pd.DataFrame,
    *,
    target_column: str = DEFAULT_TARGET_COLUMN,
    binary_target_column: str = DEFAULT_BINARY_TARGET_COLUMN,
    positive_label: str = DEFAULT_POSITIVE_TARGET_LABEL,
    allowed_labels: tuple[str, str, str] = DEFAULT_ALLOWED_TARGET_LABELS,
) -> pd.DataFrame:
    validate_multiclass_target_labels(
        frame,
        target_column=target_column,
        allowed_labels=allowed_labels,
    )

    mapped = frame[target_column].astype("string").str.strip().map(
        lambda value: 1 if str(value) == positive_label else 0
    )
    if bool(mapped.isna().any()):
        raise ValueError(
            f"Failed to derive '{binary_target_column}' due to unmapped target values."
        )

    derived = frame.copy(deep=True)
    derived[binary_target_column] = mapped.astype("int8")
    return derived


def build_feature_candidacy(
    frame: pd.DataFrame,
    *,
    identifier_columns: tuple[str, ...] = LEAKAGE_PRONE_IDENTIFIER_COLUMNS,
    target_columns: tuple[str, ...] = (DEFAULT_TARGET_COLUMN, DEFAULT_BINARY_TARGET_COLUMN),
    extra_excluded_columns: tuple[str, ...] = (),
) -> FeatureCandidacy:
    existing_identifiers = tuple(column for column in identifier_columns if column in frame.columns)
    existing_targets = tuple(column for column in target_columns if column in frame.columns)

    excluded = tuple(dict.fromkeys((*existing_identifiers, *extra_excluded_columns)))
    excluded_set = set(excluded)
    targets_set = set(existing_targets)

    candidate_features = tuple(
        column
        for column in frame.columns
        if column not in excluded_set and column not in targets_set
    )

    return FeatureCandidacy(
        identifier_columns=existing_identifiers,
        target_columns=existing_targets,
        excluded_feature_columns=excluded,
        candidate_feature_columns=candidate_features,
    )


def drop_leakage_prone_columns(
    frame: pd.DataFrame,
    excluded_columns: tuple[str, ...] | list[str],
) -> pd.DataFrame:
    to_drop = [column for column in excluded_columns if column in frame.columns]
    return frame.drop(columns=to_drop).copy()


def select_columns_for_modeling(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...] | list[str],
) -> pd.DataFrame:
    missing_columns = [column for column in feature_columns if column not in frame.columns]
    if missing_columns:
        raise KeyError(
            "Requested feature columns are not present in frame: "
            f"{', '.join(missing_columns)}"
        )
    return frame[list(feature_columns)].copy()


def _missingness_by_column(frame: pd.DataFrame) -> list[dict[str, Any]]:
    question_mark_mask, blank_or_whitespace_mask = _string_token_masks(frame)
    standard_null_mask = frame.isna()
    combined_missing_mask = standard_null_mask | question_mark_mask | blank_or_whitespace_mask

    n_rows = len(frame)
    summary: list[dict[str, Any]] = []
    for column in frame.columns:
        missing_count = int(combined_missing_mask[column].sum())
        summary.append(
            {
                "column": column,
                "missing_count": missing_count,
                "missing_rate": (missing_count / n_rows) if n_rows else 0.0,
            }
        )

    summary.sort(key=lambda item: item["missing_rate"], reverse=True)
    return summary


def _distribution(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in frame.columns:
        return {}

    counts = frame[column].astype("string").fillna("<NA>").str.strip().value_counts(dropna=False)
    return {str(label): int(count) for label, count in counts.items()}


def build_preprocessing_summary(
    raw_frame: pd.DataFrame,
    preprocessed_frame: pd.DataFrame,
    feature_candidacy: FeatureCandidacy,
    *,
    target_column: str = DEFAULT_TARGET_COLUMN,
    binary_target_column: str = DEFAULT_BINARY_TARGET_COLUMN,
) -> dict[str, Any]:
    warnings: list[str] = []
    if "patient_nbr" not in preprocessed_frame.columns:
        warnings.append("patient_nbr is missing; grouped splitting will fail.")

    return {
        "shape_original": {
            "n_rows": int(raw_frame.shape[0]),
            "n_columns": int(raw_frame.shape[1]),
        },
        "shape_after_preprocessing": {
            "n_rows": int(preprocessed_frame.shape[0]),
            "n_columns": int(preprocessed_frame.shape[1]),
        },
        "missingness_before_normalization": _missingness_by_column(raw_frame),
        "missingness_after_normalization": _missingness_by_column(preprocessed_frame),
        "target_distribution_readmitted": _distribution(preprocessed_frame, target_column),
        "target_distribution_readmitted_30d": _distribution(
            preprocessed_frame,
            binary_target_column,
        ),
        "identifier_columns": list(feature_candidacy.identifier_columns),
        "target_columns": list(feature_candidacy.target_columns),
        "excluded_feature_columns": list(feature_candidacy.excluded_feature_columns),
        "candidate_feature_columns": list(feature_candidacy.candidate_feature_columns),
        "warnings": warnings,
    }
