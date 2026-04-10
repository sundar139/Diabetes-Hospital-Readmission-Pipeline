from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


@dataclass(frozen=True)
class GroupedSplitResult:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def _validate_split_sizes(train_size: float, val_size: float, test_size: float) -> None:
    if train_size <= 0 or val_size <= 0 or test_size <= 0:
        raise ValueError("Split sizes must all be positive.")

    total = train_size + val_size + test_size
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Split sizes must sum to 1.0, got {total:.6f}.")


def grouped_split_by_patient(
    frame: pd.DataFrame,
    *,
    group_column: str = "patient_nbr",
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> GroupedSplitResult:
    _validate_split_sizes(train_size, val_size, test_size)

    if group_column not in frame.columns:
        raise KeyError(
            f"Grouped split requires '{group_column}' column, but it is missing."
        )

    group_series = frame[group_column]
    missing_group_count = int(group_series.isna().sum())
    if missing_group_count > 0:
        raise ValueError(
            f"Grouped split cannot proceed because '{group_column}' has "
            f"{missing_group_count} missing value(s)."
        )

    if frame.empty:
        raise ValueError("Cannot split an empty dataset.")

    first_splitter = GroupShuffleSplit(
        n_splits=1,
        train_size=train_size,
        random_state=random_state,
    )
    train_idx, temp_idx = next(first_splitter.split(frame, groups=group_series))

    train_frame = frame.iloc[train_idx].reset_index(drop=True)
    temp_frame = frame.iloc[temp_idx].reset_index(drop=True)

    relative_val_size = val_size / (val_size + test_size)
    second_splitter = GroupShuffleSplit(
        n_splits=1,
        train_size=relative_val_size,
        random_state=random_state + 1,
    )
    val_idx, test_idx = next(
        second_splitter.split(temp_frame, groups=temp_frame[group_column])
    )

    val_frame = temp_frame.iloc[val_idx].reset_index(drop=True)
    test_frame = temp_frame.iloc[test_idx].reset_index(drop=True)

    assert_no_group_overlap(
        train_frame,
        val_frame,
        test_frame,
        group_column=group_column,
    )

    return GroupedSplitResult(train=train_frame, val=val_frame, test=test_frame)


def assert_no_group_overlap(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    group_column: str = "patient_nbr",
) -> None:
    train_groups = set(train_frame[group_column].dropna().tolist())
    val_groups = set(val_frame[group_column].dropna().tolist())
    test_groups = set(test_frame[group_column].dropna().tolist())

    train_val_overlap = train_groups & val_groups
    train_test_overlap = train_groups & test_groups
    val_test_overlap = val_groups & test_groups

    if train_val_overlap or train_test_overlap or val_test_overlap:
        raise ValueError(
            "Leakage detected in grouped split: "
            f"train-val={len(train_val_overlap)}, "
            f"train-test={len(train_test_overlap)}, "
            f"val-test={len(val_test_overlap)} overlaps."
        )


def _distribution(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in frame.columns:
        return {}
    counts = frame[column].astype("string").fillna("<NA>").str.strip().value_counts(dropna=False)
    return {str(label): int(count) for label, count in counts.items()}


def build_split_manifest(
    split_result: GroupedSplitResult,
    *,
    group_column: str = "patient_nbr",
    multiclass_target_column: str = "readmitted",
    binary_target_column: str = "readmitted_30d",
    excluded_feature_columns: tuple[str, ...] | list[str],
    random_state: int,
) -> dict[str, Any]:
    train_frame = split_result.train
    val_frame = split_result.val
    test_frame = split_result.test

    n_rows_train = int(len(train_frame))
    n_rows_val = int(len(val_frame))
    n_rows_test = int(len(test_frame))

    n_patients_train = int(train_frame[group_column].nunique(dropna=True))
    n_patients_val = int(val_frame[group_column].nunique(dropna=True))
    n_patients_test = int(test_frame[group_column].nunique(dropna=True))

    leakage_check_passed = True
    assert_no_group_overlap(train_frame, val_frame, test_frame, group_column=group_column)

    return {
        "n_rows_total": n_rows_train + n_rows_val + n_rows_test,
        "n_rows_train": n_rows_train,
        "n_rows_val": n_rows_val,
        "n_rows_test": n_rows_test,
        "n_patients_total": n_patients_train + n_patients_val + n_patients_test,
        "n_patients_train": n_patients_train,
        "n_patients_val": n_patients_val,
        "n_patients_test": n_patients_test,
        "readmitted_distribution_by_split": {
            "train": _distribution(train_frame, multiclass_target_column),
            "val": _distribution(val_frame, multiclass_target_column),
            "test": _distribution(test_frame, multiclass_target_column),
        },
        "readmitted_30d_distribution_by_split": {
            "train": _distribution(train_frame, binary_target_column),
            "val": _distribution(val_frame, binary_target_column),
            "test": _distribution(test_frame, binary_target_column),
        },
        "leakage_check_passed": leakage_check_passed,
        "excluded_feature_columns": list(excluded_feature_columns),
        "random_state": int(random_state),
    }


def write_split_manifest(manifest: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return output_path
