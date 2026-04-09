from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.api.types import is_numeric_dtype

from src.config.settings import Settings, get_settings

REQUIRED_COLUMNS: tuple[str, ...] = (
    "encounter_id",
    "patient_nbr",
    "readmitted",
    "age",
    "race",
    "gender",
    "time_in_hospital",
    "num_lab_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
)

ALLOWED_TARGET_VALUES: tuple[str, ...] = ("NO", ">30", "<30")

COUNT_LIKE_COLUMNS: tuple[str, ...] = (
    "time_in_hospital",
    "num_lab_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
)


def _to_python_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _string_token_masks(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    question_mark_mask = pd.DataFrame(False, index=frame.index, columns=frame.columns)
    empty_string_mask = pd.DataFrame(False, index=frame.index, columns=frame.columns)
    whitespace_mask = pd.DataFrame(False, index=frame.index, columns=frame.columns)

    for column in frame.columns:
        text_series = frame[column].astype("string")
        non_null_mask = text_series.notna()
        stripped = text_series.str.strip()

        question_mark_mask[column] = non_null_mask & (stripped == "?")
        empty_string_mask[column] = non_null_mask & (text_series == "")
        whitespace_mask[column] = non_null_mask & text_series.str.fullmatch(r"\s+", na=False)

    return question_mark_mask, empty_string_mask, whitespace_mask


def validate_required_columns(
    frame: pd.DataFrame,
    required_columns: Iterable[str] = REQUIRED_COLUMNS,
) -> tuple[dict[str, bool], list[str]]:
    status = {column: column in frame.columns for column in required_columns}
    missing = [column for column, present in status.items() if not present]
    return status, missing


def summarize_identifiers(frame: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "encounter_id_exists": "encounter_id" in frame.columns,
        "encounter_id_unique": None,
        "encounter_id_non_null_count": None,
        "encounter_id_unique_count": None,
        "encounter_id_duplicate_count": None,
        "patient_nbr_exists": "patient_nbr" in frame.columns,
        "patient_nbr_unique_count": None,
        "patient_nbr_repeated_row_count": None,
        "patient_nbr_top_repeat_counts": {},
    }

    if "encounter_id" in frame.columns:
        encounter = frame["encounter_id"]
        non_null_count = int(encounter.notna().sum())
        unique_count = int(encounter.nunique(dropna=True))
        duplicate_count = int(encounter.duplicated().sum())

        summary["encounter_id_non_null_count"] = non_null_count
        summary["encounter_id_unique_count"] = unique_count
        summary["encounter_id_duplicate_count"] = duplicate_count
        summary["encounter_id_unique"] = bool(
            unique_count == non_null_count and duplicate_count == 0
        )

    if "patient_nbr" in frame.columns:
        patient_nbr = frame["patient_nbr"]
        repeated_counts = patient_nbr.value_counts(dropna=True)
        repeated_counts = repeated_counts[repeated_counts > 1].head(5)

        summary["patient_nbr_unique_count"] = int(patient_nbr.nunique(dropna=True))
        summary["patient_nbr_repeated_row_count"] = int(patient_nbr.duplicated(keep=False).sum())
        summary["patient_nbr_top_repeat_counts"] = {
            str(key): int(value) for key, value in repeated_counts.items()
        }

    return summary


def _missingness_summary(
    frame: pd.DataFrame,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    question_mark_mask, empty_string_mask, whitespace_mask = _string_token_masks(frame)
    standard_null_mask = frame.isna()
    combined_missing_mask = (
        standard_null_mask | question_mark_mask | empty_string_mask | whitespace_mask
    )

    n_rows = len(frame)
    per_column: list[dict[str, Any]] = []

    for column in frame.columns:
        missing_count = int(combined_missing_mask[column].sum())
        standard_count = int(standard_null_mask[column].sum())
        qmark_count = int(question_mark_mask[column].sum())
        empty_count = int(empty_string_mask[column].sum())
        ws_count = int(whitespace_mask[column].sum())

        per_column.append(
            {
                "column": column,
                "missing_count": missing_count,
                "missing_rate": (missing_count / n_rows) if n_rows else 0.0,
                "standard_null_count": standard_count,
                "question_mark_count": qmark_count,
                "empty_string_count": empty_count,
                "whitespace_only_count": ws_count,
            }
        )

    per_column.sort(key=lambda item: item["missing_rate"], reverse=True)

    null_like_counts = {
        "standard_nulls": int(standard_null_mask.sum().sum()),
        "question_mark": int(question_mark_mask.sum().sum()),
        "empty_string": int(empty_string_mask.sum().sum()),
        "whitespace_only": int(whitespace_mask.sum().sum()),
    }
    return per_column, null_like_counts


def _column_type_lists(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_columns = [column for column in frame.columns if is_numeric_dtype(frame[column])]
    categorical_columns = [column for column in frame.columns if column not in numeric_columns]
    return numeric_columns, categorical_columns


def _categorical_cardinality_summary(
    frame: pd.DataFrame,
    categorical_columns: Iterable[str],
    top_n: int = 5,
) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for column in categorical_columns:
        value_counts = (
            frame[column]
            .astype("string")
            .fillna("<NA>")
            .str.strip()
            .replace("", "<EMPTY>")
            .value_counts(dropna=False)
            .head(top_n)
        )

        summary[column] = {
            "unique_non_null_count": int(frame[column].nunique(dropna=True)),
            "top_values": {str(key): int(value) for key, value in value_counts.items()},
        }

    return summary


def _numeric_descriptive_summary(
    frame: pd.DataFrame,
    numeric_columns: Iterable[str],
) -> dict[str, dict[str, Any]]:
    numeric_columns_list = list(numeric_columns)
    if not numeric_columns_list:
        return {}

    described = frame[numeric_columns_list].describe().transpose()
    summary: dict[str, dict[str, Any]] = {}
    for column_name, row in described.iterrows():
        summary[column_name] = {
            key: _to_python_scalar(value) for key, value in row.to_dict().items()
        }
    return summary


def _target_distribution_and_warnings(
    frame: pd.DataFrame,
    target_column: str,
) -> tuple[dict[str, int], list[str]]:
    warnings: list[str] = []
    if target_column not in frame.columns:
        warnings.append(f"Target column '{target_column}' is missing.")
        return {}, warnings

    normalized = frame[target_column].astype("string").str.strip()
    distribution = normalized.fillna("<NA>").value_counts(dropna=False)
    target_distribution = {str(key): int(value) for key, value in distribution.items()}

    invalid_values = sorted(
        {
            str(value)
            for value in normalized.dropna().unique().tolist()
            if str(value) not in ALLOWED_TARGET_VALUES
        }
    )
    if invalid_values:
        warnings.append(
            "Unexpected target values found in readmitted: "
            f"{', '.join(invalid_values)}."
        )

    missing_expected = [
        expected
        for expected in ALLOWED_TARGET_VALUES
        if expected not in target_distribution
    ]
    if missing_expected:
        warnings.append(
            "Expected target classes are absent: "
            f"{', '.join(missing_expected)}."
        )

    return target_distribution, warnings


def _suspicious_value_warnings(frame: pd.DataFrame) -> list[str]:
    warnings: list[str] = []
    for column in COUNT_LIKE_COLUMNS:
        if column not in frame.columns:
            continue
        if not is_numeric_dtype(frame[column]):
            continue
        negative_count = int((frame[column] < 0).sum())
        if negative_count > 0:
            warnings.append(
                f"Column '{column}' contains {negative_count} negative value(s), "
                "which is suspicious for count-like fields."
            )
    return warnings


def build_raw_validation_summary(
    frame: pd.DataFrame,
    *,
    target_column: str = "readmitted",
) -> dict[str, Any]:
    n_rows, n_columns = frame.shape
    required_status, missing_required = validate_required_columns(frame)
    identifier_summary = summarize_identifiers(frame)
    duplicate_rows = int(frame.duplicated().sum())

    missingness_summary, null_like_counts = _missingness_summary(frame)
    numeric_columns, categorical_columns = _column_type_lists(frame)
    categorical_cardinality = _categorical_cardinality_summary(frame, categorical_columns)
    numeric_summary = _numeric_descriptive_summary(frame, numeric_columns)
    target_distribution, target_warnings = _target_distribution_and_warnings(frame, target_column)

    warnings: list[str] = []
    if missing_required:
        warnings.append(f"Missing required columns: {', '.join(missing_required)}.")

    if duplicate_rows > 0:
        warnings.append(f"Dataset contains {duplicate_rows} duplicate full row(s).")

    encounter_unique = identifier_summary["encounter_id_unique"]
    if encounter_unique is False:
        duplicate_count = identifier_summary["encounter_id_duplicate_count"]
        warnings.append(
            "Encounter identifier is not unique "
            f"({duplicate_count} duplicate encounter_id value(s))."
        )

    if null_like_counts["question_mark"] > 0:
        warnings.append(
            f"Found {null_like_counts['question_mark']} '?' token(s) representing missing data."
        )

    high_missing_columns = [
        item["column"]
        for item in missingness_summary
        if item["missing_rate"] >= 0.30
    ]
    if high_missing_columns:
        warnings.append(
            "High missingness (>=30%) detected in: "
            f"{', '.join(high_missing_columns)}."
        )

    warnings.extend(target_warnings)
    warnings.extend(_suspicious_value_warnings(frame))

    summary = {
        "n_rows": int(n_rows),
        "n_columns": int(n_columns),
        "required_columns_present": required_status,
        "duplicate_rows": duplicate_rows,
        "encounter_id_unique": encounter_unique,
        "patient_nbr_unique_count": identifier_summary["patient_nbr_unique_count"],
        "target_distribution": target_distribution,
        "null_like_counts": null_like_counts,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "warnings": warnings,
        "missingness_summary": missingness_summary,
        "categorical_cardinality_summary": categorical_cardinality,
        "numeric_descriptive_summary": numeric_summary,
        "identifier_summary": identifier_summary,
    }
    return summary


def _recommended_actions(summary: dict[str, Any]) -> list[str]:
    actions = [
        (
            "Normalize null-like tokens ('?', empty, and whitespace-only values) "
            "into standard missing values."
        ),
        "Define a clear strategy for high-missingness columns before feature engineering.",
        (
            "Decide leakage-safe grouping strategy for patient-level records "
            "before train/validation split."
        ),
        "Confirm target label integrity and document any out-of-vocabulary readmitted values.",
        "Resolve duplicate records and confirm deduplication policy for encounter-level analytics.",
    ]

    if not summary.get("warnings"):
        actions.append("Proceed to preprocessing after locking schema assumptions in tests.")

    deduplicated: list[str] = []
    for action in actions:
        if action not in deduplicated:
            deduplicated.append(action)
    return deduplicated


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    divider_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    row_lines = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, divider_line, *row_lines])


def render_raw_validation_report(summary: dict[str, Any]) -> str:
    required_table_rows = [
        [column, "PASS" if present else "FAIL"]
        for column, present in summary["required_columns_present"].items()
    ]
    required_table = _markdown_table(["Column", "Status"], required_table_rows)

    missingness_rows = []
    for item in summary["missingness_summary"][:20]:
        missingness_rows.append(
            [
                item["column"],
                str(item["missing_count"]),
                f"{item['missing_rate']:.2%}",
                str(item["question_mark_count"]),
                str(item["empty_string_count"]),
                str(item["whitespace_only_count"]),
            ]
        )
    missingness_table = _markdown_table(
        [
            "Column",
            "Missing Count",
            "Missing Rate",
            "? Count",
            "Empty Count",
            "Whitespace Count",
        ],
        missingness_rows,
    )

    target_rows = [
        [label, str(count)] for label, count in summary["target_distribution"].items()
    ]
    target_table = (
        _markdown_table(["readmitted", "Count"], target_rows)
        if target_rows
        else "Target column not present."
    )

    identifier_summary = summary["identifier_summary"]
    identifier_lines = [
        f"- encounter_id exists: {identifier_summary['encounter_id_exists']}",
        f"- encounter_id unique: {identifier_summary['encounter_id_unique']}",
        f"- encounter_id duplicate count: {identifier_summary['encounter_id_duplicate_count']}",
        f"- patient_nbr exists: {identifier_summary['patient_nbr_exists']}",
        f"- patient_nbr unique count: {identifier_summary['patient_nbr_unique_count']}",
        f"- patient_nbr repeated row count: {identifier_summary['patient_nbr_repeated_row_count']}",
    ]

    warnings = summary["warnings"]
    warning_lines = [f"- {warning}" for warning in warnings] if warnings else ["- None"]

    action_lines = [f"- {action}" for action in _recommended_actions(summary)]

    return "\n".join(
        [
            "# Raw Data Validation Report",
            "",
            "## Dataset Shape",
            "",
            f"- Rows: {summary['n_rows']}",
            f"- Columns: {summary['n_columns']}",
            "",
            "## Required-Column Status",
            "",
            required_table,
            "",
            "## Duplicate Summary",
            "",
            f"- Duplicate full rows: {summary['duplicate_rows']}",
            f"- encounter_id unique: {summary['encounter_id_unique']}",
            f"- patient_nbr unique count: {summary['patient_nbr_unique_count']}",
            "",
            "## Missingness Summary",
            "",
            missingness_table,
            "",
            "## Target Distribution (readmitted)",
            "",
            target_table,
            "",
            "## Identifier Observations",
            "",
            *identifier_lines,
            "",
            "## Major Warnings",
            "",
            *warning_lines,
            "",
            "## Recommended Next Preprocessing Actions",
            "",
            *action_lines,
            "",
        ]
    )


def write_validation_outputs(
    summary: dict[str, Any],
    *,
    settings: Settings | None = None,
) -> dict[str, Path]:
    if settings is None:
        settings = get_settings()

    settings.reports_dir_path.mkdir(parents=True, exist_ok=True)
    settings.artifacts_dir_path.mkdir(parents=True, exist_ok=True)

    report_path = settings.reports_dir_path / "raw_validation_report.md"
    summary_path = settings.reports_dir_path / "raw_validation_summary.json"
    artifact_summary_path = settings.artifacts_dir_path / "raw_validation_summary.json"

    report_path.write_text(render_raw_validation_report(summary), encoding="utf-8")
    summary_json = json.dumps(summary, indent=2, sort_keys=True)
    summary_path.write_text(summary_json, encoding="utf-8")
    artifact_summary_path.write_text(summary_json, encoding="utf-8")

    return {
        "raw_validation_report": report_path,
        "raw_validation_summary": summary_path,
        "raw_validation_summary_artifact": artifact_summary_path,
    }


def generate_readmitted_distribution_figure(
    frame: pd.DataFrame,
    output_path: Path,
    *,
    target_column: str = "readmitted",
) -> Path | None:
    if target_column not in frame.columns:
        return None

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    counts = frame[target_column].astype("string").str.strip().fillna("<NA>").value_counts()
    ordered_labels = ["NO", ">30", "<30", "<NA>"]
    ordered_counts = [int(counts.get(label, 0)) for label in ordered_labels]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(7, 4))
    axis.bar(ordered_labels, ordered_counts, color=["#4C72B0", "#DD8452", "#55A868", "#8C8C8C"])
    axis.set_title("Raw readmitted Class Distribution")
    axis.set_xlabel("Class")
    axis.set_ylabel("Count")

    for idx, value in enumerate(ordered_counts):
        axis.text(idx, value, str(value), ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
