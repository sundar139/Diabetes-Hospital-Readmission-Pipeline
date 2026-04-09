from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.data.load_raw import detect_column_role

UTILIZATION_COLUMNS: frozenset[str] = frozenset(
    {
        "time_in_hospital",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
    }
)

DESCRIPTIONS: dict[str, str] = {
    "encounter_id": "Encounter-level identifier for each hospital visit.",
    "patient_nbr": "Patient-level identifier shared across multiple encounters.",
    "readmitted": "Readmission label indicating timing of return admission.",
    "age": "Age bucket at time of encounter.",
    "race": "Self-reported race category.",
    "gender": "Patient gender category.",
    "time_in_hospital": "Length of stay in days.",
    "num_lab_procedures": "Number of laboratory procedures during encounter.",
    "num_medications": "Number of medications prescribed during encounter.",
    "number_outpatient": "Prior outpatient visits count.",
    "number_emergency": "Prior emergency visits count.",
    "number_inpatient": "Prior inpatient visits count.",
}


def _null_like_mask(series: pd.Series) -> pd.Series:
    text = series.astype("string")
    non_null = text.notna()
    stripped = text.str.strip()
    question_mark = non_null & (stripped == "?")
    empty_string = non_null & (text == "")
    whitespace_only = non_null & text.str.fullmatch(r"\s+", na=False)
    return series.isna() | question_mark | empty_string | whitespace_only


def _example_values(series: pd.Series, max_examples: int = 3) -> list[str]:
    mask = _null_like_mask(series)
    values = series[~mask].drop_duplicates().astype("string").head(max_examples)
    return [str(value) for value in values.tolist()]


def _description_for_column(column_name: str, role: str) -> str:
    normalized = column_name.strip().lower()

    if normalized in DESCRIPTIONS:
        return DESCRIPTIONS[normalized]
    if role == "medication-status":
        return "Medication exposure or medication-change indicator for the encounter."
    if role == "diagnosis-like":
        return "Diagnosis code feature captured during or around the encounter."
    if role == "identifier":
        return "Identifier-like field used for entity tracking and deduplication checks."
    if role == "target":
        return "Prediction target field."
    if role == "numeric":
        return "Numeric feature likely suitable for descriptive statistics and scaling decisions."
    return "Categorical feature representing grouped or coded patient/encounter context."


def build_data_dictionary(
    frame: pd.DataFrame,
    *,
    target_column: str = "readmitted",
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for column_name in frame.columns:
        series = frame[column_name]
        role = detect_column_role(column_name, series)
        normalized = column_name.strip().lower()
        null_like_mask = _null_like_mask(series)
        missing_token_rate = float(null_like_mask.mean()) if len(series) else 0.0

        record = {
            "column_name": column_name,
            "role": role,
            "dtype": str(series.dtype),
            "missing_token_rate": missing_token_rate,
            "non_null_count": int(series.notna().sum()),
            "unique_count": int(series.nunique(dropna=True)),
            "example_values": _example_values(series),
            "description": _description_for_column(column_name, role),
            "is_identifier": role == "identifier",
            "is_target": normalized == target_column.strip().lower(),
            "is_medication": role == "medication-status",
            "is_diagnosis": role == "diagnosis-like",
            "is_utilization": normalized in UTILIZATION_COLUMNS,
        }
        records.append(record)

    return records


def render_data_dictionary_markdown(records: list[dict[str, Any]]) -> str:
    lines: list[str] = [
        "# Data Dictionary",
        "",
        (
            "| Column | Role | DType | Non-null | Unique | Missing-token rate | "
            "Example values | Description | Tags |"
        ),
        "| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |",
    ]

    for record in records:
        tags = []
        if record["is_identifier"]:
            tags.append("identifier")
        if record["is_target"]:
            tags.append("target")
        if record["is_medication"]:
            tags.append("medication")
        if record["is_diagnosis"]:
            tags.append("diagnosis")
        if record["is_utilization"]:
            tags.append("utilization")

        example_values = ", ".join(record["example_values"]) if record["example_values"] else ""
        tag_text = ", ".join(tags)

        lines.append(
            "| "
            + " | ".join(
                [
                    str(record["column_name"]),
                    str(record["role"]),
                    str(record["dtype"]),
                    str(record["non_null_count"]),
                    str(record["unique_count"]),
                    f"{record['missing_token_rate']:.2%}",
                    example_values,
                    str(record["description"]),
                    tag_text,
                ]
            )
            + " |"
        )

    return "\n".join(lines) + "\n"


def write_data_dictionary_markdown(records: list[dict[str, Any]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_data_dictionary_markdown(records), encoding="utf-8")
    return output_path
