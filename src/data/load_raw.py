from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
from pandas.api.types import is_numeric_dtype

from src.config.settings import Settings, get_settings

ColumnRole = Literal[
    "identifier",
    "target",
    "numeric",
    "categorical",
    "medication-status",
    "diagnosis-like",
]

IDENTIFIER_COLUMNS: frozenset[str] = frozenset({"encounter_id", "patient_nbr"})
TARGET_COLUMNS: frozenset[str] = frozenset({"readmitted"})
MEDICATION_COLUMNS: frozenset[str] = frozenset(
    {
        "metformin",
        "repaglinide",
        "nateglinide",
        "chlorpropamide",
        "glimepiride",
        "acetohexamide",
        "glipizide",
        "glyburide",
        "tolbutamide",
        "pioglitazone",
        "rosiglitazone",
        "acarbose",
        "miglitol",
        "troglitazone",
        "tolazamide",
        "examide",
        "citoglipton",
        "insulin",
        "glyburide-metformin",
        "glipizide-metformin",
        "glimepiride-pioglitazone",
        "metformin-rosiglitazone",
        "metformin-pioglitazone",
        "change",
        "diabetesmed",
    }
)


def resolve_raw_data_path(
    csv_path: Path | str | None = None,
    settings: Settings | None = None,
) -> Path:
    if settings is None:
        settings = get_settings()

    if csv_path is None:
        candidate = settings.raw_data_path
    else:
        path_candidate = Path(csv_path).expanduser()
        candidate = (
            path_candidate
            if path_candidate.is_absolute()
            else (settings.project_root / path_candidate)
        )

    return candidate.resolve()


def load_raw_data(
    csv_path: Path | str | None = None,
    *,
    settings: Settings | None = None,
) -> pd.DataFrame:
    resolved_path = resolve_raw_data_path(csv_path=csv_path, settings=settings)

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Raw dataset file not found at {resolved_path}. "
            "Place the file at data/raw/diabetic_data.csv or pass --csv-path."
        )

    return pd.read_csv(resolved_path)


def detect_column_role(column_name: str, series: pd.Series) -> ColumnRole:
    normalized_name = column_name.strip().lower()

    if normalized_name in TARGET_COLUMNS:
        return "target"

    if normalized_name in IDENTIFIER_COLUMNS:
        return "identifier"

    if normalized_name.startswith("diag_") or "diagnosis" in normalized_name:
        return "diagnosis-like"

    if normalized_name in MEDICATION_COLUMNS:
        return "medication-status"

    if is_numeric_dtype(series):
        return "numeric"

    return "categorical"


def detect_column_roles(
    frame: pd.DataFrame,
) -> dict[str, ColumnRole]:
    return {
        column_name: detect_column_role(column_name, frame[column_name])
        for column_name in frame.columns
    }
