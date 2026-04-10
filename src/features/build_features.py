from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

ENGINEERED_FEATURE_NAMES: tuple[str, ...] = (
    "recurrency",
    "patient_severity",
    "medication_change_ratio",
    "utilization_intensity",
    "complex_discharge_flag",
    "age_bucket_risk",
)

MEDICATION_STATUS_VALUES: frozenset[str] = frozenset({"NO", "STEADY", "UP", "DOWN"})
MEDICATION_CHANGED_VALUES: frozenset[str] = frozenset({"UP", "DOWN"})
DEFAULT_HOME_DISPOSITION_IDS: frozenset[int] = frozenset({1, 6, 8})

DEFAULT_MEDICATION_STATUS_COLUMNS: tuple[str, ...] = (
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
)


@dataclass(frozen=True)
class FeatureBookkeeping:
    identifier_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    raw_base_feature_columns: tuple[str, ...]
    engineered_feature_columns: tuple[str, ...]
    excluded_feature_columns: tuple[str, ...]
    model_candidate_columns: tuple[str, ...]
    feature_source_map: dict[str, list[str]]


@dataclass(frozen=True)
class FeatureEngineeringResult:
    frame: pd.DataFrame
    bookkeeping: FeatureBookkeeping
    fallback_counts: dict[str, int]
    warnings: tuple[str, ...]
    medication_status_columns: tuple[str, ...]


def _require_columns(frame: pd.DataFrame, required_columns: tuple[str, ...], context: str) -> None:
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise KeyError(
            f"Missing required column(s) for {context}: {', '.join(missing_columns)}"
        )


def _to_numeric(frame: pd.DataFrame, column: str, *, fill_value: float = 0.0) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    return values.fillna(fill_value)


def _resolve_diagnoses_column(frame: pd.DataFrame, *, context: str) -> str:
    diagnosis_aliases = ("num_diagnoses", "number_diagnoses")
    for column in diagnosis_aliases:
        if column in frame.columns:
            return column

    raise KeyError(
        f"Missing required column(s) for {context}: num_diagnoses or number_diagnoses"
    )


def build_recurrency_feature(
    frame: pd.DataFrame,
    *,
    patient_column: str = "patient_nbr",
) -> pd.Series:
    _require_columns(frame, (patient_column,), "recurrency")

    if bool(frame[patient_column].isna().any()):
        raise ValueError(
            f"Column '{patient_column}' contains missing values; recurrency cannot be computed."
        )

    encounter_count = frame.groupby(patient_column)[patient_column].transform("size")
    recurrency = (encounter_count - 1).clip(lower=0).astype("int32")
    return recurrency.rename("recurrency")


def build_patient_severity_feature(frame: pd.DataFrame) -> pd.Series:
    severity_columns = (
        "time_in_hospital",
        "number_inpatient",
        "number_outpatient",
        "number_emergency",
    )
    _require_columns(frame, severity_columns, "patient_severity")
    diagnoses_column = _resolve_diagnoses_column(frame, context="patient_severity")

    normalized_time = (_to_numeric(frame, "time_in_hospital") / 14.0).clip(lower=0.0, upper=1.0)
    normalized_diagnoses = (_to_numeric(frame, diagnoses_column) / 16.0).clip(
        lower=0.0,
        upper=1.0,
    )
    normalized_inpatient = (_to_numeric(frame, "number_inpatient") / 10.0).clip(
        lower=0.0,
        upper=1.0,
    )
    normalized_outpatient = (_to_numeric(frame, "number_outpatient") / 10.0).clip(
        lower=0.0,
        upper=1.0,
    )
    normalized_emergency = (_to_numeric(frame, "number_emergency") / 10.0).clip(
        lower=0.0,
        upper=1.0,
    )

    severity = (
        0.30 * normalized_time
        + 0.25 * normalized_diagnoses
        + 0.20 * normalized_inpatient
        + 0.15 * normalized_outpatient
        + 0.10 * normalized_emergency
    )
    return severity.astype("float32").rename("patient_severity")


def detect_medication_status_columns(
    frame: pd.DataFrame,
    *,
    candidate_columns: tuple[str, ...] = DEFAULT_MEDICATION_STATUS_COLUMNS,
) -> tuple[str, ...]:
    detected: list[str] = []

    for column in candidate_columns:
        if column not in frame.columns:
            continue

        normalized = frame[column].astype("string").str.strip().str.upper()
        if bool(normalized.isin(MEDICATION_STATUS_VALUES).any()):
            detected.append(column)

    return tuple(detected)


def build_medication_change_ratio_feature(
    frame: pd.DataFrame,
    *,
    medication_columns: tuple[str, ...],
) -> tuple[pd.Series, int]:
    if not medication_columns:
        ratio = pd.Series(0.0, index=frame.index, dtype="float32", name="medication_change_ratio")
        return ratio, int(len(frame))

    observed = pd.Series(0, index=frame.index, dtype="int32")
    changed = pd.Series(0, index=frame.index, dtype="int32")

    for column in medication_columns:
        normalized = frame[column].astype("string").str.strip().str.upper()
        observed += normalized.isin(MEDICATION_STATUS_VALUES).astype("int32")
        changed += normalized.isin(MEDICATION_CHANGED_VALUES).astype("int32")

    zero_denominator_mask = observed == 0
    observed_float = observed.astype("float32")
    safe_denominator = observed_float.where(observed_float > 0, other=float("nan"))
    ratio = changed.astype("float32") / safe_denominator
    ratio = ratio.fillna(0.0).astype("float32")
    ratio.name = "medication_change_ratio"
    return ratio, int(zero_denominator_mask.sum())


def build_utilization_intensity_feature(frame: pd.DataFrame) -> pd.Series:
    utilization_columns = ("number_inpatient", "number_outpatient", "number_emergency")
    _require_columns(frame, utilization_columns, "utilization_intensity")

    total = (
        _to_numeric(frame, "number_inpatient")
        + _to_numeric(frame, "number_outpatient")
        + _to_numeric(frame, "number_emergency")
    )
    return total.astype("float32").rename("utilization_intensity")


def build_complex_discharge_flag_feature(
    frame: pd.DataFrame,
    *,
    home_disposition_ids: frozenset[int] = DEFAULT_HOME_DISPOSITION_IDS,
) -> tuple[pd.Series, int]:
    _require_columns(frame, ("discharge_disposition_id",), "complex_discharge_flag")

    disposition = pd.to_numeric(frame["discharge_disposition_id"], errors="coerce")
    fallback_count = int(disposition.isna().sum())
    complex_flag = (~disposition.isin(home_disposition_ids)).astype("int8")
    complex_flag.name = "complex_discharge_flag"
    return complex_flag, fallback_count


def build_age_bucket_risk_feature(frame: pd.DataFrame) -> tuple[pd.Series, int]:
    _require_columns(frame, ("age",), "age_bucket_risk")

    age_text = frame["age"].astype("string").str.strip()
    extracted = age_text.str.extract(r"^\[(\d+)-(\d+)\)$")
    lower_bound = pd.to_numeric(extracted[0], errors="coerce")

    missing_or_malformed = age_text.isna() | lower_bound.isna()
    risk = (lower_bound // 10).astype("Int64").fillna(-1).astype("int16")
    risk.name = "age_bucket_risk"
    return risk, int(missing_or_malformed.sum())


def _build_bookkeeping(
    frame_with_features: pd.DataFrame,
    *,
    original_columns: list[str],
    feature_source_map: dict[str, list[str]],
    identifier_columns: tuple[str, ...] = ("encounter_id", "patient_nbr"),
    target_columns: tuple[str, ...] = ("readmitted", "readmitted_30d"),
) -> FeatureBookkeeping:
    existing_identifiers = tuple(
        column for column in identifier_columns if column in frame_with_features.columns
    )
    existing_targets = tuple(
        column for column in target_columns if column in frame_with_features.columns
    )
    existing_engineered = tuple(
        column for column in ENGINEERED_FEATURE_NAMES if column in frame_with_features.columns
    )

    identifier_set = set(existing_identifiers)
    target_set = set(existing_targets)
    engineered_set = set(existing_engineered)

    raw_base_features = tuple(
        column
        for column in original_columns
        if column not in identifier_set
        and column not in target_set
        and column not in engineered_set
    )

    excluded_features = existing_identifiers
    excluded_set = set(excluded_features)
    model_candidates = tuple(
        column
        for column in frame_with_features.columns
        if column not in excluded_set and column not in target_set
    )

    return FeatureBookkeeping(
        identifier_columns=existing_identifiers,
        target_columns=existing_targets,
        raw_base_feature_columns=raw_base_features,
        engineered_feature_columns=existing_engineered,
        excluded_feature_columns=excluded_features,
        model_candidate_columns=model_candidates,
        feature_source_map=feature_source_map,
    )


def engineer_clinical_features(
    frame: pd.DataFrame,
    *,
    medication_status_columns: tuple[str, ...] | None = None,
) -> FeatureEngineeringResult:
    original_columns = list(frame.columns)
    output = frame.copy(deep=True)
    warnings: list[str] = []
    diagnoses_column = _resolve_diagnoses_column(output, context="patient_severity")

    recurrency = build_recurrency_feature(output)
    patient_severity = build_patient_severity_feature(output)
    utilization_intensity = build_utilization_intensity_feature(output)
    complex_discharge_flag, discharge_fallback_count = build_complex_discharge_flag_feature(output)
    age_bucket_risk, age_fallback_count = build_age_bucket_risk_feature(output)

    if medication_status_columns is None:
        medication_columns = detect_medication_status_columns(output)
    else:
        medication_columns = tuple(
            column for column in medication_status_columns if column in output.columns
        )
        dropped_columns = sorted(set(medication_status_columns) - set(medication_columns))
        if dropped_columns:
            warnings.append(
                "Some requested medication columns were missing and ignored: "
                f"{', '.join(dropped_columns)}."
            )

    medication_change_ratio, medication_ratio_fallback_count = (
        build_medication_change_ratio_feature(
            output,
            medication_columns=medication_columns,
        )
    )

    output["recurrency"] = recurrency
    output["patient_severity"] = patient_severity
    output["medication_change_ratio"] = medication_change_ratio
    output["utilization_intensity"] = utilization_intensity
    output["complex_discharge_flag"] = complex_discharge_flag
    output["age_bucket_risk"] = age_bucket_risk

    feature_source_map = {
        "recurrency": ["patient_nbr"],
        "patient_severity": [
            "time_in_hospital",
            diagnoses_column,
            "number_inpatient",
            "number_outpatient",
            "number_emergency",
        ],
        "medication_change_ratio": list(medication_columns),
        "utilization_intensity": [
            "number_inpatient",
            "number_outpatient",
            "number_emergency",
        ],
        "complex_discharge_flag": ["discharge_disposition_id"],
        "age_bucket_risk": ["age"],
    }

    fallback_counts = {
        "medication_change_ratio_zero_denominator": medication_ratio_fallback_count,
        "complex_discharge_flag_missing_or_malformed": discharge_fallback_count,
        "age_bucket_risk_missing_or_malformed": age_fallback_count,
    }

    bookkeeping = _build_bookkeeping(
        output,
        original_columns=original_columns,
        feature_source_map=feature_source_map,
    )

    if len(output) != len(frame):
        raise ValueError("Feature engineering altered row count, which is not allowed.")
    if not output.index.equals(frame.index):
        raise ValueError("Feature engineering altered row index alignment, which is not allowed.")

    return FeatureEngineeringResult(
        frame=output,
        bookkeeping=bookkeeping,
        fallback_counts=fallback_counts,
        warnings=tuple(warnings),
        medication_status_columns=medication_columns,
    )
