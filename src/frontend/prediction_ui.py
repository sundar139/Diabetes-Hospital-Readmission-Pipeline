from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd
import streamlit as st

from src.frontend.explanation_ui import render_explanation_section
from src.frontend.loaders import FrontendContext, PredictionArtifacts, build_feature_defaults
from src.frontend.prediction_engine import (
    PredictionWithExplanation,
    predict_with_deterministic_explanation,
)
from src.frontend.utils import (
    EMPTY_OPTION,
    format_probability,
    humanize_feature_name,
    probability_mapping_to_frame,
)

MEDICATION_FEATURES: tuple[str, ...] = (
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

CORE_FIELDS: tuple[str, ...] = (
    "race",
    "gender",
    "age",
    "weight",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
    "diag_1",
    "diag_2",
    "diag_3",
)

ENGINEERED_FIELDS: tuple[str, ...] = (
    "recurrency",
    "patient_severity",
    "medication_change_ratio",
    "utilization_intensity",
    "complex_discharge_flag",
    "age_bucket_risk",
)

INTEGER_FIELDS: set[str] = {
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
    "recurrency",
    "complex_discharge_flag",
    "age_bucket_risk",
}

FLOAT_FIELDS: set[str] = {
    "patient_severity",
    "medication_change_ratio",
    "utilization_intensity",
}

CATEGORICAL_OPTIONS: dict[str, list[str]] = {
    "race": [
        EMPTY_OPTION,
        "Caucasian",
        "AfricanAmerican",
        "Hispanic",
        "Asian",
        "Other",
    ],
    "gender": [EMPTY_OPTION, "Male", "Female", "Unknown/Invalid"],
    "age": [
        EMPTY_OPTION,
        "[0-10)",
        "[10-20)",
        "[20-30)",
        "[30-40)",
        "[40-50)",
        "[50-60)",
        "[60-70)",
        "[70-80)",
        "[80-90)",
        "[90-100)",
    ],
    "weight": [
        EMPTY_OPTION,
        "[0-25)",
        "[25-50)",
        "[50-75)",
        "[75-100)",
        "[100-125)",
        "[125-150)",
        "[150-175)",
        "[175-200)",
        ">200",
    ],
    "max_glu_serum": [EMPTY_OPTION, "Norm", ">200", ">300"],
    "A1Cresult": [EMPTY_OPTION, "Norm", ">7", ">8"],
    "change": [EMPTY_OPTION, "No", "Ch"],
    "diabetesMed": [EMPTY_OPTION, "No", "Yes"],
}


def _feature_widget_key(feature_name: str) -> str:
    return f"prediction_feature__{feature_name}"


def _normalize_for_widget(feature_name: str, value: Any) -> Any:
    if feature_name in INTEGER_FIELDS:
        if value is None:
            return 0
        return int(value)

    if feature_name in FLOAT_FIELDS:
        if value is None:
            return 0.0
        return float(value)

    options = _options_for_feature(feature_name, value)
    if options:
        if value is None or str(value).strip() == "":
            return EMPTY_OPTION
        return str(value)

    if value is None:
        return ""
    return str(value)


def _options_for_feature(feature_name: str, default_value: Any) -> list[str]:
    if feature_name in MEDICATION_FEATURES:
        options = [EMPTY_OPTION, "No", "Steady", "Up", "Down"]
    else:
        options = list(CATEGORICAL_OPTIONS.get(feature_name, []))

    if not options:
        return []

    if default_value is not None:
        as_text = str(default_value)
        if as_text and as_text not in options:
            options.append(as_text)

    return options


def _initialize_feature_state(defaults: Mapping[str, Any]) -> None:
    for feature_name, value in defaults.items():
        key = _feature_widget_key(feature_name)
        if key not in st.session_state:
            st.session_state[key] = _normalize_for_widget(feature_name, value)


def _set_feature_state(features: Mapping[str, Any]) -> None:
    for feature_name, value in features.items():
        st.session_state[_feature_widget_key(feature_name)] = _normalize_for_widget(
            feature_name,
            value,
        )


def _coerce_input(feature_name: str, value: Any) -> Any:
    if feature_name in INTEGER_FIELDS:
        return int(value)
    if feature_name in FLOAT_FIELDS:
        return float(value)

    if isinstance(value, str):
        normalized = value.strip()
        if not normalized or normalized == EMPTY_OPTION:
            return None
        return normalized

    return value


def _render_feature_input(feature_name: str, default_value: Any) -> Any:
    label = humanize_feature_name(feature_name)
    key = _feature_widget_key(feature_name)

    options = _options_for_feature(feature_name, default_value)
    if options:
        current_value = str(st.session_state.get(key, EMPTY_OPTION))
        if current_value not in options:
            options.append(current_value)
        selected = st.selectbox(label, options=options, key=key)
        return _coerce_input(feature_name, selected)

    if feature_name in INTEGER_FIELDS:
        value = st.number_input(
            label,
            min_value=0,
            step=1,
            key=key,
            value=int(_normalize_for_widget(feature_name, default_value)),
            format="%d",
        )
        return _coerce_input(feature_name, value)

    if feature_name in FLOAT_FIELDS:
        value = st.number_input(
            label,
            min_value=0.0,
            step=0.01,
            key=key,
            value=float(_normalize_for_widget(feature_name, default_value)),
            format="%.4f",
        )
        return _coerce_input(feature_name, value)

    text_value = st.text_input(
        label,
        key=key,
        value=str(_normalize_for_widget(feature_name, default_value)),
    )
    return _coerce_input(feature_name, text_value)


def _group_features(feature_columns: list[str]) -> list[tuple[str, list[str]]]:
    assigned: set[str] = set()

    core = [feature for feature in CORE_FIELDS if feature in feature_columns]
    assigned.update(core)

    medication = [feature for feature in MEDICATION_FEATURES if feature in feature_columns]
    for add_on_feature in ("change", "diabetesMed"):
        if add_on_feature in feature_columns and add_on_feature not in medication:
            medication.append(add_on_feature)
    assigned.update(medication)

    engineered = [feature for feature in ENGINEERED_FIELDS if feature in feature_columns]
    assigned.update(engineered)

    remaining = [feature for feature in feature_columns if feature not in assigned]

    return [
        ("Core Encounter Features", core),
        ("Medication and Therapy Features", medication),
        ("Engineered Risk Signals", engineered),
        ("Additional Features", remaining),
    ]


def _render_feature_groups(
    *,
    grouped_features: list[tuple[str, list[str]]],
    defaults: Mapping[str, Any],
) -> dict[str, Any]:
    submitted_features: dict[str, Any] = {}

    for group_index, (group_title, fields) in enumerate(grouped_features):
        if not fields:
            continue

        with st.expander(group_title, expanded=group_index == 0):
            columns = st.columns(2)
            for idx, feature_name in enumerate(fields):
                with columns[idx % 2]:
                    submitted_features[feature_name] = _render_feature_input(
                        feature_name,
                        defaults.get(feature_name),
                    )

    return submitted_features


def _render_result_metrics(prediction_result: PredictionWithExplanation) -> None:
    prediction = prediction_result.prediction

    st.subheader("Prediction Results")

    first_col, second_col, third_col, fourth_col = st.columns(4)
    first_col.metric("Binary prediction", str(prediction.binary_prediction))
    second_col.metric("Binary probability", format_probability(prediction.binary_probability))
    third_col.metric("Multiclass prediction", prediction.multiclass_prediction)

    top_multiclass_probability = 0.0
    if prediction.multiclass_probabilities:
        top_multiclass_probability = max(prediction.multiclass_probabilities.values())
    fourth_col.metric("Top multiclass probability", format_probability(top_multiclass_probability))

    probability_frame = probability_mapping_to_frame(prediction.multiclass_probabilities)
    if not probability_frame.empty:
        st.markdown("**Multiclass probability distribution**")
        chart_frame = probability_frame.set_index("class")
        st.bar_chart(chart_frame)

        display_frame = probability_frame.copy()
        display_frame["probability"] = display_frame["probability"].map(
            lambda value: f"{value * 100:.2f}%"
        )
        st.dataframe(display_frame, use_container_width=True, hide_index=True)

    with st.expander("Model metadata summary", expanded=False):
        metadata_frame = pd.DataFrame(
            [
                {"field": key, "value": value}
                for key, value in prediction.model_metadata_summary.items()
            ]
        )
        st.dataframe(metadata_frame, use_container_width=True, hide_index=True)


def render_prediction_page(*, context: FrontendContext, artifacts: PredictionArtifacts) -> None:
    st.header("Manual Prediction")
    st.caption(
        "This page runs both saved models directly from local artifacts. "
        "FastAPI and Ollama are optional local workflows and are not required here."
    )

    if context.load_warnings:
        st.info(" | ".join(context.load_warnings))

    feature_columns = artifacts.binary_feature_columns
    multiclass_only_features = [
        column for column in artifacts.multiclass_feature_columns if column not in feature_columns
    ]
    feature_columns.extend(multiclass_only_features)

    baseline_defaults = build_feature_defaults(
        feature_columns=feature_columns,
        base_features=context.baseline_features,
    )
    dummy_defaults = build_feature_defaults(
        feature_columns=feature_columns,
        base_features=context.dummy_features,
    )

    _initialize_feature_state(baseline_defaults)

    action_col_1, action_col_2, action_col_3 = st.columns((1, 1, 2))
    if action_col_1.button("Load baseline example"):
        _set_feature_state(baseline_defaults)
        st.rerun()

    if action_col_2.button("Load dummy example"):
        _set_feature_state(dummy_defaults)
        st.rerun()

    action_col_3.caption(
        "Dummy example is seeded from artifacts/sample_explain_payload.json when available."
    )

    grouped_features = _group_features(feature_columns)

    with st.form("prediction-form"):
        top_n_factors = st.slider(
            "Explanation detail (number of factors)",
            min_value=2,
            max_value=5,
            value=3,
            step=1,
        )
        submitted_features = _render_feature_groups(
            grouped_features=grouped_features,
            defaults=baseline_defaults,
        )
        submitted = st.form_submit_button("Run binary + multiclass prediction")

    if submitted:
        try:
            result = predict_with_deterministic_explanation(
                features=submitted_features,
                artifacts=artifacts,
                top_n_factors=top_n_factors,
            )
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
        else:
            st.session_state["streamlit_frontend_last_prediction"] = result

    cached_result = st.session_state.get("streamlit_frontend_last_prediction")
    if isinstance(cached_result, PredictionWithExplanation):
        _render_result_metrics(cached_result)
        render_explanation_section(
            prediction=cached_result.prediction,
            explanation=cached_result.explanation,
        )
