from __future__ import annotations

import importlib
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

_deployment_core_module = importlib.import_module("deployment_core")
DeploymentArtifacts = _deployment_core_module.DeploymentArtifacts
DeploymentContext = _deployment_core_module.DeploymentContext
PredictionBundle = _deployment_core_module.PredictionBundle
build_feature_defaults = _deployment_core_module.build_feature_defaults
load_deployment_artifacts = _deployment_core_module.load_deployment_artifacts
load_deployment_context = _deployment_core_module.load_deployment_context
missing_required_artifacts = _deployment_core_module.missing_required_artifacts
predict_with_deterministic_explanation = (
    _deployment_core_module.predict_with_deterministic_explanation
)
resolve_existing_path = _deployment_core_module.resolve_existing_path

EMPTY_OPTION = "(missing)"

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


@st.cache_data(show_spinner=False)
def _load_context_cached() -> DeploymentContext:
    return load_deployment_context()


@st.cache_resource(show_spinner="Loading saved model artifacts...")
def _load_artifacts_cached() -> DeploymentArtifacts:
    return load_deployment_artifacts()


def _inject_custom_style() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Manrope', sans-serif;
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }

        .stAlert {
            border-radius: 10px;
        }

        h1, h2, h3 {
            letter-spacing: 0.2px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _humanize_feature_name(name: str) -> str:
    return name.replace("_", " ").replace("-", " ").strip().title()


def _format_probability(value: Any) -> str:
    try:
        probability = float(value)
    except (TypeError, ValueError):
        probability = 0.0
    return f"{probability * 100:.1f}%"


def _format_metric(value: Any, *, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def _risk_band(binary_probability: float) -> str:
    if binary_probability >= 0.60:
        return "Higher relative risk"
    if binary_probability >= 0.35:
        return "Moderate relative risk"
    return "Lower relative risk"


def _feature_widget_key(feature_name: str) -> str:
    return f"deployment_feature__{feature_name}"


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
    label = _humanize_feature_name(feature_name)
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


def _extract_test_metrics(metadata: Mapping[str, Any]) -> dict[str, Any]:
    key_metrics = metadata.get("key_evaluation_metrics", {})
    if not isinstance(key_metrics, Mapping):
        return {}
    test_metrics = key_metrics.get("test", {})
    if isinstance(test_metrics, Mapping):
        return dict(test_metrics)
    return {}


def _probability_frame(probabilities: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for label, value in probabilities.items():
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            numeric_value = 0.0
        rows.append({"class": str(label), "probability": numeric_value})

    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["class", "probability"])
    return frame.sort_values("probability", ascending=False).reset_index(drop=True)


def _render_prediction_result(bundle: PredictionBundle) -> None:
    prediction = bundle.prediction
    explanation = bundle.explanation

    st.subheader("Prediction Results")

    col_1, col_2, col_3, col_4 = st.columns(4)
    col_1.metric("Binary prediction", str(prediction.binary_prediction))
    col_2.metric("Binary probability", _format_probability(prediction.binary_probability))
    col_3.metric("Multiclass prediction", prediction.multiclass_prediction)

    top_multiclass_probability = 0.0
    if prediction.multiclass_probabilities:
        top_multiclass_probability = max(prediction.multiclass_probabilities.values())
    col_4.metric("Top multiclass probability", _format_probability(top_multiclass_probability))

    probabilities = _probability_frame(prediction.multiclass_probabilities)
    if not probabilities.empty:
        st.markdown("**Multiclass probability distribution**")
        st.bar_chart(probabilities.set_index("class"))

        display_frame = probabilities.copy()
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

    st.subheader("Prediction Explanation")
    exp_col_1, exp_col_2 = st.columns(2)
    exp_col_1.metric("Risk band", _risk_band(prediction.binary_probability))
    exp_col_2.metric("Explanation mode", explanation.explanation_mode)

    st.write(explanation.explanation_text)

    factors_col_1, factors_col_2 = st.columns(2)
    with factors_col_1:
        st.markdown("**Top risk-increasing signals**")
        for item in explanation.top_risk_increasing_factors:
            st.write(f"- {item}")

    with factors_col_2:
        st.markdown("**Top risk-decreasing signals**")
        for item in explanation.top_risk_decreasing_factors:
            st.write(f"- {item}")

    st.warning(
        "Portfolio/demo disclaimer: this model output is for research and engineering discussion "
        "only. It is not medical advice and must not be used for patient care decisions."
    )


def _render_prediction_page(context: DeploymentContext, artifacts: DeploymentArtifacts) -> None:
    st.header("Manual Prediction")
    st.caption(
        "This deployment app loads committed model artifacts directly from the repository. "
        "FastAPI and Ollama are optional local workflows and are not required here."
    )

    if context.load_warnings:
        st.info(" | ".join(context.load_warnings))

    feature_columns = list(artifacts.binary_feature_columns)
    for feature_name in artifacts.multiclass_feature_columns:
        if feature_name not in feature_columns:
            feature_columns.append(feature_name)

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

    with st.form("deployment-prediction-form"):
        top_n_factors = st.slider(
            "Explanation detail (number of factors)",
            min_value=2,
            max_value=5,
            value=3,
            step=1,
        )

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
                            baseline_defaults.get(feature_name),
                        )

        submitted = st.form_submit_button("Run binary + multiclass prediction")

    if submitted:
        try:
            bundle = predict_with_deterministic_explanation(
                features=submitted_features,
                artifacts=artifacts,
                top_n_factors=top_n_factors,
            )
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
        else:
            st.session_state["deployment_last_prediction"] = bundle

    cached_bundle = st.session_state.get("deployment_last_prediction")
    if isinstance(cached_bundle, PredictionBundle):
        _render_prediction_result(cached_bundle)


def _render_confusion_matrix(
    *,
    title: str,
    matrix_value: Any,
    labels: list[str],
) -> None:
    if not isinstance(matrix_value, list) or not matrix_value:
        return

    try:
        matrix_frame = pd.DataFrame(matrix_value, index=labels, columns=labels)
    except Exception:
        return

    st.markdown(f"**{title}**")
    st.dataframe(matrix_frame, use_container_width=True)


def _collect_plot_paths(artifacts: DeploymentArtifacts) -> list[tuple[str, Path]]:
    candidates: list[tuple[str, Path]] = []

    def _add_if_exists(label: str, path_value: Any) -> None:
        path = resolve_existing_path(path_value, project_root=artifacts.paths.project_root)
        if path is not None and path.exists():
            candidates.append((label, path))

    for task_name, metadata in (
        ("binary", artifacts.binary_metadata),
        ("multiclass", artifacts.multiclass_metadata),
    ):
        shap_artifacts = metadata.get("shap_artifacts", {})
        if isinstance(shap_artifacts, Mapping):
            _add_if_exists(
                f"{task_name.title()} SHAP Top Features",
                shap_artifacts.get("shap_top_features_plot"),
            )

        evaluation_dir = resolve_existing_path(
            metadata.get("evaluation_dir"),
            project_root=artifacts.paths.project_root,
        )
        if evaluation_dir is None:
            continue

        for file_name in (
            "test_confusion_matrix.png",
            "val_confusion_matrix.png",
            "test_pr_curve.png",
            "val_pr_curve.png",
            "test_calibration_curve.png",
            "val_calibration_curve.png",
        ):
            candidate = evaluation_dir / file_name
            if candidate.exists():
                label = f"{task_name.title()} {file_name.replace('_', ' ').replace('.png', '')}"
                candidates.append((label, candidate))

    unique: dict[str, Path] = {}
    for label, path in candidates:
        unique[f"{label}:{path}"] = path

    output: list[tuple[str, Path]] = []
    for key, path in unique.items():
        output.append((key.split(":", maxsplit=1)[0], path))
    return output


def _render_analytics_page(
    context: DeploymentContext,
    artifacts: DeploymentArtifacts | None,
) -> None:
    st.header("Model Analytics")
    st.caption("Metrics and summaries are loaded from committed local report artifacts.")

    if artifacts is None:
        st.error(
            "Model artifacts are unavailable, so prediction metrics cannot be shown. "
            "Run training/evaluation locally and commit required artifacts."
        )
    else:
        binary_test = _extract_test_metrics(artifacts.binary_metadata)
        multiclass_test = _extract_test_metrics(artifacts.multiclass_metadata)

        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Binary F1", _format_metric(binary_test.get("f1"), digits=4))
        col_b.metric(
            "Binary positive rate",
            _format_metric(binary_test.get("positive_rate"), digits=4),
        )
        col_c.metric(
            "Multiclass macro F1",
            _format_metric(multiclass_test.get("macro_f1"), digits=4),
        )
        col_d.metric(
            "Multiclass accuracy",
            _format_metric(multiclass_test.get("accuracy"), digits=4),
        )

        _render_confusion_matrix(
            title="Binary confusion matrix (test)",
            matrix_value=binary_test.get("confusion_matrix"),
            labels=["0", "1"],
        )
        _render_confusion_matrix(
            title="Multiclass confusion matrix (test)",
            matrix_value=multiclass_test.get("confusion_matrix"),
            labels=["NO", ">30", "<30"],
        )

        per_class = multiclass_test.get("per_class", {})
        if isinstance(per_class, Mapping) and per_class:
            rows: list[dict[str, Any]] = []
            for label, metrics in per_class.items():
                if not isinstance(metrics, Mapping):
                    continue
                rows.append(
                    {
                        "class": str(label),
                        "precision": metrics.get("precision"),
                        "recall": metrics.get("recall"),
                        "f1": metrics.get("f1"),
                        "support": metrics.get("support"),
                    }
                )
            if rows:
                st.markdown("**Multiclass per-class metrics (test split)**")
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        for task_name, metadata in (
            ("Binary", artifacts.binary_metadata),
            ("Multiclass", artifacts.multiclass_metadata),
        ):
            shap_artifacts = metadata.get("shap_artifacts", {})
            if not isinstance(shap_artifacts, Mapping):
                continue
            top_features = shap_artifacts.get("shap_top_features", [])
            if isinstance(top_features, list) and top_features:
                st.markdown(f"**{task_name} SHAP top features**")
                st.dataframe(pd.DataFrame(top_features), use_container_width=True, hide_index=True)

        plot_candidates = _collect_plot_paths(artifacts)
        if plot_candidates:
            st.markdown("**Selected evaluation plots**")
            for label, path in plot_candidates[:8]:
                st.image(str(path), caption=label, use_container_width=True)

    st.subheader("Model comparison summary")
    if context.model_comparison_report_text:
        st.markdown(context.model_comparison_report_text)
    else:
        st.info("reports/model_comparison_report.md is missing in this repository state.")


def _render_monitoring_page(context: DeploymentContext) -> None:
    st.header("Monitoring")
    st.caption("Current monitoring summary is loaded from reports/monitoring_summary.json.")

    summary = context.monitoring_summary
    if not isinstance(summary, Mapping):
        st.info(
            "Monitoring outputs are missing. Generate them locally with "
            "uv run python scripts/run_monitoring_report.py."
        )
        return

    drift_payload = summary.get("binary_probability_drift", {})
    drift_status = "n/a"
    drift_psi = None
    if isinstance(drift_payload, Mapping):
        drift_status = str(drift_payload.get("status", "n/a"))
        drift_psi = drift_payload.get("psi")

    generated_at = summary.get("generated_at_utc", "n/a")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Generated at (UTC)", str(generated_at))
    col_b.metric("Binary drift status", drift_status)
    col_c.metric("Binary drift PSI", _format_metric(drift_psi, digits=6))

    sample_sizes = summary.get("sample_sizes", {})
    if isinstance(sample_sizes, Mapping):
        sample_frame = pd.DataFrame(
            [{"name": str(name), "value": value} for name, value in sample_sizes.items()]
        )
        st.markdown("**Sample sizes**")
        st.dataframe(sample_frame, use_container_width=True, hide_index=True)

    prediction_distribution = summary.get("prediction_distribution", {})
    if isinstance(prediction_distribution, Mapping):
        binary_rates = prediction_distribution.get("binary_prediction_rates", {})
        multiclass_rates = prediction_distribution.get("multiclass_prediction_rates", {})

        if isinstance(binary_rates, Mapping) and binary_rates:
            binary_frame = pd.DataFrame(
                [{"label": str(label), "rate": float(rate)} for label, rate in binary_rates.items()]
            )
            st.markdown("**Binary prediction rate distribution**")
            st.bar_chart(binary_frame.set_index("label"))

        if isinstance(multiclass_rates, Mapping) and multiclass_rates:
            multi_frame = pd.DataFrame(
                [
                    {"label": str(label), "rate": float(rate)}
                    for label, rate in multiclass_rates.items()
                ]
            )
            st.markdown("**Multiclass prediction rate distribution**")
            st.bar_chart(multi_frame.set_index("label"))

    feature_drift = summary.get("feature_drift", {})
    if isinstance(feature_drift, Mapping) and feature_drift:
        rows: list[dict[str, Any]] = []
        for feature_name, payload in feature_drift.items():
            if not isinstance(payload, Mapping):
                continue
            rows.append(
                {
                    "feature": str(feature_name),
                    "status": payload.get("status"),
                    "psi": payload.get("psi"),
                    "absolute_mean_shift": payload.get("absolute_mean_shift"),
                    "reference_mean": payload.get("reference_mean"),
                    "current_mean": payload.get("current_mean"),
                }
            )
        if rows:
            st.markdown("**Feature drift details**")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    narrative = summary.get("monitoring_narrative")
    if narrative:
        st.markdown("**Monitoring narrative**")
        st.write(str(narrative))

    warnings = summary.get("warnings", [])
    if isinstance(warnings, list) and warnings:
        st.warning(" | ".join(str(item) for item in warnings))

    if context.monitoring_report_text:
        with st.expander("Monitoring report markdown", expanded=False):
            st.markdown(context.monitoring_report_text)


def _render_project_overview(
    context: DeploymentContext,
    artifacts_ready: bool,
    artifact_error: str | None,
) -> None:
    st.header("Project Overview")

    st.markdown(
        """
        This app is the deployment-focused Streamlit frontend for the Diabetes Hospital
        Readmission Pipeline.

        It demonstrates:
        - direct loading of committed binary and multiclass model artifacts
        - deterministic explanation fallback with no Ollama dependency
        - analytics and monitoring summaries from committed report artifacts

        It is a technical portfolio demo and not a clinical decision-support system.
        """
    )

    st.subheader("Deployment behavior")
    st.markdown(
        """
        - Entry file: app/streamlit_app.py
        - Dependency file: app/requirements.txt
        - Root uv.lock is retained for local development workflows
        - Public app does not require FastAPI or Ollama
        """
    )

    if artifacts_ready:
        st.success("Required prediction artifacts are available.")
    else:
        st.error(
            "Required artifacts are missing for prediction. Commit model artifacts and metadata "
            "for public app usage."
        )
        if artifact_error:
            st.code(artifact_error)

    if context.readme_excerpt:
        with st.expander("README excerpt", expanded=False):
            st.markdown(context.readme_excerpt)


def main() -> None:
    st.set_page_config(
        page_title="Hospital Readmission Deployment App",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_custom_style()

    st.title("Hospital Readmission Models: Deployment Frontend")
    st.caption(
        "Community Cloud-ready app entrypoint using app/requirements.txt with a minimal runtime "
        "dependency set."
    )

    context = _load_context_cached()

    artifacts: DeploymentArtifacts | None = None
    artifact_error: str | None = None
    try:
        artifacts = _load_artifacts_cached()
    except Exception as exc:
        artifact_error = str(exc)

    page = st.sidebar.radio(
        "Navigate",
        options=("Prediction", "Analytics", "Monitoring", "Project Overview"),
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("Deployment target: Streamlit Community Cloud")

    if artifacts is None:
        st.sidebar.warning("Model artifacts are not loaded.")
        missing = missing_required_artifacts(context.paths)
        if missing:
            st.sidebar.write("Missing files:")
            for path in missing:
                st.sidebar.write(f"- {path}")
    else:
        st.sidebar.success("Model artifacts loaded")

    if page == "Prediction":
        if artifacts is None:
            st.error("Prediction page requires committed model artifacts and metadata.")
            if artifact_error:
                st.code(artifact_error)
            return
        _render_prediction_page(context, artifacts)
        return

    if page == "Analytics":
        _render_analytics_page(context, artifacts)
        return

    if page == "Monitoring":
        _render_monitoring_page(context)
        return

    _render_project_overview(
        context=context,
        artifacts_ready=artifacts is not None,
        artifact_error=artifact_error,
    )


if __name__ == "__main__":
    main()
