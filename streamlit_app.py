from __future__ import annotations

import streamlit as st
from src.frontend.analytics_ui import render_analytics_page
from src.frontend.loaders import (
    FrontendContext,
    PredictionArtifacts,
    load_frontend_context,
    load_prediction_artifacts,
    missing_prediction_artifacts,
)
from src.frontend.monitoring_ui import render_monitoring_page
from src.frontend.prediction_ui import render_prediction_page


@st.cache_data(show_spinner=False)
def _load_context_cached() -> FrontendContext:
    return load_frontend_context()


@st.cache_resource(show_spinner="Loading saved model artifacts...")
def _load_prediction_artifacts_cached() -> PredictionArtifacts:
    return load_prediction_artifacts()


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


def _render_project_overview(
    *,
    context: FrontendContext,
    artifacts_ready: bool,
    artifact_error: str | None,
) -> None:
    st.header("Project Overview")

    st.markdown(
        """
        This app is a portfolio frontend for the Diabetes Hospital Readmission Pipeline.

        It demonstrates:
        - direct loading of saved binary and multiclass model artifacts
        - deterministic explanation behavior for public deployment
        - analytics and monitoring summaries from generated report artifacts

        It is a technical demo and not a clinical decision-support system.
        """
    )

    st.subheader("Pipeline at a glance")
    st.write(
        "Data validation -> leakage-aware preprocessing -> feature engineering -> "
        "model training/evaluation -> local monitoring summaries -> frontend demo app"
    )

    st.subheader("Model and stack")
    st.markdown(
        """
        - Binary model: XGBoost pipeline for 30-day readmission risk
        - Multiclass model: XGBoost pipeline for NO / >30 / <30 classes
        - Local stack: Python, pandas, scikit-learn, XGBoost, Streamlit, MLflow, FastAPI
        """
    )

    st.subheader("Deployment notes")
    st.markdown(
        """
        - Streamlit deployment target: streamlit_app.py
        - Public app flow does not require a running FastAPI server
        - Public app flow does not require local Ollama
        - FastAPI and Ollama remain available for advanced local workflows
        """
    )

    if artifacts_ready:
        st.success("Required prediction artifacts are available for interactive scoring.")
    else:
        st.error(
            "Prediction artifacts are missing or failed to load. "
            "Run training/evaluation scripts locally before using prediction features."
        )
        if artifact_error:
            st.code(artifact_error)

    with st.expander("Repository references", expanded=False):
        st.markdown(
            """
            - README.md
            - docs/local_workflow.md
            - docs/results_summary.md
            - docs/troubleshooting.md
            - docs/release_checklist.md
            """
        )

    if context.readme_excerpt:
        with st.expander("README excerpt", expanded=False):
            st.markdown(context.readme_excerpt)


def main() -> None:
    st.set_page_config(
        page_title="Hospital Readmission Portfolio Frontend",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_custom_style()

    st.title("Hospital Readmission Models: Portfolio Frontend")
    st.caption(
        "Public demo app that loads saved model artifacts directly from this repository. "
        "No FastAPI or Ollama dependency is required for core predictions."
    )

    context = _load_context_cached()

    prediction_artifacts: PredictionArtifacts | None = None
    artifact_error: str | None = None
    try:
        prediction_artifacts = _load_prediction_artifacts_cached()
    except Exception as exc:
        artifact_error = str(exc)

    page = st.sidebar.radio(
        "Navigate",
        options=("Prediction", "Analytics", "Monitoring", "Project Overview"),
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("Deployment target: Streamlit Community Cloud")

    if prediction_artifacts is None:
        st.sidebar.warning("Model artifacts are not loaded.")
        missing = missing_prediction_artifacts(context.paths)
        if missing:
            st.sidebar.write("Missing files:")
            for path in missing:
                st.sidebar.write(f"- {path}")
    else:
        st.sidebar.success("Model artifacts loaded")

    if page == "Prediction":
        if prediction_artifacts is None:
            st.error(
                "Prediction page needs trained model artifacts. "
                "Run scripts/train_binary.py, scripts/train_multiclass.py, "
                "and scripts/run_evaluation.py."
            )
            if artifact_error:
                st.code(artifact_error)
            return

        render_prediction_page(context=context, artifacts=prediction_artifacts)
        return

    if page == "Analytics":
        render_analytics_page(context=context, artifacts=prediction_artifacts)
        return

    if page == "Monitoring":
        render_monitoring_page(context=context)
        return

    _render_project_overview(
        context=context,
        artifacts_ready=prediction_artifacts is not None,
        artifact_error=artifact_error,
    )


if __name__ == "__main__":
    main()
