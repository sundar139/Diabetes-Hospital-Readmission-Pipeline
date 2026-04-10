from __future__ import annotations

import streamlit as st

from src.frontend.prediction_engine import DeterministicExplanation, FrontendPrediction
from src.frontend.utils import format_probability


def _risk_band(binary_probability: float) -> str:
    if binary_probability >= 0.60:
        return "Higher relative risk"
    if binary_probability >= 0.35:
        return "Moderate relative risk"
    return "Lower relative risk"


def render_explanation_section(
    *,
    prediction: FrontendPrediction,
    explanation: DeterministicExplanation,
) -> None:
    st.subheader("Prediction Explanation")

    col_risk, col_mode = st.columns(2)
    col_risk.metric("Risk band", _risk_band(prediction.binary_probability))
    col_mode.metric("Explanation mode", explanation.explanation_mode)

    st.write(explanation.explanation_text)

    factors_col_up, factors_col_down = st.columns(2)
    with factors_col_up:
        st.markdown("**Top risk-increasing signals**")
        for item in explanation.top_risk_increasing_factors:
            st.write(f"- {item}")

    with factors_col_down:
        st.markdown("**Top risk-decreasing signals**")
        for item in explanation.top_risk_decreasing_factors:
            st.write(f"- {item}")

    st.caption(
        "Prediction summary: "
        f"binary probability {format_probability(prediction.binary_probability)}; "
        f"multiclass label {prediction.multiclass_prediction}."
    )

    if explanation.warnings:
        st.info("Additional notes: " + " | ".join(explanation.warnings))

    st.warning(
        "Portfolio/demo disclaimer: this model output is for research and engineering discussion "
        "only. It is not medical advice and must not be used for patient care decisions."
    )
