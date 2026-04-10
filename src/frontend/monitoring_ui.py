from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd
import streamlit as st

from src.frontend.loaders import FrontendContext
from src.frontend.utils import format_metric


def _render_prediction_distribution(distribution: Mapping[str, Any]) -> None:
    binary_rates = distribution.get("binary_prediction_rates", {})
    multiclass_rates = distribution.get("multiclass_prediction_rates", {})

    if isinstance(binary_rates, Mapping) and binary_rates:
        binary_frame = pd.DataFrame(
            [{"label": str(label), "rate": float(rate)} for label, rate in binary_rates.items()]
        )
        st.markdown("**Binary prediction rate distribution**")
        st.bar_chart(binary_frame.set_index("label"))

    if isinstance(multiclass_rates, Mapping) and multiclass_rates:
        multiclass_frame = pd.DataFrame(
            [{"label": str(label), "rate": float(rate)} for label, rate in multiclass_rates.items()]
        )
        st.markdown("**Multiclass prediction rate distribution**")
        st.bar_chart(multiclass_frame.set_index("label"))


def _render_feature_drift_table(feature_drift: Mapping[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for feature_name, drift_payload in feature_drift.items():
        if not isinstance(drift_payload, Mapping):
            continue
        rows.append(
            {
                "feature": feature_name,
                "status": drift_payload.get("status"),
                "psi": drift_payload.get("psi"),
                "absolute_mean_shift": drift_payload.get("absolute_mean_shift"),
                "reference_mean": drift_payload.get("reference_mean"),
                "current_mean": drift_payload.get("current_mean"),
            }
        )

    if not rows:
        st.info("Feature drift details are unavailable in monitoring summary.")
        return

    frame = pd.DataFrame(rows).sort_values("psi", ascending=False)
    st.markdown("**Feature drift details**")
    st.dataframe(frame, use_container_width=True, hide_index=True)


def render_monitoring_page(*, context: FrontendContext) -> None:
    st.header("Monitoring")
    st.caption("Current monitoring summary is loaded from reports/monitoring_summary.json.")

    summary = context.monitoring_summary
    if not isinstance(summary, Mapping):
        st.info(
            "Monitoring outputs are missing. Generate them with: "
            "uv run python scripts/run_monitoring_report.py"
        )
        return

    drift_payload = summary.get("binary_probability_drift", {})
    drift_status = "n/a"
    drift_psi = None
    if isinstance(drift_payload, Mapping):
        drift_status = str(drift_payload.get("status", "n/a"))
        drift_psi = drift_payload.get("psi")

    generated_at = summary.get("generated_at_utc", "n/a")
    sample_sizes = summary.get("sample_sizes", {})
    prediction_distribution = summary.get("prediction_distribution", {})

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Generated at (UTC)", str(generated_at))
    col_b.metric("Binary drift status", drift_status)
    col_c.metric("Binary drift PSI", format_metric(drift_psi, digits=6))

    if isinstance(sample_sizes, Mapping):
        sample_frame = pd.DataFrame(
            [{"name": str(name), "value": value} for name, value in sample_sizes.items()]
        )
        st.markdown("**Sample sizes**")
        st.dataframe(sample_frame, use_container_width=True, hide_index=True)

    if isinstance(prediction_distribution, Mapping):
        _render_prediction_distribution(prediction_distribution)

    feature_drift = summary.get("feature_drift", {})
    if isinstance(feature_drift, Mapping):
        _render_feature_drift_table(feature_drift)

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
