from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.frontend.loaders import FrontendContext, PredictionArtifacts
from src.frontend.utils import format_metric, resolve_existing_path


def _extract_test_metrics(metadata: Mapping[str, Any]) -> dict[str, Any]:
    key_metrics = metadata.get("key_evaluation_metrics", {})
    if not isinstance(key_metrics, Mapping):
        return {}
    test_metrics = key_metrics.get("test", {})
    if isinstance(test_metrics, Mapping):
        return dict(test_metrics)
    return {}


def _render_primary_metrics(artifacts: PredictionArtifacts) -> None:
    binary_test = _extract_test_metrics(artifacts.binary_metadata)
    multiclass_test = _extract_test_metrics(artifacts.multiclass_metadata)

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Binary F1", format_metric(binary_test.get("f1"), digits=4))
    col_b.metric("Binary positive rate", format_metric(binary_test.get("positive_rate"), digits=4))
    col_c.metric("Multiclass macro F1", format_metric(multiclass_test.get("macro_f1"), digits=4))
    col_d.metric("Multiclass accuracy", format_metric(multiclass_test.get("accuracy"), digits=4))


def _render_multiclass_per_class_table(artifacts: PredictionArtifacts) -> None:
    multiclass_test = _extract_test_metrics(artifacts.multiclass_metadata)
    per_class = multiclass_test.get("per_class", {})
    if not isinstance(per_class, Mapping) or not per_class:
        st.info("Per-class metrics are not available in multiclass metadata.")
        return

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

    if not rows:
        st.info("Per-class metrics were present but could not be parsed.")
        return

    frame = pd.DataFrame(rows)
    st.markdown("**Multiclass per-class metrics (test split)**")
    st.dataframe(frame, use_container_width=True, hide_index=True)


def _render_confusion_matrix_table(
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


def _collect_candidate_plot_paths(artifacts: PredictionArtifacts) -> list[tuple[str, Path]]:
    candidates: list[tuple[str, Path]] = []

    def add_if_exists(label: str, path_value: Any) -> None:
        path = resolve_existing_path(path_value, project_root=artifacts.paths.project_root)
        if path is not None and path.exists():
            candidates.append((label, path))

    for task_name, metadata in (
        ("binary", artifacts.binary_metadata),
        ("multiclass", artifacts.multiclass_metadata),
    ):
        shap_artifacts = metadata.get("shap_artifacts", {})
        if isinstance(shap_artifacts, Mapping):
            add_if_exists(
                f"{task_name.title()} SHAP Top Features",
                shap_artifacts.get("shap_top_features_plot"),
            )

        evaluation_dir = resolve_existing_path(
            metadata.get("evaluation_dir"),
            project_root=artifacts.paths.project_root,
        )
        if evaluation_dir is None:
            continue

        for filename in (
            "test_confusion_matrix.png",
            "val_confusion_matrix.png",
            "test_pr_curve.png",
            "val_pr_curve.png",
            "test_calibration_curve.png",
            "val_calibration_curve.png",
        ):
            candidate = evaluation_dir / filename
            if candidate.exists():
                label = f"{task_name.title()} {filename.replace('_', ' ').replace('.png', '')}"
                candidates.append((label, candidate))

    unique: dict[str, Path] = {}
    for label, path in candidates:
        unique[f"{label}:{path}"] = path

    deduped: list[tuple[str, Path]] = []
    for key, path in unique.items():
        deduped.append((key.split(":", maxsplit=1)[0], path))
    return deduped


def _render_shap_tables(artifacts: PredictionArtifacts) -> None:
    for task_name, metadata in (
        ("Binary", artifacts.binary_metadata),
        ("Multiclass", artifacts.multiclass_metadata),
    ):
        shap_artifacts = metadata.get("shap_artifacts", {})
        if not isinstance(shap_artifacts, Mapping):
            continue

        top_features = shap_artifacts.get("shap_top_features", [])
        if not isinstance(top_features, list) or not top_features:
            continue

        frame = pd.DataFrame(top_features)
        if frame.empty:
            continue

        st.markdown(f"**{task_name} SHAP top features**")
        st.dataframe(frame, use_container_width=True, hide_index=True)


def render_analytics_page(
    *,
    context: FrontendContext,
    artifacts: PredictionArtifacts | None,
) -> None:
    st.header("Model Analytics")
    st.caption("Metrics and artifact summaries are loaded from local reports and metadata files.")

    if artifacts is None:
        st.error(
            "Model artifacts are unavailable, so prediction metrics cannot be rendered. "
            "Train and evaluate models to populate analytics content."
        )
    else:
        _render_primary_metrics(artifacts)

        binary_test = _extract_test_metrics(artifacts.binary_metadata)
        multiclass_test = _extract_test_metrics(artifacts.multiclass_metadata)

        _render_confusion_matrix_table(
            title="Binary confusion matrix (test)",
            matrix_value=binary_test.get("confusion_matrix"),
            labels=["0", "1"],
        )
        _render_confusion_matrix_table(
            title="Multiclass confusion matrix (test)",
            matrix_value=multiclass_test.get("confusion_matrix"),
            labels=["NO", ">30", "<30"],
        )

        _render_multiclass_per_class_table(artifacts)
        _render_shap_tables(artifacts)

        plot_candidates = _collect_candidate_plot_paths(artifacts)
        if plot_candidates:
            st.markdown("**Selected evaluation plots**")
            for label, path in plot_candidates[:8]:
                st.image(str(path), caption=label, use_container_width=True)
        else:
            st.info("No evaluation image artifacts were found in the current workspace.")

    st.subheader("Model Comparison Summary")
    if context.model_comparison_report_text:
        st.markdown(context.model_comparison_report_text)
    else:
        st.info(
            "model_comparison_report.md was not found. Run scripts/run_evaluation.py "
            "to regenerate it."
        )
