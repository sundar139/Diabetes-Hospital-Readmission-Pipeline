from __future__ import annotations

import numpy as np
import pandas as pd
from src.models.evaluate import (
    evaluate_binary_predictions,
    evaluate_multiclass_predictions,
)


def test_binary_evaluation_contains_required_metrics(tmp_path) -> None:
    y_true = np.array([0, 1, 0, 1, 1, 0, 0, 1], dtype=int)
    y_pred = np.array([0, 1, 0, 0, 1, 0, 1, 1], dtype=int)
    y_score = np.array([0.12, 0.85, 0.30, 0.40, 0.76, 0.22, 0.63, 0.88], dtype=float)

    result = evaluate_binary_predictions(
        y_true,
        y_pred,
        y_score,
        output_dir=tmp_path,
        run_name="binary_unit",
        create_calibration_plot=True,
    )

    assert "roc_auc" in result.metrics
    assert "precision" in result.metrics
    assert "recall" in result.metrics
    assert "f1" in result.metrics
    assert "confusion_matrix" in result.metrics
    assert "pr_curve" in result.metrics
    assert result.metrics_path.exists()


def test_multiclass_evaluation_contains_required_metrics(tmp_path) -> None:
    y_true = pd.Series(["NO", ">30", "<30", "NO", "<30", ">30", "NO"])
    y_pred = pd.Series(["NO", ">30", "NO", "NO", "<30", ">30", "<30"])

    result = evaluate_multiclass_predictions(
        y_true,
        y_pred,
        class_labels=("NO", ">30", "<30"),
        output_dir=tmp_path,
        run_name="multiclass_unit",
    )

    assert "accuracy" in result.metrics
    assert "macro_f1" in result.metrics
    assert "per_class" in result.metrics
    assert "confusion_matrix" in result.metrics
    assert set(result.metrics["per_class"].keys()) == {"NO", ">30", "<30"}
    assert result.metrics_path.exists()
