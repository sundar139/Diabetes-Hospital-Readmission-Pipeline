from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

TaskType = Literal["binary", "multiclass"]


@dataclass(frozen=True)
class EvaluationResult:
    task_type: TaskType
    metrics: dict[str, Any]
    metrics_path: Path
    artifact_paths: tuple[Path, ...]
    warnings: tuple[str, ...]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(data), indent=2, sort_keys=True), encoding="utf-8")


def _plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str],
    path: Path,
    *,
    title: str,
    warnings: list[str],
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        warnings.append(f"Confusion matrix plot skipped: {exc}")
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)

    for row_idx in range(cm.shape[0]):
        for col_idx in range(cm.shape[1]):
            ax.text(col_idx, row_idx, int(cm[row_idx, col_idx]), ha="center", va="center")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_pr_curve(
    recall_values: np.ndarray,
    precision_values: np.ndarray,
    path: Path,
    *,
    warnings: list[str],
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        warnings.append(f"PR curve plot skipped: {exc}")
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall_values, precision_values, linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_calibration_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    path: Path,
    *,
    warnings: list[str],
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        warnings.append(f"Calibration plot skipped: {exc}")
        return

    try:
        frac_pos, mean_pred = calibration_curve(y_true, y_score, n_bins=10)
    except Exception as exc:
        warnings.append(f"Calibration curve skipped: {exc}")
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(mean_pred, frac_pos, marker="o", linewidth=2, label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate_binary_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    *,
    output_dir: Path,
    run_name: str,
    create_calibration_plot: bool = True,
) -> EvaluationResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    roc_auc: float | None
    try:
        roc_auc = float(roc_auc_score(y_true, y_score))
    except Exception as exc:
        roc_auc = None
        warnings.append(f"ROC-AUC unavailable: {exc}")

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(y_true, y_score)

    metrics: dict[str, Any] = {
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm.tolist(),
        "pr_curve": {
            "precision": pr_precision.tolist(),
            "recall": pr_recall.tolist(),
            "thresholds": pr_thresholds.tolist(),
        },
        "n_samples": int(len(y_true)),
        "positive_rate": float(np.mean(y_true == 1)),
    }

    metrics_path = output_dir / f"{run_name}_metrics.json"
    _write_json(metrics, metrics_path)

    artifacts: list[Path] = [metrics_path]

    confusion_path = output_dir / f"{run_name}_confusion_matrix.png"
    _plot_confusion_matrix(
        cm,
        labels=["0", "1"],
        path=confusion_path,
        title="Binary Confusion Matrix",
        warnings=warnings,
    )
    if confusion_path.exists():
        artifacts.append(confusion_path)

    pr_path = output_dir / f"{run_name}_pr_curve.png"
    _plot_pr_curve(pr_recall, pr_precision, pr_path, warnings=warnings)
    if pr_path.exists():
        artifacts.append(pr_path)

    if create_calibration_plot:
        calibration_path = output_dir / f"{run_name}_calibration_curve.png"
        _plot_calibration_curve(y_true, y_score, calibration_path, warnings=warnings)
        if calibration_path.exists():
            artifacts.append(calibration_path)

    return EvaluationResult(
        task_type="binary",
        metrics=metrics,
        metrics_path=metrics_path,
        artifact_paths=tuple(artifacts),
        warnings=tuple(warnings),
    )


def evaluate_multiclass_predictions(
    y_true: pd.Series,
    y_pred: pd.Series,
    *,
    class_labels: tuple[str, ...],
    output_dir: Path,
    run_name: str,
) -> EvaluationResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    cm = confusion_matrix(y_true, y_pred, labels=list(class_labels))
    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    report = classification_report(
        y_true,
        y_pred,
        labels=list(class_labels),
        output_dict=True,
        zero_division=0,
    )

    per_class: dict[str, dict[str, float]] = {}
    for class_label in class_labels:
        class_metrics = report.get(class_label, {})
        per_class[class_label] = {
            "precision": float(class_metrics.get("precision", 0.0)),
            "recall": float(class_metrics.get("recall", 0.0)),
            "f1": float(class_metrics.get("f1-score", 0.0)),
            "support": float(class_metrics.get("support", 0.0)),
        }

    metrics: dict[str, Any] = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "n_samples": int(len(y_true)),
    }

    metrics_path = output_dir / f"{run_name}_metrics.json"
    _write_json(metrics, metrics_path)

    artifacts: list[Path] = [metrics_path]

    confusion_path = output_dir / f"{run_name}_confusion_matrix.png"
    _plot_confusion_matrix(
        cm,
        labels=list(class_labels),
        path=confusion_path,
        title="Multiclass Confusion Matrix",
        warnings=warnings,
    )
    if confusion_path.exists():
        artifacts.append(confusion_path)

    return EvaluationResult(
        task_type="multiclass",
        metrics=metrics,
        metrics_path=metrics_path,
        artifact_paths=tuple(artifacts),
        warnings=tuple(warnings),
    )


def _positive_class_probability(model: Any, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x)
        if probabilities.ndim == 2 and probabilities.shape[1] >= 2:
            classes = getattr(model, "classes_", np.array([0, 1]))
            classes_array = np.asarray(classes)
            if 1 in classes_array:
                positive_index = int(np.where(classes_array == 1)[0][0])
            else:
                positive_index = int(probabilities.shape[1] - 1)
            return probabilities[:, positive_index]

    if hasattr(model, "decision_function"):
        decision_values = model.decision_function(x)
        return 1.0 / (1.0 + np.exp(-np.asarray(decision_values, dtype=float)))

    return np.asarray(model.predict(x), dtype=float)


def evaluate_model(
    *,
    model: Any,
    x: pd.DataFrame,
    y: pd.Series,
    task_type: TaskType,
    output_dir: Path,
    run_name: str,
    class_labels: tuple[str, ...] | None = None,
    label_decoder: dict[int, str] | None = None,
) -> EvaluationResult:
    if task_type == "binary":
        y_true = pd.to_numeric(y, errors="coerce").fillna(0).astype("int32").to_numpy()
        y_pred = np.asarray(model.predict(x), dtype=int)
        y_score = _positive_class_probability(model, x)
        return evaluate_binary_predictions(
            y_true,
            y_pred,
            y_score,
            output_dir=output_dir,
            run_name=run_name,
            create_calibration_plot=True,
        )

    y_pred_raw = pd.Series(model.predict(x), index=y.index)
    y_true_raw = y.copy()

    if label_decoder is not None:
        if pd.api.types.is_integer_dtype(y_true_raw):
            y_true_eval = y_true_raw.map(label_decoder).astype("string")
        else:
            y_true_eval = y_true_raw.astype("string")

        if pd.api.types.is_integer_dtype(y_pred_raw):
            y_pred_eval = y_pred_raw.map(label_decoder).astype("string")
        else:
            y_pred_eval = y_pred_raw.astype("string")
    else:
        y_true_eval = y_true_raw.astype("string")
        y_pred_eval = y_pred_raw.astype("string")

    labels = class_labels
    if labels is None:
        labels = tuple(sorted({str(value) for value in y_true_eval.dropna().unique().tolist()}))

    return evaluate_multiclass_predictions(
        y_true_eval,
        y_pred_eval,
        class_labels=labels,
        output_dir=output_dir,
        run_name=run_name,
    )
