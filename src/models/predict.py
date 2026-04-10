from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse


@dataclass(frozen=True)
class PredictionResult:
    predictions: list[str]
    probabilities_by_class: dict[str, list[float]]
    positive_class_probability: list[float] | None


def load_model(model_path: Path) -> Any:
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    return joblib.load(model_path)


def load_model_metadata(metadata_path: Path) -> dict[str, Any]:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Model metadata not found: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def select_model_features(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    missing_columns = [column for column in feature_columns if column not in frame.columns]
    if missing_columns:
        raise KeyError(
            "Missing feature columns for inference: " + ", ".join(missing_columns)
        )
    return frame[feature_columns].copy()


def _decode_predictions(
    values: np.ndarray,
    *,
    label_decoder: dict[int, str] | None,
) -> list[str]:
    if label_decoder is None:
        return [str(value) for value in values.tolist()]

    decoded: list[str] = []
    for value in values.tolist():
        try:
            decoded.append(label_decoder[int(value)])
        except Exception:
            decoded.append(str(value))
    return decoded


def predict_from_frame(
    *,
    model: Any,
    frame: pd.DataFrame,
    feature_columns: list[str],
    task_type: str,
    label_decoder: dict[int, str] | None = None,
) -> PredictionResult:
    x = select_model_features(frame, feature_columns)

    raw_predictions = np.asarray(model.predict(x))
    decoded_predictions = _decode_predictions(raw_predictions, label_decoder=label_decoder)

    probabilities_by_class: dict[str, list[float]] = {}
    positive_probability: list[float] | None = None

    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(x), dtype=float)
        classes = np.asarray(getattr(model, "classes_", np.arange(probabilities.shape[1])))

        for idx, class_value in enumerate(classes.tolist()):
            class_key = str(class_value)
            if label_decoder is not None:
                try:
                    class_key = label_decoder[int(class_value)]
                except Exception:
                    class_key = str(class_value)
            probabilities_by_class[class_key] = probabilities[:, idx].tolist()

        if task_type == "binary" and probabilities.shape[1] >= 2:
            positive_index = 1
            if 1 in classes:
                positive_index = int(np.where(classes == 1)[0][0])
            positive_probability = probabilities[:, positive_index].tolist()

    return PredictionResult(
        predictions=decoded_predictions,
        probabilities_by_class=probabilities_by_class,
        positive_class_probability=positive_probability,
    )


def predict_with_artifacts(
    *,
    frame: pd.DataFrame,
    model_path: Path,
    metadata_path: Path,
) -> PredictionResult:
    model = load_model(model_path)
    metadata = load_model_metadata(metadata_path)

    feature_columns = [str(column) for column in metadata.get("feature_columns", [])]
    task_type = str(metadata.get("task_type", "binary"))

    mapping_raw = metadata.get("label_mapping", {})
    label_decoder: dict[int, str] | None = None
    if isinstance(mapping_raw, dict) and mapping_raw:
        label_decoder = {int(key): str(value) for key, value in mapping_raw.items()}

    return predict_from_frame(
        model=model,
        frame=frame,
        feature_columns=feature_columns,
        task_type=task_type,
        label_decoder=label_decoder,
    )


def _extract_transformed_matrix_and_feature_names(
    *,
    model: Any,
    x: pd.DataFrame,
) -> tuple[np.ndarray | sparse.spmatrix, list[str]]:
    if not hasattr(model, "named_steps"):
        raise TypeError("SHAP utilities expect a fitted pipeline with named_steps.")

    named_steps: dict[str, Any] = model.named_steps
    if "preprocessor" not in named_steps:
        raise KeyError("Pipeline is missing required 'preprocessor' step.")

    preprocessor = named_steps["preprocessor"]
    transformed = preprocessor.transform(x)
    feature_names = preprocessor.get_feature_names_out().tolist()

    selector = named_steps.get("feature_selector")
    support_mask = getattr(selector, "support_mask_", None)
    use_feature_selection = bool(getattr(selector, "use_feature_selection_", False))

    if use_feature_selection and support_mask is not None:
        transformed = transformed[:, support_mask]
        feature_names = [
            name for name, keep in zip(feature_names, support_mask, strict=False) if bool(keep)
        ]

    return transformed, feature_names


def _dense_if_needed(matrix: np.ndarray | sparse.spmatrix) -> np.ndarray:
    if sparse.issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix)


def _mean_abs_shap_values(shap_values: Any) -> np.ndarray:
    if isinstance(shap_values, list):
        stacked = np.stack([np.abs(np.asarray(values)) for values in shap_values], axis=0)
        return stacked.mean(axis=(0, 1))

    values = np.asarray(shap_values)
    if values.ndim == 3:
        return np.abs(values).mean(axis=(0, 1))
    if values.ndim == 2:
        return np.abs(values).mean(axis=0)
    return np.abs(values)


def generate_tree_shap_summary(
    *,
    model: Any,
    x: pd.DataFrame,
    output_dir: Path,
    run_name: str,
    top_n: int = 25,
    max_rows: int = 1000,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    warnings: list[str] = []

    if not hasattr(model, "named_steps"):
        return {}, ("SHAP skipped: expected an sklearn/imblearn pipeline.",)

    classifier = model.named_steps.get("classifier")
    if classifier is None:
        return {}, ("SHAP skipped: classifier step not found.",)

    classifier_name = classifier.__class__.__name__
    if classifier_name not in {"XGBClassifier", "RandomForestClassifier"}:
        return {}, (f"SHAP skipped: unsupported classifier {classifier_name}.",)

    try:
        import shap
    except Exception as exc:  # pragma: no cover - optional dependency
        return {}, (f"SHAP skipped: {exc}",)

    sample = x.head(max_rows).copy()
    transformed, feature_names = _extract_transformed_matrix_and_feature_names(
        model=model,
        x=sample,
    )

    if sparse.issparse(transformed):
        total_cells = transformed.shape[0] * transformed.shape[1]
        if total_cells > 8_000_000:
            return {}, ("SHAP skipped: transformed matrix too large to densify safely.",)

    dense_matrix = _dense_if_needed(transformed)

    try:
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(dense_matrix)
        mean_abs_values = _mean_abs_shap_values(shap_values)
    except Exception as exc:
        return {}, (f"SHAP computation failed: {exc}",)

    top_indices = np.argsort(mean_abs_values)[::-1][:top_n]
    top_rows: list[dict[str, Any]] = []
    for idx in top_indices.tolist():
        top_rows.append(
            {
                "feature": feature_names[idx],
                "mean_abs_shap": float(mean_abs_values[idx]),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{run_name}_shap_top_features.json"
    json_path.write_text(json.dumps(top_rows, indent=2), encoding="utf-8")

    plot_path = output_dir / f"{run_name}_shap_top_features.png"
    try:
        import matplotlib.pyplot as plt

        labels = [row["feature"] for row in top_rows][::-1]
        values = [row["mean_abs_shap"] for row in top_rows][::-1]

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(labels, values)
        ax.set_xlabel("Mean Absolute SHAP Value")
        ax.set_title("Top SHAP Feature Contributions")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
    except Exception as exc:  # pragma: no cover - optional dependency
        warnings.append(f"SHAP plot skipped: {exc}")

    artifacts: dict[str, Any] = {
        "shap_top_features_json": str(json_path),
        "shap_top_features": top_rows,
    }
    if plot_path.exists():
        artifacts["shap_top_features_plot"] = str(plot_path)

    return artifacts, tuple(warnings)


def explain_single_prediction(
    *,
    model: Any,
    row_frame: pd.DataFrame,
    top_n: int = 10,
) -> list[dict[str, float | str]]:
    if len(row_frame) != 1:
        raise ValueError("row_frame must contain exactly one row.")

    if not hasattr(model, "named_steps"):
        raise TypeError("Single-row explanation requires a fitted pipeline model.")

    classifier = model.named_steps.get("classifier")
    if classifier is None:
        raise KeyError("Pipeline classifier step is missing.")

    classifier_name = classifier.__class__.__name__
    if classifier_name not in {"XGBClassifier", "RandomForestClassifier"}:
        raise RuntimeError(f"Single-row SHAP explanation unsupported for {classifier_name}.")

    try:
        import shap
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(f"SHAP unavailable: {exc}") from exc

    transformed, feature_names = _extract_transformed_matrix_and_feature_names(
        model=model,
        x=row_frame,
    )
    dense_matrix = _dense_if_needed(transformed)

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(dense_matrix)

    if isinstance(shap_values, list):
        stacked = np.stack([np.asarray(values)[0] for values in shap_values], axis=0)
        contribution = stacked.mean(axis=0)
    else:
        values = np.asarray(shap_values)
        if values.ndim == 3:
            contribution = values[0].mean(axis=0)
        elif values.ndim == 2:
            contribution = values[0]
        else:
            contribution = values

    order = np.argsort(np.abs(contribution))[::-1][:top_n]
    output: list[dict[str, float | str]] = []
    for idx in order.tolist():
        output.append(
            {
                "feature": feature_names[idx],
                "contribution": float(contribution[idx]),
            }
        )

    return output
