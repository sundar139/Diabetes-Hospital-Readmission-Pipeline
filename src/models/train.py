from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier

from src.config.settings import Settings, get_settings
from src.models.evaluate import evaluate_model
from src.models.pipeline_factory import (
    ModelFamily,
    TaskType,
    build_estimator,
    build_preprocessor,
    estimator_params_for_logging,
    resolve_feature_columns,
    resolve_xgboost_runtime_device,
)
from src.models.predict import generate_tree_shap_summary

SamplingStrategy = Literal["none", "over", "under"]
FeatureSelectionStrategy = Literal["none", "boruta"]


@dataclass(frozen=True)
class TrainingRunSummary:
    run_name: str
    task_type: TaskType
    model_family: ModelFamily
    sampling_strategy: SamplingStrategy
    feature_selection_strategy: FeatureSelectionStrategy
    model_path: Path
    eval_dir: Path
    val_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    shap_artifacts: dict[str, Any]
    xgboost_device_requested: str | None
    xgboost_device_used: str | None
    xgboost_device_used_for_training: str | None
    xgboost_device_used_for_inference: str | None
    xgboost_inference_used_fallback_path: bool | None
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class TaskTrainingResult:
    task_type: TaskType
    best_model_path: Path
    best_metadata_path: Path
    training_results_path: Path
    best_val_metrics: dict[str, Any]
    best_test_metrics: dict[str, Any]
    model_family: ModelFamily
    sampling_strategy: SamplingStrategy
    feature_selection_strategy: FeatureSelectionStrategy
    xgboost_device_used: str | None
    xgboost_device_used_for_training: str | None
    xgboost_device_used_for_inference: str | None
    xgboost_inference_used_fallback_path: bool | None


class OptionalBorutaSelector(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        *,
        strategy: FeatureSelectionStrategy,
        random_state: int,
        max_features_for_dense: int = 5000,
    ) -> None:
        self.strategy = strategy
        self.random_state = random_state
        self.max_features_for_dense = max_features_for_dense
        self.use_feature_selection_: bool = False
        self.support_mask_: np.ndarray | None = None
        self.warning_message_: str | None = None

    def fit(self, x: Any, y: Any) -> OptionalBorutaSelector:
        self.use_feature_selection_ = False
        self.support_mask_ = None
        self.warning_message_ = None

        if self.strategy != "boruta":
            return self

        try:
            from boruta import BorutaPy
        except Exception as exc:
            self.warning_message_ = f"Boruta unavailable and skipped: {exc}"
            return self

        if sparse.issparse(x):
            _, n_cols = x.shape
            if n_cols > self.max_features_for_dense:
                self.warning_message_ = (
                    "Boruta skipped because transformed feature count exceeds dense threshold."
                )
                return self
            x_dense = x.toarray()
        else:
            x_dense = np.asarray(x)

        if x_dense.ndim != 2:
            self.warning_message_ = "Boruta skipped due to unexpected feature matrix shape."
            return self

        if x_dense.shape[1] == 0:
            self.warning_message_ = "Boruta skipped because no transformed features are available."
            return self

        estimator = RandomForestClassifier(
            n_estimators=300,
            random_state=self.random_state,
            n_jobs=-1,
            class_weight="balanced",
        )

        try:
            selector = BorutaPy(
                estimator,
                n_estimators="auto",
                verbose=0,
                random_state=self.random_state,
            )
            selector.fit(x_dense, np.asarray(y))
        except Exception as exc:
            self.warning_message_ = f"Boruta execution failed and was skipped: {exc}"
            return self

        support_mask = np.asarray(selector.support_, dtype=bool)
        if support_mask.size == 0 or int(support_mask.sum()) == 0:
            self.warning_message_ = "Boruta selected no features and was skipped."
            return self

        self.use_feature_selection_ = True
        self.support_mask_ = support_mask
        return self

    def transform(self, x: Any) -> Any:
        if not self.use_feature_selection_ or self.support_mask_ is None:
            return x

        if sparse.issparse(x):
            return x[:, self.support_mask_]
        return np.asarray(x)[:, self.support_mask_]


def _write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _read_feature_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Feature metadata not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_feature_splits(processed_dir: Path) -> dict[str, pd.DataFrame]:
    split_paths = {
        "train": processed_dir / "train_features.parquet",
        "val": processed_dir / "val_features.parquet",
        "test": processed_dir / "test_features.parquet",
    }

    for split_name, path in split_paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing split '{split_name}' at {path}. Run scripts/build_feature_sets.py first."
            )

    return {
        split_name: pd.read_parquet(path)
        for split_name, path in split_paths.items()
    }


def _build_sampler(
    strategy: SamplingStrategy,
    *,
    random_state: int,
) -> RandomOverSampler | RandomUnderSampler | None:
    if strategy == "none":
        return None
    if strategy == "over":
        return RandomOverSampler(random_state=random_state)
    if strategy == "under":
        return RandomUnderSampler(random_state=random_state)
    raise ValueError(f"Unsupported sampling strategy: {strategy}")


def _flatten_numeric_metrics(prefix: str, payload: dict[str, Any]) -> dict[str, float]:
    def sanitize_metric_name(raw_name: str) -> str:
        replaced = raw_name.replace(">", "gt").replace("<", "lt")
        return re.sub(r"[^A-Za-z0-9_\-./ ]", "_", replaced)

    flattened: dict[str, float] = {}
    for key, value in payload.items():
        name = sanitize_metric_name(f"{prefix}.{key}" if prefix else str(key))
        if isinstance(value, dict):
            flattened.update(_flatten_numeric_metrics(name, value))
            continue
        if isinstance(value, (int, float)):
            flattened[name] = float(value)
    return flattened


def _score_for_selection(task_type: TaskType, metrics: dict[str, Any]) -> tuple[float, float]:
    if task_type == "binary":
        return (
            float(metrics.get("f1", 0.0)),
            float(metrics.get("roc_auc") or 0.0),
        )
    return (
        float(metrics.get("macro_f1", 0.0)),
        float(metrics.get("accuracy", 0.0)),
    )


def _resolve_transformed_feature_names(
    model: ImbPipeline,
) -> list[str]:
    preprocessor = model.named_steps["preprocessor"]
    transformed_names = list(preprocessor.get_feature_names_out())

    selector = model.named_steps.get("feature_selector")
    support_mask = getattr(selector, "support_mask_", None)
    use_feature_selection = bool(getattr(selector, "use_feature_selection_", False))

    if use_feature_selection and support_mask is not None:
        transformed_names = [
            name
            for name, keep in zip(transformed_names, support_mask, strict=False)
            if bool(keep)
        ]

    return transformed_names


def _encode_multiclass_target(
    target: pd.Series,
    *,
    class_labels: tuple[str, ...],
) -> pd.Series:
    label_to_index = {label: idx for idx, label in enumerate(class_labels)}
    normalized = target.astype("string").str.strip()
    encoded = normalized.map(label_to_index)

    if bool(encoded.isna().any()):
        bad_values = sorted({str(value) for value in normalized[encoded.isna()].unique().tolist()})
        raise ValueError(
            "Unexpected multiclass labels encountered during encoding: "
            + ", ".join(bad_values)
        )

    return encoded.astype("int32")


def _log_mlflow_run(
    *,
    enable_mlflow: bool,
    settings: Settings,
    run_summary: TrainingRunSummary,
    estimator_params: dict[str, Any],
    model_metadata_path: Path,
) -> tuple[str, ...]:
    if not enable_mlflow:
        return ()

    try:
        import mlflow
    except Exception as exc:  # pragma: no cover - optional dependency
        return (f"MLflow logging skipped: {exc}",)

    warnings: list[str] = []
    try:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)

        with mlflow.start_run(run_name=run_summary.run_name):
            run_params: dict[str, str] = {
                "task_type": run_summary.task_type,
                "model_family": run_summary.model_family,
                "sampling_strategy": run_summary.sampling_strategy,
                "feature_selection_strategy": run_summary.feature_selection_strategy,
            }
            if run_summary.xgboost_device_requested is not None:
                run_params["xgboost_device_requested"] = run_summary.xgboost_device_requested
            if run_summary.xgboost_device_used is not None:
                run_params["xgboost_device_used"] = run_summary.xgboost_device_used
            if run_summary.xgboost_device_used_for_training is not None:
                run_params["xgboost_device_used_for_training"] = (
                    run_summary.xgboost_device_used_for_training
                )
            if run_summary.xgboost_device_used_for_inference is not None:
                run_params["xgboost_device_used_for_inference"] = (
                    run_summary.xgboost_device_used_for_inference
                )
            if run_summary.xgboost_inference_used_fallback_path is not None:
                run_params["xgboost_inference_used_fallback_path"] = str(
                    run_summary.xgboost_inference_used_fallback_path
                ).lower()

            mlflow.log_params(run_params)
            mlflow.set_tags(
                {
                    "runtime.inference_device": (
                        run_summary.xgboost_device_used_for_inference or "n/a"
                    ),
                    "runtime.inference_fallback": str(
                        bool(run_summary.xgboost_inference_used_fallback_path)
                    ).lower(),
                }
            )
            mlflow.log_params({f"model_param.{k}": v for k, v in estimator_params.items()})

            numeric_metrics = {
                **_flatten_numeric_metrics("val", run_summary.val_metrics),
                **_flatten_numeric_metrics("test", run_summary.test_metrics),
            }
            if numeric_metrics:
                mlflow.log_metrics(numeric_metrics)

            mlflow.log_artifact(str(run_summary.model_path), artifact_path="model")
            mlflow.log_artifact(str(model_metadata_path), artifact_path="model")

            for artifact in run_summary.eval_dir.glob("*"):
                if artifact.is_file():
                    mlflow.log_artifact(str(artifact), artifact_path="evaluation")

            for artifact_key in ("shap_top_features_json", "shap_top_features_plot"):
                artifact_path = run_summary.shap_artifacts.get(artifact_key)
                if artifact_path:
                    artifact_file = Path(str(artifact_path))
                    if artifact_file.exists():
                        mlflow.log_artifact(str(artifact_file), artifact_path="interpretability")
    except Exception as exc:
        warnings.append(f"MLflow logging failed: {exc}")

    return tuple(warnings)


def train_task(
    *,
    task_type: TaskType,
    model_families: tuple[ModelFamily, ...],
    sampling_strategies: tuple[SamplingStrategy, ...] = ("none",),
    feature_selection_strategy: FeatureSelectionStrategy = "none",
    enable_mlflow: bool = True,
    settings: Settings | None = None,
    processed_dir: Path | None = None,
    feature_metadata_path: Path | None = None,
    artifacts_dir: Path | None = None,
) -> TaskTrainingResult:
    resolved_settings = settings or get_settings()

    resolved_processed_dir = processed_dir or resolved_settings.processed_data_dir_path
    resolved_artifacts_dir = artifacts_dir or resolved_settings.artifacts_dir_path
    resolved_metadata_path = (
        feature_metadata_path
        if feature_metadata_path is not None
        else resolved_settings.artifacts_dir_path / "feature_metadata.json"
    )

    split_frames = _load_feature_splits(resolved_processed_dir)
    feature_metadata = _read_feature_metadata(resolved_metadata_path)

    target_column = "readmitted_30d" if task_type == "binary" else resolved_settings.target_column

    feature_columns = resolve_feature_columns(
        feature_metadata,
        split_frames["train"],
        target_columns=(resolved_settings.target_column, "readmitted_30d"),
        identifier_columns=tuple(str(v) for v in feature_metadata.get("identifier_columns", [])),
    )

    x_train = split_frames["train"][list(feature_columns)].copy()
    x_val = split_frames["val"][list(feature_columns)].copy()
    x_test = split_frames["test"][list(feature_columns)].copy()

    y_train_raw = split_frames["train"][target_column].copy()
    y_val_raw = split_frames["val"][target_column].copy()
    y_test_raw = split_frames["test"][target_column].copy()

    class_labels = tuple(resolved_settings.multiclass_labels)
    label_decoder: dict[int, str] | None = None

    if task_type == "binary":
        y_train = pd.to_numeric(y_train_raw, errors="coerce").fillna(0).astype("int32")
        y_val = pd.to_numeric(y_val_raw, errors="coerce").fillna(0).astype("int32")
        y_test = pd.to_numeric(y_test_raw, errors="coerce").fillna(0).astype("int32")
    else:
        y_train = _encode_multiclass_target(y_train_raw, class_labels=class_labels)
        y_val = _encode_multiclass_target(y_val_raw, class_labels=class_labels)
        y_test = _encode_multiclass_target(y_test_raw, class_labels=class_labels)
        label_decoder = {idx: label for idx, label in enumerate(class_labels)}

    if task_type == "multiclass":
        sampling_strategies = ("none",)

    run_summaries: list[TrainingRunSummary] = []

    for model_family in model_families:
        for sampling_strategy in sampling_strategies:
            run_name = (
                f"{task_type}_{model_family}_sampling-{sampling_strategy}_"
                f"fs-{feature_selection_strategy}"
            )

            preprocessor = build_preprocessor(x_train, feature_columns)

            class_weight: str | dict[int, float] | None = None
            scale_pos_weight: float | None = None
            run_xgboost_device_requested: str | None = None
            run_xgboost_device_used: str | None = None
            run_xgboost_device_used_for_training: str | None = None
            run_xgboost_device_used_for_inference: str | None = None
            run_xgboost_inference_used_fallback_path: bool | None = None
            estimator_xgboost_device: str = "cpu"
            device_warnings: list[str] = []

            if task_type == "binary" and sampling_strategy == "none":
                class_weight = "balanced"
                positives = float((y_train == 1).sum())
                negatives = float((y_train == 0).sum())
                if positives > 0:
                    scale_pos_weight = negatives / positives

            if model_family == "xgboost":
                run_xgboost_device_requested = resolved_settings.xgboost_device
                run_xgboost_device_used_for_training, xgboost_warnings = (
                    resolve_xgboost_runtime_device(
                    requested_device=resolved_settings.xgboost_device
                    )
                )
                run_xgboost_device_used = run_xgboost_device_used_for_training
                estimator_xgboost_device = run_xgboost_device_used
                device_warnings.extend(xgboost_warnings)

                if run_xgboost_device_used_for_training == "cuda":
                    run_xgboost_device_used_for_inference = "cpu"
                    run_xgboost_inference_used_fallback_path = True
                    device_warnings.append(
                        "XGBoost training used CUDA, but inference is pinned to CPU because "
                        "preprocessing outputs CPU-resident feature matrices."
                    )
                else:
                    run_xgboost_device_used_for_inference = run_xgboost_device_used_for_training
                    run_xgboost_inference_used_fallback_path = False

            estimator = build_estimator(
                model_family=model_family,
                task_type=task_type,
                random_state=resolved_settings.random_state,
                num_multiclass_labels=len(class_labels),
                class_weight=class_weight,
                scale_pos_weight=scale_pos_weight,
                xgboost_device=estimator_xgboost_device,
            )

            selector = OptionalBorutaSelector(
                strategy=feature_selection_strategy,
                random_state=resolved_settings.random_state,
            )

            pipeline_steps: list[tuple[str, Any]] = [("preprocessor", preprocessor)]
            sampler = _build_sampler(sampling_strategy, random_state=resolved_settings.random_state)
            if sampler is not None:
                pipeline_steps.append(("sampler", sampler))
            pipeline_steps.append(("feature_selector", selector))
            pipeline_steps.append(("classifier", estimator))

            model = ImbPipeline(steps=pipeline_steps)
            model.fit(x_train, y_train)

            eval_dir = resolved_artifacts_dir / "evaluations" / task_type / run_name
            val_result = evaluate_model(
                model=model,
                x=x_val,
                y=y_val,
                task_type=task_type,
                output_dir=eval_dir,
                run_name="val",
                class_labels=class_labels if task_type == "multiclass" else None,
                label_decoder=label_decoder,
            )
            test_result = evaluate_model(
                model=model,
                x=x_test,
                y=y_test,
                task_type=task_type,
                output_dir=eval_dir,
                run_name="test",
                class_labels=class_labels if task_type == "multiclass" else None,
                label_decoder=label_decoder,
            )

            shap_artifacts: dict[str, Any] = {}
            shap_warnings: tuple[str, ...] = ()
            if model_family == "xgboost":
                shap_artifacts, shap_warnings = generate_tree_shap_summary(
                    model=model,
                    x=x_train,
                    output_dir=eval_dir,
                    run_name="train",
                )

            model_path = resolved_artifacts_dir / "models" / task_type / f"{run_name}.joblib"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, model_path)

            run_warnings = [
                *device_warnings,
                *val_result.warnings,
                *test_result.warnings,
                *shap_warnings,
            ]
            selector_warning = getattr(selector, "warning_message_", None)
            if selector_warning:
                run_warnings.append(selector_warning)

            transformed_feature_names = _resolve_transformed_feature_names(model)

            run_metadata_path = eval_dir / "run_metadata.json"
            run_metadata: dict[str, Any] = {
                "run_name": run_name,
                "task_type": task_type,
                "model_family": model_family,
                "sampling_strategy": sampling_strategy,
                "feature_selection_strategy": feature_selection_strategy,
                "target_column": target_column,
                "class_labels": [0, 1] if task_type == "binary" else list(class_labels),
                "label_mapping": (
                    {}
                    if label_decoder is None
                    else {str(k): v for k, v in label_decoder.items()}
                ),
                "input_feature_columns": list(feature_columns),
                "transformed_feature_columns": transformed_feature_names,
                "val_metrics": val_result.metrics,
                "test_metrics": test_result.metrics,
                "shap_artifacts": shap_artifacts,
                "xgboost_device_requested": run_xgboost_device_requested,
                "xgboost_device_used": run_xgboost_device_used,
                "xgboost_device_used_for_training": run_xgboost_device_used_for_training,
                "xgboost_device_used_for_inference": run_xgboost_device_used_for_inference,
                "xgboost_inference_used_fallback_path": run_xgboost_inference_used_fallback_path,
                "warnings": run_warnings,
                "timestamp_utc": datetime.now(UTC).isoformat(),
            }
            _write_json(run_metadata, run_metadata_path)

            summary = TrainingRunSummary(
                run_name=run_name,
                task_type=task_type,
                model_family=model_family,
                sampling_strategy=sampling_strategy,
                feature_selection_strategy=feature_selection_strategy,
                model_path=model_path,
                eval_dir=eval_dir,
                val_metrics=val_result.metrics,
                test_metrics=test_result.metrics,
                shap_artifacts=shap_artifacts,
                xgboost_device_requested=run_xgboost_device_requested,
                xgboost_device_used=run_xgboost_device_used,
                xgboost_device_used_for_training=run_xgboost_device_used_for_training,
                xgboost_device_used_for_inference=run_xgboost_device_used_for_inference,
                xgboost_inference_used_fallback_path=run_xgboost_inference_used_fallback_path,
                warnings=tuple(run_warnings),
            )

            mlflow_warnings = _log_mlflow_run(
                enable_mlflow=enable_mlflow,
                settings=resolved_settings,
                run_summary=summary,
                estimator_params=estimator_params_for_logging(estimator),
                model_metadata_path=run_metadata_path,
            )

            if mlflow_warnings:
                summary = TrainingRunSummary(
                    run_name=summary.run_name,
                    task_type=summary.task_type,
                    model_family=summary.model_family,
                    sampling_strategy=summary.sampling_strategy,
                    feature_selection_strategy=summary.feature_selection_strategy,
                    model_path=summary.model_path,
                    eval_dir=summary.eval_dir,
                    val_metrics=summary.val_metrics,
                    test_metrics=summary.test_metrics,
                    shap_artifacts=summary.shap_artifacts,
                    xgboost_device_requested=summary.xgboost_device_requested,
                    xgboost_device_used=summary.xgboost_device_used,
                    xgboost_device_used_for_training=summary.xgboost_device_used_for_training,
                    xgboost_device_used_for_inference=summary.xgboost_device_used_for_inference,
                    xgboost_inference_used_fallback_path=(
                        summary.xgboost_inference_used_fallback_path
                    ),
                    warnings=summary.warnings + tuple(mlflow_warnings),
                )

            run_summaries.append(summary)

    if not run_summaries:
        raise RuntimeError("No model training runs were executed.")

    best_summary = max(
        run_summaries,
        key=lambda item: _score_for_selection(task_type, item.val_metrics),
    )

    best_model = joblib.load(best_summary.model_path)

    canonical_model_path = resolved_artifacts_dir / (
        "binary_model.joblib" if task_type == "binary" else "multiclass_model.joblib"
    )
    canonical_metadata_path = (
        resolved_artifacts_dir
        / (
            "binary_model_metadata.json"
            if task_type == "binary"
            else "multiclass_model_metadata.json"
        )
    )

    canonical_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, canonical_model_path)

    canonical_metadata: dict[str, Any] = {
        "task_type": task_type,
        "model_family": best_summary.model_family,
        "target_column": target_column,
        "class_labels": [0, 1] if task_type == "binary" else list(class_labels),
        "label_mapping": (
            {}
            if label_decoder is None
            else {str(k): v for k, v in label_decoder.items()}
        ),
        "training_timestamp_utc": datetime.now(UTC).isoformat(),
        "feature_columns": list(feature_columns),
        "sampling_strategy": best_summary.sampling_strategy,
        "feature_selection_strategy": best_summary.feature_selection_strategy,
        "xgboost_device_requested": best_summary.xgboost_device_requested,
        "xgboost_device_used": best_summary.xgboost_device_used,
        "xgboost_device_used_for_training": best_summary.xgboost_device_used_for_training,
        "xgboost_device_used_for_inference": best_summary.xgboost_device_used_for_inference,
        "xgboost_inference_used_fallback_path": best_summary.xgboost_inference_used_fallback_path,
        "key_evaluation_metrics": {
            "val": best_summary.val_metrics,
            "test": best_summary.test_metrics,
        },
        "evaluation_dir": str(best_summary.eval_dir),
        "shap_artifacts": best_summary.shap_artifacts,
        "warnings": list(best_summary.warnings),
    }
    _write_json(canonical_metadata, canonical_metadata_path)

    results_payload = {
        "task_type": task_type,
        "target_column": target_column,
        "feature_selection_strategy": feature_selection_strategy,
        "runs": [
            {
                "run_name": run.run_name,
                "model_family": run.model_family,
                "sampling_strategy": run.sampling_strategy,
                "feature_selection_strategy": run.feature_selection_strategy,
                "model_path": str(run.model_path),
                "eval_dir": str(run.eval_dir),
                "val_metrics": run.val_metrics,
                "test_metrics": run.test_metrics,
                "shap_artifacts": run.shap_artifacts,
                "xgboost_device_requested": run.xgboost_device_requested,
                "xgboost_device_used": run.xgboost_device_used,
                "xgboost_device_used_for_training": run.xgboost_device_used_for_training,
                "xgboost_device_used_for_inference": run.xgboost_device_used_for_inference,
                "xgboost_inference_used_fallback_path": run.xgboost_inference_used_fallback_path,
                "warnings": list(run.warnings),
            }
            for run in run_summaries
        ],
        "best_run": {
            "run_name": best_summary.run_name,
            "model_family": best_summary.model_family,
            "sampling_strategy": best_summary.sampling_strategy,
            "feature_selection_strategy": best_summary.feature_selection_strategy,
            "xgboost_device_requested": best_summary.xgboost_device_requested,
            "xgboost_device_used": best_summary.xgboost_device_used,
            "xgboost_device_used_for_training": best_summary.xgboost_device_used_for_training,
            "xgboost_device_used_for_inference": best_summary.xgboost_device_used_for_inference,
            "xgboost_inference_used_fallback_path": (
                best_summary.xgboost_inference_used_fallback_path
            ),
            "model_path": str(canonical_model_path),
            "metadata_path": str(canonical_metadata_path),
            "val_metrics": best_summary.val_metrics,
            "test_metrics": best_summary.test_metrics,
        },
    }

    training_results_path = resolved_artifacts_dir / (
        "binary_training_results.json"
        if task_type == "binary"
        else "multiclass_training_results.json"
    )
    _write_json(results_payload, training_results_path)

    return TaskTrainingResult(
        task_type=task_type,
        best_model_path=canonical_model_path,
        best_metadata_path=canonical_metadata_path,
        training_results_path=training_results_path,
        best_val_metrics=best_summary.val_metrics,
        best_test_metrics=best_summary.test_metrics,
        model_family=best_summary.model_family,
        sampling_strategy=best_summary.sampling_strategy,
        feature_selection_strategy=best_summary.feature_selection_strategy,
        xgboost_device_used=best_summary.xgboost_device_used,
        xgboost_device_used_for_training=best_summary.xgboost_device_used_for_training,
        xgboost_device_used_for_inference=best_summary.xgboost_device_used_for_inference,
        xgboost_inference_used_fallback_path=best_summary.xgboost_inference_used_fallback_path,
    )
