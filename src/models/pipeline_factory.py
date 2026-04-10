from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

TaskType = Literal["binary", "multiclass"]
ModelFamily = Literal["logistic_regression", "random_forest", "xgboost"]


@dataclass(frozen=True)
class FeatureColumnSpec:
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]


def resolve_feature_columns(
    feature_metadata: dict[str, Any],
    frame: pd.DataFrame,
    *,
    target_columns: tuple[str, ...],
    identifier_columns: tuple[str, ...] = (),
) -> tuple[str, ...]:
    metadata_columns = feature_metadata.get("model_candidate_columns", [])
    candidate_columns = [str(column) for column in metadata_columns]

    excluded = set(target_columns) | set(identifier_columns)

    if candidate_columns:
        resolved = [
            column
            for column in candidate_columns
            if column in frame.columns and column not in excluded
        ]
    else:
        resolved = [
            column
            for column in frame.columns
            if column not in excluded
        ]

    deduplicated = tuple(dict.fromkeys(resolved))
    if not deduplicated:
        raise ValueError("No model feature columns resolved from metadata and frame.")
    return deduplicated


def infer_feature_column_spec(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> FeatureColumnSpec:
    missing_columns = [column for column in feature_columns if column not in frame.columns]
    if missing_columns:
        raise KeyError(
            "Feature columns missing from frame: " + ", ".join(missing_columns)
        )

    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    for column in feature_columns:
        if pd.api.types.is_numeric_dtype(frame[column]):
            numeric_columns.append(column)
        else:
            categorical_columns.append(column)

    if not numeric_columns and not categorical_columns:
        raise ValueError("No usable numeric or categorical features were inferred.")

    return FeatureColumnSpec(
        numeric_columns=tuple(numeric_columns),
        categorical_columns=tuple(categorical_columns),
    )


def build_preprocessor(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> ColumnTransformer:
    column_spec = infer_feature_column_spec(frame, feature_columns)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers: list[tuple[str, Pipeline, list[str] | tuple[str, ...]]] = []
    if column_spec.numeric_columns:
        transformers.append(("numeric", numeric_pipeline, list(column_spec.numeric_columns)))
    if column_spec.categorical_columns:
        transformers.append(
            ("categorical", categorical_pipeline, list(column_spec.categorical_columns))
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_estimator(
    *,
    model_family: ModelFamily,
    task_type: TaskType,
    random_state: int,
    num_multiclass_labels: int = 3,
    class_weight: str | dict[int, float] | None = None,
    scale_pos_weight: float | None = None,
    extra_params: dict[str, Any] | None = None,
) -> LogisticRegression | RandomForestClassifier | XGBClassifier:
    params = dict(extra_params or {})

    if model_family == "logistic_regression":
        default_params: dict[str, Any] = {
            "max_iter": 2000,
            "class_weight": class_weight,
            "random_state": random_state,
            "solver": "lbfgs",
        }

        default_params.update(params)
        return LogisticRegression(**default_params)

    if model_family == "random_forest":
        default_params = {
            "n_estimators": 400,
            "max_depth": None,
            "min_samples_leaf": 1,
            "class_weight": class_weight,
            "n_jobs": -1,
            "random_state": random_state,
        }
        default_params.update(params)
        return RandomForestClassifier(**default_params)

    if model_family == "xgboost":
        default_params = {
            "n_estimators": 350,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": 1.0,
            "n_jobs": -1,
            "random_state": random_state,
            "tree_method": "hist",
        }
        if task_type == "binary":
            default_params.update({"objective": "binary:logistic", "eval_metric": "logloss"})
            if scale_pos_weight is not None and scale_pos_weight > 0.0:
                default_params["scale_pos_weight"] = float(scale_pos_weight)
        else:
            default_params.update(
                {
                    "objective": "multi:softprob",
                    "eval_metric": "mlogloss",
                    "num_class": int(num_multiclass_labels),
                }
            )

        default_params.update(params)
        return XGBClassifier(**default_params)

    raise ValueError(f"Unsupported model family: {model_family}")


def estimator_params_for_logging(estimator: Any) -> dict[str, Any]:
    if not hasattr(estimator, "get_params"):
        return {}

    params = estimator.get_params(deep=False)
    safe_params: dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe_params[key] = value
        else:
            safe_params[key] = str(value)
    return safe_params
