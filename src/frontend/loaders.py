from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config.settings import get_settings
from src.models.predict import load_model, load_model_metadata

JsonDict = dict[str, Any]


@dataclass(frozen=True)
class FrontendPaths:
    project_root: Path
    artifacts_dir: Path
    reports_dir: Path
    docs_dir: Path
    readme_path: Path
    binary_model_path: Path
    multiclass_model_path: Path
    binary_metadata_path: Path
    multiclass_metadata_path: Path
    sample_payload_path: Path
    sample_batch_payload_path: Path
    sample_explain_payload_path: Path
    monitoring_summary_path: Path
    monitoring_report_path: Path
    model_comparison_report_path: Path
    demo_summary_path: Path

    @property
    def required_prediction_paths(self) -> dict[str, Path]:
        return {
            "binary_model": self.binary_model_path,
            "multiclass_model": self.multiclass_model_path,
            "binary_metadata": self.binary_metadata_path,
            "multiclass_metadata": self.multiclass_metadata_path,
        }


@dataclass(frozen=True)
class PredictionArtifacts:
    paths: FrontendPaths
    binary_model: Any
    multiclass_model: Any
    binary_metadata: JsonDict
    multiclass_metadata: JsonDict

    @property
    def binary_feature_columns(self) -> list[str]:
        raw = self.binary_metadata.get("feature_columns", [])
        if not isinstance(raw, list):
            return []
        return [str(column) for column in raw]

    @property
    def multiclass_feature_columns(self) -> list[str]:
        raw = self.multiclass_metadata.get("feature_columns", [])
        if not isinstance(raw, list):
            return []
        return [str(column) for column in raw]


@dataclass(frozen=True)
class FrontendContext:
    paths: FrontendPaths
    baseline_features: JsonDict
    dummy_features: JsonDict
    batch_example_rows: list[JsonDict]
    monitoring_summary: JsonDict | None
    monitoring_report_text: str | None
    model_comparison_report_text: str | None
    demo_summary_text: str | None
    readme_excerpt: str | None
    load_warnings: tuple[str, ...] = ()


def resolve_frontend_paths(project_root: Path | None = None) -> FrontendPaths:
    settings = get_settings()
    resolved_root = (project_root or settings.project_root).resolve()

    artifacts_dir = (resolved_root / "artifacts").resolve()
    reports_dir = (resolved_root / "reports").resolve()

    return FrontendPaths(
        project_root=resolved_root,
        artifacts_dir=artifacts_dir,
        reports_dir=reports_dir,
        docs_dir=(resolved_root / "docs").resolve(),
        readme_path=(resolved_root / "README.md").resolve(),
        binary_model_path=(artifacts_dir / "binary_model.joblib").resolve(),
        multiclass_model_path=(artifacts_dir / "multiclass_model.joblib").resolve(),
        binary_metadata_path=(artifacts_dir / "binary_model_metadata.json").resolve(),
        multiclass_metadata_path=(artifacts_dir / "multiclass_model_metadata.json").resolve(),
        sample_payload_path=(artifacts_dir / "sample_payload.json").resolve(),
        sample_batch_payload_path=(artifacts_dir / "sample_batch_payload.json").resolve(),
        sample_explain_payload_path=(artifacts_dir / "sample_explain_payload.json").resolve(),
        monitoring_summary_path=(reports_dir / "monitoring_summary.json").resolve(),
        monitoring_report_path=(reports_dir / "monitoring_report.md").resolve(),
        model_comparison_report_path=(reports_dir / "model_comparison_report.md").resolve(),
        demo_summary_path=(reports_dir / "demo_summary.md").resolve(),
    )


def missing_prediction_artifacts(paths: FrontendPaths | None = None) -> list[Path]:
    resolved_paths = paths or resolve_frontend_paths()
    return [path for path in resolved_paths.required_prediction_paths.values() if not path.exists()]


def _read_required_json(path: Path) -> JsonDict:
    if not path.exists():
        raise FileNotFoundError(f"Expected JSON artifact is missing: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON in {path}, got {type(payload).__name__}.")
    return payload


def _read_optional_json(path: Path) -> JsonDict | None:
    if not path.exists():
        return None

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return payload
    return None


def _read_optional_text(path: Path, *, max_chars: int = 6000) -> str | None:
    if not path.exists():
        return None

    text = path.read_text(encoding="utf-8")
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n\n..."


def _extract_features(payload: Mapping[str, Any] | None) -> JsonDict:
    if payload is None:
        return {}

    features = payload.get("features")
    if not isinstance(features, Mapping):
        return {}

    return {str(key): value for key, value in features.items()}


def _extract_batch_rows(payload: Mapping[str, Any] | None) -> list[JsonDict]:
    if payload is None:
        return []

    rows = payload.get("rows")
    if not isinstance(rows, list):
        return []

    output_rows: list[JsonDict] = []
    for row in rows:
        if isinstance(row, Mapping):
            output_rows.append({str(key): value for key, value in row.items()})
    return output_rows


def build_feature_defaults(
    *,
    feature_columns: list[str],
    base_features: Mapping[str, Any],
) -> JsonDict:
    return {feature_name: base_features.get(feature_name) for feature_name in feature_columns}


def load_prediction_artifacts(paths: FrontendPaths | None = None) -> PredictionArtifacts:
    resolved_paths = paths or resolve_frontend_paths()
    missing = missing_prediction_artifacts(resolved_paths)
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Required model artifacts for Streamlit are missing: "
            f"{missing_text}. Run training/evaluation first."
        )

    return PredictionArtifacts(
        paths=resolved_paths,
        binary_model=load_model(resolved_paths.binary_model_path),
        multiclass_model=load_model(resolved_paths.multiclass_model_path),
        binary_metadata=load_model_metadata(resolved_paths.binary_metadata_path),
        multiclass_metadata=load_model_metadata(resolved_paths.multiclass_metadata_path),
    )


def load_frontend_context(paths: FrontendPaths | None = None) -> FrontendContext:
    resolved_paths = paths or resolve_frontend_paths()

    warnings: list[str] = []

    try:
        sample_payload = _read_required_json(resolved_paths.sample_payload_path)
    except Exception:
        sample_payload = _read_optional_json(resolved_paths.sample_payload_path)
        warnings.append(
            "sample_payload.json is missing or invalid; using fallback defaults where possible."
        )

    try:
        sample_explain_payload = _read_required_json(resolved_paths.sample_explain_payload_path)
    except Exception:
        sample_explain_payload = _read_optional_json(resolved_paths.sample_explain_payload_path)
        warnings.append(
            "sample_explain_payload.json is missing or invalid; dummy example may be limited."
        )

    sample_batch_payload = _read_optional_json(resolved_paths.sample_batch_payload_path)

    batch_rows = _extract_batch_rows(sample_batch_payload)
    baseline_features = _extract_features(sample_payload)
    if not baseline_features and batch_rows:
        baseline_features = dict(batch_rows[0])

    dummy_features = _extract_features(sample_explain_payload)
    if not dummy_features and len(batch_rows) >= 2:
        dummy_features = dict(batch_rows[1])
    if not dummy_features:
        dummy_features = dict(baseline_features)

    if not baseline_features:
        warnings.append(
            "No baseline feature payload was loaded; form defaults may require manual input."
        )

    return FrontendContext(
        paths=resolved_paths,
        baseline_features=baseline_features,
        dummy_features=dummy_features,
        batch_example_rows=batch_rows,
        monitoring_summary=_read_optional_json(resolved_paths.monitoring_summary_path),
        monitoring_report_text=_read_optional_text(resolved_paths.monitoring_report_path),
        model_comparison_report_text=_read_optional_text(resolved_paths.model_comparison_report_path),
        demo_summary_text=_read_optional_text(resolved_paths.demo_summary_path),
        readme_excerpt=_read_optional_text(resolved_paths.readme_path, max_chars=3500),
        load_warnings=tuple(warnings),
    )
