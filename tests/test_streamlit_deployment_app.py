from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

FORBIDDEN_DEPLOYMENT_DEPENDENCIES: tuple[str, ...] = (
    "shap",
    "numba",
    "llvmlite",
    "mlflow",
    "ollama",
)


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_deployment_modules_import_safely() -> None:
    project_root = Path(__file__).resolve().parents[1]
    deployment_core_path = project_root / "app" / "deployment_core.py"
    streamlit_entrypoint_path = project_root / "app" / "streamlit_app.py"

    deployment_core = _load_module("app_deployment_core", deployment_core_path)
    assert hasattr(deployment_core, "load_deployment_artifacts")
    assert hasattr(deployment_core, "resolve_deployment_paths")

    streamlit_app = _load_module("app_streamlit_entrypoint", streamlit_entrypoint_path)
    assert hasattr(streamlit_app, "main")


def test_deployment_path_resolution_points_to_project_artifacts() -> None:
    project_root = Path(__file__).resolve().parents[1]
    deployment_core = _load_module(
        "app_deployment_core_paths",
        project_root / "app" / "deployment_core.py",
    )

    paths = deployment_core.resolve_deployment_paths(project_root=project_root)

    assert paths.project_root == project_root.resolve()
    assert paths.app_dir == (project_root / "app").resolve()
    assert paths.artifacts_dir == (project_root / "artifacts").resolve()
    assert paths.reports_dir == (project_root / "reports").resolve()

    required_map = paths.required_prediction_paths
    assert set(required_map) == {
        "binary_model",
        "multiclass_model",
        "binary_metadata",
        "multiclass_metadata",
    }


def test_app_requirements_exclude_local_only_heavy_dependencies() -> None:
    project_root = Path(__file__).resolve().parents[1]
    requirements_path = project_root / "app" / "requirements.txt"
    requirements_text = requirements_path.read_text(encoding="utf-8").lower()

    for forbidden in FORBIDDEN_DEPLOYMENT_DEPENDENCIES:
        assert forbidden not in requirements_text


def test_docs_reference_app_deployment_entrypoint() -> None:
    project_root = Path(__file__).resolve().parents[1]

    local_workflow_text = (project_root / "docs" / "local_workflow.md").read_text(
        encoding="utf-8"
    )
    readme_text = (project_root / "README.md").read_text(encoding="utf-8")

    assert "app/streamlit_app.py" in local_workflow_text
    assert "app/requirements.txt" in local_workflow_text

    assert "app/streamlit_app.py" in readme_text
    assert "app/requirements.txt" in readme_text
