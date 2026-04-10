from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

from src.config.settings import Settings


def _load_reset_module() -> ModuleType:
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "reset_mlflow_dev_store.py"
    spec = importlib.util.spec_from_file_location("reset_mlflow_dev_store", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reset_mlflow_dev_store = _load_reset_module()


def _build_settings(
    project_root: Path,
    *,
    artifacts_destination: str = "./mlartifacts",
) -> Settings:
    return Settings(
        _env_file=None,
        project_root=project_root,
        raw_data_dir=Path("data/raw"),
        processed_data_dir=Path("data/processed"),
        artifacts_dir=Path("artifacts"),
        mlflow_backend_store_uri="sqlite:///mlflow.db",
        mlflow_artifacts_destination=artifacts_destination,
    )


def test_reset_mlflow_dev_store_deletes_only_mlflow_targets(tmp_path: Path) -> None:
    settings = _build_settings(tmp_path)

    db_path = tmp_path / "mlflow.db"
    artifacts_store = tmp_path / "mlartifacts"
    project_artifacts = tmp_path / "artifacts"

    db_path.write_text("dev-db", encoding="utf-8")
    (artifacts_store / "nested").mkdir(parents=True, exist_ok=True)
    (artifacts_store / "nested" / "artifact.txt").write_text("artifact", encoding="utf-8")
    project_artifacts.mkdir(parents=True, exist_ok=True)
    (project_artifacts / "keep.txt").write_text("keep", encoding="utf-8")

    exit_code = reset_mlflow_dev_store.reset_mlflow_dev_store(settings=settings, assume_yes=True)

    assert exit_code == 0
    assert not db_path.exists()
    assert not artifacts_store.exists()
    assert project_artifacts.exists()
    assert (project_artifacts / "keep.txt").exists()


def test_reset_mlflow_dev_store_aborts_without_confirmation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    settings = _build_settings(tmp_path)

    db_path = tmp_path / "mlflow.db"
    db_path.write_text("dev-db", encoding="utf-8")

    monkeypatch.setattr("builtins.input", lambda _: "NO")
    exit_code = reset_mlflow_dev_store.reset_mlflow_dev_store(settings=settings, assume_yes=False)

    assert exit_code == 1
    assert db_path.exists()


def test_reset_mlflow_dev_store_handles_non_local_artifact_uri(tmp_path: Path) -> None:
    settings = _build_settings(tmp_path, artifacts_destination="s3://bucket/prefix")

    db_path = tmp_path / "mlflow.db"
    db_path.write_text("dev-db", encoding="utf-8")

    exit_code = reset_mlflow_dev_store.reset_mlflow_dev_store(settings=settings, assume_yes=True)

    assert exit_code == 0
    assert not db_path.exists()
