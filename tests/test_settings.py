from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import ValidationError
from src.config.settings import Settings


def test_mlflow_defaults_use_local_tracking_server() -> None:
    settings = Settings(_env_file=None)

    assert settings.mlflow_tracking_uri == "http://127.0.0.1:5000"
    assert settings.mlflow_backend_store_uri == "sqlite:///mlflow.db"
    assert settings.mlflow_server_host == "127.0.0.1"
    assert settings.mlflow_server_port == 5000
    assert settings.xgboost_device == "auto"


def test_mlflow_backend_uri_and_path_resolve_from_project_root(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        project_root=tmp_path,
        mlflow_backend_store_uri="sqlite:///nested/mlflow.db",
        mlflow_artifacts_destination="./mlartifacts",
    )

    expected_db_path = (tmp_path / "nested" / "mlflow.db").resolve().as_posix()
    assert settings.mlflow_backend_store_uri_resolved == f"sqlite:///{expected_db_path}"
    assert settings.mlflow_backend_store_path == (tmp_path / "nested" / "mlflow.db").resolve()
    assert settings.mlflow_server_url == "http://127.0.0.1:5000"


def test_mlflow_artifacts_destination_relative_path_normalizes_to_file_uri(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        project_root=tmp_path,
        mlflow_artifacts_destination="./mlartifacts",
    )

    expected_path = (tmp_path / "mlartifacts").resolve()
    assert settings.mlflow_artifacts_destination_path == expected_path
    assert settings.mlflow_artifacts_destination_uri_resolved == expected_path.as_uri()


@pytest.mark.skipif(os.name != "nt", reason="Windows path normalization test")
def test_mlflow_artifacts_destination_windows_abs_path_normalizes_to_file_uri(
    tmp_path: Path,
) -> None:
    windows_artifact_path = (tmp_path / "win_artifacts").resolve()
    settings = Settings(
        _env_file=None,
        project_root=tmp_path,
        mlflow_artifacts_destination=str(windows_artifact_path),
    )

    assert settings.mlflow_artifacts_destination_path == windows_artifact_path
    assert settings.mlflow_artifacts_destination_uri_resolved == windows_artifact_path.as_uri()


def test_mlflow_artifacts_destination_uri_is_preserved(tmp_path: Path) -> None:
    configured_uri = "file:///C:/temp/mlartifacts"
    settings = Settings(
        _env_file=None,
        project_root=tmp_path,
        mlflow_artifacts_destination=configured_uri,
    )

    assert settings.mlflow_artifacts_destination_uri_resolved == configured_uri


def test_mlflow_server_and_xgboost_config_validation() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, mlflow_server_port=70000)

    with pytest.raises(ValidationError):
        Settings(_env_file=None, xgboost_device="gpu")
