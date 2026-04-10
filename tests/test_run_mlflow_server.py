from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from src.config.settings import Settings


def _load_runner_module() -> ModuleType:
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "run_mlflow_server.py"
    spec = importlib.util.spec_from_file_location("run_mlflow_server", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_mlflow_server = _load_runner_module()


def _build_settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        project_root=tmp_path,
        mlflow_tracking_uri="http://127.0.0.1:5000",
        mlflow_backend_store_uri="sqlite:///mlflow.db",
        mlflow_artifacts_destination="./mlartifacts",
        mlflow_server_host="127.0.0.1",
        mlflow_server_port=5000,
    )


def test_build_mlflow_server_command_uses_resolved_values(tmp_path: Path) -> None:
    settings = _build_settings(tmp_path)

    command = run_mlflow_server.build_mlflow_server_command(settings)

    assert command[:4] == [sys.executable, "-m", "mlflow", "server"]
    assert "--backend-store-uri" in command
    assert settings.mlflow_backend_store_uri_resolved in command
    assert "--artifacts-destination" in command
    assert settings.mlflow_artifacts_destination_uri_resolved in command


def test_run_mlflow_server_raises_when_mlflow_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings = _build_settings(tmp_path)

    monkeypatch.setattr(run_mlflow_server.importlib.util, "find_spec", lambda _: None)

    with pytest.raises(RuntimeError, match="MLflow is not installed"):
        run_mlflow_server.run_mlflow_server(settings=settings)


def test_run_mlflow_server_invokes_subprocess(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings = _build_settings(tmp_path)
    captured: dict[str, Any] = {}

    monkeypatch.setattr(run_mlflow_server.importlib.util, "find_spec", lambda _: object())

    def fake_run(command: list[str], *, cwd: Path, check: bool) -> subprocess.CompletedProcess[str]:
        captured["command"] = command
        captured["cwd"] = cwd
        captured["check"] = check
        return subprocess.CompletedProcess(args=command, returncode=0)

    monkeypatch.setattr(run_mlflow_server.subprocess, "run", fake_run)

    exit_code = run_mlflow_server.run_mlflow_server(settings=settings)

    assert exit_code == 0
    assert captured["check"] is True
    assert captured["cwd"] == settings.project_root
    assert "--port" in captured["command"]
    assert str(settings.mlflow_server_port) in captured["command"]
    assert settings.mlflow_artifacts_destination_path.exists()
