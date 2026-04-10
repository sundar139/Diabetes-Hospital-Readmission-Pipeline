from __future__ import annotations

import importlib.util
import subprocess
import sys

from src.config.settings import Settings, get_settings


def _ensure_mlflow_available() -> None:
    if importlib.util.find_spec("mlflow") is None:
        raise RuntimeError(
            "MLflow is not installed in this environment. "
            "Install dependencies with: uv sync --group dev --extra eda"
        )


def build_mlflow_server_command(settings: Settings) -> list[str]:
    return [
        sys.executable,
        "-m",
        "mlflow",
        "server",
        "--backend-store-uri",
        settings.mlflow_backend_store_uri_resolved,
        "--artifacts-destination",
        settings.mlflow_artifacts_destination_uri_resolved,
        "--host",
        settings.mlflow_server_host,
        "--port",
        str(settings.mlflow_server_port),
    ]


def run_mlflow_server(*, settings: Settings | None = None) -> int:
    resolved_settings = settings or get_settings()
    _ensure_mlflow_available()

    try:
        local_artifact_path = resolved_settings.mlflow_artifacts_destination_path
    except ValueError:
        local_artifact_path = None

    if local_artifact_path is not None:
        local_artifact_path.mkdir(parents=True, exist_ok=True)

    try:
        artifact_destination_uri = resolved_settings.mlflow_artifacts_destination_uri_resolved
        command = build_mlflow_server_command(resolved_settings)
    except ValueError as exc:
        raise RuntimeError(f"Invalid MLflow server configuration: {exc}") from exc

    print("Starting local MLflow tracking server")
    print(f"- tracking_uri: {resolved_settings.mlflow_tracking_uri}")
    print(f"- backend_store_uri: {resolved_settings.mlflow_backend_store_uri_resolved}")
    print(f"- artifacts_destination_configured: {resolved_settings.mlflow_artifacts_destination}")
    print(
        "- artifacts_destination_server_uri: "
        f"{artifact_destination_uri}"
    )
    print(f"- ui_url: {resolved_settings.mlflow_server_url}")

    try:
        subprocess.run(
            command,
            cwd=resolved_settings.project_root,
            check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("Unable to launch MLflow server command.") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"MLflow server exited with status code {exc.returncode}.") from exc

    return 0


def main() -> int:
    try:
        return run_mlflow_server()
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
