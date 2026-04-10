from __future__ import annotations

import importlib
import json
import os
import platform
import socket
import sys
from collections.abc import Iterable
from pathlib import Path
from urllib.parse import urlparse

from src.config.settings import get_settings

REQUIRED_MODULES: tuple[str, ...] = (
    "pandas",
    "numpy",
    "pyarrow",
    "sklearn",
    "xgboost",
    "imblearn",
    "shap",
    "joblib",
    "mlflow",
    "fastapi",
    "uvicorn",
    "pydantic",
    "pydantic_settings",
    "httpx",
)

TRACKED_ENV_KEYS: tuple[str, ...] = (
    "PIPELINE_ENVIRONMENT",
    "PIPELINE_MLFLOW_TRACKING_URI",
    "PIPELINE_MLFLOW_BACKEND_STORE_URI",
    "PIPELINE_MLFLOW_ARTIFACTS_DESTINATION",
    "PIPELINE_MLFLOW_SERVER_HOST",
    "PIPELINE_MLFLOW_SERVER_PORT",
    "PIPELINE_MLFLOW_SERVER_WORKERS",
    "PIPELINE_XGBOOST_DEVICE",
    "PIPELINE_OLLAMA_HOST",
    "PIPELINE_API_HOST",
    "PIPELINE_API_PORT",
)


def _check_python_version(minimum: tuple[int, int]) -> tuple[bool, str]:
    current = sys.version_info
    is_valid = (current.major, current.minor) >= minimum
    return is_valid, platform.python_version()


def _check_module_imports(modules: Iterable[str]) -> tuple[bool, list[str]]:
    missing: list[str] = []
    for module_name in modules:
        try:
            importlib.import_module(module_name)
        except Exception:
            missing.append(module_name)
    return not missing, missing


def _check_directories(paths: Iterable[Path]) -> tuple[bool, list[Path]]:
    missing = [path for path in paths if not path.exists()]
    return not missing, missing


def _check_env_keys(keys: Iterable[str]) -> dict[str, bool]:
    return {key: bool(os.getenv(key)) for key in keys}


def _check_tcp_reachability(*, host: str, port: int, timeout_seconds: float = 1.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_seconds):
            return True
    except OSError:
        return False


def _check_mlflow_reachability(mlflow_tracking_uri: str) -> tuple[bool, str]:
    parsed = urlparse(mlflow_tracking_uri)
    if parsed.scheme not in {"http", "https"}:
        return False, f"tracking URI is not http/https: {mlflow_tracking_uri}"

    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if host is None:
        return False, f"tracking URI host is missing: {mlflow_tracking_uri}"

    reachable = _check_tcp_reachability(host=host, port=port)
    if reachable:
        return True, f"reachable at {host}:{port}"
    return False, f"not reachable at {host}:{port}"


def _collect_xgboost_runtime_guardrails(artifacts_dir: Path) -> list[str]:
    guardrail_lines: list[str] = []

    metadata_paths = {
        "binary": artifacts_dir / "binary_model_metadata.json",
        "multiclass": artifacts_dir / "multiclass_model_metadata.json",
    }
    evaluation_metric_paths = {
        "binary": artifacts_dir / "evaluations" / "binary" / "final" / "test_final_metrics.json",
        "multiclass": (
            artifacts_dir / "evaluations" / "multiclass" / "final" / "test_final_metrics.json"
        ),
    }

    for task_name, metadata_path in metadata_paths.items():
        if not metadata_path.exists():
            guardrail_lines.append(f"{task_name}: metadata missing ({metadata_path})")
            continue

        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        requested = payload.get("xgboost_device_requested")
        used_training = payload.get("xgboost_device_used_for_training")
        if used_training is None:
            used_training = payload.get("xgboost_device_used")

        used_inference = payload.get("xgboost_device_used_for_inference")
        if used_inference is None:
            used_inference = used_training

        fallback_flag = payload.get("xgboost_inference_used_fallback_path")
        if fallback_flag is None:
            fallback_flag = bool(used_training == "cuda" and used_inference == "cpu")

        runtime_inference = None
        runtime_fallback = None
        metrics_path = evaluation_metric_paths[task_name]
        if metrics_path.exists():
            metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            runtime_payload = metrics_payload.get("inference_runtime", {})
            if isinstance(runtime_payload, dict):
                runtime_inference = runtime_payload.get("xgboost_device_used_for_inference")
                runtime_fallback = runtime_payload.get("inference_used_fallback_path")

        effective_inference = runtime_inference or used_inference
        effective_fallback = runtime_fallback if runtime_fallback is not None else fallback_flag

        guardrail_lines.append(
            f"{task_name}: requested={requested}, training={used_training}, "
            f"inference={effective_inference}, fallback_path={effective_fallback}"
        )

    return guardrail_lines


def _print_title(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def main() -> int:
    settings = get_settings()

    python_ok, python_version = _check_python_version((3, 11))
    imports_ok, missing_modules = _check_module_imports(REQUIRED_MODULES)
    dirs_ok, missing_dirs = _check_directories(settings.required_directories())
    env_status = _check_env_keys(TRACKED_ENV_KEYS)
    mlflow_reachable, mlflow_message = _check_mlflow_reachability(settings.mlflow_tracking_uri)
    xgboost_guardrails = _collect_xgboost_runtime_guardrails(settings.artifacts_dir_path)

    _print_title("Environment")
    print(f"Python version: {python_version}")
    print(f"Python >= 3.11: {'PASS' if python_ok else 'FAIL'}")

    _print_title("Imports")
    if imports_ok:
        print("Required module imports: PASS")
    else:
        print("Required module imports: FAIL")
        print("Missing modules:")
        for module_name in missing_modules:
            print(f"  - {module_name}")

    _print_title("Directory Structure")
    if dirs_ok:
        print("Required directories: PASS")
    else:
        print("Required directories: FAIL")
        print("Missing directories:")
        for directory in missing_dirs:
            print(f"  - {directory}")

    _print_title("Environment Variables")
    for key, is_set in env_status.items():
        status = "SET" if is_set else "USING DEFAULT"
        print(f"{key}: {status}")

    _print_title("Dataset")
    if settings.raw_data_path.exists():
        print(f"Raw dataset found: {settings.raw_data_path}")
    else:
        print(f"Raw dataset not found yet: {settings.raw_data_path}")

    _print_title("MLflow Reachability")
    print(f"Tracking URI: {settings.mlflow_tracking_uri}")
    print(f"Reachable: {'PASS' if mlflow_reachable else 'WARN'}")
    print(f"Details: {mlflow_message}")

    _print_title("XGBoost Runtime Guardrails")
    for line in xgboost_guardrails:
        print(f"- {line}")

    success = python_ok and imports_ok and dirs_ok
    _print_title("Summary")
    print("Healthcheck: PASS" if success else "Healthcheck: FAIL")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
