from __future__ import annotations

import importlib
import os
import platform
import sys
from collections.abc import Iterable
from pathlib import Path

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


def _print_title(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def main() -> int:
    settings = get_settings()

    python_ok, python_version = _check_python_version((3, 11))
    imports_ok, missing_modules = _check_module_imports(REQUIRED_MODULES)
    dirs_ok, missing_dirs = _check_directories(settings.required_directories())
    env_status = _check_env_keys(TRACKED_ENV_KEYS)

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

    success = python_ok and imports_ok and dirs_ok
    _print_title("Summary")
    print("Healthcheck: PASS" if success else "Healthcheck: FAIL")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
