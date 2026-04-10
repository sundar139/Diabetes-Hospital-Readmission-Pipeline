from __future__ import annotations

from pathlib import Path

from src.config.settings import get_settings


def _format_path_line(name: str, path: Path) -> str:
    status = "OK" if path.exists() else "MISSING"
    return f"{name:<22} {path} [{status}]"


def main() -> int:
    settings = get_settings()
    paths = settings.important_paths()

    print("Project Settings")
    print("----------------")
    print(f"project_name           {settings.project_name}")
    print(f"environment            {settings.environment}")
    print(f"target_column          {settings.target_column}")
    print(f"multiclass_labels      {settings.multiclass_labels}")
    print(f"binary_positive_label  {settings.binary_positive_label}")
    print(f"default_model_name     {settings.default_model_name}")
    print(f"mlflow_tracking_uri    {settings.mlflow_tracking_uri}")
    print(f"mlflow_backend_store   {settings.mlflow_backend_store_uri_resolved}")
    print(f"mlflow_artifacts_raw   {settings.mlflow_artifacts_destination}")
    print(f"mlflow_artifacts_uri   {settings.mlflow_artifacts_destination_uri_resolved}")
    try:
        print(f"mlflow_artifacts_path  {settings.mlflow_artifacts_destination_path}")
    except ValueError:
        print("mlflow_artifacts_path  non-local-uri")
    print(f"mlflow_server_host     {settings.mlflow_server_host}")
    print(f"mlflow_server_port     {settings.mlflow_server_port}")
    print(f"mlflow_experiment      {settings.mlflow_experiment_name}")
    print(f"xgboost_device         {settings.xgboost_device}")
    print(f"api_host               {settings.api_host}")
    print(f"api_port               {settings.api_port}")
    print(f"ollama_host            {settings.ollama_host}")
    print(f"ollama_model           {settings.ollama_model}")

    print("\nResolved Paths")
    print("--------------")
    for name, path in paths.items():
        print(_format_path_line(name, path))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
