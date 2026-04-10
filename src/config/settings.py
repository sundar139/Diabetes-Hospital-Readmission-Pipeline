from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, cast
from urllib.parse import urlparse
from urllib.request import url2pathname

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _looks_like_windows_drive(path_value: str) -> bool:
    normalized = path_value.replace("/", "\\")
    return len(normalized) >= 2 and normalized[1] == ":"


def _strip_windows_drive_leading_slash(path_value: str) -> str:
    if path_value.startswith("/") and _looks_like_windows_drive(path_value[1:]):
        return path_value[1:]
    return path_value


def _has_uri_scheme(path_value: str) -> bool:
    value = path_value.strip()
    if not value:
        return False
    if _looks_like_windows_drive(value):
        return False
    return bool(urlparse(value).scheme)


def _resolve_local_path(path_value: str, *, project_root: Path) -> Path:
    candidate = Path(_strip_windows_drive_leading_slash(path_value.strip()))
    if candidate.is_absolute() or _looks_like_windows_drive(path_value):
        return candidate.resolve()
    return (project_root / candidate).resolve()


def _resolve_sqlite_backend_store_uri(uri: str, *, project_root: Path) -> str:
    sqlite_prefix = "sqlite:///"
    if not uri.lower().startswith(sqlite_prefix):
        return uri

    sqlite_path = _strip_windows_drive_leading_slash(uri[len(sqlite_prefix) :])
    if not sqlite_path or sqlite_path == ":memory:":
        return uri

    resolved_path = _resolve_local_path(sqlite_path, project_root=project_root)

    return f"sqlite:///{resolved_path.as_posix()}"


def _resolve_mlflow_artifacts_destination_uri(raw_value: str, *, project_root: Path) -> str:
    value = raw_value.strip()
    if not value:
        raise ValueError("mlflow_artifacts_destination must not be empty.")

    if _has_uri_scheme(value):
        return value

    return _resolve_local_path(value, project_root=project_root).as_uri()


def _resolve_mlflow_artifacts_destination_local_path(
    raw_value: str,
    *,
    project_root: Path,
) -> Path | None:
    value = raw_value.strip()
    if not value:
        return None

    if _has_uri_scheme(value):
        parsed = urlparse(value)
        if parsed.scheme.lower() != "file":
            return None

        parsed_path = url2pathname(parsed.path)
        if parsed.netloc and parsed.netloc.lower() not in {"", "localhost"}:
            path_value = f"//{parsed.netloc}{parsed_path}"
        else:
            path_value = parsed_path
        return _resolve_local_path(path_value, project_root=project_root)

    return _resolve_local_path(value, project_root=project_root)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="PIPELINE_",
        case_sensitive=False,
        enable_decoding=False,
        extra="ignore",
    )

    project_name: str = "Diabetes Hospital Readmission Pipeline"
    environment: str = "local"
    project_root: Path = Field(default_factory=_default_project_root)

    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    artifacts_dir: Path = Path("artifacts")
    reports_dir: Path = Path("reports")
    figures_dir: Path = Path("reports/figures")
    raw_data_filename: str = "diabetic_data.csv"

    target_column: str = "readmitted"
    multiclass_labels: tuple[str, str, str] = ("NO", ">30", "<30")
    binary_positive_label: str = "<30"
    random_state: int = 42
    test_size: float = 0.2

    default_model_name: str = "xgboost"
    model_registry_name: str = "diabetes_readmission"

    mlflow_tracking_uri: str = "http://127.0.0.1:5000"
    mlflow_backend_store_uri: str = "sqlite:///mlflow.db"
    mlflow_artifacts_destination: str = "./mlartifacts"
    mlflow_server_host: str = "127.0.0.1"
    mlflow_server_port: int = 5000
    mlflow_server_workers: int = 1
    mlflow_experiment_name: str = "diabetes-readmission"

    xgboost_device: Literal["auto", "cuda", "cpu"] = "auto"

    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_reload: bool = False

    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    ollama_timeout_seconds: float = 30.0

    @field_validator("test_size")
    @classmethod
    def _validate_test_size(cls, value: float) -> float:
        if not 0.0 < value < 1.0:
            raise ValueError("test_size must be between 0 and 1.")
        return value

    @field_validator("api_port")
    @classmethod
    def _validate_api_port(cls, value: int) -> int:
        if not 1 <= value <= 65535:
            raise ValueError("api_port must be in range 1..65535.")
        return value

    @field_validator("mlflow_server_port")
    @classmethod
    def _validate_mlflow_server_port(cls, value: int) -> int:
        if not 1 <= value <= 65535:
            raise ValueError("mlflow_server_port must be in range 1..65535.")
        return value

    @field_validator("mlflow_server_workers")
    @classmethod
    def _validate_mlflow_server_workers(cls, value: int) -> int:
        if value < 1:
            raise ValueError("mlflow_server_workers must be at least 1.")
        return value

    @field_validator("xgboost_device")
    @classmethod
    def _validate_xgboost_device(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"auto", "cuda", "cpu"}:
            raise ValueError("xgboost_device must be one of: auto, cuda, cpu.")
        return normalized

    @field_validator("multiclass_labels", mode="before")
    @classmethod
    def _parse_and_validate_multiclass_labels(
        cls,
        value: tuple[str, str, str] | str,
    ) -> tuple[str, str, str]:
        if isinstance(value, str):
            labels = tuple(label.strip() for label in value.split(",") if label.strip())
        else:
            labels = tuple(value)

        expected = {"NO", ">30", "<30"}
        if len(labels) != 3 or set(labels) != expected:
            raise ValueError("multiclass_labels must contain exactly: NO, >30, <30.")
        return cast(tuple[str, str, str], labels)

    def _resolve_path(self, path_value: Path) -> Path:
        if path_value.is_absolute():
            return path_value
        return (self.project_root / path_value).resolve()

    @property
    def raw_data_dir_path(self) -> Path:
        return self._resolve_path(self.raw_data_dir)

    @property
    def processed_data_dir_path(self) -> Path:
        return self._resolve_path(self.processed_data_dir)

    @property
    def artifacts_dir_path(self) -> Path:
        return self._resolve_path(self.artifacts_dir)

    @property
    def reports_dir_path(self) -> Path:
        return self._resolve_path(self.reports_dir)

    @property
    def figures_dir_path(self) -> Path:
        return self._resolve_path(self.figures_dir)

    @property
    def mlflow_artifacts_destination_path(self) -> Path:
        local_path = _resolve_mlflow_artifacts_destination_local_path(
            self.mlflow_artifacts_destination,
            project_root=self.project_root,
        )
        if local_path is None:
            raise ValueError(
                "mlflow_artifacts_destination is a non-local URI and cannot be represented "
                "as a filesystem path."
            )
        return local_path

    @property
    def mlflow_artifacts_destination_uri_resolved(self) -> str:
        return _resolve_mlflow_artifacts_destination_uri(
            self.mlflow_artifacts_destination,
            project_root=self.project_root,
        )

    @property
    def mlflow_backend_store_uri_resolved(self) -> str:
        return _resolve_sqlite_backend_store_uri(
            self.mlflow_backend_store_uri,
            project_root=self.project_root,
        )

    @property
    def mlflow_backend_store_path(self) -> Path | None:
        resolved_uri = self.mlflow_backend_store_uri_resolved
        sqlite_prefix = "sqlite:///"
        if not resolved_uri.lower().startswith(sqlite_prefix):
            return None

        db_path = _strip_windows_drive_leading_slash(resolved_uri[len(sqlite_prefix) :])
        if not db_path or db_path == ":memory:":
            return None
        return Path(db_path).resolve()

    @property
    def mlflow_server_url(self) -> str:
        return f"http://{self.mlflow_server_host}:{self.mlflow_server_port}"

    @property
    def raw_data_path(self) -> Path:
        return self.raw_data_dir_path / self.raw_data_filename

    def required_directories(self) -> tuple[Path, ...]:
        return (
            self.raw_data_dir_path,
            self.processed_data_dir_path,
            self.artifacts_dir_path,
            self.reports_dir_path,
            self.figures_dir_path,
        )

    def important_paths(self) -> dict[str, Path]:
        paths = {
            "project_root": self.project_root.resolve(),
            "raw_data_dir": self.raw_data_dir_path,
            "raw_data_path": self.raw_data_path,
            "processed_data_dir": self.processed_data_dir_path,
            "artifacts_dir": self.artifacts_dir_path,
            "reports_dir": self.reports_dir_path,
            "figures_dir": self.figures_dir_path,
        }

        try:
            paths["mlflow_artifacts_destination"] = self.mlflow_artifacts_destination_path
        except ValueError:
            pass

        return paths


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
