from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import cast

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="PIPELINE_",
        case_sensitive=False,
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

    mlflow_tracking_uri: str = "file:./mlruns"
    mlflow_experiment_name: str = "diabetes-readmission"

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
        return {
            "project_root": self.project_root.resolve(),
            "raw_data_dir": self.raw_data_dir_path,
            "raw_data_path": self.raw_data_path,
            "processed_data_dir": self.processed_data_dir_path,
            "artifacts_dir": self.artifacts_dir_path,
            "reports_dir": self.reports_dir_path,
            "figures_dir": self.figures_dir_path,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
