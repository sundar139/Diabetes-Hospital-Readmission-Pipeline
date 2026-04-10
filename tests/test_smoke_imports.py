from __future__ import annotations

import importlib

import pytest

MODULES: tuple[str, ...] = (
    "src",
    "src.config",
    "src.config.settings",
    "src.data",
    "src.data.preprocess",
    "src.data.split",
    "src.features",
    "src.features.build_features",
    "src.models",
    "src.models.evaluate",
    "src.models.pipeline_factory",
    "src.models.predict",
    "src.models.train",
    "src.serving.api",
    "src.serving.schemas",
    "src.serving",
    "src.llm.explain",
    "src.llm.prompting",
    "src.monitoring.drift_monitor",
    "src.monitoring",
    "src.llm",
)


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name: str) -> None:
    imported_module = importlib.import_module(module_name)
    assert imported_module is not None


def test_settings_path_resolution() -> None:
    from src.config.settings import get_settings

    settings = get_settings()
    assert settings.project_root.exists()
    assert settings.raw_data_path.parent == settings.raw_data_dir_path
