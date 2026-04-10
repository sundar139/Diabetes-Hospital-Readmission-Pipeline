from __future__ import annotations

import importlib

import pytest

MODULES: tuple[str, ...] = (
    "src",
    "src.config",
    "src.config.settings",
    "src.data",
    "src.features",
    "src.features.build_features",
    "src.data.preprocess",
    "src.data.split",
    "src.models",
    "src.serving",
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
