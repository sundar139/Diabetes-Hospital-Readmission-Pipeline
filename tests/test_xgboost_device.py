from __future__ import annotations

from src.models import pipeline_factory


def test_resolve_xgboost_runtime_device_auto_prefers_cuda(monkeypatch) -> None:
    monkeypatch.setattr(
        pipeline_factory,
        "_detect_xgboost_cuda_support",
        lambda: (True, None),
    )

    device, warnings = pipeline_factory.resolve_xgboost_runtime_device(requested_device="auto")

    assert device == "cuda"
    assert warnings == ()


def test_resolve_xgboost_runtime_device_auto_falls_back_to_cpu(monkeypatch) -> None:
    monkeypatch.setattr(
        pipeline_factory,
        "_detect_xgboost_cuda_support",
        lambda: (False, "cuda runtime missing"),
    )

    device, warnings = pipeline_factory.resolve_xgboost_runtime_device(requested_device="auto")

    assert device == "cpu"
    assert any("using cpu" in warning.lower() for warning in warnings)


def test_resolve_xgboost_runtime_device_cuda_falls_back_when_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(
        pipeline_factory,
        "_detect_xgboost_cuda_support",
        lambda: (False, "cuda runtime missing"),
    )

    device, warnings = pipeline_factory.resolve_xgboost_runtime_device(requested_device="cuda")

    assert device == "cpu"
    assert any("requested" in warning.lower() for warning in warnings)


def test_build_estimator_xgboost_sets_device() -> None:
    estimator = pipeline_factory.build_estimator(
        model_family="xgboost",
        task_type="binary",
        random_state=42,
        xgboost_device="cpu",
    )

    params = estimator.get_params(deep=False)
    assert params["device"] == "cpu"
