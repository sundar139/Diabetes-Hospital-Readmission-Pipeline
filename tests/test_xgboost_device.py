from __future__ import annotations

from src.models import pipeline_factory
from src.models.predict import ensure_cpu_inference_for_xgboost


class XGBClassifier:
    def __init__(self, *, device: str) -> None:
        self.device = device

    def get_params(self, deep: bool = False) -> dict[str, str]:
        return {"device": self.device}

    def set_params(self, **kwargs: str) -> XGBClassifier:
        if "device" in kwargs:
            self.device = kwargs["device"]
        return self


class _FakePipeline:
    def __init__(self, classifier: XGBClassifier) -> None:
        self.named_steps = {"classifier": classifier}


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


def test_ensure_cpu_inference_for_xgboost_pins_cuda_runtime_to_cpu() -> None:
    classifier = XGBClassifier(device="cuda")
    model = _FakePipeline(classifier)

    runtime = ensure_cpu_inference_for_xgboost(model)

    assert runtime.is_xgboost_model is True
    assert runtime.xgboost_device_requested == "cuda"
    assert runtime.xgboost_device_used_for_inference == "cpu"
    assert runtime.inference_used_fallback_path is True
    assert classifier.device == "cpu"


def test_ensure_cpu_inference_for_xgboost_keeps_cpu_runtime() -> None:
    classifier = XGBClassifier(device="cpu")
    model = _FakePipeline(classifier)

    runtime = ensure_cpu_inference_for_xgboost(model)

    assert runtime.is_xgboost_model is True
    assert runtime.xgboost_device_requested == "cpu"
    assert runtime.xgboost_device_used_for_inference == "cpu"
    assert runtime.inference_used_fallback_path is False
