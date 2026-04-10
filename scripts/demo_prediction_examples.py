from __future__ import annotations

import argparse
import json
import webbrowser
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from src.config.settings import Settings, get_settings
from src.serving.schemas import (
    BatchPredictionRequest,
    ExplainRequest,
    SinglePredictionRequest,
)

RequestJsonFn = Callable[[str, str, Mapping[str, Any] | None, float], Any]


def build_default_api_base_url(settings: Settings) -> str:
    return f"http://{settings.api_host}:{settings.api_port}"


def _read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Payload file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_single_payload(payload: Any) -> dict[str, Any]:
    model = SinglePredictionRequest.model_validate(payload)
    return model.model_dump(mode="json")


def _validate_batch_payload(payload: Any) -> dict[str, Any]:
    model = BatchPredictionRequest.model_validate(payload)
    return model.model_dump(mode="json")


def _validate_explain_payload(payload: Any) -> dict[str, Any]:
    model = ExplainRequest.model_validate(payload)
    return model.model_dump(mode="json")


def _http_request_json(
    method: str,
    url: str,
    payload: Mapping[str, Any] | None,
    timeout_seconds: float,
) -> Any:
    timeout = httpx.Timeout(timeout_seconds)

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.request(
                method=method,
                url=url,
                json=dict(payload) if payload is not None else None,
            )
        response.raise_for_status()
    except httpx.ConnectError as exc:
        raise RuntimeError(
            "Unable to reach local API. Start it with: uv run python scripts/run_api.py"
        ) from exc
    except httpx.HTTPStatusError as exc:
        body = exc.response.text.strip()
        detail = body[:400] if body else "<no body>"
        raise RuntimeError(
            f"API request failed for {method.upper()} {url} with status "
            f"{exc.response.status_code}: {detail}"
        ) from exc
    except httpx.RequestError as exc:
        raise RuntimeError(f"API request error for {method.upper()} {url}: {exc}") from exc

    return response.json()


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def run_demo_prediction_examples(
    *,
    api_base_url: str,
    output_dir: Path,
    single_payload_path: Path,
    batch_payload_path: Path,
    explain_payload_path: Path,
    timeout_seconds: float = 20.0,
    mode: str = "all",
    requester: RequestJsonFn | None = None,
    open_docs: bool = False,
) -> dict[str, Path]:
    normalized_mode = mode.strip().lower()
    allowed_modes = {"all", "health", "predict", "predict-batch", "explain"}
    if normalized_mode not in allowed_modes:
        raise ValueError(
            "Unsupported mode. Choose one of: all, health, predict, predict-batch, explain"
        )

    base_url = api_base_url.rstrip("/")
    docs_url = f"{base_url}/docs"

    output_dir.mkdir(parents=True, exist_ok=True)
    request_fn = requester or _http_request_json

    print("Running local demo prediction examples")
    print(f"- api_base_url: {base_url}")
    print(f"- docs_url: {docs_url}")
    print(f"- mode: {normalized_mode}")
    print(f"- output_dir: {output_dir}")

    if open_docs:
        webbrowser.open(docs_url)

    generated: dict[str, Path] = {}

    if normalized_mode in {"all", "health"}:
        health_payload = request_fn("GET", f"{base_url}/health", None, timeout_seconds)
        generated["health_response"] = _write_json(
            output_dir / "sample_health_response.json",
            health_payload,
        )

    if normalized_mode in {"all", "predict"}:
        single_payload = _validate_single_payload(_read_json(single_payload_path))
        predict_payload = request_fn(
            "POST",
            f"{base_url}/predict",
            single_payload,
            timeout_seconds,
        )
        generated["predict_response"] = _write_json(
            output_dir / "sample_prediction_response.json",
            predict_payload,
        )

    if normalized_mode in {"all", "predict-batch"}:
        batch_payload = _validate_batch_payload(_read_json(batch_payload_path))
        batch_response = request_fn(
            "POST",
            f"{base_url}/predict-batch",
            batch_payload,
            timeout_seconds,
        )
        generated["predict_batch_response"] = _write_json(
            output_dir / "sample_batch_response.json",
            batch_response,
        )

    if normalized_mode in {"all", "explain"}:
        explain_payload = _validate_explain_payload(_read_json(explain_payload_path))
        explain_response = request_fn(
            "POST",
            f"{base_url}/explain",
            explain_payload,
            timeout_seconds,
        )
        generated["explain_response"] = _write_json(
            output_dir / "sample_explanation_response.json",
            explain_response,
        )

    manifest_payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "api_base_url": base_url,
        "docs_url": docs_url,
        "mode": normalized_mode,
        "outputs": {key: str(path) for key, path in generated.items()},
    }
    generated["request_manifest"] = _write_json(
        output_dir / "demo_requests_manifest.json",
        manifest_payload,
    )

    print("Demo request execution succeeded")
    for key, path in generated.items():
        print(f"- {key}: {path}")

    return generated


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Run local API demo requests and persist real response examples."
    )
    parser.add_argument(
        "--api-base-url",
        default=build_default_api_base_url(settings),
        help="Base URL of local FastAPI server.",
    )
    parser.add_argument(
        "--single-payload",
        type=Path,
        default=settings.artifacts_dir_path / "sample_payload.json",
        help="Single prediction request payload file.",
    )
    parser.add_argument(
        "--batch-payload",
        type=Path,
        default=settings.artifacts_dir_path / "sample_batch_payload.json",
        help="Batch prediction request payload file.",
    )
    parser.add_argument(
        "--explain-payload",
        type=Path,
        default=settings.artifacts_dir_path / "sample_explain_payload.json",
        help="Explain request payload file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=settings.artifacts_dir_path / "demo",
        help="Directory where demo response files are written.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=20.0,
        help="HTTP timeout for local API calls.",
    )
    parser.add_argument(
        "--mode",
        choices=["all", "health", "predict", "predict-batch", "explain"],
        default="all",
        help="Subset of demo requests to execute.",
    )
    parser.add_argument(
        "--open-docs",
        action="store_true",
        help="Open local docs URL in default browser before requests.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    run_demo_prediction_examples(
        api_base_url=args.api_base_url,
        output_dir=args.output_dir,
        single_payload_path=args.single_payload,
        batch_payload_path=args.batch_payload,
        explain_payload_path=args.explain_payload,
        timeout_seconds=args.timeout_seconds,
        mode=args.mode,
        open_docs=args.open_docs,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
