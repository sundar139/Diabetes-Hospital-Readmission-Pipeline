from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config.settings import get_settings


def _load_demo_examples_module() -> Any:
    module_path = Path(__file__).resolve().parent / "demo_prediction_examples.py"
    spec = importlib.util.spec_from_file_location("demo_prediction_examples", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load demo_prediction_examples from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Expected JSON file was not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_demo_manifest_payload(
    *,
    api_base_url: str,
    generated_paths: dict[str, Path],
) -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "api_base_url": api_base_url.rstrip("/"),
        "generated_files": {key: str(path) for key, path in generated_paths.items()},
    }


def build_demo_summary_markdown(
    *,
    api_base_url: str,
    prediction_response: dict[str, Any],
    batch_response: dict[str, Any],
    explain_response: dict[str, Any],
    monitoring_summary: dict[str, Any],
    generated_paths: dict[str, Path],
) -> str:
    base_url = api_base_url.rstrip("/")
    binary_probability = prediction_response.get("binary_probability")
    multiclass_prediction = prediction_response.get("multiclass_prediction")

    batch_rows = int(batch_response.get("n_rows", 0))
    explanation_mode = explain_response.get("explanation_mode")

    monitoring_drift = monitoring_summary.get("binary_probability_drift", {})
    drift_status = monitoring_drift.get("status", "not_generated")
    drift_psi = monitoring_drift.get("psi", "n/a")

    output_lines = [
        "# Demo Summary",
        "",
        f"Generated at (UTC): {datetime.now(UTC).isoformat()}",
        "",
        "## Local Endpoints",
        "",
        f"- API docs: {base_url}/docs",
        f"- Health endpoint: {base_url}/health",
        f"- Predict endpoint: {base_url}/predict",
        f"- Predict batch endpoint: {base_url}/predict-batch",
        f"- Explain endpoint: {base_url}/explain",
        "",
        "## Demo Response Highlights",
        "",
        f"- Single prediction binary_probability: {binary_probability}",
        f"- Single prediction multiclass_prediction: {multiclass_prediction}",
        f"- Batch response n_rows: {batch_rows}",
        f"- Explain response mode: {explanation_mode}",
        "",
        "## Monitoring Highlights",
        "",
        f"- Binary probability drift status: {drift_status}",
        f"- Binary probability drift PSI: {drift_psi}",
        f"- Monitoring warnings count: {len(monitoring_summary.get('warnings', []))}",
        "",
        "## Generated Artifacts",
        "",
    ]

    for key in sorted(generated_paths):
        output_lines.append(f"- {key}: {generated_paths[key]}")

    return "\n".join(output_lines) + "\n"


def _run_monitoring_report(
    *,
    summary_output: Path,
    report_output: Path,
    prefer_ollama: bool,
) -> None:
    settings = get_settings()

    command = [
        sys.executable,
        str(settings.project_root / "scripts" / "run_monitoring_report.py"),
        "--summary-output",
        str(summary_output),
        "--report-output",
        str(report_output),
    ]
    if prefer_ollama:
        command.append("--prefer-ollama")

    subprocess.run(command, cwd=settings.project_root, check=True)


def parse_args() -> argparse.Namespace:
    settings = get_settings()

    parser = argparse.ArgumentParser(
        description=(
            "Generate showcase-ready local demo artifacts using real API and "
            "monitoring outputs."
        )
    )
    parser.add_argument(
        "--api-base-url",
        default=f"http://{settings.api_host}:{settings.api_port}",
        help="Base URL of local FastAPI server.",
    )
    parser.add_argument(
        "--demo-output-dir",
        type=Path,
        default=settings.artifacts_dir_path / "demo",
        help="Directory for generated demo response JSON files.",
    )
    parser.add_argument(
        "--demo-summary-output",
        type=Path,
        default=settings.reports_dir_path / "demo_summary.md",
        help="Demo summary markdown output path.",
    )
    parser.add_argument(
        "--demo-manifest-output",
        type=Path,
        default=settings.artifacts_dir_path / "demo" / "demo_manifest.json",
        help="Demo manifest JSON output path.",
    )
    parser.add_argument(
        "--monitoring-summary-output",
        type=Path,
        default=settings.reports_dir_path / "monitoring_summary.json",
        help="Monitoring summary JSON output path.",
    )
    parser.add_argument(
        "--monitoring-report-output",
        type=Path,
        default=settings.reports_dir_path / "monitoring_report.md",
        help="Monitoring markdown report output path.",
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
        "--timeout-seconds",
        type=float,
        default=20.0,
        help="HTTP timeout for local API calls.",
    )
    parser.add_argument(
        "--open-docs",
        action="store_true",
        help="Open local docs URL in default browser before requests.",
    )
    parser.add_argument(
        "--skip-monitoring",
        action="store_true",
        help="Skip monitoring report generation during showcase build.",
    )
    parser.add_argument(
        "--prefer-ollama-monitoring",
        action="store_true",
        help="Ask monitoring generation to prefer Ollama summary mode.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    demo_examples = _load_demo_examples_module()

    print("Generating showcase-ready demo artifacts")
    print(f"- api_base_url: {args.api_base_url.rstrip('/')}")
    print(f"- demo_output_dir: {args.demo_output_dir}")

    demo_outputs = demo_examples.run_demo_prediction_examples(
        api_base_url=args.api_base_url,
        output_dir=args.demo_output_dir,
        single_payload_path=args.single_payload,
        batch_payload_path=args.batch_payload,
        explain_payload_path=args.explain_payload,
        timeout_seconds=args.timeout_seconds,
        mode="all",
        open_docs=args.open_docs,
    )

    generated_paths: dict[str, Path] = dict(demo_outputs)

    if args.skip_monitoring:
        print("- monitoring_generation: skipped")
    else:
        print("- monitoring_generation: running")
        _run_monitoring_report(
            summary_output=args.monitoring_summary_output,
            report_output=args.monitoring_report_output,
            prefer_ollama=args.prefer_ollama_monitoring,
        )
        generated_paths["monitoring_summary"] = args.monitoring_summary_output
        generated_paths["monitoring_report"] = args.monitoring_report_output

    prediction_response = _read_json(generated_paths["predict_response"])
    batch_response = _read_json(generated_paths["predict_batch_response"])
    explain_response = _read_json(generated_paths["explain_response"])

    monitoring_summary: dict[str, Any] = {"warnings": []}
    if "monitoring_summary" in generated_paths:
        monitoring_summary = _read_json(generated_paths["monitoring_summary"])

    summary_markdown = build_demo_summary_markdown(
        api_base_url=args.api_base_url,
        prediction_response=prediction_response,
        batch_response=batch_response,
        explain_response=explain_response,
        monitoring_summary=monitoring_summary,
        generated_paths=generated_paths,
    )

    args.demo_summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.demo_summary_output.write_text(summary_markdown, encoding="utf-8")
    generated_paths["demo_summary"] = args.demo_summary_output

    manifest_payload = build_demo_manifest_payload(
        api_base_url=args.api_base_url,
        generated_paths=generated_paths,
    )
    generated_paths["demo_manifest"] = _write_json(args.demo_manifest_output, manifest_payload)

    print("Showcase artifact generation completed")
    for key in sorted(generated_paths):
        print(f"- {key}: {generated_paths[key]}")

    print(f"- open_next: {args.api_base_url.rstrip('/')}/docs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
