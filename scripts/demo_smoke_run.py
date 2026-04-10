from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from src.config.settings import get_settings


def _required_demo_artifacts(artifacts_dir: Path) -> tuple[Path, ...]:
    return (
        artifacts_dir / "binary_model.joblib",
        artifacts_dir / "binary_model_metadata.json",
        artifacts_dir / "multiclass_model.joblib",
        artifacts_dir / "multiclass_model_metadata.json",
        artifacts_dir / "sample_payload.json",
        artifacts_dir / "sample_batch_payload.json",
        artifacts_dir / "sample_explain_payload.json",
    )


def validate_demo_prerequisites(artifacts_dir: Path) -> list[Path]:
    return [path for path in _required_demo_artifacts(artifacts_dir) if not path.exists()]


def parse_args() -> argparse.Namespace:
    settings = get_settings()

    parser = argparse.ArgumentParser(
        description="Run a local demo smoke workflow using trained artifacts and running API."
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
        help="Directory for generated demo response outputs.",
    )
    parser.add_argument(
        "--demo-summary-output",
        type=Path,
        default=settings.reports_dir_path / "demo_summary.md",
        help="Demo summary markdown path.",
    )
    parser.add_argument(
        "--demo-manifest-output",
        type=Path,
        default=settings.artifacts_dir_path / "demo" / "demo_manifest.json",
        help="Demo manifest output path.",
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
        help="Open local API docs URL in default browser.",
    )
    parser.add_argument(
        "--skip-monitoring",
        action="store_true",
        help="Skip monitoring generation stage.",
    )
    return parser.parse_args()


def main() -> int:
    settings = get_settings()
    args = parse_args()

    print("Starting local demo smoke run")
    print(f"- project_root: {settings.project_root}")
    print(f"- api_base_url: {args.api_base_url.rstrip('/')}")

    missing = validate_demo_prerequisites(settings.artifacts_dir_path)
    if missing:
        missing_text = "\n".join(f"  - {path}" for path in missing)
        raise RuntimeError(
            "Demo prerequisites are missing. Generate trained artifacts and "
            "sample payloads first:\n"
            + missing_text
        )

    command = [
        sys.executable,
        str(settings.project_root / "scripts" / "demo_generate_showcase_artifacts.py"),
        "--api-base-url",
        args.api_base_url,
        "--demo-output-dir",
        str(args.demo_output_dir),
        "--demo-summary-output",
        str(args.demo_summary_output),
        "--demo-manifest-output",
        str(args.demo_manifest_output),
        "--timeout-seconds",
        str(args.timeout_seconds),
    ]
    if args.open_docs:
        command.append("--open-docs")
    if args.skip_monitoring:
        command.append("--skip-monitoring")

    subprocess.run(command, cwd=settings.project_root, check=True)

    print("Demo smoke run completed successfully")
    print(f"- demo_summary: {args.demo_summary_output}")
    print(f"- demo_manifest: {args.demo_manifest_output}")
    print(f"- open_next: {args.api_base_url.rstrip('/')}/docs")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
