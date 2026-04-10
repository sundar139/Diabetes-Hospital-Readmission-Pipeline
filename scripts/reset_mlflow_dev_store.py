from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from src.config.settings import Settings, get_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reset local MLflow development metadata and artifacts for this project."
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip interactive confirmation.",
    )
    return parser.parse_args()


def _collect_reset_targets(settings: Settings) -> tuple[Path | None, Path | None]:
    db_path = settings.mlflow_backend_store_path

    try:
        artifacts_path = settings.mlflow_artifacts_destination_path
    except ValueError:
        artifacts_path = None

    return db_path, artifacts_path


def _assert_safe_targets(
    *,
    settings: Settings,
    db_path: Path | None,
    artifacts_path: Path | None,
) -> None:
    protected_paths = {
        settings.artifacts_dir_path.resolve(),
        settings.processed_data_dir_path.resolve(),
        settings.raw_data_dir_path.resolve(),
    }

    for target in (db_path, artifacts_path):
        if target is None:
            continue
        if target.resolve() in protected_paths:
            raise RuntimeError(
                f"Refusing to delete protected project path: {target}"
            )


def reset_mlflow_dev_store(
    *,
    settings: Settings | None = None,
    assume_yes: bool = False,
) -> int:
    resolved_settings = settings or get_settings()
    db_path, artifacts_path = _collect_reset_targets(resolved_settings)

    _assert_safe_targets(
        settings=resolved_settings,
        db_path=db_path,
        artifacts_path=artifacts_path,
    )

    print("This will delete local MLflow development state for this project only:")
    print(f"- metadata_db: {db_path if db_path is not None else 'not-applicable'}")
    print(
        "- artifacts_dir: "
        f"{artifacts_path if artifacts_path is not None else 'non-local URI (skipped)'}"
    )
    print(f"- project_artifacts_preserved: {resolved_settings.artifacts_dir_path}")
    print(f"- processed_data_preserved: {resolved_settings.processed_data_dir_path}")

    if not assume_yes:
        confirmation = input("Type RESET to continue: ").strip()
        if confirmation != "RESET":
            print("Aborted. No files were deleted.")
            return 1

    removed: list[Path] = []

    if db_path is not None and db_path.exists():
        db_path.unlink()
        removed.append(db_path)

    if artifacts_path is not None and artifacts_path.exists():
        shutil.rmtree(artifacts_path)
        removed.append(artifacts_path)

    if removed:
        print("Deleted:")
        for path in removed:
            print(f"- {path}")
    else:
        print("Nothing to delete. Local MLflow dev store is already clean.")

    return 0


def main() -> int:
    args = parse_args()
    try:
        return reset_mlflow_dev_store(assume_yes=args.yes)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
