from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config.settings import get_settings
from src.data.data_dictionary import build_data_dictionary, write_data_dictionary_markdown
from src.data.load_raw import load_raw_data
from src.data.validate_raw import (
    build_raw_validation_summary,
    generate_readmitted_distribution_figure,
    write_validation_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run raw dataset validation and generate project reports."
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=None,
        help="Optional path to the raw CSV. Defaults to configured data/raw/diabetic_data.csv.",
    )
    parser.add_argument(
        "--skip-figure",
        action="store_true",
        help="Skip generation of the readmitted class distribution figure.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()

    try:
        frame = load_raw_data(csv_path=args.csv_path, settings=settings)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    summary = build_raw_validation_summary(frame, target_column=settings.target_column)
    output_paths = write_validation_outputs(summary, settings=settings)

    dictionary_records = build_data_dictionary(frame, target_column=settings.target_column)
    dictionary_path = settings.reports_dir_path / "data_dictionary.md"
    write_data_dictionary_markdown(dictionary_records, dictionary_path)
    output_paths["data_dictionary"] = dictionary_path

    if not args.skip_figure:
        figure_path = settings.figures_dir_path / "readmitted_class_distribution.png"
        generated_figure = generate_readmitted_distribution_figure(
            frame,
            figure_path,
            target_column=settings.target_column,
        )
        if generated_figure is not None:
            output_paths["readmitted_class_distribution"] = generated_figure

    print("Raw validation completed.")
    for name, path in output_paths.items():
        print(f"- {name}: {path}")

    if summary["warnings"]:
        print("\nWarnings:")
        for warning in summary["warnings"]:
            print(f"- {warning}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
