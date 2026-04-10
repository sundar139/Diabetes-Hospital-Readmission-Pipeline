from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from src.config.settings import get_settings
from src.features.build_features import (
    ENGINEERED_FEATURE_NAMES,
    FeatureEngineeringResult,
    detect_medication_status_columns,
    engineer_clinical_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build clinically informed feature datasets from processed splits."
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help="Optional processed-data directory. Defaults to configured data/processed.",
    )
    return parser.parse_args()


def _resolve_processed_paths(processed_dir: Path) -> dict[str, Path]:
    return {
        "train": processed_dir / "train.parquet",
        "val": processed_dir / "val.parquet",
        "test": processed_dir / "test.parquet",
    }


def _read_split_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Processed split file not found: {path}. Run scripts/build_processed_data.py first."
        )
    return pd.read_parquet(path)


def _table(headers: list[str], rows: list[list[str]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    divider_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    row_lines = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, divider_line, *row_lines])


def _engineered_stats_table(frame: pd.DataFrame) -> str:
    stats = frame[list(ENGINEERED_FEATURE_NAMES)].describe().transpose()
    rows: list[list[str]] = []
    for column_name, row in stats.iterrows():
        rows.append(
            [
                column_name,
                f"{float(row['mean']):.4f}",
                f"{float(row['std']):.4f}",
                f"{float(row['min']):.4f}",
                f"{float(row['max']):.4f}",
            ]
        )

    return _table(["Feature", "Mean", "Std", "Min", "Max"], rows)


def _engineered_missingness_table(frame: pd.DataFrame) -> str:
    n_rows = len(frame)
    rows: list[list[str]] = []
    for feature_name in ENGINEERED_FEATURE_NAMES:
        missing_count = int(frame[feature_name].isna().sum())
        missing_rate = (missing_count / n_rows) if n_rows else 0.0
        rows.append([feature_name, str(missing_count), f"{missing_rate:.2%}"])

    return _table(["Feature", "Missing Count", "Missing Rate"], rows)


def _render_feature_report(
    split_results: dict[str, FeatureEngineeringResult],
    metadata: dict[str, Any],
) -> str:
    logic_lines = [
        (
            "- recurrency: number of encounters for the same patient_nbr within "
            "the split minus 1, clipped at 0."
        ),
        (
            "- patient_severity: 0.30*(time_in_hospital/14 clipped) + "
            "0.25*(diagnoses_count/16 clipped) + "
            "0.20*(number_inpatient/10 clipped) + 0.15*(number_outpatient/10 clipped) + "
            "0.10*(number_emergency/10 clipped)."
        ),
        (
            "- medication_change_ratio: count(status in {Up, Down}) / "
            "count(status in {No, Steady, Up, Down}) across medication columns."
        ),
        "- utilization_intensity: number_inpatient + number_outpatient + number_emergency.",
        (
            "- complex_discharge_flag: 1 when discharge_disposition_id is not in "
            "{1, 6, 8}; missing/malformed defaults to 1."
        ),
        (
            "- age_bucket_risk: lower decade extracted from age bucket [a-b) and "
            "mapped to a//10; missing/malformed maps to -1."
        ),
    ]

    relevance_lines = [
        (
            "- recurrency captures repeated-patient encounter burden, "
            "a strong operational readmission context signal."
        ),
        (
            "- patient_severity summarizes acute burden and prior utilization "
            "into a single clinically interpretable index."
        ),
        "- medication_change_ratio approximates treatment-instability intensity during admission.",
        "- utilization_intensity captures cumulative prior care-touch volume.",
        (
            "- complex_discharge_flag reflects transition-of-care complexity "
            "and potential post-discharge risk."
        ),
        "- age_bucket_risk injects age-associated baseline vulnerability in ordinal form.",
    ]

    assumption_lines = [
        (
            "- recurrency uses split-local repeated-patient encounter counts "
            "because explicit temporal ordering is unavailable in split files."
        ),
        (
            "- complex_discharge_flag assumes discharge disposition IDs {1, 6, 8} "
            "represent home-like discharges."
        ),
        (
            "- medication_change_ratio uses diabetes medication-status columns "
            "detected from train split and applied consistently to val/test."
        ),
    ]

    limitation_lines = [
        "- recurrency is a deterministic proxy and does not claim encounter chronology.",
        (
            "- age_bucket_risk uses coarse decade-level bins and does not model "
            "nonlinear age effects yet."
        ),
        "- no learned scaling or encoding is applied at this stage by design.",
    ]

    fallback_lines: list[str] = []
    for split_name, result in split_results.items():
        fallback_parts = [f"{key}={value}" for key, value in result.fallback_counts.items()]
        fallback_lines.append(f"- {split_name}: " + ", ".join(fallback_parts))

    split_sections: list[str] = []
    for split_name, result in split_results.items():
        split_sections.extend(
            [
                f"## Engineered Feature Stats ({split_name})",
                "",
                _engineered_stats_table(result.frame),
                "",
                f"## Engineered Feature Missingness ({split_name})",
                "",
                _engineered_missingness_table(result.frame),
                "",
            ]
        )

    return "\n".join(
        [
            "# Feature Engineering Report",
            "",
            "## Engineered Feature Logic",
            "",
            *logic_lines,
            "",
            "## Feature Source Columns",
            "",
            _table(
                ["Engineered Feature", "Source Columns"],
                [
                    [feature, ", ".join(columns)]
                    for feature, columns in metadata["feature_source_map"].items()
                ],
            ),
            "",
            "## Clinical and Operational Relevance",
            "",
            *relevance_lines,
            "",
            "## Assumptions",
            "",
            *assumption_lines,
            "",
            "## Limitations",
            "",
            *limitation_lines,
            "",
            "## Fallback Behavior",
            "",
            *fallback_lines,
            "",
            *split_sections,
            "## Recommended Next Step For Modeling",
            "",
            (
                "- Train baseline models with model_candidate_columns and compare "
                "against ablations that remove engineered features."
            ),
            "",
        ]
    )


def _save_feature_splits(
    split_results: dict[str, FeatureEngineeringResult],
    processed_dir: Path,
) -> dict[str, Path]:
    processed_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {
        "train": processed_dir / "train_features.parquet",
        "val": processed_dir / "val_features.parquet",
        "test": processed_dir / "test_features.parquet",
    }

    expected_columns = list(split_results["train"].frame.columns)
    for split_name, result in split_results.items():
        split_frame = result.frame
        if list(split_frame.columns) != expected_columns:
            raise ValueError(
                f"Feature columns mismatch detected in split '{split_name}'."
            )
        split_frame.to_parquet(output_paths[split_name], index=False)

    return output_paths


def main() -> int:
    args = parse_args()
    settings = get_settings()

    processed_dir = (
        args.processed_dir.resolve()
        if args.processed_dir is not None
        else settings.processed_data_dir_path
    )

    input_paths = _resolve_processed_paths(processed_dir)
    train_frame = _read_split_frame(input_paths["train"])
    val_frame = _read_split_frame(input_paths["val"])
    test_frame = _read_split_frame(input_paths["test"])

    medication_columns = detect_medication_status_columns(train_frame)

    split_results = {
        "train": engineer_clinical_features(
            train_frame,
            medication_status_columns=medication_columns,
        ),
        "val": engineer_clinical_features(
            val_frame,
            medication_status_columns=medication_columns,
        ),
        "test": engineer_clinical_features(
            test_frame,
            medication_status_columns=medication_columns,
        ),
    }

    output_paths = _save_feature_splits(split_results, processed_dir)

    bookkeeping = split_results["train"].bookkeeping
    metadata = {
        "engineered_feature_names": list(bookkeeping.engineered_feature_columns),
        "target_columns": list(bookkeeping.target_columns),
        "identifier_columns": list(bookkeeping.identifier_columns),
        "excluded_feature_columns": list(bookkeeping.excluded_feature_columns),
        "model_candidate_columns": list(bookkeeping.model_candidate_columns),
        "feature_source_map": bookkeeping.feature_source_map,
        "split_output_paths": {name: str(path) for name, path in output_paths.items()},
    }

    metadata_path = settings.artifacts_dir_path / "feature_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    report_path = settings.reports_dir_path / "feature_engineering_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        _render_feature_report(split_results, metadata),
        encoding="utf-8",
    )

    print("Feature set build completed.")
    print(f"- train_features: {output_paths['train']}")
    print(f"- val_features: {output_paths['val']}")
    print(f"- test_features: {output_paths['test']}")
    print(f"- feature_metadata: {metadata_path}")
    print(f"- feature_report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
