from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.config.settings import get_settings
from src.data.load_raw import load_raw_data
from src.data.preprocess import (
    FeatureCandidacy,
    build_feature_candidacy,
    build_preprocessing_summary,
    derive_binary_readmission_target,
    replace_null_like_tokens,
)
from src.data.split import (
    GroupedSplitResult,
    build_split_manifest,
    grouped_split_by_patient,
    write_split_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build processed datasets with preprocessing and grouped patient split."
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=None,
        help="Optional path to source CSV. Defaults to configured raw dataset path.",
    )
    return parser.parse_args()


def _table(headers: list[str], rows: list[list[str]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    divider_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    row_lines = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, divider_line, *row_lines])


def _missingness_table(items: list[dict[str, Any]], *, top_n: int = 20) -> str:
    rows: list[list[str]] = []
    for item in items[:top_n]:
        rows.append(
            [
                str(item["column"]),
                str(item["missing_count"]),
                f"{float(item['missing_rate']):.2%}",
            ]
        )

    return _table(["Column", "Missing Count", "Missing Rate"], rows)


def _distribution_table(distribution: dict[str, int], column_name: str) -> str:
    rows = [[label, str(count)] for label, count in distribution.items()]
    return _table([column_name, "Count"], rows)


def _render_processed_report(
    preprocessing_summary: dict[str, Any],
    split_manifest: dict[str, Any],
    feature_candidacy: FeatureCandidacy,
) -> str:
    shape_original = preprocessing_summary["shape_original"]
    shape_after = preprocessing_summary["shape_after_preprocessing"]

    readmitted_distribution = preprocessing_summary["target_distribution_readmitted"]
    readmitted_30d_distribution = preprocessing_summary["target_distribution_readmitted_30d"]

    warnings = preprocessing_summary["warnings"]
    warning_lines = [f"- {warning}" for warning in warnings] if warnings else ["- None"]

    assumptions = [
        (
            "- Grouped split uses patient_nbr and allows approximate row ratios "
            "due to group boundaries."
        ),
        "- readmitted is expected to have only NO, >30, <30 labels without missing values.",
        "- encounter_id and patient_nbr are excluded from model feature candidates.",
    ]

    recommended_next_steps = [
        (
            "- Implement feature engineering on train split only, then apply "
            "transformations to val/test."
        ),
        "- Define explicit imputation strategy for high-missingness columns.",
        "- Build model-ready feature matrix using candidate_feature_columns metadata.",
    ]

    return "\n".join(
        [
            "# Processed Data Report",
            "",
            "## Original Shape",
            "",
            f"- Rows: {shape_original['n_rows']}",
            f"- Columns: {shape_original['n_columns']}",
            "",
            "## Shape After Preprocessing",
            "",
            f"- Rows: {shape_after['n_rows']}",
            f"- Columns: {shape_after['n_columns']}",
            "",
            "## Missingness Summary Before Normalization",
            "",
            _missingness_table(preprocessing_summary["missingness_before_normalization"]),
            "",
            "## Missingness Summary After Normalization",
            "",
            _missingness_table(preprocessing_summary["missingness_after_normalization"]),
            "",
            "## Columns Excluded From Feature Candidates",
            "",
            *(f"- {column}" for column in feature_candidacy.excluded_feature_columns),
            "",
            "## Target Distribution (readmitted)",
            "",
            _distribution_table(readmitted_distribution, "readmitted"),
            "",
            "## Target Distribution (readmitted_30d)",
            "",
            _distribution_table(readmitted_30d_distribution, "readmitted_30d"),
            "",
            "## Split Row Counts",
            "",
            f"- Train rows: {split_manifest['n_rows_train']}",
            f"- Validation rows: {split_manifest['n_rows_val']}",
            f"- Test rows: {split_manifest['n_rows_test']}",
            "",
            "## Split Patient Counts",
            "",
            f"- Train patients: {split_manifest['n_patients_train']}",
            f"- Validation patients: {split_manifest['n_patients_val']}",
            f"- Test patients: {split_manifest['n_patients_test']}",
            "",
            "## Leakage Check Result",
            "",
            f"- leakage_check_passed: {split_manifest['leakage_check_passed']}",
            "",
            "## Warnings",
            "",
            *warning_lines,
            "",
            "## Assumptions",
            "",
            *assumptions,
            "",
            "## Recommended Next Step For Feature Engineering/Modeling",
            "",
            *recommended_next_steps,
            "",
        ]
    )


def _save_split_parquets(
    split_result: GroupedSplitResult,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val.parquet"
    test_path = output_dir / "test.parquet"

    split_result.train.to_parquet(train_path, index=False)
    split_result.val.to_parquet(val_path, index=False)
    split_result.test.to_parquet(test_path, index=False)

    return {
        "train": train_path,
        "val": val_path,
        "test": test_path,
    }


def main() -> int:
    args = parse_args()
    settings = get_settings()

    raw_frame = load_raw_data(csv_path=args.csv_path, settings=settings)
    normalized_frame = replace_null_like_tokens(raw_frame)
    preprocessed_frame = derive_binary_readmission_target(
        normalized_frame,
        target_column=settings.target_column,
        binary_target_column="readmitted_30d",
        positive_label=settings.binary_positive_label,
    )

    feature_candidacy = build_feature_candidacy(
        preprocessed_frame,
        target_columns=(settings.target_column, "readmitted_30d"),
    )
    preprocessing_summary = build_preprocessing_summary(
        raw_frame,
        preprocessed_frame,
        feature_candidacy,
        target_column=settings.target_column,
        binary_target_column="readmitted_30d",
    )

    split_result = grouped_split_by_patient(
        preprocessed_frame,
        group_column="patient_nbr",
        train_size=0.70,
        val_size=0.15,
        test_size=0.15,
        random_state=settings.random_state,
    )

    parquet_paths = _save_split_parquets(split_result, settings.processed_data_dir_path)

    split_manifest = build_split_manifest(
        split_result,
        group_column="patient_nbr",
        multiclass_target_column=settings.target_column,
        binary_target_column="readmitted_30d",
        excluded_feature_columns=feature_candidacy.excluded_feature_columns,
        random_state=settings.random_state,
    )

    manifest_path = write_split_manifest(
        split_manifest,
        settings.artifacts_dir_path / "split_manifest.json",
    )

    processed_report_path = settings.reports_dir_path / "processed_data_report.md"
    processed_report_path.parent.mkdir(parents=True, exist_ok=True)
    processed_report_path.write_text(
        _render_processed_report(preprocessing_summary, split_manifest, feature_candidacy),
        encoding="utf-8",
    )

    print("Processed data build completed.")
    print(f"- train_parquet: {parquet_paths['train']}")
    print(f"- val_parquet: {parquet_paths['val']}")
    print(f"- test_parquet: {parquet_paths['test']}")
    print(f"- split_manifest: {manifest_path}")
    print(f"- processed_report: {processed_report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
