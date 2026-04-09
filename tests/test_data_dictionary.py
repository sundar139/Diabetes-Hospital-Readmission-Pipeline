from __future__ import annotations

from pathlib import Path

import pandas as pd
from src.data.data_dictionary import build_data_dictionary, write_data_dictionary_markdown


def test_data_dictionary_has_expected_structure_and_roles() -> None:
    frame = pd.DataFrame(
        {
            "encounter_id": [1, 2],
            "patient_nbr": [10, 10],
            "readmitted": ["NO", "<30"],
            "metformin": ["No", "Up"],
            "diag_1": ["250.83", "401"],
            "number_emergency": [0, 1],
            "race": ["Caucasian", "AfricanAmerican"],
        }
    )

    records = build_data_dictionary(frame)
    by_column = {record["column_name"]: record for record in records}

    required_record_keys = {
        "column_name",
        "role",
        "dtype",
        "missing_token_rate",
        "non_null_count",
        "unique_count",
        "example_values",
        "description",
        "is_identifier",
        "is_target",
        "is_medication",
        "is_diagnosis",
        "is_utilization",
    }

    assert required_record_keys.issubset(records[0].keys())
    assert by_column["encounter_id"]["role"] == "identifier"
    assert by_column["readmitted"]["role"] == "target"
    assert by_column["metformin"]["role"] == "medication-status"
    assert by_column["diag_1"]["role"] == "diagnosis-like"
    assert by_column["number_emergency"]["is_utilization"] is True


def test_data_dictionary_markdown_is_written(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "encounter_id": [1],
            "readmitted": ["NO"],
        }
    )
    records = build_data_dictionary(frame)
    output_path = tmp_path / "data_dictionary.md"

    written_path = write_data_dictionary_markdown(records, output_path)
    content = written_path.read_text(encoding="utf-8")

    assert written_path.exists()
    assert "| Column | Role |" in content
    assert "encounter_id" in content
