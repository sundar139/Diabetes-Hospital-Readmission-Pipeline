from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

EMPTY_OPTION = "(missing)"


def humanize_feature_name(name: str) -> str:
    return name.replace("_", " ").replace("-", " ").strip().title()


def coerce_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text == EMPTY_OPTION:
        return None
    return text


def safe_float(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def format_probability(value: Any) -> str:
    probability = safe_float(value)
    return f"{probability * 100:.1f}%"


def format_metric(value: Any, *, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def probability_mapping_to_frame(probabilities: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for label, value in probabilities.items():
        rows.append({"class": str(label), "probability": safe_float(value)})

    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["class", "probability"])
    return frame.sort_values("probability", ascending=False).reset_index(drop=True)


def resolve_existing_path(raw_path: Any, *, project_root: Path) -> Path | None:
    if raw_path is None:
        return None

    text = str(raw_path).strip()
    if not text:
        return None

    direct_candidate = Path(text)
    if direct_candidate.exists():
        return direct_candidate.resolve()

    normalized = text.replace("\\", "/")

    if normalized.startswith("./"):
        relative_candidate = (project_root / normalized[2:]).resolve()
        if relative_candidate.exists():
            return relative_candidate

    for anchor in ("artifacts/", "reports/", "docs/", "data/"):
        anchor_index = normalized.lower().find(anchor)
        if anchor_index >= 0:
            suffix = normalized[anchor_index:]
            relative_candidate = (project_root / Path(suffix)).resolve()
            if relative_candidate.exists():
                return relative_candidate

    if normalized.startswith("/") and len(normalized) > 2 and normalized[2] == ":":
        windows_drive_candidate = Path(normalized[1:])
        if windows_drive_candidate.exists():
            return windows_drive_candidate.resolve()

    return None
