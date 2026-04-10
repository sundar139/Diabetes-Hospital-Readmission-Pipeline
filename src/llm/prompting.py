from __future__ import annotations

from collections.abc import Mapping


def build_prediction_summary(
    *,
    binary_prediction: int,
    binary_probability: float,
    multiclass_prediction: str,
    multiclass_probabilities: Mapping[str, float],
) -> str:
    probability_text = f"{binary_probability * 100:.1f}%"

    if multiclass_probabilities:
        sorted_multiclass = sorted(
            multiclass_probabilities.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        multiclass_text = ", ".join(
            f"{label}:{value * 100:.1f}%" for label, value in sorted_multiclass
        )
    else:
        multiclass_text = "unavailable"

    return (
        "Binary early-readmission prediction="
        f"{binary_prediction} ({probability_text} probability). "
        "Multiclass prediction="
        f"{multiclass_prediction}. "
        f"Multiclass probabilities: {multiclass_text}."
    )


def build_fallback_explanation_text(
    *,
    prediction_summary: str,
    increasing_factors: list[str],
    decreasing_factors: list[str],
) -> str:
    increasing_text = "; ".join(increasing_factors) if increasing_factors else "none identified"
    decreasing_text = "; ".join(decreasing_factors) if decreasing_factors else "none identified"

    return (
        f"{prediction_summary} "
        f"Risk-increasing factors: {increasing_text}. "
        f"Risk-decreasing factors: {decreasing_text}. "
        "This explanation is generated from model behavior and feature patterns for demo use only. "
        "It is not medical advice and should not replace clinical judgment."
    )


def build_ollama_system_prompt() -> str:
    return (
        "You are assisting with machine-learning readmission explanations for stakeholders. "
        "Be concise, cautious, and transparent about uncertainty. "
        "Do not present the output as medical advice. "
        "Do not claim causality; describe model signals only."
    )


def build_ollama_user_prompt(
    *,
    prediction_summary: str,
    increasing_factors: list[str],
    decreasing_factors: list[str],
) -> str:
    increasing_text = "\n".join(f"- {item}" for item in increasing_factors) or "- none"
    decreasing_text = "\n".join(f"- {item}" for item in decreasing_factors) or "- none"

    return (
        "Write a short explanation for a model prediction using the information below.\n\n"
        f"Prediction summary:\n{prediction_summary}\n\n"
        f"Top risk-increasing factors:\n{increasing_text}\n\n"
        f"Top risk-decreasing factors:\n{decreasing_text}\n\n"
        "Requirements:\n"
        "- 3 to 5 sentences\n"
        "- Mention that this is model-generated and non-diagnostic\n"
        "- Keep tone careful and stakeholder-friendly\n"
    )
