"""Evaluation metrics for measuring agent quality."""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from llm_agents.models.base import BaseModel


@dataclass
class Score:
    """A metric score with optional explanation.

    Attributes:
        value: Numeric score, typically in [0, 1].
        explanation: Optional human-readable explanation of the score.
    """

    value: float
    explanation: str = ""


# ------------------------------------------------------------------
# Built-in metric functions
# ------------------------------------------------------------------


def exact_match(predicted: str, expected: str) -> float:
    """Return 1.0 if predicted matches expected exactly, else 0.0.

    Args:
        predicted: The model's output.
        expected: The reference answer.

    Returns:
        1.0 or 0.0.
    """
    return 1.0 if predicted.strip() == expected.strip() else 0.0


def fuzzy_match(predicted: str, expected: str, threshold: float = 0.8) -> float:
    """Return the sequence similarity ratio, thresholded.

    Uses :class:`difflib.SequenceMatcher` to compute a similarity ratio.
    Returns the ratio if it meets the threshold, otherwise 0.0.

    Args:
        predicted: The model's output.
        expected: The reference answer.
        threshold: Minimum similarity to count as a match.

    Returns:
        The similarity ratio if >= threshold, else 0.0.
    """
    ratio = SequenceMatcher(None, predicted.strip(), expected.strip()).ratio()
    return ratio if ratio >= threshold else 0.0


def contains_match(predicted: str, expected: str) -> float:
    """Return 1.0 if expected is contained within predicted.

    Args:
        predicted: The model's output.
        expected: The reference answer.

    Returns:
        1.0 if expected is a substring of predicted, else 0.0.
    """
    return 1.0 if expected.strip() in predicted.strip() else 0.0


def normalized_contains(predicted: str, expected: str) -> float:
    """Case-insensitive containment check.

    Args:
        predicted: The model's output.
        expected: The reference answer.

    Returns:
        1.0 if expected is found in predicted (case-insensitive), else 0.0.
    """
    return 1.0 if expected.strip().lower() in predicted.strip().lower() else 0.0


def llm_judge(
    predicted: str,
    expected: str,
    judge_model: BaseModel,
    rubric: str = "",
) -> Score:
    """Use an LLM to grade the quality of a response.

    Args:
        predicted: The model's output to evaluate.
        expected: The reference answer.
        judge_model: The LLM to use as judge.
        rubric: Optional grading rubric or criteria.

    Returns:
        A :class:`Score` with the judge's assessment.
    """
    rubric_text = f"\nGrading rubric: {rubric}" if rubric else ""

    prompt = (
        "You are an evaluation judge. Score the following response on a "
        "scale of 0.0 to 1.0.\n\n"
        f"Expected answer: {expected}\n\n"
        f"Actual response: {predicted}\n"
        f"{rubric_text}\n\n"
        "Respond with ONLY a JSON object: "
        '{"score": <float 0.0-1.0>, "explanation": "<brief reason>"}'
    )

    result = judge_model.generate(prompt, max_tokens=256, temperature=0.1)

    # Try to parse the score from the response
    import json
    import re

    text = result.text.strip()
    try:
        data = json.loads(text)
        return Score(
            value=float(data.get("score", 0.0)),
            explanation=data.get("explanation", ""),
        )
    except (json.JSONDecodeError, ValueError):
        # Fallback: try to find a number in the response
        numbers = re.findall(r"(\d+\.?\d*)", text)
        if numbers:
            value = min(float(numbers[0]), 1.0)
            return Score(value=value, explanation=text)
        return Score(value=0.0, explanation=f"Could not parse judge response: {text}")


def factual_consistency(
    predicted: str,
    context: str,
    judge_model: BaseModel,
) -> float:
    """Check if the predicted response is supported by the given context.

    Args:
        predicted: The model's output.
        context: The reference context that should support the response.
        judge_model: An LLM to use for assessment.

    Returns:
        A score in [0, 1] indicating factual consistency.
    """
    prompt = (
        "Determine if the following response is factually consistent with "
        "the provided context. Score from 0.0 (completely inconsistent) to "
        "1.0 (fully supported).\n\n"
        f"Context: {context}\n\n"
        f"Response: {predicted}\n\n"
        "Respond with ONLY a number between 0.0 and 1.0."
    )
    result = judge_model.generate(prompt, max_tokens=32, temperature=0.1)

    import re

    numbers = re.findall(r"(\d+\.?\d*)", result.text.strip())
    if numbers:
        return min(float(numbers[0]), 1.0)
    return 0.0


class Composite:
    """Weighted combination of multiple metric functions.

    Args:
        metrics: List of metric callables ``(predicted, expected) -> float``.
        weights: Optional weights for each metric.  If not provided,
            equal weights are used.

    Raises:
        ValueError: If metrics and weights have different lengths.
    """

    def __init__(
        self,
        metrics: list[Callable[[str, str], float]],
        weights: list[float] | None = None,
    ) -> None:
        if weights is not None and len(metrics) != len(weights):
            raise ValueError("metrics and weights must have the same length")
        self._metrics = metrics
        self._weights = weights or [1.0 / len(metrics)] * len(metrics)

    def __call__(self, predicted: str, expected: str) -> float:
        """Compute the weighted composite score.

        Args:
            predicted: The model's output.
            expected: The reference answer.

        Returns:
            The weighted average score.
        """
        total = 0.0
        for metric, weight in zip(self._metrics, self._weights):
            total += weight * metric(predicted, expected)
        return total
