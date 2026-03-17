"""Extension 17: Logprob-based uncertainty quantification.

Provides functions for computing confidence scores, identifying uncertain
tokens, detecting hallucination risk, and measuring calibration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from llm_agents.logprobs.ops import entropy, perplexity
from llm_agents.models.types import LogProbResult, TokenLogProb


def confidence_score(logprob_result: LogProbResult) -> float:
    """Compute an aggregate confidence score from sequence log-probabilities.

    Uses normalized perplexity: confidence = 1 / perplexity, clamped to [0, 1].
    Lower perplexity means higher confidence.

    Args:
        logprob_result: Log-probability result from a model generation.

    Returns:
        Confidence score in [0, 1]. Higher is more confident.
    """
    if not logprob_result.tokens:
        return 0.0

    ppl = perplexity(logprob_result)
    # Map perplexity to [0, 1]: ppl=1 -> confidence=1, ppl→∞ -> confidence→0
    return 1.0 / ppl


def token_uncertainty_map(
    logprob_result: LogProbResult,
) -> list[tuple[str, float]]:
    """Compute per-token uncertainty values.

    Uncertainty for each token is defined as the negative log-probability
    (surprise / information content). Higher values indicate more
    uncertain tokens.

    Args:
        logprob_result: Log-probability result from generation.

    Returns:
        List of (token_text, uncertainty) tuples.
    """
    result: list[tuple[str, float]] = []
    for tlp in logprob_result.tokens:
        uncertainty = -tlp.logprob
        result.append((tlp.token, uncertainty))
    return result


def entropy_map(logprob_result: LogProbResult) -> list[tuple[str, float]]:
    """Compute per-position entropy from top-k alternatives.

    Uses the top-k distributions at each position to estimate the
    model's uncertainty about each token choice.

    Args:
        logprob_result: Log-probability result with top-k data.

    Returns:
        List of (chosen_token, entropy) tuples.
    """
    result: list[tuple[str, float]] = []
    tokens = logprob_result.tokens
    top_k = logprob_result.top_k_per_position

    for i, tlp in enumerate(tokens):
        if i < len(top_k) and top_k[i]:
            h = entropy(top_k[i])
        else:
            # Fallback: use surprise as a proxy
            h = -tlp.logprob
        result.append((tlp.token, h))

    return result


def is_hallucination_risk(
    logprob_result: LogProbResult,
    threshold: float = 3.0,
    min_span_length: int = 3,
) -> bool:
    """Flag responses where large spans have high uncertainty.

    Checks whether there exists a contiguous span of tokens where the
    average surprise (negative log-prob) exceeds the threshold.

    Args:
        logprob_result: Log-probability result from generation.
        threshold: Surprise threshold. Tokens with surprise above this
            are considered uncertain.
        min_span_length: Minimum number of consecutive uncertain tokens
            to trigger the flag.

    Returns:
        True if the response is likely to contain hallucinations.
    """
    if not logprob_result.tokens:
        return False

    consecutive = 0
    for tlp in logprob_result.tokens:
        if -tlp.logprob > threshold:
            consecutive += 1
            if consecutive >= min_span_length:
                return True
        else:
            consecutive = 0

    return False


def uncertain_spans(
    logprob_result: LogProbResult,
    threshold: float = 3.0,
    min_length: int = 2,
) -> list[list[tuple[str, float]]]:
    """Find contiguous spans of uncertain tokens.

    Args:
        logprob_result: Log-probability result.
        threshold: Surprise threshold for uncertainty.
        min_length: Minimum span length to include.

    Returns:
        List of spans, where each span is a list of (token, surprise) tuples.
    """
    spans: list[list[tuple[str, float]]] = []
    current_span: list[tuple[str, float]] = []

    for tlp in logprob_result.tokens:
        surprise = -tlp.logprob
        if surprise > threshold:
            current_span.append((tlp.token, surprise))
        else:
            if len(current_span) >= min_length:
                spans.append(current_span)
            current_span = []

    if len(current_span) >= min_length:
        spans.append(current_span)

    return spans


@dataclass
class CalibrationPoint:
    """A single point on the calibration curve."""

    predicted_confidence: float
    actual_accuracy: float
    count: int


def calibration_curve(
    predictions: list[float],
    actuals: list[bool],
    n_bins: int = 10,
) -> list[CalibrationPoint]:
    """Compute a calibration curve from predicted confidences and actual outcomes.

    Bins predictions by confidence and computes the actual accuracy in each bin.
    A well-calibrated model has predicted_confidence ≈ actual_accuracy.

    Args:
        predictions: Predicted confidence scores in [0, 1].
        actuals: Whether each prediction was actually correct.
        n_bins: Number of bins to divide the [0, 1] interval into.

    Returns:
        List of CalibrationPoints, one per non-empty bin.

    Raises:
        ValueError: If predictions and actuals have different lengths.
    """
    if len(predictions) != len(actuals):
        raise ValueError("predictions and actuals must have the same length.")

    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]

    for pred, actual in zip(predictions, actuals):
        bin_idx = min(int(pred * n_bins), n_bins - 1)
        bins[bin_idx].append((pred, actual))

    curve: list[CalibrationPoint] = []
    for b in bins:
        if not b:
            continue
        avg_pred = sum(p for p, _ in b) / len(b)
        avg_actual = sum(1.0 for _, a in b if a) / len(b)
        curve.append(
            CalibrationPoint(
                predicted_confidence=avg_pred,
                actual_accuracy=avg_actual,
                count=len(b),
            )
        )

    return curve


def expected_calibration_error(
    predictions: list[float],
    actuals: list[bool],
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE).

    ECE is the weighted average of |accuracy - confidence| across bins.

    Args:
        predictions: Predicted confidence scores.
        actuals: Whether each prediction was correct.
        n_bins: Number of bins.

    Returns:
        ECE value in [0, 1]. Lower is better.
    """
    curve = calibration_curve(predictions, actuals, n_bins)
    total = sum(p.count for p in curve)
    if total == 0:
        return 0.0

    ece = 0.0
    for point in curve:
        ece += point.count * abs(point.actual_accuracy - point.predicted_confidence)
    return ece / total
