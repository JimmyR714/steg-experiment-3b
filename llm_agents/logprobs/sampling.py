"""Extension 17: Diverse sampling and self-consistency.

Provides functions for generating diverse completions with varying
temperatures, measuring agreement across samples, and conformal
prediction for calibrated prediction sets.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from llm_agents.models.base import BaseModel
from llm_agents.models.types import CompletionResult


@dataclass
class ConsistencyResult:
    """Result of a self-consistency check across multiple samples.

    Attributes:
        answer: The majority answer.
        confidence: Fraction of samples agreeing with the majority.
        samples: All generated samples.
        vote_distribution: Counter mapping answer -> vote count.
    """

    answer: str
    confidence: float
    samples: list[str] = field(default_factory=list)
    vote_distribution: dict[str, int] = field(default_factory=dict)


@dataclass
class PredictionSet:
    """A conformal prediction set.

    Attributes:
        predictions: Ordered list of predictions by confidence.
        scores: Nonconformity scores for each prediction.
        alpha: Significance level used.
        coverage_guarantee: The guaranteed coverage probability (1-alpha).
    """

    predictions: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    alpha: float = 0.1
    coverage_guarantee: float = 0.9


def diverse_sample(
    model: BaseModel,
    prompt: str,
    n: int = 5,
    temperature_schedule: list[float] | None = None,
    max_tokens: int = 256,
) -> list[CompletionResult]:
    """Generate n completions with varying temperatures.

    Explores the output space by sampling at different temperatures.

    Args:
        model: The LLM model to use.
        prompt: The input prompt.
        n: Number of samples to generate.
        temperature_schedule: List of temperatures. If None, linearly
            spaces temperatures from 0.3 to 1.5.
        max_tokens: Maximum tokens per sample.

    Returns:
        List of CompletionResult objects.
    """
    if temperature_schedule is None:
        if n == 1:
            temperature_schedule = [0.7]
        else:
            temperature_schedule = [
                0.3 + (1.2 * i / (n - 1)) for i in range(n)
            ]

    # Extend schedule if n > len(schedule) by cycling
    while len(temperature_schedule) < n:
        temperature_schedule.append(temperature_schedule[-1])

    results: list[CompletionResult] = []
    for i in range(n):
        temp = temperature_schedule[i]
        result = model.generate(prompt, max_tokens=max_tokens, temperature=temp)
        results.append(result)

    return results


def _normalize_answer(text: str) -> str:
    """Normalize answer text for comparison."""
    return text.strip().lower()


def self_consistency(
    model: BaseModel,
    prompt: str,
    n: int = 5,
    temperature: float = 0.7,
    max_tokens: int = 256,
    normalize: bool = True,
) -> ConsistencyResult:
    """Generate n samples and measure agreement via majority vote.

    This implements the self-consistency method: sample multiple chain-of-thought
    reasoning paths and take the majority answer.

    Args:
        model: The LLM model.
        prompt: The input prompt.
        n: Number of samples.
        temperature: Temperature for all samples.
        max_tokens: Maximum tokens per sample.
        normalize: Whether to normalize answers before comparing.

    Returns:
        ConsistencyResult with the majority answer and confidence.
    """
    samples: list[str] = []
    for _ in range(n):
        result = model.generate(prompt, max_tokens=max_tokens, temperature=temperature)
        samples.append(result.text)

    # Count answers
    if normalize:
        normalized = [_normalize_answer(s) for s in samples]
    else:
        normalized = list(samples)

    counter = Counter(normalized)
    majority_answer, majority_count = counter.most_common(1)[0]

    # Map back to original form
    if normalize:
        for sample, norm in zip(samples, normalized):
            if norm == majority_answer:
                majority_answer = sample.strip()
                break

    return ConsistencyResult(
        answer=majority_answer,
        confidence=majority_count / n,
        samples=samples,
        vote_distribution=dict(counter),
    )


def conformal_prediction(
    model: BaseModel,
    prompt: str,
    calibration_data: list[tuple[str, str]],
    alpha: float = 0.1,
    n_samples: int = 10,
    max_tokens: int = 256,
) -> PredictionSet:
    """Build a conformal prediction set with guaranteed coverage.

    Uses split conformal prediction: calibration data determines a threshold,
    and the prediction set includes all answers with nonconformity scores
    below the threshold.

    Args:
        model: The LLM model.
        prompt: The input prompt to predict for.
        calibration_data: List of (prompt, correct_answer) pairs for
            calibration.
        alpha: Significance level. Coverage guarantee is 1-alpha.
        n_samples: Number of samples to generate for the test prompt.
        max_tokens: Maximum tokens per sample.

    Returns:
        A PredictionSet with guaranteed 1-alpha coverage.
    """
    # Step 1: Compute nonconformity scores on calibration data
    cal_scores: list[float] = []
    for cal_prompt, correct_answer in calibration_data:
        results = diverse_sample(model, cal_prompt, n=3, max_tokens=max_tokens)
        answers = [_normalize_answer(r.text) for r in results]
        correct_norm = _normalize_answer(correct_answer)

        # Nonconformity: 1 - (fraction of samples matching correct answer)
        match_count = sum(1 for a in answers if a == correct_norm)
        score = 1.0 - (match_count / len(answers))
        cal_scores.append(score)

    # Step 2: Compute quantile threshold
    cal_scores.sort()
    n_cal = len(cal_scores)
    if n_cal == 0:
        threshold = 1.0
    else:
        quantile_idx = min(
            n_cal - 1,
            int(math.ceil((1 - alpha) * (n_cal + 1))) - 1,
        )
        quantile_idx = max(0, quantile_idx)
        threshold = cal_scores[quantile_idx]

    # Step 3: Generate samples for the test prompt
    test_results = diverse_sample(model, prompt, n=n_samples, max_tokens=max_tokens)
    test_answers = [r.text.strip() for r in test_results]

    # Step 4: Compute nonconformity scores for each unique answer
    answer_counter = Counter(_normalize_answer(a) for a in test_answers)
    unique_answers: list[tuple[str, float]] = []

    for answer_norm, count in answer_counter.most_common():
        score = 1.0 - (count / n_samples)
        # Find original form
        original = next(a for a in test_answers if _normalize_answer(a) == answer_norm)
        unique_answers.append((original, score))

    # Step 5: Build prediction set (include answers below threshold)
    predictions: list[str] = []
    scores: list[float] = []
    for answer, score in unique_answers:
        if score <= threshold:
            predictions.append(answer)
            scores.append(score)

    # Always include at least the top answer
    if not predictions and unique_answers:
        predictions.append(unique_answers[0][0])
        scores.append(unique_answers[0][1])

    return PredictionSet(
        predictions=predictions,
        scores=scores,
        alpha=alpha,
        coverage_guarantee=1.0 - alpha,
    )
