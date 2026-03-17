"""Comparison and statistical significance testing for evaluation reports."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from llm_agents.eval.runner import EvalReport


@dataclass
class ComparisonResult:
    """Side-by-side comparison of multiple evaluation reports.

    Attributes:
        reports: The reports being compared.
        summary: A list of dicts with per-report summary statistics.
        winner: Name of the report with the highest aggregate score.
        significant: Whether the difference between the top two is
            statistically significant (via bootstrap test).
        p_value: Bootstrap p-value for the top-two comparison.
    """

    reports: list[EvalReport] = field(default_factory=list)
    summary: list[dict[str, Any]] = field(default_factory=list)
    winner: str = ""
    significant: bool = False
    p_value: float = 1.0


def _bootstrap_p_value(
    scores_a: list[float],
    scores_b: list[float],
    n_bootstrap: int = 10000,
    seed: int | None = 42,
) -> float:
    """Compute a bootstrap p-value testing whether A and B differ.

    Tests the null hypothesis that the mean of A equals the mean of B.

    Args:
        scores_a: Scores from the first system.
        scores_b: Scores from the second system.
        n_bootstrap: Number of bootstrap iterations.
        seed: Random seed for reproducibility.

    Returns:
        The two-sided p-value.
    """
    if not scores_a or not scores_b:
        return 1.0

    rng = random.Random(seed)
    observed_diff = abs(
        sum(scores_a) / len(scores_a) - sum(scores_b) / len(scores_b)
    )

    combined = scores_a + scores_b
    n_a = len(scores_a)
    count_extreme = 0

    for _ in range(n_bootstrap):
        rng.shuffle(combined)
        sample_a = combined[:n_a]
        sample_b = combined[n_a:]
        mean_a = sum(sample_a) / len(sample_a)
        mean_b = sum(sample_b) / len(sample_b)
        if abs(mean_a - mean_b) >= observed_diff:
            count_extreme += 1

    return count_extreme / n_bootstrap


def compare(
    reports: list[EvalReport],
    significance_level: float = 0.05,
    n_bootstrap: int = 10000,
) -> ComparisonResult:
    """Compare multiple evaluation reports side-by-side.

    Args:
        reports: List of :class:`EvalReport` instances to compare.
        significance_level: Threshold for statistical significance.
        n_bootstrap: Number of bootstrap iterations for significance
            testing.

    Returns:
        A :class:`ComparisonResult` with summary statistics and
        significance testing.

    Raises:
        ValueError: If fewer than 2 reports are provided.
    """
    if len(reports) < 2:
        raise ValueError("compare requires at least 2 reports")

    summary: list[dict[str, Any]] = []
    for report in reports:
        summary.append({
            "name": report.name,
            "aggregate_score": report.aggregate_score,
            "std_dev": report.std_dev,
            "confidence_interval": report.confidence_interval,
            "num_examples": report.num_examples,
            "total_latency_ms": report.total_latency_ms,
        })

    # Sort by aggregate score descending
    ranked = sorted(reports, key=lambda r: r.aggregate_score, reverse=True)
    winner = ranked[0]

    # Test significance between top two
    p_value = _bootstrap_p_value(
        ranked[0].scores,
        ranked[1].scores,
        n_bootstrap=n_bootstrap,
    )
    significant = p_value < significance_level

    return ComparisonResult(
        reports=reports,
        summary=summary,
        winner=winner.name,
        significant=significant,
        p_value=p_value,
    )


def format_comparison(result: ComparisonResult) -> str:
    """Format a comparison result as a human-readable table.

    Args:
        result: The comparison result to format.

    Returns:
        A formatted string table.
    """
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("Evaluation Comparison")
    lines.append("=" * 70)
    lines.append(
        f"{'Name':<25} {'Score':>8} {'StdDev':>8} {'CI 95%':>16} {'N':>5}"
    )
    lines.append("-" * 70)

    for s in sorted(result.summary, key=lambda x: x["aggregate_score"], reverse=True):
        ci = s["confidence_interval"]
        marker = " *" if s["name"] == result.winner else ""
        lines.append(
            f"{s['name']:<25} {s['aggregate_score']:>8.4f} "
            f"{s['std_dev']:>8.4f} "
            f"[{ci[0]:.4f}, {ci[1]:.4f}] "
            f"{s['num_examples']:>5}{marker}"
        )

    lines.append("-" * 70)
    lines.append(f"Winner: {result.winner}")
    sig_str = "Yes" if result.significant else "No"
    lines.append(f"Statistically significant: {sig_str} (p={result.p_value:.4f})")
    lines.append("=" * 70)

    return "\n".join(lines)
