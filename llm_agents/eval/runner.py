"""Evaluation runner and report generation."""

from __future__ import annotations

import math
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable

from llm_agents.agents.agent import Agent
from llm_agents.eval.dataset import EvalDataset, EvalExample


@dataclass
class ExampleResult:
    """Result for a single evaluation example.

    Attributes:
        example: The original evaluation example.
        predicted: The model's output.
        score: The metric score for this example.
        latency_ms: Time taken in milliseconds.
    """

    example: EvalExample
    predicted: str
    score: float
    latency_ms: float = 0.0


@dataclass
class EvalReport:
    """Aggregated evaluation results.

    Attributes:
        name: Name of this evaluation run.
        results: Per-example results.
        aggregate_score: Mean score across all examples.
        std_dev: Standard deviation of scores.
        confidence_interval: 95% confidence interval ``(lower, upper)``.
        total_latency_ms: Total evaluation time in milliseconds.
        metadata: Additional run metadata.
    """

    name: str
    results: list[ExampleResult] = field(default_factory=list)
    aggregate_score: float = 0.0
    std_dev: float = 0.0
    confidence_interval: tuple[float, float] = (0.0, 0.0)
    total_latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def scores(self) -> list[float]:
        """Return the list of per-example scores."""
        return [r.score for r in self.results]

    @property
    def num_examples(self) -> int:
        """Return the number of evaluated examples."""
        return len(self.results)


def _compute_stats(scores: list[float]) -> tuple[float, float, tuple[float, float]]:
    """Compute mean, std dev, and 95% CI for a list of scores."""
    if not scores:
        return 0.0, 0.0, (0.0, 0.0)
    n = len(scores)
    mean = sum(scores) / n
    if n < 2:
        return mean, 0.0, (mean, mean)
    variance = sum((s - mean) ** 2 for s in scores) / (n - 1)
    std = math.sqrt(variance)
    margin = 1.96 * std / math.sqrt(n)
    return mean, std, (max(0.0, mean - margin), min(1.0, mean + margin))


class EvalRunner:
    """Run evaluation of an agent against a dataset with specified metrics.

    Args:
        agent: The agent to evaluate.
        dataset: The evaluation dataset.
        metric: A callable ``(predicted, expected) -> float`` that scores
            each example.
        concurrency: Number of examples to evaluate in parallel.
            Set to 1 for sequential execution.
        name: Name for the evaluation run.
    """

    def __init__(
        self,
        agent: Agent,
        dataset: EvalDataset,
        metric: Callable[[str, str], float],
        concurrency: int = 1,
        name: str = "",
    ) -> None:
        self._agent = agent
        self._dataset = dataset
        self._metric = metric
        self._concurrency = max(1, concurrency)
        self._name = name or f"eval-{agent.name}"

    def run(self) -> EvalReport:
        """Execute the evaluation and return a report.

        Returns:
            An :class:`EvalReport` with per-example and aggregate results.
        """
        start = time.time()
        results: list[ExampleResult] = []

        if self._concurrency == 1:
            for example in self._dataset:
                result = self._evaluate_one(example)
                results.append(result)
        else:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=self._concurrency) as executor:
                futures = {
                    executor.submit(self._evaluate_one, ex): ex
                    for ex in self._dataset
                }
                for future in as_completed(futures):
                    results.append(future.result())

        total_ms = (time.time() - start) * 1000
        scores = [r.score for r in results]
        mean, std, ci = _compute_stats(scores)

        return EvalReport(
            name=self._name,
            results=results,
            aggregate_score=mean,
            std_dev=std,
            confidence_interval=ci,
            total_latency_ms=total_ms,
            metadata={
                "agent": self._agent.name,
                "dataset_size": len(self._dataset),
                "concurrency": self._concurrency,
            },
        )

    def _evaluate_one(self, example: EvalExample) -> ExampleResult:
        """Evaluate a single example."""
        self._agent.reset()
        t0 = time.time()
        response = self._agent.run(example.input)
        latency = (time.time() - t0) * 1000
        score = self._metric(response.content, example.expected)
        return ExampleResult(
            example=example,
            predicted=response.content,
            score=score,
            latency_ms=latency,
        )
