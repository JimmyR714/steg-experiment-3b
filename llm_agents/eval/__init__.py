"""Evaluation and benchmarking framework for agent quality measurement."""

from llm_agents.eval.compare import ComparisonResult, compare, format_comparison
from llm_agents.eval.dataset import EvalDataset, EvalExample
from llm_agents.eval.metrics import (
    Composite,
    Score,
    contains_match,
    exact_match,
    factual_consistency,
    fuzzy_match,
    llm_judge,
    normalized_contains,
)
from llm_agents.eval.runner import EvalReport, EvalRunner, ExampleResult

__all__ = [
    "ComparisonResult",
    "Composite",
    "EvalDataset",
    "EvalExample",
    "EvalReport",
    "EvalRunner",
    "ExampleResult",
    "Score",
    "compare",
    "contains_match",
    "exact_match",
    "factual_consistency",
    "format_comparison",
    "fuzzy_match",
    "llm_judge",
    "normalized_contains",
]
