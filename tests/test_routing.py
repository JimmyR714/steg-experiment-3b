"""Tests for Extension 7: Adaptive Routing & Model Selection."""

from __future__ import annotations

import pytest

from llm_agents.models.base import BaseModel
from llm_agents.models.types import CompletionResult, LogProbResult
from llm_agents.routing.budget import BudgetExhaustedError, BudgetRouter, UsageRecord
from llm_agents.routing.classifier import (
    Complexity,
    ComplexityClassifier,
)
from llm_agents.routing.router import CascadeRouter, LatencyRouter, ModelRouter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockModel(BaseModel):
    def __init__(self, name: str = "mock", response: str = "ok") -> None:
        self.name = name
        self._response = response
        self.call_count = 0

    def generate(self, prompt: str, **kwargs) -> CompletionResult:
        self.call_count += 1
        return CompletionResult(text=self._response)

    def get_logprobs(self, prompt: str, **kwargs) -> LogProbResult:
        return LogProbResult(prompt=prompt)


class FailingModel(BaseModel):
    def generate(self, prompt: str, **kwargs) -> CompletionResult:
        raise RuntimeError("Model failed")

    def get_logprobs(self, prompt: str, **kwargs) -> LogProbResult:
        return LogProbResult(prompt=prompt)


# ---------------------------------------------------------------------------
# ComplexityClassifier tests
# ---------------------------------------------------------------------------


class TestComplexityClassifier:
    def test_simple_prompt(self):
        classifier = ComplexityClassifier()
        result = classifier.classify("Hello")
        assert result.complexity == Complexity.SIMPLE
        assert result.score < 0.3

    def test_complex_prompt(self):
        classifier = ComplexityClassifier()
        prompt = (
            "Explain how the following Python code implements a recursive "
            "descent parser. Then compare it with an LR parser approach. "
            "First analyze the time complexity, then optimize it. "
            "```python\ndef parse(tokens):\n    pass\n```\n"
            "After that, implement the changes and evaluate the performance. "
            "Finally, describe the trade-offs between both approaches."
        )
        result = classifier.classify(prompt)
        assert result.complexity in (Complexity.MEDIUM, Complexity.HARD)
        assert result.score > 0.3

    def test_medium_prompt(self):
        classifier = ComplexityClassifier()
        prompt = "How does a hash table work? Explain the difference between chaining and open addressing."
        result = classifier.classify(prompt)
        assert result.complexity in (Complexity.SIMPLE, Complexity.MEDIUM)

    def test_features_present(self):
        classifier = ComplexityClassifier()
        result = classifier.classify("What is 2+2?")
        assert "word_count" in result.features
        assert "length_score" in result.features
        assert "aggregate_score" in result.features


# ---------------------------------------------------------------------------
# ModelRouter tests
# ---------------------------------------------------------------------------


class TestModelRouter:
    def test_routes_by_complexity(self):
        simple_model = MockModel("simple", "simple_answer")
        hard_model = MockModel("hard", "hard_answer")
        router = ModelRouter(
            routes={
                Complexity.SIMPLE: simple_model,
                Complexity.HARD: hard_model,
            }
        )
        # Simple prompt should route to simple model
        result = router.generate("Hi")
        assert result.text == "simple_answer"
        assert simple_model.call_count == 1
        assert hard_model.call_count == 0

    def test_fallback_to_available_model(self):
        model = MockModel("only", "answer")
        router = ModelRouter(routes={Complexity.MEDIUM: model})
        # Even though prompt is simple, should fall back to medium
        result = router.generate("Hello")
        assert result.text == "answer"

    def test_route_returns_model(self):
        model = MockModel("m", "r")
        router = ModelRouter(routes={Complexity.SIMPLE: model})
        selected = router.route("Hi")
        assert selected is model


# ---------------------------------------------------------------------------
# CascadeRouter tests
# ---------------------------------------------------------------------------


class TestCascadeRouter:
    def test_first_model_passes(self):
        cheap = MockModel("cheap", "cheap_answer")
        expensive = MockModel("expensive", "expensive_answer")
        router = CascadeRouter([cheap, expensive])
        result = router.generate("test")
        assert result.text == "cheap_answer"
        assert cheap.call_count == 1
        assert expensive.call_count == 0

    def test_escalation_on_validation_failure(self):
        cheap = MockModel("cheap", "bad")
        expensive = MockModel("expensive", "good")
        router = CascadeRouter(
            [cheap, expensive],
            validator=lambda r: r.text == "good",
        )
        result = router.generate("test")
        assert result.text == "good"
        assert cheap.call_count == 1
        assert expensive.call_count == 1

    def test_empty_models(self):
        with pytest.raises(ValueError, match="at least one model"):
            CascadeRouter([])

    def test_all_fail_returns_last(self):
        m1 = MockModel("m1", "bad1")
        m2 = MockModel("m2", "bad2")
        router = CascadeRouter(
            [m1, m2],
            validator=lambda r: False,
        )
        result = router.generate("test")
        assert result.text == "bad2"


# ---------------------------------------------------------------------------
# LatencyRouter tests
# ---------------------------------------------------------------------------


class TestLatencyRouter:
    def test_first_valid_returned(self):
        m1 = MockModel("m1", "first")
        m2 = MockModel("m2", "second")
        router = LatencyRouter([m1, m2])
        result = router.generate("test")
        assert result.text == "first"

    def test_skips_failing_model(self):
        failing = FailingModel()
        good = MockModel("good", "answer")
        router = LatencyRouter([failing, good])
        result = router.generate("test")
        assert result.text == "answer"

    def test_all_fail(self):
        router = LatencyRouter([FailingModel()])
        with pytest.raises(RuntimeError, match="All models failed"):
            router.generate("test")


# ---------------------------------------------------------------------------
# BudgetRouter tests
# ---------------------------------------------------------------------------


class TestBudgetRouter:
    def test_basic_usage(self):
        model = MockModel("cheap", "answer")
        router = BudgetRouter(
            models=[("cheap", model, 0.001)],
            token_budget=10000,
        )
        result = router.generate("Hello world")
        assert result.text == "answer"
        assert router.used_tokens > 0
        assert router.remaining_budget < 10000

    def test_budget_exhaustion(self):
        model = MockModel("m", "r")
        router = BudgetRouter(
            models=[("m", model, 0.001)],
            token_budget=1,  # Very small budget
        )
        # First call might succeed (depends on estimation)
        try:
            router.generate("Hello")
        except BudgetExhaustedError:
            pass
        # Keep calling until budget is exhausted
        with pytest.raises(BudgetExhaustedError):
            for _ in range(100):
                router.generate("More text here that uses tokens")

    def test_usage_log(self):
        model = MockModel("m", "r")
        router = BudgetRouter(
            models=[("m", model, 0.001)],
            token_budget=10000,
        )
        router.generate("test")
        assert len(router.usage_log) == 1
        assert router.usage_log[0].model_name == "m"

    def test_usage_record(self):
        rec = UsageRecord(model_name="test", prompt_tokens=100, completion_tokens=50)
        assert rec.total_tokens == 150
