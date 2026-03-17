"""Tests for Extension 14: Rate Limiting & Token Budgets."""

from __future__ import annotations

import pytest

from llm_agents.ratelimit.limiter import (
    AdaptiveRateLimiter,
    RateLimiter,
    RateLimitError,
)
from llm_agents.ratelimit.budget import (
    BudgetExceededError,
    BudgetTracker,
    TokenBudget,
    UsageRecord,
    allocate_budgets,
)
from llm_agents.ratelimit.middleware import RateLimitedModel, UsageReport


# ---------------------------------------------------------------------------
# RateLimiter tests
# ---------------------------------------------------------------------------


class TestRateLimiter:
    def test_basic_acquire(self):
        limiter = RateLimiter(requests_per_minute=60)
        # Should not raise
        limiter.acquire(1)

    def test_non_blocking_raises(self):
        limiter = RateLimiter(requests_per_minute=1, blocking=False)
        limiter.acquire(1)
        # Second acquire should fail (bucket exhausted)
        with pytest.raises(RateLimitError):
            limiter.acquire(1)

    def test_unlimited(self):
        limiter = RateLimiter(requests_per_minute=0, tokens_per_minute=0)
        # Should not raise
        for _ in range(100):
            limiter.acquire(1000)


# ---------------------------------------------------------------------------
# AdaptiveRateLimiter tests
# ---------------------------------------------------------------------------


class TestAdaptiveRateLimiter:
    def test_backoff_on_rate_limit(self):
        limiter = AdaptiveRateLimiter(requests_per_minute=100)
        initial_rpm = limiter.effective_rpm
        limiter.on_rate_limit()
        assert limiter.effective_rpm < initial_rpm

    def test_recovery_on_success(self):
        limiter = AdaptiveRateLimiter(requests_per_minute=100)
        limiter.on_rate_limit()
        reduced_rpm = limiter.effective_rpm
        limiter.on_success()
        assert limiter.effective_rpm > reduced_rpm

    def test_never_below_one(self):
        limiter = AdaptiveRateLimiter(requests_per_minute=2)
        for _ in range(20):
            limiter.on_rate_limit()
        assert limiter.effective_rpm >= 1.0


# ---------------------------------------------------------------------------
# TokenBudget / BudgetTracker tests
# ---------------------------------------------------------------------------


class TestTokenBudget:
    def test_defaults(self):
        budget = TokenBudget()
        assert budget.max_total_tokens == 0
        assert budget.name == "default"

    def test_custom_budget(self):
        budget = TokenBudget(
            max_prompt_tokens=1000,
            max_completion_tokens=500,
            max_total_tokens=1500,
            name="test",
        )
        assert budget.max_total_tokens == 1500


class TestBudgetTracker:
    def test_track_usage(self):
        budget = TokenBudget(max_total_tokens=100)
        tracker = BudgetTracker(budget)
        tracker.record(prompt_tokens=30, completion_tokens=20)
        assert tracker.usage.total_tokens == 50
        assert tracker.remaining_total == 50

    def test_budget_exceeded(self):
        budget = TokenBudget(max_total_tokens=50)
        tracker = BudgetTracker(budget)
        with pytest.raises(BudgetExceededError):
            tracker.record(prompt_tokens=30, completion_tokens=30)

    def test_prompt_budget_exceeded(self):
        budget = TokenBudget(max_prompt_tokens=20)
        tracker = BudgetTracker(budget)
        with pytest.raises(BudgetExceededError):
            tracker.record(prompt_tokens=25)

    def test_can_afford(self):
        budget = TokenBudget(max_total_tokens=100)
        tracker = BudgetTracker(budget)
        tracker.record(prompt_tokens=80)
        assert tracker.can_afford(10)
        assert not tracker.can_afford(30)

    def test_unlimited_budget(self):
        budget = TokenBudget()
        tracker = BudgetTracker(budget)
        tracker.record(prompt_tokens=100000)
        assert tracker.remaining_total is None
        assert tracker.can_afford(999999)

    def test_reset(self):
        budget = TokenBudget(max_total_tokens=100)
        tracker = BudgetTracker(budget)
        tracker.record(prompt_tokens=50)
        tracker.reset()
        assert tracker.usage.total_tokens == 0


class TestUsageRecord:
    def test_total_tokens(self):
        r = UsageRecord(prompt_tokens=10, completion_tokens=20)
        assert r.total_tokens == 30


# ---------------------------------------------------------------------------
# allocate_budgets tests
# ---------------------------------------------------------------------------


class TestAllocateBudgets:
    def test_equal_allocation(self):
        total = TokenBudget(max_total_tokens=100)
        budgets = allocate_budgets(total, ["a", "b"])
        assert budgets["a"].max_total_tokens == 50
        assert budgets["b"].max_total_tokens == 50

    def test_weighted_allocation(self):
        total = TokenBudget(max_total_tokens=100)
        budgets = allocate_budgets(total, ["a", "b"], weights={"a": 0.7, "b": 0.3})
        assert budgets["a"].max_total_tokens == 70
        assert budgets["b"].max_total_tokens == 30

    def test_empty_agents(self):
        total = TokenBudget(max_total_tokens=100)
        budgets = allocate_budgets(total, [])
        assert budgets == {}


# ---------------------------------------------------------------------------
# RateLimitedModel tests (with mock)
# ---------------------------------------------------------------------------


class _MockModel:
    """Minimal mock that satisfies BaseModel for testing."""

    def __init__(self, response_text: str = "mock response") -> None:
        self._text = response_text
        self.call_count = 0

    def generate(self, prompt, *, max_tokens=256, temperature=1.0, top_k=50, stop=None):
        from llm_agents.models.types import CompletionResult
        self.call_count += 1
        return CompletionResult(text=self._text)

    def get_logprobs(self, prompt, *, max_tokens=256, top_k=5):
        from llm_agents.models.types import LogProbResult
        return LogProbResult(prompt=prompt)


class TestRateLimitedModel:
    def test_passthrough(self):
        mock = _MockModel("hello")
        model = RateLimitedModel(mock)
        result = model.generate("test")
        assert result.text == "hello"
        assert model.report.total_requests == 1

    def test_with_budget(self):
        mock = _MockModel("hi")
        budget = TokenBudget(max_total_tokens=10)
        model = RateLimitedModel(mock, budget=budget)
        model.generate("test", max_tokens=5)
        # The token count from split words might vary, just check it tracked
        assert model.report.total_requests == 1

    def test_usage_report(self):
        report = UsageReport(total_prompt_tokens=10, total_completion_tokens=20)
        assert report.total_tokens == 30
