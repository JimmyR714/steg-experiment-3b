"""Token counting, cost estimation, and budget enforcement."""

from __future__ import annotations

from dataclasses import dataclass, field


class BudgetExceededError(Exception):
    """Raised when a token or cost budget is exhausted."""


@dataclass
class TokenCounter:
    """Tracks prompt and completion tokens across model calls.

    Attributes:
        prompt_tokens: Total prompt tokens consumed.
        completion_tokens: Total completion tokens consumed.
        call_count: Number of model calls tracked.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    call_count: int = 0

    @property
    def total_tokens(self) -> int:
        """Total tokens (prompt + completion)."""
        return self.prompt_tokens + self.completion_tokens

    def record(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Record tokens from a model call.

        Args:
            prompt_tokens: Number of prompt tokens used.
            completion_tokens: Number of completion tokens generated.
        """
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.call_count += 1

    def reset(self) -> None:
        """Reset all counters to zero."""
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.call_count = 0


@dataclass
class CostEstimator:
    """Estimates dollar cost based on token usage.

    Args:
        price_per_1k_prompt: Price per 1,000 prompt tokens.
        price_per_1k_completion: Price per 1,000 completion tokens.
    """

    price_per_1k_prompt: float = 0.01
    price_per_1k_completion: float = 0.03

    def estimate(self, counter: TokenCounter) -> float:
        """Estimate the cost in dollars for the given token usage.

        Args:
            counter: A :class:`TokenCounter` with usage data.

        Returns:
            Estimated cost in dollars.
        """
        prompt_cost = (counter.prompt_tokens / 1000) * self.price_per_1k_prompt
        completion_cost = (counter.completion_tokens / 1000) * self.price_per_1k_completion
        return prompt_cost + completion_cost


class BudgetGuard:
    """Enforces a maximum token budget and raises on overflow.

    Args:
        max_tokens: Maximum total tokens allowed.

    Raises:
        BudgetExceededError: When :meth:`check` detects the budget is exceeded.
    """

    def __init__(self, max_tokens: int) -> None:
        self.max_tokens = max_tokens
        self._counter = TokenCounter()

    @property
    def counter(self) -> TokenCounter:
        """The underlying token counter."""
        return self._counter

    @property
    def remaining(self) -> int:
        """Number of tokens remaining in the budget."""
        return max(0, self.max_tokens - self._counter.total_tokens)

    def record_and_check(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Record tokens and raise if budget is exceeded.

        Args:
            prompt_tokens: Prompt tokens used.
            completion_tokens: Completion tokens generated.

        Raises:
            BudgetExceededError: If total tokens exceed the budget.
        """
        self._counter.record(prompt_tokens, completion_tokens)
        if self._counter.total_tokens > self.max_tokens:
            raise BudgetExceededError(
                f"Token budget exceeded: {self._counter.total_tokens} / {self.max_tokens}"
            )

    def check(self) -> None:
        """Check if the budget has been exceeded.

        Raises:
            BudgetExceededError: If total tokens exceed the budget.
        """
        if self._counter.total_tokens > self.max_tokens:
            raise BudgetExceededError(
                f"Token budget exceeded: {self._counter.total_tokens} / {self.max_tokens}"
            )
