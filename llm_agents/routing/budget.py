"""Budget-aware model routing to stay within token limits."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from llm_agents.models.base import BaseModel
from llm_agents.models.types import CompletionResult


class BudgetExhaustedError(Exception):
    """Raised when the token budget has been exhausted."""


@dataclass
class UsageRecord:
    """Tracks token usage for a single model call.

    Attributes:
        model_name: Identifier of the model used.
        prompt_tokens: Estimated prompt tokens consumed.
        completion_tokens: Estimated completion tokens consumed.
    """

    model_name: str
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class BudgetRouter:
    """Selects models to stay within a cumulative token budget.

    Models are ordered from cheapest to most expensive.  The router tracks
    cumulative token usage and selects the most capable model that fits
    within the remaining budget.

    Args:
        models: Ordered list of (name, model, cost_per_token) tuples,
            from cheapest to most expensive.
        token_budget: Maximum total tokens allowed across all calls.
        tokens_per_word: Approximate tokens per word for estimation.
    """

    def __init__(
        self,
        models: list[tuple[str, BaseModel, float]],
        token_budget: int,
        tokens_per_word: float = 1.3,
    ) -> None:
        if not models:
            raise ValueError("BudgetRouter requires at least one model")
        self._models = models
        self._token_budget = token_budget
        self._tokens_per_word = tokens_per_word
        self._used_tokens: int = 0
        self._usage_log: list[UsageRecord] = []

    @property
    def remaining_budget(self) -> int:
        """Return the number of tokens remaining in the budget."""
        return max(0, self._token_budget - self._used_tokens)

    @property
    def used_tokens(self) -> int:
        """Return total tokens consumed so far."""
        return self._used_tokens

    @property
    def usage_log(self) -> list[UsageRecord]:
        """Return the log of all usage records."""
        return list(self._usage_log)

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count from text length."""
        return max(1, int(len(text.split()) * self._tokens_per_word))

    def _select_model(self, estimated_prompt_tokens: int) -> tuple[str, BaseModel]:
        """Select the best model that fits within budget.

        Tries from most expensive (most capable) to cheapest, picking the
        most capable one whose estimated cost fits within remaining budget.
        Falls back to cheapest if nothing fits.
        """
        remaining = self.remaining_budget
        if remaining <= 0:
            raise BudgetExhaustedError(
                f"Token budget exhausted ({self._used_tokens}/{self._token_budget})"
            )

        # Try most expensive first (reversed order)
        for name, model, cost_per_token in reversed(self._models):
            # Rough estimate: prompt + ~256 completion tokens
            estimated_total = estimated_prompt_tokens + 256
            if estimated_total <= remaining:
                return name, model

        # Fallback to cheapest
        name, model, _ = self._models[0]
        return name, model

    def generate(self, prompt: str, **kwargs: Any) -> CompletionResult:
        """Select a model within budget and generate a response.

        Args:
            prompt: The input prompt.
            **kwargs: Additional keyword arguments for the model.

        Returns:
            A :class:`CompletionResult`.

        Raises:
            BudgetExhaustedError: If the token budget is exhausted.
        """
        prompt_tokens = self._estimate_tokens(prompt)
        name, model = self._select_model(prompt_tokens)

        result = model.generate(prompt, **kwargs)

        completion_tokens = self._estimate_tokens(result.text)
        record = UsageRecord(
            model_name=name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        self._usage_log.append(record)
        self._used_tokens += record.total_tokens

        return result
