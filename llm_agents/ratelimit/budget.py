"""Extension 14: Token budget tracking and enforcement.

Provides per-run and per-agent token budget management with allocation
support for multi-agent systems.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class BudgetExceededError(Exception):
    """Raised when a token budget is exhausted."""

    def __init__(self, budget_name: str, used: int, limit: int) -> None:
        self.budget_name = budget_name
        self.used = used
        self.limit = limit
        super().__init__(
            f"Budget '{budget_name}' exceeded: {used}/{limit} tokens used."
        )


@dataclass
class TokenBudget:
    """Token budget with separate limits for prompt, completion, and total tokens.

    Set any limit to 0 to disable that specific constraint.

    Attributes:
        max_prompt_tokens: Maximum prompt tokens allowed.
        max_completion_tokens: Maximum completion tokens allowed.
        max_total_tokens: Maximum total tokens allowed.
        name: Descriptive name for this budget.
    """

    max_prompt_tokens: int = 0
    max_completion_tokens: int = 0
    max_total_tokens: int = 0
    name: str = "default"


@dataclass
class UsageRecord:
    """Accumulated token usage."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class BudgetTracker:
    """Tracks token usage across agent runs and enforces budgets.

    Args:
        budget: The token budget to enforce.
    """

    def __init__(self, budget: TokenBudget) -> None:
        self.budget = budget
        self._usage = UsageRecord()

    def record(self, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        """Record token usage and check against budget limits.

        Args:
            prompt_tokens: Number of prompt tokens used.
            completion_tokens: Number of completion tokens generated.

        Raises:
            BudgetExceededError: If any limit is exceeded.
        """
        self._usage.prompt_tokens += prompt_tokens
        self._usage.completion_tokens += completion_tokens

        self._check_limits()

    def _check_limits(self) -> None:
        b = self.budget
        u = self._usage

        if b.max_prompt_tokens > 0 and u.prompt_tokens > b.max_prompt_tokens:
            raise BudgetExceededError(
                f"{b.name}/prompt", u.prompt_tokens, b.max_prompt_tokens
            )
        if b.max_completion_tokens > 0 and u.completion_tokens > b.max_completion_tokens:
            raise BudgetExceededError(
                f"{b.name}/completion", u.completion_tokens, b.max_completion_tokens
            )
        if b.max_total_tokens > 0 and u.total_tokens > b.max_total_tokens:
            raise BudgetExceededError(
                f"{b.name}/total", u.total_tokens, b.max_total_tokens
            )

    def can_afford(self, estimated_tokens: int) -> bool:
        """Check if the budget can accommodate an estimated number of tokens.

        Args:
            estimated_tokens: Estimated total tokens for the next call.

        Returns:
            True if the budget has room, False otherwise.
        """
        if self.budget.max_total_tokens > 0:
            return (self._usage.total_tokens + estimated_tokens) <= self.budget.max_total_tokens
        return True

    @property
    def usage(self) -> UsageRecord:
        """Return current usage record."""
        return self._usage

    @property
    def remaining_total(self) -> int | None:
        """Return remaining total tokens, or None if unlimited."""
        if self.budget.max_total_tokens > 0:
            return max(0, self.budget.max_total_tokens - self._usage.total_tokens)
        return None

    def reset(self) -> None:
        """Reset usage counters."""
        self._usage = UsageRecord()


def allocate_budgets(
    total_budget: TokenBudget,
    agent_names: list[str],
    weights: dict[str, float] | None = None,
) -> dict[str, TokenBudget]:
    """Divide a total token budget across multiple agents.

    Args:
        total_budget: The total budget to divide.
        agent_names: List of agent names.
        weights: Optional per-agent weights (must sum to 1.0). If None,
            budget is split equally.

    Returns:
        Dict mapping agent name to their allocated TokenBudget.
    """
    if not agent_names:
        return {}

    if weights is None:
        weights = {name: 1.0 / len(agent_names) for name in agent_names}

    result: dict[str, TokenBudget] = {}
    for name in agent_names:
        w = weights.get(name, 0.0)
        result[name] = TokenBudget(
            max_prompt_tokens=int(total_budget.max_prompt_tokens * w) if total_budget.max_prompt_tokens else 0,
            max_completion_tokens=int(total_budget.max_completion_tokens * w) if total_budget.max_completion_tokens else 0,
            max_total_tokens=int(total_budget.max_total_tokens * w) if total_budget.max_total_tokens else 0,
            name=f"{total_budget.name}/{name}",
        )
    return result
