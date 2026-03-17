"""Extension 14: Rate-limited model middleware.

Wraps any BaseModel with rate limiting and budget enforcement, providing
automatic retry with backoff on rate-limit errors.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from llm_agents.models.base import BaseModel
from llm_agents.models.types import CompletionResult, LogProbResult
from llm_agents.ratelimit.budget import BudgetTracker, TokenBudget
from llm_agents.ratelimit.limiter import AdaptiveRateLimiter, RateLimiter


@dataclass
class UsageReport:
    """Summary of token usage and rate limiting behavior.

    Attributes:
        total_requests: Total number of requests made.
        total_prompt_tokens: Total prompt tokens used.
        total_completion_tokens: Total completion tokens generated.
        total_retries: Number of retries due to rate limiting.
        remaining_budget: Remaining total token budget (None if unlimited).
    """

    total_requests: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_retries: int = 0
    remaining_budget: int | None = None

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens


class RateLimitedModel(BaseModel):
    """Wraps a BaseModel with rate limiting and optional budget enforcement.

    Args:
        model: The underlying model to wrap.
        limiter: Rate limiter instance. If None, no rate limiting is applied.
        budget: Token budget to enforce. If None, no budget enforcement.
        max_retries: Maximum retries on rate-limit errors.
        base_delay: Base delay in seconds for exponential backoff.
    """

    def __init__(
        self,
        model: BaseModel,
        limiter: RateLimiter | None = None,
        budget: TokenBudget | None = None,
        max_retries: int = 3,
        base_delay: float = 1.0,
    ) -> None:
        self._model = model
        self._limiter = limiter
        self._tracker = BudgetTracker(budget) if budget else None
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._report = UsageReport()

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 1.0,
        top_k: int = 50,
        stop: list[str] | None = None,
    ) -> CompletionResult:
        """Generate a completion with rate limiting and budget enforcement."""
        estimated_tokens = len(prompt.split()) + max_tokens

        # Check budget
        if self._tracker and not self._tracker.can_afford(estimated_tokens):
            from llm_agents.ratelimit.budget import BudgetExceededError
            raise BudgetExceededError(
                self._tracker.budget.name,
                self._tracker.usage.total_tokens,
                self._tracker.budget.max_total_tokens,
            )

        # Rate limiting with retry
        for attempt in range(self._max_retries + 1):
            if self._limiter:
                self._limiter.acquire(estimated_tokens)

            try:
                result = self._model.generate(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    stop=stop,
                )
                break
            except Exception as exc:
                if "429" in str(exc) or "rate" in str(exc).lower():
                    if isinstance(self._limiter, AdaptiveRateLimiter):
                        self._limiter.on_rate_limit()
                    self._report.total_retries += 1
                    if attempt < self._max_retries:
                        time.sleep(self._base_delay * (2 ** attempt))
                        continue
                raise
        else:
            # All retries exhausted — re-raise via normal generation
            result = self._model.generate(
                prompt, max_tokens=max_tokens, temperature=temperature,
                top_k=top_k, stop=stop,
            )

        # Track usage
        prompt_tokens = len(prompt.split())
        completion_tokens = len(result.text.split())
        self._report.total_requests += 1
        self._report.total_prompt_tokens += prompt_tokens
        self._report.total_completion_tokens += completion_tokens

        if self._tracker:
            self._tracker.record(prompt_tokens, completion_tokens)

        if isinstance(self._limiter, AdaptiveRateLimiter):
            self._limiter.on_success()

        self._report.remaining_budget = (
            self._tracker.remaining_total if self._tracker else None
        )

        return result

    def get_logprobs(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        top_k: int = 5,
    ) -> LogProbResult:
        """Generate with log-probabilities, respecting rate limits."""
        if self._limiter:
            self._limiter.acquire(len(prompt.split()) + max_tokens)
        return self._model.get_logprobs(prompt, max_tokens=max_tokens, top_k=top_k)

    @property
    def report(self) -> UsageReport:
        """Return usage report."""
        return self._report

    @property
    def tracker(self) -> BudgetTracker | None:
        """Return the budget tracker, if any."""
        return self._tracker
