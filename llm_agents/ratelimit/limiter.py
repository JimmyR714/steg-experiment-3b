"""Extension 14: Rate limiter using the token bucket algorithm.

Provides both fixed and adaptive rate limiters that throttle API requests
based on requests-per-minute and tokens-per-minute constraints.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


class RateLimitError(Exception):
    """Raised when a rate limit would be exceeded and blocking is disabled."""

    def __init__(self, wait_seconds: float) -> None:
        self.wait_seconds = wait_seconds
        super().__init__(
            f"Rate limit exceeded. Retry after {wait_seconds:.1f}s."
        )


@dataclass
class _TokenBucket:
    """Token bucket for rate limiting.

    Tokens refill continuously at ``refill_rate`` per second up to ``capacity``.
    """

    capacity: float
    refill_rate: float  # tokens per second
    _tokens: float = 0.0
    _last_refill: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        self._tokens = self.capacity
        self._last_refill = time.monotonic()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self.capacity, self._tokens + elapsed * self.refill_rate)
        self._last_refill = now

    def try_acquire(self, tokens: float = 1.0) -> float:
        """Try to acquire tokens. Returns 0 on success, or wait time in seconds."""
        self._refill()
        if self._tokens >= tokens:
            self._tokens -= tokens
            return 0.0
        deficit = tokens - self._tokens
        return deficit / self.refill_rate

    def acquire_blocking(self, tokens: float = 1.0) -> None:
        """Acquire tokens, blocking if necessary."""
        wait = self.try_acquire(tokens)
        if wait > 0:
            time.sleep(wait)
            self._refill()
            self._tokens -= tokens

    @property
    def available(self) -> float:
        self._refill()
        return self._tokens


class RateLimiter:
    """Rate limiter with requests-per-minute and tokens-per-minute limits.

    Uses two independent token buckets — one for request count and one for
    token throughput.

    Args:
        requests_per_minute: Maximum requests per minute. 0 means unlimited.
        tokens_per_minute: Maximum tokens per minute. 0 means unlimited.
        blocking: If True, ``acquire()`` blocks until tokens are available.
            If False, raises ``RateLimitError``.
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        tokens_per_minute: int = 0,
        blocking: bool = True,
    ) -> None:
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.blocking = blocking

        self._request_bucket: _TokenBucket | None = None
        self._token_bucket: _TokenBucket | None = None

        if requests_per_minute > 0:
            self._request_bucket = _TokenBucket(
                capacity=float(requests_per_minute),
                refill_rate=requests_per_minute / 60.0,
            )
        if tokens_per_minute > 0:
            self._token_bucket = _TokenBucket(
                capacity=float(tokens_per_minute),
                refill_rate=tokens_per_minute / 60.0,
            )

    def acquire(self, estimated_tokens: int = 1) -> None:
        """Acquire permission to make a request.

        Args:
            estimated_tokens: Estimated token count for the request.

        Raises:
            RateLimitError: If blocking is disabled and the limit is exceeded.
        """
        max_wait = 0.0

        if self._request_bucket is not None:
            wait = self._request_bucket.try_acquire(1.0)
            max_wait = max(max_wait, wait)

        if self._token_bucket is not None:
            wait = self._token_bucket.try_acquire(float(estimated_tokens))
            max_wait = max(max_wait, wait)

        if max_wait > 0:
            if not self.blocking:
                raise RateLimitError(max_wait)
            time.sleep(max_wait)
            # Re-acquire after sleeping
            if self._request_bucket is not None:
                self._request_bucket.try_acquire(1.0)
            if self._token_bucket is not None:
                self._token_bucket.try_acquire(float(estimated_tokens))


class AdaptiveRateLimiter(RateLimiter):
    """Rate limiter that reduces throughput on 429 responses.

    After a rate-limit error, the effective rate is halved. It gradually
    recovers over time.

    Args:
        requests_per_minute: Base maximum requests per minute.
        tokens_per_minute: Base maximum tokens per minute.
        backoff_factor: Multiplicative factor applied on each 429 (default 0.5).
        recovery_factor: Multiplicative recovery per successful request (default 1.05).
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        tokens_per_minute: int = 0,
        backoff_factor: float = 0.5,
        recovery_factor: float = 1.05,
    ) -> None:
        super().__init__(
            requests_per_minute=requests_per_minute,
            tokens_per_minute=tokens_per_minute,
            blocking=True,
        )
        self.backoff_factor = backoff_factor
        self.recovery_factor = recovery_factor
        self._effective_rpm = float(requests_per_minute)
        self._base_rpm = float(requests_per_minute)

    def on_rate_limit(self) -> None:
        """Signal that a 429 response was received. Reduces effective rate."""
        self._effective_rpm = max(1.0, self._effective_rpm * self.backoff_factor)
        if self._request_bucket is not None:
            self._request_bucket.refill_rate = self._effective_rpm / 60.0

    def on_success(self) -> None:
        """Signal a successful request. Gradually restores rate."""
        self._effective_rpm = min(
            self._base_rpm,
            self._effective_rpm * self.recovery_factor,
        )
        if self._request_bucket is not None:
            self._request_bucket.refill_rate = self._effective_rpm / 60.0

    @property
    def effective_rpm(self) -> float:
        return self._effective_rpm
