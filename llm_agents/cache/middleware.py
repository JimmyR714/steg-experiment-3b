"""Transparent caching middleware that wraps any BaseModel."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from llm_agents.cache.exact import ExactCache
from llm_agents.models.base import BaseModel
from llm_agents.models.types import CompletionResult, LogProbResult


@dataclass
class CacheStats:
    """Aggregated cache performance statistics.

    Attributes:
        hits: Number of cache hits.
        misses: Number of cache misses.
        token_savings: Estimated tokens saved by cache hits.
        latency_savings_ms: Estimated latency saved in milliseconds.
    """

    hits: int = 0
    misses: int = 0
    token_savings: int = 0
    latency_savings_ms: float = 0.0

    @property
    def hit_rate(self) -> float:
        """Return the hit rate as a fraction in [0, 1]."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class CachedModel(BaseModel):
    """Wraps any :class:`BaseModel` with transparent caching.

    Both ``generate()`` and ``get_logprobs()`` check the cache before
    calling the underlying model.  Cache hits return stored results
    instantly.

    Args:
        model: The underlying model to wrap.
        cache: An :class:`ExactCache` instance to use for storage.
        model_name: Identifier used as the ``model`` key in cache lookups.
            Defaults to the class name of the wrapped model.
    """

    def __init__(
        self,
        model: BaseModel,
        cache: ExactCache,
        model_name: str | None = None,
    ) -> None:
        self._model = model
        self._cache = cache
        self._model_name = model_name or type(model).__name__
        self._stats = CacheStats()
        self._avg_latency_ms = 500.0  # running average for savings estimation

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 1.0,
        top_k: int = 50,
        stop: list[str] | None = None,
    ) -> CompletionResult:
        """Generate a completion, returning a cached result if available.

        Args:
            prompt: The input text to complete.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            top_k: Number of top tokens to consider during sampling.
            stop: Optional list of stop sequences.

        Returns:
            A :class:`CompletionResult`.
        """
        params = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "stop": stop,
            "method": "generate",
        }

        cached = self._cache.get(self._model_name, prompt, params)
        if cached is not None:
            self._stats.hits += 1
            self._stats.latency_savings_ms += self._avg_latency_ms
            return CompletionResult(text=cached)

        start = time.time()
        result = self._model.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            stop=stop,
        )
        elapsed_ms = (time.time() - start) * 1000
        self._avg_latency_ms = 0.9 * self._avg_latency_ms + 0.1 * elapsed_ms

        self._stats.misses += 1
        self._stats.token_savings += len(result.text.split())
        self._cache.put(self._model_name, prompt, params, result.text)
        return result

    def get_logprobs(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        top_k: int = 5,
    ) -> LogProbResult:
        """Generate with log-probabilities.  Not cached (logprob data is complex).

        Args:
            prompt: The input text to complete.
            max_tokens: Maximum number of tokens to generate.
            top_k: Number of top alternative tokens per position.

        Returns:
            A :class:`LogProbResult`.
        """
        return self._model.get_logprobs(prompt, max_tokens=max_tokens, top_k=top_k)

    @property
    def stats(self) -> CacheStats:
        """Return aggregated cache statistics."""
        return self._stats

    @property
    def cache(self) -> ExactCache:
        """Return the underlying cache instance."""
        return self._cache
