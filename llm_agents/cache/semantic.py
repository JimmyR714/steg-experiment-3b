"""Semantic cache using embedding similarity for paraphrased queries."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

    from llm_agents.memory.embeddings import Embedder


@dataclass
class _SemanticEntry:
    """Internal entry for the semantic cache."""

    prompt: str
    embedding: Any  # np.ndarray at runtime
    value: str
    created_at: float
    metadata: dict[str, Any] = field(default_factory=dict)


class SemanticCache:
    """Cache that matches queries by cosine similarity of embeddings.

    Useful for paraphrased queries that have the same intent but different
    wording.

    Args:
        embedder: An :class:`Embedder` instance for computing text embeddings.
        threshold: Minimum cosine similarity to consider a cache hit.
            Values closer to 1.0 require near-exact semantic matches.
        max_size: Maximum number of cached entries.
        ttl: Time-to-live in seconds.  ``None`` means no expiration.
    """

    def __init__(
        self,
        embedder: Embedder,
        threshold: float = 0.95,
        max_size: int = 500,
        ttl: float | None = None,
    ) -> None:
        self._embedder = embedder
        self._threshold = threshold
        self._max_size = max_size
        self._ttl = ttl
        self._entries: list[_SemanticEntry] = []
        self._hits = 0
        self._misses = 0

    def get(self, prompt: str) -> str | None:
        """Look up a semantically similar cached response.

        Args:
            prompt: The query to match against cached prompts.

        Returns:
            The cached response string if a match exceeding the threshold
            is found, otherwise ``None``.
        """
        if not self._entries:
            self._misses += 1
            return None

        query_emb = self._embedder.embed([prompt])[0]
        self._expire()

        best_score = -1.0
        best_entry: _SemanticEntry | None = None

        for entry in self._entries:
            score = self._cosine_similarity(query_emb, entry.embedding)
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_score >= self._threshold and best_entry is not None:
            self._hits += 1
            return best_entry.value

        self._misses += 1
        return None

    def put(self, prompt: str, value: str, metadata: dict[str, Any] | None = None) -> None:
        """Store a prompt-response pair in the cache.

        Args:
            prompt: The input prompt.
            value: The response to cache.
            metadata: Optional metadata to store with the entry.
        """
        embedding = self._embedder.embed([prompt])[0]
        entry = _SemanticEntry(
            prompt=prompt,
            embedding=embedding,
            value=value,
            created_at=time.time(),
            metadata=metadata or {},
        )
        self._entries.append(entry)

        # Evict oldest entries if over capacity
        while len(self._entries) > self._max_size:
            self._entries.pop(0)

    def clear(self) -> None:
        """Remove all entries from the cache."""
        self._entries.clear()
        self._hits = 0
        self._misses = 0

    @property
    def hit_rate(self) -> float:
        """Return the cache hit rate as a fraction in [0, 1]."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def stats(self) -> dict[str, Any]:
        """Return cache statistics.

        Returns:
            A dict with ``hits``, ``misses``, ``hit_rate``, and ``size``.
        """
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
            "size": len(self._entries),
        }

    def _expire(self) -> None:
        """Remove expired entries."""
        if self._ttl is None:
            return
        cutoff = time.time() - self._ttl
        self._entries = [e for e in self._entries if e.created_at >= cutoff]

    @staticmethod
    def _cosine_similarity(a: Any, b: Any) -> float:
        """Compute cosine similarity between two vectors."""
        import numpy as _np

        norm_a = _np.linalg.norm(a)
        norm_b = _np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(_np.dot(a, b) / (norm_a * norm_b))
