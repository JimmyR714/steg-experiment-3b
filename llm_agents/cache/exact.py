"""Exact-match cache with TTL and LRU eviction."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CacheEntry:
    """A single cached response.

    Attributes:
        key: The cache key (hash of request parameters).
        value: The cached response text.
        created_at: Timestamp when the entry was created.
        metadata: Optional metadata about the cached request.
    """

    key: str
    value: str
    created_at: float
    metadata: dict[str, Any] = field(default_factory=dict)


def _make_key(model: str, prompt: str, parameters: dict[str, Any]) -> str:
    """Create a deterministic cache key from request parameters.

    Args:
        model: Model identifier.
        prompt: The input prompt.
        parameters: Generation parameters (temperature, max_tokens, etc.).

    Returns:
        A hex digest string.
    """
    key_data = json.dumps(
        {"model": model, "prompt": prompt, "parameters": parameters},
        sort_keys=True,
    )
    return hashlib.sha256(key_data.encode()).hexdigest()


class ExactCache:
    """Hash-based cache keyed on (model, prompt, parameters).

    Supports in-memory dict, SQLite, and TTL-based expiration with LRU
    eviction.

    Args:
        backend: Storage backend — ``"memory"`` or ``"sqlite"``.
        max_size: Maximum number of entries before LRU eviction.
        ttl: Time-to-live in seconds.  Entries older than this are
            treated as expired.  ``None`` means no expiration.
        db_path: Path to SQLite database (only used when backend is
            ``"sqlite"``).
    """

    def __init__(
        self,
        backend: str = "memory",
        max_size: int = 1000,
        ttl: float | None = None,
        db_path: str = "cache.db",
    ) -> None:
        self._backend = backend
        self._max_size = max_size
        self._ttl = ttl
        self._hits = 0
        self._misses = 0

        if backend == "sqlite":
            self._db = sqlite3.connect(db_path)
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS cache ("
                "  key TEXT PRIMARY KEY,"
                "  value TEXT NOT NULL,"
                "  created_at REAL NOT NULL,"
                "  metadata TEXT NOT NULL"
                ")"
            )
            self._db.commit()
            self._store: OrderedDict[str, CacheEntry] | None = None
        else:
            self._store = OrderedDict()
            self._db = None  # type: ignore[assignment]

    def get(self, model: str, prompt: str, parameters: dict[str, Any]) -> str | None:
        """Look up a cached response.

        Args:
            model: Model identifier.
            prompt: The input prompt.
            parameters: Generation parameters.

        Returns:
            The cached response string, or ``None`` on a miss.
        """
        key = _make_key(model, prompt, parameters)
        entry = self._get_entry(key)

        if entry is None:
            self._misses += 1
            return None

        if self._ttl is not None and (time.time() - entry.created_at) > self._ttl:
            self._delete(key)
            self._misses += 1
            return None

        self._hits += 1
        # Move to end for LRU
        if self._store is not None and key in self._store:
            self._store.move_to_end(key)
        return entry.value

    def put(
        self,
        model: str,
        prompt: str,
        parameters: dict[str, Any],
        value: str,
    ) -> None:
        """Store a response in the cache.

        Args:
            model: Model identifier.
            prompt: The input prompt.
            parameters: Generation parameters.
            value: The response text to cache.
        """
        key = _make_key(model, prompt, parameters)
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=time.time(),
            metadata={"model": model},
        )

        if self._store is not None:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = entry
            while len(self._store) > self._max_size:
                self._store.popitem(last=False)
        else:
            self._db.execute(
                "INSERT OR REPLACE INTO cache (key, value, created_at, metadata) "
                "VALUES (?, ?, ?, ?)",
                (key, value, entry.created_at, json.dumps(entry.metadata)),
            )
            self._db.commit()

    def clear(self) -> None:
        """Remove all entries from the cache."""
        if self._store is not None:
            self._store.clear()
        else:
            self._db.execute("DELETE FROM cache")
            self._db.commit()
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
            "size": self._size(),
        }

    def _get_entry(self, key: str) -> CacheEntry | None:
        if self._store is not None:
            return self._store.get(key)
        row = self._db.execute(
            "SELECT key, value, created_at, metadata FROM cache WHERE key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        return CacheEntry(
            key=row[0],
            value=row[1],
            created_at=row[2],
            metadata=json.loads(row[3]),
        )

    def _delete(self, key: str) -> None:
        if self._store is not None:
            self._store.pop(key, None)
        else:
            self._db.execute("DELETE FROM cache WHERE key = ?", (key,))
            self._db.commit()

    def _size(self) -> int:
        if self._store is not None:
            return len(self._store)
        row = self._db.execute("SELECT COUNT(*) FROM cache").fetchone()
        return row[0] if row else 0
