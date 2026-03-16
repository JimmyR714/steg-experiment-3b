"""Memory stores for agent retrieval-augmented generation."""

from __future__ import annotations

import json
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from llm_agents.memory.embeddings import Embedder


@dataclass
class MemoryRecord:
    """A single memory entry."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: np.ndarray | None = None


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between vector *a* and matrix *b*."""
    a_norm = a / (np.linalg.norm(a) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return b_norm @ a_norm


class MemoryStore(ABC):
    """Abstract base for memory stores."""

    @abstractmethod
    def add(self, text: str, metadata: dict[str, Any] | None = None) -> None:
        """Store a piece of text with optional metadata."""

    @abstractmethod
    def search(self, query: str, k: int = 5) -> list[MemoryRecord]:
        """Retrieve the *k* most relevant memories for *query*."""

    @abstractmethod
    def clear(self) -> None:
        """Remove all memories."""

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of stored memories."""


class InMemoryStore(MemoryStore):
    """In-memory vector store using numpy cosine similarity.

    Args:
        embedder: An embedding provider implementing the :class:`Embedder` protocol.
    """

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder
        self._records: list[MemoryRecord] = []
        self._embeddings: np.ndarray | None = None

    def add(self, text: str, metadata: dict[str, Any] | None = None) -> None:
        embedding = self._embedder.embed([text])[0]
        record = MemoryRecord(text=text, metadata=metadata or {}, embedding=embedding)
        self._records.append(record)
        if self._embeddings is None:
            self._embeddings = embedding.reshape(1, -1)
        else:
            self._embeddings = np.vstack([self._embeddings, embedding.reshape(1, -1)])

    def search(self, query: str, k: int = 5) -> list[MemoryRecord]:
        if not self._records or self._embeddings is None:
            return []
        query_emb = self._embedder.embed([query])[0]
        similarities = _cosine_similarity(query_emb, self._embeddings)
        top_indices = np.argsort(similarities)[::-1][:k]
        return [self._records[i] for i in top_indices]

    def clear(self) -> None:
        self._records.clear()
        self._embeddings = None

    def __len__(self) -> int:
        return len(self._records)


class PersistentStore(MemoryStore):
    """SQLite-backed memory store with numpy embeddings.

    Args:
        embedder: An embedding provider.
        db_path: Path to the SQLite database file. Defaults to in-memory.
    """

    def __init__(self, embedder: Embedder, db_path: str = ":memory:") -> None:
        self._embedder = embedder
        self._conn = sqlite3.connect(db_path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS memories ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  text TEXT NOT NULL,"
            "  metadata TEXT NOT NULL DEFAULT '{}',"
            "  embedding BLOB NOT NULL"
            ")"
        )
        self._conn.commit()

    def add(self, text: str, metadata: dict[str, Any] | None = None) -> None:
        embedding = self._embedder.embed([text])[0]
        self._conn.execute(
            "INSERT INTO memories (text, metadata, embedding) VALUES (?, ?, ?)",
            (text, json.dumps(metadata or {}), embedding.tobytes()),
        )
        self._conn.commit()

    def search(self, query: str, k: int = 5) -> list[MemoryRecord]:
        rows = self._conn.execute(
            "SELECT text, metadata, embedding FROM memories"
        ).fetchall()
        if not rows:
            return []

        query_emb = self._embedder.embed([query])[0]
        dim = query_emb.shape[0]

        records: list[MemoryRecord] = []
        embeddings: list[np.ndarray] = []
        for text, meta_str, emb_bytes in rows:
            emb = np.frombuffer(emb_bytes, dtype=np.float32).copy()
            if emb.shape[0] != dim:
                continue
            records.append(MemoryRecord(text=text, metadata=json.loads(meta_str), embedding=emb))
            embeddings.append(emb)

        if not records:
            return []

        emb_matrix = np.vstack(embeddings)
        similarities = _cosine_similarity(query_emb, emb_matrix)
        top_indices = np.argsort(similarities)[::-1][:k]
        return [records[i] for i in top_indices]

    def clear(self) -> None:
        self._conn.execute("DELETE FROM memories")
        self._conn.commit()

    def __len__(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()
        return row[0] if row else 0
